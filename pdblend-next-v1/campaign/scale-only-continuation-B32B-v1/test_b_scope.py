"""B-only interface regressions; no synthetic measurement/proof is published."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import contract as c
import reference_map as m
import supervise as s


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=c.sha(path))


@pytest.mark.parametrize('model',[None,'7b','14b'])
def test_other_model_rejected_before_release(model,monkeypatch):
    monkeypatch.setattr(c.released,'verify_release',lambda *a,**k:pytest.fail('wrong model reached release'))
    with pytest.raises(ValueError,match='only permits B32B'):
        c.check_spec(dict(model=model),'/missing','f'*64)


def test_missing_actual_release_no_output_or_process(tmp_path,monkeypatch):
    spec=write(tmp_path/'spec.json',dict(model='32b'))
    args=SimpleNamespace(spec=Path(spec['path']),spec_sha256=spec['sha256'],release=tmp_path/'absent-release',
        release_sha256='a'*64,out=tmp_path/'out',group=None)
    monkeypatch.setattr(s,'package_check',lambda:None)
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:pytest.fail('no process without actual B150 release'))
    with pytest.raises(FileNotFoundError):s.Supervisor(args).run()
    assert not args.out.exists()


def test_binding_metadata_cannot_accept_old_model_release(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'package_check',lambda:None)
    monkeypatch.setattr(c,'released_main_records',lambda *a:pytest.fail('reference before authorization'))
    path=tmp_path/'old-release.json'
    path.write_text(json.dumps(dict(model='32b',schema=1,kind='model-main-release-per-cell',models={})))
    with pytest.raises(ValueError):m.binding_metadata(dict(model='32b'),{},path,c.sha(path),[])


def test_eco_delegate_cannot_fall_back_to_old_temporal_reader(monkeypatch):
    calls=[]
    def reject(binding):calls.append(binding);raise ValueError('new qualification missing')
    monkeypatch.setattr(c.released,'audit_eco_binding',reject)
    b=dict(model='32b',system='ecoserve',configs=dict(alpaca='unused'))
    with pytest.raises(ValueError,match='new qualification missing'):c.qualified_gate(b,['alpaca'])
    assert calls==[b]


def test_non_eco_keeps_original_actual_ordinary_pd_reader(monkeypatch):
    monkeypatch.setattr(c.released,'audit_eco_binding',lambda *a:pytest.fail('ordinary strategy must not borrow Eco qualifier'))
    b=c.read(c.CAMPAIGN/'B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json')
    v=c.qualified_gate(b,['alpaca','sharegpt','longbench'])
    assert v['verified']==dict(ordinary=True,pd=True,temporal=False)
    assert v['overall_runtime_gate_passed'] is False and v['physical']['all8_energy_j']>0


def pair(tmp_path):
    refs=[];host=tmp_path/'host';host.mkdir();write(host/'manifest.json',dict(files={}))
    for label,pid,start in [('main',11,'old'),('fresh',12,'new')]:
        folder=tmp_path/label;cfg=folder/'eco.json';write(cfg,dict(strategy='ecoserve'))
        instance=dict(id='i0',tp=2,gpus=[0,1],role='mixed',container=dict(id='cid',name='engine',image='image',StartedAt=start),
            provenance=dict(instance_id='i0',pid=1,source_files_at_import={'source':'a'*64}))
        inv=[dict(Id='cid',Name='/engine',Image='image',State=dict(Pid=pid,StartedAt=start,Running=True))]
        identity=folder/'identity.json';write(identity,inv)
        gate=folder/'gate';write(gate/'identity.after.json',[dict(container=inv[0],provenance=instance['provenance'])])
        b=dict(model='32b',system='ecoserve',protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,hostname='CPU',
            host_release=str(host),output=str(tmp_path/'output'),configs=dict(alpaca=str(cfg)),instances=[instance],
            files={str(identity):c.sha(identity)},correctness_evidence=str(gate),output_correctness_verified=True,
            mechanism_proof=dict(required=['ordinary','temporal_native_trajectory_exact']))
        refs.append(write(folder/'binding.json',b))
    return refs


@pytest.mark.parametrize('mode',['same_process','restarted'])
def test_both_identity_modes_require_eco_qualification(tmp_path,monkeypatch,mode):
    old,new=pair(tmp_path);target=old if mode=='same_process' else new;calls=[]
    monkeypatch.setattr(c,'source_policy',lambda *a:True)
    monkeypatch.setattr(c,'qualified_gate',lambda b,ds:calls.append((b,ds)))
    assert c.compatible_bindings(old,target,['alpaca'],mode)['identity_mode']==mode
    assert len(calls)==1
    def fail(*a):raise ValueError('qualification failure')
    monkeypatch.setattr(c,'qualified_gate',fail)
    with pytest.raises(ValueError,match='qualification failure'):c.compatible_bindings(old,target,['alpaca'],mode)


def test_restart_still_requires_changed_real_host_pid(tmp_path,monkeypatch):
    old,new=pair(tmp_path);b=c.reference(new);p=Path(new['path']).parent/'identity.json';inv=c.read(p);inv[0]['State']['Pid']=11
    write(p,inv);b['files'][str(p)]=c.sha(p);new=write(Path(new['path']),b)
    monkeypatch.setattr(c,'source_policy',lambda *a:True)
    monkeypatch.setattr(c,'qualified_gate',lambda *a:pytest.fail('must reject stale host PID before qualification'))
    with pytest.raises(ValueError,match='host State.Pid'):c.compatible_bindings(old,new,['alpaca'],'restarted')


def test_main_map_measurement_and_queue_bytes_unchanged():
    parent=c.CAMPAIGN/'scale-only-continuation-v3'
    for name in ['reference_map.py','supervise.py','scale_driver.py','child.py','power_evidence.frozen.py']:
        assert (parent/name).read_bytes()==(c.HERE/name).read_bytes()
    nodes=lambda path:{n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    old,new=nodes(parent/'contract.py'),nodes(c.HERE/'contract.py')
    assert old.keys()==new.keys()
    assert {name for name in old if old[name]!=new[name]}=={'qualified_gate','compatible_bindings','check_spec'}


def test_unchanged_scale_only_command_and_deadline(tmp_path,monkeypatch):
    argv=s.make_argv('/source','/binding','ecoserve',['alpaca'],'/refs','f'*64)
    assert argv[argv.index('--phase')+1]=='scale' and argv[argv.index('--max-cells')+1]=='1'
    ref=write(tmp_path/'spec.json',{});a=SimpleNamespace(out=tmp_path/'out',spec=Path(ref['path']),spec_sha256=ref['sha256'])
    supervisor=s.Supervisor(a);monkeypatch.setattr(s.time,'time',lambda:c.DEADLINE-519)
    with pytest.raises(ValueError,match='deadline'):supervisor.boundary()
