"""Focused CPU contracts, including actual retained scale/raw mechanism inputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import contract as c
import supervise as s


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return dict(path=str(p),sha256=c.sha(p))


def pair(tmp_path):
    out=tmp_path/'output';refs=[]
    host=tmp_path/'host';source=host/'source.py';host.mkdir();source.write_text('# CPU fixture\n')
    write(host/'manifest.json',dict(files={'source.py':c.sha(source)}))
    for label,pid,started in [('main',101,'old'),('fresh',102,'new')]:
        d=tmp_path/label;cfg=d/'config.json';write(cfg,dict(strategy='mixed',instances=[],topology={'runtime_dir':'/original/dynamic','engine_template':'/original/template.json'}))
        instance=dict(id='i0',tp=1,gpus=[0],role='mixed',url='http://localhost:99',port=99,kv_port=100,
            native_kind='legacy_sync_put',scheduler_cache_observed=False,engine_config='/same/engine.json',
            container=dict(id='cid',name='engine',image='image',StartedAt=started),
            provenance=dict(instance_id='i0',pid=1,model='m',source_files_at_import={'source':'a'*64}))
        inventory=[dict(Id='cid',Name='/engine',Image='image',State=dict(Pid=pid,StartedAt=started,Running=True))]
        inv=d/'identity.json';write(inv,inventory)
        gate=d/'gate';gate.mkdir();gp=dict(instance['provenance'],engine_version='observed-extra')
        write(gate/'identity.after.json',[dict(container=inventory[0],provenance=gp)])
        b=dict(protocol_id=c.PROTOCOL,deadline_s=c.DEADLINE,model='7b',system='mixed',hostname=c.barrier.HOSTS['7b'],
            host_release=str(host),output=str(out),instances=[instance],configs={'alpaca':str(cfg)},
            files={str(cfg):c.sha(cfg),str(inv):c.sha(inv),str(source):c.sha(source),str(host/'manifest.json'):c.sha(host/'manifest.json')},correctness_evidence=str(gate),
            mechanism_proof={'required':['ordinary'],'verified':{'ordinary':True}},output_correctness_verified=True)
        refs.append(write(d/'binding.json',b))
    return refs


def rewrite(ref,value):
    Path(ref['path']).write_text(json.dumps(value));ref['sha256']=c.sha(ref['path'])


def test_missing_release_no_output_or_process(tmp_path,monkeypatch):
    spec=write(tmp_path/'spec.json',{})
    args=SimpleNamespace(spec=Path(spec['path']),spec_sha256=spec['sha256'],release=tmp_path/'absent-release',release_sha256='a'*64,out=tmp_path/'out',group=None)
    monkeypatch.setattr(s,'package_check',lambda:None)
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:pytest.fail('process before release'))
    with pytest.raises((ValueError,FileNotFoundError)):s.Supervisor(args).run()
    assert not args.out.exists()


def test_exact_policy_and_frozen_topology():
    base={'strategy':'dynamollm','topology':{'runtime_dir':'/old','engine_template':'/old.json'}}
    assert c.policy_equal(dict(base,journal='/old-journal'),dict(base,journal='/new-journal'))
    with pytest.raises(ValueError):c.policy_equal(base,dict(base,service_dvfs=False))
    with pytest.raises(ValueError):c.policy_equal(base,dict(base,topology={'runtime_dir':'/new','engine_template':'/old.json'}))


def test_same_process_and_restart_pid_one(tmp_path,monkeypatch):
    main,fresh=pair(tmp_path)
    assert c.compatible_bindings(main,main,['alpaca'],'same_process')['identity_mode']=='same_process'
    with pytest.raises(ValueError):c.compatible_bindings(main,fresh,['alpaca'],'same_process')
    calls=[];monkeypatch.setattr(c,'qualified_gate',lambda b,ds:calls.append((b['correctness_evidence'],ds)))
    assert c.compatible_bindings(main,fresh,['alpaca'],'restarted')['identity_mode']=='restarted'
    assert calls and c.reference(fresh)['instances'][0]['provenance']['pid']==1


@pytest.mark.parametrize('change',['old_gate','old_started','old_host_pid','changed_output','empty_proof','large_inputs','missing_host_source'])
def test_restart_rejects_stale_binding(tmp_path,monkeypatch,change):
    main,fresh=pair(tmp_path);b=c.reference(fresh);old=c.reference(main)
    monkeypatch.setattr(c,'qualified_gate',lambda *args:None)
    if change=='old_gate':b['correctness_evidence']=old['correctness_evidence']
    if change=='old_started':b['instances'][0]['container']['StartedAt']='old'
    if change=='old_host_pid':
        inv=Path(fresh['path']).parent/'identity.json';v=c.read(inv);v[0]['State']['Pid']=101;write(inv,v);b['files'][str(inv)]=c.sha(inv)
    if change=='changed_output':b['output']+='/different'
    if change=='empty_proof':b['mechanism_proof']={}
    if change=='large_inputs':b['large_inputs']={'model':'changed'}
    if change=='missing_host_source':b['files'].pop(str(Path(b['host_release'])/'source.py'))
    rewrite(fresh,b)
    with pytest.raises(ValueError):c.compatible_bindings(main,fresh,['alpaca'],'restarted')


def invocation_fixture(tmp_path):
    main,fresh=pair(tmp_path);b=c.reference(main);out=Path(b['output']);row={'cell_id':'cell','phase':'scale','system':'mixed','dataset':'alpaca'}
    rec=out/'operations/cell/receipt.json';write(rec,dict(started_s=20,finished_s=30))
    write(out/'checkpoints/cell.json',dict(receipt=str(rec),completed_s=31))
    inv=out/'invocations/scale-1.json'
    write(inv,dict(started_s=10,finished_s=40,completed=['cell'],phase='scale',system='mixed',protocol_id=c.PROTOCOL,
        manifest_sha256='f'*64,binding_sha256=fresh['sha256'],complete=True,selected_datasets=['alpaca']))
    return row,main,fresh,inv


def test_cp_actual_execution_binding_not_main(tmp_path):
    row,main,fresh,inv=invocation_fixture(tmp_path)
    assert c.executed_binding(row,[main,fresh],'f'*64)[0]==fresh
    with pytest.raises(ValueError):c.executed_binding(row,[main],'f'*64)


@pytest.mark.parametrize('change',['skipped','unfinished','wrong_phase','wrong_manifest','wrong_group','duplicate'])
def test_cp_cannot_claim_skipped_or_nonterminal_invocation(tmp_path,change):
    row,main,fresh,p=invocation_fixture(tmp_path);v=c.read(p)
    if change=='skipped':v['completed']=[]
    if change=='unfinished':v['finished_s']=None
    if change=='wrong_phase':v['phase']='main'
    if change=='wrong_manifest':v['manifest_sha256']='0'*64
    if change=='wrong_group':v['selected_datasets']=['longbench']
    if change=='duplicate':write(p.parent/'duplicate.json',v)
    write(p,v)
    with pytest.raises(ValueError):c.executed_binding(row,[main,fresh],'f'*64)


def test_scale_only_one_cell_command():
    args=s.make_argv('/manifest','/binding','distserve',['longbench'],'/map','a'*64)
    assert args[args.index('--phase')+1]=='scale' and args[args.index('--max-cells')+1]=='1'
    assert args[-2:]==['--dataset','longbench']
    assert '--run' in args and str(c.DRIVER) in args


def test_stop_and_original_deadline(tmp_path,monkeypatch):
    ref=write(tmp_path/'spec.json',{});a=SimpleNamespace(out=tmp_path/'out',spec=Path(ref['path']),spec_sha256=ref['sha256'])
    sup=s.Supervisor(a);sup.stopped=True
    with pytest.raises(ValueError):sup.boundary()
    sup.stopped=False;monkeypatch.setattr(s.time,'time',lambda:c.DEADLINE-519)
    with pytest.raises(ValueError):sup.boundary()


def test_actual_b_gate_mixed_pass_does_not_authorize_eco(tmp_path):
    p=c.CAMPAIGN/'B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json';b=c.read(p)
    result=c.qualified_gate(b,['alpaca','sharegpt','longbench'])
    assert result['verified']=={'ordinary':True,'pd':True,'temporal':False}
    assert result['overall_runtime_gate_passed'] is False and result['physical']['all8_energy_j']>0
    broken=copy.deepcopy(b);q=tmp_path/'ecoserve.json';write(q,dict(c.read(b['configs']['alpaca']),strategy='ecoserve'))
    broken['system']='ecoserve';broken['configs']={'alpaca':str(q)}
    # Even a nonempty/falsified mechanism dictionary cannot override raw tokens.
    broken['mechanism_proof']={'required':['ordinary','temporal'],'verified':{'ordinary':True,'pd':True,'temporal':True}}
    with pytest.raises(RuntimeError,match='required mechanism failed'):c.qualified_gate(broken,['alpaca'])


@pytest.mark.parametrize('change',['host_pid','provenance'])
def test_actual_cp_process_change_is_rejected(change):
    p=c.CAMPAIGN/'C7B-five-system100-baselines-v1/bindings/mixed-resident/binding.json';b=c.read(p)
    cp=next(c.read(q) for q in (Path(b['output'])/'checkpoints').glob('*.json') if c.read(q)['row']['phase']=='scale')
    class Reader:
        def read(self,path,digest=None):
            value=copy.deepcopy(c.read(path))
            if Path(path).name=='identity.after.json':
                if change=='host_pid':value[0]['container']['State']['Pid']+=1
                else:value[0]['provenance']['source_files_at_import']={'different':'0'*64}
            return value
    with pytest.raises(ValueError):c.point_process(cp['row'],b,cp,Reader())


def test_frozen_power_checker_function_is_original_ast():
    import ast
    parent=c.CAMPAIGN.parent/'releases/five-system100-A14B-baseline-v1-runtime/src/ecopadg/serving/measurement.py'
    f=lambda path:next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='power_evidence')
    assert ast.dump(f(parent))==ast.dump(f(c.HERE/'power_evidence.frozen.py'))
