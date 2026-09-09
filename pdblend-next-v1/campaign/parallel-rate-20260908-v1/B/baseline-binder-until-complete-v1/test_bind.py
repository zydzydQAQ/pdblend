"""Synthetic HTTP-free contract fixtures only; no new GPU evidence."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).with_name('bind.py')
spec=importlib.util.spec_from_file_location('b_eco_binder_test',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path):
    engine=tmp_path/'engine.py';engine.write_text('# CPU fixture\n')
    ii=[];inventory=[]
    for k in range(4):
        cid=str(k)*64;name=f'pdb-v2-base100b{k}'
        c=dict(Id=cid,Name='/'+name,Image=m.IMAGE,Path='python3',Args=['engine.py'],Config={'Env':['CPU_FIXTURE=1']},
            HostConfig={},Mounts=[],State=dict(Pid=100+k,StartedAt='2026-09-08T03:00:00Z',Running=True,Paused=False,Restarting=False))
        p=dict(instance_id=f'base100b{k}',pid=1,tp=2,model='/models/Qwen2.5-32B-Instruct',dtype='bfloat16',
            max_model_len=8192,cuda_visible_devices=f'{2*k},{2*k+1}',source_files_at_import={str(engine):m.sha(engine)})
        ii.append(dict(id=f'base100b{k}',tp=2,gpus=[2*k,2*k+1],role='mixed',port=34100+k,kv_port=34300+32*k,
            url=f'http://127.0.0.1:{34100+k}',native_kind='legacy_sync_put',scheduler_cache_observed=False,
            container=dict(id=cid,name=name,image=m.IMAGE,StartedAt=c['State']['StartedAt']),provenance=p))
        inventory.append(c)
    identity=tmp_path/'identity.json';m.write(identity,inventory)
    b=dict(schema=1,model='32b',hostname=m.EXPECTED_HOST,protocol_id=m.PROTOCOL,deadline_s=m.DEADLINE,
        host_release=str(m.HOST),configs={},window_s=100,seeds=[701],instances=ii,
        correctness_gate_required_before_performance=True,output_correctness_verified=False,
        files={str(engine):m.sha(engine),str(identity):m.sha(identity)},identity_file=str(identity),large_inputs={})
    gate=tmp_path/'gate';gate.mkdir()
    old=dict(complete=True,measurement_valid=True,native_cleanup_complete=True,clock_restore_complete=True,
        sampling_error=None,cleanup_errors=[],finished_s=1788835000,passed=False,
        mechanism_gate={'ordinary':True,'pd':True,'temporal':False})
    references=[list(range(64)),list(range(64))];outputs=copy.deepcopy(references);outputs[1][31]=999
    temporal=dict(complete=True,reference_token_ids=references,token_ids=outputs,
        first_differences=[None,dict(position_one_based=32,reference=31,observed=999)])
    m.write(gate/'status.json',old);m.write(gate/'checks/checks.json',dict(checks={'temporal_exact':False},temporal=temporal))
    observations=[dict(container=c,provenance=i['provenance']) for c,i in zip(inventory,ii)]
    for name in ('identity.before.json','identity.after.json'):m.write(gate/name,observations)
    q=dict(schema=2,kind='temporal-default-trajectory-qualification',protocol_id=m.CORRECTNESS,passed=True,
        eligible_systems={'ecoserve':True},verified={k:True for k in m.NEEDED},legacy_single_vs_pair_exact=False,
        legacy_failure_preserved=True,original_mechanism_gate=old['mechanism_gate'],
        files={str(p):m.sha(p) for p in gate.rglob('*') if p.is_file()})
    bp=tmp_path/'bootstrap.json';m.write(bp,b);oracle=tmp_path/'oracle.json';m.write(oracle,{'CPU_fixture_only':True})
    q['inputs']=dict(gate_dir=str(gate.resolve()),binding_canonical_sha256=m.canonical(b),
        oracle_canonical_sha256=m.canonical({'CPU_fixture_only':True}))
    a=SimpleNamespace(bootstrap=bp,bootstrap_sha256=m.sha(bp),oracle=oracle,oracle_sha256=m.sha(oracle),gate=gate,
        qualifier=tmp_path/'qualifier.py',qualifier_sha256='a'*64,out=tmp_path/'bound')
    return SimpleNamespace(b=b,inventory=inventory,gate=gate,q=q,args=a,tmp=tmp_path,observations=observations)


def test_real_original_restore_inventory_parser():
    base=m.C/'B32B-native-default-reference-attempt-001/results'
    b=m.read(base/'restored-bootstrap.binding.json');m.bootstrap_contract(b)
    ready=m.full_inventory(m.read(base/'restored-ready.json'),b)
    after=m.full_inventory(m.read(base/'restored-identity.after.json'),b)
    assert len(ready)==4 and [r['State']['Pid'] for r in ready]==[r['State']['Pid'] for r in after]


@pytest.mark.parametrize('mutation',[
    lambda b:b.update(model='14b'), lambda b:b.update(output_correctness_verified=True),
    lambda b:b.update(configs={'alpaca':'old.json'}), lambda b:b.update(window_s=300),
    lambda b:b['instances'][1].update(tp=1)])
def test_wrong_bootstrap_rejected(fixture,mutation):
    mutation(fixture.b)
    with pytest.raises(RuntimeError):m.bootstrap_contract(fixture.b)


@pytest.mark.parametrize('mutation',[
    lambda q:q.update(passed=False), lambda q:q['eligible_systems'].update(ecoserve=False),
    lambda q:q['verified'].update(clock=False), lambda q:q['verified'].pop('cancel'),
    lambda q:q.update(legacy_single_vs_pair_exact=True),lambda q:q.update(legacy_failure_preserved=False),
    lambda q:q.update(kind='temporal-default-trajectory-prequalification'),lambda q:q.update(files={})])
def test_qualification_failures_do_not_publish(fixture,mutation,monkeypatch):
    f=fixture;mutation(f.q);monkeypatch.setattr(m,'PERFORMANCE_OUTPUT',f.tmp/'results')
    monkeypatch.setattr(m,'load_qualifier',lambda *_:(SimpleNamespace(audit_fresh_gate=lambda *_:f.q),{}))
    with pytest.raises(RuntimeError):m.bind(f.args)
    assert not f.args.out.exists()


def test_same_namespace_pid_one_does_not_hide_host_restart(fixture):
    f=fixture;f.inventory[1]['State']['Pid']+=1000
    with pytest.raises(RuntimeError,match='another host process'):
        m.qualification_contract(f.q,f.b,f.gate,f.inventory)


def test_success_constructs_original_policy_and_preserves_failed_legacy(fixture,monkeypatch):
    f=fixture;monkeypatch.setattr(m,'PERFORMANCE_OUTPUT',f.tmp/'results')
    calls=[]
    def audit(gate,binding,oracle):
        calls.append((gate,binding,oracle));return copy.deepcopy(f.q)
    monkeypatch.setattr(m,'load_qualifier',lambda *_:(SimpleNamespace(audit_fresh_gate=audit),{}))
    receipt=m.bind(f.args);b=m.read(f.args.out/'binding.json')
    assert len(calls)==1 and calls[0][2]=={'CPU_fixture_only':True}
    assert receipt['complete'] and receipt['main_cells']==30
    assert b['correctness_protocol_id']==m.CORRECTNESS and b['legacy_single_vs_pair_exact'] is False
    assert b['mechanism_proof']['legacy_verified']['temporal'] is False
    assert b['mechanism_proof']['verified']['temporal_native_trajectory_exact'] is True
    assert m.read(f.gate/'status.json')['passed'] is False
    original=m.plan_module()
    for dataset,path in b['configs'].items():
        assert m.read(path)==original.configuration(original.read(original.C/f'B32B-five-system100-v1/configs/{dataset}.ecoserve.json'))
    m.verify_files(b['files'])
    with pytest.raises(RuntimeError):m.bind(f.args)


def test_bad_qualifier_sha_does_not_import(tmp_path,monkeypatch):
    module=tmp_path/'qualification.py';module.write_text("raise AssertionError('must not import')\n")
    monkeypatch.setattr(m,'QUALIFIER_DIR',tmp_path)
    with pytest.raises(RuntimeError,match='SHA/path'):m.load_qualifier(module,'0'*64)


def test_derive_original_raw_is_unchanged(fixture):
    f=fixture;restore=f.tmp/'restore';restore.mkdir()
    f.b.pop('identity_file');status=restore/'status.json';f.b['restoration_evidence']=str(status)
    bp=restore/'restored-bootstrap.binding.json';m.write(bp,f.b)
    ready=restore/'restored-ready.json';after=restore/'restored-identity.after.json'
    m.write(ready,f.observations);m.write(after,f.observations)
    m.write(status,dict(restored_binding=str(bp),restored_binding_sha256=m.sha(bp),all_original_restored=True,
        measurement_valid=True,clock_restore_complete=True,sampling_error=None,finished_s=1788835000,
        restored_native={'complete':True},full_operation_energy_j=100))
    before={str(p):m.sha(p) for p in restore.iterdir()}
    a=SimpleNamespace(bootstrap=bp,bootstrap_sha256=m.sha(bp),identity=ready,identity_sha256=m.sha(ready),
        restore_status_sha256=m.sha(status),out=f.tmp/'derived')
    result=m.derive_bootstrap(a);b=m.read(result['binding'])
    assert result['performance_eligible'] is False and b['configs']=={}
    assert b['identity_file']==str(a.out/'identity.json') and b['output_correctness_verified'] is False
    assert all(m.sha(p)==h for p,h in before.items())
    m.verify_files(b['files'])


def test_actual_fresh27_missing_header_preserved():
    checks=m.read(m.C/'B32B-temporal-matched-shape-fresh-gate-v1/checks/checks.json')
    original=copy.deepcopy(checks);result=m.legacy_exact_evidence(checks)
    assert result['header_present'] is False and result['header_value'] is None and result['recomputed_exact'] is False
    assert result['first_differences']==[None,dict(position_one_based=32,reference=2776,observed=4172)]
    assert checks==original and 'temporal_exact' not in checks['checks']


@pytest.mark.parametrize('kind',['null_header','true_header','zero_header','short_output','false_summary','actually_equal'])
def test_missing_header_does_not_relax_raw_evidence(fixture,kind):
    checks=m.read(fixture.gate/'checks/checks.json');checks['checks'].pop('temporal_exact')
    if kind=='null_header':checks['checks']['temporal_exact']=None
    elif kind=='true_header':checks['checks']['temporal_exact']=True
    elif kind=='zero_header':checks['checks']['temporal_exact']=0
    elif kind=='short_output':checks['temporal']['token_ids'][1].pop()
    elif kind=='false_summary':checks['temporal']['first_differences']=[None,None]
    elif kind=='actually_equal':checks['temporal']['token_ids']=copy.deepcopy(checks['temporal']['reference_token_ids'])
    with pytest.raises(RuntimeError):m.legacy_exact_evidence(checks)
