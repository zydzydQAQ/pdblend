import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as N
import pytest
import restore as r


def original():
    return r.read(r.ORIGINAL/'deployment.json'),r.read(r.ORIGINAL/'deployment-receipt.json'),r.read(r.ORIGINAL/'containers.after.json')

def previous():
    base,receipt,inv=original();ii=[]
    for j,(tp,gpus) in enumerate([(1,[x]) for x in range(5)]+[(2,[6,7])]):
        ii.append(dict(id='hetero'+str(j),tp=tp,gpus=gpus,native_kind='legacy_sync_put',
            container=dict(name='hetero'+str(j),id=str(j)*64,image=r.IMAGE,StartedAt='future-actual-only')))
    return dict(model='14b',system='distserve',hostname=r.HOST,protocol_id=r.PROTOCOL,deadline_s=r.DEADLINE,
        configs={'longbench':'synthetic-no-performance.json'},instances=ii)

def spec(tmp_path):
    b,p,iv=original();x=r.plan(b,p,iv,previous(),tmp_path/'new',tmp_path/'future-release.json','a'*64)
    previous_path=tmp_path/'previous.json';previous_path.write_text(json.dumps(previous()));x['previous_binding']=str(previous_path)
    return x

def stopped_inventory(s):
    rows=copy.deepcopy(list(s['expected_containers'].values()))
    for x in rows:x['State'].update(Running=False,Pid=0)
    for i in previous()['instances']:rows.append(dict(Name='/'+i['container']['name'],State={'Running':True}))
    return rows

def test_actual_original_eight_ids_and_no_new_command(tmp_path):
    s=spec(tmp_path)
    assert list(s['expected_containers'])==['pdb-v2-base100ar'+str(x) for x in range(8)]
    assert s['expected_containers']['pdb-v2-base100ar0']['Id']=='9ac6c0a2386545d0e8a7c22f6784ebf12464ecf37598262c58485cd64380a44f'
    assert s['bootstrap_configs']=={} and not s['output_correctness_verified'] and not s['new_container_creation_allowed']
    assert s['instances']==original()[0]['instances']

@pytest.mark.parametrize('what',['resident','mixed','wrong-model','missing-tp2'])
def test_wrong_previous_layout_rejected(what):
    p=previous()
    if what=='resident':p['configs']['alpaca']='x'
    if what=='mixed':p['system']='mixed'
    if what=='wrong-model':p['model']='7b'
    if what=='missing-tp2':p['instances'][-1]['tp']=1
    with pytest.raises(RuntimeError):r.validate_previous(p)

@pytest.mark.parametrize('what',['image','id','env','mount','running','foreign'])
def test_exact_retained_identity_required(tmp_path,what):
    s=spec(tmp_path);iv=stopped_inventory(s)
    if what=='image':iv[0]['Image']='other'
    if what=='id':iv[0]['Id']='f'*64
    if what=='env':iv[0]['Config']['Env'].append('OTHER=1')
    if what=='mount':iv[0]['Mounts'][0]['RW']=False
    if what=='running':iv[0]['State'].update(Running=True,Pid=3)
    if what=='foreign':iv.append(dict(Name='/unowned',State={'Running':True}))
    with pytest.raises(RuntimeError):r.verify_stopped(s,iv)

def test_new_startedat_and_host_pid_required(tmp_path):
    s=spec(tmp_path);before=r.verify_stopped(s,stopped_inventory(s));after=copy.deepcopy(before)
    with pytest.raises(RuntimeError):r.verify_restarted(s,before,after)
    for j,x in enumerate(after.values()):x['State'].update(Running=True,Pid=9000+j,StartedAt='2026-09-08T15:00:00Z')
    r.verify_restarted(s,before,after)
    after[next(iter(after))]['State']['StartedAt']=before[next(iter(before))]['State']['StartedAt']
    with pytest.raises(RuntimeError):r.verify_restarted(s,before,after)

def test_archive_preserves_control_and_entire_prefix(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir();cfg=tmp_path/'e.json';cfg.write_text(json.dumps({'runtime_dir':str(runtime)}))
    control=runtime/'e.control.json';events=runtime/'e.control.events.jsonl'
    control.write_text('{"generation":98}\n');events.write_text('{"old":1}\n{"old":2}\n')
    records=r.archive_runtime({'instances':[{'id':'e','config':str(cfg)}]},tmp_path/'out')
    control.write_text('{"generation":0}\n')
    with events.open('a') as f:f.write('{"new":1}\n')
    r.verify_prefixes(records)
    assert r.read(records[str(control)]['archive'])=={'generation':98}
    events.write_text('{"old":9}\n{"old":2}\n{"new":1}\n')
    with pytest.raises(RuntimeError):r.verify_prefixes(records)

def test_legacy_tp2_drain_preserved_and_bad_proof_no_resume(monkeypatch):
    calls=[];instance={'id':'x','tp':2,'native_kind':'legacy_sync_put'}
    async def idle(*a):return {'id':'x','generation':4,'acknowledged_generation':4}
    async def http(session,i,path,body,limit,records):calls.append(path);return {'drained':False}
    def barrier(*a):raise RuntimeError('bad two-rank proof')
    monkeypatch.setattr(r,'idle',idle);monkeypatch.setattr(r,'http',http)
    record={}
    with pytest.raises(RuntimeError):asyncio.run(r.native(None,instance,r.Limit(time.time()+1),[],N(barrier=barrier),record))
    assert calls==['/drain'] and record['complete'] is False and record['proof']=={'drained':False}

def test_cancel_propagates_keeps_partial_and_no_resume(monkeypatch):
    calls=[];instance={'id':'x','tp':1,'native_kind':'legacy_sync_put'}
    async def idle(*a):return {'id':'x','generation':4,'acknowledged_generation':4}
    async def http(session,i,path,body,limit,records):
        calls.append(path);records.append({'route':path,'issued':True});await asyncio.sleep(1)
    monkeypatch.setattr(r,'idle',idle);monkeypatch.setattr(r,'http',http)
    record={};raw=[]
    async def run():
        with pytest.raises(asyncio.TimeoutError):await asyncio.wait_for(r.native(None,instance,r.Limit(time.time()+2),raw,N(),record),.01)
    started=time.monotonic();asyncio.run(run())
    assert time.monotonic()-started<.2 and calls==['/drain'] and raw and not record['complete']

def test_expired_limit_never_starts_process(monkeypatch):
    async def forbidden(*a,**kw):pytest.fail('process dispatched')
    monkeypatch.setattr(asyncio,'create_subprocess_exec',forbidden)
    with pytest.raises(asyncio.TimeoutError):asyncio.run(r.command(['docker','start','x'],r.Limit(time.time()-1),[]))

def test_full_energy_requires_all8_both_brackets_and_clock(tmp_path):
    s=N(samples=[(0.,[10.]*8),(1.,[20.]*8),(2.,[10.]*8)],frequency_samples=[(0.,[1500.]*8),(2.,[1500.]*8)],
        power_source={},power_metadata=[],error=None)
    evidence=lambda *a:{'power_source_verified':True}
    assert r.energy_evidence(s,.25,1.75,evidence)['energy_j']==195.
    with pytest.raises(RuntimeError):r.energy_evidence(s,-.1,1.75,evidence)
    s.frequency_samples[1][1][7]=0
    with pytest.raises(RuntimeError):r.energy_evidence(s,.25,1.75,evidence)

def test_old_gate_rejected_by_actual_binder_identity_function(tmp_path):
    gate=r.load(r.CAMPAIGN/'AC-baseline-binding-v2/gate_evidence.py','restore_old_gate_test')
    old=r.read(r.CAMPAIGN/'A14B-five-system100-baselines-v1/resident-correctness/identity.before.json')
    # Actual archived gate shape, with a future StartedAt in the binding only.
    rows=old;instances=[]
    for x in rows:
        p=x['provenance'];c=x['container']
        instances.append(dict(id=p['instance_id'],container=dict(id=c['Id'],image=c['Image'],StartedAt='future-restart'),provenance=p))
    (tmp_path/'identity.before.json').write_text(json.dumps(rows));(tmp_path/'identity.after.json').write_text(json.dumps(rows))
    with pytest.raises(RuntimeError,match='different container process'):gate.identities(tmp_path,instances)

def test_bootstrap_old_binder_scope_accepts_restore_contract(tmp_path):
    import sys
    sys.path.insert(0,str(r.CAMPAIGN/'AC-baseline-binding-v2'))
    binder=r.load(r.CAMPAIGN/'AC-baseline-binding-v2/bind.py','restore_binder_compat_test')
    s=spec(tmp_path);binder.scope(s);assert binder.selected(s,None,None)==[]
    assert binder.selected(s,'mixed',None)==['alpaca','sharegpt','longbench']
    with pytest.raises(RuntimeError):binder.selected(s,'distserve',['longbench'])

def test_native_real_barrier_with_two_ranks_then_applied_resume(monkeypatch):
    common=r.load(r.COMMON,'restore_actual_native_test');instance={'id':'x','tp':2,'native_kind':'legacy_sync_put'}
    calls=[];raw=dict(id='x',generation=4,acknowledged_generation=4,accepting=True)
    async def idle(*a):return copy.deepcopy(raw)
    async def http(session,i,route,body,limit,records):
        calls.append((route,body))
        if route=='/drain':
            raw.update(generation=5,acknowledged_generation=5,accepting=False)
            return dict(drained=True,accepting=False,generation=5,drain_proof_type='synchronous_put_owner_barrier',
                transfers=[dict(buffered_tensors=0,inflight_receives=0,listener_alive=True,allocations={},buffered_gpu_bytes=0) for _ in range(2)])
        raw.update(body,acknowledged_generation=body['generation'],accepting=True);return {'generation':body['generation']}
    monkeypatch.setattr(r,'idle',idle);monkeypatch.setattr(r,'http',http)
    evidence={};asyncio.run(r.native(None,instance,r.Limit(time.time()+1),[],common,evidence))
    assert evidence['complete'] and raw['generation']==6 and raw['role']=='mixed'
    assert calls[-1][1]==dict(generation=6,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
    assert 'scheduler_budget' not in calls[-1][1] # Legacy TP2 is not v3.

def test_initial_ack_minus1_requires_actual_control(monkeypatch):
    i={'id':'x','tp':1,'native_kind':'legacy_sync_put'};calls=[]
    raw=dict(id='x',generation=0,acknowledged_generation=-1,accepting=True,transport_healthy=True,
        active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},transfer_buffered_tensors=0,
        transfer_inflight_receives=0)
    original=copy.deepcopy(raw)
    async def http(session,instance,route,body,limit,records):
        calls.append((route,body))
        if route=='/provenance':return {'instance_id':'x','pid':1}
        if route=='/control':raw.update(body,acknowledged_generation=body['generation']);return {'generation':body['generation']}
        return copy.deepcopy(raw)
    async def idle(*a):
        assert raw['generation']==raw['acknowledged_generation']==1
        return copy.deepcopy(raw)
    monkeypatch.setattr(r,'http',http);monkeypatch.setattr(r,'idle',idle)
    p,end=asyncio.run(r.ready(None,i,{'instance_id':'x','pid':1},r.Limit(time.time()+1),[],N()))
    assert original['acknowledged_generation']==-1 and end['acknowledged_generation']==1
    assert [x[0] for x in calls]==['/provenance','/runtime','/control']

def test_initial_wrong_source_never_controls(monkeypatch):
    calls=[]
    async def http(session,i,route,body,limit,records):calls.append(route);return {'instance_id':'wrong','pid':1}
    monkeypatch.setattr(r,'http',http)
    with pytest.raises(RuntimeError):asyncio.run(r.ready(None,{'id':'x'},{'instance_id':'x'},r.Limit(time.time()+1),[],N()))
    assert calls==['/provenance']

def test_sampler_failure_revokes_future_work(monkeypatch):
    async def forbidden(*a,**kw):pytest.fail('new process after sampler failure')
    monkeypatch.setattr(asyncio,'create_subprocess_exec',forbidden)
    limit=r.Limit(time.time()+1,lambda:r.require(False,'power failed'))
    with pytest.raises(RuntimeError,match='power failed'):asyncio.run(r.command(['docker','start','x'],limit,[]))

@pytest.mark.parametrize('power_ready_ok',[True,False])
def test_complete_restart_flow_receipt_and_preoperation_power_failure(tmp_path,monkeypatch,power_ready_ok):
    """Execute the actual orchestration with fake OS/HTTP/NVML adapters, never GPUs."""
    import sys,types
    s=spec(tmp_path);p=previous()
    for i in p['instances']:i['provenance']={};i['url']='http://not-executed'
    Path(s['previous_binding']).write_text(json.dumps(p));s['files']={}
    out=Path(s['out']);out.mkdir();spec_path=out/'deployment.json';spec_path.write_text(json.dumps(s))
    inventory=stopped_inventory(s)
    for i,x in zip(p['instances'],inventory[8:]):
        x.update(Id=i['container']['id']);x['State'].update(StartedAt=i['container']['StartedAt'],Pid=99)
    table={x['Name'].lstrip('/'):x for x in inventory};commands=[];http_records=[]
    original_load=r.load
    common=N(validate_binding=lambda b:None)
    barrier=N(verify_release=lambda *a:{'released':True},process_scan=lambda:{'no_live_serving_child':True})
    async def power_ready(*a):
        if not power_ready_ok:raise RuntimeError('first power readiness failed')
    deploy=N(terminal_group=lambda *a,**k:{'counts':{'main':10,'scale':6}},await_power_ready=power_ready,
        free_gpu_snapshot=lambda h:{'gpus':[{'used_bytes':0,'compute_pids':[],'graphics_pids':[]} for _ in range(8)]})
    def loader(path,name):
        if path==r.COMMON:return common
        if path==r.BARRIER:return barrier
        if path==r.DEPLOY:return deploy
        return original_load(path,name)
    monkeypatch.setattr(r,'load',loader);monkeypatch.setattr(r,'package_check',lambda:None)
    monkeypatch.setattr(r,'validate_spec',lambda s:True);monkeypatch.setattr(r,'verify_files',lambda f:None)
    monkeypatch.setattr(r.socket,'gethostname',lambda:r.HOST)
    monkeypatch.setattr(r,'DEADLINE',time.time()+10000)
    p['deadline_s']=r.DEADLINE;Path(s['previous_binding']).write_text(json.dumps(p))
    async def cmd(argv,limit,records,cap=40):
        limit.left();commands.append(argv);records.append({'argv':argv,'fake_cpu':True})
        if argv[:3]==['docker','ps','-aq']:return '\n'.join(table)
        if argv[:3]==['docker','ps','--format']:return '\n'.join(n for n,x in table.items() if x['State']['Running'])
        if argv[:2]==['docker','inspect']:return json.dumps([table[n] for n in argv[2:]])
        target=next(x for x in table.values() if x['Id']==argv[-1])
        if argv[1]=='stop':target['State'].update(Running=False,Pid=0)
        else:target['State'].update(Running=True,Pid=12000+int(target['Name'][-1]),StartedAt='2026-09-08T15:00:00Z')
        return target['Id']
    async def http(*a):http_records.append(a[2]);return {}
    async def idle(*a):return {'generation':1,'acknowledged_generation':1}
    async def ready(session,i,expected,*a):return expected,{'generation':1,'acknowledged_generation':1}
    async def native(session,i,limit,records,common,result):result.update(complete=True,errors=[],before={},proof={},resumed={})
    monkeypatch.setattr(r,'command',cmd);monkeypatch.setattr(r,'http',http);monkeypatch.setattr(r,'idle',idle)
    monkeypatch.setattr(r,'ready',ready);monkeypatch.setattr(r,'native',native)
    monkeypatch.setattr(r,'archive_runtime',lambda *a:{});monkeypatch.setattr(r,'verify_prefixes',lambda *a:None)
    # All adapters below are in-memory CPU functions; avoid real executor threads.
    async def fake_to_thread(fn,*a,**kw):return fn(*a,**kw)
    monkeypatch.setattr(asyncio,'to_thread',fake_to_thread)
    class Session:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
    class Sampler:
        def __init__(self,*a,**kw):
            now=time.time();self.samples=[(now-.2,[10.]*8),(now-.1,[10.]*8)]
            self.frequency_samples=[(now-.2,[1500.]*8),(now-.1,[1500.]*8)]
            self.utilization_samples=[];self.power_source={};self.power_metadata=[];self.error=None
        def start(self):pass
        def stop(self):
            now=time.time()+1;self.samples.append((now,[10.]*8));self.frequency_samples.append((now,[1500.]*8))
    modules={
        'ecopadg.measure.backends':{'PynvmlBackend':lambda **kw:N()},
        'ecopadg.measure.power':{'PowerSampler':Sampler},
        'ecopadg.serving.measurement':{'save_raw':lambda out,*a,**kw:(out/'power.csv').write_text('CPU-only retained power fixture\n'),
            'power_evidence':lambda *a:{'power_source_verified':True}}}
    aiohttp=types.ModuleType('aiohttp');aiohttp.ClientSession=Session;monkeypatch.setitem(sys.modules,'aiohttp',aiohttp)
    for name,values in modules.items():
        module=types.ModuleType(name);module.__dict__.update(values);monkeypatch.setitem(sys.modules,name,module)
    if power_ready_ok:
        result=asyncio.run(r.launch(spec_path))
        assert result['complete'] and result['measurement_valid'] and result['all8_operation_energy_j']>0
        assert len(result['created'])==len(result['restarted'])==len(result['new_provenance'])==8
        assert result['created']==result['restarted'] and not result['new_containers_created']
        assert not result['output_correctness_verified'] and not result['scale_executed']
        expected_ids={i['container_name']:s['expected_containers'][i['container_name']]['Id'] for i in s['instances']}
        assert {x['name']:x['container_id'] for x in result['created']}==expected_ids
        assert sum(x[:2]==['docker','start'] for x in commands)==8
        assert not any('run' in x or 'create' in x or 'rm' in x for x in commands)
    else:
        with pytest.raises(RuntimeError,match='first power readiness failed'):asyncio.run(r.launch(spec_path))
        result=r.read(out/'deployment-receipt.json')
        assert not result['complete'] and not result['measurement_valid']
        assert 'operation_start_s' not in result and result['all8_operation_energy_j'] is None
        assert (out/'deployment-power/power.csv').is_file() and (out/'deployment-power/clocks.csv').is_file()
        assert not any(x[1] in ('start','stop') for x in commands)
    assert (out/'deployment-receipt.json').is_file()
