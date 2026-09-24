"""CPU contracts for same-fleet PD runtime collection; no CUDA/NVML calls."""
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.collection.native_runtime_collect import (
    NativeRuntimeCollector, RUNTIME_PLAN, collect_runtime, snapshot_sampler, validate_inventory, write_new,
)
from pdblend.profile.collection.native_runtime_audit import replay_runtime
from pdblend_runtime.probe import NativeSpec


UUIDS=['GPU-test-'+str(i) for i in range(8)]
SOURCE=dict(mode='instant',field_id=186,scope_id=0,source_id='nvml:field:186:scope:0:mW')


def state(generation=0):
    return dict(generation=generation,tp=1,pp=1,native_evidence_complete=True,
        transport_healthy=True,native_at_s=10**12,ranks=[dict(rank=0,generation=generation,
        native_evidence_complete=True,healthy=True,pending_transfers=0,transfer_allocations={},acknowledged=True)],
        all_queue=[],running=[],waiting=[],retained_kv_requests=[],pending_transfers=0,
        transfer_allocations={},kv_allocations={},total_blocks=100,free_blocks=100,reserved_blocks=0,
        total_kv_tokens=1600,max_num_seqs=32,accepting=False,acknowledged=True,drained=True)


def inventory():
    specs=[NativeSpec('i'+str(i),(i,),10000+i*4,'/models/Qwen2.5-7B-Instruct',
                      max_num_seqs=32,extra_args=('--enforce-eager','--worker-cls','PDNativeTimingWorker')) for i in range(8)]
    backend=SimpleNamespace(gpu_uuid=lambda g:UUIDS[g])
    meter=SimpleNamespace(gpus=list(range(8)),backend=backend)
    instances={s.instance_id:SimpleNamespace(spec=s,alive=lambda:True,events=[dict(kind='start',t_s=1)]) for s in specs}
    class Fleet:
        def __getitem__(self,k):return instances[k]
    fleet=Fleet();fleet.instances=instances
    sampler=SimpleNamespace(backend=backend,gpus=list(range(8)),sample_clocks=True,error=None,
        power_source=dict(SOURCE),samples=[],power_metadata=[],frequency_samples=[],
        utilization_samples=[],utilization_readings=[],utilization_errors=[],utilization_source={},
        error_at_s=None,interval=.1,_thread=SimpleNamespace(is_alive=lambda:True))
    return specs,fleet,meter,sampler


def test_inventory_binds_real_uuids_and_native32_before_mutation():
    specs,fleet,meter,sampler=inventory()
    assert validate_inventory(specs,fleet,meter,sampler,UUIDS)['gpu_uuid_binding_verified']
    assert specs[0].extra_args[-1]=='PDNativeTimingWorker'
    sampler.power_source['mode']='average'
    with pytest.raises(ValueError,match='instant'):validate_inventory(specs,fleet,meter,sampler,UUIDS)
    sampler.power_source=dict(SOURCE);meter.backend.gpu_uuid=lambda g:UUIDS[7-g]
    with pytest.raises(ValueError,match='UUIDs'):validate_inventory(specs,fleet,meter,sampler,UUIDS)


def test_inventory_rejects_historical_256_and_second_sampler_backend():
    specs,fleet,meter,sampler=inventory();specs[0]=replace(specs[0],max_num_seqs=256)
    fleet['i0'].spec=specs[0]
    with pytest.raises(ValueError,match='native32'):validate_inventory(specs,fleet,meter,sampler,UUIDS)
    specs[0]=replace(specs[0],max_num_seqs=32);sampler.backend=object()
    with pytest.raises(ValueError,match='same eight'):validate_inventory(specs,fleet,meter,sampler,UUIDS)


def test_live_sampler_snapshot_uses_complete_prefix_without_stopping():
    _,_,_,sampler=inventory();sampler.samples=[(1,[10]*8)]
    sampler.power_metadata=[dict(t_s=1),dict(t_s=2)]
    raw=snapshot_sampler(sampler,UUIDS)
    assert len(raw['samples'])==len(raw['power_metadata'])==1
    sampler.samples.append((2,[11]*8))
    assert len(raw['samples'])==1 and len(sampler.samples)==2


def test_operation_failure_preserves_raw_error_and_interval(tmp_path):
    specs,fleet,meter,sampler=inventory()
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS)
    async def fail():raise RuntimeError('native clock rejected')
    with pytest.raises(RuntimeError,match='rejected'):
        asyncio.run(runner.operation('clock',specs[0],0,fail))
    row=json.loads((tmp_path/'journal.jsonl').read_text())
    assert row['status']=='failed' and row['finished_s']>=row['started_s']
    assert 'rejected' in row['error']


def test_off_requires_owned_stop_and_actual_compute_empty(tmp_path):
    specs,fleet,meter,sampler=inventory();instance=fleet['i0']
    instance.alive=lambda:False;instance.state='off';instance.events.append(dict(kind='stop',t_s=2))
    meter.backend._handle=lambda g:g
    meter.backend._nvml=SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=lambda h:[SimpleNamespace(pid=123)])
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS)
    with pytest.raises(RuntimeError,match='retains compute'):
        asyncio.run(runner.compute_empty(specs[0],timeout_s=0))
    meter.backend._nvml.nvmlDeviceGetComputeRunningProcesses=lambda h:[]
    receipt=asyncio.run(runner.compute_empty(specs[0],timeout_s=0))
    assert receipt['compute_processes_gone'] and receipt['observations'][-1]['devices'][0]['uuid']==UUIDS[0]
    instance.events.pop()
    with pytest.raises(ValueError,match='owned'):asyncio.run(runner.compute_empty(specs[0],timeout_s=0))


def test_restore_attempts_entire_inventory_after_one_endpoint_fails(tmp_path,monkeypatch):
    specs,fleet,meter,sampler=inventory();meter.unpark=lambda g:None;meter.set_clock=lambda g,f:None
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS)
    observed=[]
    async def request(spec,endpoint,payload=None):
        observed.append((spec.instance_id,endpoint))
        if spec.instance_id=='i0':raise RuntimeError('lost endpoint')
        return state()
    async def receipt(spec,*args,**kwargs):return state()
    runner.request=request;runner.drain=receipt;runner.resume=receipt;runner.clock=receipt;runner.capability=receipt
    result=asyncio.run(runner.restore())
    assert not result['passed'] and len(result['instances'])==8
    assert ('i7','measurement/stop') in observed
    assert result['final_generation']==0 and result['measurement_stopped'] is None


def test_collection_failure_restores_inventory_and_keeps_caller_sampler_owned(tmp_path,monkeypatch):
    specs,fleet,meter,sampler=inventory();sampler.samples=[(10**12,[100]*8)]
    sampler.frequency_samples=[(10**12,[2520]*8)]
    sampler.power_metadata=[dict(read_finished_s=[10**12]*8)]
    restored=[]
    class Session:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr('pdblend.profile.collection.native_runtime_collect.aiohttp.ClientSession',Session)
    async def cap(self,spec):return dict(supported=True)
    async def receipt(self,spec,*args):return state()
    async def fail(self,spec):raise RuntimeError('actual clock failed')
    async def restore(self):
        restored.extend(s.instance_id for s in self.specs)
        return dict(passed=True,errors=[],instances=[],actual_starts={s.instance_id:[dict(kind='start',t_s=1)] for s in self.specs})
    monkeypatch.setattr(NativeRuntimeCollector,'capability',cap)
    for name in ('drain','resume','ordinary_probe','request'):monkeypatch.setattr(NativeRuntimeCollector,name,receipt)
    monkeypatch.setattr(NativeRuntimeCollector,'static_phases',fail)
    monkeypatch.setattr(NativeRuntimeCollector,'restore',restore)
    report=asyncio.run(collect_runtime(specs,fleet,meter,sampler,tmp_path/'new',gpu_uuids=UUIDS))
    assert report['status']=='failed' and 'actual clock failed' in report['error']
    assert report['restoration']['passed'] and not report['ready_for_timing']
    assert restored==[s.instance_id for s in specs] and len(sampler.samples)==1
    assert report['necessary_mechanism_loads']==0
    assert (tmp_path/'new/completion.json').is_file()


def artifact(tmp_path):
    specs,_,_,_=inventory();rows=[];clock=100.
    def add(row):
        row['sequence']=len(rows)
        if 'repeat' in row:row['purpose']='training' if row['repeat']<3 else 'holdout'
        rows.append(row)
    caps={s.instance_id:dict(supported=True,model_id='Qwen2.5-7B-Instruct',tp=1,pp=1,gpu_uuids=[UUIDS[s.gpus[0]]]) for s in specs}
    def off(at):
        return dict(compute_processes_gone=True,owned_stop=dict(kind='stop'),started_s=at,finished_s=at+.2,
                    observations=[dict(at_s=at+.1,devices=[dict(gpu=0,uuid=UUIDS[0],compute_pids=[])])])
    def clock_receipt(f,at):
        return dict(ack=dict(acknowledged=True,success=True,requested_frequency_mhz=f,gpus=[dict(gpu_uuid=UUIDS[0])]),
                    observations=[dict(at_s=at,frequencies_mhz=[f])])
    for s in specs:add(dict(kind='capacity',instance_id=s.instance_id,capability=caps[s.instance_id],state=state()))
    for _ in range(6):
        add(dict(kind='ordinary_golden',instance_id='i0',result=dict(error=None,stream_done=True,
            usage_received=True,prompt_tokens=512,completion_tokens=16,token_ids=list(range(16)))))
    for repeat in range(4):
        for name in ('active_idle@1500','active_idle@2520','active_idle_reset','L1','off','clock_target@1500','clock_target@2520'):
            add(dict(kind='static',state=name,repeat=repeat,instance_id='i0',gpus=[0],
                     settle_started_s=clock,started_s=clock+2,finished_s=clock+7,
                     before=None if name=='off' else dict(state(),native_at_s=clock-.1),
                     after=None if name=='off' else dict(state(),native_at_s=clock+7.1),
                     off_before=off(clock-.3) if name=='off' else None,
                     off_evidence=off(clock+7.1) if name=='off' else None,
                     memory_before_mhz=[405],memory_after_mhz=[405]))
            clock+=8
        for name in ('park','unpark','off','wake','clock_2520_to_1500','clock_1500_to_2520'):
            receipt={}
            if name=='off':receipt=dict(off=off(clock+.1),drain=state())
            if name=='wake':receipt=dict(ordinary=dict(error=None,stream_done=True,completion_tokens=16),
                drain=state(),resume=dict(state=state()),capability=caps['i0'])
            if name=='park':receipt=dict(drain=state(),memory_mhz=[405],observed_mhz=[210])
            if name=='unpark':receipt=dict(clock=clock_receipt(2520,clock),drain=state(),resume=dict(state=state()))
            if name.startswith('clock_'):receipt=clock_receipt(int(name.rsplit('_',1)[1]),clock)
            add(dict(kind='operation',operation=name,instance_id='i0',gpus=[0],repeat=repeat,
                     started_s=clock,finished_s=clock+1,status='passed',receipt=receipt));clock+=2
    samples=[];metadata=[];frequencies=[]
    for n in range(int((clock-99)*4)+8):
        t=99+n/4;freq=[2520]*8
        for row in rows:
            if row['kind']=='static' and row['started_s']<=t<row['finished_s']:
                name=row['state'];freq[0]=210 if name=='L1' else int(name.split('@')[1]) if '@' in name else 2520
        samples.append([t,[100]*8]);frequencies.append([t,freq])
        metadata.append(dict(t_s=t,gpus=list(range(8)),read_finished_s=[t]*8,
            mode=['instant']*8,source_id=[SOURCE['source_id']]*8,field_id=[186]*8,
            scope_id=[0]*8,return_code=[0]*8,value_type=[1]*8))
    power=dict(gpus=list(range(8)),gpu_uuids=UUIDS,samples=samples,power_metadata=metadata,
               frequency_samples=frequencies,power_source=SOURCE,error=None)
    power_ref=write_new(tmp_path/'power.json',power)
    journal=tmp_path/'journal.jsonl';journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=dict(complete=True,ready_for_timing=True,repeats=3,settle_s=2,measure_s=5,
        runtime_plan=deepcopy(RUNTIME_PLAN),
        runtime_plan_binding=write_new(tmp_path/'runtime-plan.json',RUNTIME_PLAN),
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],initial_capabilities=caps,
        restoration=dict(passed=True,errors=[],instances=[dict(instance_id=s.instance_id,drain=state(),process_alive=True,
            resume=dict(state=state()),measurement_stop=dict(ranks=[dict(rank=0,acknowledged=True)]),
            clock=dict(ack=dict(requested_frequency_mhz=2520))) for s in specs]),
        power=power_ref,journal=dict(path=str(journal),sha256=hashlib.sha256(journal.read_bytes()).hexdigest()),
        lease=dict(gpu_ids=list(range(8)),gpu_uuids=UUIDS,gpu_uuid_binding_verified=True),measured_components=['capacity','static','clock','L1','off_wake'])
    return report,power,rows


def replace_bound(tmp_path,report,key,value):
    if key=='power':report[key]=write_new(tmp_path/'new-power.json',value)
    else:
        path=tmp_path/'new-journal.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in value))
        report[key]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_raw_energy_replay_keeps_full_profile_and_incremental_energy_unqualified(tmp_path):
    report,_,_=artifact(tmp_path);audit=replay_runtime(report)
    assert audit['status']=='passed',audit['errors']
    assert audit['raw_components_complete'] and not audit['formal_eligible'] and not audit['energy_comparable']
    assert audit['phases'][0]['energy_j']==4000 and audit['phases'][0]['incremental_energy_j'] is None
    assert audit['holdout_passed'] and len(audit['holdout_comparisons'])==13


def test_real_holdout_miss_retains_raw_evidence_and_safe_timing_boundary(tmp_path):
    report,_,rows=artifact(tmp_path)
    held=next(r for r in rows if r.get('operation')=='wake' and r['repeat']==3)
    held['finished_s']+=.75
    replace_bound(tmp_path,report,'journal',rows)
    audit=replay_runtime(report)
    assert audit['raw_components_complete'] and not audit['holdout_passed']
    assert audit['holdout_error_summary']['max_relative_error']>.25
    assert report['ready_for_timing'] and not audit['component_qualified']


@pytest.mark.parametrize('failure',['average','gap','wrong_frequency','missing_wake','not_restored','wrong_capacity','changed_wake_golden'])
def test_raw_replay_refuses_missing_or_incompatible_runtime_evidence(tmp_path,failure):
    report,power,rows=artifact(tmp_path)
    if failure=='average':
        power['power_source']=dict(SOURCE,mode='average');replace_bound(tmp_path,report,'power',power)
    elif failure=='gap':
        for key in ('samples','power_metadata'):
            power[key]=[r for r in power[key] if not 102<= (r[0] if isinstance(r,list) else r['t_s']) <=104]
        replace_bound(tmp_path,report,'power',power)
    elif failure=='wrong_frequency':
        power['frequency_samples'][14][1][0]=2520;replace_bound(tmp_path,report,'power',power)
    elif failure=='missing_wake':
        rows=[r for r in rows if not (r.get('operation')=='wake' and r['repeat']==2)]
        for i,r in enumerate(rows):r['sequence']=i
        replace_bound(tmp_path,report,'journal',rows)
    elif failure=='not_restored':report['restoration']['passed']=False
    elif failure=='changed_wake_golden':
        next(r for r in reversed(rows) if r['kind']=='ordinary_golden')['result']['token_ids'][0]=99
        replace_bound(tmp_path,report,'journal',rows)
    else:
        rows[0]['state']['max_num_seqs']=256;replace_bound(tmp_path,report,'journal',rows)
    audit=replay_runtime(report)
    assert not audit['raw_components_complete'] and audit['errors']
