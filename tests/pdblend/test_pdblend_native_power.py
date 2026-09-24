import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pdblend.profile.collection.native_power_plan import build_plan,validate_plan,binding,short_context_limit
from pdblend.profile.collection.native_power_audit import audit_power_window,context_window
from test_comparison_metering import evidence,UUIDS
from test_comparison_acceptance import state
from test_pdblend_native_timing import event


def fixture_plan(tmp_path):
    ledger=dict(schema='pdblend-offline-query-ledger/v2',evaluation_read=False,min_m_floor_overridden=False,
        ledgers=[dict(model_id='Qwen2.5-7B-Instruct',selection_split='tuning',tp=1,pp=1,
            min_m_instances=4,engine_max_num_seqs=32,frequency_scope=[1500,2520],dataset='alpaca',rate_scale=.5,
            queries=[dict(method='decode_supported',args=[1.,121.0214,1500],kwargs={},finite_arguments=True),
                     dict(method='decode_power_w',args=[1.55,2520],kwargs={'ctx':482.7},finite_arguments=True),
                     dict(method='prefill_seconds',args=[33,1500],kwargs={},finite_arguments=True)])])
    path=tmp_path/'ledger.json';path.write_text(json.dumps(ledger))
    provenance=tmp_path/'bindings.json';provenance.write_text(json.dumps(dict(evaluation_read=False,
        outputs={'ledger':binding(path)})))
    return build_plan(binding(path),binding(provenance))


def test_plan_is_query_bound_and_does_not_promote_nominal_context(tmp_path):
    plan=validate_plan(fixture_plan(tmp_path))
    assert len(plan['points'])==12 and sum(p['repeats'] for p in plan['points'])==36
    assert len(plan['actual_queries'])==3
    assert not plan['formal_eligible'] and not plan['power_component_qualified']
    assert all(p['purpose']=='coverage_feasibility_pilot' for p in plan['points'])
    plan['points'][0]['prompt_tokens']=121
    with pytest.raises(ValueError):validate_plan(plan)


def test_short_context_bound_is_conditional_not_unmeasured_native_proof():
    result=short_context_limit(121.)
    assert result['necessary_constant_step_s']==pytest.approx(4.5/105.)
    assert result['native_impossibility_proved'] is False


def test_power_plan_supports_model_owned_32b_tp2_without_claiming_pilot_qualification(tmp_path):
    plan=fixture_plan(tmp_path);path=Path(plan['query_ledger']['path']);value=json.loads(path.read_text())
    value['ledgers'][0].update(model_id='Qwen2.5-32B-Instruct',tp=2);path.write_text(json.dumps(value))
    provenance=Path(plan['query_provenance']['path']);provenance.write_text(json.dumps(dict(evaluation_read=False,
        outputs={'ledger':binding(path)})))
    plan=build_plan(binding(path),binding(provenance),model_id='Qwen2.5-32B-Instruct')
    assert plan['tp']==2 and not plan['formal_eligible']
    validate_plan(plan)


def test_zero_marginal_identity_is_recorded_without_fake_gpu_request(tmp_path):
    plan=fixture_plan(tmp_path);path=Path(plan['query_ledger']['path'])
    ledger=json.loads(path.read_text());ledger['ledgers'][0]['queries'].append(
        dict(method='prefill_marginal_seconds',args=[0,1500],kwargs={},finite_arguments=True))
    path.write_text(json.dumps(ledger));ref=binding(path)
    provenance=Path(plan['query_provenance']['path']);value=json.loads(provenance.read_text())
    value['outputs']['ledger']=ref;provenance.write_text(json.dumps(value))
    result=build_plan(ref,binding(provenance))
    assert len(result['zero_marginal_queries'])==1
    assert all(p['prompt_tokens']>0 for p in result['points'])


def raw_fixture(role='decode'):
    times=[99.+i*.1 for i in range(91)];power=evidence(times,watts=100.)
    power['frequency_samples']=[(t,[1500.]+[2520.]*7) for t in times]
    drained=dict(state(1,109.,5),acknowledged=True,drained=True,total_blocks=100,free_blocks=100,reserved_blocks=0)
    rows=[event(batch=1,context=129+i,at=100.+i*.1,role=role) for i in range(70)]
    if role=='prefill':
        for row in rows:row.update(context_lengths=[128],scheduled_lengths=[128])
    requests=[dict(request_id='r0',submitted_s=99.,observed_running_through_s=107.,
        terminal=role=='prefill',completion_tokens=1 if role=='prefill' else 70,cancel_expected=role=='decode')]
    if role=='prefill':
        requests=[]
        for i,row in enumerate(rows):
            row['request_ids']=[f'r{i}']
            requests.append(dict(request_id=f'r{i}',terminal=True,completion_tokens=1,submitted_s=row['at_s']-.01))
    return dict(schema='pdblend-native-power-window/v1',system='pdblend',status='measured',
        point=dict(role=role,batch=1,prompt_tokens=128,frequency_mhz=1500),
        spec=dict(tp=1,pp=1,gpus=[0],max_num_seqs=32,generation=5),
        lease=dict(gpu_ids=list(range(8)),gpu_uuids=UUIDS,gpu_uuid_binding_verified=True),
        capability=dict(supported=True,tp=1,pp=1,model_id='Qwen2.5-7B-Instruct',gpu_uuids=UUIDS[:1]),
        measurement_start=dict(acknowledged=True),measurement_stop=dict(ranks=[dict(rank=0,acknowledged=True)]),
        settle_started_s=100.,start_s=102.,end_s=107.,cleanup_errors=[],drain=drained,
        service_end_state=dict(state(1,107.01,5),all_queue=['r0'],running=['r0']),
        sample=dict(ranks=[dict(rank=0,samples=rows)]),client_requests=requests,power=power,
        cancel_receipts={'r0':dict(acknowledged=True,cancelled=True,request_id='r0',generation=5,native_state=drained)})


def test_actual_decode_context_and_active_board_power_are_separate_from_lease_power():
    raw=raw_fixture();value=audit_power_window(raw)
    assert value['passed'] and value['effective_context_tokens']>128
    assert value['power_w']==pytest.approx(100.)
    assert value['measured_board_energy_j']==pytest.approx(500.)
    assert value['public_eight_board_metering']['energy_service_j']==pytest.approx(4000.)
    assert not value['formal_eligible'] and not value['power_component_qualified']


@pytest.mark.parametrize('role',['prefill','decode'])
def test_tp2_power_uses_both_physical_boards_and_requires_rank1(role):
    raw=raw_fixture(role);raw['spec'].update(tp=2,gpus=[0,1])
    raw['capability'].update(tp=2,model_id='Qwen2.5-32B-Instruct',gpu_uuids=UUIDS[:2])
    raw['measurement_stop']['ranks'].append(dict(rank=1,acknowledged=True))
    native_states=[raw['drain'],raw['service_end_state']]
    for value in native_states:
        value['tp']=2;value['ranks'].append(dict(deepcopy(value['ranks'][0]),rank=1))
    rows=raw['sample']['ranks'][0]['samples']
    for row in rows:row['tp']=2
    other=deepcopy(rows)
    for row in other:row['rank']=1
    raw['sample']['ranks'].append(dict(rank=1,samples=other))
    for _,clocks in raw['power']['frequency_samples']:clocks[1]=1500.
    # True TP ranks share a logical sequence but not an identical host timestamp.
    # Let rank 1's last launch cross the fixed power boundary slightly.
    raw['sample']['ranks'][0]['samples'][-1]['at_s']=106.999
    raw['sample']['ranks'][1]['samples'][-1]['at_s']=107.001
    result=audit_power_window(raw)
    assert result['power_w']==pytest.approx(200.)
    raw['sample']['ranks'].pop()
    with pytest.raises(ValueError,match='rank inventory'):audit_power_window(raw)


def test_prefill_cycle_is_not_claimed_as_active_kernel_power():
    value=audit_power_window(raw_fixture('prefill'))
    assert value['passed'] and value['input_tokens']==128
    assert 'idle_gaps' in value['power_scope']
    assert not value['active_prefill_kernel_power_qualified']


@pytest.mark.parametrize('damage',['prefill','mixed','batch','different_context','ended','native_ended','cancel','clock','gap','rank','source'])
def test_incomplete_or_contaminated_native_power_is_rejected(damage):
    raw=raw_fixture();rows=raw['sample']['ranks'][0]['samples']
    if damage in ('prefill','mixed'):rows[30]['role']=damage
    elif damage=='batch':rows[30]['batch']=2
    elif damage=='different_context':rows[30]['prompt_lengths']=[127]
    elif damage=='ended':raw['client_requests'][0]['observed_running_through_s']=106.
    elif damage=='native_ended':raw['service_end_state']['running']=[]
    elif damage=='cancel':raw['cancel_receipts']={}
    elif damage=='clock':raw['power']['frequency_samples'][40][1][0]=210
    elif damage=='gap':raw['power']['samples']=raw['power']['samples'][:30]+raw['power']['samples'][50:];raw['power']['power_metadata']=raw['power']['power_metadata'][:30]+raw['power']['power_metadata'][50:]
    elif damage=='rank':raw['sample']['ranks']=[]
    elif damage=='source':raw['power']['power_source']=dict(raw['power']['power_source'],field_id=185)
    with pytest.raises(ValueError):audit_power_window(raw)


def test_context_is_time_weighted_and_cannot_be_replaced_by_nominal_prompt():
    rows=[dict(at_s=t,context_tokens=c) for t,c in ((0.,100),(.1,200),(.9,300))]
    value=context_window(rows,0.,1.)
    assert value['effective_context_tokens']==pytest.approx(200.)
    with pytest.raises(ValueError):context_window(rows,.5,3.)


def test_barrier_propagates_early_native_request_failure():
    from pdblend.profile.collection.native_power_collect import _wait
    async def run():
        done=asyncio.create_task(asyncio.sleep(0));await done
        async def false():return False
        with pytest.raises(ValueError,match='ended'):
            await _wait(false,[done])
    asyncio.run(run())


@pytest.mark.parametrize('fail_during_measurement',[False,True])
def test_collector_arms_idle_before_prefill_and_cancels_before_native_drain(tmp_path,monkeypatch,fail_during_measurement):
    from types import SimpleNamespace
    from pdblend_runtime.probe import NativeSpec
    from pdblend.profile.collection import native_power_collect as module
    clock=[100.];calls=[];clients=[];original_sleep=asyncio.sleep
    async def sleep(seconds):clock[0]+=seconds;await original_sleep(0)
    monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0]))
    monkeypatch.setattr(module.asyncio,'sleep',sleep)
    async def request(session,url,payload,row):
        row['submitted_s']=clock[0];clients.append(row)
        try:await asyncio.Event().wait()
        except asyncio.CancelledError:row['error']='CancelledError()';raise
        finally:row['finished_s']=clock[0]
    monkeypatch.setattr(module,'_request',request)
    raw_power=evidence([99+i*.1 for i in range(120)])
    sampler=SimpleNamespace(**raw_power,frequency_samples=[(99+i*.1,[1500]*8) for i in range(120)],
        utilization_readings=[],utilization_errors=[],utilization_source={},error_at_s=None,interval=.1)
    spec=NativeSpec('pd-timing-0',(0,),20000,'/models/Qwen2.5-7B-Instruct',tp=1,max_num_seqs=32)
    class Runner:
        session=None;uuids=UUIDS;lease=dict(gpu_ids=list(range(8)),gpu_uuids=UUIDS,gpu_uuid_binding_verified=True)
        async def capability(self,s):return dict(state=dict(total_kv_tokens=1000000))
        async def drain(self,s):calls.append(('initial-drain',None));return {}
        async def resume(self,s):calls.append(('resume',None));return {}
        async def clock(self,s,f):return {}
        async def state(self,s):return dict(all_queue=[r['request_id'] for r in clients])
        async def request(self,s,endpoint,payload=None):
            calls.append((endpoint,payload))
            if endpoint=='control' and payload.get('admit_prefill') is True:
                for row in clients:row['seen_tokens']=1
            if endpoint=='measurement/samples':
                if fail_during_measurement:raise RuntimeError('sample RPC failure')
                return dict(ranks=[])
            return dict(acknowledged=True)
    runner=Runner();runner.sampler=sampler
    point=dict(role='decode',batch=2,prompt_tokens=16,output_tokens=1000,frequency_mhz=1500,seed=9701,repeat=0)
    result=asyncio.run(module.power_window(runner,spec,point,tmp_path/'raw.json'))
    assert result['status']==('failed' if fail_during_measurement else 'measured')
    assert len(result['cancel_receipts'])==2
    names=[name for name,payload in calls]
    assert names.index('measurement/start') < next(i for i,(name,payload) in enumerate(calls)
        if name=='control' and payload.get('admit_prefill') is True)
    assert names.index('measurement/samples')<names.index('cancel')<names.index('measurement/stop')<names.index('drain')
    assert json.loads((tmp_path/'raw.json').read_text())['status']==result['status']


@pytest.mark.parametrize('cause',['domain','http_error','cleanup_error'])
def test_only_expected_domain_failure_allows_unrelated_timing_after_verified_restore(tmp_path,monkeypatch,cause):
    from types import SimpleNamespace
    from pdblend_runtime.probe import NativeSpec
    from pdblend.profile.collection import native_power_collect as module
    spec=NativeSpec('pd-timing-0',(0,),20000,'/models/Qwen2.5-7B-Instruct',max_num_seqs=32,
                    extra_args=('--worker-cls',module.WORKER))
    class Runner:
        def __init__(self,*args,**kwargs):self.capabilities={}
        async def stop_measurement(self,s):return {}
        async def capability(self,s):return {}
        async def drain(self,s):return {}
        async def clock(self,s,f):
            if cause=='cleanup_error':raise RuntimeError('clock restore failed')
            return {}
        async def state(self,s):return dict(all_queue=[])
        async def request(self,*args):return {}
    monkeypatch.setattr(module,'NativeRuntimeCollector',Runner)
    monkeypatch.setattr(module,'validate_inventory',lambda *a:dict(gpu_uuids=UUIDS))
    monkeypatch.setattr(module,'validate_plan',lambda p:p)
    async def window(runner,s,point,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        raw=dict(status='failed',cleanup_errors=[],error='recorded raw failure',
                 error_kind='operational_failure' if cause=='http_error' else 'domain_unavailable')
        path.write_text(json.dumps(raw));return raw
    monkeypatch.setattr(module,'power_window',window)
    from dataclasses import replace
    specs=[replace(spec,instance_id=f'pd-timing-{i}',gpus=(i,)) for i in range(8)]
    plan=dict(model_id='Qwen2.5-7B-Instruct',tp=1,points=[dict(repeats=3)],remaining_gates=['holdout'])
    fleet={s.instance_id:SimpleNamespace(alive=lambda:True) for s in specs}
    result=asyncio.run(module.collect_power_pilot(specs,fleet,None,None,tmp_path/'out',gpu_uuids=UUIDS,plan=plan))
    assert result['ready_for_timing'] is (cause=='domain')
    assert result['safe_restore_passed'] is (cause!='cleanup_error')
    assert not result['complete'] and not result['formal_eligible']
    assert len(result['windows'])==1
    if cause!='http_error':assert result['domain_unavailable'][0]['omitted_identical_repeats']==2
