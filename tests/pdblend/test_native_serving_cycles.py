from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from pdblend.profile.collection import native_serving_cycles as collect
from pdblend.profile.collection.native_serving_cycles_audit import audit_cycle_window,occupancy
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend_runtime.probe import NativeSpec
from pdblend.results.journal import payload_receipt
from test_comparison_acceptance import state
from test_comparison_metering import evidence,UUIDS


def plan_fixture(tmp_path,monkeypatch,tp=1):
    path=tmp_path/'parent.json';path.write_text(json.dumps(dict(seed=9702,duration_s=120,
        requests=[dict(prompt=[10,i],max_tokens=2,source='alpaca') for i in range(1,50)])))
    groups=[dict(dataset='alpaca',rate_scale=s,target_rate_rps=16/60*s,trace_binding=binding(path))
            for s in (.25,.5,.75,1.)]
    monkeypatch.setattr(collect,'tuning_groups',lambda *a:groups)
    return collect.build_cycle_plan({}, {},model_id='Qwen2.5-'+('32B' if tp==2 else '7B')+'-Instruct')


@pytest.mark.parametrize('tp',[1,2])
def test_same_fleet_plan_binds_true_topology_and_disjoint_native_content(tmp_path,monkeypatch,tp):
    plan=plan_fixture(tmp_path,monkeypatch,tp);collect.validate_cycle_plan(plan)
    assert plan['resident_instances']==8//tp and len(plan['points'])==16
    sets={purpose:set() for purpose in ('training','holdout')}
    for point in plan['points']:
        trace=collect.cycle_trace(plan,point)
        sets[point['purpose']].update(r['request_content_sha256'] for r in trace['requests'])
        assert all(len(r['prompt'])==2 and r['max_tokens']==2 for r in trace['requests'])
    assert not sets['training']&sets['holdout']
    assert plan['service_seconds_per_phase']==480 and not plan['formal_eligible']
    a=collect.cycle_trace(plan,plan['points'][0]);b=collect.cycle_trace(plan,plan['points'][1])
    assert [r['request_content_sha256'] for r in a['requests']]==[r['request_content_sha256'] for r in b['requests']]
    assert [r['arrival_s'] for r in a['requests']]!=[r['arrival_s'] for r in b['requests']]


def test_holdout_has_no_gpu_side_effect_without_frozen_training_candidate(tmp_path,monkeypatch):
    import asyncio
    plan=plan_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='holdout blocked'):
        asyncio.run(collect.collect_serving_cycles([],None,None,None,tmp_path/'out',gpu_uuids=UUIDS,plan=plan,phase='holdout'))
    assert not (tmp_path/'out').exists()


def fixture(tmp_path,monkeypatch,tp=1):
    plan=plan_fixture(tmp_path,monkeypatch,tp);point=plan['points'][0];trace=collect.cycle_trace(plan,point)
    specs=[NativeSpec(f'cycle-{i}',tuple(range(i*tp,(i+1)*tp)),22000+i,'/models/'+plan['model_id'],tp=tp,
        max_num_seqs=32,generation=5,extra_args=('--worker-cls',collect.WORKER)) for i in range(8//tp)]
    def drained(at):return dict(state(tp,at,5),total_blocks=100,free_blocks=100,reserved_blocks=0,drained=True,acknowledged=True)
    raw=dict(schema='pdblend-native-request-cycle-window/v1',system='pdblend',status='measured',point=point,
        plan_sha256=digest(plan),trace=trace,candidate=None,lease=dict(gpu_ids=list(range(8)),gpu_uuids=UUIDS,gpu_uuid_binding_verified=True),
        actual_launch=[dict(spec=asdict(s),argv=s.command()) for s in specs],capabilities={},before={},after={},resume={},
        clocks={},measurement_start={},measurement_stop={},samples={},state_observations=[],routes=[],client_requests=[],
        cleanup_errors=[],hardware_executed=True,service_started_s=100.,service_end_s=160.,tail_end_s=161.,
        power_scope='complete_request_cycle_not_pure_decode_or_active_prefill_kernel')
    for spec in specs:
        iid=spec.instance_id
        raw['capabilities'][iid]=dict(supported=True,model_id=plan['model_id'],tp=tp,pp=1,gpu_uuids=UUIDS[spec.gpus[0]:spec.gpus[-1]+1],
            model_hash='m',tokenizer_hash='t',engine_revision='engine',image_digest='image',source_revision='source')
        raw['before'][iid]=drained(98.);raw['after'][iid]=drained(160.9)
        raw['resume'][iid]=dict(control=dict(acknowledged=True),state=drained(99.))
        raw['measurement_start'][iid]=dict(acknowledged=True,ranks=[dict(rank=i,acknowledged=True,
            system='pdblend',scope='runner',tp=tp,pp=1) for i in range(tp)])
        raw['clocks'][iid]=dict(ack=dict(acknowledged=True,success=True,requested_frequency_mhz=1500,
            gpus=[dict(gpu_uuid=g) for g in raw['capabilities'][iid]['gpu_uuids']]),
            observations=[dict(at_s=99.,frequencies_mhz=[1500]*tp)])
        raw['measurement_stop'][iid]=dict(ranks=[dict(rank=i,acknowledged=True) for i in range(tp)])
        raw['samples'][iid]=dict(ranks=[dict(rank=i,samples=[]) for i in range(tp)])
        raw['state_observations'] += [dict(instance_id=iid,received_s=t/2,state=state(tp,t/2,5)) for t in range(200,505)]
    counts={s.instance_id:0 for s in specs}
    for request in trace['requests']:
        rid='r'+str(request['req_id']);iid=specs[0].instance_id;at=100.+request['arrival_s']
        events=[dict(token_ids=[4+i],token_index=i+1,received_s=at+.01+i*.02,finished=i==1) for i in range(2)]
        client=dict(req_id=request['req_id'],request_id=rid,instance_id=iid,scheduled_s=at,submitted_s=at,
            finished_s=at+.04,events=events,**{k:v for k,v in payload_receipt(events,journal_path='embedded:events',request_id=rid).items() if k!='request_id'})
        raw['client_requests'].append(client)
        raw['routes'].append(dict(event='acquire',request_id=rid,instance_id=iid,at_s=at,before=dict(counts)))
        raw['routes'].append(dict(event='release',request_id=rid,instance_id=iid,at_s=at+.04,after=dict(counts)))
        for rank in raw['samples'][iid]['ranks']:
            for i in range(2):rank['samples'].append(dict(system='pdblend',measurement_scope='runner',rank=rank['rank'],tp=tp,pp=1,
                role='prefill' if i==0 else 'decode',batch=1,request_ids=[rid],prompt_lengths=[2],context_lengths=[2+i],
                scheduled_lengths=[2 if i==0 else 1],gpu_elapsed_ms=5.,at_s=at+i*.02))
    raw['power']=evidence([99.+i/10 for i in range(1531)])
    raw['power']['frequency_samples']=[(t,[1500]*8) for t,_ in raw['power']['samples']]
    return raw,plan


@pytest.mark.parametrize('tp',[1,2])
def test_real_cycle_raw_replay_accepts_all_physical_ranks_without_claiming_pure_power(tmp_path,monkeypatch,tp):
    raw,plan=fixture(tmp_path,monkeypatch,tp);audit=audit_cycle_window(raw,plan)
    assert audit['passed'],audit
    assert audit['measured']['energy_service_j']==pytest.approx(48000)
    assert len(audit['occupancy_by_replica'])==8//tp
    assert not audit['pure_decode_nodes_created'] and not audit['formal_eligible']


@pytest.mark.parametrize('damage',['rank','forward','tokens','route','power_gap','unbound_trace','native_work','clock'])
def test_cycle_missing_rank_progression_ownership_or_energy_never_qualifies(tmp_path,monkeypatch,damage):
    raw,plan=fixture(tmp_path,monkeypatch,2);iid='cycle-0'
    if damage=='rank':raw['samples'][iid]['ranks'].pop()
    elif damage=='forward':
        for rank in raw['samples'][iid]['ranks']:rank['samples'].pop()
    elif damage=='tokens':raw['client_requests'][0]['events'][0]['token_index']=2
    elif damage=='route':raw['routes'].pop()
    elif damage=='power_gap':
        for key in ('samples','power_metadata'):raw['power'][key]=raw['power'][key][:50]+raw['power'][key][80:]
    elif damage=='unbound_trace':raw['trace']['requests'][0]['prompt'][0]=999
    elif damage=='native_work':raw['state_observations'][0]['state']['all_queue']=['unowned']
    elif damage=='clock':raw['power']['frequency_samples'][50][1][5]=210
    audit=audit_cycle_window(raw,plan)
    assert not audit['passed'] and not audit['formal_eligible']


def test_occupancy_does_not_fill_a_scheduler_gap_with_zero():
    rows=[dict(state=dict(native_at_s=t,running=['r']*n)) for t,n in [(0.,2),(.5,0),(2.,4),(2.5,0)]]
    value=occupancy(rows,0.,2.5)
    assert value['coverage_fraction']==pytest.approx(.4)
    assert value['mean_running_requests']==pytest.approx(3.)
    assert value['averaging_denominator_s']==1. and value['max_gap_s']==1.5


@pytest.mark.parametrize('tp',[1,2])
def test_only_complete_frequency_mismatch_receives_recoverable_invalid_result(tmp_path,monkeypatch,tp):
    raw,plan=fixture(tmp_path,monkeypatch,tp)
    raw['power']['frequency_samples'][50][1][5]=1395
    audit=audit_cycle_window(raw,plan)
    assert not audit['passed'] and audit['non_frequency_checks_passed']
    assert audit['qualification_gap']=='observed_frequency_mismatch' and audit['frequency_data_complete']
    assert audit['frequency_evidence']['mismatch_count']==1
    assert audit['measured']['energy_service_j']==pytest.approx(48000)
    assert not audit['formal_eligible'] and not audit['full_profile_qualified']


@pytest.mark.parametrize('damage',['clock_gap','clock_nan','clock_duplicate','clock_negative','clock_missing_rank',
    'clock_ack','clock_uuid','clock_observed','arm_rank','drain_ack','drain_stale','tokens','power_gap','cuda'])
def test_frequency_mismatch_cannot_hide_another_failure(tmp_path,monkeypatch,damage):
    raw,plan=fixture(tmp_path,monkeypatch,2);iid='cycle-0'
    raw['power']['frequency_samples'][50][1][5]=1395
    samples=raw['power']['frequency_samples']
    if damage=='clock_gap':del samples[100:120]
    elif damage=='clock_nan':samples[100][1][3]=float('nan')
    elif damage=='clock_duplicate':samples.insert(100,deepcopy(samples[100]))
    elif damage=='clock_negative':samples[100][1][3]=-1
    elif damage=='clock_missing_rank':samples[100][1].pop()
    elif damage=='clock_ack':raw['clocks'][iid]['ack']['success']=False
    elif damage=='clock_uuid':raw['clocks'][iid]['ack']['gpus'][0]['gpu_uuid']='wrong'
    elif damage=='clock_observed':raw['clocks'][iid]['observations'][-1]['frequencies_mhz'][0]=1200
    elif damage=='arm_rank':raw['measurement_start'][iid]['ranks'].pop()
    elif damage=='drain_ack':raw['after'][iid]['acknowledged']=False
    elif damage=='drain_stale':raw['after'][iid]['native_at_s']=159.
    elif damage=='tokens':raw['client_requests'][0]['completion_tokens']=1
    elif damage=='power_gap':
        for key in ('samples','power_metadata'):del raw['power'][key][100:120]
    elif damage=='cuda':raw['samples'][iid]['ranks'][0]['samples'][0]['failed']=True
    audit=audit_cycle_window(raw,plan)
    assert not audit['passed'] and not audit.get('non_frequency_checks_passed')
    assert audit.get('error_kind')!='expected_measurement_qualification_gap'


@pytest.mark.parametrize('restore_failure',[False,True])
@pytest.mark.parametrize('invalid',['none','frequency','operational'])
def test_collection_never_loads_engines_and_restores_each_rank_before_next_stage(tmp_path,monkeypatch,restore_failure,invalid):
    import asyncio
    from types import SimpleNamespace
    from pdblend.profile.collection import native_serving_cycles_audit as audit
    plan=plan_fixture(tmp_path,monkeypatch,2)
    specs=[NativeSpec(f'cycle-{i}',(2*i,2*i+1),22000+i,'/models/'+plan['model_id'],tp=2,
        max_num_seqs=32,extra_args=('--worker-cls',collect.WORKER)) for i in range(4)]
    operations=[]
    class Runner:
        def __init__(self,*args,**kwargs):pass
        async def stop_measurement(self,s):operations.append(('stop_measurement',s.instance_id));return {}
        async def drain(self,s):operations.append(('drain',s.instance_id));return {}
        async def clock(self,s,f):
            operations.append(('clock',s.instance_id,f))
            if restore_failure and s.instance_id=='cycle-0':raise RuntimeError('restore clock failed')
            return {}
        async def resume(self,s):operations.append(('resume',s.instance_id));return {}
    monkeypatch.setattr(collect,'NativeRuntimeCollector',Runner)
    monkeypatch.setattr(collect,'validate_inventory',lambda *args:dict(gpu_uuids=UUIDS))
    async def window(runner,ss,p,point,path,**kwargs):
        operations.append(('window',point))
        value=dict(point=point,status='measured');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return value
    monkeypatch.setattr(collect,'collect_cycle_window',window)
    def audited(raw,p):
        if invalid=='none' or raw['point']!=plan['points'][0]:return dict(passed=True)
        if invalid=='operational':return dict(passed=False,errors=['missing frequency observations'])
        return dict(passed=False,error_kind='expected_measurement_qualification_gap',
            qualification_gap='observed_frequency_mismatch',non_frequency_checks_passed=True,
            frequency_data_complete=True,frequency_evidence=dict(mismatch_count=1))
    monkeypatch.setattr(audit,'audit_cycle_window',audited)
    fleet={s.instance_id:SimpleNamespace(alive=lambda:True) for s in specs}
    result=asyncio.run(collect.collect_serving_cycles(specs,fleet,None,None,tmp_path/'out',gpu_uuids=UUIDS,plan=plan))
    complete=invalid=='none' or (invalid=='frequency' and not restore_failure)
    assert result['collection_complete']==complete and len(result['windows'])==(8 if complete else 1)
    assert result['safe_restore_passed']==(not restore_failure)
    assert result['ready_for_timing']==(not restore_failure and invalid!='operational')
    assert result['all_observations_valid']==(invalid=='none')
    restores=8 if invalid=='frequency' else 4
    assert len([r for r in operations if r[0]=='stop_measurement'])==restores
    assert len([r for r in operations if r[0]=='clock' and r[2]==2520])==restores
    if invalid=='frequency' and not restore_failure:
        second=[i for i,r in enumerate(operations) if r[0]=='window'][1]
        assert len([r for r in operations[:second] if r[0]=='resume'])==4
        assert result['qualification_gaps'][0]['invalid_window_preserved']
    assert not result['formal_eligible'] and not result['pure_power_component_qualified']
