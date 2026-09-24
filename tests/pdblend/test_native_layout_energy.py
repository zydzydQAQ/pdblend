from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest

from pdblend.profile.collection import native_layout_energy as collect
from pdblend.profile.query import native_layout_model as model
from pdblend.profile.collection.native_timing_plan import binding,digest
from pdblend.planner.native_layout import NativeLayoutPlanner
from pdblend.planner.pool import PoolPlanner,PlannerConfig,SLO
from pdblend.planner.forecast import Forecast


def plan_fixture(tmp_path,monkeypatch):
    groups=[]
    for dataset in ('alpaca','sharegpt','longbench'):
        p=tmp_path/(dataset+'.json');p.write_text(json.dumps(dict(seed=9702,duration_s=120,
            requests=[dict(req_id=i,arrival_s=i/10,prompt=[10,i],max_tokens=2,source=dataset) for i in range(1,80)])))
        groups.extend(dict(dataset=dataset,rate_scale=s,target_rate_rps=s,trace_binding=binding(p)) for s in (.25,.5,.75,1.))
    monkeypatch.setattr(collect,'tuning_groups',lambda *args:groups)
    return collect.build_layout_plan({}, {})


def test_plan_poisson_independent_content_exact_all_domain(tmp_path,monkeypatch):
    plan=plan_fixture(tmp_path,monkeypatch);collect.validate_layout_plan(plan)
    assert len(plan['points'])==36 and plan['training_service_s']==720 and plan['holdout_service_s']==3600
    content={'training':set(),'holdout':set()}
    for point in plan['points']:
        trace=collect.layout_trace(plan,point)
        content[point['purpose']].update(digest(dict(prompt=r['prompt'],max_tokens=r['max_tokens'])) for r in trace['requests'])
        assert all(0<r['arrival_s']<point['duration_s'] for r in trace['requests'])
        assert trace['arrival_family']=='poisson' and trace['seed']!=701
        assert point['initial_roles']=={'M':4}
    assert not content['training']&content['holdout']
    assert not plan['formal_eligible']
    point=plan['points'][0];trace=collect.layout_trace(plan,point)
    times=[r['arrival_s'] for r in trace['requests']]
    assert len({round(b-a,8) for a,b in zip(times,times[1:])})>2


@pytest.mark.parametrize('field,value',[('arrival_family','paced'),('duration_s',150.),('initial_roles',{'M':8}),('seed',701)])
def test_discovery_or_evaluation_trace_cannot_enter_layout(tmp_path,monkeypatch,field,value):
    plan=plan_fixture(tmp_path,monkeypatch);point=dict(plan['points'][0],**{field:value})
    with pytest.raises(ValueError):collect.layout_trace(plan,point)


def candidate():
    return dict(kind=model.KIND,prediction_semantics='whole_eight_gpu_wall_clock_mean_w',model_id=collect.MODEL,
        nodes=[dict(dataset=d,parent_trace={'path':d,'sha256':d},frequency_mhz=f,rates_rps=[.25,1.],power_w=[500.,800.] if f==1500 else [700.,1000.])
            for d in ('alpaca','sharegpt','longbench') for f in (1500,2520)])


def query(**updates):
    return dict(model_id=collect.MODEL,tp=2,pp=1,counts={'M':4},rate_rps=.5,frequency_mhz=1500,dataset='alpaca',
        parent_trace={'path':'alpaca','sha256':'alpaca'},arrival_family='poisson',service_duration_s=150.,**updates)


def test_mean_w_is_whole_fleet_and_no_kernel_interface():
    energy=model.NativeLayoutEnergyModel(candidate())
    assert energy.predict_layout_mean_w(**query())==600.
    assert not hasattr(energy,'prefill_power_w') and not hasattr(energy,'decode_power_w')


@pytest.mark.parametrize('field,value',[('counts',{'M':8}),('counts',{'M':3,'off':1}),('arrival_family','burst4'),
    ('service_duration_s',60.),('rate_rps',.01),('rate_rps',1.01),('frequency_mhz',1800),('tp',1),
    ('dataset','unknown'),('parent_trace',{'path':'substitute','sha256':'alpaca'})])
def test_mean_w_outside_scope_is_unsupported(field,value):
    args=query();args[field]=value
    with pytest.raises(ValueError,match='missing_profile:'):model.NativeLayoutEnergyModel(candidate()).predict_layout_mean_w(**args)


class Timing:
    model=collect.MODEL;system='pdblend';tp=2;pp=1;freqs=(1500,2520);kv_capacity_tokens=100000
    bounded_coverage={'native':True};decode_power_overrides={};profile_key={'qualified_timing':'test'}
    def require_runtime_components(self,*names):pass
    def prefill_marginal_seconds(self,n,f):return n*.000001
    def prefill_seconds(self,n,f):return .001+n*.000001
    def decode_supported(self,b,c,f):return 1<=b<=32 and f in self.freqs
    def step_seconds(self,b,c,f):return .002+b*.00001
    def prefill_power_w(self,*args,**kwargs):raise AssertionError('kernel prefill power must not be queried')
    def decode_power_w(self,*args,**kwargs):raise AssertionError('kernel decode power must not be queried')
    def decode_power_supported(self,*args,**kwargs):raise AssertionError('decode power gate must not be queried')
    def static_power_w(self,*args,**kwargs):raise AssertionError('whole fleet already contains static power')


def planner(timing=None,**scope):
    return NativeLayoutPlanner(timing or Timing(),PlannerConfig(slots=4,slo=SLO(10,1),freqs=(1500,2520),max_num_seqs=32,min_m_instances=4),
        model.NativeLayoutEnergyModel(candidate()),workload_scope=dict(dataset='alpaca',parent_trace={'path':'alpaca','sha256':'alpaca'},
            arrival_family='poisson',service_duration_s=150.,**scope))


def forecast(rate=.5):return Forecast(rate_rps=rate,trend_rps=0,input_mean=20,input_p95=20,output_mean=20,inflight=0,
                                     inputs=(20,),outputs=(20,),length_pairs=((20,20),))


def test_opt_in_planner_full_canonical_layout_two_frequencies_no_kernel_queries():
    p=planner();plans=p.candidates(forecast())
    assert len(plans)==2 and plans[0].f_M==1500 and plans[0].power_w==600.
    assert all({k:v for k,v in r.counts.items() if v}=={'M':4} for r in plans)
    assert not p.unsupported
    assert 'native_whole_layout_request_cycle'==plans[0].detail['M']['energy_model']


def test_missing_timing_is_distinct_from_physical_capacity_infeasibility():
    class Missing(Timing):
        def decode_supported(self,b,c,f):return f!=1500 and super().decode_supported(b,c,f)
    p=planner(Missing());plans=p.candidates(forecast())
    assert len(plans)==1 and p.unsupported and 'actual hull' in p.unsupported[0]['reason']
    p=planner();assert p.evaluate({'M':3,'off':1},2520,2520,1500,0,forecast()) is None
    assert p.unsupported[0]['reason']=='unsupported_layout_or_backlog'


def test_original_pool_planner_still_uses_original_energy_gate():
    with pytest.raises(AssertionError,match='decode power gate'):
        PoolPlanner(Timing(),PlannerConfig(slots=4,slo=SLO(10,1),freqs=(1500,2520),max_num_seqs=32,min_m_instances=4)).candidates(forecast())


def test_raw_discovery_window_is_not_a_layout_energy_receipt(tmp_path,monkeypatch):
    from test_native_serving_cycles import fixture
    raw,plan=fixture(tmp_path,monkeypatch,2)
    result=collect.audit_layout_window(raw,plan)
    assert not result['passed'] and 'discovery' in result['errors'][0]


def test_holdout_without_frozen_selection_has_no_gpu_side_effect(tmp_path,monkeypatch):
    import asyncio
    plan=plan_fixture(tmp_path,monkeypatch)
    with pytest.raises((ValueError,TypeError)):
        asyncio.run(collect.collect_layout_energy([],None,None,None,tmp_path/'out',gpu_uuids=[],plan=plan,phase='holdout'))
    assert not (tmp_path/'out').exists()


def layout_raw(tmp_path,monkeypatch):
    from test_native_serving_cycles import fixture
    from test_comparison_acceptance import state
    from pdblend.results.journal import payload_receipt
    raw,_=fixture(tmp_path,monkeypatch,2)
    plan=plan_fixture(tmp_path,monkeypatch);point=plan['points'][0];trace=collect.layout_trace(plan,point)
    raw.update(point=point,plan_sha256=digest(plan),trace=trace,client_requests=[],routes=[])
    counts={x['spec']['instance_id']:0 for x in raw['actual_launch']};ids=sorted(counts)
    timeline=[]
    for request in trace['requests']:
        at=100+request['arrival_s'];timeline.extend([(at,'acquire',request),(at+.004,'release',request)])
    owned={}
    for at,event,request in sorted(timeline):
        rid='layout-'+str(request['req_id'])
        if event=='acquire':
            iid=min(counts,key=lambda k:(counts[k],k));owned[rid]=iid
            raw['routes'].append(dict(event=event,request_id=rid,instance_id=iid,at_s=at,before=dict(counts)));counts[iid]+=1
            events=[dict(token_ids=[4+i],token_index=i+1,received_s=at+.001+i*.002,finished=i==1) for i in range(2)]
            raw['client_requests'].append(dict(req_id=request['req_id'],instance_id=iid,scheduled_s=at,submitted_s=at,
                finished_s=at+.004,events=events,**payload_receipt(events,journal_path='embedded:events',request_id=rid)))
        else:
            iid=owned[rid];counts[iid]-=1
            raw['routes'].append(dict(event=event,request_id=rid,instance_id=iid,at_s=at,after=dict(counts)))
    for iid,value in raw['samples'].items():
        for rank in value['ranks']:rank['samples']=[]
    for request in raw['client_requests']:
        for rank in raw['samples'][request['instance_id']]['ranks']:
            for i in range(2):rank['samples'].append(dict(system='pdblend',measurement_scope='runner',rank=rank['rank'],tp=2,pp=1,
                role='prefill' if i==0 else 'decode',batch=1,request_ids=[request['request_id']],prompt_lengths=[2],context_lengths=[2+i],
                scheduled_lengths=[2 if i==0 else 1],gpu_elapsed_ms=1.,at_s=request['submitted_s']+i*.002))
    for launch in raw['actual_launch']:
        spec=launch['spec'];iid=spec['instance_id'];uuids=[raw['lease']['gpu_uuids'][g] for g in spec['gpus']]
        raw['clocks'][iid]=dict(ack=dict(acknowledged=True,success=True,requested_frequency_mhz=1500,
            gpus=[dict(gpu_uuid=u) for u in uuids]),observations=[dict(at_s=98.,frequencies_mhz=[1500,1500])])
    return raw,plan


def test_real_poisson_raw_canonical_and_tp2_clock_acceptance(tmp_path,monkeypatch):
    raw,plan=layout_raw(tmp_path,monkeypatch);audit=collect.audit_layout_window(raw,plan)
    assert audit['passed'],audit
    assert audit['metrics']['token_timing_complete'] and audit['metrics']['slo_pass']
    assert audit['measured']['service_mean_power_w']==pytest.approx(800.)
    assert not audit['formal_eligible']


@pytest.mark.parametrize('damage',['clock_ack_rank','actual_clock','cuda_rank','wrong_trace','incomplete_cohort'])
def test_layout_raw_missing_physical_or_request_evidence_is_invalid(tmp_path,monkeypatch,damage):
    raw,plan=layout_raw(tmp_path,monkeypatch)
    if damage=='clock_ack_rank':raw['clocks']['cycle-0']['ack']['gpus'].pop()
    elif damage=='actual_clock':raw['power']['frequency_samples'][50][1][7]=1200
    elif damage=='cuda_rank':raw['samples']['cycle-0']['ranks'].pop()
    elif damage=='wrong_trace':raw['trace']['requests'][0]['arrival_s']+=.001
    else:raw['client_requests'].pop()
    assert not collect.audit_layout_window(raw,plan)['passed']


def test_timing_feasibility_matches_original_pool_without_reusing_energy():
    class Original(Timing):
        def decode_power_supported(self,*a,**k):return True
        def decode_power_w(self,*a,**k):return 100.
        def prefill_power_w(self,*a,**k):return 100.
        def static_power_w(self,*a,**k):return 10.
    new=planner();old=PoolPlanner(Original(),new.cfg)
    for rate in (.25,.5,.75,1.):
        fc=forecast(rate)
        for frequency in (1500,2520):
            a=new.mixed_timing(fc,4,frequency);b=old._mixed_pool(fc.rate_rps,fc,fc.input_mean,fc.input_p95,4,frequency)
            assert all(a[k]==b[k] for k in ('ttft_s','tpot_s','busy','batch','tpot_miss'))


def component_fixture(tmp_path,monkeypatch,frequency_domain_ref=None):
    import time
    plan=plan_fixture(tmp_path,monkeypatch)
    if frequency_domain_ref is not None:
        plan=collect.build_layout_plan(plan['query_ledger'],plan['query_provenance'],frequency_domain_ref=frequency_domain_ref)
    groups=[]
    for dataset in ('alpaca','sharegpt','longbench'):
        ref=next(p['parent_trace'] for p in plan['points'] if p['dataset']==dataset)
        trace=json.loads(Path(ref['path']).read_text())
        # Actual Request schema uses idx; collector adds req_id only as the
        # native HTTP route alias, preserving the frozen Poisson idx.
        for r in trace['requests']:r['idx']=r.pop('req_id')
        groups.extend(dict(dataset=dataset,rate_scale=s,target_rate_rps=s,trace_binding=ref,trace=trace)
                      for s in (.25,.5,.75,1.))
    monkeypatch.setattr(model,'tuning_groups',lambda *args:groups)
    identity=dict(model_id=collect.MODEL,tp=2,pp=1,model_hash='m',tokenizer_hash='t',image_digest='i',engine_revision='e',source_revision='s')
    if frequency_domain_ref is not None:
        from pdblend.profile.collection.native_frequency_domain import with_domain
        identity=with_domain(identity,plan['frequency_domain'])
    holder={}
    def evidence(reference,actual_plan,phase):
        rows=[]
        for i,point in enumerate(plan['points']):
            if point['purpose']!=phase:continue
            power=(500. if point['frequency_mhz']==1500 else 700.)+(point['target_rate_rps']-.25)/.75*300.
            rows.append(dict(point=point,raw={'path':'synthetic-observation-'+str(i),'sha256':str(i)},
                audit=dict(identity=identity,measured=dict(service_mean_power_w=power),metrics=dict(slo_pass=
                    not (holder.get('slo_fail_frequency')==point['frequency_mhz']))),candidate=holder.get('candidate'),
                start_s=holder.get('held_start',1.)+i,tail_end_s=holder.get('held_start',1.)+i+1))
        report=dict(finished_s=1.) if phase=='training' else dict(started_s=holder['held_start'],candidate=holder['candidate'],selection=holder['selection'])
        if holder.get('power_error') and phase=='holdout':rows[0]['audit']['measured']['service_mean_power_w']*=2.
        if holder.get('decreasing') and phase=='training':rows[-1]['audit']['measured']['service_mean_power_w']=1.
        return dict(rows=rows,identity=identity,report=report)
    monkeypatch.setattr(model,'read_layout_collection',evidence)
    training=tmp_path/'training.json';training.write_text('{}');training_ref=binding(training)
    timing_file=tmp_path/'timing.json';timing_file.write_text('{}');timing_ref=binding(timing_file)
    timing=Timing();timing.profile_key={'layout_timing_selection_sha256':timing_ref['sha256']}
    if frequency_domain_ref is not None:timing.freqs=tuple(plan['frequency_domain']['frequencies_mhz'])
    timing.calibration_identity={k:v for k,v in identity.items() if k!='source_revision'}
    candidate_ref=model.freeze_layout_candidate(training_ref,plan,tmp_path/'candidate.json');holder['candidate']=candidate_ref
    selected=model.freeze_layout_selection(plan,candidate_ref,timing,timing_profile_ref=timing_ref,out=tmp_path/'selection.json')
    holder['selection']=selected;holder['held_start']=time.time()
    return plan,candidate_ref,selected,timing,holder


def test_freeze_full_candidate_replay_then_independent_150s_component(tmp_path,monkeypatch):
    plan,candidate_ref,selected,timing,holder=component_fixture(tmp_path,monkeypatch)
    selection=json.loads(Path(selected['path']).read_text())
    assert len(selection['groups'])==12 and all(len(r['candidates'])==2 for r in selection['groups'])
    result=model.replay_layout_component(plan,candidate_ref,selected,{},timing)
    assert result['component_qualified'] and result['duration_transfer_qualified']
    assert len(result['comparisons'])==24 and sum(r['selected'] for r in result['comparisons'])==12
    assert not result['formal_eligible']


@pytest.mark.parametrize('damage',['power','selected_slo','unselected_slo','candidate_bytes','selection_bytes','wrong_timing_object','decreasing_refit'])
def test_layout_qualification_rejects_prediction_source_or_selection_changes(tmp_path,monkeypatch,damage):
    plan,candidate_ref,selected,timing,holder=component_fixture(tmp_path,monkeypatch)
    if damage=='power':holder['power_error']=True
    elif damage=='selected_slo':holder['slo_fail_frequency']=1500
    elif damage=='unselected_slo':holder['slo_fail_frequency']=2520
    elif damage=='candidate_bytes':Path(candidate_ref['path']).write_text('{}')
    elif damage=='selection_bytes':Path(selected['path']).write_text('{}')
    elif damage=='wrong_timing_object':timing.profile_key={'layout_timing_selection_sha256':'different'}
    else:holder['decreasing']=True
    if damage in ('power','selected_slo','unselected_slo'):
        result=model.replay_layout_component(plan,candidate_ref,selected,{},timing)
        assert result['component_qualified'] is (damage=='unselected_slo')
        if damage=='selected_slo':assert not result['selected_150s_slo_passed']
    else:
        with pytest.raises(ValueError):model.replay_layout_component(plan,candidate_ref,selected,{},timing)


def test_layout_fallback_never_uses_kernel_functions():
    p=planner();assert p.fallback(forecast()).power_w==800.
    with pytest.raises(ValueError,match='missing_profile:'):p.fallback(forecast(3.))


def runtime_scope_fixture(monkeypatch,clock_error=False):
    from pdblend.profile.query import native_layout_profile as profile
    from pdblend.profile.collection import native_runtime_audit
    ident=dict(system='pdblend',model_id=collect.MODEL,tp=2,pp=1,model_hash='m',tokenizer_hash='t',image_digest='i',engine_revision='e')
    report=dict(complete=True,ready_for_timing=True,initial_capabilities={str(i):dict(ident,source_revision='s') for i in range(4)})
    comparisons=[]
    for operation,_ in profile.RUNTIME_REQUIRED:
        held=2. if clock_error else 1.05
        comparisons.append(dict(component=operation,metric='elapsed_s',training=[1.,1.,1.],training_prediction=1.,heldout=held,
            relative_error=abs(1.-held)/held))
    comparisons.append(dict(component='transfer_7168',metric='second_output_overhead_s',training=[-.00488877]*3,
        training_prediction=-.00488877,heldout=-.00686634,relative_error=.288))
    audit=dict(raw_components_complete=True,independent_holdout_collected=True,holdout_passed=False,
        capacity={str(i):100000 for i in range(4)},holdout_comparisons=comparisons)
    monkeypatch.setattr(native_runtime_audit,'replay_runtime',lambda value:audit)
    class Resolver:
        def read(self,ref):return report
    return profile,Resolver(),ident,audit


def test_unused_negative_transfer_never_becomes_zero_or_blocks_scoped_all_m_runtime(monkeypatch):
    profile,resolver,ident,audit=runtime_scope_fixture(monkeypatch)
    result=profile.replay_layout_runtime_component({},resolver,ident,['s'])
    assert result['scoped_runtime_qualified'] and not result['original_global_holdout_passed']
    assert not result['transfer_cost_consumed'] and not result['full_runtime_profile_qualified']
    assert audit['holdout_comparisons'][-1]['training_prediction']==-.00488877
    base=profile._RuntimeTimingBase(ident,result)
    base.require_runtime_components('capacity','clock_transition')
    with pytest.raises(ValueError,match='missing_profile'):base.require_runtime_components('transfer')
    with pytest.raises(ValueError,match='missing_profile'):base.transfer_seconds(7168)


def test_needed_clock_holdout_failure_still_blocks_scoped_runtime(monkeypatch):
    profile,resolver,ident,_=runtime_scope_fixture(monkeypatch,clock_error=True)
    with pytest.raises(profile.LayoutQualificationUnavailable,match='clock component holdout failed') as caught:
        profile.replay_layout_runtime_component({},resolver,ident,['s'])
    assert caught.value.gate=='scoped_clock_holdout'


def test_scoped_runtime_cannot_escape_to_7b_or_missing_raw_identity(monkeypatch):
    profile,resolver,ident,audit=runtime_scope_fixture(monkeypatch)
    with pytest.raises(ValueError,match='restricted'):
        profile.replay_layout_runtime_component({},resolver,dict(ident,model_id='Qwen2.5-7B-Instruct',tp=1),['s'])
    audit['raw_components_complete']=False
    with pytest.raises(ValueError,match='raw replay incomplete') as caught:
        profile.replay_layout_runtime_component({},resolver,ident,['s'])
    assert not isinstance(caught.value,profile.LayoutQualificationUnavailable)


@pytest.mark.parametrize('failure',['operational','fit_domain','holdout_model'])
def test_same_fleet_return_contract_distinguishes_infra_from_complete_model_failure(tmp_path,monkeypatch,failure):
    import asyncio
    from pdblend.profile.query import native_layout_profile as profile
    from pdblend.profile.collection.native_runtime_collect import write_new
    plan=plan_fixture(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(profile,'load_layout_timing',lambda ref:(Timing(),{}))
    async def phase(*args,**kwargs):
        calls.append(kwargs.get('phase','training'));out=args[4]
        value=dict(safe_restore_passed=True,collection_complete=failure!='operational',ready_for_next=failure!='operational')
        write_new(out/'completion.json',value);return value
    monkeypatch.setattr(collect,'collect_layout_energy',phase)
    def fit(*args):
        if failure=='fit_domain':raise ValueError('missing_profile: observed domain')
        return write_new(args[-1],{})
    monkeypatch.setattr(model,'freeze_layout_candidate',fit)
    monkeypatch.setattr(model,'freeze_layout_selection',lambda *args,**kwargs:write_new(kwargs['out'],{}))
    monkeypatch.setattr(model,'replay_layout_component',lambda *args:dict(component_qualified=False))
    result=asyncio.run(collect.collect_and_replay_layout_energy([],None,None,None,tmp_path/'out',gpu_uuids=[],plan=plan,timing_profile_ref={}))
    assert result['safe_restore_passed']
    assert result['operational_failure'] is (failure=='operational')
    assert result['ready_for_next'] is (failure!='operational')
    assert result['expected_model_gap'] is (failure!='operational')
    assert calls==(['training','holdout'] if failure=='holdout_model' else ['training'])
    assert not result['formal_eligible']


@pytest.mark.parametrize('failure',['holdout','source'])
def test_resident_loader_uses_typed_gap_only_after_raw_and_source_replay(tmp_path,monkeypatch,failure):
    from pdblend.profile.query import native_layout_profile as profile
    from pdblend.profile.collection import native_layout_stage as stage
    ident=dict(system='pdblend',model_id=collect.MODEL,tp=2,pp=1,model_hash='m',tokenizer_hash='t',image_digest='i',engine_revision='e')
    timing=tmp_path/'stage.json';timing.write_text(json.dumps(dict(schema=stage.SCHEMA)))
    selected=tmp_path/'timing-selection.json';selected.write_text(json.dumps(dict(kind=profile.TIMING_KIND,identity=ident,timing=binding(timing))))
    calls=[]
    def source(*args):
        if failure=='source':raise ValueError('native source bytes differ')
        return dict(calibration_source_revisions=['s'])
    def replay(*args,**kwargs):
        calls.append(True)
        return dict(identity=dict(ident,source_revision='s'),component=None,component_qualified=False)
    monkeypatch.setattr(profile,'replay_sources',source);monkeypatch.setattr(stage,'replay_resident_timing',replay)
    with pytest.raises(ValueError) as caught:profile.load_layout_timing(binding(selected))
    assert isinstance(caught.value,profile.LayoutQualificationUnavailable) is (failure=='holdout')
    assert bool(calls) is (failure=='holdout')
