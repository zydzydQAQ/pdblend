from copy import deepcopy
import json

import pytest

from ecopadg.serving import dynamo_validation_setup as setup


def test_trace_uses_real_clock_and_all_nine_public_request_shapes():
    trace = setup.trace(77)
    assert trace == setup.trace(77)
    assert trace['requests'][0]['arrival_s'] == 0
    assert trace['requests'][-1]['arrival_s'] == trace['duration_s'] == 1890
    assert {(r['prompt_len'],r['output_len']) for r in trace['requests']} == {
        (n,out) for n in (128,512,2048) for out in (64,192,384)}
    assert all(len(p)==r['prompt_len'] for p,r in zip(trace['prompts'],trace['requests']))
    with pytest.raises(ValueError): setup.trace(rates=[.1]*5)


def test_development_step_preserves_real_periods_and_later_shape_coverage():
    trace=setup.trace(rates=(.35,.08,.12,.08,.06,.04),arrival_process='periodic',
        phase_shapes=[[(7168,64)],None,None,None,None,None])
    prefix=[r for r in trace['requests'] if r['arrival_s']<=300]
    assert len(prefix)==105
    assert {(r['prompt_len'],r['output_len']) for r in prefix}=={(7168,64)}
    assert trace['requests'][-1]['arrival_s']==1890
    assert {(n,out) for n in (128,512,2048) for out in (64,192,384)} <= {
        (r['prompt_len'],r['output_len']) for r in trace['requests']}
    with pytest.raises(ValueError,match='arrivals'): setup.trace(arrival_process='forced')
    with pytest.raises(ValueError,match='shapes'): setup.trace(phase_shapes=[None]*5)
    with pytest.raises(ValueError,match='shapes'): setup.trace(phase_shapes=[[(128,-1)]]*6)


def test_shard_prediction_uses_only_arrived_prefix_and_real_policy():
    from ecopadg.serving.profiles import ProfilePoint,ProfileStore
    from ecopadg.serving.topology import InstanceSpec
    profiles=ProfileStore([ProfilePoint('mixed',tp,2520,7168,7680,1,prefill,.01,
        200*tp,50*tp,.05,1,'cpu-fixture') for tp,prefill in [(1,.5),(2,2.)]])
    instances=[InstanceSpec('fallback',1,(2,),24000,28000,'mixed'),
        InstanceSpec('native',2,(0,1),24001,28016,'mixed'),
        InstanceSpec('short',1,(3,),24002,28032,'mixed')]
    assignments=dict(fallback='LL',native='LM',short='SM')
    costs=[dict(source_tps=(2,),target_tps=(1,1),duration_upper_s=60,
        energy_upper_j=10000,source_sha256='cpu-fixture',cached_weights=True)]
    workload=setup.trace(rates=(.35,.08,.12,.08,.06,.04),arrival_process='periodic',
        phase_shapes=[[(7168,64)],None,None,None,None,None])
    cold=setup.predict_scale_shard(profiles,costs,instances,assignments,workload,512,5,.1)
    assert 'LS' not in cold['forecast'] and cold['forecast']['LL']['rate']==.4
    assert cold['status']=='cpu_prediction_not_executed' and cold['assumed_completion'] is None
    result=setup.predict_scale_shard(profiles,costs,instances,assignments,workload,512,5,.1,
        assume_first_completed_by_s=180)
    assert result['forecast']['LS']['rate']==.4
    assert result['proposal']['remove_ids']==('native',)
    assert result['proposal']['add_tps']==(1,1)
    assert result['proposal']['capacity_recovery']['required_rps']==.4
    assert result['status']=='conditional_cpu_prediction_not_executed'
    assert result['assumed_completion']['measured'] is False
    changed=deepcopy(workload)
    changed['requests']=[r for r in changed['requests'] if r['arrival_s']<=300]+[
        dict(arrival_s=300.001,prompt_len=100000,output_len=100000)]
    assert setup.predict_scale_shard(profiles,costs,instances,assignments,changed,512,5,.1,
        assume_first_completed_by_s=180)==result
    assert setup.predict_scale_shard(profiles,[],instances,assignments,workload,512,5,.1,
        assume_first_completed_by_s=180)['proposal'] is None
    # Prescribed outputs cannot change any cold prediction without an explicit
    # assumed historical completion in this offline-only scenario analysis.
    changed=deepcopy(workload)
    for r in changed['requests']: r['output_len']=32
    assert setup.predict_scale_shard(profiles,costs,instances,assignments,changed,512,5,.1)==cold
    with pytest.raises(ValueError,match='completion deadline'):
        setup.predict_scale_shard(profiles,costs,instances,assignments,workload,512,5,.1,
            assume_first_completed_by_s=300)


def cycle_fixture():
    raw = dict(period_origin_s={k:0 for k in setup.PERIODS}, errors=[], routes={'r':'survivor'},
        workload=dict(requests=[dict(prompt_len=128,output_len=64)]),
        reference_keys={'0':'1:128:64'}, reference={'1:128:64':[7]*64})
    events = [dict(kind='dynamo_control_epoch',operation=name,period_s=period,
                   at_s=period*index,executed=name=='ScaleFreq',plan=dict(frequencies=[]))
              for name,period in setup.PERIODS.items() for index in range(1,int(1890//period)+1)]
    rows = [dict(request_id='0',success=1,generated_tokens=64,
                 output_token_sha256=setup.token_hash([7]*64))]
    summary = dict(validity='ok',measurement_schema=2,gpu_count=8,energy_j=100,
                   power_mode='instant',power_source_verified=True,
                   measurement_start_s=0,measurement_end_s=1900)
    return raw,events,rows,summary


def test_real_period_coverage_never_claims_unexecuted_reconfiguration():
    raw,events,rows,summary = cycle_fixture()
    result = setup.audit(raw,events,rows,summary)
    assert result['cycles_passed'] and not result['passed']
    assert result['actual_action_counts'] == dict(ScaleInst=0,ScaleShard=0,ScaleFreq=0)
    assert result['status'] == 'cycles_only_or_missing_actions'
    assert not result['proposed_mechanism_fields']['independent_calibration']
    assert not result['proposed_mechanism_fields']['scale_shard_300s']


def test_proposal_does_not_promote_missing_actions_or_invalid_run(tmp_path):
    raw,events,rows,summary=cycle_fixture()
    result=setup.audit(raw,events,rows,summary)
    path=tmp_path/'mechanisms.json';setup.write(path,result)
    proposal=setup.mechanism_proposal(path)['dynamollm']
    assert proposal['output_correctness']['passed']
    assert not proposal['scale_inst_1800s']['passed']
    assert not proposal['independent_calibration']['passed']
    result['status']='invalid_or_incomplete';setup.write(path,result)
    assert not any(v['passed'] for v in setup.mechanism_proposal(path)['dynamollm'].values())


def test_frequency_plans_are_not_changed_driver_commands():
    raw,events,rows,summary = cycle_fixture()
    for e in events:
        if e['operation']=='ScaleFreq': e['plan']['frequencies']=[{'instance_id':'a','frequency_mhz':1500}]
    events += [dict(kind='dynamo_clock_commit',before={'a':2520},after={'a':1500}),
               dict(kind='dynamo_clock_commit',before={'a':1500},after={'a':1500})]
    result = setup.audit(raw,events,rows,summary)
    assert result['frequency_nonempty_plan_epochs']==378
    assert result['actual_action_counts']['ScaleFreq']==1


def test_compressed_or_missing_cycles_and_wrong_outputs_fail():
    raw,events,rows,summary = cycle_fixture()
    compressed=deepcopy(events)
    for index,e in enumerate(compressed): e['at_s']=index*.01
    assert not setup.audit(raw,compressed,rows,summary)['cycles_passed']
    missing=[e for e in events if not (e['operation']=='ScaleShard' and e['at_s']==600)]
    assert not setup.audit(raw,missing,rows,summary)['cycles_passed']
    rows[0]['output_token_sha256']='wrong'
    assert not setup.audit(raw,events,rows,summary)['proposed_mechanism_fields']['output_correctness']


def test_average_or_unverified_power_source_cannot_pass_current_validation():
    raw,events,rows,summary=cycle_fixture()
    assert not setup.audit(raw,events,rows,dict(summary,power_mode='average'))['measurement_valid']
    assert not setup.audit(raw,events,rows,dict(summary,power_source_verified=False))['measurement_valid']


def test_physical_commit_and_unaffected_completions_are_required():
    raw,events,rows,summary = cycle_fixture()
    epoch=next(e for e in events if e['operation']=='ScaleInst')
    epoch.update(executed=True,result={'transaction':'tx'})
    assert setup.audit(raw,events,rows,summary)['actual_action_counts']['ScaleInst']==0
    events += [dict(kind='topology_begin',transaction='tx',at_s=1800,
                    before=[dict(instance_id='old',tp=2)],after=[dict(instance_id='new',tp=1)]),
               dict(kind='topology_commit',transaction='tx',at_s=1810),
               dict(kind='request_end',request_id='r',completed=True,at_s=1805)]
    result=setup.audit(raw,events,rows,summary,[(0,[10]*8),(1900,[10]*8)])
    assert result['actual_action_counts']['ScaleInst']==1
    assert result['proposed_mechanism_fields']['staggered_switch']
    assert result['transactions'][0]['total_eight_gpu_energy_j']==pytest.approx(800)
    assert not result['proposed_mechanism_fields']['scale_shard_300s']


def generator_fixture(tmp_path, monkeypatch):
    corpus=tmp_path/'corpus';corpus.mkdir()
    setup.write(corpus/'sharegpt.json',dict(calibration=[dict(input_tokens=512,output_tokens=192)]*128,
        development='must never become predictor history',formal='must never enter configuration choice'))
    profile=tmp_path/'profiles.json'
    points=[dict(role='mixed',tp=tp,frequency_mhz=2520,input_tokens=2048,context_tokens=2560,
        batch=1,prefill_s=.5,iteration_s=.04,power_w=200,residency_w=30,error_fraction=.1,
        samples=1,source_sha256='fixture') for tp in (1,2)]
    setup.write(profile,dict(schema=2,measurement='hardware',points=points))
    root=tmp_path/'campaign';root.mkdir();setup.write(root/'budget.json',dict(started_s=100,limit_s=86400))
    monkeypatch.setattr(setup.time,'time',lambda:200)
    costs=[dict(source_tps=[2],target_tps=[1,1],duration_upper_s=40,energy_upper_j=1000,
                source_sha256='fixture',cached_weights=True)]
    monkeypatch.setattr(setup,'validated_inputs',lambda m:(dict(points=points),dict(links=[]),{},
        dict(model='/models/Qwen2.5-14B-Instruct',max_model_len=8192),[],[],costs,{}))
    monkeypatch.setattr(setup,'verify_frozen_costs',lambda *args:None)
    return dict(image='sha256:'+'a'*64,profiles=str(profile),campaign_root=str(root),
        corpus=str(corpus),retained_weights=str(tmp_path/'weights'),initial_instances=[],
        instances=[dict(instance_id=f'instance-{i}',tp=tp,gpus=gpus,port=24000+i,
            kv_port=28000+i*8,role='mixed') for i,(tp,gpus) in enumerate([(1,[0]),(1,[1]),(2,[2,3])])])


def test_generator_is_cpu_only_preserves_real_periods_and_calibration_initialization(tmp_path,monkeypatch):
    manifest=generator_fixture(tmp_path,monkeypatch)
    result=setup.generate(manifest,tmp_path/'generated')
    config=setup.read(tmp_path/'generated/runtime-config.json')
    assert result['status']=='prepared_not_executed'
    assert config['output_prior']==192 and config['dynamo_assignments']['instance-0']=='LL'
    assert config['node_gpus']==list(range(8)) and config['manage_clocks']
    assert config['power_mode']=='instant'
    assert not any(k in config for k in ('trace','future_arrivals','phase_rates','output_len'))
    assert result['periods_s']==dict(ScaleInst=1800,ScaleShard=300,ScaleFreq=5)
    stages=setup.read(tmp_path/'generated/campaign.json')['stages']
    assert len(stages)==2 and stages[1]['requires']==[stages[0]['name']]
    with pytest.raises(ValueError,match='overwrite'):setup.generate(manifest,tmp_path/'generated')


def test_generator_rejects_shortened_clock_budget_and_missing_reference_tp(tmp_path,monkeypatch):
    manifest=generator_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='real-clock'):
        setup.generate(dict(manifest,run_limit_s=300),tmp_path/'short')
    incomplete=deepcopy(manifest);incomplete['instances']=incomplete['instances'][:2]
    with pytest.raises(ValueError,match='at least three'):
        setup.generate(incomplete,tmp_path/'incomplete')
    with pytest.raises(ValueError,match='current measured inputs'):
        setup.generate(dict(manifest,require_scale_shard_prediction=True),tmp_path/'not-predicted')
