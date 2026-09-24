"""CPU regressions for fail-closed v2 evidence and independent tuning scope."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.bench.capacity_floor_v2 import validate_manifest, summarize, runtime_context
from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_pdblend_acceptance import _plan_capacity_floor
from pdblend.bench.low_m_tuning import prepare, FAMILIES, source_inventory, TrialPlanner, trace_requests, trace_rate_metadata
from pdblend.bench.resident_session import digest
from pdblend.planner.capacity import load_capacity_floors, floor_identity
from pdblend.planner.pool import PlannerConfig, PoolPlanner, QualifiedCapacityFloor, SLO, Plan
from synthetic import synthetic_model, fc


CONTEXT = dict(algorithm_source_sha256='a'*64,workload_family_sha256='b'*64,recovery_policy_sha256='c'*64)
RECOVERY = dict(startup_safety=True,deadline_safety=True,preserve_overload_capacity=True,
    safety_recovery=True,slo_routing=True,capacity_floor_reserve_canonical=True,
    shield_mode='budget_aware',safety_max_freq=2100)


def planner():
    floor = QualifiedCapacityFloor('synthetic',1,1,2,(0.,1.),(1.,4096.),(1.,512.),'raw-evidence',
        qualified=True,profile_key='{}',accepted_slo=(1.,.1),version=2,frequency_mhz=1800,context=CONTEXT)
    return PoolPlanner(synthetic_model(),PlannerConfig(8,SLO(1.,.1),min_m_instances=4,
        capacity_floors=(floor,),capacity_floor_context=CONTEXT,capacity_floor_reserve_canonical=True))


def test_v2_only_allows_exact_count_clock_context_and_m_l1_layout():
    pl = planner(); demand = fc(.05)
    assert pl.evaluate({'M':2,'L1':6},2520,2520,1800,0,demand,False) is not None
    for counts,freq,tau in [({'M':2,'L1':6},1500,0),({'M':3,'L1':5},1800,0),
                            ({'M':2,'P':1,'D':1,'L1':4},1800,1024),({'M':2,'off':6},1800,0)]:
        assert pl.evaluate(counts,2520,2520,freq,tau,demand,False) is None
    pl.cfg.capacity_floor_context = dict(CONTEXT,algorithm_source_sha256='changed')
    assert pl.mixed_floor(demand) == 4
    assert pl.evaluate({'M':2,'L1':6},2520,2520,1800,0,demand,False) is None


def test_unknown_v2_domain_restores_m4_and_replay_matches_final_layout():
    pl = planner(); demand = fc(.05)
    low = pl.evaluate({'M':2,'L1':6},2520,2520,1800,0,demand,False)
    restored = pl.enforce_capacity_floor(low,fc(2.))
    assert restored.counts['M'] == 4 and restored.f_M == 2520
    assert restored.detail['capacity_floor']['n_m'] == 4
    selection = dict(profile_key='{}',frequencies=list(pl.model.freqs),capacity_floor=dict(
        schema='pdblend-capacity-floor-selection/v2',reserve_canonical=True,canonical_floor=4,
        recovery_frequency_mhz=2520,context=CONTEXT,floors=[asdict(pl.cfg.capacity_floors[0])],
        model=dict(model='synthetic',tp=1,pp=1,profile_key={}),slo=dict(ttft_s=1.,tpot_s=.1)))
    for plan,expected in [(low,2),(restored,4)]:
        logged = dict(counts=plan.counts,f_M=plan.f_M,tau=plan.tau,query_results=plan.detail)
        assert _plan_capacity_floor(logged,selection,8) == expected


def test_v1_floor_identity_is_unchanged_by_new_default_fields():
    floor = replace(planner().cfg.capacity_floors[0],version=1,frequency_mhz=None,context={})
    previous = asdict(floor)
    for key in ('version','frequency_mhz','context'): previous.pop(key)
    assert floor_identity(floor) == hashlib.sha256(json.dumps(previous,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def test_refresh_final_estimate_preserves_identity_and_clears_stale_values(monkeypatch):
    pl = planner(); demand = fc(.05)
    plan = pl.evaluate({'M':2,'L1':6},2520,2520,1800,0,demand,False)
    plan = replace(plan,pool_id='a',generation=19,power_w=-100,ttft_s=-100,detail=dict(plan.detail,safety_note='keep'))
    result = pl.refresh_estimate(plan,demand)
    assert result.key() == plan.key() and result.power_w > 0 and result.detail['safety_note'] == 'keep'
    monkeypatch.setattr(pl,'evaluate',lambda *a,**kw:None)
    result = pl.refresh_estimate(plan,demand)
    assert result.key() == plan.key() and math.isinf(result.power_w) and math.isinf(result.ttft_s)
    assert result.detail['prediction_unavailable'] and 'M' not in result.detail


def prepared(tmp_path,monkeypatch):
    corpus = tmp_path/'corpus'; corpus.mkdir()
    records = [dict(prompt=[1]*(4+i),output_tokens=16+i) for i in range(10)]
    data = dict(model_name='synthetic',dataset='sharegpt',tuning=records,evaluation=[dict(prompt=[999],output_tokens=2)])
    (corpus/'sharegpt.json').write_text(json.dumps(data))
    (corpus/'manifest.json').write_text(json.dumps(dict(complete=True,model_name='synthetic',
        dataset_sha256={'sharegpt':binding(corpus/'sharegpt.json')['sha256']},
        tokenizer_sha256='tokenizer',model_config_sha256='model')))
    profile = tmp_path/'profile.json'; profile.write_text('{}')
    monkeypatch.setattr('pdblend.profile.query.versions.load_profile',lambda *a,**kw:SimpleNamespace(model=synthetic_model()))
    output = tmp_path/'tuning'
    manifest = prepare(output=output,corpus=corpus,profile=profile,recovery=RECOVERY,
                       frequencies=(2100,),model_id='synthetic')
    return output/'manifest.json',manifest


def test_preparation_replays_three_independent_seeds_and_all_stress_families(tmp_path,monkeypatch):
    path,manifest = prepared(tmp_path,monkeypatch)
    assert len(manifest['trials']) == 2*3*len(FAMILIES)
    assert validate_manifest(path) == manifest
    result = summarize(path,[])
    assert result['selected'] == [] and len(result['missing_trial_ids']) == len(manifest['trials'])
    for trial in manifest['trials']:
        trace = json.loads(open(trial['trace']['path']).read())
        assert all(999 not in r['prompt'] for r in trace['requests'])
        assert trial['plan']['counts']['M'] in (2,3) and trace['seed'] != 701


@pytest.mark.parametrize('seed', [8801, 8802, 8803])
@pytest.mark.parametrize('rate', [2., 4.])
@pytest.mark.parametrize('family,stages', [
    ('burst25', [(60.,1.), (30.,1.25), (60.,1.)]),
    ('initial_burst', [(30.,1.25), (120.,1.)]),
    ('tail_burst', [(120.,1.), (30.,1.25)]),
])
def test_short_bursts_replay_exact_thirty_second_stages(seed, rate, family, stages):
    from pdblend.bench.client import staged_trace
    records = [dict(prompt=[1]*64, output_tokens=16), dict(prompt=[2]*128, output_tokens=32)]
    actual = trace_requests(records, family=family, rate=rate, duration=150., seed=seed)
    expected = staged_trace(records, rate, stages, 1., seed, source='sharegpt:tuning:'+family)
    assert actual == expected
    assert all(row.idx == index and 0 <= row.arrival_s < 150 for index,row in enumerate(actual))
    metadata = trace_rate_metadata(family=family, rate=rate, duration=150.)
    assert metadata['rate_rps'] == pytest.approx(rate*1.05)
    assert metadata['peak_rate_rps'] == rate*1.25
    assert metadata['rate_rps_semantics'] == 'time_weighted_target_rate'
    bursts = [row for row in metadata['rate_stages'] if row['rate_rps'] > rate]
    assert len(bursts) == 1 and bursts[0]['end_s']-bursts[0]['start_s'] == 30.
    if family == 'burst25':
        assert bursts[0]['start_s'] == 60. and bursts[0]['end_s'] == 90.


@pytest.mark.parametrize('seed', [8801, 8802, 8803])
def test_nominal_trace_seed_semantics_remain_unchanged(seed):
    from pdblend.bench.client import poisson_trace
    records = [dict(prompt=[1]*64, output_tokens=16)]
    assert trace_requests(records,family='nominal',rate=2.,duration=150.,seed=seed) == poisson_trace(
        records,2.,150.,seed,source='sharegpt:tuning:nominal')


@pytest.mark.parametrize('fault', ['mean', 'stages', 'missing_peak', 'constant_rate_requests'])
def test_rebound_burst_trace_cannot_hide_an_entire_window_rate_increase(tmp_path, monkeypatch, fault):
    from pdblend.bench.client import poisson_trace
    from pdblend.bench.capacity_workloads import corpus_inputs
    path,manifest = prepared(tmp_path,monkeypatch)
    trial = next(row for row in manifest['trials'] if row['family']=='burst25')
    trace = json.loads(Path(trial['trace']['path']).read_text())
    rate = trial['nominal_rate_rps']
    if fault == 'mean':
        trial['rate_rps'] = trace['rate_rps'] = rate*1.25
    elif fault == 'stages':
        trial['rate_stages'] = trace['rate_stages'] = [dict(start_s=0.,end_s=150.,rate_rps=rate*1.25)]
    elif fault == 'missing_peak':
        trial.pop('peak_rate_rps'); trace.pop('peak_rate_rps')
    else:
        records,_,_ = corpus_inputs(Path(manifest['corpus']['corpus_manifest']['path']).parent,'sharegpt','tuning')
        trace['requests'] = [asdict(row) for row in poisson_trace(records,rate*1.25,150.,trial['seed'],
                                                                  source='sharegpt:tuning:burst25')]
    changed = tmp_path/'changed-burst.json'; changed.write_text(json.dumps(trace))
    trial['trace'] = binding(changed); path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='does not replay' if fault=='constant_rate_requests' else 'rate stages'):
        validate_manifest(path)


def test_stored_passed_flag_without_gpu_receipts_cannot_issue_v2_floor(tmp_path,monkeypatch):
    path,manifest = prepared(tmp_path,monkeypatch)
    artifact = tmp_path/'floor.json'
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v2',identity=manifest['identity'],
        tuning_manifest=binding(path),trial_receipts=[],floors=[dict(passed=True,qualified=True)])))
    with pytest.raises(ValueError,match='complete independent'):
        load_capacity_floors(artifact,model=synthetic_model())


def test_trace_relabel_or_seed_substitution_is_rejected(tmp_path,monkeypatch):
    path,manifest = prepared(tmp_path,monkeypatch)
    trial = manifest['trials'][0]; trace_path = tmp_path/'changed.json'
    trace = json.loads(open(trial['trace']['path']).read()); trace['requests'][0]['prompt']=[999]
    trace_path.write_text(json.dumps(trace)); trial['trace']=binding(trace_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='does not replay'):
        validate_manifest(path)
    manifest['seeds'][0]=701; path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='non-evaluation seeds'):
        validate_manifest(path)


def test_looser_manifest_and_rebound_trace_cannot_promote_original_slo(tmp_path,monkeypatch):
    path,manifest = prepared(tmp_path,monkeypatch)
    trial=manifest['trials'][0]; trial['slo']=dict(ttft_s=100.,tpot_s=1.)
    trace_path=tmp_path/'loose-trace.json'
    trace=json.loads(open(trial['trace']['path']).read()); trace['slo']=trial['slo']
    trace_path.write_text(json.dumps(trace)); trial['trace']=binding(trace_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='original ShareGPT SLO'):
        validate_manifest(path)


def test_runtime_context_recomputes_algorithm_family_and_recovery(tmp_path,monkeypatch):
    path,manifest = prepared(tmp_path,monkeypatch)
    artifact = tmp_path/'floor.json'
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v2',tuning_manifest=binding(path))))
    family = manifest['context']['workload_family_sha256']
    assert runtime_context(artifact,runtime_options=RECOVERY,workload_family_sha256=family,nominal_rate_rps=2.)==dict(manifest['context'],nominal_rate_rps=2.)
    assert runtime_context(artifact,runtime_options=RECOVERY,workload_family_sha256=family,nominal_rate_rps=3.)=={}
    assert runtime_context(artifact,runtime_options=RECOVERY,workload_family_sha256='other')=={}
    assert runtime_context(artifact,runtime_options=dict(RECOVERY,safety_max_freq=2520),workload_family_sha256=family)=={}
    monkeypatch.setattr('pdblend.bench.capacity_floor_v2.source_inventory',lambda root:{'changed.py':'sha'})
    assert runtime_context(artifact,runtime_options=RECOVERY,workload_family_sha256=family)=={}


def test_trial_planner_has_no_qualified_floor_and_uses_canonical_on_domain_exit():
    trial = dict(id='cpu',nominal_rate_rps=.1,plan=dict(counts={'M':2,'L1':6},f_P=2100,f_D=2100,f_M=1800,tau=0))
    pl = TrialPlanner(synthetic_model(),PlannerConfig(8,SLO(5.,.15),min_m_instances=4,freqs=(1800,2100)),trial)
    assert pl.cfg.capacity_floors == ()
    low=pl.plan(fc(.11))
    assert low.counts['M'] == 2 and pl.capacity_reserve_enabled
    assert pl.enforce_capacity_floor(low,fc(.11),force_canonical=True).counts['M']==4
    assert pl.enforce_capacity_floor(low,fc(.05)).counts['M']==2
    assert pl.enforce_capacity_floor(low,fc(.2)).counts['M']==4
    assert pl.plan(fc(.2)).counts['M'] >= 4


def trial_receipt(tmp_path,monkeypatch):
    """Synthetic raw-native fixture; these files are CPU test evidence only."""
    from test_comparison_pdblend_acceptance import fixture,raw,write
    from test_comparison_acceptance import put,state
    from collections import Counter
    args = fixture(tmp_path,monkeypatch,profile_key='{}')
    outcomes = raw(args,'outcomes'); outcomes[0]['sampling_seed']=8801
    write(args,'outcomes',outcomes)
    events = raw(args,'controller'); initial = next(row for row in events if row.get('kind')=='plan')
    initial.update(counts={'M':4,'L1':4},f_P=2100,f_D=2100,f_M=2100,tau=0,
        roles={f'mixed{i}':'M' if i<4 else 'L1' for i in range(8)})
    low = dict(initial,t=130.1,counts={'M':2,'L1':6},f_M=1500,
        roles={f'mixed{i}':'M' if i<2 else 'L1' for i in range(8)})
    completed = next(row for row in events if row.get('kind')=='transition_complete')
    low_complete = dict(kind='transition_complete',t=130.8,started_s=130.,finished_s=130.8,
        transition_id='low',affected=[f'mixed{i}' for i in range(4)])
    low['t']=130.7
    phases=[]
    for tid,start,parked,active in [('initial',99.1,range(4,8),range(4)),('low',130.,range(2,4),range(2))]:
        for gpu in (*active,*parked):
            operations = (['clock_set','route_publish'] if gpu in active else
                          ['route_publish','proxy_drain','native_drain','clock_reset','park'])
            for step,operation in enumerate(operations):
                begin=start+step*.06+gpu*.001; end=begin+.04
                row=dict(instance=f'mixed{gpu}',operation=operation,gpus=[gpu],started_s=begin,finished_s=end,
                    duration_s=.04,generation=5,source_role='M',source_frequency_mhz=None,
                    transition_id=tid,status='passed',energy_status='awaiting_common_sampler_integration')
                if operation=='native_drain': row['native_receipt']=dict(state(1,end,5),acknowledged=True,drained=True)
                phases.append(row)
    phases.sort(key=lambda row:row['finished_s'])
    events = [row for row in events if row.get('kind')!='transition_phase']
    events += [dict(row,kind='transition_phase',t=row['finished_s']) for row in phases]+[low,low_complete]
    next(row for row in events if row.get('kind')=='stop')['roles']=low['roles']
    events.sort(key=lambda row:row['t'])
    write(args,'controller',events)
    native=raw(args,'native_result'); native['final_roles']=low['roles']
    native['controller']=dict(events=dict(Counter(row['kind'] for row in events)),transition_phases=phases,
                              final_roles=low['roles'])
    write(args,'native_result',native)
    write(args,'transition_measurements',dict(phases=[dict(row,energy_status='measured',energy_j=1.) for row in phases],
                                            incremental_energy_j=None,formal_eligible=False))
    samples = [[99.5+i*.5,[2100 if g<4 else 210 for g in range(8)]
                if 99.5+i*.5<130 else [1500 if g<2 else 210 for g in range(8)]] for i in range(303)]
    write(args,'frequencies',samples)
    readings = [dict(gpu=g,read_started_s=t-.0001,read_finished_s=t,observed_mhz=f,error=None,power_limit_w=350)
                for t,values in samples for g,f in enumerate(values)]
    refs = args['raw_refs']
    artifacts = {key:refs[key] for key in ('outcomes','controller','native_result','drain','native_cleanup',
                                          'metering','power','frequencies','reset','routes','transition_measurements')}
    artifacts.update(requests=refs['canonical_requests'],startup=refs['startup_qualification'],
                     frequency_readings=put(tmp_path,'readings.jsonl',readings,journal=True))
    trial = dict(id='cpu-only',seed=8801,family='nominal',nominal_rate_rps=2.,rate_rps=2.,trace=refs['trace'],
        plan=dict(counts={'M':2,'L1':6},f_P=2100,f_D=2100,f_M=1500,tau=0),slo=dict(ttft_s=5.,tpot_s=.15))
    source = json.loads(open(args['startup_qualification']['source_manifest']['path']).read())
    manifest = dict(context=CONTEXT,identity=dict(model_id=args['point']['model_id'],tp=1,pp=1,profile_key={}),
        trials=[trial],recovery_policy=RECOVERY,profile_frequencies=[1500,2100,2520],source_files={k:v for k,v in source['files'].items()
            if k.startswith(('pdblend/','pdblend_runtime/','pdblend_baselines/')) and k.endswith('.py')},
        startup_plan=dict(counts={'M':4,'L1':4},f_P=2100,f_D=2100,f_M=2100,tau=0))
    manifest_ref = put(tmp_path,'tuning-manifest.json',manifest)
    receipt = dict(kind='pdblend_low_m_trial_v2',hardware_executed=True,selection_split='tuning',
        evaluation_used_for_selection=False,manifest=manifest_ref,executed_context=CONTEXT,trial_id=trial['id'],
        actual_plan=trial['plan'],engine_identity=args['engine_identity'],artifacts=artifacts)
    path = tmp_path/'trial-receipt.json'; path.write_text(json.dumps(receipt))
    return path,manifest,manifest_ref,receipt


def test_complete_raw_trial_recomputes_actual_clocks_slo_and_full_energy(tmp_path,monkeypatch):
    from pdblend.bench.capacity_floor_v2 import validate_trial
    path,manifest,ref,_ = trial_receipt(tmp_path,monkeypatch)
    result = validate_trial(path,manifest=manifest,manifest_ref=ref)
    assert result['stable_low_m_s'] > 100 and result['energy_service_tail_j'] > 0
    assert result['joint_slo_requests'] == result['offered_requests'] == 1


def test_nominal_energy_selects_frequency_while_stress_is_only_a_qualification_gate(tmp_path,monkeypatch):
    from copy import deepcopy
    from pdblend.bench import capacity_floor_v2 as module
    path,manifest=prepared(tmp_path,monkeypatch)
    alternate=[]
    for original in manifest['trials']:
        trial=deepcopy(original);trial['id']=trial['id'].replace('f2100','f1800');trial['plan']['f_M']=1800
        alternate.append(trial)
    manifest['trials']+=alternate;path.write_text(json.dumps(manifest))
    by_id={trial['id']:trial for trial in manifest['trials']}
    paths=[]
    for trial in manifest['trials']:
        receipt=tmp_path/(trial['id']+'.json');receipt.write_text('{}');paths.append(receipt)
    def measured(receipt,**kwargs):
        trial=by_id[receipt.stem];f=trial['plan']['f_M'];nominal=trial['family']=='nominal' and trial['seed']==8801
        energy=(100 if f==2100 else 110) if nominal else (1000 if f==2100 else 1)
        return dict(trial_id=trial['id'],nominal_rate_rps=trial['nominal_rate_rps'],min_m_instances=trial['plan']['counts']['M'],
            frequency_mhz=f,seed=trial['seed'],family=trial['family'],energy_service_tail_j=energy,
            input_range=[4,13],output_range=[2,25],receipt=binding(receipt))
    monkeypatch.setattr(module,'validate_trial',measured)
    selected=module.summarize(path,paths)['selected']
    assert {row['frequency_mhz'] for row in selected}=={2100}
    assert all(row['objective_energy_service_tail_j']==100 for row in selected)
    # One missing stress trial prevents issuing that frequency's domain even
    # though its nominal energy is the lowest.
    paths=[p for p in paths if p.stem!='r2-m2-f2100-s8803-tail_burst']
    selected=module.summarize(path,paths)['selected']
    assert next(row for row in selected if row['nominal_rate_rps']==2.)['frequency_mhz']==1800


@pytest.mark.parametrize('fault',['context','clock','candidate_count','energy','late_cleanup','slo','startup'])
def test_raw_trial_rejects_unqualified_candidate_and_incomplete_cycle(tmp_path,monkeypatch,fault):
    from pdblend.bench.capacity_floor_v2 import validate_trial
    from test_comparison_acceptance import put
    path,manifest,ref,receipt = trial_receipt(tmp_path,monkeypatch)
    if fault=='context': receipt['executed_context']={}
    elif fault=='candidate_count': receipt['actual_plan']=dict(receipt['actual_plan'],counts={'M':1,'L1':7})
    elif fault=='clock':
        rows=[json.loads(line) for line in open(receipt['artifacts']['frequency_readings']['path'])]
        next(row for row in rows if row['read_finished_s']==200 and row['gpu']==0)['observed_mhz']=900
        receipt['artifacts']['frequency_readings']=put(tmp_path,'bad-clock.jsonl',rows,journal=True)
    elif fault=='energy':
        meter=json.loads(open(receipt['artifacts']['metering']['path']).read()); meter['energy_service_tail_j']=0
        receipt['artifacts']['metering']=put(tmp_path,'bad-meter.json',meter)
    elif fault=='late_cleanup':
        drain=json.loads(open(receipt['artifacts']['drain']['path']).read()); drain['states'][0]['received_s']=251.
        receipt['artifacts']['drain']=put(tmp_path,'late-drain.json',drain)
    elif fault=='slo':
        manifest['trials'][0]['slo']['ttft_s']=.01
    else:
        manifest['startup_plan']['counts']={'M':2,'L1':6}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): validate_trial(path,manifest=manifest,manifest_ref=ref)


@pytest.mark.parametrize('fault',['roles_count','no_park','missing_ack','overlap'])
def test_rebound_summary_cannot_hide_forged_physical_low_m_execution(tmp_path,monkeypatch,fault):
    from collections import Counter
    from pdblend.bench.capacity_floor_v2 import validate_trial
    from test_comparison_acceptance import put
    path,manifest,ref,receipt = trial_receipt(tmp_path,monkeypatch)
    events=[json.loads(line) for line in open(receipt['artifacts']['controller']['path'])]
    if fault=='roles_count':
        next(row for row in events if row.get('kind')=='plan' and row['counts']['M']==2)['roles']['mixed2']='M'
    elif fault=='no_park':
        events=[row for row in events if not (row.get('transition_id')=='low'
                and row.get('instance')=='mixed2' and row.get('operation')=='park')]
    elif fault=='missing_ack':
        next(row for row in events if row.get('operation')=='native_drain')['native_receipt']['acknowledged']=False
    else:
        actions=[row for row in events if row.get('kind')=='transition_phase'
                 and row.get('transition_id')=='low' and row.get('instance')=='mixed2']
        actions[1]['started_s']=actions[0]['started_s']
    # Rebind *both* summaries, so rejection must be a physical invariant rather
    # than a convenient mismatch against an unmodified cached summary.
    bare=[{key:value for key,value in row.items() if key not in ('kind','t')}
          for row in events if row.get('kind')=='transition_phase']
    native=json.loads(open(receipt['artifacts']['native_result']['path']).read())
    native['controller']['transition_phases']=bare
    native['controller']['events']=dict(Counter(row['kind'] for row in events))
    receipt['artifacts']['native_result']=put(tmp_path,'forged-native.json',native)
    receipt['artifacts']['controller']=put(tmp_path,'forged-controller.jsonl',events,journal=True)
    receipt['artifacts']['transition_measurements']=put(tmp_path,'forged-transitions.json',dict(
        phases=[dict(row,energy_status='measured',energy_j=1.) for row in bare],
        incremental_energy_j=None,formal_eligible=False))
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): validate_trial(path,manifest=manifest,manifest_ref=ref)


def test_session_completion_is_written_after_actual_cleanup_and_on_error(tmp_path,monkeypatch):
    import asyncio
    from pdblend.bench import low_m_tuning_runtime as module
    group = tmp_path/'group.json'; group.write_text(json.dumps(dict(scope='independent_low_m_tuning/v2')))
    class FakeAdapter:
        def __init__(self,out,**kw): self.out=out
        async def start(self,group): self.tuning_manifest={'trials':[{'id':'one'}]}
        async def execute_trial(self,*args): raise ValueError('injected GPU-free execution failure')
        async def close(self):
            assert not (self.out/'completion.json').exists()
            return dict(passed=True,process_cleanup_verified=True,errors=[])
    monkeypatch.setattr(module,'LowMTuningAdapter',FakeAdapter)
    out=tmp_path/'session'
    with pytest.raises(ValueError,match='injected'):
        asyncio.run(module.run_group(group,out))
    completion=json.loads((out/'completion.json').read_text())
    assert completion['status']=='failed' and completion['complete'] is False
    assert completion['cleanup']['process_cleanup_verified'] is True
    assert completion['cleanup_receipt']==binding(out/'cleanup.json')


def test_successful_tuning_session_receipt_is_accepted_by_real_lease_worker(tmp_path,monkeypatch):
    import asyncio
    from pdblend.bench import low_m_tuning_runtime as module
    from pdblend.experimentation.worker import receipt
    group=tmp_path/'group.json';group.write_text(json.dumps(dict(scope='independent_low_m_tuning/v2')))
    class FakeAdapter:
        def __init__(self,out,**kwargs):self.out=out
        async def start(self,group):self.tuning_manifest={'trials':[{'id':'one'}]}
        async def execute_trial(self,*args):return dict(accepted=True)
        async def close(self):
            assert not (self.out/'completion.json').exists()
            return dict(passed=True,process_cleanup_verified=True,errors=[])
    monkeypatch.setattr(module,'LowMTuningAdapter',FakeAdapter)
    asyncio.run(module.run_group(group,tmp_path/'session'))
    verified=receipt(tmp_path,['session/completion.json'])
    assert verified['session/completion.json']==binding(tmp_path/'session/completion.json')['sha256']
    complete=json.loads((tmp_path/'session/completion.json').read_text())
    assert complete['status']=='passed' and complete['complete'] and complete['all_trials_qualified']


def test_tuning_stop_file_preserves_current_trial_and_continuation_after_cleanup(tmp_path,monkeypatch):
    import asyncio
    from pdblend.bench import low_m_tuning_runtime as module
    group=tmp_path/'group.json';group.write_text(json.dumps(dict(scope='independent_low_m_tuning/v2')))
    calls=[]
    class FakeAdapter:
        def __init__(self,out,**kwargs):self.out=out
        async def start(self,group):self.tuning_manifest={'trials':[{'id':'one'},{'id':'two'}]}
        async def execute_trial(self,trial_id,out):
            calls.append(trial_id)
            (self.out/'stop-after-window').touch()
            # The sentinel cannot interrupt this trial's receipt persistence.
            out.mkdir(parents=True);(out/'trial-receipt.json').write_text('{"preserved":true}')
            return dict(accepted=True)
        async def close(self):
            assert (self.out/'trials/one/trial-receipt.json').is_file()
            assert not (self.out/'completion.json').exists()
            return dict(passed=True,process_cleanup_verified=True,errors=[])
    monkeypatch.setattr(module,'LowMTuningAdapter',FakeAdapter)
    out=tmp_path/'session';asyncio.run(module.run_group(group,out))
    completion=json.loads((out/'completion.json').read_text())
    assert calls==['one'] and completion['status']=='interrupted'
    assert completion['complete'] is False and completion['all_trials_qualified'] is False
    assert completion['remaining_trial_ids']==['two'] and completion['continuation_required']
    assert completion['cleanup']['passed'] is True and completion['group']==binding(group)
