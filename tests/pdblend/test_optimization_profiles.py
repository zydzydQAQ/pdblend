"""CPU protocol/qualification checks; synthetic fixtures are not GPU evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.profile.calibration import optimization_profiles as cal
from pdblend.profile.collection.window_sampling import summarize_window
from pdblend.profile.query.optimization import OptimizationPowerOverlay, attach_component
from pdblend.profile.query.versions import VersionError, load_profile
from pdblend.profile.query.power_table import LOW_BATCH_KIND, PowerCoverageError, CompiledPowerTable, validate, predict
from pdblend.profile.identity import sha256_value
from test_power_override import model


IDENTITY = dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',model_hash='fixture-model',
                tokenizer_hash='fixture-tokenizer',tp=1,pp=1)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def base(tmp_path):
    m = model()
    m.model = IDENTITY['model_id']
    path = tmp_path/'base.json'
    m.save(path)
    return path, load_profile(path,system='pdblend',model_id=m.model,tp=1)


def native_trace(point,start,end):
    background=[f'native-background-{i}' for i in range(point['batch'])]
    def event(seq,t,prefill=False):
        queue=background+(['native-probe'] if prefill else [])
        return dict(seq=seq,timestamp=t,kind='schedule',generation=0,
                    prefill_mode=prefill,schedule_queue=queue)
    before=dict(events=[event(1,start-.01)],next_seq=1,first_seq=1,gap=False)
    rows=[event(i+2,start+i*.1,point['family']=='mixed' and i%10==0) for i in range(50)]
    return dict(before=before,after=dict(events=rows,next_seq=51,first_seq=1,gap=False),
                before_received_s=start,after_requested_s=end,background_request_ids=background)


def archive(root,base_path,frequency=900):
    """Complete artificial training/holdout archive for validation tests only."""
    plan=cal.make_plan(IDENTITY,frequencies=[frequency])
    write(root/'plan.json',plan)
    manifest=dict(kind=cal.KIND,synthetic_test_only=True)
    write(root/'package-manifest.json',manifest)
    layout={'fixture':['GPU-fixture']}
    receipt=dict(complete=True,cross_job=True,passed=True,cohort_id='synthetic-test-only',
        members=['fixture'],isolated=[dict(gpu_uuids=['GPU-fixture'])],parallel=[dict(gpu_uuids=['GPU-fixture'])])
    write(root/'samples'/'qualification.json',receipt)
    qsha=cal.digest(root/'samples'/'qualification.json')
    qualification=dict(samples_file='samples/qualification.json',samples_sha256=qsha,
        epoch_id='synthetic-test-only',layout_sha256=sha256_value(layout))
    stamp=dict(qualification_sha256=qsha,epoch_id='synthetic-test-only',layout_sha256=sha256_value(layout),layout=layout)
    raw=dict(binding=dict(serving_entrypoint=cal.SERVING_ENTRYPOINT,plan_sha256=cal.digest(root/'plan.json'),
                          package_sha256=cal.digest(root/'package-manifest.json')),training={},holdout={})
    count=0
    for phase in ('training','holdout'):
        for point in plan[phase]:
            row=dict(point=point,repeats=[])
            raw[phase][cal.key(point)]=row
            for repeat in range(3):
                start=1000.+100*count; end=start+5; count+=1
                times=[[start-1+i*.1 for i in range(62)] for _ in range(point['batch'])]
                watts=100 if point['family']=='decode' else 180
                power=[[start+i*.1,[watts]] for i in range(51)]
                clocks=[[start+i*.1,[frequency]] for i in range(51)]
                summary=summarize_window(token_times=times,context=point['context_tokens'],start_s=start,end_s=end,
                    power=power,frequency=clocks,gpu_count=1,settle_s=2,measurement_s=5)
                summary.update(raw_window_mean_power_w=summary['power_w'],energy_j=watts*5,power_w=float(watts))
                probes=[dict(submitted_s=start+i,first_token_s=start+i+.01,finished_s=start+i+.02)
                        for i in range(5)] if point['family']=='mixed' else []
                data=dict(point=point,repeat=repeat,start_s=start,end_s=end,settle_start_s=start-2,
                    serving_entrypoint=cal.SERVING_ENTRYPOINT,
                    token_times_s=times,power=power,frequency=clocks,gpu_count=1,summary=summary,probes=probes,
                    plan_sha256=raw['binding']['plan_sha256'],epoch_binding=stamp)
                data['native_schedule']=native_trace(point,start,end)
                data['native_batch_observation']=cal.native_batch_observation(data)
                path=root/'samples'/f'{count}.json';write(path,data)
                row['repeats'].append(dict(samples_file=str(path.relative_to(root)),samples_sha256=cal.digest(path),
                    qualification=qualification))
    write(root/'raw.json',raw)
    observed=cal.observations(raw,root,plan)
    candidate=cal.fit_component(plan,observed['training'],cal.digest(base_path))
    audit=cal.audit_component(candidate,plan,observed['holdout'])
    write(root/'candidate.json',candidate);write(root/'audit.json',audit)
    completion=dict(complete=True,components_passed=audit['passed'],
        **{name+'_sha256':cal.digest(root/(name+'.json')) for name in ('raw','candidate','audit')})
    write(root/'completion.json',completion)
    return plan,raw,candidate,audit


def test_integrate_common_window_energy_and_endpoint_coverage():
    assert cal.integrate_power([[0,[10,20]],[1,[20,30]]],0,1,2)==40
    with pytest.raises(ValueError,match='endpoint'):
        cal.integrate_power([[1,[10]],[2,[10]]],0,3,1)
    with pytest.raises(ValueError):
        cal.integrate_power([[0,[10]],[0,[20]]],0,1,1)


def test_new_low_batch_schema_is_exact_and_does_not_bridge_legacy_diagnostics():
    m=model();spec=deepcopy(m.decode_power_overrides[900])
    spec['nodes'] += [dict(batch=b,context_min=c,context_max=c+10,power_w=100+b)
                      for b in (2,3) for c in (500,1000)]
    with pytest.raises(ValueError,match='invalid measured power batch'):validate(spec)
    spec['kind']=LOW_BATCH_KIND
    with pytest.raises(ValueError,match='qualification'):validate(spec)
    spec['low_batch_qualification']=dict(independent_holdout_passed=True,exact_batches=[2,3],
        evidence_sha256=['synthetic-test-only'],batch_interpolation_qualified=False)
    table=CompiledPowerTable(spec)
    for batch in (1,2,3,4,6,8):
        assert table.predict(batch,777)==pytest.approx(predict(spec,batch,777))
    for batch in (1.5,2.5,3.5):
        with pytest.raises(PowerCoverageError):table.predict(batch,777)
        with pytest.raises(PowerCoverageError):predict(spec,batch,777)


def test_training_holdout_and_query_remain_separate(tmp_path):
    path,loaded=base(tmp_path)
    plan,raw,candidate,audit=archive(tmp_path/'component',path)
    assert audit['passed']
    combined=attach_component(loaded,tmp_path/'component')
    assert combined.model.decode_power_w(2,900,ctx=1500)==100
    assert combined.model.decode_power_w(3,900,ctx=1500)==100
    assert combined.model.mixed_power_w(8,1500,900,chunk_tokens=512,prefill_rate_rps=1)==180
    assert combined.model.mixed_power_residual_bound_w(8,1500,900,chunk_tokens=512,prefill_rate_rps=1)==0
    assert not combined.model.mixed_power_supported(8,1500,900,chunk_tokens=512,prefill_rate_rps=.5)
    assert not combined.model.decode_power_supported(2.5,1500,900)
    assert combined.model.prefill_seconds(512,900)==loaded.model.prefill_seconds(512,900)
    assert combined.model.step_seconds(2,1500,900)==loaded.model.step_seconds(2,1500,900)
    assert not combined.qualification['optimization_formal_eligible']
    assert combined.model.serving_entrypoint=='pdblend_runtime.serve'
    assert combined.qualification['native_timing_crosscheck_passed'] is False
    measurements=cal.observations(raw,tmp_path/'component',plan)
    bad=deepcopy(measurements['holdout']);bad[0]['repeats'][0]['power_w']=300
    assert not cal.audit_component(candidate,plan,bad)['passed']
    with pytest.raises(ValueError,match='training matrix'):
        cal.fit_component(plan,measurements['holdout'],cal.digest(path))


def test_component_rejects_truncated_repeats_changed_samples_and_other_base(tmp_path):
    path,loaded=base(tmp_path);root=tmp_path/'component'
    plan,raw,_,_=archive(root,path)
    broken=deepcopy(raw);next(iter(broken['holdout'].values()))['repeats'].pop()
    with pytest.raises(ValueError,match='three exact repeats'):cal.observations(broken,root,plan)
    loaded.profile_key['profile_sha256']='different'
    with pytest.raises(VersionError,match='another explicit base'):attach_component(loaded,root)
    sample=root/next(iter(raw['training'].values()))['repeats'][0]['samples_file']
    sample.write_text(sample.read_text()+' ')
    with pytest.raises(ValueError,match='changed or reused'):cal.observations(raw,root,plan)


def test_union_reaudits_disjoint_frequency_panels_without_refit(tmp_path):
    path,loaded=base(tmp_path)
    archive(tmp_path/'f900',path,900);archive(tmp_path/'f2520',path,2520)
    union=cal.merge_components([tmp_path/'f900',tmp_path/'f2520'],tmp_path/'union.json')
    assert union['refit_performed'] is False
    result=attach_component(loaded,tmp_path/'union.json')
    for frequency in (900,2520):
        assert result.model.decode_power_w(2,frequency,ctx=1500)==100
    with pytest.raises(ValueError,match='overlap'):
        cal.merge_components([tmp_path/'f900',tmp_path/'f900'],tmp_path/'invalid-union.json')


def test_missing_domain_priority_uses_demand_and_decision_sensitivity():
    queries=[dict(family='decode',batch=2,context_tokens=1000,freq_mhz=900,query_count=5,decision_sensitivity_j=1),
             dict(family='decode',batch=3,context_tokens=1000,freq_mhz=900,query_count=3,decision_sensitivity_j=10),
             dict(family='decode',batch=8,context_tokens=1000,freq_mhz=900)]
    rows=cal.missing_domain_ledger(model(),queries)
    assert [r['batch'] for r in rows]==[3,2]


def test_mixed_collector_meters_actual_shared_window_and_keeps_decode_timing(monkeypatch):
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from pdblend.profile.collection import optimization_profiles as collect
    clock=SimpleNamespace(now=10.)
    async def sleep(seconds):clock.now+=seconds
    monkeypatch.setattr(collect.time,'time',lambda:clock.now)
    monkeypatch.setattr(collect.asyncio,'sleep',sleep)
    class Live:
        @property
        def token_times_s(self):
            return [9.+i*.02 for i in range(int((clock.now-9.)/.02)+1)]
    @asynccontextmanager
    async def background(*unused):yield [Live() for _ in range(8)],[]
    class Sampler:
        error=None
        def start(self):self.started=clock.now
        def stop(self):
            self.samples=[[self.started+i*.05,[150.]] for i in range(int((clock.now-self.started)/.05)+1)]
            self.frequency_samples=[[t,[1500.]] for t,_ in self.samples]
    class Client:
        async def complete(self,*args):
            submitted=clock.now;await sleep(.01)
            return SimpleNamespace(error=None,stream_done=True,submitted_s=submitted,
                first_token_s=clock.now,finished_s=clock.now,token_times_s=[clock.now])
    profiler=SimpleNamespace(_background=background,_require_running=lambda tasks:None,
                             meter=SimpleNamespace(sampler=lambda gpus:Sampler()),
                             raw=dict(serving_entrypoint=cal.SERVING_ENTRYPOINT))
    point=next(p for p in cal.make_plan(IDENTITY)['training'] if p['family']=='mixed')
    trace=native_trace(point,12.,17.)
    async def events(after_seq):return trace['before'] if after_seq==0 else trace['after']
    evidence=asyncio.run(collect.measure_repeat(profiler,Client(),[0],point,0,_events=events))
    assert len(evidence['warmup_probes'])==2 and len(evidence['probes'])==5
    assert evidence['summary']['energy_j']==pytest.approx(750)
    assert evidence['summary']['power_w']==pytest.approx(150)
    assert evidence['summary']['step_seconds']==pytest.approx(.02)
    assert evidence['summary']['power_scope']=='common_wall_clock_window_with_decode_and_periodic_prefill'
    assert evidence['native_timing_qualified'] is False
    assert evidence['serving_entrypoint']=='pdblend_runtime.serve'
    assert evidence['native_batch_observation']['passed']
    assert any(row['prefill_mode'] for row in evidence['native_batch_observation']['histogram'])


def test_optimization_launch_variant_is_explicit_without_changing_old_specs():
    from pdblend.engine.launcher import InstanceSpec
    from pdblend.profile.collection.optimization_profiles import native_specs
    old=InstanceSpec('i0',(0,),8100,'/models/Qwen2.5-7B-Instruct')
    native=native_specs([old])[0]
    assert old.native_control is False and old.command()[:2]==['vllm','serve']
    assert native.native_control is True and native.command()[1:3]==['-m','pdblend_runtime.serve']


def test_native_optimization_preflight_requires_verified_model_receipt(tmp_path,monkeypatch):
    from pdblend.profile.collection import optimization_profiles as collect
    monkeypatch.setattr(collect,'load_package',lambda _: (dict(tp=1),{}))
    monkeypatch.delenv('PDBLEND_MODEL_VERIFICATION_RECEIPT',raising=False)
    write(tmp_path/'cohort.json',dict(cohort_id='fixture',members=['m']))
    with pytest.raises(ValueError,match='verified model receipt'):
        collect.preflight(package='fixture',model='/models/Qwen2.5-7B-Instruct',gpus=[0],
                          epochs_root=tmp_path,member='m')


def test_native_actual_batch_rejects_gaps_wrong_batch_and_rpc_overlap():
    point=cal.make_plan(IDENTITY)['training'][0]
    evidence=dict(point=point,start_s=10.,end_s=15.,native_schedule=native_trace(point,10.,15.))
    observed=cal.native_batch_observation(evidence)
    assert observed['exact_decode_batch_fraction']==1
    changed=deepcopy(evidence);changed['native_schedule']['after']['gap']=True
    with pytest.raises(ValueError,match='overflow/gap'):cal.native_batch_observation(changed)
    changed=deepcopy(evidence);changed['native_schedule']['after']['events'].pop(2)
    with pytest.raises(ValueError,match='sequence incomplete'):cal.native_batch_observation(changed)
    changed=deepcopy(evidence);changed['native_schedule']['before_received_s']=10.01
    with pytest.raises(ValueError,match='overlapped'):cal.native_batch_observation(changed)
    changed=deepcopy(evidence)
    for row in changed['native_schedule']['after']['events']:row['schedule_queue']=row['schedule_queue'][:1]
    observed=cal.native_batch_observation(changed)
    assert not observed['passed'] and observed['exact_decode_batch_fraction']<.01
