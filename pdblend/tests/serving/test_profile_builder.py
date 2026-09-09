import json
import pytest
from ecopadg.serving.evidence import sha256
from ecopadg.serving.profile_builder import build, instant_phase_coverage
from ecopadg.serving.profiles import ProfilePoint,ProfileStore


def engine_provenance():
    return [dict(instance_id='m',image_id='sha256:'+'a'*64,
        model='/models/Qwen2.5-14B-Instruct',engine_version='0.9.2',
        source_files_at_import={'/opt/pdblend/src/ecopadg/serving/engine.py':'b'*64})]


def raw_mixed(tmp_path):
    runtime=tmp_path/'runtime';runtime.mkdir()
    events=[dict(instance='m',prefill=1,decode=0,tokens=8,request_ids=[str(i)],
                 started_s=2+i,finished_s=2.1+i) for i in range(2)]
    events.extend(dict(instance='m',prefill=0,decode=2,tokens=2,request_ids=['0','1'],
        started_s=4+i*.1,finished_s=4.05+i*.1) for i in range(3))
    raw=dict(schema=3,complete=True,sampling_error=None,runtime_dir=str(runtime),
        engine_provenance=engine_provenance(),
        topology={'mixed':dict(id='m',tp=1,gpus=[0])},power_limit_w=[350]*8,
        power_samples=[(i*.01,[100 if i<=100 else 200]*8) for i in range(501)],
        frequency_samples=[(i*.01,[2505]*8) for i in range(501)],
        runs=[dict(skipped=False,input_tokens=8,batch=2,frequency_mhz=2520,output_tokens=4,
            idle_start_s=0,idle_end_s=1,layout='mixed',events=events,commanded_frequencies={'0':2520})])
    path=tmp_path/'raw.json';path.write_text(json.dumps(raw))
    return path,raw


def as_instant(raw):
    # Positive epoch timestamps are required by the actual provenance check.
    raw['power_samples']=[(t+100,ws) for t,ws in raw['power_samples']]
    raw['frequency_samples']=[(t+100,ws) for t,ws in raw['frequency_samples']]
    for run in raw['runs']:
        for key in ('idle_start_s','idle_end_s'):run[key]+=100
        for event in run['events']:
            for key in ('started_s','finished_s'):event[key]+=100
    source=dict(mode='instant',source_id='nvml:field:186:scope:0:mW',field_id=186,scope_id=0,unit='W')
    raw['power_source']=source
    raw['power_metadata']=[dict(t_s=t,gpus=list(range(8)),mode=['instant']*8,
        source_id=[source['source_id']]*8,field_id=[186]*8,scope_id=[0]*8,value_type=[1]*8,
        return_code=[0]*8,nvml_timestamp_us=[round(t*1e6)]*8,nvml_latency_us=[10]*8,
        read_started_s=[t]*8,read_finished_s=[t]*8) for t,_ in raw['power_samples']]
    return raw


def test_phase_power_and_actual_late_context_spacing_include_adapter_gaps(tmp_path):
    path,_=raw_mixed(tmp_path)
    profiles,transfers=build(path)
    p=profiles['points'][0]
    assert p['batch']==2 and p['context_tokens']==12
    assert p['prefill_s']==pytest.approx(.1)
    assert p['iteration_s']==pytest.approx(.1)  # not the .05-second kernel span
    assert p['prefill_power_w']==350 and p['decode_power_w']==200
    assert not profiles['heldout_calibration_complete']
    assert not profiles['mixed_interference_measured']
    assert transfers==[]


def test_partial_actual_batch_cannot_be_relabelled_as_full(tmp_path):
    path,raw=raw_mixed(tmp_path)
    raw['runs'][0]['events'][-1]['decode']=1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='observed'):
        build(path)


def test_safety_recovery_to_full_frequency_cannot_be_labelled_as_medium(tmp_path):
    path,raw=raw_mixed(tmp_path)
    raw['runs'][0]['frequency_mhz']=1500
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='differs from the clock command'):
        build(path)


def test_prefill_context_excludes_target_decode_work(tmp_path):
    path,raw=raw_mixed(tmp_path)
    runtime=tmp_path/'runtime'
    raw['topology']={'prefill':dict(id='p',tp=1,gpus=[0]),
                     'decode':dict(id='d',tp=1,gpus=[1])}
    run=raw['runs'][0]
    run['layout']='pd'
    run['commanded_frequencies']['1']=2520
    prefills=run['events'][:2]
    run['events']=[dict(e,instance='p') for e in prefills]+[
        dict(e,instance='d') for e in run['events']]
    for engine in ('p','d'):
        transfers=[dict(engine_id=engine,request_ids=e['request_ids'],
            started_s=e['started_s']+.05,finished_s=e['finished_s']) for e in prefills]
        (runtime/(engine+'.control.json.kv.0.jsonl')).write_text(
            '\n'.join(json.dumps(e) for e in transfers)+'\n')
    path.write_text(json.dumps(raw))
    profiles,_=build(path)
    p=next(p for p in profiles['points'] if p['role']=='prefill')
    d=next(p for p in profiles['points'] if p['role']=='decode')
    assert p['context_tokens']==9 and p['batch']==1
    assert d['context_tokens']==12 and d['batch']==2


@pytest.mark.parametrize('tp',[1,8])
@pytest.mark.parametrize('duration',[.18,.21])
def test_supported_instant_phases_do_not_jump_to_device_limit_at_legacy_duration(tmp_path,tp,duration):
    path,value=raw_mixed(tmp_path);value=as_instant(value)
    value['topology']['mixed'].update(tp=tp,gpus=list(range(tp)))
    value['runs'][0]['commanded_frequencies']={str(g):2520 for g in range(tp)}
    for event in value['runs'][0]['events'][:2]:event['finished_s']=event['started_s']+duration
    path.write_text(json.dumps(value));profiles,_=build(path)
    point=profiles['points'][0]
    assert profiles['power_source_verified']
    assert point['prefill_power_w']==pytest.approx(200*tp)
    evidence=[p for p in profiles['phase_power_sources'] if p['phase']=='prefill']
    assert all(p['source']=='integrated_nvml_instant' for p in evidence)
    # Equal power values are legitimate time-stamped readings, not proof of
    # either a stale field or independent sensor updates.
    assert all(p['power_evidence']['sampling_supported'] for p in evidence)
    assert all(len(p['power_evidence']['per_gpu'])==tp for p in evidence)


@pytest.mark.parametrize('kind',['too_short','sparse_frames','repeated_timestamps'])
def test_instant_without_sufficient_time_support_keeps_explicit_limit(tmp_path,kind):
    path,value=raw_mixed(tmp_path);value=as_instant(value)
    if kind=='too_short':
        for event in value['runs'][0]['events'][:2]:event['finished_s']=event['started_s']+.005
    elif kind=='sparse_frames':
        value['power_samples']=value['power_samples'][::20]
        value['power_metadata']=value['power_metadata'][::20]
    else:
        for row in value['power_metadata']:
            row['nvml_timestamp_us']=[int(row['t_s']*5)/5*1e6]*8
            row['nvml_timestamp_us']=[round(t) for t in row['nvml_timestamp_us']]
    path.write_text(json.dumps(value));profiles,_=build(path)
    assert profiles['power_source_verified']
    assert profiles['points'][0]['prefill_power_w']==350
    evidence=[p for p in profiles['phase_power_sources'] if p['phase']=='prefill']
    assert all(p['source']=='enforced_device_limit_upper_bound' for p in evidence)
    assert all('insufficient_instant_sample_support' in p['power_evidence']['fallback_reasons'] for p in evidence)


def test_supported_instant_power_below_residency_stays_conservative(tmp_path):
    path,value=raw_mixed(tmp_path);value=as_instant(value)
    value['power_samples']=[(t,[100 if t<=101 else 50]*8) for t,_ in value['power_samples']]
    path.write_text(json.dumps(value));profiles,_=build(path)
    assert profiles['points'][0]['prefill_power_w']==350
    evidence=[p['power_evidence'] for p in profiles['phase_power_sources'] if p['phase']=='prefill']
    assert all(p['sampling_supported'] for p in evidence)
    assert all(p['fallback_reasons']==['integrated_power_below_residency'] for p in evidence)


def test_instant_phase_coverage_requires_distributed_timestamps_on_every_gpu():
    assert instant_phase_coverage({0:[6,8]},[0],5,9)['sampling_supported']
    assert not instant_phase_coverage({0:[5.1,5.2,5.3]},[0],5,9)['sampling_supported']
    assert not instant_phase_coverage({0:[6,8],1:[6,6,6]},[0,1],5,9)['sampling_supported']
    assert not instant_phase_coverage({0:[5,9]},[0],5,9)['sampling_supported']


def test_instant_label_without_per_sample_provenance_is_rejected(tmp_path):
    path,value=raw_mixed(tmp_path);value=as_instant(value);value['power_metadata'].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='per-sample power metadata'):build(path)


def settled_raw(tmp_path,name='settled.json',watts=100):
    raw=dict(complete=True,sampling_error=None,wakeup=dict(passed=True),
        engine_provenance=engine_provenance(),residency=[
            dict(instance_id='settled-m',tp=1,frequency_mhz=2520,parked=True,watts=900),
            dict(instance_id='settled-m',tp=1,frequency_mhz=2520,parked=False,watts=watts)])
    path=tmp_path/name;path.write_text(json.dumps(raw));return path,raw


def dirty_local_prelude(tmp_path,*,instant=True,phase_w=200):
    path,raw=raw_mixed(tmp_path)
    raw['power_samples']=[(t,[300 if t<=1 else phase_w]*8) for t,_ in raw['power_samples']]
    if instant:raw=as_instant(raw)
    path.write_text(json.dumps(raw));return path,raw


def test_independent_settled_reference_replaces_dirty_local_idle_without_rewriting_raw(tmp_path):
    path,raw=dirty_local_prelude(tmp_path)
    low,_=settled_raw(tmp_path,'settled-low.json',95)
    high,_=settled_raw(tmp_path,'settled-high.json',100)
    originals={p:p.read_bytes() for p in (path,low,high)}
    profiles,_=build(path,residency_paths=(low,high))
    assert profiles['power_source_verified']
    point=profiles['points'][0]
    assert point['residency_w']==100  # no merger's 1.05 model margin here
    assert point['prefill_power_w']==pytest.approx(200)
    assert point['decode_power_w']==pytest.approx(200)
    assert point['power_w']==pytest.approx(200)
    reference=profiles['residency_reference']
    assert reference['method']=='maximum independently measured settled power; no model margin'
    assert reference['artifacts']=={str(low.resolve()):sha256(low),str(high.resolve()):sha256(high)}
    selected=next(p for p in reference['points'] if p['tp']==1 and p['frequency_mhz']==2520)
    assert selected['watts']==100
    assert selected['source_sha256']==sha256(high)
    assert selected['source_path']==str(high.resolve()) and selected['sample_index']==1
    for phase in profiles['phase_power_sources']:
        evidence=phase['power_evidence']
        assert phase['source']=='integrated_nvml_instant'
        assert evidence['local_prelude_power_w']==pytest.approx(300)
        assert evidence['residency_w']==100
        assert evidence['residency_reference']==selected
        assert evidence['fallback_reasons']==[]
    assert originals=={p:p.read_bytes() for p in originals}
    assert profiles['source_sha256']==sha256(path)


@pytest.mark.parametrize('phase_w', [80,200])
def test_independent_reference_cannot_turn_unsupported_window_into_measured_power(tmp_path,phase_w):
    path,raw=dirty_local_prelude(tmp_path,phase_w=phase_w)
    reference,_=settled_raw(tmp_path)
    for event in raw['runs'][0]['events'][:2]:event['finished_s']=event['started_s']+.005
    path.write_text(json.dumps(raw))
    profiles,_=build(path,residency_paths=(reference,))
    assert profiles['points'][0]['residency_w']==100
    assert profiles['points'][0]['prefill_power_w']==350
    reason='insufficient_instant_sample_support'
    phases=[p for p in profiles['phase_power_sources'] if p['phase']=='prefill']
    assert all(p['source']=='enforced_device_limit_upper_bound' for p in phases)
    assert all(reason in p['power_evidence']['fallback_reasons'] for p in phases)


@pytest.mark.parametrize('layout', ['mixed','pd'])
def test_supported_instant_below_independent_max_retains_observation_and_signed_difference(tmp_path,layout):
    path,raw=dirty_local_prelude(tmp_path,phase_w=80)
    reference,_=settled_raw(tmp_path,watts=100)
    if layout=='pd':
        raw['topology']={'prefill':dict(id='p',tp=1,gpus=[0]),
                         'decode':dict(id='d',tp=1,gpus=[1])}
        raw['engine_provenance']=[dict(engine_provenance()[0],instance_id=i) for i in ('p','d')]
        run=raw['runs'][0];run['layout']='pd';run['commanded_frequencies']['1']=2520
        prefills=run['events'][:2]
        run['events']=[dict(e,instance='p') for e in prefills]+[
            dict(e,instance='d') for e in run['events']]
        for engine in ('p','d'):
            transfers=[dict(engine_id=engine,request_ids=e['request_ids'],
                started_s=e['started_s']+.05,finished_s=e['finished_s']) for e in prefills]
            (tmp_path/'runtime'/(engine+'.control.json.kv.0.jsonl')).write_text(
                '\n'.join(json.dumps(e) for e in transfers)+'\n')
        path.write_text(json.dumps(raw))
    originals={p:p.read_bytes() for p in (path,reference)}
    profiles,_=build(path,residency_paths=(reference,))
    assert profiles['power_source_verified']
    assert {p['role'] for p in profiles['points']}==({'mixed'} if layout=='mixed' else {'prefill','decode'})
    for p in profiles['points']:
        assert p['residency_w']==100
        assert p['power_w']==pytest.approx(80)
        for phase in ('prefill','decode'):
            expected=80 if p['role']=='mixed' or p['role']==phase else 100
            assert p[phase+'_power_w']==pytest.approx(expected)
        # Raw observation tables are not already-reconciled online models.
        with pytest.raises(ValueError,match='invalid or unmeasured'):
            ProfileStore([ProfilePoint(**p)])
    for phase in profiles['phase_power_sources']:
        evidence=phase['power_evidence']
        assert phase['source']=='integrated_nvml_instant'
        assert evidence['sampling_supported'] and evidence['fallback_reasons']==[]
        assert evidence['integrated_power_w']==pytest.approx(80)
        assert evidence['local_prelude_power_w']==pytest.approx(300)
        assert evidence['residency_w']==100 and evidence['below_reference'] is True
        assert evidence['signed_power_above_reference_w']==pytest.approx(-20)
    assert originals=={p:p.read_bytes() for p in originals}


@pytest.mark.parametrize('watts', [-1,float('nan'),float('inf')])
def test_independent_reference_never_masks_invalid_operator_power(tmp_path,watts):
    path,_=dirty_local_prelude(tmp_path,phase_w=watts)
    reference,_=settled_raw(tmp_path)
    with pytest.raises(ValueError,match='invalid power samples'):
        build(path,residency_paths=(reference,))


@pytest.mark.parametrize('defect', ['missing_metadata','wrong_field','wrong_source','stale_timestamp'])
def test_below_reference_observation_still_requires_exact_instant_provenance(tmp_path,defect):
    path,raw=dirty_local_prelude(tmp_path,phase_w=80)
    reference,_=settled_raw(tmp_path)
    if defect=='missing_metadata':raw['power_metadata'].pop()
    elif defect=='wrong_field':raw['power_metadata'][205]['field_id'][0]=185
    elif defect=='wrong_source':raw['power_source']['source_id']='nvml:average'
    else:raw['power_metadata'][205]['nvml_timestamp_us'][0]-=1_000_000
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='per-sample power metadata'):
        build(path,residency_paths=(reference,))


def test_no_reference_preserves_legacy_local_idle_behavior(tmp_path):
    path,_=dirty_local_prelude(tmp_path)
    profiles,_=build(path)
    assert profiles['points'][0]['residency_w']==pytest.approx(300)
    assert profiles['points'][0]['prefill_power_w']==350


@pytest.mark.parametrize('phase_w', [80,200])
def test_settled_reference_does_not_promote_short_average_phase_to_instant(tmp_path,phase_w):
    path,_=dirty_local_prelude(tmp_path,instant=False,phase_w=phase_w)
    reference,_=settled_raw(tmp_path)
    profiles,_=build(path,residency_paths=(reference,))
    assert profiles['power_source']['mode']=='legacy_average'
    assert not profiles['power_source_verified']
    assert profiles['points'][0]['residency_w']==100
    assert profiles['points'][0]['prefill_power_w']==350
    phases=[p for p in profiles['phase_power_sources'] if p['phase']=='prefill']
    assert all(p['source']=='enforced_device_limit_upper_bound' for p in phases)
    assert all('legacy_average_short_phase' in p['power_evidence']['fallback_reasons'] for p in phases)
    if phase_w<100:
        assert all('integrated_power_below_residency' in p['power_evidence']['fallback_reasons'] for p in phases)


@pytest.mark.parametrize('missing', ['tp','frequency','nonparked'])
def test_explicit_reference_requires_every_used_group(tmp_path,missing):
    path,_=dirty_local_prelude(tmp_path);reference,raw=settled_raw(tmp_path)
    if missing=='tp':raw['residency'][1]['tp']=2
    elif missing=='frequency':raw['residency'][1]['frequency_mhz']=1500
    else:raw['residency'][1]['parked']=True
    reference.write_text(json.dumps(raw))
    with pytest.raises(ValueError):build(path,residency_paths=(reference,))


@pytest.mark.parametrize('field,value', [
    ('image_id','sha256:'+'c'*64),('model','/models/another-model'),
    ('engine_version','0.10.0'),('source_files_at_import',{'/other/serving/engine.py':'d'*64})])
def test_independent_reference_requires_matching_engine_identity(tmp_path,field,value):
    path,_=dirty_local_prelude(tmp_path);reference,raw=settled_raw(tmp_path)
    raw['engine_provenance'][0][field]=value;reference.write_text(json.dumps(raw))
    with pytest.raises(ValueError):build(path,residency_paths=(reference,))


@pytest.mark.parametrize('operator', [False,True])
def test_independent_reference_rejects_missing_engine_provenance(tmp_path,operator):
    path,work=dirty_local_prelude(tmp_path);reference,resident=settled_raw(tmp_path)
    target,raw=(path,work) if operator else (reference,resident)
    raw.pop('engine_provenance');target.write_text(json.dumps(raw))
    with pytest.raises(ValueError):build(path,residency_paths=(reference,))


@pytest.mark.parametrize('change', [dict(complete=False),dict(sampling_error='NVML failure'),
    dict(wakeup={}),dict(wakeup=dict(passed=False))])
def test_independent_reference_requires_complete_wakeup_evidence(tmp_path,change):
    path,_=dirty_local_prelude(tmp_path);reference,raw=settled_raw(tmp_path)
    raw.update(change);reference.write_text(json.dumps(raw))
    with pytest.raises(ValueError):build(path,residency_paths=(reference,))


@pytest.mark.parametrize('field,value', [
    ('watts',float('nan')),('watts',float('inf')),('watts',-1),
    ('tp',3),('tp',0),('frequency_mhz',0),('frequency_mhz',-900)])
def test_independent_reference_rejects_invalid_measurements(tmp_path,field,value):
    path,_=dirty_local_prelude(tmp_path);reference,raw=settled_raw(tmp_path)
    raw['residency'][1][field]=value;reference.write_text(json.dumps(raw))
    with pytest.raises(ValueError):build(path,residency_paths=(reference,))
