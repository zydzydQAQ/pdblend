from dataclasses import asdict,replace
import json

import pytest

from ecopadg.serving.evidence import sha256
from ecopadg.serving.profile_validation import apply_settled_residency,merge,validate_envelope,validate_heldout_matrix
from ecopadg.serving.profiles import ProfilePoint,ProfileStore,validate_profile_observations


def test_heldout_validation_keeps_a_failed_envelope_failed():
    point=ProfilePoint('mixed',1,1500,2048,2560,8,.5,.04,200,60,.1,3,'train',
                       prefill_power_w=250,decode_power_w=200)
    store=ProfileStore([point])
    actual=replace(point,input_tokens=1024,context_tokens=1152,batch=2,
                   prefill_s=.3,iteration_s=.05,source_sha256='independent')
    result=validate_envelope(store,[actual])
    assert not result['passed'] and result['observations'][0]['ratios']['iteration_s']>1
    assert store.points[0]==point
    assert validate_envelope(store,[replace(actual,iteration_s=.03)])['passed']
    assert not validate_envelope(store,[replace(actual,tp=2)])['passed']


def test_prefill_validation_does_not_compare_an_unused_decode_phase():
    point=ProfilePoint('prefill',1,2520,2048,2049,2,.5,0,250,60,.1,3,'train',prefill_power_w=250)
    actual=replace(point,input_tokens=1024,context_tokens=1025,prefill_s=.3,decode_power_w=60,
                   source_sha256='independent')
    assert validate_envelope(ProfileStore([point]),[actual])['passed']


def test_resident_park_power_is_distinct_from_unallocated_gpu_power():
    point=ProfilePoint('mixed',1,1500,2048,2560,8,.5,.04,200,80,.1,3,'measured')
    store=ProfileStore([point],idle_unallocated_gpu_w=35)
    assert store.parked_residency(1)==80
    measured=ProfileStore([point],idle_unallocated_gpu_w=35,parked_residency_w_by_tp={1:55})
    assert measured.parked_residency(1)==55


def test_short_unresolved_phase_validates_energy_not_device_limit_watts():
    point=ProfilePoint('prefill',1,900,2048,2049,1,1.,0,160,60,.1,3,'train',prefill_power_w=160)
    short=replace(point,input_tokens=64,context_tokens=65,prefill_s=.05,
                  prefill_power_w=350,power_w=350,source_sha256='heldout')
    row=validate_envelope(ProfileStore([point]),[short])['observations'][0]
    assert row['passed'] and row['power_ratios_diagnostic']['prefill']>2
    assert row['ratios']['prefill_incremental_energy_j']<1
    # Faster execution alone cannot excuse an underestimated phase energy.
    expensive=replace(short,prefill_s=.9)
    result=validate_envelope(ProfileStore([point]),[expensive])
    assert not result['passed']
    assert result['observations'][0]['ratios']['prefill_s']<1
    assert result['observations'][0]['ratios']['prefill_energy_j']>1


def test_validation_uses_online_single_prefill_query_not_decode_batch_bucket():
    single=ProfilePoint('mixed',1,900,128,256,1,.01,.01,100,50,0,1,'train')
    large=replace(single,batch=8,prefill_s=1.,power_w=200)
    actual=replace(single,input_tokens=64,context_tokens=192,batch=2,prefill_s=.05,source_sha256='heldout')
    row=validate_envelope(ProfileStore([single,large]),[actual])['observations'][0]
    assert not row['passed'] and row['ratios']['prefill_s']==5
    assert row['queries']['prefill']==dict(context_tokens=65,batch=1)


def test_prefill_context_does_not_include_future_output_and_zero_phase_fails():
    single=ProfilePoint('prefill',1,900,128,129,1,.01,0,100,50,0,1,'train')
    large=replace(single,context_tokens=256,prefill_s=1.)
    actual=replace(single,input_tokens=64,context_tokens=192,prefill_s=.05,source_sha256='heldout')
    result=validate_envelope(ProfileStore([single,large]),[actual])
    assert not result['passed'] and result['observations'][0]['ratios']['prefill_s']==5
    assert not validate_envelope(ProfileStore([single]),[replace(actual,prefill_s=0)])['passed']


def test_mixed_capacity_bucket_also_needs_a_valid_prefill_bound():
    single=ProfilePoint('mixed',1,900,128,256,1,1.,.01,100,50,0,1,'train')
    batch=replace(single,batch=8,prefill_s=.001)
    actual=replace(single,input_tokens=64,context_tokens=192,batch=2,prefill_s=.5,source_sha256='heldout')
    row=validate_envelope(ProfileStore([single,batch]),[actual])['observations'][0]
    assert row['ratios']['prefill_s']<1
    assert row['ratios']['mixed_bucket_prefill_s']>1 and not row['passed']


def test_short_prelude_residency_cannot_hide_incremental_underprediction():
    point=ProfilePoint('prefill',1,900,128,129,1,1.,0,100,80,0,1,'train')
    actual=replace(point,input_tokens=64,context_tokens=65,prefill_s=.5,
                   power_w=150,residency_w=140,source_sha256='heldout')
    row=validate_envelope(ProfileStore([point]),[actual])['observations'][0]
    assert row['ratios']['prefill_energy_j']==.75
    assert row['ratios']['prefill_incremental_energy_j']==1.75 and not row['passed']


def settled(watts=80,source='settled-raw'):
    return {(1,1500):dict(watts=watts,source_sha256=source)}


def training(**changes):
    point=ProfilePoint('mixed',1,1500,128,256,1,.1,.04,100,60,.05,2,'train',
        prefill_power_w=67.731,decode_power_w=110,energy_error_fraction=.05)
    return replace(point,**changes)


def test_training_floor_preserves_observations_and_audits_conservative_residency():
    point=training();original=asdict(point);evidence=settled(67.20633098189671)
    saved=json.dumps(evidence[(1,1500)],sort_keys=True)
    updated,audit=apply_settled_residency([point],evidence)
    revised=updated[0];floor=67.20633098189671*1.05
    assert revised.prefill_power_w==revised.residency_w==floor
    assert revised.power_w==100 and revised.decode_power_w==110
    assert asdict(point)==original and json.dumps(evidence[(1,1500)],sort_keys=True)==saved
    assert revised.error_fraction==point.error_fraction
    assert revised.energy_error_fraction==point.energy_error_fraction
    assert revised.prefill_s==point.prefill_s and revised.iteration_s==point.iteration_s
    assert audit[0]['source_sha256']=='train'
    assert audit[0]['settled_evidence']==evidence[(1,1500)]
    assert audit[0]['original']['prefill_power_w']==67.731
    assert audit[0]['modeled_effective_phase_w']['prefill']==floor
    assert audit[0]['residency_multiplier']==1.05 and not audit[0]['measurement_changed']
    assert json.loads(json.dumps(audit))==audit


@pytest.mark.parametrize('role', ['mixed','prefill','decode'])
@pytest.mark.parametrize('watts', [50,200])
def test_zero_active_phase_retains_whole_power_fallback_and_unused_phase_is_idle(role,watts):
    point=training(role=role,prefill_power_w=0,decode_power_w=0)
    updated,audit=apply_settled_residency([point],settled(watts))
    revised=updated[0];floor=watts*1.05
    assert revised.power_w==max(point.power_w,floor)
    for phase in ('prefill','decode'):
        if role=='mixed' or role==phase:
            assert getattr(revised,phase+'_power_w')==0
            assert revised.phase_power(phase)==max(point.phase_power(phase),floor)
        else:
            assert getattr(revised,phase+'_power_w')==floor
    assert audit[0]['original_effective_phase_w']==dict(prefill=100,decode=100)


@pytest.mark.parametrize('role', ['mixed','prefill','decode'])
def test_settled_floor_never_lowers_whole_or_active_phase_power(role):
    point=training(role=role,power_w=150,prefill_power_w=200,decode_power_w=300)
    updated,_=apply_settled_residency([point],settled(180))
    revised=updated[0]
    assert revised.power_w>=point.power_w
    for phase in ('prefill','decode'):
        if role=='mixed' or role==phase:
            assert revised.phase_power(phase)==max(point.phase_power(phase),189)
        else:
            assert getattr(revised,phase+'_power_w')==189


@pytest.mark.parametrize('changes', [
    dict(power_w=-1),dict(power_w=float('nan')),dict(power_w=float('inf')),
    dict(prefill_power_w=-1),dict(prefill_power_w=float('nan')),
    dict(decode_power_w=float('inf')),dict(error_fraction=-1),dict(source_sha256='')])
def test_floor_rejects_original_invalid_observations_before_maximum(changes):
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        apply_settled_residency([training(**changes)],settled(200))


@pytest.mark.parametrize('evidence', [settled(-1),settled(float('nan')),
                                     settled(float('inf')),settled(source='')])
def test_floor_rejects_invalid_settled_measurement(evidence):
    with pytest.raises(ValueError,match='invalid measured settled'):
        apply_settled_residency([training()],evidence)


def test_unmeasured_residency_does_not_change_a_point():
    point=training()
    assert apply_settled_residency([point],{})==([point],[])


@pytest.mark.parametrize('role', ['mixed','prefill','decode'])
def test_observed_power_below_idle_max_is_valid_but_requires_a_model_floor(role):
    original=training(role=role,power_w=50,prefill_power_w=50,decode_power_w=50,
                      residency_w=100)
    saved=asdict(original)
    validate_profile_observations([original])
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        ProfileStore([original])
    modeled,audit=apply_settled_residency([original],settled(100))
    assert modeled[0].power_w==modeled[0].residency_w==105
    assert modeled[0].prefill_power_w==modeled[0].decode_power_w==105
    assert asdict(original)==saved
    assert audit[0]['original']['power_w']==50
    assert audit[0]['original']['residency_w']==100
    assert not audit[0]['measurement_changed']
    assert ProfileStore(modeled).points==tuple(modeled)


@pytest.mark.parametrize('role', ['mixed','prefill','decode'])
@pytest.mark.parametrize('field,value', [
    ('power_w',-1),('prefill_power_w',-1),('decode_power_w',-1),
    ('residency_w',-1),('power_w',float('nan')),('power_w',float('inf')),
    ('prefill_power_w',float('nan')),('decode_power_w',float('inf'))])
def test_invalid_heldout_values_cannot_disappear_in_incremental_energy_clamping(role,field,value):
    trained=training(role=role,power_w=150,prefill_power_w=150,decode_power_w=150)
    actual=replace(trained,source_sha256='heldout',**{field:value})
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        validate_envelope(ProfileStore([trained]),[actual])


def test_model_floor_does_not_turn_real_incremental_underprediction_into_success():
    point=training(role='prefill',prefill_s=1,iteration_s=0,power_w=100,
                   residency_w=80,prefill_power_w=0,decode_power_w=0,
                   error_fraction=0,energy_error_fraction=0)
    actual=replace(point,input_tokens=64,context_tokens=65,prefill_s=.5,
                   power_w=150,residency_w=140,source_sha256='heldout')
    updated,_=apply_settled_residency([point],settled(80))
    result=validate_envelope(ProfileStore(updated),[actual])
    row=result['observations'][0]
    assert not result['passed']
    assert row['ratios']['prefill_energy_j']==.75
    assert row['ratios']['prefill_incremental_energy_j']==2.0625
    assert actual.residency_w==140 and actual.power_w==150


def test_merge_floors_training_only_with_raw_provenance_and_keeps_heldout_failure(tmp_path):
    def save(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return str(path)
    provenance=[dict(image_id='image',engine_version='0.9.2',model='model',
                     source_files_at_import={'/src/serving/engine.py':'engine-sha'})]
    train_raw=save('train.raw.json',dict(complete=True,engine_provenance=provenance,partition='train'))
    held_raw=save('held.raw.json',dict(complete=True,engine_provenance=provenance,partition='heldout'))
    resident=save('resident.raw.json',dict(complete=True,engine_provenance=provenance,wakeup=dict(passed=True),
        residency=[dict(tp=1,frequency_mhz=1500,parked=False,watts=67.20633098189671),
                   dict(tp=1,frequency_mhz=1500,parked=True,watts=50)]))
    point=training(source_sha256=sha256(train_raw))
    actual=replace(point,input_tokens=64,context_tokens=192,prefill_power_w=300,power_w=300,
                   source_sha256=sha256(held_raw))
    def table(name,raw,point):
        return save(name,dict(schema=2,measurement='hardware',source_sha256=sha256(raw),
            frequency_samples_source_sha256=sha256(raw),frequency_commands_verified=True,points=[asdict(point)]))
    train_table=table('train.json',train_raw,point);held_table=table('held.json',held_raw,actual)
    manifest=dict(raw=[train_raw,held_raw,resident],tables=[train_table],heldout_tables=[held_table],
                  residency=[resident],engine_image='image')
    originals={p:sha256(p) for p in manifest['raw']+[train_table,held_table]}
    result=merge(manifest)
    assert result['status']=='development_only' and not result['heldout_calibration_complete']
    row=result['residency_model_adjustments'][0]
    assert row['settled_evidence']['source_sha256']==sha256(resident)
    assert row['settled_evidence']['source_path']==resident
    assert row['settled_evidence']['sample_index']==0
    assert result['points'][0]['prefill_power_w']==67.20633098189671*1.05
    assert result['heldout_validation']['observations'][0]['ratios']['prefill_energy_j']>1
    assert originals=={p:sha256(p) for p in originals}
    # Invalid original training must fail even though the floor would hide it.
    broken=json.loads((tmp_path/'train.json').read_text())
    broken['points'][0]['prefill_power_w']=-1
    (tmp_path/'train.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        merge(manifest)


@pytest.mark.parametrize('role', ['mixed','prefill','decode'])
def test_merge_reconciles_low_training_power_without_rewriting_low_heldout_observations(tmp_path,role):
    def save(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return str(path)
    provenance=[dict(image_id='image',engine_version='0.9.2',model='model',
                     source_files_at_import={'/src/serving/engine.py':'engine-sha'})]
    train_raw=save('train.raw.json',dict(complete=True,engine_provenance=provenance,partition='train'))
    held_raw=save('held.raw.json',dict(complete=True,engine_provenance=provenance,partition='heldout'))
    resident=save('resident.raw.json',dict(complete=True,engine_provenance=provenance,wakeup=dict(passed=True),
        residency=[dict(tp=1,frequency_mhz=1500,parked=False,watts=100),
                   dict(tp=1,frequency_mhz=None,parked=True,watts=90)]))
    original=training(role=role,source_sha256=sha256(train_raw),power_w=80,residency_w=100,
        prefill_power_w=100 if role=='decode' else 80,decode_power_w=100 if role=='prefill' else 80,
        prefill_s=0 if role=='decode' else .1,iteration_s=0 if role=='prefill' else .04,
        context_tokens=129 if role=='prefill' else 256,error_fraction=0,energy_error_fraction=0)
    held=replace(original,source_sha256=sha256(held_raw),input_tokens=64,
        context_tokens=65 if role=='prefill' else 192,power_w=65,
        prefill_power_w=100 if role=='decode' else 65,decode_power_w=100 if role=='prefill' else 65,
        prefill_s=0 if role=='decode' else .05,iteration_s=0 if role=='prefill' else .02)
    def measured_table(raw,value):
        return dict(schema=2,measurement='hardware',source_sha256=sha256(raw),
            frequency_samples_source_sha256=sha256(raw),frequency_commands_verified=True,
            power_source={'mode':'instant'},power_source_verified=True,points=[asdict(value)],
            phase_power_sources=[] if role=='decode' else [dict(role=role,tp=1,frequency_mhz=1500,
                input_tokens=value.input_tokens,batch=1,phase='prefill',source='integrated_nvml_instant',
                power_evidence=dict(started_s=10,finished_s=10+value.prefill_s,
                    integrated_power_w=value.prefill_power_w,sampling_supported=True,fallback_reasons=[]))])
    train_table=save('train.json',measured_table(train_raw,original))
    held_table=save('held.json',measured_table(held_raw,held))
    manifest=dict(raw=[train_raw,held_raw,resident],tables=[train_table],heldout_tables=[held_table],
                  residency=[resident],engine_image='image')
    originals={path:sha256(path) for path in manifest['raw']+[train_table,held_table]}
    result=merge(manifest)
    modeled=ProfilePoint(**result['points'][0])
    assert modeled.power_w==modeled.residency_w==105
    assert modeled.prefill_power_w==modeled.decode_power_w==105
    assert result['heldout_validation']['passed']
    ratios=result['heldout_validation']['observations'][0]['ratios']
    for phase in (('prefill','decode') if role=='mixed' else (role,)):
        # Observed energy is 65 W times half the training duration; using a
        # 105 W floor on the heldout value would incorrectly produce .5 here.
        assert ratios[phase+'_energy_j']==pytest.approx(65*.5/105)
        assert phase+'_incremental_energy_j' not in ratios  # zero above-reference activity
    assert originals=={path:sha256(path) for path in originals}
    bad=measured_table(held_raw,replace(held,power_w=-1))
    save('held.json',bad)
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        merge(manifest)
    # Source agreement remains mandatory even for numerically valid low power.
    bad=measured_table(held_raw,replace(held,source_sha256='foreign-source'))
    save('held.json',bad)
    with pytest.raises(ValueError,match='point source differs'):
        merge(manifest)


def test_missing_shape_cannot_pass_by_retaining_role_tp_frequency_coverage():
    point=ProfilePoint('mixed',1,1500,128,256,2,.1,.04,200,60,.1,1,'independent')
    large=replace(point,input_tokens=4096,context_tokens=4224,batch=6)
    fields=('role','tp','frequency_mhz','input_tokens','context_tokens','batch')
    expected=[{field:getattr(p,field) for field in fields} for p in (point,large)]
    assert validate_heldout_matrix([point,large],expected)['passed']
    omitted=validate_heldout_matrix([point],expected)
    assert not omitted['passed'] and omitted['expected_observations']==2
    assert omitted['missing']==[dict(expected[1],count=1)]
    # A duplicate easy point cannot stand in for the missing difficult shape.
    duplicate=validate_heldout_matrix([point,point],expected)
    assert duplicate['actual_observations']==duplicate['expected_observations']
    assert not duplicate['passed'] and duplicate['unexpected']==[dict(expected[0],count=1)]


def test_shape_matrix_includes_context_and_declared_repetition_counts():
    point=ProfilePoint('decode',2,900,128,256,6,0,.04,400,120,.1,1,'heldout')
    fields=('role','tp','frequency_mhz','input_tokens','context_tokens','batch')
    expected={field:getattr(point,field) for field in fields}
    assert not validate_heldout_matrix([replace(point,context_tokens=255)],[expected])['passed']
    assert validate_heldout_matrix([point,point],[expected,expected])['passed']
    assert not validate_heldout_matrix([point],[expected,expected])['passed']


def test_merge_cannot_certify_an_incomplete_declared_shape_matrix(tmp_path,monkeypatch):
    def save(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return str(path)
    provenance=[dict(image_id='image',model='model',engine_version='0.9.2',
                     source_files_at_import={'/src/serving/engine.py':'engine'})]
    train_raw=save('train-raw.json',dict(complete=True,engine_provenance=provenance,partition='train'))
    held_raw=save('held-raw.json',dict(complete=True,engine_provenance=provenance,partition='held'))
    resident=save('idle-raw.json',dict(complete=True,engine_provenance=provenance,wakeup={'passed':True},
        residency=[dict(tp=1,frequency_mhz=1500,parked=False,watts=60),
                   dict(tp=1,frequency_mhz=None,parked=True,watts=50)]))
    point=ProfilePoint('mixed',1,1500,2048,2176,1,.2,.1,200,60,.1,1,sha256(train_raw))
    held=[replace(point,input_tokens=n,context_tokens=n+128,prefill_s=.01,iteration_s=.02,
                  source_sha256=sha256(held_raw)) for n in (128,1024)]
    def table(raw,values):
        return dict(schema=2,measurement='hardware',source_sha256=sha256(raw),
            frequency_samples_source_sha256=sha256(raw),frequency_commands_verified=True,
            power_source={'mode':'instant'},power_source_verified=True,points=[asdict(p) for p in values])
    train_table=save('train.json',table(train_raw,[point]));held_table=save('held.json',table(held_raw,held))
    fields=('role','tp','frequency_mhz','input_tokens','context_tokens','batch')
    manifest=dict(raw=[train_raw,held_raw,resident],tables=[train_table],heldout_tables=[held_table],
        residency=[resident],engine_image='image',require_instant_calibration=True,
        require_complete_heldout_matrix=True,
        expected_heldout_points=[{field:getattr(p,field) for field in fields} for p in held])
    assert merge(manifest)['status']=='validated_envelope'
    # Role-only profiles cannot supply an empty-card baseline. The explicit
    # prepared reference replaces even a larger value in an older table.
    from ecopadg.serving import profile_reference
    legacy=table(train_raw,[point]);legacy['idle_unallocated_gpu_w']=900
    save('train.json',legacy)
    manifest['require_unallocated_residency_reference']=True
    with pytest.raises(ValueError,match='empty-GPU residency evidence'):
        merge(manifest)
    manifest['unallocated_residency_reference']={'raw':'empty-raw','preparation':'prepared'}
    def independent(spec):
        assert spec==manifest['unallocated_residency_reference']
        return 35.5,{'free_gpus':[2,3]}, {'empty-proof':'hash'}
    monkeypatch.setattr(profile_reference,'measured_unallocated_reference',independent)
    corrected=merge(manifest)
    assert corrected['status']=='validated_envelope' and corrected['idle_unallocated_gpu_w']==35.5
    assert corrected['certification_artifacts']['empty-proof']=='hash'
    save('held.json',table(held_raw,held[:1]))
    result=merge(manifest)
    assert result['heldout_validation']['missing_role_tp_frequency_groups']==[]
    assert result['status']=='development_only' and not result['heldout_calibration_complete']
    assert result['heldout_validation']['matrix']['missing'][0]['input_tokens']==1024
    manifest.pop('expected_heldout_points')
    with pytest.raises(ValueError,match='explicit complete held-out matrix'):
        merge(manifest)
