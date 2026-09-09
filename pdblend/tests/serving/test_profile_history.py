from copy import deepcopy
from dataclasses import asdict,replace
import json

import pytest

from ecopadg.serving.profile_history import apply_prefill_history_bounds
from ecopadg.serving.evidence import sha256
from ecopadg.serving.profile_validation import apply_settled_residency,merge,validate_envelope
from ecopadg.serving.profiles import ProfilePoint,ProfileStore


def point(source='cold',**changes):
    value=ProfilePoint('mixed',1,1500,128,512,1,.5,.04,150,50,.1,2,source,
        prefill_power_w=100,decode_power_w=160,energy_error_fraction=.1)
    return replace(value,**changes)


def table(value,*,event_batches=None,powers=None,durations=None):
    batches=event_batches or [value.batch]
    powers=powers or [value.phase_power('prefill')]*len(batches)
    durations=durations or [.1]*len(batches)
    return dict(source_sha256=value.source_sha256,power_source=dict(mode='instant'),
        power_source_verified=True,points=[asdict(value)],phase_power_sources=[
            dict(role=value.role,tp=value.tp,frequency_mhz=value.frequency_mhz,
                input_tokens=value.input_tokens,batch=batch,phase='prefill',
                source='integrated_nvml_instant',power_evidence=dict(started_s=10+i,
                    finished_s=10+i+duration,integrated_power_w=power,sampling_supported=True,
                    fallback_reasons=[])) for i,(batch,power,duration) in
                        enumerate(zip(batches,powers,durations))])


def test_single_prefill_shares_measured_history_across_mixed_batch_context_and_p_role():
    cold=point()
    warm=point('warm',batch=8,context_tokens=4096,prefill_power_w=280,energy_error_fraction=.2)
    source=point('source',role='prefill',context_tokens=129,prefill_power_w=250,iteration_s=0)
    original=[cold,warm,source];tables=[table(p) for p in original];saved=deepcopy(tables)
    updated,audit=apply_prefill_history_bounds(original,tables)
    for before,after in zip(original,updated):
        assert after.prefill_power_upper_w==336
        assert after.phase_power_bound('prefill')==336
        assert after.phase_power('prefill')==before.phase_power('prefill')
        assert after.phase_power_bound('decode')==before.phase_power_bound('decode')
        assert after.batch_service_s(128)==before.batch_service_s(128)
        bounds={'prefill_power_upper_w','prefill_duration_upper_s'}
        assert {k:v for k,v in asdict(after).items() if k not in bounds}=={
            k:v for k,v in asdict(before).items() if k not in bounds}
    assert tables==saved and original==[cold,warm,source]
    assert not audit['nominal_estimates_changed'] and audit['prefill_bounds_changed']
    assert not audit['decode_bound_changed']
    assert len(audit['groups'])==1
    assert {r['source_sha256'] for r in audit['groups'][0]['contributors']}=={'cold','warm','source'}
    assert json.loads(json.dumps(audit))==audit


def test_history_groups_do_not_leak_across_tp_frequency_or_input():
    points=[point(),point('other-tp',tp=2,prefill_power_w=400,prefill_s=2),
        point('other-frequency',frequency_mhz=900,prefill_power_w=500,prefill_s=3),
        point('other-input',input_tokens=256,prefill_power_w=600,prefill_s=4)]
    updated,audit=apply_prefill_history_bounds(points,[table(p) for p in points])
    assert updated[0].prefill_power_upper_w==pytest.approx(110)
    assert [p.prefill_duration_upper_s for p in updated]==pytest.approx([.55,2.2,3.3,4.4])
    assert len(audit['groups'])==4
    assert {(x['tp'],x['frequency_mhz'],x['input_tokens']) for x in audit['groups']}=={
        (1,1500,128),(2,1500,128),(1,900,128),(1,1500,256)}


def test_real_batched_p_neither_contributes_nor_receives_single_prefill_bounds():
    mixed=point(prefill_power_w=200)
    batched=point('batched',role='prefill',batch=4,prefill_power_w=900,prefill_s=4,iteration_s=0)
    decoder=point('decoder',role='decode',prefill_power_w=1000,prefill_s=8)
    updated,audit=apply_prefill_history_bounds([mixed,batched,decoder],
        [table(mixed),table(batched),table(decoder)])
    assert updated[0].prefill_power_upper_w==pytest.approx(220)
    assert updated[0].prefill_duration_upper_s==pytest.approx(.55)
    assert updated[1]==batched and updated[2]==decoder
    assert [x['source_sha256'] for x in audit['groups'][0]['contributors']]==['cold']


@pytest.mark.parametrize('defect', ['legacy','unverified','no_rows','device_limit','missing_evidence',
    'unsupported','fallback_reason','missing_power','nonfinite_power','negative_power','invalid_duration'])
def test_any_unsupported_relevant_prefill_evidence_prevents_that_bucket_contributing(defect):
    cold=point();invalid=point('unusable',batch=8,prefill_power_w=1000,prefill_s=4)
    bad=table(invalid,event_batches=[8,8])
    row=bad['phase_power_sources'][1]
    if defect=='legacy':bad['power_source']['mode']='legacy_average'
    elif defect=='unverified':bad['power_source_verified']=False
    elif defect=='no_rows':bad['phase_power_sources']=[]
    elif defect=='device_limit':row['source']='enforced_device_limit_upper_bound'
    elif defect=='missing_evidence':row.pop('power_evidence')
    elif defect=='unsupported':row['power_evidence']['sampling_supported']=False
    elif defect=='fallback_reason':row['power_evidence']['fallback_reasons']=['insufficient_instant_sample_support']
    elif defect=='missing_power':row['power_evidence'].pop('integrated_power_w')
    elif defect=='nonfinite_power':row['power_evidence']['integrated_power_w']=float('nan')
    elif defect=='negative_power':row['power_evidence']['integrated_power_w']=-1
    else:row['power_evidence']['finished_s']=row['power_evidence']['started_s']
    updated,audit=apply_prefill_history_bounds([cold,invalid],[table(cold),bad])
    assert updated[0].prefill_power_upper_w==pytest.approx(110)
    assert updated[0].prefill_duration_upper_s==pytest.approx(.55)
    assert [x['source_sha256'] for x in audit['groups'][0]['contributors']]==['cold']
    # An ineligible measurement is not erased: its nominal power and its own
    # existing uncertainty still apply, even if a shared bound is received.
    assert updated[1].phase_power('prefill')==1000
    assert updated[1].phase_power_bound('prefill')==1100
    assert updated[1].phase_time_bound('prefill')==pytest.approx(4.4)


def test_p_batch_one_matches_events_labelled_with_held_target_decode_batch():
    mixed=point()
    producer=point('producer',role='prefill',batch=1,context_tokens=129,prefill_s=.8,
                   prefill_power_w=250,energy_error_fraction=.2)
    evidence=table(producer,event_batches=[2,6],powers=[260,300],durations=[.7,.85])
    updated,audit=apply_prefill_history_bounds([mixed,producer],[table(mixed),evidence])
    assert all(p.phase_power_bound('prefill')==300 for p in updated)
    assert [p.phase_time_bound('prefill') for p in updated]==pytest.approx([.88,.88])
    contribution=next(x for x in audit['groups'][0]['contributors'] if x['role']=='prefill')
    assert contribution['batch']==1 and contribution['events']==2
    assert contribution['maximum_observed_power_w']==300
    assert contribution['maximum_observed_duration_s']==pytest.approx(.85)


def test_mixed_evidence_cannot_borrow_a_different_target_batch_label():
    single=point();candidate=point('different-batch',batch=8,prefill_power_w=600)
    mismatched=table(candidate,event_batches=[2])
    updated,audit=apply_prefill_history_bounds([single,candidate],[table(single),mismatched])
    assert updated[0].phase_power_bound('prefill')==pytest.approx(110)
    assert [x['source_sha256'] for x in audit['groups'][0]['contributors']]==['cold']


def test_original_repeat_uncertainty_must_cover_own_events_even_with_prior_history_bound():
    value=point(prefill_power_upper_w=1000)
    evidence=table(value,powers=[120])  # original 100 W with 10% error covers only 110 W
    with pytest.raises(ValueError,match='own prefill power observations'):
        apply_prefill_history_bounds([value],[evidence])


def test_foreign_heldout_source_is_rejected_even_when_shape_matches_training():
    value=point();heldout=replace(value,source_sha256='independent-heldout')
    with pytest.raises(ValueError,match='must belong to training'):
        apply_prefill_history_bounds([value],[table(value),table(heldout)])


def test_merge_rejects_point_source_that_disagrees_with_its_measured_table(tmp_path):
    raw_path=tmp_path/'raw.json'
    raw_path.write_text(json.dumps(dict(complete=True,engine_provenance=[dict(
        image_id='image',model='model',engine_version='0.9.2',
        source_files_at_import={'/src/serving/engine.py':'engine-hash'})])))
    measured=table(point(source=sha256(raw_path)))
    measured.update(schema=2,measurement='hardware',frequency_commands_verified=True,
                    frequency_samples_source_sha256=sha256(raw_path))
    path=tmp_path/'training.json';path.write_text(json.dumps(measured))
    manifest=dict(raw=[str(raw_path)],tables=[str(path)],engine_image='image')
    assert merge(manifest)['points'][0]['source_sha256']==sha256(raw_path)
    # The table header still points to a valid frozen raw, but one point has
    # been relabelled as an independent source. Reject before history pooling.
    measured['points'][0]['source_sha256']='foreign-heldout-or-fabricated-source'
    path.write_text(json.dumps(measured))
    with pytest.raises(ValueError,match='point source differs from its measured table'):
        merge(manifest)


def test_repeated_application_is_idempotent_without_error_multiplication():
    points=[point(),point('hot',batch=8,prefill_power_w=300,energy_error_fraction=.2,
                         prefill_s=.8,error_fraction=.2)]
    tables=[table(p) for p in points]
    once,audit=apply_prefill_history_bounds(points,tables)
    twice,repeated_audit=apply_prefill_history_bounds(once,tables)
    assert once==twice and audit==repeated_audit
    assert all(p.prefill_power_upper_w==360 for p in twice)
    assert [p.prefill_duration_upper_s for p in twice]==pytest.approx([.96,.96])
    assert twice[0].energy_error_fraction==.1 and twice[1].energy_error_fraction==.2
    assert twice[0].error_fraction==.1 and twice[1].error_fraction==.2


def test_legacy_same_key_error_contamination_cannot_become_an_instant_history_contributor():
    receiver=point('receiver',energy_error_fraction=0)
    original=point('instant-p',role='prefill',context_tokens=129,
                   prefill_power_w=150,iteration_s=0,energy_error_fraction=.1)
    legacy=replace(original,source_sha256='legacy-p',prefill_power_w=350,power_w=350,
                   energy_error_fraction=0)
    legacy_table=table(legacy)
    legacy_table['power_source']['mode']='legacy_average'
    legacy_table['power_source_verified']=False
    legacy_table['phase_power_sources'][0]['source']='enforced_device_limit_upper_bound'
    # Simulate an upstream cross-table error field influenced by that legacy
    # point. Only the unmodified instantaneous source table may supply a
    # contributor's repeat-derived uncertainty, not this model's 600 W bound.
    contaminated=replace(original,energy_error_fraction=3,prefill_power_upper_w=700)
    updated,audit=apply_prefill_history_bounds([receiver,contaminated,legacy],
        [table(receiver),table(original,event_batches=[2,6],powers=[155,160]),legacy_table])
    assert updated[0].phase_power_bound('prefill')==165
    contributors=audit['groups'][0]['contributors']
    assert {x['source_sha256'] for x in contributors}=={'receiver','instant-p'}
    assert next(x for x in contributors if x['source_sha256']=='instant-p')['power_upper_w']==165
    assert audit['groups'][0]['prefill_power_upper_w']==165
    assert updated[1].prefill_power_upper_w==700  # local pre-existing bound is not rewritten
    assert updated[1].energy_error_fraction==3


def test_settled_model_floor_is_applied_without_using_changed_model_error():
    original=point(energy_error_fraction=.1)
    modeled=replace(original,residency_w=120,prefill_power_w=120,energy_error_fraction=2)
    updated,audit=apply_prefill_history_bounds([modeled],[table(original,powers=[105])])
    assert updated[0].prefill_power_upper_w==132
    assert audit['groups'][0]['contributors'][0]['original_phase_power_w']==100
    assert audit['groups'][0]['contributors'][0]['original_energy_error_fraction']==.1


def test_settled_floor_cannot_hide_original_repeat_evidence_that_exceeds_its_bound():
    original=point(energy_error_fraction=.1)
    modeled=replace(original,residency_w=200,power_w=200,prefill_power_w=200,decode_power_w=200)
    with pytest.raises(ValueError,match='own prefill power observations'):
        apply_prefill_history_bounds([modeled],[table(original,powers=[120])])


@pytest.mark.parametrize('role', ['mixed','prefill'])
def test_history_preserves_valid_below_reference_training_observations_while_model_is_floored(role):
    original=point(role=role,power_w=80,residency_w=100,prefill_power_w=80,decode_power_w=80,
                   energy_error_fraction=.1)
    evidence=table(original,powers=[85],durations=[.54])
    saved=deepcopy(evidence)
    modeled,_=apply_settled_residency([original],{(1,1500):dict(watts=100,source_sha256='settled')})
    updated,audit=apply_prefill_history_bounds(modeled,[evidence])
    assert updated[0].residency_w==105
    assert updated[0].phase_power('prefill')==105
    assert updated[0].prefill_power_upper_w==pytest.approx(115.5)
    assert updated[0].prefill_duration_upper_s==pytest.approx(.55)
    contributor=audit['groups'][0]['contributors'][0]
    assert contributor['original_phase_power_w']==80
    assert contributor['maximum_observed_power_w']==85
    assert contributor['original_energy_error_fraction']==.1
    assert original.phase_power('prefill')==80 and original.residency_w==100
    assert evidence==saved


@pytest.mark.parametrize('role', ['mixed','prefill'])
@pytest.mark.parametrize('defect', ['power','duration'])
def test_below_reference_original_repeat_failure_is_not_hidden_by_training_floor(role,defect):
    original=point(role=role,power_w=80,residency_w=100,prefill_power_w=80,decode_power_w=80,
                   prefill_power_upper_w=1000,prefill_duration_upper_s=10)
    evidence=table(original,powers=[90 if defect=='power' else 85],
                   durations=[.6 if defect=='duration' else .54])
    modeled,_=apply_settled_residency([original],{(1,1500):dict(watts=100,source_sha256='settled')})
    # Raw 80 W / .5 s with 10% repeat error cannot cover 90 W / .6 s.
    # Model floors and prior explicit bounds must not rescue that evidence.
    with pytest.raises(ValueError,match='own prefill '+defect+' observations'):
        apply_prefill_history_bounds(modeled,[evidence])


@pytest.mark.parametrize('field,value', [('power_w',-1),('prefill_power_w',float('nan')),
                                       ('decode_power_w',float('inf'))])
def test_model_validity_does_not_replace_original_history_observation_validation(field,value):
    modeled=point()
    invalid=replace(modeled,**{field:value})
    evidence=table(invalid,powers=[100])
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        apply_prefill_history_bounds([modeled],[evidence])


@pytest.mark.parametrize('field', ['prefill_power_upper_w','prefill_duration_upper_s'])
@pytest.mark.parametrize('value', [-1,float('nan'),float('inf'),float('-inf')])
def test_invalid_new_prefill_bound_is_rejected_before_history_pooling(field,value):
    invalid=point(**{field:value})
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        ProfileStore([invalid])
    with pytest.raises(ValueError,match='invalid or unmeasured'):
        apply_prefill_history_bounds([invalid],[table(invalid)])


def test_phase_power_bound_preserves_zero_phase_fallback_and_existing_decode_error():
    value=point(prefill_power_w=0,decode_power_w=0,power_w=150,prefill_power_upper_w=300)
    assert value.phase_power('prefill')==value.phase_power('decode')==150
    assert value.phase_power_bound('prefill')==300
    assert value.phase_power_bound('decode')==pytest.approx(165)


@pytest.mark.parametrize('phase', ['mixed','prefill_power_w','',None])
def test_unknown_execution_phase_cannot_silently_use_a_decode_bound(phase):
    with pytest.raises(ValueError,match='unknown execution phase'):
        point().phase_power_bound(phase)
    with pytest.raises(ValueError,match='unknown execution phase'):
        point().phase_time_bound(phase)


def test_envelope_uses_training_prefill_bound_without_relaxing_decode_or_latency():
    cold=point(prefill_s=1,iteration_s=.04,error_fraction=0,energy_error_fraction=0)
    hot=point('hot',batch=8,prefill_s=1,iteration_s=.04,prefill_power_w=300,
              error_fraction=0,energy_error_fraction=0)
    actual=replace(cold,input_tokens=64,context_tokens=128,prefill_s=.75,
                   prefill_power_w=250,source_sha256='heldout')
    assert not validate_envelope(ProfileStore([cold,hot]),[actual])['passed']
    updated,_=apply_prefill_history_bounds([cold,hot],[table(cold),table(hot)])
    result=validate_envelope(ProfileStore(updated),[actual])
    assert result['passed']
    assert result['observations'][0]['ratios']['prefill_energy_j']==.625
    expensive_decode=replace(actual,decode_power_w=200)
    failed=validate_envelope(ProfileStore(updated),[expensive_decode])
    ratios=failed['observations'][0]['ratios']
    assert not failed['passed'] and ratios['decode_energy_j']==1.25
    assert ratios['prefill_energy_j']==.625
    slow=replace(actual,prefill_s=1.1,prefill_power_w=100)
    assert not validate_envelope(ProfileStore(updated),[slow])['passed']
    # A heldout point's own optional bound cannot inflate its observation or
    # enter the training envelope; validation always reads observed power.
    assert validate_envelope(ProfileStore(updated),[replace(actual,prefill_power_upper_w=9999)])==result


def test_warm_prefill_time_is_shared_without_changing_nominal_values_or_decode_bounds():
    cold=point(prefill_s=.25,error_fraction=.1)
    warm=point('warm',batch=6,context_tokens=4096,prefill_s=.8,error_fraction=.2,
               iteration_s=.06,prefill_power_w=250)
    producer=point('producer',role='prefill',batch=1,context_tokens=129,
                   prefill_s=.3,error_fraction=.05,iteration_s=0)
    points=[cold,warm,producer]
    tables=[table(cold,durations=[.26]),table(warm,durations=[.9]),
            table(producer,event_batches=[2,6],durations=[.3,.31])]
    before=deepcopy(tables)
    updated,audit=apply_prefill_history_bounds(points,tables)
    assert [p.phase_time_bound('prefill') for p in updated]==pytest.approx([.96]*3)
    for original,modeled in zip(points,updated):
        assert modeled.prefill_s==original.prefill_s
        assert modeled.iteration_s==original.iteration_s
        assert modeled.error_fraction==original.error_fraction
        assert modeled.phase_time_bound('decode')==original.phase_time_bound('decode')
        assert modeled.phase_power_bound('decode')==original.phase_power_bound('decode')
    assert tables==before
    group=audit['groups'][0]
    assert group['prefill_duration_upper_s']==pytest.approx(.96)
    contributor=next(c for c in group['contributors'] if c['source_sha256']=='warm')
    assert contributor['original_prefill_s']==.8
    assert contributor['original_error_fraction']==.2
    assert contributor['maximum_observed_duration_s']==pytest.approx(.9)


@pytest.mark.parametrize('prior_explicit_bound', [0,10])
def test_original_duration_observations_cannot_be_hidden_by_later_model_error_or_bound(prior_explicit_bound):
    original=point(prefill_s=.5,error_fraction=.1,
                   prefill_duration_upper_s=prior_explicit_bound)
    modeled=replace(original,error_fraction=3,prefill_duration_upper_s=20)
    # The original bucket covers .55 s; neither an old explicit bound nor a
    # later model error may make the recorded .6 s prefill a valid contributor.
    with pytest.raises(ValueError,match='own prefill duration observations'):
        apply_prefill_history_bounds([modeled],[table(original,durations=[.6])])


def test_prior_explicit_duration_and_model_error_are_not_shared_back_to_other_histories():
    cold=point('cold',prefill_s=.25,error_fraction=0)
    original=point('warm',batch=6,context_tokens=4096,prefill_s=.5,error_fraction=.1,
                   prefill_duration_upper_s=5)
    modeled=replace(original,error_fraction=3,prefill_duration_upper_s=10)
    updated,audit=apply_prefill_history_bounds([cold,modeled],
        [table(cold),table(original,durations=[.54])])
    assert updated[0].phase_time_bound('prefill')==pytest.approx(.55)
    assert updated[1].phase_time_bound('prefill')==10
    assert updated[1].error_fraction==3
    assert updated[1].phase_time_bound('decode')==modeled.phase_time_bound('decode')
    assert audit['groups'][0]['prefill_duration_upper_s']==pytest.approx(.55)
    warm=next(x for x in audit['groups'][0]['contributors'] if x['source_sha256']=='warm')
    assert warm['duration_upper_s']==pytest.approx(.55)


def test_batch_service_uses_single_prefill_per_mixed_request_but_one_true_batched_p_kernel():
    mixed=point(batch=4,prefill_s=.25,iteration_s=.04,error_fraction=.1,
                prefill_duration_upper_s=.75)
    producer=replace(mixed,role='prefill',prefill_s=.8,iteration_s=0,
                     prefill_duration_upper_s=0)
    assert mixed.batch_service_s(3)==pytest.approx(4*.75+2*.04*1.1)
    assert mixed.batch_service_s(1)==pytest.approx(4*.75)
    # A true batched P profile already measures the whole batch, so multiplying
    # its measured prefill duration by four would count the same work twice.
    assert producer.batch_service_s(3)==pytest.approx(.8*1.1)
    assert producer.batch_service_s(1)==pytest.approx(.8*1.1)


def test_envelope_uses_training_duration_bound_but_ignores_heldout_optional_bounds():
    cold=point(prefill_s=.5,error_fraction=0,energy_error_fraction=0)
    warm=point('warm',batch=6,context_tokens=4096,prefill_s=.8,error_fraction=.1,
               energy_error_fraction=0)
    actual=replace(cold,input_tokens=64,context_tokens=128,prefill_s=.75,
                   source_sha256='heldout',prefill_duration_upper_s=100,
                   prefill_power_upper_w=10000)
    assert not validate_envelope(ProfileStore([cold,warm]),[actual])['passed']
    updated,_=apply_prefill_history_bounds([cold,warm],
        [table(cold,durations=[.5]),table(warm,durations=[.85])])
    store=ProfileStore(updated)
    result=validate_envelope(store,[actual])
    assert result['passed']
    ratios=result['observations'][0]['ratios']
    assert ratios['prefill_s']==pytest.approx(.75/.88)
    assert ratios['prefill_energy_j']==pytest.approx(.75/.88)
    assert ratios['mixed_bucket_prefill_s']==pytest.approx(.75/.88)
    assert validate_envelope(store,[replace(actual,prefill_duration_upper_s=0,
                                           prefill_power_upper_w=0)])==result
    # The decode query still uses its own original 40 ms iteration bound.
    failed=validate_envelope(store,[replace(actual,iteration_s=.05)])
    assert not failed['passed']
    assert failed['observations'][0]['ratios']['iteration_s']==pytest.approx(1.25)
    assert failed['observations'][0]['ratios']['prefill_s']==ratios['prefill_s']
    too_slow=validate_envelope(store,[replace(actual,prefill_s=.9)])
    assert not too_slow['passed']
    assert too_slow['observations'][0]['ratios']['prefill_s']==pytest.approx(.9/.88)
