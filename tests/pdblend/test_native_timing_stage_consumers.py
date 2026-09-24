"""Only the explicit new terminal kind reaches its strict stage replayer."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pdblend.profile.collection import native_timing_stage as stage
from pdblend.profile.query import native_composition as composition
from pdblend.profile.query import native_layout_profile as layout
from pdblend.profile.query.native_timing import attach_native_timing
from pdblend.profile.query.versions import LoadedVersion, VersionError
from test_native_timing_stage import staged, collected_v2, collected, plans
from test_native_timing_replay import put
from test_native_layout_stage import resident


def selection_identity(x):
    return {k: x.v2_fitted['component']['identity'][k] for k in composition.IDENTITY}


def base_profile(identity):
    base = SimpleNamespace(system=identity['system'], model=identity['model_id'], tp=identity['tp'], pp=identity['pp'],
                           query_qualification={})
    return LoadedVersion(base, deepcopy(identity), {}, {'usage': 'development'}, {})


def test_development_bridge_consumes_real_terminal_raw_but_keeps_all_formal_gates(staged):
    x = staged; ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    loaded = base_profile(selection_identity(x)); result = attach_native_timing(loaded, ref)
    assert result.model.prefill_seconds(128, 1500) > 0
    assert result.model.native_timing_replay['parent_job_status'] == 'failed'
    assert result.qualification['native_timing_component_qualified']
    assert not result.qualification['formal_eligible'] and not result.qualification['energy_comparable']
    assert not result.model.decode_supported(10000, 10000, 1500)
    formal = base_profile(selection_identity(x)); formal.qualification['usage'] = 'formal'
    with pytest.raises(VersionError, match='development-only'): attach_native_timing(formal, ref)
    changed = base_profile(dict(selection_identity(x), model_hash='different'))
    with pytest.raises(VersionError, match='model_hash'): attach_native_timing(changed, ref)


def test_full_composition_new_timing_kind_does_not_supply_missing_runtime_or_energy(staged, monkeypatch):
    x = staged; ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    source = x.v2_fitted['component']['identity']['source_revision']
    # This test isolates consumer dispatch. Actual terminal source, raw,
    # interference, holdout, phase and cleanup still replay without mocking.
    monkeypatch.setattr(composition, 'replay_sources', lambda *a: {'calibration_source_revisions': [source]})
    value = dict(kind=composition.KIND, identity=selection_identity(x), timing=ref)
    audit, model = composition.audit_native_profile(value)
    assert audit['gates']['timing_raw_and_holdout']
    assert not audit['gates']['runtime_raw_and_holdout'] and not audit['gates']['power_training_and_holdout']
    assert model is None and not audit['formal_eligible'] and not audit['full_profile_qualified']
    audit, _ = composition.audit_native_profile(dict(value, identity=dict(value['identity'], tokenizer_hash='other')))
    assert not audit['gates']['timing_raw_and_holdout']
    monkeypatch.setattr(composition, 'replay_sources', lambda *a: {'calibration_source_revisions': ['other-source']})
    audit, _ = composition.audit_native_profile(value)
    assert not audit['gates']['timing_raw_and_holdout'] and not audit['formal_eligible']


def test_layout_consumer_checks_new_terminal_lifetime_before_scoped_runtime(staged, monkeypatch):
    x = staged; ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    source = x.v2_fitted['component']['identity']['source_revision']; calls = []
    monkeypatch.setattr(layout, 'replay_sources', lambda *a: {'calibration_source_revisions': [source]})
    def runtime(*args):
        calls.append('runtime')
        return dict(capacity_tokens=16384, values={key: .01 for key in layout.RUNTIME_REQUIRED}, audit={},
                    scoped_runtime_qualified=True, full_runtime_profile_qualified=False)
    monkeypatch.setattr(layout, 'replay_layout_runtime_component', runtime)
    selected = dict(kind=layout.TIMING_KIND, identity=selection_identity(x), timing=ref, runtime={})
    model, audit = layout.load_layout_timing(put(x.tmp/'selection.json', selected))
    assert model.prefill_seconds(128, 1500) > 0 and calls == ['runtime']
    assert audit['physical_cleanup_verified'] and audit['queue_terminal_verified'] and not audit['resident_stage_only']
    assert not audit['formal_eligible'] and not audit['runtime_scope']['full_runtime_profile_qualified']
    # A changed terminal receipt cannot fall back to the live/legacy replayer.
    final = x.root/'completion.json'; saved = final.read_bytes(); final.write_bytes(saved+b' ')
    try:
        with pytest.raises(ValueError, match='checksum'): layout.load_layout_timing(put(x.tmp/'bad-selection.json', selected))
        assert calls == ['runtime']
    finally: final.write_bytes(saved)
    wrong_model = dict(selected, identity=dict(selected['identity'], model_id='Qwen2.5-7B-Instruct', tp=1))
    with pytest.raises(ValueError, match='model-owned'):
        layout.load_layout_timing(put(x.tmp/'wrong-model.json', wrong_model))


def test_original_resident_layout_stage_finalization_explicitly_accepts_new_terminal_kind(staged):
    from pdblend.profile.collection import native_layout_stage as resident_stage
    from pdblend.profile.collection.native_timing_plan import binding
    x = staged; report, specs, fleet = resident(x)
    live_ref = resident_stage.capture_resident_timing(report, input_manifest_ref=x.inputs_ref,
        attempt_manifest_ref=binding(x.attempt/'manifest.json'), specs=specs, fleet=fleet,
        out=x.root/'layout-resident-stage.json')
    terminal_ref = stage.capture_terminal_evidence(x.attempt, x.queue, x.tmp/'terminal.json')
    result = resident_stage.verify_final_timing(live_ref, terminal_ref)
    assert result['supported_fit'] == x.v2_fitted and result['parent_job_status'] == 'failed'
    assert result['physical_cleanup_verified'] and not result['formal_eligible']
