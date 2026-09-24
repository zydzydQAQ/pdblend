import json
from pathlib import Path

import pytest

from pdblend.profile.collection import native_readiness as readiness
from pdblend.profile.collection.native_timing_plan import binding


def put(path, value):
    path.write_text(json.dumps(value))
    return binding(path)


def test_replays_runtime_instead_of_trusting_success_flags(tmp_path, monkeypatch):
    ref = put(tmp_path/'runtime.json', dict(complete=True, component_qualified=True,
        runtime_holdout_passed=True, runtime_plan={'holdout_limits': {'max_relative_error': .25}}))
    audit = dict(raw_components_complete=True, holdout_passed=False, errors=[],
        holdout_comparisons=[dict(component='transfer_7168', metric='second_output_overhead_s',
            training_prediction=-.005, relative_error=.3)])
    seen = []
    monkeypatch.setattr(readiness, 'replay_runtime', lambda report: seen.append(report) or audit)
    result = readiness.runtime_readiness(ref)
    assert seen and result['raw_components_complete']
    assert not result['independent_holdout_passed']
    assert not result['usable_as_full_runtime']
    assert result['nonpositive_training_nodes'][0]['training_prediction'] == -.005
    assert len(result['failed_max_error_nodes']) == 1


def test_stale_nested_runtime_binding_fails_before_replay(tmp_path, monkeypatch):
    raw = tmp_path/'raw.json'
    raw_ref = put(raw, {'old': True})
    ref = put(tmp_path/'runtime.json', {'power': raw_ref})
    raw.write_text('{}')
    monkeypatch.setattr(readiness, 'replay_runtime', lambda _: pytest.fail('must reject stale raw first'))
    with pytest.raises(ValueError, match='checksum'):
        readiness.runtime_readiness(ref)


def test_raw_complete_and_pilot_success_do_not_skip_timing():
    attempts = [dict(qualified_timing=False, timing_component_claimed=True,
        runtime=dict(raw_components_complete=True, completion={'path': 'raw', 'sha256': 'x'}),
        auxiliaries={'power-pilot': {'complete': True}})]
    decision = readiness.collection_reuse_decision(attempts)
    assert decision['collect_timing']
    assert len(decision['raw_runtime_to_retain']) == 1
    assert not decision['raw_runtime_qualifies_profile']
    assert not decision['power_pilot_qualifies_profile']


def test_qualified_replayed_timing_is_reused_without_recollection():
    assert not readiness.collection_reuse_decision([{'qualified_timing': True}])['collect_timing']


def test_terminal_inventory_keeps_failure_and_omits_lease_tokens():
    job = dict(job_id='pdblend-native-timing-7b-x', status='blocked', attempts=0,
        last_error='historical preference', lease_token='private',
        payload=dict(model_id='Qwen2.5-7B-Instruct'))
    a = readiness.terminal_jobs({'jobs': {job['job_id']: job}})
    b = readiness.terminal_jobs({'jobs': [job]})
    assert a == b
    assert a[0]['status'] == 'blocked'
    assert a[0]['last_error'] == 'historical preference'
    assert 'private' not in json.dumps(a)
