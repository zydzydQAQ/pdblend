"""Isolated read-only review fixtures; never access experiment artifacts."""
import importlib.util
import gzip
import hashlib
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2]/'scripts/2026-09-25_extract_failed_distserve_native.py'
    spec = importlib.util.spec_from_file_location('failed_native_review', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def point():
    return {'trace': {'sha256': 'trace'}, 'duration_s': 150.,
            'slo': {'ttft_s': 5., 'tpot_s': .15}}


def outcome(i, arrival=0, delay=.001, ttft=.1, n=10):
    return dict(request_id=f'distserve-701-{i}', arrival_s=arrival,
                scheduled_s=1000+arrival, submitted_s=1000+arrival+delay,
                finished_s=1000+arrival+ttft+1, ttft_s=ttft, tpot_s=.1,
                completion_tokens=n, output_tokens=n, ok=True, terminal=True,
                native_receipts_complete=True)


def native(rows):
    return dict(trace_sha256='trace', service_started_s=1000., outcomes=rows,
                status='failed', complete=False, cleanup_errors=['drain 500'])


def test_native_success_does_not_erase_outer_cleanup_failure():
    got = module().summarize(native([outcome(0)]), point=point(), expected_count=1)
    assert got['native_request_slo_pass'] is True
    assert got['native_complete'] is False
    assert got['canonical_energy_service_j'] is None
    assert got['canonical_modified'] is got['used_for_ranking'] is False
    assert got['native_cleanup_errors'] == ['drain 500']


def test_failed_and_missing_outcomes_keep_full_denominator():
    failed = outcome(1); failed.update(ok=False, completion_tokens=0, ttft_s=None,
                                      tpot_s=None, error='timeout')
    got = module().summarize(native([outcome(0), failed]), point=point(), expected_count=3)
    assert got['counts']['native_successful_requests'] == 1
    assert got['counts']['native_failed_or_missing_requests'] == 2
    assert got['counts']['native_joint_slo_rate'] == pytest.approx(1/3)
    assert got['missing_outcome_indices'] == [2]
    assert got['native_request_slo_pass'] is False


def test_late_request_goodput_and_pre_dispatch_not_actual_send():
    a = outcome(0, arrival=149.5); b = outcome(1, arrival=.5, delay=.2)
    got = module().summarize(native([a, b]), point=point(), expected_count=2)
    assert got['native_goodput_finished_window_request_s_lower_bound'] == 1/150
    assert got['native_cohort_goodput_request_s'] == pytest.approx(2/150.6)
    assert got['client_pre_dispatch_delay']['p99_s'] == pytest.approx(.2)
    assert got['client_peak_pre_dispatch_outstanding'] == 1
    assert got['client_send_queue_status'] == 'unknown_actual_send_and_connector_queue'


def test_outstanding_end_is_half_open():
    a = outcome(0); a['finished_s'] = 1001.1
    b = outcome(1, arrival=1.099, delay=.001)
    got = module().summarize(native([a, b]), point=point(), expected_count=2)
    assert got['client_peak_pre_dispatch_outstanding'] == 1


@pytest.mark.parametrize('kind', ['trace', 'schedule', 'duplicate', 'range'])
def test_identity_or_event_mismatch_rejected(kind):
    rows = [outcome(0)]
    obj = native(rows)
    if kind == 'trace': obj['trace_sha256'] = 'other'
    if kind == 'schedule': rows[0]['scheduled_s'] += 1
    if kind == 'duplicate': rows.append(outcome(0))
    if kind == 'range': rows[0]['request_id'] = 'distserve-701-4'
    with pytest.raises(ValueError):
        module().summarize(obj, point=point(), expected_count=1)


def test_phase_requires_exclusive_new_lease_and_absent_service_dirs(tmp_path):
    m = module(); queue = tmp_path/'queue.json'; attempt = tmp_path/'attempt'
    lease = dict(job_id='wanted', status='active', attempt_dir=str(attempt), claimed_at=10)
    data = {'leases': {'one': lease}}
    queue.write_text(json.dumps(data)); assert m.phase(queue, 'wanted')['ready']
    (attempt/'session/windows/point').mkdir(parents=True)
    (attempt/'session/windows/point/point.json').write_text('{}')
    assert not m.phase(queue, 'wanted')['ready']
    (attempt/'session/windows/point/point.json').unlink()
    data['leases']['two'] = dict(lease, job_id='other')
    queue.write_text(json.dumps(data)); assert not m.phase(queue, 'wanted')['ready']


def test_sha_mismatch_rejected(tmp_path):
    p = tmp_path/'native.json'; p.write_text('{}')
    with pytest.raises(ValueError): module().read_bound(p, 'wrong')


def power_module(monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/2026-09-25_recover_failed_distserve_service_power.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('failed_service_power', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def power_fixture():
    rows = [dict(_journal_schema='pdblend-journal-v1', kind='samples', values=[1, [2]*8]),
            dict(_journal_schema='pdblend-journal-v1', kind='power_metadata', source_epoch=0,
                 values={'read_finished_s': [1]*8})]
    raw = gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())
    manifest = dict(schema='pdblend-power-v1', counts={'samples': 1, 'power_metadata': 1},
                    raw_sha256=hashlib.sha256(raw).hexdigest(),
                    power_source_epochs=[{'gpus': list(range(8)), 'mode': ['instant']*8}])
    return manifest, raw


def test_power_decode_preserves_original_epochs(monkeypatch):
    manifest, raw = power_fixture()
    value = power_module(monkeypatch).decode_snapshot(manifest, raw)
    assert value['samples'] == [[1, [2]*8]]
    assert value['power_metadata'][0]['gpus'] == list(range(8))
    assert value['power_metadata'][0]['read_finished_s'] == [1]*8


@pytest.mark.parametrize('problem', ['hash', 'count', 'epoch'])
def test_power_decode_rejects_bad_raw_bindings(monkeypatch, problem):
    manifest, raw = power_fixture()
    if problem == 'hash': manifest['raw_sha256'] = 'bad'
    if problem == 'count': manifest['counts']['samples'] = 2
    if problem == 'epoch': manifest['power_source_epochs'] = []
    with pytest.raises(ValueError): power_module(monkeypatch).decode_snapshot(manifest, raw)


def mixed_module(monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/2026-09-25_extract_mixed_client_timing.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('mixed_client_review', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_mixed_stream_counts_native_failure_without_actual_send_claim(monkeypatch):
    rows = [dict(idx=0, arrival_s=0., scheduled_s=1000., submitted_s=1000.1,
                 finished_s=1002., correct=False, error='timeout'),
            dict(idx=1, arrival_s=.5, scheduled_s=1000.5, submitted_s=1000.6,
                 finished_s=1002., correct=True)]
    raw = gzip.compress(('\n'.join(json.dumps(dict(_journal_schema='pdblend-journal-v1', **r)) for r in rows)+'\n').encode())
    m = mixed_module(monkeypatch)
    v = m.reduce_timing(m.rows_from_bytes(raw), origin=1000., duration=150., offered=2)
    assert v['observed_outcomes'] == 2
    assert v['native_correct_requests'] == 1
    assert v['peak_pre_dispatch_outstanding'] == 2
    assert v['scheduled_to_pre_dispatch_delay']['p99_s'] == pytest.approx(.1)
    assert v['client_concurrency_limit'] is None
    assert v['actual_send_queue_status'] == 'unknown_actual_send_and_connector_queue'


def test_mixed_missing_cohort_rejected(monkeypatch):
    with pytest.raises(ValueError):
        mixed_module(monkeypatch).reduce_timing([], origin=1000., duration=150., offered=1)


def test_mixed_missing_finished_time_keeps_peak_unknown(monkeypatch):
    row = dict(idx=0, arrival_s=0., scheduled_s=1000., submitted_s=1000.1)
    v = mixed_module(monkeypatch).reduce_timing([row], origin=1000., duration=150., offered=1)
    assert v['peak_pre_dispatch_outstanding'] is None
    assert v['missing_client_timing_indices'] == [0]
