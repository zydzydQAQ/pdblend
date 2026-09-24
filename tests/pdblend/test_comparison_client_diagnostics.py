import gzip
import hashlib
import json

import pytest

from pdblend.bench import comparison_client_diagnostics as diagnostics


def outcome(idx, scheduled, submitted, finished):
    return dict(idx=idx, scheduled_s=scheduled, submitted_s=submitted, finished_s=finished)


def test_observable_delays_and_half_open_outstanding_intervals():
    rows = [outcome(0, 1., 1.1, 2.), outcome(1, 1., 1.2, 3.), outcome(2, 2., 2., 4.)]
    value = diagnostics.summarize(rows, expected_requests=3)
    assert value['client_schedule_delay_p50_s'] == pytest.approx(.1)
    assert value['client_schedule_delay_p99_s'] == pytest.approx(.2)
    assert value['client_schedule_delay_mean_s'] == pytest.approx(.1)
    assert value['client_peak_outstanding'] == 2
    assert value['client_timing_complete'] and value['client_timing_coverage_fraction'] == 1.


def test_missing_invalid_and_duplicate_timing_is_explicit():
    rows = [outcome(0, 1., 2., 3.), dict(idx=1), outcome(2, 2., 1., 3.), outcome(0, 1., 2., 3.)]
    value = diagnostics.summarize(rows, expected_requests=4)
    assert value['client_timing_missing_records'] == value['client_timing_invalid_records'] == 1
    assert value['client_timing_duplicate_records'] == 1
    assert value['client_schedule_delay_samples'] == 1 and not value['client_timing_complete']


def put(path, rows, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row)+'\n' for row in rows) if lines else json.dumps(rows)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt') as stream:
            stream.write(text)
    else:
        path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pd_own_outcomes_fallback_is_cached_by_artifact_hash_and_never_actual_send_queue(tmp_path, monkeypatch):
    path = tmp_path/'window/receipt.json'
    canonical = 'run/comparison-requests.json'; raw = 'run/outcomes.jsonl.gz'
    rows = [outcome(i, 1., 1., 2.) for i in range(2048)]
    receipt = dict(artifacts={canonical: put(path.parent/canonical, [dict(idx=i) for i in range(2048)]),
                             raw: put(path.parent/raw, rows, lines=True)})
    metrics = dict(offered_requests=2048, ttft_p99_s=5., slo_pass=False)
    before = dict(metrics)
    value = diagnostics.annotate(path, receipt, dict(system='pdblend', run_id='current'), metrics)
    assert value['client_timing_source'] == raw
    assert value['client_peak_outstanding'] == 2048
    assert value['client_bottleneck_status'] == 'potential_client_concurrency_limit'
    assert value['client_submitted_timestamp_semantics'] == 'before_client_concurrency_semaphore'
    assert value['client_send_queue_status'] == 'unknown_send_queue'
    assert metrics == before
    monkeypatch.setattr(diagnostics, '_records', lambda path: pytest.fail('artifact parsed twice'))
    assert diagnostics.annotate(path, receipt, dict(system='pdblend', run_id='current'), metrics) == value


def test_baseline_does_not_borrow_pd_fallback_and_unbound_outcome_is_never_read(tmp_path):
    path = tmp_path/'window/receipt.json'; canonical = 'run/comparison-requests.json'
    receipt = dict(artifacts={canonical: put(path.parent/canonical, [dict(idx=0)])})
    put(path.parent/'run/outcomes.jsonl', [outcome(0, 1., 2., 3.)], lines=True)
    for system in ('pdblend', 'dynamollm'):
        value = diagnostics.annotate(path, receipt, dict(system=system, run_id='current'), dict(offered_requests=1))
        assert value['client_bottleneck_status'] == 'unsupported_missing_timestamps'
        assert value['client_peak_outstanding'] is None
        assert value['potential_client_concurrency_limit'] is None


def test_canonical_timestamps_supported_without_claiming_baseline_concurrency_limit(tmp_path):
    path = tmp_path/'window/receipt.json'; name = 'run/comparison-requests.json'
    receipt = dict(artifacts={name: put(path.parent/name, [outcome(0, 1., 2., 3.)])})
    value = diagnostics.annotate(path, receipt, dict(system='mixed', run_id='current'), dict(offered_requests=1))
    assert value['client_timing_complete'] and value['client_peak_outstanding'] == 1
    assert value['client_configured_concurrency_limit'] is None
    assert value['potential_client_concurrency_limit'] is None
    assert value['client_send_queue_status'] == 'unknown_send_queue'


@pytest.mark.parametrize('run_id', [None, 'previous-round'])
def test_historical_rows_do_not_read_canonical_or_outcomes(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(diagnostics, '_records', lambda path: pytest.fail('historical artifact read'))
    receipt = dict(artifacts={'run/comparison-requests.json': 'a'*64, 'run/outcomes.jsonl': 'b'*64})
    value = diagnostics.annotate(tmp_path/'receipt.json', receipt,
        dict(system='pdblend', run_id=run_id), dict(offered_requests=1), active_run_id='current')
    assert value['client_bottleneck_status'] == 'not_recomputed_historical'
    assert value['client_timing_source_sha256'] == '' and value['client_peak_outstanding'] is None
