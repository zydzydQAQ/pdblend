import asyncio
import json
import time

import pytest

from pdblend.bench.client import LoadClient, Request
from pdblend.bench.comparison_metrics import (
    canonical_outcomes, event_token_count, reduce_comparison, safe_ratio,
)


def outcome(idx, first, last, terminal, *, n=2, **extra):
    return dict(idx=idx, first_token_s=first, last_token_s=last, terminal_s=terminal,
        finished_s=terminal + 10, terminal=True, completion_tokens=n,
        token_events=[dict(received_s=first, count=1), dict(received_s=last, count=n-1)],
        token_events_complete=True, **extra)


def test_window_goodput_excludes_tail_but_throughput_includes_delivered_partial_tokens():
    trace = [Request(0, 0, [1], 2), Request(1, 149, [1], 2), Request(2, 20, [1], 3)]
    rows = [outcome(0, 1000.2, 1000.4, 1000.41),
            outcome(1, 1149.1, 1150.5, 1150.6),
            dict(idx=2, error='timeout', terminal=False, finished_s=1030,
                 completion_tokens=1, token_events_complete=True,
                 token_events=[dict(received_s=1020.1, count=1)])]
    result = reduce_comparison(trace, rows, service_started_s=1000, slo=(2, 2))
    assert result['offered_requests'] == 3
    assert result['successful_requests'] == 2 and result['joint_slo_requests'] == 2
    assert result['window_good_requests'] == 1
    assert result['goodput_request_s'] == pytest.approx(1 / 150)
    assert result['goodput_token_s'] == pytest.approx(2 / 150)
    assert result['window_delivered_tokens'] == 4 and result['tail_delivered_tokens'] == 1
    assert result['throughput_token_s'] == pytest.approx(4 / 150)
    assert result['cohort_goodput_request_s'] == pytest.approx(2 / 150.6)
    assert result['request_tail_s'] == pytest.approx(.6)
    assert result['pending_at_window_end_requests'] == 1
    assert result['tail_completed_requests'] == 1
    assert result['slo_pass'] is False


def test_latency_uses_scheduled_arrival_and_last_token_not_submission_or_http_close():
    trace = [Request(0, 0, [1], 2)]
    row = outcome(0, 105, 105.1, 105.2, submitted_s=104.9)
    result = reduce_comparison(trace, [row], service_started_s=100, slo=(1, .2))
    assert result['ttft_mean_s'] == 5
    assert result['tpot_mean_s'] == pytest.approx(.1)
    assert result['successful_requests'] == 1
    assert result['joint_slo_requests'] == 0 and result['slo_pass'] is False


def test_exact_service_end_is_in_window_but_later_terminal_is_not():
    trace = [Request(0, 149, [1], 2), Request(1, 149, [1], 2)]
    rows = [outcome(0, 249.1, 249.8, 250), outcome(1, 249.1, 249.8, 250.001)]
    result = reduce_comparison(trace, rows, service_started_s=100, slo=(1, 1))
    assert result['window_good_requests'] == 1
    assert result['window_delivered_tokens'] == 4
    assert result['cohort_good_requests'] == 2


def test_nearest_rank_percentiles_and_low_sample_flags():
    trace = [Request(i, 0, [1], 2) for i in range(57)]
    rows = [outcome(i, 100+i, 100+i+.1, 100+i+.2) for i in range(57)]
    result = reduce_comparison(trace, rows, service_started_s=100, slo=(100, 1))
    assert result['ttft_p50_s'] == 28
    assert result['ttft_p90_s'] == 51
    assert result['ttft_p95_s'] == 54
    assert result['ttft_p99_s'] == result['ttft_max_s'] == 56
    assert result['ttft_samples'] == 57 and result['ttft_p99_low_sample'] is True


def test_missing_and_truncated_outcomes_do_not_disappear_from_denominator():
    trace = [Request(i, 0, [1], 3) for i in range(2)]
    rows = [outcome(0, 100.1, 100.2, 100.3, n=2)]
    result = reduce_comparison(trace, rows, service_started_s=100, observed_until_s=300)
    assert result['offered_requests'] == result['failed_requests'] == 2
    assert result['success_rate'] == 0 and result['joint_slo_rate'] == 0
    assert result['cohort_goodput_request_s'] == 0
    assert result['throughput_token_s'] is None
    assert result['ttft_p99_s'] is None
    assert safe_ratio(100, result['good_output_tokens']) is None
    unknown = reduce_comparison(trace, rows, service_started_s=100)
    assert unknown['cohort_goodput_request_s'] is None
    assert unknown['request_elapsed_s'] is None


def test_empty_cohort_and_missing_or_unproven_token_times_stay_explicit():
    result = reduce_comparison([], [], service_started_s=0)
    assert result['success_rate'] is None and result['joint_slo_rate'] is None
    assert result['slo_pass'] is False
    row = outcome(0, 1, 2, 3)
    row['token_events'][0]['exact'] = False
    result = reduce_comparison([Request(0, 0, [1], 2)], [row], service_started_s=0, slo=(3, 3))
    assert result['throughput_token_s'] is None
    assert result['token_timing_complete'] is False


@pytest.mark.parametrize('system,kind', [
    ('mixed', 'mixed_client_sse'), ('distserve', 'distserve_client_sse'),
    ('ecoserve', 'eco_client_sse'), ('dynamollm', 'dynamo_sse'), ('pdblend', 'pdblend_client_sse'),
])
def test_native_adapters_use_client_stream_and_not_engine_time_or_duplicate_native_stream(system, kind):
    trace = [Request(0, .5, [1], 2)]
    rid = 'r0' if system == 'pdblend' else system+'-701-0'
    original = dict(request_id=rid, completion_tokens=2, ok=True, finished_s=999)
    events = [dict(event=kind, request_id=rid, at_s=101,
                   payload=dict(token_ids=[1], at_s=0, finished=False)),
              dict(event=kind, request_id=rid, at_s=101.1,
                   payload=dict(token_ids=[2], at_s=0, finished=True)),
              dict(event='distserve_native_sse', request_id=rid, at_s=100.1,
                   payload=dict(token_ids=[1, 2], finished=True))]
    rows = canonical_outcomes(system, trace, [original], service_started_s=100, journal=events)
    result = reduce_comparison(trace, rows, service_started_s=100, slo=(1, .2))
    assert result['ttft_p99_s'] == .5
    assert result['tpot_p99_s'] == pytest.approx(.1)
    assert result['window_delivered_tokens'] == 2
    assert result['request_tail_s'] == 0  # HTTP/context cleanup at 999 is not request tail.
    assert result['slo_pass'] is True


def test_client_receipt_timestamp_beats_journal_write_timestamp():
    trace = [Request(0, 0, [1], 2)]
    events = [dict(kind='mixed_client_sse', request_id='mixed-701-0', at_s=999,
                   payload=dict(received_s=101 + i, token_ids=[i], finished=i == 1)) for i in range(2)]
    rows = canonical_outcomes('mixed', trace, [dict(idx=0, correct=True, completion_tokens=2)],
                              service_started_s=100, journal=events)
    assert rows[0]['first_token_s'] == 101 and rows[0]['last_token_s'] == 102
    assert rows[0]['terminal_s'] == 102


def test_token_count_requires_authoritative_metadata_and_never_retokenizes_text():
    assert event_token_count(dict(choices=[dict(text='several words')]))[:2] == (1, False)
    assert event_token_count(dict(choices=[dict(logprobs=dict(tokens=['a', 'b']))]))[:2] == (2, True)
    assert event_token_count(dict(pdblend_generated_tokens=1))[:2] == (1, True)
    with pytest.raises(ValueError, match='regressed'):
        event_token_count(dict(token_index=1), 2)


def test_frozen_identity_and_event_monotonicity_are_checked():
    trace = [Request(0, 0, [1], 2)]
    row = outcome(0, 101, 102, 103, scheduled_s=99)
    with pytest.raises(ValueError, match='scheduled clock'):
        reduce_comparison(trace, [row], service_started_s=100)
    row.pop('scheduled_s')
    with pytest.raises(ValueError, match='duplicate'):
        reduce_comparison(trace, [row, row], service_started_s=100)
    row['token_events'].reverse()
    with pytest.raises(ValueError, match='delivery event'):
        reduce_comparison(trace, [row], service_started_s=100)


def test_public_client_diagnostics_records_actual_token_and_terminal_times(monkeypatch):
    def frame(tokens, **extra):
        value = dict(choices=[dict(index=0, text='', logprobs=dict(tokens=tokens))], **extra)
        return ('data: ' + json.dumps(value) + '\n\n').encode()
    chunks = [frame(['token_id:1', 'token_id:2']),
              frame(['token_id:3'], usage=dict(prompt_tokens=2, completion_tokens=3, total_tokens=5)),
              b'data: [DONE]\n\n']
    class Response:
        status = 200
        headers = {}
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        @property
        def content(self): return self
        async def iter_any(self):
            for chunk in chunks: yield chunk
    class Session:
        def post(self, url, json):
            assert json['pdblend_token_diagnostics'] is True
            return Response()
    ticks = iter([110., 112., 114., 115., 116.])
    monkeypatch.setattr('pdblend.bench.client.time.time', lambda: next(ticks))
    client = LoadClient('http://unused', sampling_seed=701, token_diagnostics=True)
    row = asyncio.run(client._one(Session(), Request(0, 5, [1, 2], 3), 100.))
    assert row.scheduled_s == 105
    assert row.first_token_s == 112 and row.last_token_s == 114 and row.terminal_s == 115
    assert row.finished_s == 116 and row.ttft_s == 7 and row.tpot_s == 1
    assert row.terminal is True and row.token_events_complete is True
    assert [e['count'] for e in row.token_events] == [2, 1]
    assert client.args[-1] is True


def test_detached_replay_returns_actual_child_clock_without_changing_list_interface():
    client = LoadClient('http://unused', sampling_seed=701, token_diagnostics=True)
    before = time.time()
    result = asyncio.run(client.replay_detached([]))
    assert result == []
    assert before <= client.replay_started_s <= client.replay_finished_s <= time.time()


def test_failed_native_stream_preserves_exact_partial_delivery_and_reported_count():
    trace = [Request(0, 0, [1], 4)]
    native = dict(idx=0, correct=False, completion_tokens=0, error='TimeoutError', finished_s=105.)
    journal = [dict(kind='mixed_client_sse', request_id='mixed-701-0', at_s=102.,
                    payload=dict(token_ids=[7, 8], received_s=102., finished=False))]
    rows = canonical_outcomes('mixed', trace, [native], service_started_s=100., journal=journal)
    assert rows[0]['completion_tokens'] == 2 and rows[0]['native_reported_completion_tokens'] == 0
    metrics = reduce_comparison(trace, rows, service_started_s=100.)
    assert metrics['token_timing_complete'] and metrics['window_delivered_tokens'] == 2
    assert metrics['failed_requests'] == metrics['timeout_requests'] == 1
    assert metrics['goodput_request_s'] == metrics['success_rate'] == 0


def test_explicit_zero_delivery_failure_is_complete_but_missing_outcome_is_unknown():
    trace = [Request(0, 0, [1], 4)]
    native = dict(idx=0, correct=False, completion_tokens=0, error='no accepting replica', finished_s=101.)
    rows = canonical_outcomes('mixed', trace, [native], service_started_s=100., journal=[])
    metrics = reduce_comparison(trace, rows, service_started_s=100.)
    assert metrics['token_timing_complete'] and metrics['window_delivered_tokens'] == 0
    assert metrics['failed_requests'] == 1 and metrics['unresolved_requests'] == 0
    missing = canonical_outcomes('mixed', trace, [], service_started_s=100., journal=[])
    assert not reduce_comparison(trace, missing, service_started_s=100.)['token_timing_complete']
    without_journal = canonical_outcomes('mixed', trace, [native], service_started_s=100.)
    assert not reduce_comparison(trace, without_journal, service_started_s=100.)['token_timing_complete']


def test_claimed_success_with_wrong_count_cannot_be_repaired_into_success():
    trace = [Request(0, 0, [1], 4)]
    native = dict(idx=0, correct=True, completion_tokens=4, finished_s=105.)
    journal = [dict(kind='mixed_client_sse', request_id='mixed-701-0', at_s=102.,
                    payload=dict(token_ids=[7, 8], received_s=102., finished=True))]
    rows = canonical_outcomes('mixed', trace, [native], service_started_s=100., journal=journal)
    assert rows[0]['completion_tokens'] == 4
    assert rows[0]['error'] == 'native_client_token_count_mismatch'
    metrics = reduce_comparison(trace, rows, service_started_s=100.)
    assert not metrics['token_timing_complete'] and metrics['successful_requests'] == 0


def test_public_client_http_rejection_proves_zero_delivered_tokens():
    class Response:
        status = 409
        headers = {}
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def text(self): return 'admission full'
    class Session:
        def post(self, url, json): return Response()
    client = LoadClient('http://unused', token_diagnostics=True)
    row = asyncio.run(client._one(Session(), Request(0, 0, [1], 4), time.time()))
    assert row.error and row.completion_tokens == 0 and row.token_events_complete
