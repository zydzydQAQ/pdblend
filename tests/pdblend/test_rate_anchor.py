import asyncio
import json
from types import SimpleNamespace

import pytest

from pdblend.bench.rate_anchor import next_rate
from pdblend.bench.native_mixed import metrics
from pdblend.bench import native_mixed
from pdblend.bench.client import Request
from pdblend.results.journal import iter_journal


def test_anchor_brackets_without_promoting_a_failed_window():
    row = lambda r, ok:dict(rate_rps=r, metrics=dict(passed=ok))
    assert next_rate([row(1, False)]) == .5
    assert next_rate([row(1, True)]) == 2
    assert next_rate([row(1, True), row(2, False)]) == 1.5
    assert next_rate([row(1, False), row(.5, True)]) == .75


def test_failed_and_tail_requests_count_in_slo():
    good = dict(correct=True, completion_tokens=16, ttft_s=.3, tpot_s=.05)
    result = metrics([good]*9+[dict(correct=False, completion_tokens=0)], 10, (1, .1))
    assert result['joint_slo_rate'] == .9 and result['passed'] is False
    late = dict(good, ttft_s=2.)
    result = metrics([good]*99+[late], 100, (1, .1))
    assert result['joint_slo_rate'] == .99 and result['ttft_p99_s'] == .3
    with pytest.raises(ValueError):
        metrics([good], 2, (1, .1))


def test_native_mixed_routes_and_releases_after_actual_transport_failure(monkeypatch, tmp_path):
    async def generate(session, url, payload, observe=None):
        if payload['request_id'].endswith('-1'):
            raise RuntimeError('stream failed')
        now = native_mixed.time.time()
        event = dict(received_s=now, token_ids=[1, 2], finished=True)
        if observe is not None:
            observe(event)
        return dict(events=[event], token_ids=[1,2])
    monkeypatch.setattr(native_mixed, 'generate', generate)
    # The unreachable cancel endpoint must become an explicit error receipt;
    # it must not leak the admission count or hide the failed request.
    specs = [SimpleNamespace(instance_id='a', tp=1, max_num_seqs=8, base_url='http://127.0.0.1:1')]
    trace = [Request(i, 0., [1]*16, 2) for i in range(2)]
    result = asyncio.run(native_mixed.execute(specs, trace, tmp_path/'out', duration_s=.01,
                                             slo=(1., .1), seed=9701))
    assert result['counts_reclaimed']
    assert result['metrics']['success_rate'] == .5
    rows = list(iter_journal(tmp_path/'out/outcomes.jsonl'))
    assert any(r.get('cancel_error') for r in rows)
    assert all('events' not in row and 'token_ids' not in row for row in rows)
