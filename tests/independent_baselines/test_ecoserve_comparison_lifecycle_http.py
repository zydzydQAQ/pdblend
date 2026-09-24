"""Real runner/controller/native HTTP routes with CPU engine fixtures only."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
from pdblend_baselines.ecoserve import run_native
from pdblend_baselines.ecoserve.comparison_lifecycle import MODE, _request
from pdblend_baselines.ecoserve.runtime import HttpEcoServeTransport
from pdblend.bench.comparison_ecoserve_lifecycle import audit_lifecycle_events
from pdblend.results.journal import iter_journal

spec = importlib.util.spec_from_file_location('lifecycle_native_http_fixture',
    Path(__file__).with_name('test_ecoserve_run_native_http.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.asyncio
@pytest.mark.parametrize('timeout', [False, True])
async def test_actual_runner_opt_in_keeps_native_drain_and_replays_close_receipts(tmp_path, monkeypatch, timeout):
    config, identity, trace = fixture.inputs(tmp_path, monkeypatch)
    config['eco_comparison_lifecycle'] = MODE
    if timeout:
        config['request_timeout_s'] = .005
        original = HttpEcoServeTransport._call
        async def stalled_request_read(self, method, path, body=None):
            # A request-owned admission refresh stalls before its per-request
            # output deadline starts. Other actual native HTTP calls proceed.
            if _request.get() is not None and method == 'GET' and path == '/baseline/state':
                await asyncio.Event().wait()
            return await original(self, method, path, body)
        monkeypatch.setattr(HttpEcoServeTransport, '_call', stalled_request_read)
    async with fixture.services(monkeypatch, identity) as (engines, endpoints, _):
        result = await run_native.execute(config, endpoints, trace, tmp_path/'run', .02 if timeout else .08)
    events = list(iter_journal(tmp_path/'run/events.jsonl.gz'))
    proof = audit_lifecycle_events(events, result, config)
    assert proof['independently_replayed'] and not proof['policy_changed']
    assert result['drain_kv_released'] and not result['cleanup_errors']
    assert len(result['outcomes']) == 1
    assert all(not engine.scheduler.requests for engine in engines.values())
    assert not any(row['kind'] == 'eco_membership_rollback' for row in events)
    if timeout:
        assert result['error'] == 'TimeoutError()' and result['status'] == 'failed'
        begin = next(r for r in events if r['kind'] == 'eco_comparison_cohort_cancel_begin')
        assert begin['reason'] == 'cohort_timeout'
        assert begin['pending_http'] and proof['allowed_cancelled_read_ids']
        assert begin['at_s'] <= result['outcomes'][0]['finished_s']
    else:
        assert result['status'] == 'passed' and not result['comparison_lifecycle']['cohort_cancelled']
