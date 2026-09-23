"""GPU-probe orchestration through actual HTTP routes; only GPU execution fake."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
pytest.importorskip('fastapi')

spec = importlib.util.spec_from_file_location('dist_gpu_http_fixture',
    Path(__file__).with_name('test_distserve_request_runtime_http.py'))
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)

from pdblend_baselines.distserve.gpu_probe import run_resident, specifications, trace_rows


def test_fixed_work_trace_and_two_owned_groups():
    rows = trace_rows()
    assert rows == trace_rows()
    assert [len(row['prompt']) for row in rows] == [512, 2048, 7168, 512, 512, 512]
    assert [row['max_tokens'] for row in rows] == [16, 16, 16, 1, 512, 16]
    assert all(row['seed'] == 701 and row['ignore_eos'] for row in rows)
    pair = specifications('/models/Qwen2.5-32B-Instruct', [0, 1, 2, 3], 2, 12000)
    assert pair[0].gpus == (0, 1) and pair[1].gpus == (2, 3)
    assert pair[0].port == 12000 and pair[1].port == 12016
    assert pair[0].zmq_address == '127.0.0.1:32000'
    with pytest.raises(ValueError): specifications('m', [0, 0], 1, 12000)


@pytest.mark.asyncio
async def test_resident_probe_real_http_full_pipeline_cancel_and_recovery(monkeypatch, tmp_path):
    async with fixture.services(monkeypatch) as (engines, transport):
        # The low-level fake models carried decoding as +1; ordinary goldens
        # instead start at token 100, matching the prefill first token.
        original = engines['D'].generate
        async def gpu_model(prompt, params, rid):
            async for value in original(prompt, params, rid):
                if rid.startswith('distserve-reference-'):
                    for choice in value.outputs: choice.token_ids = [token-1 for token in choice.token_ids]
                yield value
        engines['D'].generate = gpu_model
        result = await asyncio.wait_for(run_resident(transport.prefill_url, transport.decode_url,
            transport.prefill_address, transport.decode_address, 1, tmp_path/'probe'), 20)
        assert result['complete'], result
        assert result['status'] == 'passed' and len(result['outcomes']) == 5
        assert result['concurrent_requests_observed']
        assert result['cancel']['acknowledgement']['status'] == 'cancelled'
        assert result['cancel']['result']['tokens'] >= 2
        assert all(not row['all_queue'] and not row['kv_allocations'] for row in result['recovery']['states'].values())
        assert all(row['golden_match'] for row in result['outcomes'].values())
        assert not result['formal_eligible'] and not result['energy_comparable'] and not result['profiles_collected']
        assert not result['gpu_batch_equivalence_qualified'] and result['recovery_scope'] == 'real_cancellation_only'
        assert json.loads((tmp_path/'probe/runtime.json').read_text())['complete']
        assert all(not engine.scheduler.requests for engine in engines.values())


@pytest.mark.asyncio
async def test_probe_does_not_hide_golden_mismatch(monkeypatch, tmp_path):
    # Uncorrected low-level fake deliberately produces an inconsistent golden.
    async with fixture.services(monkeypatch) as (_, transport):
        result = await asyncio.wait_for(run_resident(transport.prefill_url, transport.decode_url,
            transport.prefill_address, transport.decode_address, 1, tmp_path/'mismatch'), 20)
        assert result['status'] == 'failed' and not result['complete']
        assert 'golden' in result['error']
        assert not result['cleanup_errors']
