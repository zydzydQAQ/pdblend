import asyncio
import json
from pathlib import Path
import pytest
import child
import run
from fixed_contract import validate_row


def test_real_benchmark_late_epoch_dispatches_nothing(monkeypatch):
    called = []
    original = child.bench_vllm.evaluation_headers
    gate = child.EpochGate(original, {'latest_arrival_epoch_s': 0}, called.append)
    monkeypatch.setattr(child.bench_vllm, 'evaluation_headers', gate)
    async def no_request(*a, **kw):
        raise AssertionError('a request was dispatched after phase admission closed')
    monkeypatch.setattr(child.bench_vllm, 'send_request', no_request)
    with pytest.raises(RuntimeError, match='arrival epoch'):
        asyncio.run(child.bench_vllm.run_trace(
            {'requests': [{'arrival_s': 0, 'output_len': 1}], 'prompts': ['hello']},
            'http://127.0.0.1:18080', 'test', evaluation_protocol=child.bench_vllm.EVALUATION_V3))
    assert not called and not gate.checked


def test_epoch_gate_checks_once_and_preserves_request_headers():
    original = child.bench_vllm.evaluation_headers
    records = []
    gate = child.EpochGate(original, {'latest_arrival_epoch_s': 12}, records.append)
    assert gate('http://127.0.0.1:18080', 'evaluation-v3', 10, 10) == original(
        'http://127.0.0.1:18080', 'evaluation-v3', 10, 10)
    assert gate('http://127.0.0.1:18080', 'evaluation-v3', 100, 101) == original(
        'http://127.0.0.1:18080', 'evaluation-v3', 100, 101)
    assert len(records) == 1


def test_actual_generator_rows_accept_variable_n():
    rows = json.loads((Path(__file__).parent / 'runspec.json').read_text())['cells']
    for row in rows:
        validate_row(run, row)
    assert min(row['n_requests'] for row in rows) < 1000


def test_wrong_slo_is_rejected():
    row = json.loads((Path(__file__).parent / 'runspec.json').read_text())['cells'][0]
    with pytest.raises(RuntimeError, match='SLO'):
        validate_row(run, dict(row, slo_ttft_s=5))
