"""Dynamo route fault/activation tests with mocked V1 RPCs; never loads vLLM."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def bridge(monkeypatch):
    # The host CPU environment does not install the serving FastAPI dependency.
    # Route decorators are inert here; endpoint logic and protocol are real.
    fastapi = ModuleType('fastapi')
    class Router:
        def __init__(self, **kwargs): pass
        def get(self, *args, **kwargs): return lambda fn: fn
        def post(self, *args, **kwargs): return lambda fn: fn
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail
            super().__init__(str(detail))
    fastapi.APIRouter, fastapi.HTTPException, fastapi.Request = Router, HTTPException, object
    responses = ModuleType('fastapi.responses')
    responses.StreamingResponse = object
    monkeypatch.setitem(sys.modules, 'fastapi', fastapi)
    monkeypatch.setitem(sys.modules, 'fastapi.responses', responses)
    path = Path(__file__).resolve().parents[2]/'src/pdblend_runtime/serve.py'
    spec = importlib.util.spec_from_file_location('dynamo_test_native_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment(bridge, monkeypatch):
    import time
    monkeypatch.setenv('DYNAMO_DUMMY', '1')
    golden = dict(model_id='Qwen2.5-32B-Instruct', tp=2, engine_revision='vllm-0.10.1.1',
                  seed=701, prompt=[10, 20], token_ids=[100, 101], source_sha256='a'*64)
    monkeypatch.setenv('DYNAMO_GOLDEN_JSON', json.dumps(golden))
    app = SimpleNamespace(state=SimpleNamespace(native_tp=2, native_pp=1,
        native_identity=dict(model_id=golden['model_id'], engine_revision=golden['engine_revision'])))
    native = dict(generation=1, acknowledged_generation=1, accepting=False, active=0,
                  running=[], waiting=[], all_queue=[], kv_allocations={}, transfer_allocations={},
                  transport_healthy=True, evidence_complete=True, total_kv_tokens=128, free_kv_tokens=128,
                  num_gpu_blocks=9, free_blocks=8, reserved_blocks=1, pending_transfers=0)
    calls = []
    async def state(_request):
        return dict(native, timestamp=time.time(), native_at_s=time.time())
    async def scheduler(_request, operation, payload=None):
        calls.append(('scheduler', operation, payload))
        if operation == 'control':
            native['accepting'] = payload.get('accepting', native['accepting'])
        return await state(_request)
    async def workers(_request, method, operation=None, payload=None):
        calls.append(('worker', operation, payload))
        rows = [dict(ok=True, rank=rank, generation=1, transaction_id=payload.get('transaction_id')) for rank in range(2)]
        for row in rows:
            if operation == 'transfer':
                row.update(source=False, operation_id=payload['operation_id'], target_complete=True,
                           received_bytes=1024, parameter_count=339)
            if operation == 'drain':
                row.update(drained=True, cuda_synchronized=True, active_weight_sessions=0)
            if operation == 'mark_ready':
                row.update(weights_ready=True)
        return rows
    async def generate(_request, payload, *, private):
        assert private and not native['accepting']
        assert payload['prompt'] == golden['prompt'] and payload['seed'] == 701
        calls.append(('golden', payload, None))
        yield dict(token_ids=[100, 101], finished=True)
    monkeypatch.setattr(bridge, 'state', state)
    monkeypatch.setattr(bridge, 'scheduler', scheduler)
    monkeypatch.setattr(bridge, 'workers', workers)
    monkeypatch.setattr(bridge, 'generate_events', generate)
    class Request:
        def __init__(self, body): self.app, self.body = app, body
        async def json(self): return self.body
    return Request, native, calls


@pytest.mark.asyncio
async def test_target_requires_transfer_golden_and_all_rank_ready_before_activation(bridge, environment):
    Request, native, calls = environment
    body = dict(transaction_id='tx', operation_id='tx', expected_generation=1, session_id='tx-session')
    await bridge.dynamollm(Request(dict(body, rank_offset=2)), 'open')
    await bridge.dynamollm(Request(dict(body, source_ranks=[0, 1])), 'transfer')
    assert native['accepting'] is False
    await bridge.dynamollm(Request(body), 'close')
    verified = await bridge.dynamollm(Request(body), 'verify')
    assert verified['token_ids'] == [100, 101] and native['accepting'] is False
    activated = await bridge.dynamollm(Request(body), 'activate')
    assert activated['activated'] is True and native['accepting'] is True
    kinds = [(row[0], row[1] if isinstance(row[1], str) else None) for row in calls]
    assert kinds.index(('golden', None)) < kinds.index(('worker', 'mark_ready'))


@pytest.mark.asyncio
async def test_missing_golden_ack_never_activates_and_quarantines_native_session(bridge, environment):
    Request, native, calls = environment
    body = dict(transaction_id='tx', operation_id='tx', expected_generation=1)
    with pytest.raises(bridge.HTTPException, match='verified transaction'):
        await bridge.dynamollm(Request(body), 'activate')
    assert not native['accepting']
    assert not any(row[:2] == ('worker', 'mark_ready') for row in calls)
    with pytest.raises(bridge.HTTPException, match='uncertain'):
        await bridge.dynamollm(Request(body), 'resume')


@pytest.mark.asyncio
async def test_stale_generation_rejected_without_mutating_target(bridge, environment):
    Request, native, calls = environment
    with pytest.raises(bridge.HTTPException, match='generation differs'):
        await bridge.dynamollm(Request(dict(expected_generation=0)), 'describe')
    assert not any(row[0] == 'worker' for row in calls)
    assert not native['accepting']


@pytest.mark.asyncio
async def test_partial_rank_ack_marks_communication_uncertain(bridge, environment, monkeypatch):
    Request, native, _calls = environment
    async def partial(*args, **kwargs):
        return [dict(ok=True, rank=0, generation=1, transaction_id='tx')]
    monkeypatch.setattr(bridge, 'workers', partial)
    with pytest.raises(RuntimeError, match='rank/generation'):
        await bridge.dynamollm(Request(dict(transaction_id='tx', session_id='s', rank_offset=0)), 'open')
    with pytest.raises(bridge.HTTPException, match='uncertain'):
        await bridge.dynamollm(Request({}), 'describe')
    assert not native['accepting']
