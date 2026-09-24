"""Author queues + actual Mapped transport + native HTTP retained-KV routes.

Only the GPU model execution, KV tensor copying and worker RPC wire are fake;
the retained-KV manager, native routes and DistServe runtime are real.
"""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
pytest.importorskip('fastapi')
from fastapi import FastAPI
import uvicorn

_spec = importlib.util.spec_from_file_location('dist_runtime_http_base',
    Path(__file__).with_name('test_ecoserve_native_http.py'))
_base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_base)
from pdblend_baselines.distserve.runtime import DistServeRuntime, MappedDistServeTransport, DistServeCapabilityError
from pdblend_runtime import serve
from pdblend_runtime.kv import NativeKV


class Engine(_base.Engine):
    def __init__(self, identifier, scheduler):
        super().__init__(identifier, scheduler)
        self.worker = NS(_native_generation=0, parallel_config=NS(tensor_parallel_size=1),
            model_config=NS(hf_config=NS(num_hidden_layers=2)),
            send_kv_layer=lambda tensor, slot, **kwargs: dict(submitted=True),
            wait_for_sent=lambda tx, rows: dict(acknowledged=True, ranks=rows),
            start_load_kv=lambda **kwargs: True, consume_loaded_kv=lambda **kwargs: True)
        self.kv = NativeKV(self.worker)
        self.bad_load_ack = False
        self.started, self.allow_prefill = asyncio.Event(), asyncio.Event()
        self.allow_prefill.set()
        scheduler._free_blocks = lambda request: scheduler.requests.pop(request.request_id)

    async def abort(self, rid):
        # Match V1's output-processor filter: already-completed retained
        # prefill has no live output request, so ordinary abort is a no-op.
        # The dedicated pinned-body HTTP test below verifies this behavior.
        if rid in self.scheduler.native_held:
            self.calls.append(('abort_filtered', dict(request_id=rid)))
            return
        await super().abort(rid)

    async def collective_rpc(self, method, kwargs):
        self.calls.append((method, dict(kwargs)))
        if method == 'native_generation_set':
            self.worker_generation = self.worker._native_generation = kwargs['generation']
        if method == 'native_kv_operation':
            operation, payload = kwargs['operation'], kwargs['payload']
            if operation == 'state': value = self.kv.state()
            elif operation == 'release': value = self.kv.release(payload['held_request_id'])
            elif operation == 'cancel': value = self.kv.cancel(payload['request_id'])
            else: value = getattr(self.kv, operation)(payload)
            if operation == 'query_load' and self.bad_load_ack: value['loaded_layers'] = 1
            return [dict(value, rank=0, generation=self.worker_generation, tp=1, pp=1)]
        return [dict(rank=0, generation=self.worker_generation, native_evidence_complete=True,
            healthy=True, acknowledged=True, pending_transfers=self.kv.state()['receiving_transactions'],
            transfer_allocations={}, retained_kv_supported=True)]

    async def generate(self, prompt, params, rid):
        source = rid.startswith('distserve-hold-')
        if source:
            self.started.set(); await self.allow_prefill.wait()
        else:
            for key in list(self.kv.expected):
                if key[0] == rid:
                    for layer in range(2):
                        self.kv.load_ack(dict(target_request_id=rid, transaction_id=key[1],
                                             generation=self.worker_generation, layer=layer))
        async for value in super().generate(prompt, params, rid):
            if source and value.finished:
                self.kv.hold(rid, '', {0: 'GPU-layer-0', 1: 'GPU-layer-1'}, {0: [1], 1: [1]})
                self.scheduler.native_held.add(rid)
                self.scheduler.requests[rid] = NS(request_id=rid, is_finished=lambda: True)
            elif not source:
                for choice in value.outputs:
                    choice.token_ids = [token+1 for token in choice.token_ids]
            yield value


@asynccontextmanager
async def services(monkeypatch):
    module = ModuleType('vllm'); module.SamplingParams = lambda **kwargs: NS(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setenv('DYNAMO_GENERATION', '0'); monkeypatch.setenv('DYNAMO_DUMMY', '0')
    endpoints, engines, tasks, servers, sockets = {}, {}, [], [], []
    try:
        for index, role in enumerate(('P', 'D')):
            core = _base.scheduler_class()(); engine = Engine(role, core)
            app = FastAPI(); app.include_router(serve.router)
            app.state.engine_client = engine; app.state.native_tp = app.state.native_pp = 1
            app.state.native_kv_address = '127.0.0.1:'+str(31000+16*index)
            app.state.native_identity = dict(model_id='Qwen2.5-7B-Instruct', model_hash='a'*64,
                tokenizer_hash='b'*64, image_digest='sha256:'+'c'*64, source_revision='d'*64,
                engine_revision='vllm-0.10.1.1', gpu_uuids=['GPU-'+role])
            sock = socket.socket(); sock.bind(('127.0.0.1', 0)); sock.listen(128)
            server = uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error', access_log=False))
            tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
            servers.append(server); sockets.append(sock); engines[role] = engine
            endpoints[role] = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        while not all(server.started for server in servers): await asyncio.sleep(.001)
        transport = MappedDistServeTransport(endpoints['P'], endpoints['D'],
            prefill_address='127.0.0.1:31000', decode_address='127.0.0.1:31016')
        yield engines, transport
    finally:
        for engine in engines.values(): engine.resume.set(); engine.allow_prefill.set()
        for server in servers: server.should_exit = True
        await asyncio.gather(*tasks)
        for sock in sockets: sock.close()


def payload(length=128, output=16):
    return dict(prompt=list(range(100, 100+length)), max_tokens=output, seed=701)


async def consume(runtime, rid, value=None):
    return [event async for event in runtime.handle(value or payload(), rid)]


@pytest.mark.asyncio
async def test_real_runtime_executes_overlapping_requests_with_complete_native_kv_receipts(monkeypatch):
    async with services(monkeypatch) as (engines, transport):
        rows = []
        runtime = DistServeRuntime(transport, poll_s=.002, journal=lambda kind, **kw: rows.append(dict(kind=kind, **kw)))
        try:
            await runtime.start()
            first, second = await asyncio.wait_for(asyncio.gather(consume(runtime, 'first'), consume(runtime, 'second')), 10)
            for rid, values in [('first', first), ('second', second)]:
                result = runtime.results[rid]
                assert result.status == 'completed' and result.tokens == 16
                assert result.token_ids == list(range(100, 116))
                assert [row['token_index'] for row in values] == list(range(1, 17))
                assert not values[0]['finished'] and values[-1]['finished']
                assert {r['step'] for r in result.receipts} == {'prefill', 'expect_load', 'transfer', 'load_ack', 'release'}
                assert not result.formal_eligible and not result.energy_comparable
            assert any(row['kind'] == 'distserve_prefill_admission' and len(row['request_ids']) == 2 for row in rows)
            assert any(row['kind'] == 'distserve_decode_selected' for row in rows)
            assert not runtime.prefill.retained and not runtime.decode.active
            assert all(not engine.scheduler.requests for engine in engines.values())
        finally: await runtime.close()


@pytest.mark.asyncio
async def test_one_token_request_releases_prefill_without_remote_transfer(monkeypatch):
    async with services(monkeypatch) as (_, transport):
        runtime = DistServeRuntime(transport, poll_s=.002)
        try:
            values = await asyncio.wait_for(consume(runtime, 'one', payload(output=1)), 5)
            assert len(values) == 1 and values[0]['finished']
            assert [r['step'] for r in runtime.results['one'].receipts] == ['prefill', 'release']
        finally: await runtime.close()


@pytest.mark.asyncio
async def test_client_cancel_during_decode_has_real_cancel_receipts_and_releases_both_sides(monkeypatch):
    async with services(monkeypatch) as (engines, transport):
        engines['D'].pause_at = 2
        runtime = DistServeRuntime(transport, poll_s=.002)
        task = asyncio.create_task(consume(runtime, 'cancel-live'))
        await asyncio.wait_for(engines['D'].paused.wait(), 5)
        task.cancel(); await asyncio.gather(task, return_exceptions=True)
        result = runtime.results['cancel-live']
        assert result.status == 'cancelled' and result.tokens >= 2
        assert {'cancel_P', 'cancel_D'} <= {row['step'] for row in result.receipts}
        assert all(not engine.scheduler.requests for engine in engines.values())
        await runtime.close()


@pytest.mark.asyncio
async def test_bad_layer_ack_is_not_success_and_recovery_requires_actual_empty_states(monkeypatch):
    async with services(monkeypatch) as (engines, transport):
        engines['D'].bad_load_ack = True
        runtime = DistServeRuntime(transport, poll_s=.002)
        with pytest.raises(DistServeCapabilityError):
            await asyncio.wait_for(consume(runtime, 'bad-load'), 5)
        assert runtime.results['bad-load'].status == 'failed'
        assert not any(row['finished'] for row in runtime.results['bad-load'].events)
        assert runtime.quarantined == {'P', 'D'}
        assert {'cancel_P', 'cancel_D'} <= {row['step'] for row in runtime.results['bad-load'].receipts}
        engines['D'].bad_load_ack = False
        engines['D'].scheduler.free_override = 1023
        with pytest.raises(DistServeCapabilityError, match='recovery'):
            await runtime.recover()
        engines['D'].scheduler.free_override = None
        assert (await runtime.recover())['acknowledged']
        await asyncio.wait_for(consume(runtime, 'recovered'), 5)
        assert runtime.results['recovered'].status == 'completed'
        await runtime.close()


@pytest.mark.asyncio
async def test_actual_trace_entrypoint_binds_seed_identity_and_preserves_request_evidence(monkeypatch, tmp_path):
    from pdblend_baselines.distserve.run_native import execute
    async with services(monkeypatch) as (_, transport):
        trace = tmp_path/'trace.json'
        trace.write_text(json.dumps(dict(seed=701, model_id='Qwen2.5-7B-Instruct', tokenizer_hash='b'*64,
            requests=[dict(arrival_s=0., **payload()), dict(arrival_s=.01, **payload(output=1))])))
        options = NS(trace=trace, out=tmp_path/'run', tp=1, pp=1, max_batch_size=2, request_timeout=10.,
            prefill_url=transport.prefill_url, decode_url=transport.decode_url,
            prefill_address=transport.prefill_address, decode_address=transport.decode_address)
        result = await execute(options)
        assert result['status'] == 'passed' and result['complete'] and result['trace_model_identity_complete']
        assert len(result['outcomes']) == 2 and all(row['ok'] for row in result['outcomes'])
        assert not result['gpu_batch_equivalence_qualified'] and not result['complete_reproduction']
        assert not result['formal_eligible'] and not result['energy_comparable']
        assert len(result['events_sha256']) == len(result['trace_sha256']) == 64
        assert json.loads((options.out/'completion.json').read_text())['status'] == 'passed'
        assert all(row['ttft_s'] > 0 for row in result['outcomes'])
        assert all(row['native_receipts_complete'] and row['terminal_observed'] for row in result['outcomes'])


@pytest.mark.asyncio
async def test_trace_observation_window_keeps_real_idle_tail(monkeypatch, tmp_path):
    from pdblend_baselines.distserve.run_native import execute
    async with services(monkeypatch) as (_, transport):
        trace = tmp_path/'trace.json'
        trace.write_text(json.dumps(dict(seed=701, requests=[dict(arrival_s=0., **payload(output=1))])))
        options = NS(trace=trace, out=tmp_path/'run', tp=1, pp=1, max_batch_size=2, request_timeout=10.,
            prefill_url=transport.prefill_url, decode_url=transport.decode_url,
            prefill_address=transport.prefill_address, decode_address=transport.decode_address, duration=.15)
        result = await execute(options)
        assert result['complete'] and result['observed_duration_s'] >= .15
        assert not result['request_golden_comparison_performed']
        options.duration = 0.
        with pytest.raises(ValueError, match='duration'):
            await execute(options)


@pytest.mark.asyncio
async def test_state_http_preserves_owner_timestamp_and_orders_worker_observation_first(monkeypatch):
    import time
    async with services(monkeypatch) as (engines, transport):
        engine = engines['P']; old_rpc = engine.collective_rpc
        observed = []
        async def slow_workers(method, kwargs):
            if method == 'native_worker_state':
                await asyncio.sleep(.02)
                observed.append(time.time())
            return await old_rpc(method, kwargs)
        engine.collective_rpc = slow_workers
        result = await transport.state('P')
        assert observed[-1] <= result['native_at_s'] == result['scheduler_at_s']
        assert result['rank_observation_started_s'] <= result['rank_observation_finished_s'] <= result['scheduler_at_s']
        assert result['scheduler_at_s'] <= result['response_at_s']
        assert result['rank_observed_at_s'][0]['rank'] == 0
        assert result['atomic_snapshot'] is result['atomic_rank_scheduler_snapshot'] is False


@pytest.mark.asyncio
async def test_completed_retained_cancel_uses_release_when_pinned_async_abort_filters_id(monkeypatch):
    import ast
    from types import MethodType
    # Execute the shipped abort bodies without importing vLLM or initializing
    # CUDA. This is the real API/output-processor filter that the old fake
    # abort shortcut bypassed, hiding retained-source leaks.
    package = importlib.util.find_spec('vllm')
    if package is None:
        pytest.skip('requires pinned vLLM source; run this contract in the CPU-only campaign image')
    from importlib.metadata import version
    assert version('vllm') == '0.10.1.1', 'abort contract must use the campaign-pinned engine source'
    root = Path(next(iter(package.submodule_search_locations)))
    namespace = {'as_list': list}
    for filename, method in [('async_llm.py', 'abort'), ('output_processor.py', 'abort_requests')]:
        tree = ast.parse((root/'v1/engine'/filename).read_text())
        node = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                    for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
                     '<pinned-abort-contract>', 'exec'), namespace)
    async with services(monkeypatch) as (engines, transport):
        engine = engines['P']; forwarded = []
        processor = NS(request_states={}, parent_requests={})
        processor.abort_requests = MethodType(namespace['abort_requests'], processor)
        engine.output_processor, engine.log_requests = processor, False
        async def abort_requests_async(ids):
            forwarded.extend(ids)
            for rid in ids: engine.scheduler.finish_requests(rid, 'cancelled')
        engine.abort_requests_async = abort_requests_async
        engine.abort = MethodType(namespace['abort'], engine)
        saved = await transport.prefill(dict(request_id='completed-retained', **payload(output=1)))
        rid = saved['retained_handle']
        assert rid in engine.scheduler.requests and rid in engine.scheduler.native_held
        receipt = await transport.cancel('P', dict(request_id=rid, timeout_s=.1))
        assert forwarded == []  # Actual pinned filter ignores the completed ID.
        assert receipt['acknowledged'] and receipt['retained_release']['released']
        assert receipt['retained_release']['ranks'][0]['released']
        assert rid not in engine.scheduler.requests and rid not in engine.scheduler.native_held
        assert rid not in engine.kv.held and not receipt['native_state']['kv_allocations']


@pytest.mark.asyncio
@pytest.mark.parametrize('uncertainty', ['pending', 'received', 'incomplete_rank'])
async def test_retained_cancel_never_releases_uncertain_transport(monkeypatch, uncertainty):
    async with services(monkeypatch) as (engines, transport):
        engine = engines['P']
        saved = await transport.prefill(dict(request_id='uncertain-held', **payload(output=1)))
        rid = saved['retained_handle']; original = engine.collective_rpc
        async def uncertain_rpc(method, kwargs):
            rows = await original(method, kwargs)
            if method == 'native_worker_state':
                if uncertainty == 'pending': rows[0]['pending_transfers'] = 1
                if uncertainty == 'received': rows[0]['transfer_allocations'] = {rid+'#layer': 'received'}
                if uncertainty == 'incomplete_rank': rows[0]['native_evidence_complete'] = False
            return rows
        engine.collective_rpc = uncertain_rpc
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(transport.prefill_url+'/baseline/cancel',
                                    json=dict(request_id=rid, timeout_s=0)) as response:
                assert response.status == 409
                detail = (await response.json())['detail']
                assert isinstance(detail, str)
                assert json.loads(detail)['reason'] == 'cancellation resources not released'
        assert rid in engine.kv.held and rid in engine.scheduler.native_held and rid in engine.scheduler.requests
        assert not any(name == 'native_kv_operation' and value['operation'] == 'release' for name, value in engine.calls)
