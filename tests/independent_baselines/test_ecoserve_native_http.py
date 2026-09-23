"""Real EcoServe controller/urllib/FastAPI integration, with CPU native RPCs.

NativeScheduler's actual state/event/control/schedule methods generate every
HTTP schema. Only its GPU scheduler superclass and engine token work are fake.
"""
import ast
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import os
from pathlib import Path
import socket
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip('fastapi')
from fastapi import FastAPI
import uvicorn

from pdblend_baselines.ecoserve.runtime import EcoServeRuntime, MappedEcoServeTransport
from pdblend_runtime import serve


class SchedulerBase:
    """CPU executor state beneath the unmodified native scheduler adapter."""
    def __init__(self):
        self.max_num_running_reqs, self.max_num_scheduled_tokens = 32, 8192
        self.max_model_len, self.block_size = 8192, 16
        self.cache_config = SimpleNamespace(num_gpu_blocks=1025)
        self.parallel_config = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
        self.requests, self.running, self.waiting = {}, [], []
        self.free_override = None
        self.kv_cache_manager = SimpleNamespace(
            block_pool=SimpleNamespace(get_num_free_blocks=lambda:
                self.free_override if self.free_override is not None else 1024-2*len(self.requests)),
            get_block_ids=lambda rid: ([1, 2],))
        self.connector, self.policy = None, 'fcfs'

    def schedule(self):
        request = self.requests[self.executing]
        count = request.num_prompt_tokens if not request.num_computed_tokens else 1
        return SimpleNamespace(num_scheduled_tokens={request.request_id: count}, total_num_scheduled_tokens=count)

    def update_from_output(self, output, rid, finished):
        self.requests[rid].num_computed_tokens += output.num_scheduled_tokens[rid]
        if finished:
            self.requests.pop(rid)
            self.running = [request for request in self.running if request.request_id != rid]
        return None

    def finish_requests(self, ids, status):
        for rid in ids:
            self.requests.pop(rid, None)
            self.running = [request for request in self.running if request.request_id != rid]


def scheduler_class():
    path = Path(__file__).resolve().parents[2]/'src/pdblend_runtime/native_v1.py'
    tree = ast.parse(path.read_text())
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NativeScheduler')
    namespace = dict(Scheduler=SchedulerBase, os=os, time=time, deque=deque, create_request_queue=lambda _: [])
    module = ast.Module(body=[native], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['NativeScheduler']


class Engine:
    def __init__(self, identifier, scheduler):
        self.identifier, self.scheduler = identifier, scheduler
        self.engine_core = self
        self.worker_generation = 0
        self.calls, self.outputs = [], []
        self.pause_at = None
        self.paused, self.resume = asyncio.Event(), asyncio.Event()

    async def call_utility_async(self, method, operation, payload):
        assert method == 'native_operation'
        self.calls.append((operation, dict(payload)))
        return self.scheduler.native_operation(operation, payload)

    async def collective_rpc(self, method, kwargs):
        self.calls.append((method, dict(kwargs)))
        if method == 'native_generation_set':
            self.worker_generation = kwargs['generation']
        return [dict(rank=0, generation=self.worker_generation, native_evidence_complete=True,
                     healthy=True, pending_transfers=0, transfer_allocations={}, acknowledged=True)]

    async def generate(self, prompt, params, rid):
        assert set(prompt) == {'prompt_token_ids'}
        assert params.ignore_eos and params.seed == 701 and params.temperature == 0
        self.calls.append(('generate', dict(request_id=rid, prompt=prompt, max_tokens=params.max_tokens)))
        request = SimpleNamespace(request_id=rid, num_prompt_tokens=len(prompt['prompt_token_ids']), num_computed_tokens=0)
        self.scheduler.requests[rid] = request
        self.scheduler.running.append(request)
        tokens = []
        for index in range(params.max_tokens):
            if rid not in self.scheduler.requests:
                return
            self.scheduler.executing = rid
            scheduled = self.scheduler.schedule()
            await asyncio.sleep(.002)
            finished = index+1 == params.max_tokens
            self.scheduler.update_from_output(scheduled, rid, finished)
            tokens.append(100+index)
            self.outputs.append((rid, list(tokens), finished))
            yield SimpleNamespace(finished=finished, outputs=[SimpleNamespace(index=0, text='',
                token_ids=list(tokens), finish_reason='length' if finished else None)])
            if index+1 == self.pause_at:
                self.paused.set()
                await self.resume.wait()

    async def abort(self, rid):
        self.calls.append(('abort', dict(request_id=rid)))
        self.scheduler.finish_requests(rid, 'cancelled')
        self.resume.set()


@asynccontextmanager
async def native_services(monkeypatch, identifiers=('a', 'b'), engine_factory=Engine):
    sampling = ModuleType('vllm')
    sampling.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', sampling)
    monkeypatch.setenv('DYNAMO_GENERATION', '0')
    monkeypatch.setenv('DYNAMO_DUMMY', '0')
    monkeypatch.setattr(serve, 'clock_operation', lambda frequency:
        dict(acknowledged=True, success=True, frequency_mhz=frequency, gpus=[]))
    servers, tasks, sockets = [], [], []
    engines, endpoints = {}, {}
    try:
        for identifier in identifiers:
            native = scheduler_class()()
            # Exercise the real generation-change route during controller startup.
            native.native_role = 'prefill'
            engine = engine_factory(identifier, native)
            app = FastAPI()
            app.include_router(serve.router)
            app.state.engine_client = engine
            app.state.native_tp, app.state.native_pp = 1, 1
            if hasattr(engine, 'native_identity'):
                app.state.native_identity = engine.native_identity
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            sock.listen(128)
            config = uvicorn.Config(app, log_level='error', lifespan='off', access_log=False)
            server = uvicorn.Server(config)
            task = asyncio.create_task(server.serve(sockets=[sock]))
            servers.append(server); tasks.append(task); sockets.append(sock)
            engines[identifier] = engine
            endpoints[identifier] = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        async def started():
            while not all(server.started for server in servers):
                await asyncio.sleep(.001)
        await asyncio.wait_for(started(), 5)
        yield engines, endpoints
    finally:
        for engine in engines.values():
            engine.resume.set()
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks)
        for sock in sockets:
            sock.close()


def runtime(tmp_path, endpoints):
    profile = tmp_path/'eco-own-cpu-contract.csv'
    profile.write_text('Length,Prefill Time\n16,1\n128,8\n4096,256\n')
    rows = []
    def journal(kind, **fields):
        rows.append(dict(kind=kind, **fields))
    config = dict(instances=[dict(id=i, tp=1, gpus=[index]) for index, i in enumerate(endpoints)],
        eco_prefill_csv=str(profile), slo_ttft_s=5., slo_tpot_s=.15,
        eco_state_poll_s=.005, eco_scale_period_s=30., request_timeout_s=5., eco_drain_timeout_s=2.)
    transport = MappedEcoServeTransport(endpoints)
    return EcoServeRuntime(config, transport, journal), rows


@pytest.mark.asyncio
async def test_real_controller_admission_http_stream_and_native_terminal_cleanup(tmp_path, monkeypatch):
    async with native_services(monkeypatch) as (engines, endpoints):
        eco, journal = runtime(tmp_path, endpoints)
        try:
            await eco.start()
            assert all(engine.worker_generation == engine.scheduler.native_generation == 1 for engine in engines.values())
            for engine in engines.values():
                assert any(operation == 'native_generation_set' for operation, _ in engine.calls)
            # Force author macro rotation using actual native free-block state.
            engines['a'].scheduler.free_override = 0
            engines['a'].scheduler._native_event('resource_observed')
            result = [event async for event in eco.handle(dict(prompt=list(range(128)), max_tokens=16,
                                                           seed=701, ignore_eos=True), 'http-701')]
            await eco.controller.refresh()
            assert [token for event in result for token in event['token_ids']] == list(range(100, 116))
            assert result[-1]['finished'] and result[-1]['choices'][0]['finish_reason'] == 'length'
            admission = next(row for row in journal if row['kind'] == 'eco_admission')
            assert admission['instance_id'] == 'b' and admission['predicted_prefill_ms'] == 8
            assert admission['controls'] == [dict(instance_id='a', send_output=True), dict(instance_id='b', send_output=False)]
            assert not eco.controller.active and not eco.controller.failure
            assert not any(member.requests for member in eco.controller.members.values())
            assert not any(engine.scheduler.requests for engine in engines.values())
            assert not any(row[0] == 'generate' for row in engines['a'].calls)
            steps = [row for row in journal if row['kind'] == 'eco_native_step']
            assert any(row.get('native_kind') == 'step_completed' and not row['all_queue'] for row in steps)
        finally:
            await eco.close()


@pytest.mark.asyncio
async def test_native_schedule_and_completed_events_count_each_decode_iteration_once(tmp_path, monkeypatch):
    async with native_services(monkeypatch) as (engines, endpoints):
        eco, journal = runtime(tmp_path, endpoints)
        task = None
        try:
            await eco.start()
            engines['a'].pause_at = 3
            async def consume():
                return [event async for event in eco.handle(dict(prompt=list(range(128)), max_tokens=16,
                                            seed=701, ignore_eos=True), 'credit-701')]
            task = asyncio.create_task(consume())
            await asyncio.wait_for(engines['a'].paused.wait(), 3)
            await eco.controller.refresh()
            request = next(row for row in eco.controller.members['a'].requests if row.request_id == 'credit-701')
            # One prefill execution and two decode executions occurred.
            assert request.num_iterations == 2
            before = request.num_iterations
            engines['a'].scheduler.native_operation('control', {'mode': 'temporal'})
            await eco.controller.refresh()
            assert request.num_iterations == before, 'control ACK cannot mint saved-TPOT credit'
        finally:
            engines['a'].resume.set()
            if task:
                await task
            await eco.close()


@pytest.mark.asyncio
async def test_native_single_token_completion_does_not_leave_author_ghost_request(tmp_path, monkeypatch):
    async with native_services(monkeypatch) as (_, endpoints):
        eco, _ = runtime(tmp_path, endpoints)
        try:
            result = [event async for event in eco.handle(dict(prompt=list(range(128)), max_tokens=1,
                                             seed=701, ignore_eos=True), 'single-701')]
            assert result[-1]['finished']
            await eco.controller.refresh()
            assert not any(member.requests for member in eco.controller.members.values())
        finally:
            await eco.close()


@pytest.mark.asyncio
async def test_startup_reopens_drained_native_admission_before_reporting_ready(tmp_path, monkeypatch):
    async with native_services(monkeypatch) as (engines, endpoints):
        for engine in engines.values():
            engine.scheduler.native_operation('control', dict(accepting=False))
        eco, _ = runtime(tmp_path, endpoints)
        try:
            await eco.start()
            assert all(engine.scheduler.native_accepting for engine in engines.values())
            result = [event async for event in eco.handle(dict(prompt=[100]*16, max_tokens=2,
                                             seed=701, ignore_eos=True), 'reopened-701')]
            assert result[-1]['finished']
        finally:
            await eco.close()


@pytest.mark.asyncio
async def test_mapped_native_streams_remain_bound_to_each_real_http_endpoint(tmp_path, monkeypatch):
    async with native_services(monkeypatch) as (engines, endpoints):
        eco, _ = runtime(tmp_path, endpoints)
        try:
            await eco.start()
            base = eco.transport.base_url
            async def stream(identifier):
                return [event async for event in eco.transport.stream(identifier,
                    dict(prompt=[100]*16, max_tokens=3, request_id='mapped-'+identifier,
                         seed=701, ignore_eos=True))]
            left, right = await asyncio.gather(stream('a'), stream('b'))
            assert {row['request_id'] for row in left} == {'mapped-a'}
            assert {row['request_id'] for row in right} == {'mapped-b'}
            assert eco.transport.base_url == base
            for identifier, engine in engines.items():
                assert [row['request_id'] for kind, row in engine.calls if kind == 'generate'] == ['mapped-'+identifier]
        finally:
            await eco.close()
