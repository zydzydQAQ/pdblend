"""CPU native-scheduler/FastAPI execution of the real EcoServe campaign entry.

Only GPU token work, CUDA profiling observations and NVML calls are fixtures;
the runner, policy, HTTP routes, generation ACKs and scheduler inventories run.
"""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
pytest.importorskip('fastapi')
from fastapi import FastAPI
import uvicorn

from pdblend_baselines.ecoserve import run_native
from pdblend_baselines.native_profile import ECO_LENGTHS, _sha
from pdblend_runtime import serve

_spec = importlib.util.spec_from_file_location('eco_campaign_native_fixture', Path(__file__).with_name('test_ecoserve_native_http.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def inputs(tmp_path, monkeypatch, count=1):
    source, image = 'a'*64, 'sha256:'+'b'*64
    model = 'Qwen2.5-7B-Instruct'
    inventory = [dict(path='weights.safetensors', bytes=123, sha256='c'*64, kind='weight'),
                 dict(path='tokenizer.json', bytes=42, sha256='d'*64, kind='tokenizer')]
    receipt = tmp_path/'model-receipt.json'
    receipt.write_text(json.dumps(dict(all_pass=True, models={'7b':dict(model_id=model, verified=True, files=inventory)})))
    monkeypatch.setenv('PDBLEND_SOURCE_SHA256', source)
    monkeypatch.setenv('PDBLEND_IMAGE_ID', image)
    monkeypatch.setenv('PDBLEND_MODEL_VERIFICATION_RECEIPT', str(receipt))
    monkeypatch.setenv('PDBLEND_GPU_UUIDS', ','.join('GPU-'+str(i) for i in range(count)))
    config = dict(model_id=model, instances=[dict(id='eco'+str(i), tp=1, pp=1, gpus=[i]) for i in range(count)],
                  slo_ttft_s=5, slo_tpot_s=.15, eco_macro_lower=1, eco_macro_upper=2,
                  eco_initial_instances=1, eco_scale_period_s=.02, eco_state_poll_s=.005,
                  eco_active_frequency_mhz=2520, request_timeout_s=2, eco_drain_timeout_s=1)
    identity = run_native.expected_identity(config)
    meta = dict(model_hash=identity['model_hash'], tokenizer_hash=identity['tokenizer_hash'],
                image_digest=image, source_revision=source, engine_version='vllm-0.10.1.1',
                gpu_uuids=['GPU-0'], tp=1, pp=1, frequency_mhz=2520)
    rows = []
    for length in ECO_LENGTHS:
        repeats = []
        for i in range(5):
            sample = dict(role='prefill', input_tokens=length, context_tokens=length, batch=1,
                          gpu_elapsed_ms=length/128, measurement_scope='forward', system='ecoserve',
                          request_ids=[f'unit-{length}-{i}'], rank=0, tp=1, pp=1)
            repeats.append(dict(ranks=[dict(rank=0, samples=[sample])]))
        rows.append(dict(input_tokens=length, repetitions=repeats, minimum_ms=length/128,
                         samples_sha256=[_sha(value) for value in repeats]))
    profile = tmp_path/'own-profile.csv'
    profile.write_text('Length,Prefill Time\n'+''.join(f'{row["input_tokens"]},{row["minimum_ms"]}\n' for row in rows))
    Path(str(profile)+'.manifest.json').write_text(json.dumps(dict(schema='pdblend-baseline-profile-v1',
        complete=True, system='ecoserve', model=model, metadata=meta, rows=rows)))
    config['eco_prefill_csv'] = str(profile)
    trace = tmp_path/'trace.json'
    trace.write_text(json.dumps(dict(seed=701, requests=[dict(arrival_s=.01, prompt=[100]*128, max_tokens=4)])))
    return config, identity, trace


@asynccontextmanager
async def services(monkeypatch, identity, count=1, *, fault=None, slow_first=False, engine_class=None):
    sampling = ModuleType('vllm')
    sampling.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', sampling)
    monkeypatch.setenv('DYNAMO_GENERATION', '0')
    monkeypatch.setenv('DYNAMO_DUMMY', '0')
    owner = ContextVar('eco_owner')
    calls = []
    def clock(frequency):
        identifier = owner.get()
        calls.append((identifier, frequency))
        if fault == 'cleanup' and frequency is None:
            raise RuntimeError('injected physical clock reset failure')
        return dict(acknowledged=True, success=True, requested_frequency_mhz=frequency,
                    gpus=[dict(gpu_uuid='GPU-'+identifier[3:], frequency_mhz=frequency or 210,
                               power_w=100)], at_s=time.time())
    monkeypatch.setattr(serve, 'clock_operation', clock)
    if fault == 'drain_ack':
        original = serve.drain_engine
        async def missing_ack(*args):
            value = await original(*args)
            value['acknowledged'] = False
            return value
        monkeypatch.setattr(serve, 'drain_engine', missing_ack)
    class Engine(engine_class or _fixture.Engine):
        async def collective_rpc(self, method, kwargs):
            if method == 'native_generation_set':
                await asyncio.sleep(.03)
            rows = await super().collective_rpc(method, kwargs)
            for row in rows:
                row['at_s'] = time.time() - (5 if fault == 'stale_rank' else 0)
            return [] if fault == 'ranks' else rows
        async def generate(self, *args):
            if args[1].seed == 9701:
                self.calls.append(('warmup_seed', dict(seed=9701, prompt=args[0])))
                # Only fake GPU execution normalizes the fixture assertion;
                # the actual HTTP SamplingParams carried 9701 above.
                args = (args[0], SimpleNamespace(**dict(vars(args[1]), seed=701)), args[2])
            if slow_first:
                await asyncio.sleep(.04)
            async for event in super().generate(*args):
                if fault == 'truncated' and event.finished:
                    return
                yield event
    servers, tasks, sockets, engines, endpoints = [], [], [], {}, {}
    try:
        for i in range(count):
            identifier = 'eco'+str(i)
            scheduler = _fixture.scheduler_class()()
            scheduler.native_role = 'prefill'
            engine = Engine(identifier, scheduler)
            app = FastAPI()
            app.include_router(serve.router)
            app.state.engine_client = engine
            app.state.native_tp, app.state.native_pp = 1, 1
            app.state.native_kv_address = '127.0.0.1:'+str(31000+16*i)
            app.state.native_identity = dict(identity, gpu_uuids=['GPU-'+str(i)])
            if fault == 'model':
                app.state.native_identity['model_hash'] = 'e'*64
            if fault == 'uuid':
                app.state.native_identity['gpu_uuids'] = ['GPU-other']
            @app.middleware('http')
            async def context(request, next_call, iid=identifier):
                token = owner.set(iid)
                try:
                    return await next_call(request)
                finally:
                    owner.reset(token)
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0)); sock.listen(128)
            server = uvicorn.Server(uvicorn.Config(app, log_level='critical', lifespan='off', access_log=False))
            task = asyncio.create_task(server.serve(sockets=[sock]))
            servers.append(server); tasks.append(task); sockets.append(sock)
            engines[identifier] = engine
            endpoints[identifier] = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        async def ready():
            while not all(s.started for s in servers):
                await asyncio.sleep(.001)
        await asyncio.wait_for(ready(), 5)
        yield engines, endpoints, calls
    finally:
        for engine in engines.values():
            engine.resume.set()
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*tasks)
        for sock in sockets:
            sock.close()


@pytest.mark.asyncio
async def test_real_http_functional_success_quiet_policy_inconclusive_and_exact_service_start(tmp_path, monkeypatch):
    config, identity, trace = inputs(tmp_path, monkeypatch)
    async with services(monkeypatch, identity) as (engines, endpoints, calls):
        result = await run_native.execute(config, endpoints, trace, tmp_path/'run', .08)
    assert result['functional_status'] == 'passed', result
    assert result['status'] == 'passed' and result['automatic_policy_status'] == 'inconclusive'
    assert result['complete'] and not result['automatic_policy_triggered']
    assert not result['complete_reproduction']
    assert result['service_finished_s']-result['service_started_s'] >= .08
    assert result['outcomes'][0]['submitted_s']-result['service_started_s'] >= .009
    startup = next(r for r in result['journal'] if r['kind'] == 'eco_startup')
    assert startup['at_s'] < result['service_started_s']
    assert result['outcomes'][0]['token_ids'] == result['outcomes'][0]['native_token_ids'] == [100,101,102,103]
    assert result['drain_receipts']['eco0']['acknowledged'] and result['drain_kv_released']
    assert not result['cleanup_errors'] and ('eco0', None) in calls
    assert any(r['kind'] == 'eco_scale_observation' for r in result['journal'])
    assert all(not engine.scheduler.requests for engine in engines.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['model', 'uuid', 'ranks', 'stale_rank', 'drain_ack', 'cleanup', 'truncated'])
async def test_real_http_identity_drain_or_cleanup_failure_never_passes(tmp_path, monkeypatch, fault):
    config, identity, trace = inputs(tmp_path, monkeypatch)
    async with services(monkeypatch, identity, fault=fault) as (engines, endpoints, _):
        result = await run_native.execute(config, endpoints, trace, tmp_path/'run', .04)
    assert result['status'] == 'failed' and not result['complete'], result
    assert result['functional_status'] == 'failed'
    assert result.get('error') or result['cleanup_errors'] or any(not row['ok'] for row in result['outcomes'])
    if fault in ('model', 'uuid', 'ranks', 'stale_rank'):
        assert not any(op == 'generate' for e in engines.values() for op, _ in e.calls)


@pytest.mark.asyncio
async def test_rehashed_csv_cannot_replace_native_forward_profile(tmp_path, monkeypatch):
    config, identity, trace = inputs(tmp_path, monkeypatch)
    path = Path(config['eco_prefill_csv'])
    path.write_text(path.read_text().replace('128,1.0', '128,99.0'))
    async with services(monkeypatch, identity) as (engines, endpoints, calls):
        result = await run_native.execute(config, endpoints, trace, tmp_path/'run', .04)
    assert result['status'] == 'failed' and 'CSV differs' in result['error']
    assert not calls


@pytest.mark.asyncio
async def test_real_automatic_add_requires_native_clock_prepare_commit_receipts(tmp_path, monkeypatch):
    config, identity, trace = inputs(tmp_path, monkeypatch, count=2)
    config['slo_ttft_s'] = .01
    async with services(monkeypatch, identity, count=2, slow_first=True) as (_, endpoints, calls):
        result = await run_native.execute(config, endpoints, trace, tmp_path/'run', .2)
    assert result['functional_status'] == 'passed', result
    assert result['automatic_policy_triggered'] and result['automatic_policy_qualified'], result
    assert result['status'] == 'passed' and result['complete']
    assert result['automatic_actions'][0]['trigger'] == 'mean_ttft'
    assert ('eco1', 2520) in calls and result['drain_kv_released']
    assert not result['formal_eligible'] and not result['energy_comparable']
