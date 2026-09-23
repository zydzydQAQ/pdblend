"""Real HTTP collector + native AST instrumentation, CPU simulated GPU clock.

Only CUDA-event time, NVML power and engine tensor work are simulated. Serving
routes, SSE framing, worker measurement reducers, collector windows/fit/resume
and generation-aware V1 transport execute their actual source implementations.
"""
import ast
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip('fastapi')
_spec = importlib.util.spec_from_file_location('dynamo_profile_http_fixture', Path(__file__).with_name('test_ecoserve_native_http.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)

from pdblend_baselines.dynamollm import profile_v1
from pdblend_baselines.dynamollm.deployment import sha
from pdblend_baselines.dynamollm.transport import V1Transport


class Clock:
    def __init__(self):
        self.now, self.meter = 1800000000., None
        self.delivered = asyncio.Event()
    def time(self): return self.now
    def monotonic(self): return self.now
    def time_ns(self): return int(self.now*1e9)
    def advance(self, duration):
        self.now += duration
        if self.meter is not None and not self.meter.closed:
            row = dict(timestamp=self.now, gpu=0, gpu_uuid='GPU-profile', power_w=100.,
                       frequency_mhz=self.meter.frequency, source='nvml:field:186:scope:0:mW')
            self.meter.readings.append(row)
            self.meter.journal('dynamo_power', **row)


def worker_type(clock, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'src/pdblend_runtime/native_v1.py'
    tree = ast.parse(path.read_text())
    original = next(row for row in tree.body if isinstance(row, ast.ClassDef) and row.name == 'NativeWorker')
    names = {'_native_shape', '_native_begin', 'native_measurement_start',
             'native_measurement_samples', 'native_measurement_stop'}
    methods = [row for row in original.body if isinstance(row, ast.FunctionDef) and row.name in names]
    namespace = dict(time=clock)
    module = ast.Module(body=[ast.ClassDef(name='Worker', bases=[], keywords=[], body=methods, decorator_list=[])],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    class Event:
        def __init__(self, enable_timing): assert enable_timing
        def record(self): self.at = clock.now
        def synchronize(self): pass
        def elapsed_time(self, end): return (end.at-self.at)*1000
    torch = ModuleType('torch'); torch.cuda = SimpleNamespace(Event=Event)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    distributed = ModuleType('vllm.distributed')
    distributed.get_tp_group = distributed.get_pp_group = lambda: SimpleNamespace(rank_in_group=0)
    monkeypatch.setitem(sys.modules, 'vllm.distributed', distributed)
    return namespace['Worker']


@pytest.fixture
def hardware(monkeypatch):
    clock = Clock()
    worker_cls = worker_type(clock, monkeypatch)
    monkeypatch.setattr(profile_v1, 'time', clock)
    monkeypatch.setattr(profile_v1, 'model_identity', lambda _: dict(model='Qwen2.5-7B-Instruct', manifest_sha256='a'*64))
    monkeypatch.setenv('PDBLEND_SOURCE_SHA256', 'c'*64)
    monkeypatch.setenv('PDBLEND_IMAGE_ID', 'sha256:'+'b'*64)
    transitions = []

    class Meter:
        def __init__(self, gpus, journal):
            assert gpus == [0]
            self.uuids, self.readings, self.frequency = {0:'GPU-profile'}, [], 2520
            self.journal, self.closed = journal, False
            clock.meter = self
        def start(self): pass
        def clock(self, gpus, frequency):
            assert gpus == [0]
            self.frequency = frequency
            transitions.append(frequency)
        async def close(self): self.closed = True

    class TrackingTransport(V1Transport):
        generations = []
        async def json(self, iid, path, payload=None, **kwargs):
            if path == '/baseline/measurement/start':
                self.generations.append(self.instances[iid]['generation'])
            return await super().json(iid, path, payload, **kwargs)
        async def stream(self, iid, payload):
            async for event in super().stream(iid, payload):
                yield event
                clock.delivered.set()

    def forbidden_lifecycle(*args, **kwargs):
        raise AssertionError('external profile must not construct a model lifecycle')
    monkeypatch.setattr(profile_v1, 'GroupTelemetry', Meter)
    monkeypatch.setattr(profile_v1, 'SubprocessLifecycle', forbidden_lifecycle)
    monkeypatch.setattr(profile_v1, 'V1Transport', TrackingTransport)

    class Engine(_fixture.Engine):
        def __init__(self, identifier, scheduler):
            super().__init__(identifier, scheduler)
            self.scheduler.native_operation('control', dict(generation=2, role='mixed'))
            self.worker_generation = 2
            self.native_identity = dict(model_id='Qwen2.5-7B-Instruct', engine_revision='vllm-0.10.1.1',
                model_hash='d'*64, tokenizer_hash='e'*64, verification_receipt_sha256='f'*64,
                source_revision='c'*64, image_digest='sha256:'+'b'*64, gpu_uuids=['GPU-profile'])
            self.worker = worker_cls()
            self.worker.rank = 0
            self.worker.parallel_config = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
            self.worker.model_config = SimpleNamespace(enforce_eager=True)
            self.worker._native_scope = self.worker._native_system = None
            self.worker._native_forward_hooks, self.worker._native_pending = [], []
            self.worker._native_requests = {}
        async def collective_rpc(self, method, kwargs):
            if method.startswith('native_measurement_'):
                self.calls.append((method, dict(kwargs)))
                return [getattr(self.worker, method)(**kwargs)]
            return await super().collective_rpc(method, kwargs)
        async def generate(self, prompt, params, rid):
            self.calls.append(('generate', dict(request_id=rid, scope=self.worker._native_scope,
                system=self.worker._native_system, submitted_s=clock.now)))
            request = SimpleNamespace(request_id=rid, num_prompt_tokens=len(prompt['prompt_token_ids']), num_computed_tokens=0)
            self.scheduler.requests[rid] = request
            self.scheduler.running.append(request)
            tokens = []
            for index in range(params.max_tokens):
                self.scheduler.executing = rid
                scheduled = self.scheduler.schedule()
                scheduled.finished_req_ids = []
                scheduled.scheduled_new_reqs = [SimpleNamespace(req_id=rid, prompt_token_ids=prompt['prompt_token_ids'],
                    num_computed_tokens=0)] if index == 0 else []
                scheduled.scheduled_cached_reqs = SimpleNamespace(req_ids=[] if index == 0 else [rid],
                    num_computed_tokens=[] if index == 0 else [request.num_computed_tokens])
                shape = self.worker._native_shape(scheduled)
                pair = self.worker._native_begin(shape) if self.worker._native_scope else None
                clock.advance(.5 if index == 0 else .125)
                if pair:
                    pair[2].record(); self.worker._native_pending.append(pair)
                finished = index+1 == params.max_tokens
                self.scheduler.update_from_output(scheduled, rid, finished)
                tokens.append(index+100)
                yield SimpleNamespace(finished=finished, outputs=[SimpleNamespace(index=0, text='',
                    token_ids=list(tokens), finish_reason='length' if finished else None)])
                # Keep simulated GPU time synchronized with actual HTTP delivery;
                # loopback TCP buffering must not turn a 2s simulated request into
                # a zero-iteration observation at the CPU collector.
                await asyncio.wait_for(clock.delivered.wait(), 5)
                clock.delivered.clear()
            self.worker._native_requests.pop(rid, None)
    return SimpleNamespace(clock=clock, Engine=Engine, transitions=transitions, transport=TrackingTransport)


def arguments(tmp_path, url, *, resume=False):
    return SimpleNamespace(model=tmp_path/'model', tp=1, gpus=[0], out=tmp_path/'profile', resume=resume,
        base_port=17000, instance_id='resident-generation-two', existing_url=url,
        freqs=list(profile_v1.FREQUENCIES), inputs=[128], outputs=[16], batches=[1],
        settle=2., measure=5., label_corpus_root=None)


@pytest.mark.asyncio
async def test_resident_six_frequency_collector_has_real_warmup_separate_fit_and_immutable_resume(tmp_path, monkeypatch, hardware):
    async with _fixture.native_services(monkeypatch, ('resident',), hardware.Engine) as (engines, endpoints):
        engine = engines['resident']
        args = arguments(tmp_path, endpoints['resident'])
        result = await profile_v1.collect(args)
        assert len(result['points']) == 6 and result['coverage']['all_six_frequencies'], result['coverage']
        assert hardware.transitions == list(profile_v1.FREQUENCIES)
        assert set(hardware.transport.generations) == {2}
        assert sum(name == 'native_measurement_start' for name, _ in engine.calls) == 24
        assert sum(name == 'native_measurement_stop' for name, _ in engine.calls) == 1
        assert not engine.scheduler.requests and engine.worker._native_scope is None
        cells = [json.loads(path.read_text()) for path in sorted((args.out/'cells').glob('*.json'))]
        assert {cell['point']['frequency_mhz'] for cell in cells} == set(profile_v1.FREQUENCIES)
        for cell in cells:
            assert cell['identity']['system'] == 'dynamollm' and cell['identity']['model_id'] == 'Qwen2.5-7B-Instruct'
            assert sha(args.out/cell['capability_path']) == cell['capability_sha256']
            for relative in cell['artifacts']:
                if 'repeat' not in relative and 'holdout' not in relative: continue
                window = json.loads((args.out/relative).read_text())
                warm = {row['request_id'] for row in window['warmup_requests']}
                measured = {row['request_id'] for row in window['requests']}
                assert warm and measured and warm.isdisjoint(measured)
                assert window['settle_s'] >= 2 and window['finished_s']-window['started_s'] >= 5
                assert all(row['finished_s'] <= window['started_s'] for row in window['warmup_requests'])
                assert {rid for row in window['warmup_native']['ranks'][0]['samples'] for rid in row['request_ids']} <= warm
                assert {rid for row in window['native']['ranks'][0]['samples'] for rid in row['request_ids']} <= measured
        completion = json.loads((args.out/'completion.json').read_text())
        assert completion['complete'] and completion['status'] == 'passed' and not completion['formal_eligible']
        legacy_sha = sha(args.out/'capability.json')
        previous = {path.name: sha(path) for path in (args.out/'cells').glob('*.json')}
        engine.scheduler.native_operation('control', {'generation': 3})
        engine.worker_generation = 3
        resumed = await profile_v1.collect(arguments(tmp_path, endpoints['resident'], resume=True))
        assert len(resumed['points']) == 6
        assert sha(args.out/'capability.json') == legacy_sha
        assert previous == {path.name: sha(path) for path in (args.out/'cells').glob('*.json')}
        assert len(list((args.out/'capabilities').glob('*.json'))) == 2
        assert sum(name == 'native_measurement_start' for name, _ in engine.calls) == 24
        assert sum(name == 'native_measurement_stop' for name, _ in engine.calls) == 1
        assert hardware.transitions == list(profile_v1.FREQUENCIES)
        first = cells[0]
        (args.out/first['capability_path']).write_text('{}')
        with pytest.raises(ValueError, match='capability checksum'):
            profile_v1.completed_cells(args.out, first['identity'])


@pytest.mark.asyncio
@pytest.mark.parametrize('field,bad', [('gpu_uuids', ['GPU-other']), ('source_revision', 'wrong'), ('image_digest', 'wrong')])
async def test_rejected_external_identity_never_stops_or_clocks_foreign_instance(tmp_path, monkeypatch, hardware, field, bad):
    async with _fixture.native_services(monkeypatch, ('resident',), hardware.Engine) as (engines, endpoints):
        engine = engines['resident']
        engine.native_identity[field] = bad
        engine.worker._native_scope, engine.worker._native_system = 'runner', 'distserve'
        with pytest.raises(RuntimeError, match='differs'):
            await profile_v1.collect(arguments(tmp_path, endpoints['resident']))
        assert not any(name.startswith('native_measurement_') or name == 'generate' for name, _ in engine.calls)
        assert not hardware.transitions
        assert engine.worker._native_system == 'distserve' and engine.worker._native_scope == 'runner'
        assert not json.loads((tmp_path/'profile'/'completion.json').read_text())['complete']
