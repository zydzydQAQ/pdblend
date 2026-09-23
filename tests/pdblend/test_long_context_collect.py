import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from pdblend.profile import long_context_collect as collector
from pdblend.profile.long_context_plan import FREQUENCIES


def plan_and_source():
    source = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=2, pp=1,
                  kv_capacity_tokens=400000, model_hash='m'*64, tokenizer_hash='t'*64)
    point = dict(batch=4, context_tokens=7168, max_tokens=1024, repeats=3,
                 settle_s=2, measure_s=5, purpose='training_extension')
    plan = dict(system='pdblend', model_id=source['model_id'], tp=2, pp=1,
                fit_existing_holdout=False,
                training=[dict(point, freq_mhz=f) for f in FREQUENCIES],
                holdout=[dict(point, freq_mhz=1500, context_tokens=6144, purpose='independent_holdout_after_candidate_freeze')])
    return plan, source


def test_plan_reads_training_only_and_never_merges_holdout():
    plan, source = plan_and_source()
    points = collector.validate_training_plan(plan, source)
    assert len(points) == 6 and all(p['context_tokens'] == 7168 for p in points)
    plan['training'][0]['purpose'] = 'independent_holdout_after_candidate_freeze'
    with pytest.raises(ValueError, match='holdout'):
        collector.validate_training_plan(plan, source)


def test_plan_rejects_identity_mismatch_and_oversized_context_tail():
    plan, source = plan_and_source()
    bad_source = dict(source, system='distserve')
    with pytest.raises(ValueError, match='PDBlend'):
        collector.validate_training_plan(plan, bad_source)
    plan['training'][0]['max_tokens'] = 1025
    with pytest.raises(ValueError, match='shape'):
        collector.validate_training_plan(plan, source)
    assert collector.point_capacity_error(dict(batch=4, context_tokens=7168, max_tokens=1024), 33000)


class Clock:
    def __init__(self):
        self.now = 1000.
        self.live = []
        self.frequency = 1500

    def __call__(self): return self.now

    async def sleep(self, seconds):
        self.now += seconds
        for r in self.live:
            while r.token_times_s[-1] + .1 <= self.now + 1e-8:
                r.token_times_s.append(r.token_times_s[-1] + .1)

    def background(self, enters):
        @asynccontextmanager
        async def run(profiler, client, point, tag):
            enters.append(tag)
            self.live = [SimpleNamespace(token_times_s=[self.now-1.6+i*.1 for i in range(17)], error=None)
                         for _ in range(point['batch'])]
            yield self.live, []
        return run


class Sampler:
    error = None

    def __init__(self, clock): self.clock = clock
    def start(self): self.begin = self.clock()
    def stop(self):
        self.samples = [(self.begin+.5, [100, 100]), (self.clock(), [100, 100])]
        self.frequency_samples = [(self.begin+1, [self.clock.frequency, self.clock.frequency])]


def profiler(root, clock):
    return SimpleNamespace(tp=2, out_dir=root, parallel_layout={'instances': ['tp2']},
                           raw=dict(kv_capacity_tokens=400000, training_plan_sha256='p'*64),
                           meter=SimpleNamespace(sampler=lambda gpus: Sampler(clock)))


@pytest.mark.asyncio
async def test_training_windows_have_actual_context_and_resume_only_missing_repeats(tmp_path):
    plan, _ = plan_and_source()
    point = plan['training'][0]
    clock, enters, checkpoint = Clock(), [], []
    p = profiler(tmp_path, clock)

    def interrupted(rows):
        checkpoint[:] = rows
        raise RuntimeError('simulate interruption after durable first window')

    with pytest.raises(RuntimeError, match='interruption'):
        await collector.collect_training_point(p, None, [0, 1], point, on_window=interrupted,
            _clock=clock, _sleep=clock.sleep, _background_factory=clock.background(enters))
    assert len(checkpoint) == 1
    row = await collector.collect_training_point(p, None, [0, 1], point, previous=checkpoint,
        _clock=clock, _sleep=clock.sleep, _background_factory=clock.background(enters))
    assert len(enters) == 3 and len(row['repeats']) == 3
    assert all(r['effective_context_tokens'] > 7168 for r in row['repeats'])
    assert row['max_tokens'] == 1024
    assert row['independent_holdout'] is False
    for rep in row['repeats']:
        collector.validate_repeat(tmp_path, rep, expected_point=point, expected_plan_sha256='p'*64)
    checkpoint_raw = dict(decode=[row], decode_pending={}, training_plan_sha256='p'*64)
    assert collector.resume_points(checkpoint_raw, tmp_path, [point]) == {collector.point_key(point)}
    file = tmp_path / row['repeats'][0]['samples_file']
    file.write_text('{}')
    with pytest.raises(ValueError, match='corrupted'):
        collector.resume_points(checkpoint_raw, tmp_path, [point])


def test_cli_run_keeps_one_engine_resident_and_executes_only_training(tmp_path, monkeypatch):
    from pdblend.profile import profiler as profiler_module, wave as wave_module
    from pdblend.engine import launcher, client as client_module

    plan, source = plan_and_source()
    source_path = tmp_path/'source.json'; source_path.write_text(json.dumps(source))
    plan.update(training_source=str(source_path), training_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    output = tmp_path/'out'
    clock, enters, events = Clock(), [], []

    class FakeProfiler:
        def __init__(self, model, gpus, **kw):
            self.out_dir = kw['out_dir']; self.out_dir.mkdir()
            self.tp = 2
            self.raw = dict(source, config={}, prefill=[], decode=[], mixed=[], transfer=[], static={})
            self.model_spec = SimpleNamespace(model_id=source['model_id'], model_hash=source['model_hash'], tokenizer_hash=source['tokenizer_hash'])
            self.specs = [SimpleNamespace(instance_id='p', gpus=tuple(gpus), base_url='http://unused')]
            self.parallel_layout = {'instances': ['p']}
            self.meter = SimpleNamespace(sampler=lambda gpus: Sampler(clock), reset_all=lambda: events.append('reset'))
        def _kv_capacity(self, instance): return 400000
        def _lock(self, f, gpus): clock.frequency = f; events.append(('frequency', f))
        def _checkpoint(self): (self.out_dir/'raw.json').write_text(json.dumps(self.raw))

    class Fleet:
        def __init__(self, specs, logs): self.specs = specs
        def __enter__(self): return self
        def __exit__(self, *exc): events.append('stop')
        def start_all(self): events.append('load')
        def __getitem__(self, key): return SimpleNamespace(spec=self.specs[0])

    class Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): pass

    class Wave:
        async def qualify_external(self, profiler, fleet): events.append('qualify')
        @asynccontextmanager
        async def measurement(self):
            events.append('measurement'); yield
        def write(self, phase, value): events.append((phase, value))

    original_collect = collector.collect_training_point

    async def fake_collect(*a, **kw):
        return await original_collect(*a, **kw, _clock=clock, _sleep=clock.sleep,
                                      _background_factory=clock.background(enters))

    monkeypatch.setattr(profiler_module, 'Profiler', FakeProfiler)
    monkeypatch.setattr(launcher, 'Fleet', Fleet)
    monkeypatch.setattr(client_module, 'EngineClient', Client)
    monkeypatch.setattr(wave_module.ProfileWave, 'from_environment', lambda: Wave())
    monkeypatch.setattr(collector, 'collect_training_point', fake_collect)
    monkeypatch.setattr(collector.asyncio, 'sleep', clock.sleep)
    result = collector.run(plan_path=plan_path, training_raw_path=source_path,
        model_path='/models/Qwen2.5-7B-Instruct', gpus=[0, 1], base_port=19000, out=output)
    assert result['complete'], result
    assert events.count('load') == 1 and events.count('stop') == 1
    assert events.index('qualify') < events.index('measurement')
    assert len(enters) == 18
    assert result['holdout_points_consumed'] == 0 and result['fit_performed'] is False
    measured = json.loads((output/'raw.json').read_text())
    assert len(measured['decode']) == 6 and measured['decode_pending'] == {}
    assert all(row['context_tokens'] == 7168 for row in measured['decode'])
    assert not result['formal_eligible']
