import copy
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile import calibration_repair as repair
from pdblend.profile.long_context_collect import collect_bounded_decode_point
from pdblend.profile.calibration import evaluate_holdout


class Model:
    system = 'pdblend'
    model = '/models/Qwen2.5-14B-Instruct'
    tp = pp = 1
    freqs = (900, 1200, 1500, 1800, 2100, 2520)

    def decode_supported(self, b, c, f):
        return f in self.freqs and b*c <= 42508.8
    def step_seconds(self, b, c, f):
        if not self.decode_supported(b, c, f): raise ValueError('outside coverage')
        return .1
    def decode_power_w(self, b, f, *, ctx=None): return 100.


def plan():
    return dict(system='pdblend', model_id='Qwen2.5-14B-Instruct', tp=1, pp=1,
        fit_existing_holdout=False, independent_holdout=True, kv_capacity_tokens=47232,
        points=[dict(freq_mhz=f, original=dict(batch=128, context_tokens=256),
            replacement=dict(batch=96, max_tokens=n), repeats=3, settle_s=2, measure_s=5)
            for f, n in zip(repair.REPAIR_FREQUENCIES, (180, 183, 185, 186))])


def test_repair_reservation_is_exactly_four_independent_bounded_points():
    points = repair.repair_points(plan(), Model())
    assert [p['max_tokens'] for p in points] == [180, 183, 185, 186]
    assert all(p['purpose'] == 'independent_holdout_repair' and p['batch'] == 96 for p in points)
    bad = plan(); bad['points'].append(copy.deepcopy(bad['points'][0]))
    with pytest.raises(ValueError, match='exactly'):
        repair.repair_points(bad, Model())
    bad = plan(); bad['points'][0]['replacement']['max_tokens'] = 512
    with pytest.raises(ValueError, match='reservation'):
        repair.repair_points(bad, Model())


def original_fixture(root):
    root.mkdir()
    source = dict(prefill=[], decode=[], mixed=[])
    for f in Model.freqs:
        ctx = 311 if f in (900, 1200) else 340
        repeats = []
        for i in range(3):
            name = f'{f}-{i}.json'; path = root/name
            path.write_text(json.dumps(dict(power=[[1, [100]], [6, [100]]], frequency=[[2, [f]]])))
            repeats.append(dict(effective_context_tokens=ctx, step_seconds=.1, power_w=100,
                steady_window_s=5, min_steps=50, power_samples=2, frequency_samples=1,
                samples_file=name, samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        source['decode'].append(dict(freq_mhz=f, batch=128, context_tokens=256,
            effective_context_tokens=ctx, step_seconds=.1, power_w=100, repeats=repeats))
    mixed = root/'mixed.json'; mixed.write_text('{}')
    source['mixed'] = [dict(valid=True, base_step_s=.1, alone_prefill_s=.1, probe_ttft_s=.2,
        samples_file='mixed.json', samples_sha256=hashlib.sha256(mixed.read_bytes()).hexdigest()) for _ in range(12)]
    manifest = dict(plan=dict(prefill=[], decode=[dict(freq_mhz=f, batch=128, context_tokens=256) for f in Model.freqs]))
    return source, manifest


class Clock:
    def __init__(self): self.now=1000.; self.live=[]; self.freq=1500
    def __call__(self): return self.now
    async def sleep(self, seconds):
        self.now += seconds
        for r in self.live:
            while r.token_times_s[-1]+.1 <= self.now+1e-8:
                r.token_times_s.append(r.token_times_s[-1]+.1)


class Sampler:
    error = None
    def __init__(self, clock): self.clock=clock
    def start(self): self.begin=self.clock()
    def stop(self):
        self.samples=[(self.begin+.5, [100]), (self.clock(), [100])]
        self.frequency_samples=[(self.begin+1, [self.clock.freq])]


@pytest.mark.asyncio
async def test_four_point_sampling_and_composite_audit_preserve_originals(tmp_path):
    original, manifest = original_fixture(tmp_path/'original')
    before = copy.deepcopy(original)
    before_files = {p.name: p.read_bytes() for p in (tmp_path/'original').iterdir()}
    clock, launches = Clock(), []
    out = tmp_path/'repair'; out.mkdir()
    profiler = SimpleNamespace(tp=1, out_dir=out, parallel_layout={'instances': ['tp1']},
        raw=dict(kv_capacity_tokens=47232, measurement_plan_sha256='a'*64),
        meter=SimpleNamespace(sampler=lambda gpus: Sampler(clock)))

    @asynccontextmanager
    async def background(p, client, point, tag):
        launches.append((point['freq_mhz'], point['max_tokens']))
        clock.live=[SimpleNamespace(token_times_s=[clock.now-1.6+i*.1 for i in range(17)], error=None)
                    for _ in range(point['batch'])]
        yield clock.live, []

    measured=[]
    for point in repair.repair_points(plan(), Model()):
        clock.freq=point['freq_mhz']
        row = await collect_bounded_decode_point(profiler, None, [0], point,
            purpose='independent_holdout_repair', _clock=clock, _sleep=clock.sleep, _background_factory=background)
        row['requires_per_window_evaluation']=True
        measured.append(row)
    assert len(launches)==12
    assert all(r['independent_holdout'] and r['evidence_class']=='independent_holdout_repair' for r in measured)
    view, expected, superseded = repair.compose_audit_view(original, dict(decode=measured), manifest, plan(), Model())
    audit = evaluate_holdout(view, Model(), out, expected_plan=expected,
                            evidence_roots={'original': tmp_path/'original', 'repair': out})
    assert audit['passed'], audit['failures']
    assert len(superseded)==4 and all(x['reason']=='outside_coverage' for x in superseded)
    assert original==before
    assert {p.name: p.read_bytes() for p in (tmp_path/'original').iterdir()}==before_files
    assert all(row['batch']==96 for row in expected['decode'] if row['freq_mhz'] in repair.REPAIR_FREQUENCIES)


def test_no_selective_replacement_of_in_domain_measurements(tmp_path):
    original, manifest = original_fixture(tmp_path/'original')
    for rep in original['decode'][2]['repeats']: rep['effective_context_tokens']=310
    measured=[dict(freq_mhz=f, batch=96, context_tokens=256) for f in repair.REPAIR_FREQUENCIES]
    with pytest.raises(ValueError, match='in-domain'):
        repair.compose_audit_view(original, dict(decode=measured), manifest, plan(), Model())


def test_unknown_evidence_root_cannot_fallback_to_another_archive(tmp_path):
    original, _ = original_fixture(tmp_path/'original')
    original['decode'][0]['evidence_source']='unregistered'
    with pytest.raises(ValueError, match='undeclared'):
        evaluate_holdout(original, Model(), tmp_path/'original', evidence_roots={'original': tmp_path/'original'})


def test_repair_cli_runtime_loads_once_without_prefill_or_mixed_grid(tmp_path, monkeypatch):
    from pdblend.profile import profiler as profiler_module, wave as wave_module
    from pdblend.engine import launcher, client as client_module

    original_dir = tmp_path/'original'
    original, manifest = original_fixture(original_dir)
    original.update(system='pdblend', model_id='Qwen2.5-14B-Instruct', tp=1, pp=1,
                    model_hash='m'*64, tokenizer_hash='t'*64)
    (original_dir/'raw.json').write_text(json.dumps(original))
    candidate_dir=tmp_path/'candidate'; candidate_dir.mkdir()
    (candidate_dir/'candidate.json').write_text('{}')
    candidate_hash=repair.digest(candidate_dir/'candidate.json')
    manifest.update(candidate_sha256=candidate_hash, training_raw_sha256='r'*64,
                    model_hash='m'*64, tokenizer_hash='t'*64)
    (candidate_dir/'manifest.json').write_text(json.dumps(manifest))
    (original_dir/'completion.json').write_text(json.dumps(dict(complete=True,
        candidate_sha256=candidate_hash, raw_sha256=repair.digest(original_dir/'raw.json'))))
    repaired_plan=plan(); repaired_plan.update(candidate_sha256=candidate_hash, training_raw_sha256='r'*64)
    plan_path=tmp_path/'repair-plan.json'; plan_path.write_text(json.dumps(repaired_plan))
    original_hashes={p.name:repair.digest(p) for p in original_dir.iterdir()}
    clock, events, launches = Clock(), [], []

    class Profiler:
        def __init__(self, model, gpus, **kw):
            self.tp=1; self.out_dir=kw['out_dir']; self.out_dir.mkdir()
            self.raw=dict(system='pdblend', model_id='Qwen2.5-14B-Instruct', tp=1, pp=1,
                          config={}, decode=[], prefill=[], mixed=[], static={}, transfer=[])
            self.model_spec=SimpleNamespace(model_hash='m'*64, tokenizer_hash='t'*64)
            self.specs=[SimpleNamespace(instance_id='p', base_url='http://unused', gpus=tuple(gpus))]
            self.parallel_layout={'instances':['p']}
            self.meter=SimpleNamespace(sampler=lambda g:Sampler(clock), reset_all=lambda:events.append('reset'))
        def _kv_capacity(self, inst): return 47232
        def _lock(self, f, gpus): clock.freq=f
        def _checkpoint(self): (self.out_dir/'raw.json').write_text(json.dumps(self.raw))
        def _prefill(self, *args): raise AssertionError('must not repeat prefill grid')
        def _mixed(self, *args): raise AssertionError('must not repeat mixed grid')

    class Fleet:
        def __init__(self, specs, logs):self.specs=specs
        def __enter__(self):return self
        def __exit__(self,*exc):events.append('stop')
        def start_all(self):events.append('load')
        def __getitem__(self,k):return SimpleNamespace(spec=self.specs[0])

    class Client:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*exc):pass

    class Wave:
        async def qualify_external(self, p, fleet):events.append('qualified')
        @asynccontextmanager
        async def measurement(self):yield
        def write(self,phase,value):events.append((phase,value))

    @asynccontextmanager
    async def background(profiler,client,point,tag):
        launches.append(point['freq_mhz'])
        clock.live=[SimpleNamespace(token_times_s=[clock.now-1.6+i*.1 for i in range(17)],error=None)
                    for _ in range(point['batch'])]
        yield clock.live,[]

    async def sample(*args,**kwargs):
        return await collect_bounded_decode_point(*args,**kwargs,_clock=clock,_sleep=clock.sleep,
                                                   _background_factory=background)

    monkeypatch.setattr(profiler_module,'Profiler',Profiler)
    monkeypatch.setattr(launcher,'Fleet',Fleet)
    monkeypatch.setattr(client_module,'EngineClient',Client)
    monkeypatch.setattr(wave_module.ProfileWave,'from_environment',lambda:Wave())
    monkeypatch.setattr(repair.PerfModel,'load',lambda p:Model())
    monkeypatch.setattr(repair,'collect_bounded_decode_point',sample)
    monkeypatch.setattr(repair.asyncio,'sleep',clock.sleep)
    result=repair.run(candidate_dir=candidate_dir,repair_plan=plan_path,original_holdout=original_dir,
        model_path=Model.model,gpus=[0],base_port=19000,out=tmp_path/'output')
    assert result['complete'] and result['calibration_passed'],result
    assert result['measured_decode_points']==4 and result['superseded_points']==4
    assert result['measured_prefill_grid_points']==result['measured_mixed_grid_points']==0
    assert len(launches)==12 and events.count('load')==events.count('stop')==1
    assert {p.name:repair.digest(p) for p in original_dir.iterdir()}==original_hashes
