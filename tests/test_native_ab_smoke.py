import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from pdblend.bench import native_ab_smoke as smoke
from pdblend.bench.client import Request
from pdblend.planner.pool import Plan


def trace_file(tmp_path, *, split='evaluation', seed=701, duration=300):
    path = tmp_path / (split + '.json')
    rows = [asdict(Request(i, float(i), [101] * n, 128, 'development'))
            for i, n in enumerate((512, 1024, 2048))]
    path.write_text(json.dumps(dict(seed=seed, split=split, duration_s=duration, requests=rows)))
    return path


def test_tuning_uses_an_independent_seed_and_evaluation_keeps_701(tmp_path):
    assert len(smoke.read_trace(trace_file(tmp_path), 'evaluation', 300)) == 3
    tuning = trace_file(tmp_path, split='tuning', seed=9701, duration=60)
    assert len(smoke.read_trace(tuning, 'tuning', 60)) == 3
    with pytest.raises(ValueError, match='identity'):
        smoke.read_trace(trace_file(tmp_path, seed=9701), 'evaluation', 300)


@pytest.mark.parametrize('change', ['nan_time', 'duplicate_id', 'bad_shape', 'short_output'])
def test_trace_validation_rejects_unsupported_work(change, tmp_path):
    path = trace_file(tmp_path)
    data = json.loads(path.read_text())
    if change == 'nan_time': data['requests'][0]['arrival_s'] = float('nan')
    if change == 'duplicate_id': data['requests'][1]['idx'] = data['requests'][0]['idx']
    if change == 'bad_shape': data['requests'][0]['prompt'] = [101] * 7168
    if change == 'short_output': data['requests'][0]['max_tokens'] = 16
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='trace shape'):
        smoke.read_trace(path, 'evaluation', 300)


def test_interference_requires_both_latency_and_group_power_within_five_percent():
    solo = dict(median_latency_s=1., mean_power_w=200.)
    assert smoke.compare_probes(solo, dict(median_latency_s=1.04, mean_power_w=208.))['passed']
    assert not smoke.compare_probes(solo, dict(median_latency_s=1., mean_power_w=211.))['passed']
    assert not smoke.compare_probes(solo, dict(median_latency_s=1.06, mean_power_w=200.))['passed']
    with pytest.raises(ValueError):
        smoke.compare_probes(solo, dict(median_latency_s=float('nan'), mean_power_w=200.))


def test_failed_member_does_not_block_stage_and_receipt_is_atomic(tmp_path):
    smoke.write(tmp_path / 'cohort.json', dict(schema=1, cohort_id='test', members=['a', 'b'], qualification_limit=.05))
    first, failed = smoke.Cohort(tmp_path, 'a', .1), smoke.Cohort(tmp_path, 'b', .1)
    failed.publish(status='failed')
    rows = first.barrier('functional')
    assert rows['b']['status'] == 'failed'
    assert first.active_members() == ['a']
    assert not list(tmp_path.glob('*.tmp'))


def test_timed_planner_restores_method_on_exception_and_keeps_actual_duration(tmp_path, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('real planner failure')
    monkeypatch.setattr(smoke.PoolPlanner, 'plan', fail)
    path = tmp_path / 'planning-times.jsonl'
    with pytest.raises(RuntimeError):
        with smoke.timed_planning(path):
            smoke.PoolPlanner.plan(None)
    assert smoke.PoolPlanner.plan is fail
    receipt = json.loads(path.read_text())
    assert receipt['duration_s'] >= 0 and receipt['origin'] == 'periodic_controller'


def test_warmup_does_not_share_evaluation_request_ids():
    source = [Request(0, .01, [1]*512, 128), Request(1, .1, [1]*1024, 128)]
    warm = smoke.warmup_trace(source)
    assert {r.idx for r in warm}.isdisjoint({r.idx for r in source})
    assert warm[0].max_tokens == source[0].max_tokens
    assert source[0].idx == 0


def test_fixed_control_retains_shield_actions_without_calling_planner(monkeypatch):
    initial = Plan({'M': 2}, 900, 900, 900, 0, 200, 1, .1)
    safe = Plan({'M': 2}, 900, 900, 1500, 0, 300, 1, .1)
    stop, actions = asyncio.Event(), []
    class Shield:
        floor_active = 0
        def observe(self, *args): return object()
        def update(self, *args): return 1
        def apply(self, *args): return safe
    ctl = NS(initial_plan=initial, tick_s=0, shield=Shield(), max_freq=2520,
             forecaster=NS(set_backlog=lambda value: None), roles={'i0': 'M', 'i1': 'M'},
             router=NS(observation_records=lambda *args: []), log=lambda *args, **kwargs: None)
    async def execute(plan):
        ctl.plan_now = plan
        actions.append(plan)
        if len(actions) == 2: stop.set()
    ctl.execute = execute
    monkeypatch.setattr(smoke, 'backlog_snapshot', lambda router: ())
    monkeypatch.setattr(smoke.PoolPlanner, 'plan', lambda *args: pytest.fail('fixed A must not optimize'))
    asyncio.run(smoke.FixedPlanShieldController.run(ctl, stop))
    assert actions == [initial, safe]


def test_arm_controller_class_restored_after_failure():
    original = smoke.run_module.Controller
    with pytest.raises(RuntimeError):
        with smoke.arm_controller('A'):
            assert smoke.run_module.Controller is smoke.FixedPlanShieldController
            raise RuntimeError('abort arm')
    assert smoke.run_module.Controller is original


@pytest.mark.parametrize('functional_ok,interference_ok', [(True, True), (False, True), (True, False)])
def test_shared_fleet_a_b_and_functional_failure_isolation(tmp_path, monkeypatch, functional_ok, interference_ok):
    cohort_root = tmp_path / 'cohort'
    smoke.write(cohort_root / 'cohort.json', dict(schema=1, cohort_id='run', members=['7b'], qualification_limit=.05))
    cohort = smoke.Cohort(cohort_root, '7b', .1)
    calls = []
    class Instance:
        def __init__(self, spec, pid):
            self.spec, self.process, self.state = spec, NS(pid=pid), 'ready'
        def start(self): calls.append(('start', self.spec.instance_id))
        def wait_ready(self): return .01
        def alive(self): return True
    class Fleet:
        def __init__(self, specs, logs):
            self.instances = {s.instance_id: Instance(s, i+100) for i, s in enumerate(specs)}
        def __enter__(self): return self
        def __exit__(self, *args): calls.append(('close', len(self.instances)))
        def events(self): return []
    class Meter:
        def __init__(self, ids): self.gpus = ids
        def reset_all(self): pass
    async def functional(*args): return dict(functional_passed=functional_ok)
    async def empty(*args): return {}
    async def probe(*args):
        return dict(median_latency_s=1., mean_power_w=100. if interference_ok or args[3] == 'solo' else 110.)
    async def point(*args, **kwargs):
        calls.append(('point', kwargs['fixed_plan'], kwargs['initial_plan'], kwargs['observation_duration_s']))
        return dict(slo={}, controller={}, native_cleanup_complete=True)
    monkeypatch.setattr(smoke, 'Fleet', Fleet)
    monkeypatch.setattr(smoke, 'Gpus', Meter)
    monkeypatch.setattr(smoke, 'qualify', functional)
    monkeypatch.setattr(smoke, 'drain', empty)
    monkeypatch.setattr(smoke, 'reset_native', empty)
    monkeypatch.setattr(smoke, 'representative_probe', probe)
    monkeypatch.setattr(smoke, '_point', point)
    specs = [NS(instance_id=x) for x in ('i0', 'i1')]
    initial = Plan({'M': 2}, 1500, 1500, 1500, 0, 200, 1, .1)
    trace = [Request(0, .1, [101]*512, 128)]
    args = NS(out=tmp_path / 'out', base_port=18000, member='7b', pd_eight=False)
    bindings = dict(model_id='Qwen2.5-7B-Instruct', cohort_id='run', gpu_uuids='a,b')
    prepared = ([0, 1], NS(model=object()), object(), object(), initial, specs, trace, trace, cohort, bindings)
    result = smoke.run(args, prepared)
    assert result['complete'] is functional_ok
    points = [c for c in calls if c[0] == 'point']
    if functional_ok:
        assert len(points) == (2 if interference_ok else 4)
        assert points[0][1] is initial and points[1][1] is None
        assert all(p[2] is initial and p[3] == 300 for p in points)
        assert result['shared_lifecycle_pids'] == {'i0': 100, 'i1': 101}
        assert result['status'] == 'passed'
        if not interference_ok:
            assert (args.out / 'parallel-unqualified' / 'A' / 'summary.json').exists()
            assert (args.out / 'qualification' / 'serial-execution.json').exists()
            assert result['comparison_mode'] == 'serial_after_interference'
    else:
        assert not points and result['errors']
    assert len([c for c in calls if c[0] == 'start']) == 2
    assert result['formal_eligible'] is False and result['energy_comparable'] is False
    assert calls[-1] == ('close', 2)
