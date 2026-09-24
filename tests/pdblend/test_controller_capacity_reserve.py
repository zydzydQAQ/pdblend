"""Control-loop regressions for exact capacity domains and tuning recovery."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pdblend.bench.low_m_tuning import TrialPlanner
from pdblend.online.shield import Pressure, Shield
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, QualifiedCapacityFloor, SLO
from synthetic import fc, synthetic_model
from test_controller import make_controller


def reserve_controller(kind):
    context = dict(algorithm_source_sha256='code', workload_family_sha256='workload',
                   recovery_policy_sha256='recovery')
    cfg = PlannerConfig(8, SLO(5., .15), freqs=(1800, 2100), min_m_instances=4,
                        capacity_floor_reserve_canonical=True, capacity_floor_context=context)
    frequency = 2100 if kind == 'v1' else 1800
    initial = Plan({'M':2, 'L1':6}, 2100, 2100, frequency, 0, 100., .1, .01)
    if kind == 'tuning':
        planner = TrialPlanner(synthetic_model(), cfg, dict(id='cpu', nominal_rate_rps=2.,
            plan=dict(counts=initial.counts, f_P=2100, f_D=2100, f_M=frequency, tau=0)))
        assert planner.cfg.capacity_floors == ()
    else:
        cfg.capacity_floors = (QualifiedCapacityFloor('synthetic', 1, 1, 2, (2., 2.5),
            (1., 4096.), (1., 512.), 'synthetic-evidence', qualified=True, profile_key='{}',
            accepted_slo=(5., .15), version=1 if kind == 'v1' else 2,
            frequency_mhz=None if kind == 'v1' else frequency,
            context={} if kind == 'v1' else context),)
        planner = PoolPlanner(synthetic_model(), cfg)
    ctl, _, _ = make_controller(n=8, shield=False)
    ctl.planner, ctl.max_freq, ctl.initial_plan = planner, 2100, initial
    ctl.period_s, ctl.tick_s = 5., 1.
    ctl.safety_recovery, ctl.hold_initial = True, True
    ctl._informed = lambda *args: True
    return ctl


def run_ticks(ctl, monkeypatch, *, seconds=15, demand=None):
    """Exercise the real decision loop with deterministic time, no GPU actions."""
    from pdblend.online import controller as module
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(time=lambda:clock.now))
    ctl.forecaster = SimpleNamespace(set_backlog=lambda work:None,
        forecast=lambda now:fc(demand(now) if demand else 2.1))
    executed = []

    async def execute(plan):
        executed.append(plan)
        ctl.plan_now, ctl._last_plan_change_s = plan, clock.now
    ctl.execute = execute

    async def run():
        stop = asyncio.Event()
        async def tick(delay):
            clock.now += delay
            if clock.now >= 100.+seconds:
                stop.set()
        async def inline(function, *args):
            return function(*args)
        monkeypatch.setattr(module, 'asyncio', SimpleNamespace(sleep=tick, to_thread=inline))
        await ctl.run(stop)
    asyncio.run(run())
    return executed


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
def test_supported_low_m_keeps_normal_periodic_planning(monkeypatch, kind):
    ctl = reserve_controller(kind)
    planned = []
    def plan(demand, current):
        planned.append(demand)
        return current
    ctl.planner.plan = plan
    executed = run_ticks(ctl, monkeypatch)
    observations = [row for row in ctl._log if row['kind'] == 'forecast']
    assert len(planned) == len(observations) == 3
    assert all(row['planner_invoked'] and row['decision_reason'] == 'unchanged'
               for row in observations)
    assert [row['t'] for row in observations] == [105., 110., 115.]
    assert len(executed) == 1 and executed[0].counts['M'] == 2
    assert executed[0].f_M == (2100 if kind == 'v1' else 1800)


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
def test_domain_exit_restores_before_period_and_during_hold(monkeypatch, kind):
    ctl = reserve_controller(kind)
    ctl.period_s, ctl.min_plan_hold_s = 60., 600.
    def forbidden(*args):
        raise AssertionError('domain restoration must precede ordinary planning')
    ctl.planner.plan = forbidden
    executed = run_ticks(ctl, monkeypatch, seconds=1, demand=lambda now:2.1 if now == 100 else 2.6)
    assert [plan.counts['M'] for plan in executed] == [2, 4]
    assert executed[1].f_M == 2100
    observation = next(row for row in ctl._log if row['kind'] == 'forecast')
    assert observation['decision_reason'] == 'capacity_floor_restore'
    assert observation['planner_invoked'] is False


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
def test_shield_receives_canonical_reserve_before_frequency_protection(monkeypatch, kind):
    ctl = reserve_controller(kind)
    ctl.period_s, ctl.min_plan_hold_s = 60., 600.
    ctl._informed = lambda *args:False
    shield = Shield(ctl.planner.cfg.slo, mode='budget_aware')
    shield.observe = lambda *args:Pressure(decode=True)
    def escalate(*args):
        shield.level = 1
        return 1
    shield.update = escalate
    actual_apply, applied = shield.apply, []
    def apply(plan, pressure, frequency):
        applied.append(plan)
        return actual_apply(plan, pressure, frequency)
    shield.apply, ctl.shield = apply, shield
    executed = run_ticks(ctl, monkeypatch, seconds=1)
    assert applied and applied[0].counts['M'] == 4 and applied[0].f_M == 2100
    assert [plan.counts['M'] for plan in executed] == [2, 4]
    assert executed[1].f_M == 2100


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
def test_initial_unqualified_clock_restores_before_execution(monkeypatch, kind):
    ctl = reserve_controller(kind)
    ctl.initial_plan = replace(ctl.initial_plan, f_M=900)
    ctl.period_s = 60.
    executed = run_ticks(ctl, monkeypatch, seconds=1)
    assert len(executed) == 1
    assert executed[0].counts['M'] == 4 and executed[0].f_M == 2100


def collective_controller(kind, instances=2, frequency=2100):
    ctl = reserve_controller(kind)
    ctl.initial_plan = replace(ctl.initial_plan, counts={'M':instances, 'L1':8-instances}, f_M=frequency)
    if kind == 'tuning':
        ctl.planner.trial['plan'].update(counts=ctl.initial_plan.counts, f_M=frequency)
        ctl.planner.minimum, ctl.planner.frequency = instances, frequency
    else:
        ctl.planner.cfg.capacity_floors = tuple(replace(floor, min_m_instances=instances,
            frequency_mhz=frequency if floor.version == 2 else None)
            for floor in ctl.planner.cfg.capacity_floors)
    ctl.period_s, ctl.min_plan_hold_s = 5., 600.
    ctl.planner.plan = lambda demand, current: current
    ctl.shield = Shield(ctl.planner.cfg.slo, mode='budget_aware')
    return ctl


def short_collective():
    return Pressure(decode=True, decode_paths=('M',), decode_stalled=2,
                    decode_stalled_fraction=.5, longest_token_gap_s=.2, tpot_p90=.025,
                    mode='budget_aware')


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
@pytest.mark.parametrize('instances', [2, 3])
def test_collective_clock_only_episode_at_maximum_never_adds_reserved_m(monkeypatch, kind, instances):
    ctl = collective_controller(kind, instances)
    # A quiet scheduled tick within the level-1 episode must retain its origin;
    # testing only the currently observed gap would restore M4 on the next tick.
    ctl.shield.observe = lambda records, now:short_collective() if now == 101. else Pressure()
    executed = run_ticks(ctl, monkeypatch, seconds=6)
    assert [plan.counts['M'] for plan in executed] == [instances]
    assert ctl.plan_now.f_M == 2100 and ctl.shield.level == 1
    assert ctl.shield.capacity_target_active == ctl.shield.floor_active == 0


@pytest.mark.parametrize('kind', ['v1', 'v2', 'tuning'])
@pytest.mark.parametrize('risk', ['sustained', 'deadline', 'capacity_target', 'capacity_floor'])
def test_collective_exemption_never_bypasses_real_recovery(monkeypatch, kind, risk):
    ctl = collective_controller(kind)
    ctl.period_s = 1.
    if risk == 'deadline': ctl._deadline_floor_mhz = 2100
    if risk == 'capacity_target': ctl.shield.capacity_target_active = 3
    if risk == 'capacity_floor': ctl.shield.floor_active = 3
    def observe(records, now):
        return (Pressure(decode=True, decode_paths=('M',), decode_sustained_stalls=1,
                         longest_token_gap_s=1.2, mode='budget_aware')
                if risk == 'sustained' and now >= 102. else short_collective())
    ctl.shield.observe = observe
    executed = run_ticks(ctl, monkeypatch, seconds=2)
    assert executed[-1].counts['M'] >= 4 and executed[-1].f_M == 2100


@pytest.mark.parametrize('kind', ['v2', 'tuning'])
def test_collective_clock_change_outside_exact_frequency_still_restores_m4(monkeypatch, kind):
    ctl = collective_controller(kind, frequency=1800)
    ctl.shield.observe = lambda *args:short_collective()
    executed = run_ticks(ctl, monkeypatch, seconds=1)
    assert [plan.counts['M'] for plan in executed] == [2, 4]
    assert executed[-1].f_M == 2100
