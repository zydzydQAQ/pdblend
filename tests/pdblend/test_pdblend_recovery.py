"""Regression cases for overloaded PD capacity and interrupted recovery."""
import asyncio
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pdblend.online import controller as controller_module
from pdblend.online.shield import Pressure
from pdblend.planner.forecast import InFlightWork
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
from synthetic import fc, synthetic_model
from test_controller import make_controller


def deployment(counts, *, frequency=2520, tau=4096):
    return Plan(counts, frequency, frequency, frequency, tau, 1000., 1., .02,
                tp=1, pool_id="resident", generation=7, profile_key="bound")


def owned_work(rate=500):
    return replace(fc(rate, in_mean=6000, out_mean=20), backlog=(
        InFlightWork("pd", 6000, 20, waiting_prefill_tokens=6000, branch="PD"),
        InFlightWork("mixed", 512, 20, branch="M")))


def test_overload_never_reduces_existing_prefill_capacity_or_lowers_threshold():
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(15, .2),
        min_m_instances=4, preserve_overload_capacity=True))
    current = deployment({"P": 3, "D": 1, "M": 4})
    result = planner.plan(owned_work(), current)
    assert result.counts == current.counts and result.tau == 4096
    assert result.pool_id == "resident" and result.generation == 7
    assert result.detail["capacity_insufficient"]
    assert not result.detail["capacity_estimate_available"]
    assert result.detail["fallback_reason"] == "retain_uncertified_capacity"


def test_finite_fallback_can_use_spare_capacity_without_changing_owned_branches(monkeypatch):
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(15, .2),
        preserve_overload_capacity=True))
    current = deployment({"P": 2, "D": 1, "M": 4, "off": 1}, frequency=900)
    evaluated = []

    def evaluate(counts, fp, fd, fm, tau, demand, strict=True):
        evaluated.append((counts, tau))
        # The spare prefill instance has a finite estimate; the other layouts do not.
        if counts.get("P") == 3:
            return deployment(counts, tau=tau)
        return None

    monkeypatch.setattr(planner, "evaluate", evaluate)
    result = planner.fallback(owned_work(), current)
    assert result.counts == {"P": 3, "D": 1, "M": 4}
    assert all(c["P"] >= 2 and c["D"] >= 1 and c["M"] >= 4 and t == 4096
               for c, t in evaluated)
    assert result.detail["capacity_estimate_available"] and math.isfinite(result.ttft_s)


def test_baseline_fallback_remains_legacy_without_pdblend_opt_in():
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(15, .2), min_m_instances=4))
    result = planner.plan(owned_work(), deployment({"P": 3, "D": 1, "M": 4}))
    assert result.counts == {"P": 1, "D": 3, "M": 4} and result.tau == 1024


def test_specialized_legacy_fallback_keeps_its_one_argument_interface(monkeypatch):
    class Specialized(PoolPlanner):
        def fallback(self, demand):
            return deployment({"M": 8}, tau=0)
    planner = Specialized(synthetic_model(), PlannerConfig(8, SLO(15, .2)))
    monkeypatch.setattr(planner, "candidates", lambda demand: [])
    assert planner.plan(owned_work()).counts == {"M": 8}


def test_rescue_bypasses_hold_but_does_not_drop_ownership_or_accept_unknown_capacity():
    ctl, _, _ = make_controller(n=8)
    ctl.safety_recovery = True
    ctl.min_plan_hold_s, ctl.down_plan_votes = 30, 2
    ctl.plan_now = deployment({"P": 1, "D": 3, "M": 4})
    ctl._last_plan_change_s = 100
    rescue = deployment({"P": 3, "D": 1, "M": 4})
    pressure = Pressure(prefill=True)
    result, reason = ctl._gate_plan_change(rescue, 101, pressure=pressure, forecast=owned_work())
    assert result is rescue and reason == "safety_recovery"
    invalid = replace(rescue, counts={"P": 7, "D": 1}, ttft_s=float("inf"))
    result, reason = ctl._gate_plan_change(invalid, 101, pressure=pressure, forecast=owned_work())
    assert result is ctl.plan_now and reason == "minimum_hold"
    # The same redistribution cannot evade dwell based only on a cheaper prediction.
    assert ctl._gate_plan_change(rescue, 101, forecast=owned_work())[1] == "minimum_hold"


def test_all_m_capacity_increase_bypasses_hold_only_for_corrected_pdblend():
    ctl, _, _ = make_controller(n=8)
    ctl.plan_now = deployment({"M": 4, "off": 4}, frequency=900, tau=0)
    ctl.min_plan_hold_s, ctl.down_plan_votes, ctl._last_plan_change_s = 30, 2, 100
    increase = deployment({"M": 8}, tau=0)
    assert ctl._gate_plan_change(increase, 101)[1] == "minimum_hold"
    ctl.safety_recovery = True
    assert ctl._gate_plan_change(increase, 101) == (increase, "safety_recovery")


def test_repeated_shield_escalations_keep_rescue_and_periodic_planning(monkeypatch):
    ctl, _, _ = make_controller(n=8)
    ctl.safety_recovery = True
    ctl.min_plan_hold_s, ctl.down_plan_votes = 30, 2
    ctl.period_s, ctl.tick_s = 10., 1.
    initial = deployment({"P": 1, "D": 3, "M": 4})
    rescue = deployment({"P": 3, "D": 1, "M": 4})
    ctl.initial_plan = initial
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(controller_module, "time", SimpleNamespace(time=lambda: clock.now))
    scheduled_ticks, executed = [], []
    gate = ctl._gate_plan_change

    def tracked_gate(plan, now, **kwargs):
        if kwargs["scheduled"]:
            scheduled_ticks.append(now)
        return gate(plan, now, **kwargs)

    ctl._gate_plan_change = tracked_gate
    ctl._informed = lambda *args: True
    ctl.forecaster = SimpleNamespace(set_backlog=lambda items: None, forecast=lambda now: owned_work())
    ctl.planner.plan = lambda *args: rescue

    class Escalating:
        floor_active = level = 0
        events = []
        def observe(self, records, now): return Pressure(prefill=True)
        def needs_collective_clock_probe(self, pressure, now): return False
        def update(self, pressure, now):
            self.level += 1
            return self.level
        def apply(self, plan, pressure, frequency): return plan
        def protection_active(self, now): return False

    ctl.shield = Escalating()

    async def execute(plan):
        executed.append(plan)
        ctl.plan_now, ctl._last_plan_change_s = plan, clock.now

    ctl.execute = execute

    async def run():
        stop = asyncio.Event()
        async def tick(delay):
            clock.now += delay
            if clock.now >= 112:
                stop.set()
        async def inline(function, *args): return function(*args)
        monkeypatch.setattr(controller_module, "asyncio", SimpleNamespace(sleep=tick, to_thread=inline))
        await ctl.run(stop)

    asyncio.run(run())
    assert executed == [initial, rescue]
    assert scheduled_ticks == [110.]
    assert next(row for row in ctl._log if row["kind"] == "forecast")["decision_reason"] == "safety_recovery"


def test_unchanged_fallback_candidates_are_counted_without_reexecuting_plan(monkeypatch):
    from pdblend.bench.pdblend_runtime_options import runtime_receipt
    ctl, _, router = make_controller(n=8, shield=False)
    ctl.safety_recovery = True
    initial = deployment({'P': 2, 'D': 2, 'M': 4})
    initial.detail = dict(fallback=True, fallback_reason='initial_uncertified')
    ctl.initial_plan = initial
    ctl.period_s = ctl.tick_s = 1.
    ctl._informed = lambda *args: True
    ctl.forecaster = SimpleNamespace(set_backlog=lambda work: None, forecast=lambda now: owned_work())
    ctl.planner.plan = lambda *args: replace(initial, detail=dict(fallback=True,
        fallback_reason='retain_uncertified_capacity', capacity_estimate_available=False))
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(controller_module, 'time', SimpleNamespace(time=lambda: clock.now))
    executed = []
    async def execute(plan):
        executed.append(plan)
        ctl.plan_now, ctl._last_plan_change_s = plan, clock.now
        ctl.log('plan', counts=plan.counts, fallback=bool(plan.detail.get('fallback')), query_results=plan.detail)
    ctl.execute = execute
    async def run():
        stop = asyncio.Event()
        async def tick(delay):
            clock.now += delay
            if clock.now >= 102: stop.set()
        async def inline(function, *args): return function(*args)
        monkeypatch.setattr(controller_module, 'asyncio', SimpleNamespace(sleep=tick, to_thread=inline))
        await ctl.run(stop)
    asyncio.run(run())
    assert executed == [initial]
    forecasts = [row for row in ctl._log if row['kind'] == 'forecast']
    assert len(forecasts) == 2 and all(row['candidate_fallback'] for row in forecasts)
    assert all(row['candidate_capacity_estimate_available'] is False for row in forecasts)
    result = runtime_receipt(requested={}, enabled={}, controller=ctl, router=router, profile_key={})
    assert result['trigger_counts']['fallback_plans'] == 1
    assert result['trigger_counts']['fallback_candidates'] == 2
    assert result['trigger_counts']['capacity_preserving_fallback_candidates'] == 2
    assert result['fallback_candidate_reasons'] == {'retain_uncertified_capacity': 2}


def test_unknown_clock_is_really_reset_once_and_failed_reset_is_not_cached(monkeypatch):
    ctl, _, _ = make_controller(n=1, shield=False)
    calls = []
    fail = [True]

    async def hardware(iid, operation, action):
        calls.append(operation)
        if fail[0]:
            fail[0] = False
            raise RuntimeError("reset failed")
        action()

    monkeypatch.setattr(ctl, "_hardware", hardware)
    async def run():
        with pytest.raises(RuntimeError, match="reset failed"):
            await ctl._set_clock("i0", None)
        await ctl._set_clock("i0", None)
        await ctl._set_clock("i0", None)
    asyncio.run(run())
    assert calls == ["clock_reset", "clock_reset"]
    assert ctl.gpus.clocks == {0: None}
