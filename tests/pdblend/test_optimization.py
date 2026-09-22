"""PDblend-only startup and stability behavior, including safety bypasses."""
import asyncio
import pytest

from pdblend.control.forecast import Forecast, Forecaster
from pdblend.control.planner import Plan
from pdblend.control.policies import get_policy
from pdblend.control.shield import Pressure
from synthetic import fc
from test_controller import make_controller


def plan(n, freq=2100):
    return Plan({"M": n, "L1": 4 - n}, 2520, 2520, freq, 0, 1, .1, .02)


def test_prior_waits_for_actual_traffic_and_does_not_fabricate_samples():
    f = Forecaster(initial=fc(8, out_mean=320))
    assert f.forecast(1000).samples == 0
    assert f.forecast(1100).rate_rps == 8
    for i in range(80):
        f.arrive(512, 1200 + i / 8)
    got = f.forecast(1210)
    assert got.rate_rps == pytest.approx(8)
    assert got.recent_rate_rps == pytest.approx(8)
    assert got.samples == 80 and got.completed_bins == 3


def test_short_completions_do_not_erase_prior_and_prior_expires():
    prior = Forecast(8, 0, 512, 768, 320, 0, (512,) * 50, (320,) * 10)
    f = Forecaster(initial=prior)
    f.arrive(512, 1000)
    for i in range(10):
        f.finish(20, 1001 + i / 10)
    early = f.forecast(1010)
    assert early.output_mean > 290 and 320 in early.outputs
    assert f.forecast(1120).output_mean == 20
    assert all(v == 20 for v in f.forecast(1120).outputs)


def test_prior_tracks_load_step_and_decays_during_idle():
    f = Forecaster(initial=fc(8))
    for i in range(1200):
        f.arrive(512, 1000 + i / 20)
    assert f.forecast(1060).rate_rps > 17
    assert f.forecast(1420).rate_rps < .01


def test_baselines_do_not_opt_into_adaptive_changes():
    for name in ("mixed", "distserve_static", "dynamollm", "ecoserve", "static_best"):
        p = get_policy(name)
        assert not p.bootstrap_forecast and p.plan_hold_s == 0 and p.down_plan_votes == 1
    baseline = Forecaster()
    for i in range(80):
        baseline.arrive(512, 1000 + i / 8)
    assert baseline.forecast(1010).rate_rps < 6
    assert get_policy("pdblend").plan_hold_s == 30


def controller():
    ctl, _, _ = make_controller()
    ctl.min_plan_hold_s, ctl.down_plan_votes = 30, 2
    ctl.plan_now, ctl._last_plan_change_s = plan(3), 1000
    return ctl


def test_hold_applies_to_expansion_and_clock_changes():
    ctl = controller()
    for candidate in (plan(4), plan(3, 2520), plan(2)):
        held, reason = ctl._gate_plan_change(candidate, 1029)
        assert held is ctl.plan_now and reason == "minimum_hold"
    accepted, reason = ctl._gate_plan_change(plan(4), 1030)
    assert accepted.active() == 4 and reason == "planner_change"


def test_downshift_votes_reset_and_only_count_scheduled_windows():
    ctl = controller()
    assert ctl._gate_plan_change(plan(2), 1030)[1] == "downshift_confirmation"
    assert ctl._gate_plan_change(plan(2), 1031, scheduled=False)[1] == "downshift_confirmation"
    assert ctl._gate_plan_change(plan(3), 1040)[1] == "unchanged"
    assert ctl._gate_plan_change(plan(2), 1050)[1] == "downshift_confirmation"
    assert ctl._gate_plan_change(plan(1), 1060)[1] == "downshift_confirmation"
    assert ctl._gate_plan_change(plan(1), 1070)[1] == "confirmed_downshift"


def test_clock_downshift_ignores_unused_PD_clocks():
    ctl = controller()
    candidate = plan(3, 1800)
    assert ctl._gate_plan_change(candidate, 1030)[1] == "downshift_confirmation"
    assert ctl._gate_plan_change(candidate, 1040)[1] == "confirmed_downshift"


def test_shield_escalates_during_hold_in_real_control_loop():
    ctl = controller()
    ctl.initial_plan = plan(2, 900)
    ctl.hold_initial = True
    hot = Pressure(decode=True, tpot_p90=.2)
    ctl.shield.observe = lambda *args: hot

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(ctl.run(stop))
        await asyncio.sleep(.08)
        stop.set()
        await task

    asyncio.run(run())
    assert ctl.plan_now.f_M == ctl.max_freq
    assert any(e.get("decision_reason") == "shield_override" for e in ctl._log)
