"""CPU tests for the ported baseline decision logic."""
import time

from pdblend.control.forecast import Forecaster
from pdblend.control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from pdblend.control.policies import get_policy
from pdblend.control.policies.baselines import (DYNAMO_PERIODS, DynamoPlanner, EcoPlanner, EcoRouter, capacity_rps,
                                                 distserve_plan)
from synthetic import fc, synthetic_model


def planner(slots=8, ttft=5.0, tpot=0.15):
    return PoolPlanner(synthetic_model(), PlannerConfig(slots=slots, slo=SLO(ttft, tpot)))


def test_forecaster_tracks_completed_60s_bin_peak():
    f = Forecaster()
    t0 = 1000.0
    for i in range(120):                        # 2 rps over [t0, t0+60)
        f.arrive(512, t0 + i * 0.5)
    for i in range(300):                        # 5 rps over [t0+60, t0+120)
        f.arrive(512, t0 + 60 + i * 0.2)
    got = f.forecast(t0 + 130.0)                # third bin still partial: not counted
    assert got.completed_bins == 2 and abs(got.peak_rps - 5.0) < 1e-9


def test_distserve_picks_split_with_highest_capacity():
    p = planner(slots=4)
    plan = distserve_plan(p, fc(2.0))
    assert plan.counts["P"] + plan.counts["D"] == 4 and plan.counts.get("M", 0) == 0
    assert plan.f_P == plan.f_D == 2520 and plan.tau == 0
    caps = {n: capacity_rps(p, fc(2.0), {"P": n, "D": 4 - n}, 2520, 2520, 2520, 0) for n in (1, 2, 3)}
    assert abs(plan.detail["capacity_rps"] - max(caps.values())) < 1e-9


def test_dynamollm_fail_open_then_sizes_from_observed_peak():
    base = planner(slots=8)
    one = capacity_rps(base, fc(1.0), {"M": 1}, 2520, 2520, 2520, 0)
    dyn = DynamoPlanner(base)
    cold = dyn.plan(fc(0.2))
    assert cold.counts["M"] == 8 and "off" not in cold.counts   # no completed bins: fail-open, all on
    plan = dyn.plan(fc(0.2, peak_rps=one * 2.5, completed_bins=3))
    assert plan.counts["M"] == 3 and plan.counts["off"] == 5
    assert plan.f_M < 2520                     # ScaleFreq lowers the clock at light load
    grew = dyn.plan(fc(one * 2.4, peak_rps=one * 5.5, completed_bins=4))
    assert grew.counts["M"] == 6               # epoch 0 re-evaluates as the observed peak grows
    dyn.t0 -= DYNAMO_PERIODS["inst"]           # enter epoch 1: boundary fires once ...
    boundary = dyn.plan(fc(one * 2.4, peak_rps=one * 7.5, completed_bins=5))
    assert boundary.counts["M"] == 8
    same = dyn.plan(fc(one * 2.4, peak_rps=one * 3.5, completed_bins=5))
    assert same.counts["M"] == 8               # ... but not again inside the same epoch
    over = dyn.plan(fc(one * 10, peak_rps=one * 30, completed_bins=5))
    assert over.counts["M"] == 8 and over.f_M == 2520 and over.detail.get("emergency")


def test_eco_router_rotates_prefill_instance_when_budget_exhausted():
    model, slo = synthetic_model(), SLO(1.0, 0.1)
    r = EcoRouter(["a", "b", "c"], model, slo)
    first = r.dispatch("r0", 512, 64)
    assert first.decode_instance == r.groups[0][0]
    # A very long pending prompt exhausts the TTFT budget of the current prefill instance: rotate.
    r.dispatch("r1", 7168, 64)
    r.dispatch("r2", 7168, 64)
    third = r.dispatch("r3", 7168, 64)
    assert third.decode_instance != first.decode_instance
    r.set_roles({"a": "parked", "b": "parked"})
    assert r.choose(100)[1] == "c"


def test_eco_planner_scales_on_ttft_and_credits():
    base = planner(slots=4, ttft=1.0, tpot=0.1)
    router = EcoRouter(list("abcd"), base.model, base.cfg.slo)
    eco = EcoPlanner(base, router)
    cur = Plan({"M": 3, "idle": 1}, 2520, 2520, 2520, 0, 0, 0, 0)
    now = time.time()
    rec = router.dispatch("x", 256, 64)
    rec.submitted_s = now - 3.0
    router.first_token(rec, now - 0.5)          # TTFT 2.5 s > 1 s SLO
    up = eco.plan(fc(0.5), cur)
    assert up.counts["M"] == 4 and "idle" not in up.counts
    router.finish(rec, 64)
    rec2 = router.dispatch("y", 256, 64)
    rec2.submitted_s = now - 0.05
    router.first_token(rec2, now - 0.04)
    for _ in range(40):
        router.first_token(rec2, now)           # plenty of saved TPOT credit, TTFT fine
    router.records.clear()
    router.records.append(rec2)
    down = eco.plan(fc(0.5), Plan({"M": 4}, 2520, 2520, 2520, 0, 0, 0, 0))
    assert down.counts["M"] == 3 and down.counts["idle"] == 1
    assert down.f_M == 2520


def test_policy_table_marks_ported_baselines():
    for name in ("distserve_static", "dynamollm", "ecoserve"):
        assert get_policy(name).ported
    assert not get_policy("pdblend").ported
