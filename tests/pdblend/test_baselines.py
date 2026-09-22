"""CPU tests for the ported baseline decision logic."""
import time

from pdblend2.control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from pdblend2.control.policies import get_policy
from pdblend2.control.policies.baselines import (DynamoPlanner, EcoPlanner, EcoRouter, capacity_rps, distserve_plan,
                                                 epoch_peaks)
from synthetic import fc, synthetic_model


def planner(slots=8, ttft=5.0, tpot=0.15):
    return PoolPlanner(synthetic_model(), PlannerConfig(slots=slots, slo=SLO(ttft, tpot)))


def test_epoch_peaks_uses_bin_maximum():
    arrivals = [i * 0.5 for i in range(240)] + [1800 + i * 0.1 for i in range(600)]
    peaks = epoch_peaks(arrivals, epoch_s=1800, bin_s=60)
    assert len(peaks) == 2 and abs(peaks[0] - 2.0) < 1e-9 and abs(peaks[1] - 10.0) < 1e-9


def test_distserve_picks_split_with_highest_capacity():
    p = planner(slots=4)
    plan = distserve_plan(p, fc(2.0))
    assert plan.counts["P"] + plan.counts["D"] == 4 and plan.counts.get("M", 0) == 0
    assert plan.f_P == plan.f_D == 2520 and plan.tau == 0
    caps = {n: capacity_rps(p, fc(2.0), {"P": n, "D": 4 - n}, 2520, 2520, 2520, 0) for n in (1, 2, 3)}
    assert abs(plan.detail["capacity_rps"] - max(caps.values())) < 1e-9


def test_dynamollm_sizes_fleet_from_epoch_peak_and_lowers_clock():
    base = planner(slots=8)
    one = capacity_rps(base, fc(1.0), {"M": 1}, 2520, 2520, 2520, 0)
    dyn = DynamoPlanner(base, peaks=[one * 2.5])
    plan = dyn.plan(fc(0.2))
    assert plan.counts["M"] == 3 and plan.counts["off"] == 5
    assert plan.f_M < 2520                     # ScaleFreq lowers the clock at light load
    hot = dyn.plan(fc(one * 2.4), plan)
    assert hot.counts["M"] == 3                # ScaleInst does not react inside an epoch
    assert hot.f_M >= plan.f_M
    over = dyn.plan(fc(one * 10), hot)
    assert over.counts["M"] == 3 and over.f_M == 2520 and over.detail.get("emergency")


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
