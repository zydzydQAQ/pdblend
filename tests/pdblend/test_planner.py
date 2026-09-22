import time

import pytest

from pdblend.control.forecast import Forecaster
from pdblend.control.planner import SLO, PlannerConfig, PoolPlanner, assign_roles
from synthetic import fc, synthetic_model

@pytest.fixture
def planner():
    return PoolPlanner(synthetic_model(), PlannerConfig(slots=8, slo=SLO(5.0, 0.15)))


def test_low_load_parks_and_lowers_clock(planner):
    plan = planner.plan(fc(0.5))
    assert plan.active() < 8
    assert plan.counts.get("off", 0) + plan.counts.get("L1", 0) == 8 - plan.active()
    assert plan.ttft_s <= 5.0 * 0.85 and plan.tpot_s <= 0.15 * 0.85


def test_power_monotone_in_rate(planner):
    powers = [planner.plan(fc(r)).power_w for r in (0.5, 2, 5, 10)]
    assert powers == sorted(powers)


def test_tight_slo_raises_frequency():
    model = synthetic_model()
    loose = PoolPlanner(model, PlannerConfig(8, SLO(5.0, 0.15))).plan(fc(6))
    tight = PoolPlanner(model, PlannerConfig(8, SLO(5.0, 0.05))).plan(fc(6))
    assert tight.power_w >= loose.power_w
    assert max(tight.f_D, tight.f_M) >= max(loose.f_D, loose.f_M)


def test_fixed_mixed_baseline_only_uses_M():
    cfg = PlannerConfig(8, SLO(5.0, 0.15), fixed_mixed=True, allow_dvfs=False)
    plan = PoolPlanner(synthetic_model(), cfg).plan(fc(3))
    assert plan.counts == {"M": 8} and plan.f_M == 2520


def test_hysteresis_keeps_current_when_saving_small(planner):
    first = planner.plan(fc(3.0))
    again = planner.plan(fc(3.05), current=first)
    assert again.key() == first.key()


def test_overload_falls_back_to_full_capacity(planner):
    plan = planner.plan(fc(500))
    assert plan.detail.get("fallback") and plan.active() == 8 and plan.f_M == 2520


def test_long_inputs_can_prefer_pd():
    model = synthetic_model()
    p = PoolPlanner(model, PlannerConfig(8, SLO(15.0, 0.2)))
    inputs = [7000] * 50
    plan = p.plan(fc(1.5, in_mean=7000, out_mean=100, inputs=inputs))
    p_no_pd = PoolPlanner(model, PlannerConfig(8, SLO(15.0, 0.2), allow_pd=False))
    alt = p_no_pd.plan(fc(1.5, in_mean=7000, out_mean=100, inputs=inputs))
    assert plan.power_w <= alt.power_w


def test_assign_roles_minimises_changes():
    current = {"i0": "M", "i1": "M", "i2": "M", "i3": "L1", "i4": "off"}
    out = assign_roles(current, {"P": 1, "D": 2, "M": 1, "L1": 1}, load={"i0": 5, "i1": 1, "i2": 3})
    assert sorted(out.values()) == sorted(["P", "D", "D", "M", "L1"])
    assert out["i3"] == "L1"            # stays asleep
    assert out["i4"] in ("P", "D")      # off slot woken only because 4 active are needed
    assert sum(1 for i in ("i0", "i1", "i2") if out[i] == "M") == 1


def test_forecaster_rate_and_lengths():
    f = Forecaster(short_s=5, long_s=20, window_s=60)
    t0 = 1000.0
    for i in range(200):
        f.arrive(256 + (i % 10) * 100, now=t0 + i * 0.1)   # 10 rps
    for i in range(50):
        f.finish(100, now=t0 + 20 + i * 0.01)
    est = f.forecast(now=t0 + 20.5)
    assert 7 < est.rate_rps < 13
    assert 600 < est.input_mean < 800
    assert est.output_mean == 100
    share, hi, lo = est.split(1000)
    assert abs(share - 0.2) < 1e-9 and hi >= 1000 > lo


def test_pressure_controls_only_enable_pd_in_pressure_mode():
    model = synthetic_model()
    p = PoolPlanner(model, PlannerConfig(8, SLO(15.0, .2), min_m_instances=4,
                                        pressure_controls=True))
    demand = fc(5.0, in_mean=2048, out_mean=32, inputs=[512, 1024, 2048, 4096] * 12)
    m = p.plan(demand)
    assert m.counts.get("P", 0) == 0 and m.counts.get("M", 0) >= 4
    p.cfg.pd_pressure_active = True
    pd = p.plan(demand)
    assert pd.counts.get("P", 0) > 0 and pd.counts.get("D", 0) > 0
    assert pd.tau in (0, 1024)


def test_m_floor_does_not_forbid_pure_pd_escape_hatch():
    model = synthetic_model()
    p = PoolPlanner(model, PlannerConfig(8, SLO(15.0, .2), min_m_instances=4,
                                        pressure_controls=True))
    demand = fc(8.0, in_mean=7000, out_mean=32, inputs=[7000] * 50)
    p.cfg.pd_pressure_active = True
    plans = p.candidates(demand)
    assert any(x.counts.get('P', 0) and x.counts.get('D', 0) and not x.counts.get('M', 0)
               for x in plans)
