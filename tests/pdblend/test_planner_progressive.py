"""Regression cases for conditional work, bounded allocation and transition economics."""
from dataclasses import replace

import pytest

from pdblend.planner.forecast import Forecast, Forecaster, InFlightWork
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, QualifiedCapacityFloor, SLO
from pdblend.planner.topology import ResidentAllocationPlanner, ResidentPool, Topology
from synthetic import fc, synthetic_model


def test_conditional_outputs_use_explicit_pairs_not_deque_position():
    demand = Forecast(2, 0, 2304, 4096, 528, 0, (512, 4096), (32, 1024),
                      length_pairs=((512, 32), (4096, 1024)))
    high, low = demand.split_forecasts(1024)
    assert (high.rate_rps, high.input_mean, high.output_mean, high.input_p95) == (1, 4096, 1024, 4096)
    assert (low.rate_rps, low.input_mean, low.output_mean, low.input_p95) == (1, 512, 32, 512)
    unpaired = replace(demand, length_pairs=())
    assert all(part.output_mean == 528 for part in unpaired.split_forecasts(1024))


def test_completed_pairs_survive_out_of_order_completion_and_ignore_failed_requests():
    forecaster = Forecaster(window_s=60)
    forecaster.arrive(4096, now=1, request_id="long")
    forecaster.arrive(512, now=2, request_id="short")
    forecaster.arrive(2048, now=3, request_id="failed")
    forecaster.finish(32, now=4, request_id="short")
    forecaster.finish(0, now=5, request_id="failed")
    forecaster.finish(1024, now=6, request_id="long")
    estimate = forecaster.forecast(now=7)
    assert estimate.length_pairs == ((512, 32), (4096, 1024))
    assert forecaster.pending_inputs == {}
    assert [part.output_mean for part in estimate.split_forecasts(1024)] == [1024, 32]


def test_bound_backlog_is_not_expired_or_reclassified_by_new_threshold():
    forecaster = Forecaster(window_s=60)
    forecaster.arrive(512, now=1, request_id="long")
    work = InFlightWork("long", 512, 10000, 0, 4096, "PD", "pool-a")
    forecaster.set_backlog([work])
    demand = forecaster.forecast(now=301)
    assert not demand.inputs and demand.backlog == (work,)
    pd, mixed = demand.split_forecasts(4096)
    assert pd.backlog == (work,) and not mixed.backlog
    assert demand.for_pool("pool-a", 0).remaining_decode_tokens == 10000
    assert not demand.for_pool("pool-b", 1).backlog


def test_planner_consumes_conditional_branch_outputs():
    demand = Forecast(.2, 0, 2304, 4096, 528, 0, (512, 4096), (32, 1024),
                      length_pairs=((512, 32), (4096, 1024)))
    planner = PoolPlanner(synthetic_model(), PlannerConfig(3, SLO(20, .5)))
    plan = planner.evaluate({"P": 1, "D": 1, "M": 1}, 2520, 2520, 2520, 1024, demand)
    assert plan is not None
    assert plan.detail["forecast"]["pd_output_mean"] == 1024
    assert plan.detail["forecast"]["mixed_output_mean"] == 32


def test_existing_decode_work_prevents_idle_forecast_from_shrinking_capacity():
    model = synthetic_model()
    planner = PoolPlanner(model, PlannerConfig(8, SLO(20, .5)))
    idle = fc(0)
    assert planner.evaluate({"M": 1, "L1": 7}, 2520, 2520, 2520, 0, idle) is not None
    work = tuple(InFlightWork(str(i), 512, 100, kv_tokens=5000, branch="M") for i in range(100))
    occupied = replace(idle, backlog=work, inflight=100)
    assert planner.evaluate({"M": 1, "L1": 7}, 2520, 2520, 2520, 0, occupied) is None
    assert planner.evaluate({"M": 8}, 2520, 2520, 2520, 0, occupied) is not None


def test_no_plan_drops_the_branch_of_an_active_request():
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(20, .5)))
    demand = replace(fc(.1), backlog=(InFlightWork("p", 4096, 50, branch="PD"),))
    assert planner.evaluate({"M": 8}, 2520, 2520, 2520, 0, demand) is None


def test_rank_all_candidates_after_transition_cost_and_retain_current(monkeypatch):
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(20, .5)))
    current = Plan({"M": 1, "L1": 7}, 2520, 2520, 2520, 0, 100, 1, .1)
    cold = Plan({"M": 8}, 2520, 2520, 2520, 0, 80, 1, .1)
    warm = Plan({"M": 1, "L1": 7}, 2520, 2520, 2100, 0, 90, 1, .1)
    monkeypatch.setattr(planner, "candidates", lambda demand: [cold, warm])
    monkeypatch.setattr(planner, "evaluate", lambda *args, **kwargs: replace(current, detail={}))
    planner.cfg.transition_estimator = lambda old, new: 2054 if new is cold else 6.6
    assert planner.plan(fc(.1), current).f_M == 2100
    planner.cfg.transition_estimator = lambda old, new: 10000
    retained = planner.plan(fc(.1), current)
    assert retained.key() == current.key()
    assert retained.detail["objective"]["retained_current"]


def test_memoization_preserves_every_candidate_and_reduces_role_queries(monkeypatch):
    model = synthetic_model()
    planner = PoolPlanner(model, PlannerConfig(5, SLO(20, .5)))
    counts = {"calls": 0}
    original = planner._mixed_pool
    def counted(*args):
        counts["calls"] += 1
        return original(*args)
    monkeypatch.setattr(planner, "_mixed_pool", counted)
    demand = replace(fc(1, inputs=[512, 4096] * 20), length_pairs=((512, 32), (4096, 1024)))
    cached = planner.candidates(demand)
    calls = counts["calls"]
    planner.cfg.memoize_roles = False
    counts["calls"] = 0
    uncached = planner.candidates(demand)
    assert cached == uncached
    assert calls < counts["calls"] / 2
    # Changing the same object's backlog must invalidate per-call memoization.
    planner.cfg.memoize_roles = True
    demand.backlog = (InFlightWork("oversize", 512, 100, kv_tokens=10000000, branch="M"),)
    assert not planner.candidates(demand)


def test_capacity_floor_requires_matching_qualified_domain():
    model = synthetic_model()
    qualified = QualifiedCapacityFloor("synthetic", 1, 1, 1, (0, 1), (1, 4096), (1, 256),
                                       "gpu-receipt.json", qualified=True)
    planner = PoolPlanner(model, PlannerConfig(4, SLO(20, .5), min_m_instances=4,
                                               capacity_floors=(replace(qualified, qualified=False),)))
    assert planner.mixed_floor(fc(.5)) == 4
    planner.cfg.capacity_floors = (qualified,)
    assert planner.mixed_floor(fc(.5)) == 1
    assert planner.mixed_floor(fc(2)) == 4
    planner.cfg.capacity_floors = (replace(qualified, tp=2),)
    assert planner.mixed_floor(fc(.5)) == 4


def _resident(*, require_measured_energy=False):
    pools = (ResidentPool("tp1", Topology(1, gpus=(0, 1)), 2),
             ResidentPool("tp2", Topology(2, gpus=(2, 3)), 1))
    models = {pool.pool_id: replace(synthetic_model(), tp=pool.topology.tp,
              bounded_coverage={"prefill_tokens": (1, 8192)}) for pool in pools}
    return ResidentAllocationPlanner(pools, models, PlannerConfig(4, SLO(20, .5), allow_pd=False),
                                     shares=(0, .5, 1), require_measured_energy=require_measured_energy,
                                     unused_static_power_w=40)


def test_outer_allocator_counts_every_resident_replica_and_static_power():
    planner = _resident()
    planner.observe_dispatches({"tp1": 2, "tp2": 1})
    proposal = planner.plan(fc(.5))
    assert sum(proposal.shares.values()) == 1
    assert proposal.power_w == pytest.approx(40 + sum(p.power_w for p in proposal.plans.values()))
    assert proposal.detail["resident_replicas"] == {"tp1": 2, "tp2": 1}
    assert proposal.detail["observed_shares"]["tp1"] == pytest.approx(2 / 3)
    assert all(sum(p.counts.values()) == planner.planners[pool].cfg.slots
               for pool, p in proposal.plans.items())
    assert not proposal.detail["formal_eligible"]
    assert not proposal.detail["native_tp_change"]


def test_outer_default_refuses_unmeasured_mixed_energy_and_unknown_backlog():
    planner = _resident(require_measured_energy=True)
    with pytest.raises(ValueError, match="missing_profile"):
        planner.plan(fc(.5))
    with pytest.raises(ValueError, match="pool ownership"):
        planner.plan(replace(fc(.5), backlog=(InFlightWork("r", 512, 100),)))


def test_outer_does_not_invent_energy_savings_to_override_current_margin():
    planner = _resident()
    proposal = planner.plan(fc(.5))
    for inner in planner.planners.values():
        inner.cfg.transition_estimator = lambda old, new: 100000
    retained = planner.plan(fc(.5), current_plans=proposal.plans, current_shares=proposal.shares)
    assert retained.detail["retained_current"]
    assert retained.shares == proposal.shares
    assert retained.detail["transition_energy_j"] == 0


def test_zero_arrival_branch_can_keep_serving_bound_backlog():
    planner = PoolPlanner(synthetic_model(), PlannerConfig(4, SLO(20, .5)))
    demand = replace(fc(.1), backlog=(InFlightWork("old", 4096, 1024, kv_tokens=4096, branch="PD"),))
    plan = planner.evaluate({"P": 1, "D": 1, "M": 2}, 2520, 2520, 2520, 1024, demand)
    assert plan is not None
    assert plan.detail["D"]["remaining_decode_tokens"] == 1024
    assert plan.detail["M"]["remaining_decode_tokens"] == 0


def test_overload_fallback_preserves_existing_pd_and_mixed_paths():
    planner = PoolPlanner(synthetic_model(), PlannerConfig(8, SLO(20, .5)))
    demand = replace(fc(500), backlog=(InFlightWork("pd", 4096, 1024, branch="PD"),
                                      InFlightWork("m", 512, 100, branch="M")))
    plan = planner.plan(demand)
    assert plan.detail["fallback"]
    assert all(plan.counts[role] > 0 for role in ("P", "D", "M"))
    assert plan.active() == 8


def test_correlated_bootstrap_prior_fades_without_pairing_independent_samples():
    prior = Forecast(1, 0, 512, 512, 1000, 0, (512,), (1000,), length_pairs=((512, 1000),))
    forecaster = Forecaster(window_s=100, initial=prior)
    forecaster.arrive(512, now=1, request_id="fast")
    forecaster.finish(10, now=2, request_id="fast")
    early = forecaster.forecast(now=11).split_forecasts(1024)[1]
    assert 850 < early.output_mean < 950
    assert set(early.length_pairs) == {(512, 1000), (512, 10)}
    # The original paired observations are retained, not the resampled mixture.
    expired = forecaster.forecast(now=101).split_forecasts(1024)[1]
    assert expired.output_mean == 10
    no_pairs = Forecaster(initial=replace(prior, length_pairs=()))
    no_pairs.arrive(512, now=1)
    no_pairs.finish(10, now=2)
    assert no_pairs.forecast(now=3).length_pairs == ()
