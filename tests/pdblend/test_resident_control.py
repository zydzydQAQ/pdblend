import asyncio
from types import SimpleNamespace

import pytest

from pdblend.online.resident_control import ResidentCoordinator
from pdblend.planner.pool import Plan
from pdblend.planner.topology import ResidentAllocation
from synthetic import fc


class JointPlanner:
    models = {"left": object(), "right": object()}
    require_measured_energy = True
    def __init__(self):
        self.feedback = None
        self.error = None
        self.plans = {key: Plan({"M": 1}, 2520, 2520, 1800, 0, 100, 1, .01, pool_id=key)
                      for key in self.models}
    def observe_dispatches(self, counts):
        self.feedback = counts
    def plan(self, forecast, **kwargs):
        if self.error:
            raise ValueError(self.error)
        return ResidentAllocation({"left": .25, "right": .75}, self.plans, 200,
                                  dict(query_qualified=True, measured_energy_required=True))


class FakeController:
    shield = None
    def __init__(self, pool, events):
        self.pool, self.events = pool, events
        self.roles = {pool: "M"}
        self.freqs = {pool: 2520}
        self.router = SimpleNamespace(loads={pool: SimpleNamespace(accepting=True, role="M")})
        self.plan_now = None
        self.logs = []
        self.fail = False
        self.leave_wrong_clock = False
    def log(self, kind, **row):
        self.logs.append(dict(kind=kind, **row))
    async def execute(self, plan):
        await asyncio.sleep(.001)
        if self.fail:
            raise RuntimeError("native readiness failed")
        if not self.leave_wrong_clock:
            self.freqs[self.pool] = plan.f_M
        self.plan_now = plan
        self.events.append(self.pool)


class FakeRouter:
    def __init__(self, events):
        self.events = events
        self.pools = {"left": object(), "right": object()}
        self.target_shares = {"left": .5, "right": .5}
    def dispatch_feedback(self, reset=False):
        return dict(counts={"left": 2, "right": 1}, total=3)
    def set_target_shares(self, shares):
        self.events.append("shares")
        self.target_shares = dict(shares)


def fixture():
    events = []
    planner = JointPlanner()
    router = FakeRouter(events)
    controllers = {pool: FakeController(pool, events) for pool in router.pools}
    return ResidentCoordinator(controllers, router, planner), events


def test_joint_share_publication_waits_for_both_ready_plans():
    coordinator, events = fixture()
    result = asyncio.run(coordinator.step(fc(1)))
    assert result is not None
    assert events[-1] == "shares" and set(events[:-1]) == {"left", "right"}
    assert coordinator.planner.feedback == {"left": 2, "right": 1}
    assert coordinator.router.target_shares == {"left": .25, "right": .75}


def test_missing_qualification_does_not_change_hardware_or_routing():
    coordinator, events = fixture()
    coordinator.planner.error = "missing_profile: absent Mixed energy windows"
    assert asyncio.run(coordinator.step(fc(1))) is None
    assert not events
    assert coordinator.router.target_shares == {"left": .5, "right": .5}
    assert coordinator.events[-1]["status"] == "missing_qualification"


@pytest.mark.parametrize("failure", ["fail", "leave_wrong_clock"])
def test_partial_failure_or_false_readiness_does_not_publish_and_blocks_retry(failure):
    coordinator, events = fixture()
    setattr(coordinator.controllers["right"], failure, True)
    assert asyncio.run(coordinator.step(fc(1))) is None
    assert "shares" not in events
    assert coordinator.failed_transition
    before = list(events)
    assert asyncio.run(coordinator.step(fc(1))) is None
    assert events == before
    assert coordinator.events[-1]["status"] == "blocked_after_failed_transition"


def test_shield_guard_preserves_fast_safety_decision():
    coordinator, events = fixture()
    coordinator.controllers["left"].shield = SimpleNamespace(level=1, floor_active=0)
    assert asyncio.run(coordinator.step(fc(1))) is None
    assert not events
    assert coordinator.events[-1]["status"] == "shield_guard"


def test_online_coordinator_refuses_exploratory_unqualified_planner():
    coordinator, _ = fixture()
    coordinator.planner.require_measured_energy = False
    with pytest.raises(ValueError, match="measured energy"):
        ResidentCoordinator(coordinator.controllers, coordinator.router, coordinator.planner)
