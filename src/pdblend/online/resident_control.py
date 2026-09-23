"""Guarded publication of a measured, joint allocation for existing resident pools."""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from pdblend.planner.forecast import Forecast
from pdblend.planner.pool import ACTIVE, Plan
from pdblend.planner.topology import ResidentAllocation, ResidentAllocationPlanner


@dataclass
class ResidentCoordinator:
    """Thin periodic step; the caller serializes this with child shield actions.

    Child Controller.run loops must not replan concurrently with this step.
    Shield observation/escalation remains the caller's fast loop. A hardware
    failure blocks further optimization until the caller reconciles native
    state and constructs a fresh coordinator; it does not guess a rollback.
    """
    controllers: Mapping[str, object]
    router: object
    planner: ResidentAllocationPlanner
    events: list[dict] = field(default_factory=list)
    failed_transition: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        expected = set(self.planner.models)
        if set(self.controllers) != expected or set(self.router.pools) != expected:
            raise ValueError("joint coordinator requires the exact explicit resident pool set")
        if not self.planner.require_measured_energy:
            raise ValueError("online joint coordination requires measured energy qualification")

    def _log(self, status: str, **data):
        row = dict(t=time.time(), status=status, **data)
        self.events.append(row)
        for controller in self.controllers.values():
            controller.log("resident_allocation", status=status, **data)

    @staticmethod
    def _ready(controller, plan: Plan) -> bool:
        actual = Counter(controller.roles.values())
        expected = {role: count for role, count in plan.counts.items() if count}
        if dict(actual) != expected:
            return False
        for iid, role in controller.roles.items():
            load = controller.router.loads[iid]
            if role in ACTIVE:
                if (not load.accepting or load.role != role
                        or controller.freqs.get(iid) != getattr(plan, f"f_{role}")):
                    return False
            elif load.accepting or load.role != "parked":
                return False
        return True

    async def step(self, forecast: Forecast, current_plans: Mapping[str, Plan] | None = None
                   ) -> ResidentAllocation | None:
        async with self._lock:
            if self.failed_transition:
                self._log("blocked_after_failed_transition")
                return None
            now = time.time()
            for pool, controller in self.controllers.items():
                shield = getattr(controller, "shield", None)
                if shield is not None and (shield.level or shield.floor_active or shield.protection_active(now)):
                    self._log("shield_guard", pool_id=pool)
                    return None
            current = dict(current_plans or {pool: c.plan_now for pool, c in self.controllers.items()
                                             if c.plan_now is not None})
            feedback = self.router.dispatch_feedback(reset=True)
            self.planner.observe_dispatches(feedback["counts"])
            shares = dict(self.router.target_shares)
            try:
                proposal = await asyncio.to_thread(self.planner.plan, forecast,
                                                   current_plans=current, current_shares=shares)
            except ValueError as exc:
                if "missing_profile:" not in str(exc) and "coverage" not in str(exc):
                    raise
                self._log("missing_qualification", reason=str(exc), dispatch_feedback=feedback)
                return None
            if not proposal.detail.get("query_qualified") or not proposal.detail.get("measured_energy_required"):
                self._log("rejected_unqualified_proposal")
                return None
            for pool, controller in self.controllers.items():
                plan = proposal.plans[pool]
                gate = getattr(controller, "_gate_plan_change", None)
                if gate is not None:
                    gated, reason = gate(plan, now, scheduled=True, shield_protected=False)
                    if gated.key() != plan.key():
                        self._log("controller_guard", pool_id=pool, reason=reason)
                        return None
            tasks = []
            changing = []
            for pool, controller in self.controllers.items():
                plan = proposal.plans[pool]
                if controller.plan_now is None or controller.plan_now.key() != plan.key():
                    changing.append(pool)
                    tasks.append(controller.execute(plan))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = {pool: repr(result) for pool, result in zip(changing, results)
                        if isinstance(result, BaseException)}
            for pool, controller in self.controllers.items():
                if pool not in failures and not self._ready(controller, proposal.plans[pool]):
                    failures[pool] = "role/clock/admission state differs from completed plan"
            if failures:
                self.failed_transition = True
                self._log("transition_failed", errors=failures, target_shares_unchanged=shares,
                          requested_shares=proposal.shares,
                          actual_roles={pool: dict(c.roles) for pool, c in self.controllers.items()})
                return None
            self.router.set_target_shares(proposal.shares)
            self._log("published", shares=proposal.shares, power_w=proposal.power_w,
                      plans={pool: dict(counts=plan.counts, tp=plan.tp, pp=plan.pp,
                                        frequencies=dict(P=plan.f_P, D=plan.f_D, M=plan.f_M))
                             for pool, plan in proposal.plans.items()},
                      dispatch_feedback=feedback, qualification=proposal.detail)
            return proposal
