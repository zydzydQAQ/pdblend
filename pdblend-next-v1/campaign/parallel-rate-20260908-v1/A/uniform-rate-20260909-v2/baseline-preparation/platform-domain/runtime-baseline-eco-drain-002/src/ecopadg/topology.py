"""Transactional coordinator for slow spatial rematerialization.

Spatial mixed↔P/D changes are intentionally excluded from fast control.  This
state machine is advanced explicitly by a slow coordinator thread or tests:

STEADY -> DRAINING -> RESTARTING -> STARTING -> VALIDATING -> STEADY
                                                   |
                                                   v
                                              ROLLBACK -> STEADY/DEGRADED
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, List, Optional, Tuple

from ecopadg.pool_manager import (
    MODE_RESTART,
    PoolManager,
    RestartTransaction,
)
from ecopadg.types import Partition


class TopologyState(str, Enum):
    STEADY = "STEADY"
    DRAINING = "DRAINING"
    RESTARTING = "RESTARTING"
    STARTING = "STARTING"
    VALIDATING = "VALIDATING"
    ROLLBACK = "ROLLBACK"
    DEGRADED = "DEGRADED"


STEADY = TopologyState.STEADY
DRAINING = TopologyState.DRAINING
RESTARTING = TopologyState.RESTARTING
STARTING = TopologyState.STARTING
VALIDATING = TopologyState.VALIDATING
ROLLBACK = TopologyState.ROLLBACK
DEGRADED = TopologyState.DEGRADED


def partition_dict(partition: Optional[Partition]) -> Optional[dict]:
    if partition is None:
        return None
    return {
        "n_mixed": int(partition.n_mixed),
        "n_prefill": int(partition.n_prefill),
        "n_decode": int(partition.n_decode),
        "tp_mixed": int(partition.tp_mixed),
        "tp_prefill": int(partition.tp_prefill),
        "tp_decode": int(partition.tp_decode),
        "pp_mixed": int(partition.pp_mixed),
        "pp_prefill": int(partition.pp_prefill),
        "pp_decode": int(partition.pp_decode),
        "gpus_total": int(partition.gpus_total),
    }


@dataclass(frozen=True)
class TopologyDecision:
    """Result of a request or one state-machine advancement."""

    accepted: bool
    state: TopologyState
    reason: str
    target: Optional[Partition] = None
    transaction_id: int = 0
    committed: bool = False
    rolled_back: bool = False
    reasons: Tuple[str, ...] = ()

    @property
    def migrated(self) -> bool:
        return self.committed

    def to_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "state": self.state.value,
            "reason": self.reason,
            "reasons": list(self.reasons or (self.reason,)),
            "target": partition_dict(self.target),
            "transaction_id": int(self.transaction_id),
            "committed": bool(self.committed),
            "rolled_back": bool(self.rolled_back),
        }


@dataclass(frozen=True)
class TopologyStatus:
    """Serializable snapshot safe for controller status endpoints."""

    state: TopologyState
    current_partition: Partition
    target_partition: Optional[Partition]
    prior_partition: Optional[Partition]
    transaction_id: int
    state_since: float
    last_switch: float
    min_dwell_s: float
    role_switch_cost_j: float
    projected_savings_w: float
    error: str = ""
    abort_requested: bool = False
    history: Tuple[str, ...] = ()

    @property
    def steady(self) -> bool:
        return self.state == TopologyState.STEADY

    @property
    def blocks_requests(self) -> bool:
        return not self.steady

    @property
    def prior_partition_snapshot(self) -> Optional[Partition]:
        return self.prior_partition

    def to_dict(self) -> dict:
        cost = float(self.role_switch_cost_j)
        savings = float(self.projected_savings_w)
        return {
            "state": self.state.value,
            "steady": self.steady,
            "blocks_requests": self.blocks_requests,
            "current_partition": partition_dict(self.current_partition),
            "target_partition": partition_dict(self.target_partition),
            "prior_partition": partition_dict(self.prior_partition),
            "transaction_id": int(self.transaction_id),
            "state_since": float(self.state_since),
            "last_switch": float(self.last_switch),
            "min_dwell_s": float(self.min_dwell_s),
            "role_switch_cost_j": cost if math.isfinite(cost) else None,
            "projected_savings_w": (
                savings if math.isfinite(savings) else None
            ),
            "error": self.error,
            "abort_requested": bool(self.abort_requested),
            "history": list(self.history),
        }


Gate = Callable[[], bool]
PartitionHook = Callable[[Partition], None]


class TopologyCoordinator:
    """Single-flight, lock-protected physical topology transaction."""

    def __init__(
        self,
        pool_manager: PoolManager,
        initial_partition: Partition,
        *,
        min_dwell_s: float = 300.0,
        role_switch_cost_j: float = float("nan"),
        drain_timeout_s: float = 60.0,
        health_timeout_s: float = 600.0,
        rollback_timeout_s: Optional[float] = None,
        queue_empty: Optional[Gate] = None,
        slo_safe: Optional[Gate] = None,
        trusted: Optional[Gate] = None,
        canary_validator: Optional[Callable[[Partition], bool]] = None,
        on_commit: Optional[PartitionHook] = None,
        on_rollback: Optional[PartitionHook] = None,
        clock=time.time,
        last_switch: float = -1e30,
    ):
        if (
            not math.isfinite(float(min_dwell_s))
            or float(min_dwell_s) < 0.0
        ):
            raise ValueError("min_dwell_s must be finite and non-negative")
        if (
            not math.isfinite(float(drain_timeout_s))
            or float(drain_timeout_s) <= 0.0
        ):
            raise ValueError("drain_timeout_s must be positive")
        if (
            not math.isfinite(float(health_timeout_s))
            or float(health_timeout_s) <= 0.0
        ):
            raise ValueError("health_timeout_s must be positive")
        self.pm = pool_manager
        self.current_partition = replace(initial_partition)
        self.min_dwell_s = float(min_dwell_s)
        self.role_switch_cost_j = float(role_switch_cost_j)
        self.drain_timeout_s = float(drain_timeout_s)
        self.health_timeout_s = float(health_timeout_s)
        self.rollback_timeout_s = float(
            rollback_timeout_s
            if rollback_timeout_s is not None else health_timeout_s
        )
        if (
            not math.isfinite(self.rollback_timeout_s)
            or self.rollback_timeout_s <= 0.0
        ):
            raise ValueError("rollback_timeout_s must be positive")
        self._queue_empty = queue_empty or self.pm.queues_empty
        self._slo_safe = slo_safe or (lambda: True)
        self._trusted = trusted or (lambda: True)
        self._canary_validator = canary_validator or (lambda _target: True)
        self._on_commit = on_commit
        self._on_rollback = on_rollback
        self._clock = clock
        now = float(clock())
        self.topology_lock = threading.RLock()
        self.lock = self.topology_lock
        self._state = TopologyState.STEADY
        self._state_since = now
        self._last_switch = float(last_switch)
        self._target: Optional[Partition] = None
        self._prior: Optional[Partition] = None
        self._tx: Optional[RestartTransaction] = None
        self._transaction_id = 0
        self._projected_savings_w = float("nan")
        self._error = ""
        self._abort_requested = False
        self._history: List[str] = [TopologyState.STEADY.value]

    @property
    def state(self) -> TopologyState:
        with self.topology_lock:
            return self._state

    @property
    def is_steady(self) -> bool:
        return self.state == TopologyState.STEADY

    @property
    def blocks_requests(self) -> bool:
        return not self.is_steady

    def _set_state(self, state: TopologyState, now: float) -> None:
        self._state = state
        self._state_since = float(now)
        self._history.append(state.value)

    @staticmethod
    def _call_gate(gate: Gate) -> bool:
        try:
            return bool(gate())
        except Exception:
            return False

    def status(self) -> TopologyStatus:
        with self.topology_lock:
            return TopologyStatus(
                state=self._state,
                current_partition=replace(self.current_partition),
                target_partition=(
                    replace(self._target) if self._target is not None else None
                ),
                prior_partition=(
                    replace(self._prior) if self._prior is not None else None
                ),
                transaction_id=self._transaction_id,
                state_since=self._state_since,
                last_switch=self._last_switch,
                min_dwell_s=self.min_dwell_s,
                role_switch_cost_j=self.role_switch_cost_j,
                projected_savings_w=self._projected_savings_w,
                error=self._error,
                abort_requested=self._abort_requested,
                history=tuple(self._history),
            )

    def reconcile_steady(
        self,
        partition: Partition,
        *,
        last_switch: Optional[float] = None,
    ) -> None:
        """Reconcile intervening inplace park/unpark state.

        This operation is accepted only in STEADY and never while a physical
        transaction owns the topology.
        """
        with self.topology_lock:
            if self._state != TopologyState.STEADY or self._tx is not None:
                raise RuntimeError("cannot reconcile an active topology")
            self.current_partition = replace(partition)
            if last_switch is not None:
                self._last_switch = float(last_switch)

    def _decision(
        self,
        accepted: bool,
        reason: str,
        *,
        committed: bool = False,
        rolled_back: bool = False,
        reasons: Tuple[str, ...] = (),
        decision_target: Optional[Partition] = None,
    ) -> TopologyDecision:
        target = (
            decision_target
            if decision_target is not None else self._target
        )
        return TopologyDecision(
            accepted=accepted,
            state=self._state,
            reason=reason,
            reasons=reasons or (reason,),
            target=replace(target) if target is not None else None,
            transaction_id=self._transaction_id,
            committed=committed,
            rolled_back=rolled_back,
        )

    def request_transition(
        self,
        target: Partition,
        *,
        projected_savings_w: float = float("nan"),
        now: Optional[float] = None,
        queue_empty: Optional[bool] = None,
        slo_ok: Optional[bool] = None,
        trusted: Optional[bool] = None,
        reason: str = "requested",
    ) -> TopologyDecision:
        """Evaluate gates and enter DRAINING without changing current layout."""
        now_f = float(self._clock() if now is None else now)
        with self.topology_lock:
            if self._state != TopologyState.STEADY:
                return self._decision(False, "topology-busy")
            counts = (
                target.n_mixed, target.n_prefill, target.n_decode,
                target.tp_mixed, target.tp_prefill, target.tp_decode,
                target.pp_mixed, target.pp_prefill, target.pp_decode,
            )
            if (
                any(int(value) < 0 for value in counts[:3])
                or any(int(value) <= 0 for value in counts[3:])
                or target.total_gpus() <= 0
            ):
                return self._decision(False, "invalid-partition")
            if target.total_gpus() > len(self.pm.gpus):
                return self._decision(False, "gpu-budget")
            if target.n_prefill != target.n_decode:
                return self._decision(False, "pd-pairing-required")
            if not self.pm.all_ready():
                return self._decision(False, "pool-not-steady")
            steps = self.pm.plan_transition(target)
            if not steps:
                return self._decision(False, "already-steady")
            if not any(step.mode == MODE_RESTART for step in steps):
                return self._decision(False, "inplace-transition")
            if now_f - self._last_switch < self.min_dwell_s:
                return self._decision(False, "dwell")
            queue_ok = (
                self._call_gate(self._queue_empty)
                if queue_empty is None else bool(queue_empty)
            )
            if not queue_ok:
                return self._decision(False, "queue-not-empty")
            slo_gate = (
                self._call_gate(self._slo_safe)
                if slo_ok is None else bool(slo_ok)
            )
            if not slo_gate:
                return self._decision(False, "slo-unsafe")
            trust_gate = (
                self._call_gate(self._trusted)
                if trusted is None else bool(trusted)
            )
            if not trust_gate:
                return self._decision(False, "untrusted")
            cost = float(self.role_switch_cost_j)
            if not math.isfinite(cost) or cost < 0.0:
                return self._decision(False, "restart-cost-unknown")
            savings = float(projected_savings_w)
            if not math.isfinite(savings) or savings <= 0.0:
                return self._decision(False, "savings-unknown")
            if savings * self.min_dwell_s + 1e-12 < cost:
                return self._decision(False, "payback")
            try:
                tx = self.pm.begin_restart_transaction(target, steps)
            except Exception as exc:
                self._error = "prepare:%s" % type(exc).__name__
                return self._decision(False, "prepare-failed")
            self._transaction_id += 1
            self._target = replace(target)
            self._prior = replace(self.current_partition)
            self._tx = tx
            self._projected_savings_w = savings
            self._error = ""
            self._abort_requested = False
            self._set_state(TopologyState.DRAINING, now_f)
            return self._decision(
                True, reason, reasons=(reason, "gates-passed")
            )

    # Concise API used by PeriodicCoordinator and tests.
    request = request_transition

    def _enter_rollback(self, now: float, error: str) -> TopologyDecision:
        self._error = str(error)
        self._set_state(TopologyState.ROLLBACK, now)
        return self._decision(True, str(error))

    def _finish_rollback(self, now: float) -> TopologyDecision:
        prior = replace(self._prior) if self._prior is not None else None
        if prior is None:
            return self._degrade_locked(now, "rollback-missing-snapshot")
        try:
            if self._on_rollback is not None:
                self._on_rollback(prior)
        except Exception as exc:
            return self._degrade_locked(
                now, "rollback-adapter:%s" % type(exc).__name__
            )
        self.current_partition = prior
        self._target = None
        self._prior = None
        self._tx = None
        self._abort_requested = False
        self._set_state(TopologyState.STEADY, now)
        return self._decision(
            True,
            "rolled-back",
            rolled_back=True,
            decision_target=prior,
        )

    def _degrade_locked(
        self, now: float, error: str
    ) -> TopologyDecision:
        self._error = str(error)
        self._set_state(TopologyState.DEGRADED, now)
        return self._decision(False, self._error)

    def fail_closed(
        self, error: str, *, now: Optional[float] = None
    ) -> TopologyDecision:
        """Enter DEGRADED after an adapter/supervisor ownership failure."""
        now_f = float(self._clock() if now is None else now)
        with self.topology_lock:
            return self._degrade_locked(now_f, error)

    def abort(self, *, now: Optional[float] = None) -> TopologyDecision:
        now_f = float(self._clock() if now is None else now)
        with self.topology_lock:
            if self._state == TopologyState.STEADY:
                return self._decision(False, "nothing-to-abort")
            if self._state == TopologyState.DEGRADED:
                return self._decision(False, "degraded")
            self._abort_requested = True
            return self._advance_locked(now_f)

    def _advance_locked(self, now: float) -> TopologyDecision:
        if self._state == TopologyState.STEADY:
            return self._decision(False, "steady")
        if self._state == TopologyState.DEGRADED:
            return self._decision(False, "degraded")
        tx = self._tx
        if tx is None:
            return self._degrade_locked(now, "missing-transaction")

        if self._abort_requested and self._state == TopologyState.DRAINING:
            try:
                self.pm.cancel_restart_transaction(tx)
            except Exception as exc:
                return self._degrade_locked(
                    now, "abort:%s" % type(exc).__name__
                )
            return self._finish_rollback(now)
        if self._abort_requested and self._state != TopologyState.ROLLBACK:
            return self._enter_rollback(now, "abort-requested")

        if self._state == TopologyState.DRAINING:
            try:
                drained = self.pm.drain_restart_transaction(tx)
            except Exception as exc:
                return self._enter_rollback(
                    now, "drain:%s" % type(exc).__name__
                )
            if drained:
                self._set_state(TopologyState.RESTARTING, now)
                return self._decision(True, "drained")
            if now - self._state_since >= self.drain_timeout_s:
                return self._enter_rollback(now, "drain-timeout")
            return self._decision(True, "draining")

        if self._state == TopologyState.RESTARTING:
            try:
                self.pm.execute_restart_transaction(tx)
            except Exception as exc:
                return self._enter_rollback(
                    now, "restart:%s" % type(exc).__name__
                )
            self._set_state(TopologyState.STARTING, now)
            return self._decision(True, "restarted")

        if self._state == TopologyState.STARTING:
            try:
                healthy = self.pm.restart_transaction_healthy(tx)
            except Exception as exc:
                return self._enter_rollback(
                    now, "health:%s" % type(exc).__name__
                )
            if healthy:
                self._set_state(TopologyState.VALIDATING, now)
                return self._decision(True, "healthy")
            if now - self._state_since >= self.health_timeout_s:
                return self._enter_rollback(now, "health-timeout")
            return self._decision(True, "starting")

        if self._state == TopologyState.VALIDATING:
            if not self._call_gate(self._trusted):
                return self._enter_rollback(now, "validation-untrusted")
            if not self._call_gate(self._slo_safe):
                return self._enter_rollback(now, "validation-slo")
            try:
                valid = (
                    self.pm.restart_transaction_canary(tx)
                    and bool(self._canary_validator(tx.target))
                )
            except Exception:
                valid = False
            if not valid:
                return self._enter_rollback(now, "canary-failed")
            target = replace(tx.target)
            try:
                # The runtime adapter is part of validation.  PoolManager is
                # committed only after the adapter has accepted the new URLs.
                if self._on_commit is not None:
                    self._on_commit(target)
                self.pm.commit_restart_transaction(tx)
            except Exception as exc:
                return self._enter_rollback(
                    now, "commit:%s" % type(exc).__name__
                )
            self.current_partition = target
            self._last_switch = now
            self._target = None
            self._prior = None
            self._tx = None
            self._abort_requested = False
            self._error = ""
            self._set_state(TopologyState.STEADY, now)
            return self._decision(
                True,
                "committed",
                committed=True,
                decision_target=target,
            )

        if self._state == TopologyState.ROLLBACK:
            # A drain timeout/abort before physical restart can be restored
            # without stopping a healthy engine.
            if not tx.executed and not tx.restarted_names:
                try:
                    self.pm.cancel_restart_transaction(tx)
                except Exception as exc:
                    return self._degrade_locked(
                        now, "rollback-cancel:%s" % type(exc).__name__
                    )
                return self._finish_rollback(now)
            try:
                complete = self.pm.rollback_restart_transaction(tx)
            except Exception as exc:
                return self._degrade_locked(
                    now, "rollback:%s" % type(exc).__name__
                )
            if complete:
                return self._finish_rollback(now)
            if now - self._state_since >= self.rollback_timeout_s:
                return self._degrade_locked(now, "rollback-timeout")
            return self._decision(True, "rolling-back")

        return self._degrade_locked(now, "invalid-state")

    def step(self, now: Optional[float] = None) -> TopologyDecision:
        now_f = float(self._clock() if now is None else now)
        with self.topology_lock:
            return self._advance_locked(now_f)

