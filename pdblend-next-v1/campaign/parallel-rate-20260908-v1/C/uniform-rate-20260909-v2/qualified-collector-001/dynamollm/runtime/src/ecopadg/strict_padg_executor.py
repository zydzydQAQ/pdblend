"""Explicit execution boundary for strict-PaDG controller actions."""
from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ecopadg.engine_telemetry import FileTelemetryRegistry
from ecopadg.mpc.types import ControlAction
from ecopadg.telemetry import ControlState
from ecopadg.temporal_coordinator import (
    MODE_CONTINUOUS,
    MODE_TEMPORAL,
    GlobalTemporalCoordinator,
    TemporalConfig,
)


@dataclass(frozen=True)
class ExecutionResult:
    applied: bool
    action: ControlAction
    reason: str
    generation: Optional[int] = None
    ack_latency_s: Optional[float] = None
    rolled_back: bool = False
    rollback_generation: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "applied": bool(self.applied),
            "action": self.action.to_dict(),
            "reason": str(self.reason),
            "generation": self.generation,
            "ack_latency_s": self.ack_latency_s,
            "rolled_back": bool(self.rolled_back),
            "rollback_generation": self.rollback_generation,
        }


@runtime_checkable
class StrictPaDGExecutor(Protocol):
    """Only this protocol is allowed to cross into an actuation backend."""

    def apply(
        self, action: ControlAction, state: Optional[ControlState] = None
    ) -> ExecutionResult:
        ...


class NoOpStrictPaDGExecutor:
    """Production-safe adapter used while active MPC remains feature-gated."""

    def apply(
        self, action: ControlAction, state: Optional[ControlState] = None
    ) -> ExecutionResult:
        del state
        return ExecutionResult(
            applied=False, action=action, reason="executor-noop"
        )


class InMemoryStrictPaDGExecutor:
    """Side-effect-free adapter for tests and control-plane integration."""

    def __init__(self, initial: Optional[ControlAction] = None):
        self.current = initial
        self.history: List[ControlAction] = []
        self._lock = Lock()

    def apply(
        self, action: ControlAction, state: Optional[ControlState] = None
    ) -> ExecutionResult:
        del state
        with self._lock:
            self.current = action
            self.history.append(action)
        return ExecutionResult(
            applied=True, action=action, reason="in-memory-applied"
        )


class EcoSpdTemporalExecutor:
    """Atomically publish one joint temporal action to four mixed engines."""

    CONTROL_SCHEMA_VERSION = 1
    PHYSICAL_REPLICAS = 4
    _LOCAL_TARGET_ATTRIBUTES = (
        "_temporal_target_mode",
        "_temporal_target_n_prefill_active",
        "_temporal_target_window_s",
        "_temporal_target_token_budget",
        "_temporal_fP_target",
        "_temporal_fD_target",
        "_last_temporal_change_s",
        "_engine_control_generation",
        "_engine_applied_generation",
        "_last_mpc_fast_switch",
    )
    _COORDINATOR_ATTRIBUTES = (
        "_config",
        "_available",
        "_cursor",
        "_temporary_until_s",
        "_deferred_marker",
        "_state",
    )

    def __init__(
        self,
        controller: Any,
        coordinator: GlobalTemporalCoordinator,
        telemetry_registry: FileTelemetryRegistry,
        control_directory: str,
        ack_timeout_s: float = 5.0,
        ack_poll_s: float = 0.05,
        clock=time.time,
        sleep=time.sleep,
    ):
        timeout = float(ack_timeout_s)
        poll = float(ack_poll_s)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("ack_timeout_s must be finite and non-negative")
        if not math.isfinite(poll) or poll <= 0.0:
            raise ValueError("ack_poll_s must be finite and positive")
        self.controller = controller
        self.coordinator = coordinator
        self.telemetry_registry = telemetry_registry
        self.control_directory = str(control_directory or "")
        self.ack_timeout_s = timeout
        self.ack_poll_s = poll
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._generation = self._discover_generation()
        self.last_requested_generation: Optional[int] = None
        self.last_applied_generation: Optional[int] = None
        self.last_rollback_generation: Optional[int] = None
        self.last_requested_mode: Optional[str] = None
        self.last_result: Optional[ExecutionResult] = None
        self.last_telemetry: Dict[str, Dict[str, Any]] = {}
        self._last_applied_action: Optional[ControlAction] = None
        self._last_applied_config: Optional[TemporalConfig] = None

    @property
    def generation(self) -> int:
        with self._lock:
            return int(self._generation)

    def control_path(self, instance_id: str) -> str:
        instance = str(instance_id)
        if (
            not instance
            or os.path.basename(instance) != instance
            or instance in (".", "..")
        ):
            raise ValueError("invalid engine instance id: %r" % instance_id)
        return os.path.join(self.control_directory, "%s.json" % instance)

    def _discover_generation(self) -> int:
        highest = 0
        if not self.control_directory:
            return highest
        instance_ids = tuple(
            getattr(self.telemetry_registry, "instance_ids", ())
        )
        for instance_id in instance_ids:
            try:
                path = self.control_path(instance_id)
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                generation = payload.get("generation")
                if (
                    isinstance(generation, int)
                    and not isinstance(generation, bool)
                ):
                    highest = max(highest, generation)
            except (
                AttributeError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
        return highest

    def _next_generation(self) -> int:
        self._generation = max(
            int(self._generation), self._discover_generation()
        ) + 1
        return self._generation

    def _atomic_write_control(
        self, instance_id: str, generation: int, mode: str
    ) -> None:
        path = self.control_path(instance_id)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temporary = "%s.tmp.%d.%d.%d" % (
            path,
            os.getpid(),
            threading.get_ident(),
            int(generation),
        )
        payload = {
            "schema_version": self.CONTROL_SCHEMA_VERSION,
            "generation": int(generation),
            "mode": str(mode),
        }
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _write_generation(
        self, generation: int, mode: str, instance_ids: tuple[str, ...]
    ) -> List[str]:
        errors = []
        for instance_id in instance_ids:
            try:
                self._atomic_write_control(instance_id, generation, mode)
            except Exception as exc:
                errors.append(
                    "%s:%s:%s"
                    % (instance_id, type(exc).__name__, str(exc))
                )
        return errors

    def _topology_context(self):
        topology = getattr(self.controller, "topology_coordinator", None)
        lock = getattr(topology, "topology_lock", None)
        if lock is None:
            lock = getattr(self.controller, "topology_lock", None)
        if lock is None:
            lock = getattr(self.controller, "_topology_lock", None)
        return nullcontext() if lock is None else lock

    def _active_indices(self) -> tuple[int, ...]:
        active_mixed = getattr(self.controller, "active_mixed", None)
        if callable(active_mixed):
            return tuple(int(index) for index in active_mixed())
        mixed = tuple(getattr(self.controller, "mixed", ()))
        return tuple(
            index
            for index, instance in enumerate(mixed)
            if not bool(getattr(instance, "parked", False))
            and not bool(getattr(instance, "draining", False))
        )

    def _validate(
        self, action: ControlAction
    ) -> tuple[Optional[TemporalConfig], tuple[str, ...], str]:
        mode = str(action.fast.mode)
        if mode not in (MODE_CONTINUOUS, MODE_TEMPORAL):
            return None, (), "invalid-mode:%s" % mode
        if int(action.slow.active_replicas) != self.PHYSICAL_REPLICAS:
            return (
                None,
                (),
                "physical-replicas-must-be-4:got-%d"
                % int(action.slow.active_replicas),
            )
        instance_ids = tuple(
            str(value)
            for value in getattr(
                self.telemetry_registry, "instance_ids", ()
            )
        )
        if len(instance_ids) != self.PHYSICAL_REPLICAS:
            return (
                None,
                instance_ids,
                "expected-4-engine-telemetry-instances:got-%d"
                % len(instance_ids),
            )
        if len(set(instance_ids)) != len(instance_ids):
            return None, instance_ids, "duplicate-engine-instance-id"
        if not self.control_directory:
            return None, instance_ids, "engine-control-directory-missing"
        mixed = tuple(getattr(self.controller, "mixed", ()))
        if len(mixed) != self.PHYSICAL_REPLICAS:
            return (
                None,
                instance_ids,
                "expected-4-physical-mixed-engines:got-%d" % len(mixed),
            )
        active = self._active_indices()
        if tuple(sorted(set(active))) != tuple(
            range(self.PHYSICAL_REPLICAS)
        ):
            return (
                None,
                instance_ids,
                "all-4-physical-mixed-engines-must-be-unparked:%s"
                % ",".join(str(index) for index in active),
            )
        try:
            candidate = TemporalConfig(
                mode=mode,
                n_prefill_active=int(action.fast.n_prefill_active),
                window_s=float(action.fast.window_s),
                token_budget=int(action.fast.token_budget),
                fP=int(action.fast.prefill_freq_mhz),
                fD=int(action.fast.decode_freq_mhz),
            )
            for instance_id in instance_ids:
                self.control_path(instance_id)
        except (TypeError, ValueError) as exc:
            return (
                None,
                instance_ids,
                "invalid-temporal-action:%s" % str(exc),
            )
        return candidate, instance_ids, ""

    @staticmethod
    def _view_status(view: Any) -> Dict[str, Any]:
        if view is None:
            return {
                "present": False,
                "fresh": False,
                "requested_generation": None,
                "applied_generation": None,
                "applied_mode": None,
                "pending_generation": None,
                "requested_mode": None,
                "pending_mode": None,
                "control_error": None,
                "chunked_prefill_enabled": None,
            }
        return {
            "present": True,
            "fresh": not bool(getattr(view, "stale", True)),
            "requested_generation": getattr(
                view, "requested_generation", None
            ),
            "applied_generation": getattr(
                view, "applied_generation", None
            ),
            "applied_mode": getattr(view, "applied_mode", None),
            "pending_generation": getattr(
                view, "pending_generation", None
            ),
            "requested_mode": getattr(view, "requested_mode", None),
            "pending_mode": getattr(view, "pending_mode", None),
            "control_error": getattr(view, "control_error", None),
            "chunked_prefill_enabled": getattr(
                view, "chunked_prefill_enabled", None
            ),
        }

    def _poll_ack(
        self,
        generation: int,
        mode: str,
        instance_ids: tuple[str, ...],
        *,
        timeout_s: Optional[float] = None,
        allow_lazy_idle: bool = False,
    ) -> tuple[bool, str, float]:
        started = float(self._clock())
        timeout = (
            self.ack_timeout_s
            if timeout_s is None
            else max(float(timeout_s), 0.0)
        )
        deadline = started + timeout
        while True:
            now = float(self._clock())
            elapsed = max(now - started, 0.0)
            try:
                views = self.telemetry_registry.read_all(now=now)
            except Exception as exc:
                return (
                    False,
                    "telemetry-read-error:%s:%s"
                    % (type(exc).__name__, str(exc)),
                    elapsed,
                )
            statuses = {
                instance_id: self._view_status(views.get(instance_id))
                for instance_id in instance_ids
            }
            self.last_telemetry = statuses
            lazy_idle = set()
            if allow_lazy_idle:
                mixed = tuple(getattr(self.controller, "mixed", ()))
                for index, instance_id in enumerate(instance_ids):
                    if index >= len(mixed):
                        continue
                    instance = mixed[index]
                    scheduler = getattr(instance, "sched", None)
                    locally_idle = (
                        int(getattr(instance, "inflight", 0) or 0) == 0
                        and not bool(getattr(scheduler, "buffer", ()))
                        and not bool(getattr(scheduler, "active", ()))
                    )
                    if (
                        statuses[instance_id]["present"]
                        and locally_idle
                        and self._control_files_match(
                            generation, mode, (instance_id,))
                    ):
                        lazy_idle.add(instance_id)
            errors = [
                "%s:%s" % (instance_id, status["control_error"])
                for instance_id, status in statuses.items()
                if (
                    status["fresh"]
                    and status["control_error"]
                    and (
                        status["requested_generation"] in (None, generation)
                        or status["pending_generation"] == generation
                        or status["applied_generation"] == generation
                    )
                )
            ]
            if errors:
                return (
                    False,
                    "engine-control-error:" + "|".join(errors),
                    elapsed,
                )
            def acknowledged(instance_id, status):
                return (
                    instance_id in lazy_idle
                    or (
                        status["fresh"]
                        and status["requested_generation"] == generation
                        and status["requested_mode"] == mode
                        and status["applied_generation"] == generation
                        and status["applied_mode"] == mode
                        and status["pending_generation"] is None
                        and not status["control_error"]
                        and (
                            status["chunked_prefill_enabled"] is None
                            or status["chunked_prefill_enabled"]
                            == (mode == MODE_CONTINUOUS)
                        )
                    )
                )

            if all(acknowledged(instance_id, status)
                   for instance_id, status in statuses.items()):
                reason = "all-engine-acks"
                if lazy_idle:
                    reason += ";lazy-idle=" + ",".join(sorted(lazy_idle))
                return True, reason, elapsed
            if now >= deadline:
                details = []
                for instance_id, status in statuses.items():
                    if not status["fresh"]:
                        state = "missing-or-stale"
                    elif status["pending_generation"] == generation:
                        state = "pending"
                    else:
                        state = "applied=%s/%s" % (
                            status["applied_generation"],
                            status["applied_mode"],
                        )
                    details.append("%s=%s" % (instance_id, state))
                return (
                    False,
                    "ack-timeout:" + ",".join(details),
                    elapsed,
                )
            self._sleep(min(self.ack_poll_s, max(deadline - now, 0.0)))

    @staticmethod
    def _scheduler_budgets(controller: Any) -> tuple[int, ...]:
        values = []
        for instance in tuple(getattr(controller, "mixed", ())):
            scheduler = getattr(instance, "sched", None)
            if scheduler is None or not hasattr(
                scheduler, "prefill_token_budget"
            ):
                raise AttributeError(
                    "mixed scheduler lacks prefill_token_budget"
                )
            values.append(int(scheduler.prefill_token_budget))
        return tuple(values)

    @staticmethod
    def _set_scheduler_budgets(controller: Any, budget: int) -> None:
        for instance in tuple(getattr(controller, "mixed", ())):
            instance.sched.prefill_token_budget = int(budget)

    @staticmethod
    def _restore_scheduler_budgets(
        controller: Any, budgets: tuple[int, ...]
    ) -> List[str]:
        errors = []
        for index, (instance, budget) in enumerate(zip(
            tuple(getattr(controller, "mixed", ())), budgets
        )):
            try:
                instance.sched.prefill_token_budget = int(budget)
            except Exception as exc:
                errors.append(
                    "budget[%d]:%s:%s"
                    % (index, type(exc).__name__, str(exc))
                )
        return errors

    @staticmethod
    def _controller_targets(config: TemporalConfig, now: float) -> dict:
        return {
            "_temporal_target_mode": config.mode,
            "_temporal_target_n_prefill_active": config.n_prefill_active,
            "_temporal_target_window_s": config.window_s,
            "_temporal_target_token_budget": config.token_budget,
            "_temporal_fP_target": config.fP,
            "_temporal_fD_target": config.fD,
            "_last_temporal_change_s": now,
        }

    @staticmethod
    def _snapshot_attributes(target: Any, names: tuple[str, ...]) -> dict:
        return {
            name: (hasattr(target, name), getattr(target, name, None))
            for name in names
        }

    @staticmethod
    def _restore_attributes(target: Any, snapshot: dict) -> None:
        for name, (present, value) in snapshot.items():
            if present:
                setattr(target, name, value)
            elif hasattr(target, name):
                delattr(target, name)

    def _snapshot_coordinator(self) -> dict:
        lock = getattr(self.coordinator, "_lock", None)
        with nullcontext() if lock is None else lock:
            prior_state = self.coordinator.snapshot()
            snapshot_config = getattr(
                self.coordinator, "snapshot_config", None
            )
            previous = (
                snapshot_config()
                if callable(snapshot_config)
                else TemporalConfig(
                    mode=prior_state.mode,
                    n_prefill_active=prior_state.n_prefill_active,
                    window_s=prior_state.window_s,
                    token_budget=prior_state.token_budget,
                    fP=prior_state.fP,
                    fD=prior_state.fD,
                )
            )
            attributes = self._snapshot_attributes(
                self.coordinator, self._COORDINATOR_ATTRIBUTES
            )
            events_present = hasattr(self.coordinator, "_events")
            events = (
                tuple(getattr(self.coordinator, "_events"))
                if events_present
                else ()
            )
        return {
            "state": prior_state,
            "config": previous,
            "attributes": attributes,
            "events_present": events_present,
            "events": events,
        }

    def _restore_coordinator(self, snapshot: dict) -> None:
        lock = getattr(self.coordinator, "_lock", None)
        with nullcontext() if lock is None else lock:
            self._restore_attributes(
                self.coordinator, snapshot["attributes"]
            )
            if snapshot["events_present"]:
                events = getattr(self.coordinator, "_events")
                events.clear()
                events.extend(snapshot["events"])
            elif hasattr(self.coordinator, "_events"):
                delattr(self.coordinator, "_events")

    def _restore_local_state(
        self,
        coordinator_snapshot: dict,
        budgets: tuple[int, ...],
        target_snapshot: dict,
    ) -> List[str]:
        errors = []
        try:
            self._restore_coordinator(coordinator_snapshot)
        except Exception as exc:
            errors.append(
                "coordinator:%s:%s"
                % (type(exc).__name__, str(exc))
            )
        errors.extend(
            self._restore_scheduler_budgets(self.controller, budgets)
        )
        try:
            self._restore_attributes(self.controller, target_snapshot)
        except Exception as exc:
            errors.append(
                "targets:%s:%s" % (type(exc).__name__, str(exc))
            )
        return errors

    def _control_files_match(
        self,
        generation: int,
        mode: str,
        instance_ids: tuple[str, ...],
    ) -> bool:
        for instance_id in instance_ids:
            try:
                with open(
                    self.control_path(instance_id), encoding="utf-8"
                ) as handle:
                    payload = json.load(handle)
            except (
                AttributeError,
                OSError,
                TypeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return False
            if (
                payload.get("schema_version")
                != self.CONTROL_SCHEMA_VERSION
                or payload.get("generation") != generation
                or payload.get("mode") != mode
            ):
                return False
        return True

    def _local_publication_matches(
        self,
        action: ControlAction,
        candidate: TemporalConfig,
        previous: TemporalConfig,
        prior_state: Any,
        budgets: tuple[int, ...],
    ) -> bool:
        generation = self.last_applied_generation
        if (
            generation is None
            or self.last_rollback_generation is not None
            or self._last_applied_action != action
            or self._last_applied_config != candidate
            or previous != candidate
            or tuple(budgets)
            != (candidate.token_budget,) * self.PHYSICAL_REPLICAS
            or tuple(getattr(self.coordinator, "_available", ()))
            != tuple(range(self.PHYSICAL_REPLICAS))
        ):
            return False
        expected = {
            "_temporal_target_mode": candidate.mode,
            "_temporal_target_n_prefill_active": (
                candidate.n_prefill_active
            ),
            "_temporal_target_window_s": candidate.window_s,
            "_temporal_target_token_budget": candidate.token_budget,
            "_temporal_fP_target": candidate.fP,
            "_temporal_fD_target": candidate.fD,
            "_engine_control_generation": generation,
            "_engine_applied_generation": generation,
        }
        if any(
            not hasattr(self.controller, name)
            or getattr(self.controller, name) != value
            for name, value in expected.items()
        ):
            return False
        return (
            prior_state.mode == candidate.mode
            and prior_state.window_s == candidate.window_s
            and prior_state.token_budget == candidate.token_budget
            and prior_state.fP == candidate.fP
            and prior_state.fD == candidate.fD
        )

    def _request_rollback(
        self,
        previous_mode: str,
        instance_ids: tuple[str, ...],
    ) -> tuple[Optional[int], bool, str]:
        try:
            generation = self._next_generation()
            self.last_rollback_generation = generation
            errors = self._write_generation(
                generation, previous_mode, instance_ids
            )
            acknowledged, ack_reason, ack_latency = self._poll_ack(
                generation, previous_mode, instance_ids,
                allow_lazy_idle=True,
            )
        except Exception as exc:
            return (
                self.last_rollback_generation,
                False,
                "rollback-error:%s:%s"
                % (type(exc).__name__, str(exc)),
            )
        details = [
            "rollback-generation=%d" % generation,
            "mode=%s" % previous_mode,
            "acknowledged=%s" % str(bool(acknowledged)).lower(),
            "ack-latency-s=%.6f" % ack_latency,
            ack_reason,
        ]
        if errors:
            details.insert(2, "write-errors=%s" % "|".join(errors))
        return generation, bool(acknowledged), " ".join(details)

    def _fail_closed(self) -> None:
        try:
            self.controller._force_max_freq = True
            self.controller._temporal_fail_closed_pending = True
        except Exception:
            pass
        unpark = getattr(self.controller, "_unpark_all", None)
        if callable(unpark):
            try:
                unpark()
            except Exception:
                pass

    def status(self) -> dict:
        with self._lock:
            result = self.last_result
            return {
                "generation": int(self._generation),
                "requested_generation": self.last_requested_generation,
                "applied_generation": self.last_applied_generation,
                "rollback_generation": self.last_rollback_generation,
                "requested_mode": self.last_requested_mode,
                "executor_generation": (
                    None if result is None else result.generation
                ),
                "ack_latency_s": (
                    None if result is None else result.ack_latency_s
                ),
                "rolled_back": (
                    False if result is None else bool(result.rolled_back)
                ),
                "applied": (
                    None if result is None else bool(result.applied)
                ),
                "reason": (
                    "" if result is None else str(result.reason)
                ),
                "telemetry": dict(self.last_telemetry),
            }

    def _result(
        self,
        applied: bool,
        action: ControlAction,
        reason: str,
        *,
        generation: Optional[int] = None,
        ack_latency_s: Optional[float] = None,
        rolled_back: bool = False,
        rollback_generation: Optional[int] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            applied=bool(applied),
            action=action,
            reason=str(reason),
            generation=generation,
            ack_latency_s=ack_latency_s,
            rolled_back=bool(rolled_back),
            rollback_generation=rollback_generation,
        )
        self.last_result = result
        return result

    def apply(
        self, action: ControlAction, state: Optional[ControlState] = None
    ) -> ExecutionResult:
        del state
        with self._topology_context():
            with self._lock:
                unpark = getattr(self.controller, "_unpark_all", None)
                if callable(unpark):
                    unpark()
                candidate, instance_ids, invalid = self._validate(action)
                if candidate is None:
                    self._fail_closed()
                    return self._result(
                        False, action, "validation-failed:%s" % invalid
                    )

                try:
                    coordinator_snapshot = self._snapshot_coordinator()
                    prior_state = coordinator_snapshot["state"]
                    previous = coordinator_snapshot["config"]
                    prior_budgets = self._scheduler_budgets(
                        self.controller
                    )
                    target_snapshot = self._snapshot_attributes(
                        self.controller, self._LOCAL_TARGET_ATTRIBUTES
                    )
                except Exception as exc:
                    self._fail_closed()
                    return self._result(
                        False,
                        action,
                        "snapshot-failed:%s:%s"
                        % (type(exc).__name__, str(exc)),
                    )

                if self._local_publication_matches(
                    action,
                    candidate,
                    previous,
                    prior_state,
                    prior_budgets,
                ):
                    applied_generation = int(self.last_applied_generation)
                    if self._control_files_match(
                        applied_generation, candidate.mode, instance_ids
                    ):
                        try:
                            acknowledged, _, ack_latency = self._poll_ack(
                                applied_generation,
                                candidate.mode,
                                instance_ids,
                                timeout_s=0.0,
                            )
                        except Exception:
                            acknowledged = False
                            ack_latency = None
                        if acknowledged:
                            return self._result(
                                True,
                                action,
                                "no-change",
                                generation=applied_generation,
                                ack_latency_s=ack_latency,
                            )

                generation = self._next_generation()
                self.last_requested_generation = generation
                self.last_requested_mode = candidate.mode
                self.last_rollback_generation = None
                write_errors = self._write_generation(
                    generation, candidate.mode, instance_ids
                )
                if write_errors:
                    restore_errors = self._restore_local_state(
                        coordinator_snapshot,
                        prior_budgets,
                        target_snapshot,
                    )
                    self._fail_closed()
                    rollback_generation, rolled_back, rollback = (
                        self._request_rollback(
                            previous.mode, instance_ids
                        )
                    )
                    restore_detail = (
                        ""
                        if not restore_errors
                        else "; local-restore-errors=%s"
                        % "|".join(restore_errors)
                    )
                    return self._result(
                        False,
                        action,
                        (
                            "control-write-failed:generation=%d errors=%s; %s%s"
                        )
                        % (
                            generation,
                            "|".join(write_errors),
                            rollback,
                            restore_detail,
                        ),
                        generation=generation,
                        rolled_back=rolled_back,
                        rollback_generation=rollback_generation,
                    )

                try:
                    acknowledged, ack_reason, ack_latency = self._poll_ack(
                        generation, candidate.mode, instance_ids,
                        allow_lazy_idle=True,
                    )
                except Exception as exc:
                    acknowledged = False
                    ack_latency = None
                    ack_reason = "ack-poll-error:%s:%s" % (
                        type(exc).__name__, str(exc))
                if not acknowledged:
                    restore_errors = self._restore_local_state(
                        coordinator_snapshot,
                        prior_budgets,
                        target_snapshot,
                    )
                    self._fail_closed()
                    rollback_generation, rolled_back, rollback = (
                        self._request_rollback(
                            previous.mode, instance_ids
                        )
                    )
                    restore_detail = (
                        ""
                        if not restore_errors
                        else "; local-restore-errors=%s"
                        % "|".join(restore_errors)
                    )
                    return self._result(
                        False,
                        action,
                        "generation=%d mode=%s %s; %s%s"
                        % (
                            generation,
                            candidate.mode,
                            ack_reason,
                            rollback,
                            restore_detail,
                        ),
                        generation=generation,
                        ack_latency_s=ack_latency,
                        rolled_back=rolled_back,
                        rollback_generation=rollback_generation,
                    )

                try:
                    published_at = float(self._clock())
                    targets = self._controller_targets(
                        candidate, published_at
                    )
                    targets.update({
                        "_engine_control_generation": generation,
                        "_engine_applied_generation": generation,
                        "_last_mpc_fast_switch": published_at,
                    })
                    self._set_scheduler_budgets(
                        self.controller, candidate.token_budget
                    )
                    for name, value in targets.items():
                        setattr(self.controller, name, value)
                    self.coordinator.configure(
                        candidate,
                        active=tuple(range(self.PHYSICAL_REPLICAS)),
                        now=float(self._clock()),
                        reason="engine-ack-generation-%d" % generation,
                    )
                except Exception as exc:
                    restore_errors = self._restore_local_state(
                        coordinator_snapshot,
                        prior_budgets,
                        target_snapshot,
                    )
                    self._fail_closed()
                    rollback_generation, rolled_back, rollback = (
                        self._request_rollback(
                            previous.mode, instance_ids
                        )
                    )
                    restore_detail = (
                        ""
                        if not restore_errors
                        else "; local-restore-errors=%s"
                        % "|".join(restore_errors)
                    )
                    return self._result(
                        False,
                        action,
                        "publish-failed:generation=%d %s:%s; %s%s"
                        % (
                            generation,
                            type(exc).__name__,
                            str(exc),
                            rollback,
                            restore_detail,
                        ),
                        generation=generation,
                        ack_latency_s=ack_latency,
                        rolled_back=rolled_back,
                        rollback_generation=rollback_generation,
                    )

                self.last_applied_generation = generation
                self._last_applied_action = action
                self._last_applied_config = candidate
                return self._result(
                    True,
                    action,
                    "applied:generation=%d mode=%s instances=%s"
                    % (
                        generation,
                        candidate.mode,
                        ",".join(instance_ids),
                    ),
                    generation=generation,
                    ack_latency_s=ack_latency,
                )

