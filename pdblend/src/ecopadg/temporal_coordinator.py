# -*- coding: utf-8 -*-
"""Global admission windows for KV-local mixed engines.

The coordinator changes only where a *new* request may start.  Once selected,
an engine owns the request (and its KV cache) through decode.  Engine-local
schedulers remain responsible for immediate continuous admission.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, List, Mapping, Optional, Sequence, Tuple, Union

from ecopadg.types import TOKEN_BUDGET

MODE_CONTINUOUS = "continuous"
MODE_TEMPORAL = "temporal"
TEMPORAL_MODES = (MODE_CONTINUOUS, MODE_TEMPORAL)
PREFILL_COUNTS = (1, 2, 4)
MIN_WINDOW_S = 1.5

_UNSET = object()


@dataclass(frozen=True)
class TemporalConfig:
    """One atomic coordinator configuration.

    ``fP`` and ``fD`` are deliberately inert placeholders for the later
    MPC/executor integration; this module never writes engine controls.
    """

    mode: str = MODE_CONTINUOUS
    n_prefill_active: int = 4
    window_s: float = 3.0
    token_budget: int = TOKEN_BUDGET
    fP: Optional[int] = None
    fD: Optional[int] = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in TEMPORAL_MODES:
            raise ValueError(
                "mode must be one of %s" % ", ".join(TEMPORAL_MODES)
            )
        if isinstance(self.n_prefill_active, bool):
            raise ValueError("n_prefill_active must be one of 1, 2, 4")
        n_prefill = int(self.n_prefill_active)
        if n_prefill not in PREFILL_COUNTS:
            raise ValueError("n_prefill_active must be one of 1, 2, 4")
        window_s = float(self.window_s)
        if not math.isfinite(window_s) or window_s < MIN_WINDOW_S:
            raise ValueError("window_s must be finite and at least 1.5")
        if isinstance(self.token_budget, bool):
            raise ValueError("token_budget must be positive")
        token_budget = int(self.token_budget)
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        for name, value in (("fP", self.fP), ("fD", self.fD)):
            if value is not None and int(value) <= 0:
                raise ValueError("%s must be positive or None" % name)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "n_prefill_active", n_prefill)
        object.__setattr__(self, "window_s", window_s)
        object.__setattr__(self, "token_budget", token_budget)
        object.__setattr__(
            self, "fP", None if self.fP is None else int(self.fP)
        )
        object.__setattr__(
            self, "fD", None if self.fD is None else int(self.fD)
        )

    @property
    def fP_target(self) -> Optional[int]:
        return self.fP

    @property
    def fD_target(self) -> Optional[int]:
        return self.fD


@dataclass(frozen=True)
class TemporalState:
    generation: int
    mode: str
    n_prefill_active: int
    window_s: float
    token_budget: int
    fP: Optional[int]
    fD: Optional[int]
    active_set: Tuple[int, ...]
    epoch: int
    window_start_s: float
    window_end_s: float
    deferred_rotations: int
    fallback_reason: str

    @property
    def window_start(self) -> float:
        return self.window_start_s

    @property
    def window_end(self) -> float:
        return self.window_end_s

    @property
    def fP_target(self) -> Optional[int]:
        return self.fP

    @property
    def fD_target(self) -> Optional[int]:
        return self.fD


@dataclass(frozen=True)
class TemporalEvent:
    timestamp_s: float
    event: str
    generation: int
    mode: str
    n_prefill_active: int
    active_set: Tuple[int, ...]
    epoch: int
    window_start_s: float
    window_end_s: float
    window_s: float
    token_budget: int
    fP: Optional[int]
    fD: Optional[int]
    deferred_rotations: int
    fallback_reason: str
    reason: str = ""

    @property
    def kind(self) -> str:
        return self.event

    def to_row(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "event": self.event,
            "generation": self.generation,
            "mode": self.mode,
            "n_prefill_active": self.n_prefill_active,
            "active_set": ";".join(str(i) for i in self.active_set),
            "epoch": self.epoch,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "window_s": self.window_s,
            "token_budget": self.token_budget,
            "fP_target": "" if self.fP is None else self.fP,
            "fD_target": "" if self.fD is None else self.fD,
            "deferred_rotations": self.deferred_rotations,
            "fallback_reason": self.fallback_reason,
            "reason": self.reason,
        }


ActiveInput = Union[int, Sequence[int]]
PhaseViews = Union[Mapping[int, object], Sequence[object]]


class GlobalTemporalCoordinator:
    """Thread-safe global admission coordinator for mixed engines."""

    def __init__(
        self,
        available_engines: Union[ActiveInput, TemporalConfig] = 4,
        config: Optional[TemporalConfig] = None,
        *,
        active: Optional[Sequence[int]] = None,
        clock=time.time,
    ):
        self._clock = clock
        self._lock = threading.RLock()
        self._events: Deque[TemporalEvent] = deque()
        if isinstance(available_engines, TemporalConfig):
            if config is not None:
                raise TypeError("config was provided twice")
            config = available_engines
            available_engines = 4
        self._base_active = self._normalize_available(available_engines)
        self._available = self._normalize_active(
            self._base_active if active is None else active
        )
        self._cursor = 0
        self._temporary_until_s = 0.0
        self._deferred_marker: Optional[Tuple[int, int, float]] = None

        default_n = self._largest_prefill_count(len(self._base_active))
        initial = config or TemporalConfig(
            mode=MODE_CONTINUOUS,
            n_prefill_active=default_n,
        )
        now = self._now(None)
        self._config = initial
        self._state = TemporalState(
            generation=0,
            mode=initial.mode,
            n_prefill_active=0,
            window_s=initial.window_s,
            token_budget=initial.token_budget,
            fP=initial.fP,
            fD=initial.fD,
            active_set=(),
            epoch=0,
            window_start_s=now,
            window_end_s=now + initial.window_s,
            deferred_rotations=0,
            fallback_reason="",
        )
        self._configure_locked(initial, self._available, now, "initial")

    @staticmethod
    def _normalize_available(value: ActiveInput) -> Tuple[int, ...]:
        if isinstance(value, bool):
            raise ValueError("available_engines must be non-negative")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("available_engines must be non-negative")
            return tuple(range(value))
        return GlobalTemporalCoordinator._normalize_active(value)

    @staticmethod
    def _normalize_active(active: Sequence[int]) -> Tuple[int, ...]:
        out = []
        seen = set()
        for raw in active:
            if isinstance(raw, bool):
                raise ValueError("engine indices must be non-negative integers")
            index = int(raw)
            if index < 0:
                raise ValueError("engine indices must be non-negative integers")
            if index not in seen:
                seen.add(index)
                out.append(index)
        return tuple(sorted(out))

    @staticmethod
    def _largest_prefill_count(available: int) -> int:
        if available <= 1:
            return 1
        if available <= 3:
            return 2
        return 4

    @staticmethod
    def _effective_prefill_count(requested: int, available: int) -> int:
        if available <= 0:
            return 0
        bounded = min(int(requested), int(available))
        return max(value for value in PREFILL_COUNTS if value <= bounded)

    def _now(self, now: Optional[float]) -> float:
        value = float(self._clock() if now is None else now)
        if not math.isfinite(value):
            raise ValueError("now must be finite")
        return value

    def _window_set(
        self, active: Tuple[int, ...], n_prefill: int
    ) -> Tuple[int, ...]:
        if not active or n_prefill <= 0:
            return ()
        start = self._cursor % len(active)
        return tuple(
            active[(start + offset) % len(active)]
            for offset in range(min(n_prefill, len(active)))
        )

    def _state_for_config(
        self,
        config: TemporalConfig,
        active: Tuple[int, ...],
        now: float,
        *,
        generation: int,
        epoch: int,
        deferred_rotations: int,
        fallback_reason: str,
    ) -> TemporalState:
        if config.mode == MODE_CONTINUOUS:
            active_set = active
            n_prefill = len(active)
        else:
            n_prefill = self._effective_prefill_count(
                config.n_prefill_active, len(active)
            )
            active_set = self._window_set(active, n_prefill)
        return TemporalState(
            generation=generation,
            mode=config.mode,
            n_prefill_active=n_prefill,
            window_s=config.window_s,
            token_budget=config.token_budget,
            fP=config.fP,
            fD=config.fD,
            active_set=active_set,
            epoch=epoch,
            window_start_s=now,
            window_end_s=now + config.window_s,
            deferred_rotations=deferred_rotations,
            fallback_reason=fallback_reason,
        )

    def _append_event_locked(
        self, event: str, now: float, reason: str = ""
    ) -> None:
        state = self._state
        self._events.append(
            TemporalEvent(
                timestamp_s=now,
                event=event,
                generation=state.generation,
                mode=state.mode,
                n_prefill_active=state.n_prefill_active,
                active_set=state.active_set,
                epoch=state.epoch,
                window_start_s=state.window_start_s,
                window_end_s=state.window_end_s,
                window_s=state.window_s,
                token_budget=state.token_budget,
                fP=state.fP,
                fD=state.fD,
                deferred_rotations=state.deferred_rotations,
                fallback_reason=state.fallback_reason,
                reason=str(reason or ""),
            )
        )

    def _configure_locked(
        self,
        config: TemporalConfig,
        active: Tuple[int, ...],
        now: float,
        reason: str,
    ) -> TemporalState:
        self._config = config
        self._available = active
        self._cursor = 0
        self._temporary_until_s = 0.0
        self._deferred_marker = None
        self._state = self._state_for_config(
            config,
            active,
            now,
            generation=self._state.generation + 1,
            epoch=0,
            deferred_rotations=0,
            fallback_reason="",
        )
        self._append_event_locked("configuration", now, reason)
        return self._state

    def configure(
        self,
        config: Optional[TemporalConfig] = None,
        *,
        mode: Optional[str] = None,
        n_prefill_active: Optional[int] = None,
        window_s: Optional[float] = None,
        token_budget: Optional[int] = None,
        fP=_UNSET,
        fD=_UNSET,
        active: Optional[Sequence[int]] = None,
        now: Optional[float] = None,
        reason: str = "configure",
    ) -> TemporalState:
        """Validate and publish an entire configuration under one lock."""
        current = self._now(now)
        with self._lock:
            base = self._config if config is None else config
            if not isinstance(base, TemporalConfig):
                raise TypeError("config must be a TemporalConfig")
            candidate = TemporalConfig(
                mode=base.mode if mode is None else mode,
                n_prefill_active=(
                    base.n_prefill_active
                    if n_prefill_active is None
                    else n_prefill_active
                ),
                window_s=base.window_s if window_s is None else window_s,
                token_budget=(
                    base.token_budget if token_budget is None else token_budget
                ),
                fP=base.fP if fP is _UNSET else fP,
                fD=base.fD if fD is _UNSET else fD,
            )
            available = (
                self._available
                if active is None
                else self._normalize_active(active)
            )
            return self._configure_locked(
                candidate, available, current, reason
            )

    def _sync_active_locked(
        self, active: Tuple[int, ...]
    ) -> TemporalState:
        if active == self._available:
            return self._state
        self._available = active
        state = self._state
        if state.mode == MODE_CONTINUOUS:
            active_set = active
            n_prefill = len(active)
        else:
            n_prefill = self._effective_prefill_count(
                self._config.n_prefill_active, len(active)
            )
            active_set = self._window_set(active, n_prefill)
        self._state = replace(
            state,
            n_prefill_active=n_prefill,
            active_set=active_set,
        )
        return self._state

    def eligible_indices(self, active: Sequence[int]) -> List[int]:
        """Return engines eligible to receive a new full request."""
        available = self._normalize_active(active)
        with self._lock:
            state = self._sync_active_locked(available)
            if state.mode == MODE_CONTINUOUS:
                return list(available)
            eligible = [i for i in state.active_set if i in available]
            if eligible:
                return eligible
            if not available:
                return []
            self._fallback_locked(
                "no-eligible-engine", available, self._now(None), temporary=False
            )
            return list(available)

    @staticmethod
    def _inflight_for(inflight, index: int) -> int:
        try:
            if isinstance(inflight, Mapping):
                return max(int(inflight.get(index, 0)), 0)
            return max(int(inflight[index]), 0)
        except (IndexError, KeyError, TypeError, ValueError):
            return 0

    def choose_prefill_target(
        self, active: Sequence[int], inflight
    ) -> int:
        """Choose the least-loaded eligible engine with stable tie-breaking."""
        available = self._normalize_active(active)
        with self._lock:
            state = self._sync_active_locked(available)
            eligible = (
                list(available)
                if state.mode == MODE_CONTINUOUS
                else [i for i in state.active_set if i in available]
            )
            if not eligible and available:
                self._fallback_locked(
                    "no-eligible-engine",
                    available,
                    self._now(None),
                    temporary=False,
                )
                eligible = list(available)
            if not eligible:
                raise RuntimeError("no active mixed engine")
            return min(
                eligible,
                key=lambda index: (
                    self._inflight_for(inflight, index),
                    eligible.index(index),
                ),
            )

    @staticmethod
    def _phase_value(view: object, name: str, default=None):
        if isinstance(view, Mapping):
            return view.get(name, default)
        return getattr(view, name, default)

    @classmethod
    def _rotation_blocker(cls, index: int, view: object) -> str:
        if view is None:
            return "telemetry-missing:%d" % index
        if bool(cls._phase_value(view, "stale", False)):
            return "telemetry-stale:%d" % index
        fresh = cls._phase_value(view, "fresh", None)
        if fresh is not None and not bool(fresh):
            return "telemetry-stale:%d" % index
        phase = str(
            view if isinstance(view, str)
            else cls._phase_value(view, "phase", "unknown")
        ).lower()
        try:
            n_prefill = int(
                cls._phase_value(view, "n_prefill_groups", 0) or 0
            )
        except (TypeError, ValueError):
            n_prefill = 0
        if n_prefill > 0 or phase in ("prefill", "overlap"):
            return "prefill-active:%d" % index
        if phase not in ("decode", "idle"):
            return "telemetry-phase-%s:%d" % (phase, index)
        return ""

    @staticmethod
    def _view_for(phase_views: PhaseViews, index: int):
        if isinstance(phase_views, Mapping):
            if index in phase_views:
                return phase_views[index]
            if str(index) in phase_views:
                return phase_views[str(index)]
            return phase_views.get("mixed-%d" % index)
        try:
            return phase_views[index]
        except (IndexError, TypeError):
            return None

    def _recover_temporary_locked(self, now: float) -> None:
        if self._temporary_until_s <= 0.0 or now < self._temporary_until_s:
            return
        self._temporary_until_s = 0.0
        self._deferred_marker = None
        self._state = self._state_for_config(
            self._config,
            self._available,
            now,
            generation=self._state.generation,
            epoch=self._state.epoch,
            deferred_rotations=self._state.deferred_rotations,
            fallback_reason="",
        )
        self._append_event_locked(
            "fallback-recovery", now, "temporary-fallback-expired"
        )

    def step(self, now: float, phase_views: PhaseViews) -> TemporalState:
        """Advance an expired temporal window when telemetry proves it safe."""
        current = self._now(now)
        with self._lock:
            self._recover_temporary_locked(current)
            state = self._state
            if state.mode != MODE_TEMPORAL:
                return state
            if current < state.window_end_s:
                return state

            blocker = ""
            if not state.active_set:
                blocker = "no-active-engine"
            else:
                for index in state.active_set:
                    blocker = self._rotation_blocker(
                        index, self._view_for(phase_views, index)
                    )
                    if blocker:
                        break
            if blocker:
                marker = (
                    state.generation,
                    state.epoch,
                    state.window_end_s,
                )
                if marker != self._deferred_marker:
                    self._deferred_marker = marker
                    self._state = replace(
                        state,
                        deferred_rotations=state.deferred_rotations + 1,
                    )
                    self._append_event_locked(
                        "rotation-deferred", current, blocker
                    )
                return self._state

            self._deferred_marker = None
            if self._available:
                self._cursor = (
                    self._cursor + max(state.n_prefill_active, 1)
                ) % len(self._available)
            n_prefill = self._effective_prefill_count(
                self._config.n_prefill_active, len(self._available)
            )
            active_set = self._window_set(self._available, n_prefill)
            self._state = replace(
                state,
                n_prefill_active=n_prefill,
                active_set=active_set,
                epoch=state.epoch + 1,
                window_start_s=current,
                window_end_s=current + state.window_s,
            )
            self._append_event_locked("rotation", current)
            return self._state

    def _fallback_locked(
        self,
        reason: str,
        active: Tuple[int, ...],
        now: float,
        *,
        temporary: bool,
        hold_s: Optional[float] = None,
    ) -> TemporalState:
        reason = str(reason or "emergency")
        if temporary and self._config.mode == MODE_CONTINUOUS:
            self._sync_active_locked(active)
            return self._state
        if (
            self._state.mode == MODE_CONTINUOUS
            and self._state.fallback_reason == reason
            and self._available == active
        ):
            if temporary:
                duration = (
                    self._state.window_s if hold_s is None else float(hold_s)
                )
                self._temporary_until_s = max(
                    self._temporary_until_s, now + duration
                )
            return self._state

        self._available = active
        self._deferred_marker = None
        if temporary:
            duration = self._state.window_s if hold_s is None else float(hold_s)
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError("hold_s must be finite and non-negative")
            self._temporary_until_s = max(
                self._temporary_until_s, now + duration
            )
            generation = self._state.generation
        else:
            self._temporary_until_s = 0.0
            configured_n = self._largest_prefill_count(len(active))
            self._config = replace(
                self._config,
                mode=MODE_CONTINUOUS,
                n_prefill_active=configured_n,
            )
            generation = self._state.generation + 1
        self._state = TemporalState(
            generation=generation,
            mode=MODE_CONTINUOUS,
            n_prefill_active=len(active),
            window_s=self._state.window_s,
            token_budget=self._state.token_budget,
            fP=self._state.fP,
            fD=self._state.fD,
            active_set=active,
            epoch=self._state.epoch,
            window_start_s=now,
            window_end_s=now + self._state.window_s,
            deferred_rotations=self._state.deferred_rotations,
            fallback_reason=reason,
        )
        self._append_event_locked("fallback", now, reason)
        return self._state

    def emergency_continuous(
        self,
        reason: str = "emergency",
        active: Optional[Sequence[int]] = None,
        now: Optional[float] = None,
        *,
        temporary: bool = False,
        hold_s: Optional[float] = None,
    ) -> TemporalState:
        """Fail safe to continuous admission over every active engine."""
        current = self._now(now)
        available = (
            None if active is None else self._normalize_active(active)
        )
        with self._lock:
            return self._fallback_locked(
                reason,
                self._available if available is None else available,
                current,
                temporary=temporary,
                hold_s=hold_s,
            )

    def snapshot(self) -> TemporalState:
        with self._lock:
            return self._state

    def snapshot_config(self) -> TemporalConfig:
        """Return the last fully published coordinator configuration."""
        with self._lock:
            return self._config

    def drain_events(self) -> List[TemporalEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events


# Explicit aliases make the public typed API discoverable under either name.
TemporalCoordinatorConfig = TemporalConfig
TemporalCoordinatorState = TemporalState
TemporalCoordinatorEvent = TemporalEvent
