"""Routing table and per-request path selection. Roles are labels the controller rewrites at will."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

ROLES = ("P", "D", "M", "parked")


@dataclass
class RequestRecord:
    request_id: str
    path: str                     # "M" or "PD"
    prefill_instance: str
    decode_instance: str
    input_tokens: int
    max_tokens: int
    submitted_s: float
    first_token_s: Optional[float] = None
    finished_s: Optional[float] = None
    completion_tokens: int = 0
    tokens_so_far: int = 0        # streamed tokens seen so far (in-flight TPOT for the shield)
    error: Optional[str] = None
    route_pressure: float = 0.0
    route_reason: str = ""

    @property
    def ttft_s(self) -> Optional[float]:
        return None if self.first_token_s is None else self.first_token_s - self.submitted_s

    @property
    def tpot_s(self) -> Optional[float]:
        if self.first_token_s is None or self.finished_s is None or self.completion_tokens < 2:
            return None
        return (self.finished_s - self.first_token_s) / (self.completion_tokens - 1)


@dataclass
class InstanceLoad:
    role: str = "M"
    inflight_prefill_tokens: int = 0     # prompts dispatched, first token not yet seen
    inflight_seqs: int = 0               # sequences dispatched, not yet finished
    accepting: bool = True


class Router:
    """Chooses (prefill, decode) instances for a request from the current role table."""

    def __init__(self, instance_ids, pd_threshold_tokens: int = 0, history: int = 20000,
                 pd_pressure_enter: float = 0.75, pd_pressure_exit: float = 0.55,
                 pd_route_hold_s: float = 0.0, pd_route_stable_windows: int = 1,
                 pd_min_input_tokens: int = 1024):
        self.loads = {i: InstanceLoad() for i in instance_ids}
        self.active: dict[str, list[RequestRecord]] = {i: [] for i in instance_ids}   # in flight, by decode instance
        self.pd_threshold_tokens = pd_threshold_tokens
        self.records: deque[RequestRecord] = deque(maxlen=history)
        self.rejected = 0
        self.listeners: list = []       # objects with arrive(input_tokens) / finish(output_tokens)
        self.pd_pressure_enter = float(pd_pressure_enter)
        self.pd_pressure_exit = float(pd_pressure_exit)
        self.pd_route_hold_s = float(pd_route_hold_s)
        self.pd_route_stable_windows = max(1, int(pd_route_stable_windows))
        self.pd_min_input_tokens = int(pd_min_input_tokens)
        self._m_pressure = 0.0
        self._decode_risk = False
        self._prefill_risk = False
        self._shield_active = False
        self._pd_pressure_active = False
        self._pd_pressure_until = 0.0
        self._pd_clear_windows = 0
        self._pd_last_window = None
        self._pd_route_reason = "disabled"
        self.pressure_gate_enabled = False

    def configure_pressure_gate(self, *, enter: Optional[float] = None, exit: Optional[float] = None,
                                hold_s: Optional[float] = None, stable_windows: Optional[int] = None,
                                min_input_tokens: Optional[int] = None) -> None:
        """Configure the PDblend-only pressure gate; defaults preserve legacy routing."""
        self.pressure_gate_enabled = True
        if enter is not None:
            self.pd_pressure_enter = float(enter)
        if exit is not None:
            self.pd_pressure_exit = float(exit)
        if hold_s is not None:
            self.pd_route_hold_s = float(hold_s)
        if stable_windows is not None:
            self.pd_route_stable_windows = max(1, int(stable_windows))
        if min_input_tokens is not None:
            self.pd_min_input_tokens = int(min_input_tokens)

    def set_pressure_state(self, *, m_pressure: float = 0.0, decode_risk: bool = False,
                           prefill_risk: bool = False, shield_active: bool = False,
                           now: Optional[float] = None, stable_window: bool = False,
                           reason: str = "") -> bool:
        """Update the pressure-aware PD gate and return whether its mode changed.

        The controller supplies one stable_window tick per planning period. The
        actual route mode is committed with the matching evaluated plan, so the
        router never silently changes the planner's assumed load split.
        """
        if not self.pressure_gate_enabled:
            return False
        now = time.time() if now is None else now
        self._m_pressure = max(0.0, float(m_pressure))
        self._decode_risk = bool(decode_risk)
        self._prefill_risk = bool(prefill_risk)
        self._shield_active = bool(shield_active)
        high = (self._shield_active or self._m_pressure >= self.pd_pressure_enter
                or self._decode_risk or self._prefill_risk)
        changed = False
        if high:
            if not self._pd_pressure_active:
                changed = True
                self._pd_route_reason = reason or "pressure_enter"
            self._pd_pressure_active = True
            self._pd_pressure_until = max(self._pd_pressure_until, now + self.pd_route_hold_s)
            self._pd_clear_windows = 0
        elif self._pd_pressure_active:
            if stable_window and now != self._pd_last_window:
                self._pd_last_window = now
                if self._m_pressure <= self.pd_pressure_exit:
                    self._pd_clear_windows += 1
                else:
                    self._pd_clear_windows = 0
            if now >= self._pd_pressure_until and self._pd_clear_windows >= self.pd_route_stable_windows:
                self._pd_pressure_active = False
                self._pd_route_reason = reason or "pressure_exit"
                changed = True
        return changed

    def pressure_state(self) -> dict:
        return dict(m_pressure=self._m_pressure, decode_risk=self._decode_risk,
                    prefill_risk=self._prefill_risk, shield_active=self._shield_active,
                    pd_active=self._pd_pressure_active, pd_until=self._pd_pressure_until,
                    clear_windows=self._pd_clear_windows, reason=self._pd_route_reason)

    def set_roles(self, roles: dict[str, str], pd_threshold_tokens: Optional[int] = None) -> None:
        for instance_id, role in roles.items():
            if role not in ROLES:
                raise ValueError(role)
            self.loads[instance_id].role = role
        if pd_threshold_tokens is not None:
            self.pd_threshold_tokens = pd_threshold_tokens

    def roles(self) -> dict[str, str]:
        return {i: l.role for i, l in self.loads.items()}

    def set_accepting(self, instance_id: str, accepting: bool) -> None:
        self.loads[instance_id].accepting = accepting

    def inflight(self) -> dict[str, int]:
        return {i: l.inflight_seqs for i, l in self.loads.items()}

    def _pool(self, role: str) -> list[str]:
        return [i for i, l in self.loads.items() if l.role == role and l.accepting]

    def _least_prefill(self, ids) -> str:
        return min(ids, key=lambda i: (self.loads[i].inflight_prefill_tokens, self.loads[i].inflight_seqs))

    def _least_seqs(self, ids) -> str:
        return min(ids, key=lambda i: (self.loads[i].inflight_seqs, self.loads[i].inflight_prefill_tokens))

    def choose(self, input_tokens: int) -> Optional[tuple[str, str, str]]:
        """Returns (path, prefill_id, decode_id) or None if nothing accepts requests."""
        mixed, prefill, decode = self._pool("M"), self._pool("P"), self._pool("D")
        pd_possible = bool(prefill and decode)
        if self.pressure_gate_enabled:
            # Commit the evaluated split, even while waiting for a mode change.
            pressure_pd = input_tokens >= max(self.pd_min_input_tokens, self.pd_threshold_tokens)
        else:
            pressure_pd = input_tokens >= self.pd_threshold_tokens
        prefer_pd = pd_possible and (not mixed or pressure_pd)
        if prefer_pd:
            return "PD", self._least_prefill(prefill), self._least_seqs(decode)
        if mixed:
            m = self._least_seqs(mixed)
            return "M", m, m
        if pd_possible:
            return "PD", self._least_prefill(prefill), self._least_seqs(decode)
        return None

    def dispatch(self, request_id: str, input_tokens: int, max_tokens: int) -> Optional[RequestRecord]:
        choice = self.choose(input_tokens)
        if choice is None:
            self.rejected += 1
            return None
        path, p, d = choice
        self.loads[p].inflight_prefill_tokens += input_tokens
        self.loads[d].inflight_seqs += 1
        record = RequestRecord(request_id, path, p, d, input_tokens, max_tokens, time.time(),
                               route_pressure=self._m_pressure,
                               route_reason=("pressure_pd" if self.pressure_gate_enabled and path == "PD"
                                             else ("threshold_pd" if path == "PD" else "m_capacity")))
        self.records.append(record)
        self.active[d].append(record)
        for l in self.listeners:
            l.arrive(input_tokens)
        return record

    def first_token(self, record: RequestRecord, at_s: Optional[float] = None) -> None:
        record.tokens_so_far += 1
        if record.first_token_s is None:
            record.first_token_s = at_s or time.time()
            self.loads[record.prefill_instance].inflight_prefill_tokens -= record.input_tokens

    def finish(self, record: RequestRecord, completion_tokens: int, error: Optional[str] = None) -> None:
        if record.first_token_s is None:
            self.loads[record.prefill_instance].inflight_prefill_tokens -= record.input_tokens
        record.finished_s = time.time()
        record.completion_tokens = completion_tokens
        record.error = error
        self.loads[record.decode_instance].inflight_seqs -= 1
        self.active[record.decode_instance].remove(record)
        for l in self.listeners:
            l.finish(completion_tokens if error is None else 0)

    def recent(self, window_s: float, now: Optional[float] = None) -> list[RequestRecord]:
        now = now or time.time()
        return [r for r in self.records if r.submitted_s >= now - window_s]
