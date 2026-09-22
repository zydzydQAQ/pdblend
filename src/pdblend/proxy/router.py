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

    def __init__(self, instance_ids, pd_threshold_tokens: int = 0, history: int = 20000):
        self.loads = {i: InstanceLoad() for i in instance_ids}
        self.active: dict[str, list[RequestRecord]] = {i: [] for i in instance_ids}   # in flight, by decode instance
        self.pd_threshold_tokens = pd_threshold_tokens
        self.records: deque[RequestRecord] = deque(maxlen=history)
        self.rejected = 0
        self.listeners: list = []       # objects with arrive(input_tokens) / finish(output_tokens)

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
        prefer_pd = pd_possible and (not mixed or input_tokens >= self.pd_threshold_tokens)
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
        record = RequestRecord(request_id, path, p, d, input_tokens, max_tokens, time.time())
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
