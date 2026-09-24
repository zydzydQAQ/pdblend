"""Routing table and per-request path selection. Roles are labels the controller rewrites at will."""
from __future__ import annotations

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

ROLES = ("P", "D", "M", "parked")


class DuplicateRequestError(ValueError):
    """The request still owns a live or uncertain engine execution."""


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
    tp: int = 1
    pp: int = 1
    pool_id: str = ""
    generation: int = 0
    profile_key: str = ""
    last_token_s: Optional[float] = None
    terminal_state: Optional[str] = None
    engine_instances: set[str] = field(default_factory=set)
    route_estimate: dict = field(default_factory=dict)

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
    tp: int = 1
    pp: int = 1
    pool_id: str = ""
    generation: int = 0
    profile_key: str = ""
    model_id: str = ""


class Router:
    """Chooses (prefill, decode) instances for a request from the current role table."""

    def __init__(self, instance_ids, pd_threshold_tokens: int = 0, history: int = 20000,
                 pd_pressure_enter: float = 0.75, pd_pressure_exit: float = 0.55,
                 pd_route_hold_s: float = 0.0, pd_route_stable_windows: int = 1,
                 pd_min_input_tokens: int = 1024, instance_metadata: Optional[dict] = None):
        self.loads = {i: InstanceLoad(**(instance_metadata or {}).get(i, {})) for i in instance_ids}
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
        self._active_ids: dict[str, RequestRecord] = {}
        self.quarantined: set[str] = set()
        self._quarantine_accepting: dict[str, bool] = {}

    def set_instance_metadata(self, instance_id: str, *, tp: int, pp: int = 1, pool_id: str = "",
                              generation: int = 0, profile_key: str = "", model_id: str = "") -> None:
        if instance_id not in self.loads:
            raise KeyError(instance_id)
        if tp < 1 or pp < 1 or generation < 0:
            raise ValueError("TP and PP must be positive")
        if self.loads[instance_id].inflight_seqs or self.loads[instance_id].inflight_prefill_tokens:
            raise RuntimeError("cannot change topology identity while requests are in flight")
        self.loads[instance_id].tp, self.loads[instance_id].pp, self.loads[instance_id].pool_id = int(tp), int(pp), pool_id
        self.loads[instance_id].generation = generation
        self.loads[instance_id].profile_key = profile_key
        self.loads[instance_id].model_id = model_id

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
        if instance_id in self.quarantined:
            self._quarantine_accepting[instance_id] = accepting
            self.loads[instance_id].accepting = False
        else:
            self.loads[instance_id].accepting = accepting

    def inflight(self) -> dict[str, int]:
        return {i: l.inflight_seqs for i, l in self.loads.items()}

    def _pool(self, role: str) -> list[str]:
        return [i for i, l in self.loads.items() if l.role == role and l.accepting]

    def _least_prefill(self, ids) -> str:
        return min(ids, key=lambda i: (self.loads[i].inflight_prefill_tokens, self.loads[i].inflight_seqs))

    def _least_seqs(self, ids) -> str:
        return min(ids, key=lambda i: (self.loads[i].inflight_seqs, self.loads[i].inflight_prefill_tokens))

    def _compatible_pd(self, prefills, decodes) -> list[tuple[str, str]]:
        # P2P KV currently has no shape remap. Pair only identical TP/PP and,
        # when declared, the same resident pool generation.
        def identity(iid):
            load = self.loads[iid]
            return load.tp, load.pp, load.pool_id, load.generation, load.model_id
        return [(p, d) for p in prefills for d in decodes if p != d
                and self.loads[p].pp == 1 and identity(p) == identity(d)]

    def _least_pd(self, pairs: list[tuple[str, str]]) -> tuple[str, str]:
        return min(pairs, key=lambda pair: (self.loads[pair[0]].inflight_prefill_tokens,
                                           self.loads[pair[1]].inflight_seqs,
                                           self.loads[pair[1]].inflight_prefill_tokens, pair))

    def choose(self, input_tokens: int) -> Optional[tuple[str, str, str]]:
        """Returns (path, prefill_id, decode_id) or None if nothing accepts requests."""
        mixed, prefill, decode = self._pool("M"), self._pool("P"), self._pool("D")
        pairs = self._compatible_pd(prefill, decode)
        pd_possible = bool(pairs)
        if self.pressure_gate_enabled:
            # Commit the evaluated split, even while waiting for a mode change.
            pressure_pd = input_tokens >= max(self.pd_min_input_tokens, self.pd_threshold_tokens)
        else:
            pressure_pd = input_tokens >= self.pd_threshold_tokens
        prefer_pd = pd_possible and (not mixed or pressure_pd)
        if prefer_pd:
            return ("PD", *self._least_pd(pairs))
        if mixed:
            m = self._least_seqs(mixed)
            return "M", m, m
        if pd_possible:
            return ("PD", *self._least_pd(pairs))
        return None

    def candidates(self, input_tokens: int) -> list[tuple[str, str, str]]:
        """Enumerate the evaluated path without prematurely picking an instance."""
        chosen = self.choose(input_tokens)
        if chosen is None:
            return []
        if chosen[0] == "PD":
            return [("PD", p, d) for p, d in self._compatible_pd(self._pool("P"), self._pool("D"))]
        return [("M", m, m) for m in self._pool("M")]

    def has_request(self, request_id: str) -> bool:
        return request_id in self._active_ids

    def dispatch(self, request_id: str, input_tokens: int, max_tokens: int, *,
                 choice: Optional[tuple[str, str, str]] = None) -> Optional[RequestRecord]:
        if self.has_request(request_id):
            raise DuplicateRequestError("request_id is already active or awaiting native cleanup")
        choice = choice if choice is not None else self.choose(input_tokens)
        if choice is None:
            self.rejected += 1
            return None
        path, p, d = choice
        if (path == "PD" and (p, d) not in self._compatible_pd(self._pool("P"), self._pool("D"))):
            raise ValueError("P/D route does not match current topology/generation")
        if path == "M" and (p != d or p not in self._pool("M")):
            raise ValueError("mixed route is not accepting")
        self.loads[p].inflight_prefill_tokens += input_tokens
        self.loads[d].inflight_seqs += 1
        record = RequestRecord(request_id, path, p, d, input_tokens, max_tokens, time.time(),
                               route_pressure=self._m_pressure,
                               route_reason=("pressure_pd" if self.pressure_gate_enabled and path == "PD"
                                             else ("threshold_pd" if path == "PD" else "m_capacity")),
                               tp=self.loads[d].tp, pp=self.loads[d].pp, pool_id=self.loads[d].pool_id,
                               generation=self.loads[d].generation, profile_key=self.loads[d].profile_key)
        self.records.append(record)
        self.active[d].append(record)
        self._active_ids[request_id] = record
        for l in self.listeners:
            if hasattr(l, "arrive_request"):
                l.arrive_request(input_tokens, max_tokens, request_id=request_id)
            else:
                l.arrive(input_tokens)
        return record

    def first_token(self, record: RequestRecord, at_s: Optional[float] = None) -> None:
        self.token(record, at_s=at_s)

    def token(self, record: RequestRecord, at_s: Optional[float] = None, count: int = 1) -> None:
        if count < 1:
            return
        at_s = time.time() if at_s is None else at_s
        record.tokens_so_far += count
        record.last_token_s = at_s
        if record.first_token_s is None:
            record.first_token_s = at_s
            self.loads[record.prefill_instance].inflight_prefill_tokens -= record.input_tokens

    def _release_accounting(self, record):
        if self._active_ids.get(record.request_id) is not record:
            return
        if record.first_token_s is None:
            self.loads[record.prefill_instance].inflight_prefill_tokens -= record.input_tokens
        self.loads[record.decode_instance].inflight_seqs -= 1
        self.active[record.decode_instance].remove(record)
        del self._active_ids[record.request_id]

    def finish(self, record: RequestRecord, completion_tokens: int, error: Optional[str] = None,
               *, terminal_state: Optional[str] = None) -> None:
        if record.finished_s is not None:
            return
        state = terminal_state or ("completed" if error is None else "uncertain")
        if state not in ("completed", "rejected_before_engine", "uncertain"):
            raise ValueError("unknown terminal state")
        if state == "rejected_before_engine" and record.engine_instances:
            raise ValueError("cannot roll back after engine submission")
        record.finished_s = time.time()
        record.completion_tokens = completion_tokens
        record.error = error
        record.terminal_state = state
        if state == "uncertain":
            for iid in {record.prefill_instance, record.decode_instance}:
                self._quarantine_accepting.setdefault(iid, self.loads[iid].accepting)
                self.quarantined.add(iid)
                self.loads[iid].accepting = False
        else:
            self._release_accounting(record)
        for l in self.listeners:
            if hasattr(l, "finish_request"):
                l.finish_request(completion_tokens if error is None else 0,
                                 request_id=record.request_id, input_tokens=record.input_tokens)
            else:
                l.finish(completion_tokens if error is None else 0)

    def recover_cancel(self, record, receipts, *, engine_request_id: str) -> bool:
        if record.terminal_state != "uncertain" or self._active_ids.get(record.request_id) is not record:
            return False
        validate_cancel_receipts(record, receipts, engine_request_id=engine_request_id)
        record.terminal_state = "cancelled_acknowledged"
        record.route_estimate['native_cancel_receipts'] = receipts
        self._release_accounting(record)
        for iid in {record.prefill_instance, record.decode_instance}:
            uncertain = any(r.terminal_state == "uncertain" and iid in
                            {r.prefill_instance, r.decode_instance} for r in self._active_ids.values())
            if not uncertain:
                self.quarantined.discard(iid)
                self.loads[iid].accepting = self._quarantine_accepting.pop(iid, False)
        return True

    def recent(self, window_s: float, now: Optional[float] = None) -> list[RequestRecord]:
        now = time.time() if now is None else now
        return [r for r in self.records if r.submitted_s >= now - window_s]

    def observation_records(self, window_s: float, now: Optional[float] = None) -> list[RequestRecord]:
        records = {id(r): r for r in self.recent(window_s, now)}
        records.update((id(r), r) for rows in self.active.values() for r in rows)
        return list(records.values())


def validate_cancel_receipts(record, receipts, *, engine_request_id):
    """An HTTP response alone cannot retire native request/KV ownership."""
    from pdblend.online.native_control import validate_state
    expected = record.engine_instances or {record.prefill_instance, record.decode_instance}
    if not isinstance(receipts, dict) or set(receipts) != expected:
        raise ValueError("cancel requires native cleanup receipts for every submitted engine")
    for iid, receipt in receipts.items():
        if not isinstance(receipt, dict):
            raise ValueError("cancel receipt must contain native evidence")
        state = receipt.get("native_state", {})
        try:
            validate_state(state, generation=record.generation, tp=record.tp, pp=record.pp,
                           request_id=engine_request_id, observed_after_s=record.finished_s)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        ranks = state.get("ranks", [])
        if (receipt.get("instance_id") != iid or receipt.get("request_id") != engine_request_id
                or receipt.get("generation") != record.generation
                or receipt.get("acknowledged") is not True or receipt.get("cancelled") is not True
                or state.get("generation") != record.generation or state.get("tp") != record.tp
                or state.get("pp") != record.pp or state.get("native_evidence_complete") is not True
                or state.get("transport_healthy") is not True
                or state.get("pending_transfers") != 0 or state.get("transfer_allocations") != {}
                or len(ranks) != record.tp * record.pp
                or {r.get("rank") for r in ranks} != set(range(record.tp * record.pp))):
            raise ValueError("cancel lacks complete topology-bound native cleanup ACK")
        for key in ("all_queue", "waiting", "running", "retained_kv_requests"):
            values = state.get(key)
            if not isinstance(values, list) or engine_request_id in values:
                raise ValueError("cancel native request inventory is missing or still owns the request")
        for rank in ranks:
            if (rank.get("generation") != record.generation or rank.get("native_evidence_complete") is not True
                    or rank.get("healthy") is not True or rank.get("pending_transfers") != 0
                    or rank.get("transfer_allocations") != {}):
                raise ValueError("cancel rank cleanup ACK is incomplete")
            retained = rank.get("retained", {})
            if retained.get("receiving_transactions", 0) or engine_request_id in retained.get("held_requests", []):
                raise ValueError("cancel rank still owns request KV")


class ResidentRouter(Router):
    """Compose independent per-TP controllers behind one real proxy.

    Each child retains its own model, role plan, clocks and observations. The
    admission layer compares its current offered path with other resident pools
    using their own measured costs and available KV capacity.
    """
    def __init__(self, pools: dict[str, Router], models: dict[str, object]):
        from pdblend.online.tp_modes import PoolMember, ResidentTPRouter
        self.pools, self.models = dict(pools), dict(models)
        ids = [iid for router in pools.values() for iid in router.loads]
        if len(ids) != len(set(ids)) or set(pools) != set(models):
            raise ValueError("resident pools require unique instances and one model per pool")
        super().__init__(ids)
        self.loads = {iid: load for router in pools.values() for iid, load in router.loads.items()}
        self.active = {iid: active for router in pools.values() for iid, active in router.active.items()}
        self._owners = {iid: pool for pool, router in pools.items() for iid in router.loads}
        self.quarantined: set[str] = set()
        self.frequency_provider = lambda iid: max(models[self._owners[iid]].freqs)
        self._output_tokens = 1
        self.energy_slo = None
        self.energy_estimator = None
        self.target_shares: dict[str, float] = {}
        self.pool_dispatches = {pool: 0 for pool in pools}
        members = []
        for iid, load in self.loads.items():
            model = models[self._owners[iid]]
            if model.kv_capacity_tokens <= 0:
                raise ValueError("resident routing requires measured per-instance KV capacity")
            members.append(PoolMember(iid, load.model_id, load.tp, load.pp, load.pool_id,
                                      load.generation, profile_key=load.profile_key,
                                      capacity_tokens=model.kv_capacity_tokens))
        self.selector = ResidentTPRouter(members, score=self._score)

    def configure_energy_routing(self, *, slo=None, estimator=None):
        """Opt in only with a qualified per-route incremental-energy estimator.

        estimator(choice, context) returns {qualified, incremental_energy_j,
        profile_keys, coverage}. Context includes bounded timing, current
        frequencies, reservation/queue counts and measured PD transfer time.
        Missing qualification falls back to timing for the entire choice set.
        """
        if estimator is not None and slo is None:
            raise ValueError("energy routing requires explicit TTFT/TPOT constraints")
        self.energy_slo, self.energy_estimator = slo, estimator

    def set_target_shares(self, shares):
        if (set(shares) != set(self.pools) or any(not math.isfinite(v) or v < 0 for v in shares.values())
                or not math.isclose(sum(shares.values()), 1.0)):
            raise ValueError("target shares must cover all resident pools and sum to one")
        self.target_shares = dict(shares)
        self.pool_dispatches = dict.fromkeys(self.pools, 0)

    def dispatch_feedback(self, *, reset=False):
        counts = dict(self.pool_dispatches)
        total = sum(counts.values())
        result = dict(counts=counts, total=total,
                      shares={pool: n / total if total else 0.0 for pool, n in counts.items()},
                      target_shares=dict(self.target_shares))
        if reset:
            self.pool_dispatches = dict.fromkeys(self.pools, 0)
        return result

    def has_request(self, request_id):
        return request_id in self.selector.active or any(r.has_request(request_id) for r in self.pools.values())

    def _route_context(self, choice, input_tokens, max_tokens):
        path, p, d = choice
        model = self.models[self._owners[d]]
        frequencies = {iid: self.frequency_provider(iid) or max(model.freqs) for iid in {p, d}}
        # A queued prompt may not be queried as a fictitious oversized prompt:
        # sum measured request costs to stay within the calibration domain.
        waiting = [r for rows in self.active.values() for r in rows
                   if r.prefill_instance == p and r.first_token_s is None]
        queue_s = sum(model.prefill_seconds(r.input_tokens, frequencies[p]) for r in waiting)
        prefill_s = model.prefill_seconds(input_tokens, frequencies[p])
        batch = self.selector._requests[d] + 1
        context = max(input_tokens + max_tokens,
                      self.selector._tokens[d] / max(1, self.selector._requests[d]))
        step_s = model.step_seconds(batch, context, frequencies[d])
        transfer_s = model.transfer_seconds(input_tokens) if path == "PD" else 0.0
        return dict(input_tokens=input_tokens, max_tokens=max_tokens, batch=batch, context_tokens=context,
                    frequencies=frequencies, queued_prefill_s=queue_s,
                    queued_prefill_tokens=sum(r.input_tokens for r in waiting), prefill_s=prefill_s,
                    transfer_s=transfer_s, ttft_s=queue_s + prefill_s + transfer_s + step_s,
                    tpot_s=step_s, profile_keys=[self.loads[iid].profile_key for iid in dict.fromkeys((p, d))],
                    reservation_tokens={iid: self.selector._tokens[iid] for iid in {p, d}})

    def _score(self, member, input_tokens, tokens, requests):
        model = self.models[self._owners[member.instance_id]]
        frequency = self.frequency_provider(member.instance_id) or max(model.freqs)
        batch = requests + 1
        context = max(input_tokens, tokens / max(requests, 1))
        prefill = model.prefill_seconds(input_tokens, frequency) if member.role in ('P', 'M') else 0.0
        decode = (model.step_seconds(batch, context, frequency) * self._output_tokens
                  if member.role in ('D', 'M') else 0.0)
        return prefill + decode

    def set_roles(self, roles, pd_threshold_tokens=None):
        for pool, router in self.pools.items():
            router.set_roles({iid: role for iid, role in roles.items() if self._owners[iid] == pool},
                             pd_threshold_tokens)
        if pd_threshold_tokens is not None:
            self.pd_threshold_tokens = pd_threshold_tokens

    def dispatch(self, request_id, input_tokens, max_tokens, **kwargs):
        from dataclasses import replace
        if self.has_request(request_id):
            raise DuplicateRequestError("request_id is already active or awaiting native cleanup")
        self._output_tokens = max_tokens
        offers = {}
        allowed = set()
        for pool, router in self.pools.items():
            choices = router.candidates(input_tokens)
            if choices:
                offers[pool] = choices
                for choice in choices:
                    allowed.update(choice[1:])
        for iid, member in tuple(self.selector.members.items()):
            load = self.loads[iid]
            active = (iid in allowed and load.accepting and iid not in self.quarantined
                      and input_tokens + max_tokens <= 8192)
            updated = replace(member, accepting=active,
                              role=load.role if load.role in ('P', 'D', 'M') else member.role)
            if active:
                try:
                    self._score(updated, input_tokens, self.selector._tokens[iid], self.selector._requests[iid])
                except ValueError as exc:
                    if 'outside measured coverage' not in str(exc):
                        raise
                    updated = replace(updated, accepting=False)
            self.selector.members[iid] = updated
        has_mixed = any(choices[0][0] == 'M' for choices in offers.values())
        prefer_pd = any(choices[0][0] == 'PD' and
                        (not has_mixed or input_tokens >= self.pools[pool].pd_threshold_tokens)
                        for pool, choices in offers.items())
        routes = {((p, d) if path == 'PD' else (p,)): (path, p, d)
                  for choices in offers.values() for path, p, d in choices}
        estimates, scores = {}, None
        if self.energy_slo is not None:
            for ids, choice in routes.items():
                if any(not self.selector.members[i].accepting or
                       self.selector._tokens[i] + input_tokens + max_tokens > self.selector.members[i].capacity_tokens or
                       self.selector._requests[i] >= self.selector.members[i].max_concurrent for i in ids):
                    continue
                try:
                    context = self._route_context(choice, input_tokens, max_tokens)
                except ValueError as exc:
                    if 'coverage' not in str(exc):
                        raise
                    continue
                if context['ttft_s'] > self.energy_slo.ttft_s or context['tpot_s'] > self.energy_slo.tpot_s:
                    continue
                try:
                    energy = self.energy_estimator(choice, context) if self.energy_estimator else None
                except ValueError as exc:
                    if 'coverage' not in str(exc):
                        raise
                    energy = None
                qualified = (isinstance(energy, dict) and energy.get('qualified') is True
                             and energy.get('profile_keys') == context['profile_keys']
                             and bool(energy.get('coverage')))
                value = energy.get('incremental_energy_j') if qualified else None
                if qualified and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError("incremental energy must be finite and nonnegative")
                if qualified and (energy.get('measured_ttft_s', context['ttft_s']) > self.energy_slo.ttft_s
                                  or energy.get('measured_tpot_s', context['tpot_s']) > self.energy_slo.tpot_s):
                    continue
                estimates[ids] = dict(context, energy=energy if qualified else None)
            all_energy = bool(estimates) and all(v['energy'] is not None for v in estimates.values())
            scores = {ids: (v['energy']['incremental_energy_j'] if all_energy else
                            v['ttft_s'] + max_tokens * v['tpot_s']) for ids, v in estimates.items()}
        if self.target_shares:
            feasible = [ids for ids in routes if (scores is None or ids in scores)
                        and all(self.selector.members[i].accepting and
                                self.selector._tokens[i] + input_tokens + max_tokens <= self.selector.members[i].capacity_tokens and
                                self.selector._requests[i] < self.selector.members[i].max_concurrent for i in ids)]
            # Weighted deficit dispatch implements the outer planner's offered
            # load split. If its target pool is infeasible, serve another pool.
            if feasible:
                total = sum(self.pool_dispatches.values()) + 1
                pool = max({self._owners[ids[-1]] for ids in feasible},
                           key=lambda pool: (self.target_shares[pool] * total - self.pool_dispatches[pool], pool))
                routes = {ids: route for ids, route in routes.items() if self._owners[ids[-1]] == pool}
                prefer_pd = next(iter(routes.values()))[0] == 'PD'
        receipt = self.selector.route(request_id, input_tokens, max_tokens=max_tokens, prefer_pd=prefer_pd,
                                      allowed_routes=set(routes), choice_scores=scores)
        if receipt is None:
            self.rejected += 1
            return None
        pool = self._owners[receipt['decode_instance']]
        choice = (receipt['path'], receipt['prefill_instance'], receipt['decode_instance'])
        try:
            record = self.pools[pool].dispatch(request_id, input_tokens, max_tokens, choice=choice)
        except Exception:
            # Reservation happened synchronously; no engine can have started.
            self.selector.release(request_id, generation=receipt['generation'], terminal_ack=True)
            raise
        record.route_estimate = estimates.get(tuple(receipt['instance_ids']), {})
        self.pool_dispatches[pool] += 1
        self.records.append(record)
        return record

    def first_token(self, record, at_s=None):
        self.pools[self._owners[record.decode_instance]].first_token(record, at_s)

    def token(self, record, at_s=None, count=1):
        self.pools[self._owners[record.decode_instance]].token(record, at_s, count)

    def finish(self, record, completion_tokens, error=None, *, terminal_state=None):
        self.pools[self._owners[record.decode_instance]].finish(record, completion_tokens, error,
                                                               terminal_state=terminal_state)
        if record.terminal_state in ('completed', 'rejected_before_engine'):
            self.selector.release(record.request_id, generation=record.generation, terminal_ack=True)
        else:
            # An HTTP error alone does not prove native KV/cancel completion.
            # Keep ownership and prohibit new work until the fleet is cleaned up.
            for iid in {record.prefill_instance, record.decode_instance}:
                self.quarantined.add(iid)
                self.loads[iid].accepting = False

    def recover_cancel(self, record, receipts, *, engine_request_id):
        child = self.pools[self._owners[record.decode_instance]]
        if not child.recover_cancel(record, receipts, engine_request_id=engine_request_id):
            return False
        self.selector.release(record.request_id, generation=record.generation, terminal_ack=True)
        self.quarantined = set().union(*(r.quarantined for r in self.pools.values()))
        return True
