"""PDBlend TP mode boundaries and cost accounting.

The first two modes are executable with existing symmetric TP1/2/4 pools. The
slow reshard mode deliberately remains experimental until a native vLLM
backend supplies generation/rank/KV/cancellation receipts.
"""
from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, asdict, replace
from enum import Enum
from threading import RLock
from typing import Callable, Iterable
from .reshard import ReshardCoordinator, TopologyTarget


class TPMode(str, Enum):
    FIXED = 'fixed_tp'
    OFFLINE = 'offline_tp'
    RESIDENT = 'resident_hetero_tp'
    SLOW_RESHARD = 'slow_reshard_tp'


@dataclass(frozen=True)
class TransitionCost:
    startup_j: float = 0.0
    drain_j: float = 0.0
    weight_transfer_j: float = 0.0
    kv_transfer_j: float = 0.0
    rollback_j: float = 0.0
    downtime_s: float = 0.0
    retries: int = 0

    def __post_init__(self):
        for field, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"transition cost {field} must be finite and nonnegative")

    @property
    def total_j(self) -> float:
        return sum((self.startup_j, self.drain_j, self.weight_transfer_j,
                    self.kv_transfer_j, self.rollback_j))


@dataclass(frozen=True)
class PoolMember:
    instance_id: str
    model_id: str
    tp: int
    pp: int
    pool_id: str
    generation: int
    accepting: bool = True
    role: str = "M"
    profile_key: str = ""
    capacity_tokens: int = 0
    max_input_tokens: int = 8192
    max_concurrent: int = 256

    def __post_init__(self):
        if self.tp not in (1, 2, 4) or self.pp != 1:
            raise ValueError("PDBlend resident members require TP1/2/4 and PP1")
        if (not self.instance_id or not self.model_id or not self.pool_id
                or self.generation < 0 or self.role not in ("M", "P", "D")
                or self.capacity_tokens < 0 or self.max_input_tokens < 1 or self.max_concurrent < 1):
            raise ValueError("invalid resident pool member")


class ResidentTPRouter:
    """Reserve bounded capacity and pin each request to one generation.

    A supplied scorer may use this pool's measured profile.  Without one the
    default is token-capacity and concurrency pressure, not a claimed energy
    optimum.  Completion/cancel callers release only after engine cleanup ACK.
    """
    def __init__(self, members: Iterable[PoolMember], *,
                 score: Callable[[PoolMember, int, int, int], float] | None = None):
        members = tuple(members)
        self.members = {m.instance_id: m for m in members}
        if len(self.members) != len(members):
            raise ValueError("duplicate resident instance id")
        self.score = score
        self.routes: list[dict] = []
        self.active: dict[str, dict] = {}
        self._tokens = {m.instance_id: 0 for m in members}
        self._requests = {m.instance_id: 0 for m in members}
        self._lock = RLock()

    def set_accepting(self, instance_id: str, accepting: bool) -> None:
        with self._lock:
            self.members[instance_id] = replace(self.members[instance_id], accepting=accepting)

    def inflight(self) -> dict[str, int]:
        with self._lock:
            return dict(self._requests)

    def route(self, request_id: str, input_tokens: int, *, prefer_tp: int | None = None,
              model_id: str | None = None, max_tokens: int = 1, prefer_pd: bool = False) -> dict | None:
        if not request_id or input_tokens < 1 or max_tokens < 1:
            raise ValueError("request id and positive token lengths are required")
        with self._lock:
            if request_id in self.active:
                receipt = self.active[request_id]
                if (receipt['input_tokens'] != input_tokens or receipt['max_tokens'] != max_tokens
                        or model_id is not None and receipt['model_id'] != model_id):
                    raise ValueError("request id is already bound to different work")
                return deepcopy(receipt)
            models = {m.model_id for m in self.members.values()}
            if model_id is None and len(models) > 1:
                raise ValueError("model_id is required for a multi-model resident router")
            reservation = input_tokens + max_tokens
            candidates = [m for m in self.members.values() if m.accepting
                          and (model_id is None or m.model_id == model_id)
                          and (prefer_tp is None or m.tp == prefer_tp)
                          and input_tokens <= m.max_input_tokens
                          and self._requests[m.instance_id] < m.max_concurrent
                          and (m.capacity_tokens == 0 or
                               self._tokens[m.instance_id] + reservation <= m.capacity_tokens)]
            choices = [(m,) for m in candidates if m.role == "M"]
            if prefer_pd:
                pairs = [(p, d) for p in candidates if p.role == "P"
                         for d in candidates if d.role == "D"
                         and (p.model_id, p.tp, p.pp, p.pool_id, p.generation) ==
                             (d.model_id, d.tp, d.pp, d.pool_id, d.generation)]
                if pairs:
                    choices = pairs
            if not choices:
                return None

            def rank(choice):
                value = 0.0
                for member in choice:
                    tokens, requests = self._tokens[member.instance_id], self._requests[member.instance_id]
                    cost = (self.score(member, input_tokens, tokens, requests) if self.score else
                            (tokens + reservation) / (member.capacity_tokens or member.max_input_tokens)
                            + requests / member.max_concurrent)
                    if not math.isfinite(cost) or cost < 0:
                        raise ValueError("resident score must be finite and nonnegative")
                    value += cost
                return value, tuple((m.tp, m.pool_id, m.instance_id) for m in choice)

            selected = min(choices, key=rank)
            first, last = selected[0], selected[-1]
            receipt = {'request_id': request_id, 'input_tokens': int(input_tokens), 'max_tokens': int(max_tokens),
                       'instance_id': last.instance_id, 'instance_ids': [m.instance_id for m in selected],
                       'prefill_instance': first.instance_id, 'decode_instance': last.instance_id,
                       'path': 'PD' if len(selected) == 2 else 'M', 'model_id': first.model_id,
                       'tp': first.tp, 'pp': first.pp, 'pool_id': first.pool_id, 'generation': first.generation,
                       'profile_keys': [m.profile_key for m in selected],
                       'kv_pairing': 'same_tp_pp1' if len(selected) == 2 else 'none',
                       'reservation_tokens': reservation, 'formal_eligible': False}
            for member in selected:
                self._requests[member.instance_id] += 1
                self._tokens[member.instance_id] += reservation
            self.active[request_id] = receipt
            self.routes.append(deepcopy(receipt))
            return deepcopy(receipt)

    def release(self, request_id: str, *, generation: int, terminal_ack: bool) -> bool:
        """Reclaim once, after a completed stream or confirmed native cancel."""
        with self._lock:
            receipt = self.active.get(request_id)
            if receipt is None:
                return False
            if terminal_ack is not True or generation != receipt['generation']:
                raise ValueError("release requires terminal ACK for the request's pinned generation")
            for instance_id in receipt['instance_ids']:
                self._requests[instance_id] -= 1
                self._tokens[instance_id] -= receipt['reservation_tokens']
            del self.active[request_id]
            return True


class SlowTPController:
    def __init__(self, coordinator: ReshardCoordinator):
        self.coordinator = coordinator
        self.transitions: list[dict] = []

    def request(self, source: TopologyTarget, target: TopologyTarget, *, cost: TransitionCost) -> dict:
        if cost.total_j <= 0:
            raise ValueError('reshard requires measured positive transition cost')
        # Cross-TP reshard is allowed only after all live KV and requests drain;
        # online P/D handoff remains symmetric within a resident pool.
        record = self.coordinator.transition(source, target)
        result = record.json()
        result.update(mode=TPMode.SLOW_RESHARD.value, transition_cost=asdict(cost),
                      total_transition_j=cost.total_j, formal_eligible=False,
                      hardware_qualified=False)
        self.transitions.append(result)
        return result
