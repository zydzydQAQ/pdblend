"""Independent fixed-TP Mixed least-load policy.

Mixed has no P/D or TP reconfiguration.  The policy only selects among
identical fixed-TP replicas and records routing decisions for the baseline
manifest; it does not consume PDblend's profile or planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class MixedReplica:
    instance_id: str
    tp: int
    max_num_seqs: int = 256
    active_requests: int = 0
    accepting: bool = True
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.instance_id or self.tp < 1 or self.max_num_seqs < 1 or self.active_requests < 0:
            raise ValueError("Mixed replica identity, TP and queue limits must be positive")


@dataclass(frozen=True)
class MixedRoute:
    request_id: str
    instance_id: str
    tp: int
    load: float
    policy: str = "fixed_tp_least_load"


@dataclass
class MixedLeastLoadPolicy:
    tp: int
    routes: list[MixedRoute] = field(default_factory=list)

    def route(self, request_id: str, replicas: Iterable[MixedReplica]) -> MixedRoute | None:
        candidates = [replica for replica in replicas
                      if replica.accepting and replica.tp == self.tp
                      and replica.active_requests < replica.max_num_seqs]
        if not candidates:
            return None
        selected = min(candidates, key=lambda replica: (
            replica.active_requests / replica.max_num_seqs,
            replica.active_requests,
            replica.instance_id))
        load = selected.active_requests / selected.max_num_seqs
        selected.active_requests += 1
        result = MixedRoute(request_id, selected.instance_id, selected.tp, load)
        self.routes.append(result)
        return result

    @staticmethod
    def complete(instance_id: str, replicas: Iterable[MixedReplica]) -> None:
        for replica in replicas:
            if replica.instance_id == instance_id:
                replica.active_requests = max(0, replica.active_requests - 1)
                return
        raise KeyError(instance_id)
