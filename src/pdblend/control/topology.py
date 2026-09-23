"""Topology-aware offline planning and resident-pool accounting.

The existing role/frequency planner remains useful inside one topology.  This
module adds the missing outer dimension without allowing a profile from one
TP/PP layout to be reused for another layout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Mapping

from ..model_registry import ModelSpec
from ..profile.model import PerfModel
from .forecast import Forecast
from .planner import Plan, PlannerConfig, PoolPlanner


@dataclass(frozen=True)
class Topology:
    tp: int
    pp: int = 1
    gpus: tuple[int, ...] = ()

    @property
    def gpu_count(self) -> int:
        return self.tp * self.pp

    @property
    def key(self) -> str:
        return f"tp{self.tp}-pp{self.pp}"


@dataclass(frozen=True)
class ResidentPool:
    pool_id: str
    topology: Topology
    replicas: int
    role: str = "M"
    standby: bool = False

    @property
    def gpu_count(self) -> int:
        return self.replicas * self.topology.gpu_count


def candidate_topologies(spec: ModelSpec, *, include_pp: bool = False,
                         require_memory: bool = True, gpu_budget: int = 8,
                         require_pd_pair: bool = True) -> tuple[Topology, ...]:
    """PDBlend's measured TP space, including room for a symmetric P/D pair.

    Filter PP1 before testing memory; projecting a legal TP1/PP2 layout to
    TP1/PP1 incorrectly admits 32B on one L20.  Baseline PP enumeration belongs
    to its independent planner and is deliberately not enabled here.
    """
    if include_pp:
        raise ValueError("PDBlend topology planning only supports PP1")
    if not 1 <= gpu_budget <= 8:
        raise ValueError("PDBlend requires a GPU budget between one and eight")
    return tuple(Topology(tp, pp) for tp, pp in spec.legal_topologies(
        available_gpus=gpu_budget, require_memory=require_memory)
        if pp == 1 and tp in (1, 2, 4)
        and (not require_pd_pair or 2 * tp <= gpu_budget))


def partition_pools(topologies: tuple[Topology, ...], gpu_budget: int = 8) -> tuple[tuple[Topology, ...], ...]:
    """Return deterministic non-overlapping pool shapes for resident experiments."""
    result = []
    for i, left in enumerate(topologies):
        for right in topologies[i:]:
            if left.gpu_count + right.gpu_count <= gpu_budget:
                result.append((left, right))
    return tuple(result)


class TopologyPlanner:
    def __init__(self, spec: ModelSpec, profiles: Mapping[tuple[int, int], PerfModel],
                 *, gpu_budget: int = 8, min_saving: float = .03,
                 planner_config: PlannerConfig | None = None,
                 require_pd_pair: bool = True):
        self.spec = spec
        self.profiles = dict(profiles)
        self.gpu_budget = gpu_budget
        self.min_saving = min_saving
        self.planner_config = planner_config
        self.topologies = candidate_topologies(spec, gpu_budget=gpu_budget,
                                               require_pd_pair=require_pd_pair)
        for (tp, pp), model in self.profiles.items():
            if model.system != "pdblend":
                raise ValueError("PDBlend cannot consume another system's profile")
            if (model.tp, model.pp) != (tp, pp):
                raise ValueError("profile topology does not match its lookup key")
            identity = model.profile_key
            if identity and (identity.get("model_id") != spec.model_id
                             or identity.get("system") != "pdblend"
                             or (identity.get("tp"), identity.get("pp")) != (tp, pp)):
                raise ValueError("profile identity does not match model/system/topology")

    def _planner(self, topology: Topology, model: PerfModel, slo) -> PoolPlanner:
        config = self.planner_config or PlannerConfig(slots=1, slo=slo)
        return PoolPlanner(model, replace(config, slots=self.gpu_budget // topology.gpu_count,
                                          slo=slo, freqs=model.freqs))

    def search(self, forecast: Forecast, slo, *, mode: str = "offline_tp",
               fixed_topology: tuple[int, int] | None = None) -> Plan:
        if mode not in {"fixed_tp", "offline_tp", "resident_hetero_tp", "slow_reshard_tp"}:
            raise ValueError(f"unknown TP mode: {mode}")
        if mode == "fixed_tp" and fixed_topology is None:
            raise ValueError("fixed_tp requires an explicit fixed_topology")
        candidates = []
        rejected = []
        legal = {(top.tp, top.pp) for top in self.topologies}
        for (tp, pp), model in sorted(self.profiles.items()):
            topo = Topology(tp, pp)
            if (tp, pp) not in legal:
                continue
            if mode == "fixed_tp" and fixed_topology is not None and (tp, pp) != tuple(fixed_topology):
                continue
            planner = self._planner(topo, model, slo)
            try:
                plans = planner.candidates(forecast)
            except ValueError as exc:
                if "outside measured coverage" not in str(exc):
                    raise
                rejected.append({"tp": tp, "pp": pp, "reason": str(exc)})
                continue
            if not plans:
                continue
            plan = min(plans, key=lambda p: (p.power_w, p.ttft_s, p.tpot_s))
            plan.tp, plan.pp = tp, pp
            plan.profile_key = json.dumps(model.profile_key, sort_keys=True, separators=(",", ":"))
            standby = sum(plan.counts.get(role, 0) * model.static_power_w(state)
                          for role, state in (("idle", "active_idle_reset"), ("L1", "parked"), ("off", "off")))
            plan.detail.update({"topology": {"tp": tp, "pp": pp}, "tp_mode": mode,
                                "replicas": sum(plan.counts.values()),
                                "active_replicas": plan.active(),
                                "standby_power_w": standby,
                                "gpu_count": sum(plan.counts.values()) * topo.gpu_count,
                                "formal_eligible": False})
            candidates.append(plan)
        if not candidates:
            raise ValueError(f"no qualified profile for {self.spec.model_id}")
        if mode == "fixed_tp":
            chosen = min(candidates, key=lambda p: p.power_w)
        else:
            chosen = min(candidates, key=lambda p: (p.power_w, p.tpot_s, p.ttft_s))
        chosen.detail["candidate_count"] = len(candidates)
        chosen.detail["qualified_topologies"] = [c.detail["topology"] for c in candidates]
        chosen.detail["rejected_topologies"] = rejected
        if mode == "resident_hetero_tp":
            pools = []
            for left, right in partition_pools(tuple(Topology(c.tp, c.pp) for c in candidates), self.gpu_budget):
                if left.tp == right.tp:
                    continue
                if (left.tp, left.pp) not in self.profiles or (right.tp, right.pp) not in self.profiles:
                    continue
                pools.append({"pool_id": f"{left.key}+{right.key}",
                              "topologies": [left.__dict__, right.__dict__],
                              "runtime_status": "experimental_pending_gpu_router_validation"})
            chosen.detail["resident_pools"] = pools
            chosen.detail["runtime_status"] = "experimental_requires_explicit_pool_layout"
        elif mode == "slow_reshard_tp":
            chosen.detail["transition_controller"] = "prepare/drain/transfer/verify/activate/retire/rollback"
            chosen.detail["runtime_status"] = "experimental_pending_native_backend"
        return chosen


def require_profiles(spec: ModelSpec, profiles: Mapping[tuple[int, int], PerfModel], *, include_pp: bool = False) -> None:
    required = {(t.tp, t.pp) for t in candidate_topologies(spec, include_pp=include_pp)}
    missing = sorted(required - set(profiles))
    if missing:
        raise ValueError(f"missing profile for {spec.model_id}: {missing}")
