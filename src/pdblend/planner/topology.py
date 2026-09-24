"""Topology-aware offline planning and resident-pool accounting.

The existing role/frequency planner remains useful inside one topology.  This
module adds the missing outer dimension without allowing a profile from one
TP/PP layout to be reused for another layout.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Mapping

from pdblend.model_registry import ModelSpec
from pdblend.profile.query.model import PerfModel
from pdblend.planner.forecast import Forecast
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner


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


@dataclass
class ResidentAllocation:
    """A proposal for existing pools; no topology change or request migration."""
    shares: dict[str, float]
    plans: dict[str, Plan]
    power_w: float
    detail: dict


class ResidentAllocationPlanner:
    """Conservative two-pool allocation, using each pool's independent inner planner.

    Callers publish a returned proposal only after normal role/clock readiness
    checks. Missing measured energy produces a coverage error and preserves the
    caller's current service configuration. Native TP resharding is out of scope.
    """
    def __init__(self, pools: tuple[ResidentPool, ...], models: Mapping[str, PerfModel],
                 config: PlannerConfig, *, shares: tuple[float, ...] = tuple(i / 10 for i in range(11)),
                 require_measured_energy: bool = True, unused_static_power_w: float = 0.0):
        if len(pools) != 2 or len({p.pool_id for p in pools}) != 2:
            raise ValueError("joint resident allocation requires exactly two named pools")
        if len({p.topology.tp for p in pools}) != 2:
            raise ValueError("joint resident allocation requires distinct TP values")
        if set(models) != {p.pool_id for p in pools}:
            raise ValueError("resident layout and model keys must match")
        if not shares or any(not 0 <= share <= 1 for share in shares):
            raise ValueError("allocation shares must be within [0, 1]")
        if not math.isfinite(unused_static_power_w) or unused_static_power_w < 0:
            raise ValueError("unused resident power must be nonnegative")
        if sum(pool.gpu_count for pool in pools) > 8:
            raise ValueError("resident allocation exceeds eight GPUs")
        if not math.isfinite(config.dwell_s) or config.dwell_s <= 0:
            raise ValueError("planning horizon must be positive and finite")
        used = set()
        model_ids, revisions = set(), set()
        for pool in pools:
            model = models[pool.pool_id]
            if (not pool.pool_id or pool.replicas < 1 or pool.topology.pp != 1
                    or pool.topology.tp not in (1, 2, 4) or pool.standby or pool.role != "M"
                    or model.system != "pdblend"
                    or (model.tp, model.pp) != (pool.topology.tp, pool.topology.pp)):
                raise ValueError("resident pool identity/topology does not match independent profile")
            if pool.topology.gpus:
                allocation = set(pool.topology.gpus)
                if len(allocation) != pool.gpu_count or allocation & used:
                    raise ValueError("explicit resident GPU allocations overlap or have wrong size")
                used.update(allocation)
            model_ids.add(model.profile_key.get("model_id", model.model))
            revisions.add((model.profile_key.get("engine_revision"), model.profile_key.get("hardware_id")))
        if len(model_ids) != 1 or len(revisions) != 1:
            raise ValueError("resident profiles must share model, engine and hardware identities")
        self.pools, self.models, self.cfg = tuple(pools), dict(models), config
        self.shares = tuple(sorted(set(shares)))
        self.require_measured_energy = require_measured_energy
        self.unused_static_power_w = unused_static_power_w
        self.planners = {pool.pool_id: PoolPlanner(models[pool.pool_id], replace(
            config, slots=pool.replicas, freqs=models[pool.pool_id].freqs)) for pool in pools}
        self.observed_shares: dict[str, float] = {}

    def observe_dispatches(self, counts: Mapping[str, int]) -> None:
        """Previous interval's actual split, used as an additional candidate and audited."""
        if set(counts) - set(self.models) or any(n < 0 for n in counts.values()):
            raise ValueError("dispatch feedback has unknown pools or negative counts")
        total = sum(counts.values())
        self.observed_shares = ({key: counts.get(key, 0) / total for key in self.models}
                                if total else {})

    def _query_qualified(self, pool: ResidentPool, forecast: Forecast, plan: Plan) -> bool:
        model = self.models[pool.pool_id]
        if not model.bounded_coverage:
            return False
        if self.require_measured_energy:
            if not model.decode_power_overrides:
                return False
            if plan.counts.get("M", 0) and plan.detail.get("M", {}).get("energy_model") != "measured_mixed_window":
                return False
        # Means alone do not qualify a workload with an uncovered tail.
        # This is a conservative coverage envelope, not invented input/output
        # correlation used by the workload model.
        out_bound = max(forecast.outputs or (forecast.output_mean,))
        shapes = tuple(forecast.length_pairs) + tuple((n, out_bound) for n in
                    (forecast.inputs or (forecast.input_mean, forecast.input_p95)))
        for inp, out in shapes:
            for role, frequency in (("P", plan.f_P), ("D", plan.f_D), ("M", plan.f_M)):
                if not plan.counts.get(role, 0):
                    continue
                if plan.counts.get("P", 0) and plan.counts.get("M", 0):
                    use_pd = inp >= plan.tau
                    if (role == "M") == use_pd:
                        continue
                try:
                    if role in ("P", "M"):
                        model.prefill_seconds(int(inp), frequency)
                    if role in ("D", "M"):
                        for context in (inp, inp + max(out - 1, 0)):
                            if not model.decode_supported(1, context, frequency):
                                return False
                            # Decode power at the model's actual operating batch was
                            # checked by the inner planner. Check timing at both tails.
                            model.step_seconds(1, context, frequency)
                except ValueError as exc:
                    if "coverage" not in str(exc) and "missing_profile:" not in str(exc):
                        raise
                    return False
        return True

    def plan(self, forecast: Forecast, *, current_plans: Mapping[str, Plan] | None = None,
             current_shares: Mapping[str, float] | None = None) -> ResidentAllocation:
        current_plans = dict(current_plans or {})
        if set(current_plans) - set(self.models):
            raise ValueError("current plans contain an unknown resident pool")
        if any(not work.pool_id or work.pool_id not in self.models for work in forecast.backlog):
            raise ValueError("resident backlog must retain its known pool ownership")
        candidates, rejected = [], []
        left, right = self.pools
        fractions = set(self.shares)
        for feedback in (self.observed_shares, current_shares or {}):
            if feedback:
                if (set(feedback) != set(self.models) or any(not 0 <= x <= 1 for x in feedback.values())
                        or abs(sum(feedback.values()) - 1) > 1e-9):
                    raise ValueError("resident shares must name both pools and sum to one")
                fractions.add(feedback[left.pool_id])
        for fraction in sorted(fractions):
            shares = {left.pool_id: fraction, right.pool_id: 1.0 - fraction}
            chosen = {}
            cost = self.cfg.dwell_s * self.unused_static_power_w
            for pool in self.pools:
                planner = self.planners[pool.pool_id]
                part = forecast.for_pool(pool.pool_id, shares[pool.pool_id])
                # Retain all resident replicas in every accounting path, even
                # when they receive no new arrivals. No uncharged standby pool.
                proposals = planner.candidates(part)
                proposals = [p for p in proposals if self._query_qualified(pool, part, p)]
                if not proposals:
                    rejected.append(dict(shares=shares, pool_id=pool.pool_id,
                                         reason="no SLO- and coverage-qualified inner plan"))
                    break
                for proposal in proposals:
                    proposal.tp, proposal.pp, proposal.pool_id = pool.topology.tp, pool.topology.pp, pool.pool_id
                    proposal.generation = current_plans[pool.pool_id].generation if pool.pool_id in current_plans else 0
                    proposal.profile_key = json.dumps(self.models[pool.pool_id].profile_key, sort_keys=True, separators=(",", ":"))
                def objective(p):
                    return self.cfg.dwell_s * p.power_w + planner.switch_energy_j(current_plans.get(pool.pool_id), p)
                proposals = [p for p in proposals if math.isfinite(objective(p))]
                if not proposals:
                    rejected.append(dict(shares=shares, pool_id=pool.pool_id,
                                         reason="no qualified transition for inner plan"))
                    break
                best = min(proposals, key=lambda p: (objective(p), p.power_w))
                best.tp, best.pp, best.pool_id = pool.topology.tp, pool.topology.pp, pool.pool_id
                best.profile_key = json.dumps(self.models[pool.pool_id].profile_key, sort_keys=True, separators=(",", ":"))
                chosen[pool.pool_id] = best
                cost += objective(best)
            if len(chosen) == len(self.pools):
                power = self.unused_static_power_w + sum(p.power_w for p in chosen.values())
                candidates.append(ResidentAllocation(shares, chosen, power, dict(
                    total_energy_j=cost, horizon_s=self.cfg.dwell_s,
                    transition_energy_j=cost - self.cfg.dwell_s * power,
                    observed_shares=dict(self.observed_shares), query_qualified=True,
                    measured_energy_required=self.require_measured_energy,
                    formal_eligible=False, native_tp_change=False)))
        if not candidates:
            raise ValueError("missing_profile: no qualified joint resident allocation")
        best = min(candidates, key=lambda proposal: (proposal.detail["total_energy_j"],
                                                     proposal.power_w))
        # Reevaluate the actual current configuration independently: the best
        # inner plan for the same share need not be the current one.
        if current_shares and len(current_plans) == 2:
            retained = {}
            for pool in self.pools:
                old = current_plans[pool.pool_id]
                part = forecast.for_pool(pool.pool_id, current_shares[pool.pool_id])
                evaluated = self.planners[pool.pool_id].evaluate(
                    old.counts, old.f_P, old.f_D, old.f_M, old.tau, part)
                if evaluated is None or not self._query_qualified(pool, part, evaluated):
                    break
                evaluated.tp, evaluated.pp, evaluated.pool_id = old.tp, old.pp, old.pool_id
                evaluated.generation, evaluated.profile_key = old.generation, old.profile_key
                retained[pool.pool_id] = evaluated
            if len(retained) == 2:
                power = self.unused_static_power_w + sum(p.power_w for p in retained.values())
                if best.detail["total_energy_j"] >= power * self.cfg.dwell_s * (1 - self.cfg.margin):
                    best = ResidentAllocation(dict(current_shares), retained, power, dict(
                        total_energy_j=self.cfg.dwell_s * power, horizon_s=self.cfg.dwell_s,
                        transition_energy_j=0.0, retained_current=True,
                        observed_shares=dict(self.observed_shares), query_qualified=True,
                        measured_energy_required=self.require_measured_energy,
                        formal_eligible=False, native_tp_change=False))
        best.detail.update(candidate_count=len(candidates), rejected=rejected,
                           unused_static_power_w=self.unused_static_power_w,
                           resident_replicas={p.pool_id: p.replicas for p in self.pools})
        return best
