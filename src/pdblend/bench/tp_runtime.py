"""CPU preparation for PDBlend TP modes; native launch stays in bench.run."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..control.forecast import Forecast
from ..control.planner import Plan, PlannerConfig, SLO
from ..control.policies import Policy
from ..control.topology import ResidentPool, TopologyPlanner, candidate_topologies
from ..control.tp_modes import TPMode
from ..engine.launcher import InstanceSpec, make_specs
from ..model_registry import ModelRegistry
from ..profile.model import PerfModel


class UnsupportedTPMode(RuntimeError):
    status = 'unsupported_engine'


@dataclass
class TPRuntime:
    mode: str
    specs: list[InstanceSpec]
    model: PerfModel
    profile_path: Path
    selected_plan: Plan | None = None
    pool_models: dict[str, PerfModel] = field(default_factory=dict)
    profile_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _identity(model: PerfModel, model_id: str, topology: tuple[int, int]) -> None:
    identity = model.profile_key
    if (model.system != 'pdblend' or not identity or identity.get('system') != 'pdblend'
            or identity.get('model_id') != model_id
            or (identity.get('tp'), identity.get('pp')) != topology
            or (model.tp, model.pp) != topology or Path(model.model).name != model_id):
        raise ValueError('TP runtime requires this model/system/topology\'s independent profile')
    if not identity.get('engine_revision') or not identity.get('hardware_id'):
        raise ValueError('TP runtime requires engine and hardware profile identity')
    if not model.bounded_coverage:
        raise ValueError('TP runtime requires bounded measured coverage; legacy extrapolation is disabled')
    if any(frequency not in model.decode_overrides for frequency in model.freqs):
        raise ValueError('TP runtime requires decode coverage at every permitted frequency')


def _covers(model: PerfModel, requests) -> bool:
    try:
        for request in requests:
            if request.input_tokens + request.max_tokens > 8192:
                return False
            for frequency in model.freqs:
                model.prefill_seconds(request.input_tokens, frequency)
                for context in (request.input_tokens, request.input_tokens + request.max_tokens - 1):
                    model.step_seconds(1, context, frequency)
    except ValueError as exc:
        if 'outside measured coverage' not in str(exc):
            raise
        return False
    return True


def prepare_tp_runtime(*, model_name: str, gpus: Sequence[int], fixed_tp: int, mode: str,
                       profiles: Mapping[tuple[int, int], Path], policy: Policy,
                       forecast: Forecast, slo: SLO, requests: Sequence = (), base_port: int = 8100,
                       kv_connector: str | None = 'P2pNcclConnector',
                       resident_pools: Sequence[ResidentPool] = ()) -> TPRuntime:
    mode = TPMode(mode)
    if not policy.name.startswith('pdblend'):
        raise ValueError('PDBlend TP runtime cannot execute an independent baseline')
    if mode is TPMode.SLOW_RESHARD:
        raise UnsupportedTPMode('slow_reshard_tp requires a GPU-qualified native transaction backend')
    if not gpus or len(set(gpus)) != len(gpus) or len(gpus) > 8:
        raise ValueError('TP runtime requires one to eight unique leased GPUs')
    spec = ModelRegistry().get(model_name)
    # Fixed/offline and P/D mechanism layouts need a symmetric pair budget.
    # Resident M pools are single mixed engines, so admit a TP topology when
    # its own GPUs fit the lease; the per-pool allocation checks below remain
    # strict and never invent a second P/D replica.
    resident_mode = mode is TPMode.RESIDENT
    legal = {(t.tp, t.pp) for t in candidate_topologies(
        spec, gpu_budget=len(gpus), require_pd_pair=not resident_mode)}
    models, paths = {}, {}
    for topology, path in profiles.items():
        topology = tuple(topology)
        if topology not in legal:
            raise ValueError(f'unsupported PDBlend TP/PP topology: {topology}')
        model = PerfModel.load(Path(path))
        _identity(model, spec.model_id, topology)
        models[topology], paths[topology] = model, Path(path)
    if not models:
        raise ValueError('missing_profile: no independent PDBlend topology profile')
    revisions = {(m.profile_key['engine_revision'], m.profile_key['hardware_id']) for m in models.values()}
    if len(revisions) != 1:
        raise ValueError('topology profiles have different engine or hardware identities')
    metadata = dict(tp_mode=mode.value, formal_eligible=False, policy_m_floor=policy.min_m_instances,
                    profiles={f'tp{tp}-pp{pp}': str(paths[tp, pp]) for tp, pp in models},
                    policy_floor_override=False)
    if mode is not TPMode.RESIDENT:
        covered = {topology: model for topology, model in models.items() if _covers(model, requests)}
        cfg = policy.planner_config(PlannerConfig(slots=len(gpus), slo=slo))
        planner = TopologyPlanner(spec, covered, gpu_budget=len(gpus), planner_config=cfg,
                                  require_pd_pair=True)
        plan = planner.search(forecast, slo, mode=mode.value, fixed_topology=(fixed_tp, 1))
        chosen = models[plan.tp, plan.pp]
        key = json.dumps(chosen.profile_key, sort_keys=True, separators=(',', ':'))
        specs = make_specs(model_name, gpus, tp=plan.tp, pp=1, base_port=base_port,
                           kv_connector=kv_connector if policy.allow_pd else None,
                           pool_id=f'tp{plan.tp}', generation=0, profile_key=key)
        metadata.update(selected_topology=dict(tp=plan.tp, pp=1), selection=plan.detail)
        return TPRuntime(mode.value, specs, chosen, paths[plan.tp, 1], plan,
                         profile_paths={f'tp{plan.tp}': str(paths[plan.tp, 1])}, metadata=metadata)
    if len({pool.topology.tp for pool in resident_pools}) < 2:
        raise ValueError('resident_hetero_tp requires explicit pools with at least two distinct TP values')
    if len({pool.pool_id for pool in resident_pools}) != len(resident_pools):
        raise ValueError('resident pool ids must be unique')
    specs, pool_models, pool_paths = [], {}, {}
    next_port = base_port
    available = list(gpus)
    used = set()
    for pool in resident_pools:
        topology = (pool.topology.tp, pool.topology.pp)
        if not pool.pool_id or topology not in models or pool.replicas < 1 or pool.standby or pool.role != 'M':
            raise ValueError('resident pools require measured active topology and positive replica count')
        count = pool.gpu_count
        # A declared pool allocation contains all replicas, otherwise use the
        # next unassigned leased GPUs. No hidden GPU or TP fallback is allowed.
        allocation = tuple(pool.topology.gpus) or tuple(g for g in available if g not in used)[:count]
        if (len(allocation) != count or len(set(allocation)) != count
                or not set(allocation) <= set(gpus) or set(allocation) & used):
            raise ValueError('resident pool GPU allocation is overlapping or exceeds the lease')
        used.update(allocation)
        model = models[topology]
        key = json.dumps(model.profile_key, sort_keys=True, separators=(',', ':'))
        for offset in range(pool.replicas):
            group = allocation[offset * pool.topology.tp:(offset + 1) * pool.topology.tp]
            # P2P binds one port per TP rank at HTTP port + 20000 + rank.
            # Counting instances collides for TP2 followed by TP4; reserve
            # the complete rank block even for an M-only resident pool.
            specs.append(InstanceSpec(f'{pool.pool_id}-{offset}', group, next_port, model_name,
                                      tp=pool.topology.tp, pp=1, pool_id=pool.pool_id, generation=0,
                                      profile_key=key, kv_connector=kv_connector if policy.allow_pd else None))
            next_port += pool.topology.tp
        pool_models[pool.pool_id], pool_paths[pool.pool_id] = model, str(paths[topology])
    if any(not any(_covers(model, (request,)) for model in pool_models.values()) for request in requests):
        raise ValueError('missing_profile: resident pools do not cover every request shape')
    if requests and any(not any(_covers(model, (request,)) for request in requests) for model in pool_models.values()):
        raise ValueError('missing_profile: a resident pool covers no workload request shapes')
    metadata.update(layout=[dict(instance_id=s.instance_id, tp=s.tp, pp=s.pp, pool_id=s.pool_id,
                                 generation=s.generation, gpus=list(s.gpus), profile_key=s.profile_key) for s in specs],
                    unused_metered_gpus=[g for g in gpus if g not in used],
                    bootstrap='covered_trace_per_pool_conservative', profile_paths=pool_paths)
    first = next(iter(pool_models))
    return TPRuntime(mode.value, specs, pool_models[first], Path(pool_paths[first]),
                     pool_models=pool_models, profile_paths=pool_paths, metadata=metadata)


def mechanism_pd_plan(instance_count: int, *, frequency_mhz: int = 2520, threshold_tokens: int = 1024) -> Plan:
    """Force a labeled hardware P/D probe without changing formal policy floors."""
    if instance_count < 2:
        raise ValueError('P/D mechanism probe needs at least two instances')
    counts = {'P': 1, 'D': 1}
    if instance_count > 2:
        counts['M'] = instance_count - 2
    return Plan(counts, frequency_mhz, frequency_mhz, frequency_mhz, threshold_tokens,
                0.0, 0.0, 0.0, detail=dict(mechanism_forced_roles=True, formal_eligible=False,
                                          policy_decision=False, measured_power_prediction=False))
