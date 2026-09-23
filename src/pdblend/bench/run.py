"""One benchmark point: fleet + proxy + controller + open-loop load with 8-GPU power metering."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

from aiohttp import web

from ..control.controller import Controller
from ..control.forecast import Forecast, Forecaster
from ..control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from ..control.policies import Policy, get_policy
from ..control.policies.baselines import EcoRouter, build_control
from ..control.shield import Shield
from ..control.topology import ResidentPool
from ..engine.client import PDTransfer
from ..engine.launcher import Fleet, make_specs
from ..profile.model import PerfModel
from pdblend.profile.query.versions import load_profile
from pdblend.model_registry import ModelRegistry
from ..proxy.router import ResidentRouter, Router
from ..proxy.server import Proxy
from .client import LoadClient, Request, dump_outcomes, nearest_rank, poisson_trace, slo_attainment, trace_summary
from .metering import Gpus
from .tp_runtime import UnsupportedTPMode, _covers, prepare_tp_runtime
from pdblend.online.native_control import NativeControl
from pdblend.online.transition_measurement import measure_transitions
from pdblend.online.observations import backlog_snapshot
from pdblend.online.resident_control import ResidentCoordinator
from pdblend.planner.topology import ResidentAllocationPlanner


def offline_forecast(trace: list[Request]) -> Forecast:
    s = trace_summary(trace)
    return Forecast(s["mean_rps"], 0.0, s["input_mean"], s["input_p95"], s["output_mean"], 0,
                    tuple(r.input_tokens for r in trace), tuple(r.max_tokens for r in trace),
                    length_pairs=tuple((r.input_tokens, r.max_tokens) for r in trace))


def window_energy(samples, start: float, end: float) -> tuple[float, float]:
    """Trapezoid energy and mean power over [start, end] from (t, [w...]) rows."""
    rows = [(t, sum(w)) for t, w in samples if start <= t <= end]
    if len(rows) < 2:
        return 0.0, 0.0
    e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(rows, rows[1:]))
    return e, e / (rows[-1][0] - rows[0][0])


def series_stats(values) -> dict:
    vals = [float(v) for v in values if v is not None]
    return {"mean": (sum(vals) / len(vals) if vals else None),
            "p50": nearest_rank(vals, .50), "p90": nearest_rank(vals, .90),
            "p95": nearest_rank(vals, .95), "p99": nearest_rank(vals, .99),
            "max": (max(vals) if vals else None), "samples": len(vals),
            "missing": 0}


def gpu_series_stats(samples, gpus: Sequence[int]) -> dict:
    rows = [row for _, row in samples]
    fleet = [v for row in rows for v in row]
    per_gpu = {str(g): series_stats([row[i] for row in rows if len(row) > i]) for i, g in enumerate(gpus)}
    active = [sum(1 for v in row if float(v) > 1.0) / max(len(gpus), 1) for row in rows]
    return {"fleet": series_stats(fleet), "per_gpu": per_gpu,
            "active_gpu_fraction": series_stats(active)}


async def _serve_proxy(proxy: Proxy, port: int) -> web.AppRunner:
    runner = web.AppRunner(proxy.app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def _make_controller(fleet, router, gpus, model, policy, slo, trace, out_dir, period_s,
                     fixed_plan=None, min_warm_s=20.0, initial_plan=None, native_control=None,
                     transition_catalog_path=None, capacity_floor_path=None, transition_qualified_only=False):
    cfg = policy.planner_config(PlannerConfig(slots=len(fleet.instances), slo=slo, freqs=model.freqs))
    cfg.pressure_controls = policy.dynamic_m_floor
    if transition_catalog_path is not None or capacity_floor_path is not None:
        if not policy.name.startswith('pdblend'):
            raise ValueError('optimization artifacts require the PDBlend policy')
        from pdblend.planner.capacity import load_capacity_floors, select_artifact
        from pdblend.planner.transitions import TransitionCatalog
        if capacity_floor_path is not None:
            cfg.capacity_floors = load_capacity_floors(capacity_floor_path, model=model)
        if transition_catalog_path is not None:
            cfg.transition_estimator = TransitionCatalog.load(select_artifact(transition_catalog_path, model),
                model=model, qualified_only=transition_qualified_only)
    planner = PoolPlanner(model, cfg)
    freeze = policy.freeze or fixed_plan is not None
    prior = offline_forecast(trace) if policy.bootstrap_forecast else None
    initial = fixed_plan or initial_plan or (planner.plan(prior or offline_forecast(trace)) if (policy.freeze or policy.warm_start) else None)
    if policy.dynamic_m_floor and initial is not None and fixed_plan is None:
        demand = prior or offline_forecast(trace)
        pressure = planner.mixed_pressure(demand, initial.counts.get('M', 0), initial.f_M)
        cfg = planner.cfg
        cfg.pd_pressure_active = (demand.input_p95 >= cfg.pd_min_input_tokens
                                  and pressure['pressure'] >= policy.pd_pressure_enter)
        initial = planner.plan(demand)
    if policy.ported and fixed_plan is None:
        planner, initial, freeze, ported_period = build_control(policy.name, planner, router, offline_forecast(trace))
        period_s = ported_period or period_s
    ctl = Controller(fleet, router, gpus, planner,
                     Shield(slo, protect_s=policy.shield_protect_s) if policy.shield else None,
                     Forecaster(initial=prior),
                     period_s=period_s, log_path=out_dir / "controller.jsonl", initial_plan=initial, freeze=freeze,
                     hold_initial=policy.warm_start, min_warm_s=min_warm_s,
                     min_plan_hold_s=policy.plan_hold_s, down_plan_votes=policy.down_plan_votes,
                     home_margin=policy.home_margin,
                     dynamic_m_floor=policy.dynamic_m_floor,
                     base_m_floor=policy.min_m_instances,
                     low_load_m_floor=policy.low_load_min_m_instances,
                     m_floor_pressure_enter=policy.m_floor_pressure_enter,
                     m_floor_pressure_exit=policy.m_floor_pressure_exit,
                     m_floor_stable_windows=policy.m_floor_stable_windows,
                     m_floor_hold_s=policy.m_floor_hold_s,
                     pd_pressure_enter=policy.pd_pressure_enter,
                     pd_pressure_exit=policy.pd_pressure_exit,
                     pd_route_hold_s=policy.pd_route_hold_s,
                     pd_route_stable_windows=policy.pd_route_stable_windows,
                     shield_protect_s=policy.shield_protect_s,
                     transition_cooldown_s=policy.transition_cooldown_s,
                     native_control=native_control)
    return ctl


class _PoolControllers:
    def __init__(self, controllers, *, coordinator=None, forecaster=None, period_s=10.):
        self.controllers = controllers
        self.coordinator, self.forecaster, self.period_s = coordinator, forecaster, period_s

    @property
    def roles(self):
        return {iid: role for controller in self.controllers.values() for iid, role in controller.roles.items()}

    async def run(self, stop):
        if self.coordinator is not None:
            return await self._run_joint(stop)
        tasks = [asyncio.create_task(controller.run(stop)) for controller in self.controllers.values()]
        try:
            await asyncio.gather(*tasks)
        finally:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_joint(self, stop):
        # One sequencer owns child transitions and joint publication. Shield
        # actions cannot race a plan that was evaluated against old roles.
        await asyncio.gather(*(c.execute(c.initial_plan or c._fail_open_plan())
                               for c in self.controllers.values()))
        next_plan = time.time() + self.period_s
        while not stop.is_set():
            await asyncio.sleep(min(c.tick_s for c in self.controllers.values()))
            now = time.time()
            for c in self.controllers.values():
                c.forecaster.set_backlog(backlog_snapshot(c.router))
                if c.shield is not None:
                    pressure = c.shield.observe(c.router.observation_records(60., now), now)
                    level = c.shield.update(pressure, now)
                    if level or c.shield.floor_active:
                        safe = c.shield.apply(c.plan_now, pressure, c.max_freq)
                        if safe.key() != c.plan_now.key():
                            await c.execute(safe)
            if now >= next_plan:
                self.forecaster.set_backlog(backlog_snapshot(self.coordinator.router))
                await self.coordinator.step(self.forecaster.forecast(now))
                next_plan = time.time() + self.period_s

    def summary(self):
        return {'mode': 'resident_hetero_tp', 'pools': {pool: ctl.summary() for pool, ctl in self.controllers.items()},
                'joint_optimization': self.coordinator is not None,
                'joint_events': self.coordinator.events if self.coordinator else []}

    @property
    def transition_events(self):
        return [dict(row, pool_id=pool) for pool, controller in self.controllers.items()
                for row in controller.transition_events]


async def _point(fleet: Fleet, gpus: Gpus, model: PerfModel, policy: Policy, slo: SLO, trace: list[Request],
                 warmup: list[Request], out_dir: Path, proxy_port: int, period_s: float, tail_timeout_s: float,
                 fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0,
                 sampling_seed: Optional[int] = None, *, initial_plan: Optional[Plan] = None,
                 pool_models: Optional[dict[str, PerfModel]] = None,
                 pool_fixed_plans: Optional[dict[str, Plan]] = None,
                 planning_trace: Optional[list[Request]] = None,
                 observation_duration_s: Optional[float] = None,
                 joint_resident: bool = False,
                 resident_pools: Sequence[ResidentPool] = (),
                 incremental_energy_path: Optional[Path] = None,
                 transition_catalog_path: Optional[Path] = None,
                 capacity_floor_path: Optional[Path] = None,
                 transition_qualified_only: bool = False) -> dict:
    selection_trace = trace if planning_trace is None else planning_trace
    if not selection_trace:
        raise ValueError('nonempty calibration/tuning bootstrap trace required')
    if observation_duration_s is not None and (observation_duration_s <= 0
            or any(r.arrival_s >= observation_duration_s for r in trace)):
        raise ValueError('observation duration must contain all evaluation arrivals')
    urls = {iid: inst.spec.base_url for iid, inst in fleet.instances.items()}
    specs = [inst.spec for inst in fleet.instances.values()]
    native = NativeControl({s.instance_id: s for s in specs}) if policy.name.startswith('pdblend') else None
    transfer = PDTransfer(specs[0].kv_connector, {s.instance_id: s.zmq_address for s in specs})
    metadata = {s.instance_id: dict(tp=getattr(s, 'tp', 1), pp=getattr(s, 'pp', 1),
                                   pool_id=getattr(s, 'pool_id', ''), generation=getattr(s, 'generation', 0),
                                   profile_key=getattr(s, 'profile_key', ''), model_id=Path(getattr(s, 'model', model.model)).name)
                for s in specs}
    if pool_models:
        if fixed_plan is not None:
            raise ValueError('resident layouts require per-pool fixed plans, not a fleet-wide role count')
        pool_routers, controllers = {}, {}
        for pool_id, pool_model in pool_models.items():
            ids = [s.instance_id for s in specs if s.pool_id == pool_id]
            sub_fleet = Fleet([], out_dir / 'logs')
            sub_fleet.instances = {iid: fleet.instances[iid] for iid in ids}
            sub_router = Router(ids, instance_metadata={iid: metadata[iid] for iid in ids})
            pool_dir = out_dir / 'pools' / pool_id
            pool_dir.mkdir(parents=True, exist_ok=True)
            pool_trace = [request for request in selection_trace if _covers(pool_model, (request,))]
            if not pool_trace:
                raise ValueError(f'missing_profile: resident pool {pool_id} covers no workload requests')
            controllers[pool_id] = _make_controller(
                sub_fleet, sub_router, gpus, pool_model, policy, slo, pool_trace, pool_dir, period_s,
                (pool_fixed_plans or {}).get(pool_id), min_warm_s, native_control=native,
                transition_catalog_path=transition_catalog_path, capacity_floor_path=capacity_floor_path,
                transition_qualified_only=transition_qualified_only)
            pool_routers[pool_id] = sub_router
        router = ResidentRouter(pool_routers, pool_models)
        router.frequency_provider = lambda iid: controllers[router._owners[iid]].freqs.get(iid)
        ctl = _PoolControllers(controllers)
        if incremental_energy_path is not None:
            from pdblend.online.energy_routing import load_energy_estimator
            router.configure_energy_routing(slo=slo, estimator=load_energy_estimator(incremental_energy_path, router=router))
        if joint_resident:
            if pool_fixed_plans or policy.dynamic_m_floor:
                raise ValueError('joint resident optimization requires ordinary periodic inner plans')
            if set(fleet.instances) != set(router.loads) or sum(p.gpu_count for p in resident_pools) != len(gpus.gpus):
                raise ValueError('joint resident optimization must account for every metered GPU')
            cfg = policy.planner_config(PlannerConfig(slots=len(specs), slo=slo, freqs=model.freqs))
            outer = ResidentAllocationPlanner(tuple(resident_pools), pool_models, cfg)
            for pool_id, inner in outer.planners.items():
                inner.cfg.capacity_floors = controllers[pool_id].planner.cfg.capacity_floors
                inner.cfg.transition_estimator = controllers[pool_id].planner.cfg.transition_estimator
            coordinator = ResidentCoordinator(controllers, router, outer)
            forecaster = Forecaster(initial=offline_forecast(selection_trace))
            for child in pool_routers.values():
                child.listeners.append(forecaster)
            ctl = _PoolControllers(controllers, coordinator=coordinator, forecaster=forecaster, period_s=period_s)
    else:
        if joint_resident or incremental_energy_path is not None:
            raise ValueError('joint/energy routing requires explicit resident pools')
        router = (EcoRouter(list(urls), model, slo) if policy.name == 'ecoserve'
                  else Router(list(urls), instance_metadata=metadata))
        ctl = _make_controller(fleet, router, gpus, model, policy, slo, selection_trace, out_dir, period_s,
                               fixed_plan, min_warm_s, initial_plan, native_control=native,
                               transition_catalog_path=transition_catalog_path, capacity_floor_path=capacity_floor_path,
                               transition_qualified_only=transition_qualified_only)
    proxy = Proxy(urls, router, transfer=transfer, native_cancel=native.cancel if native else None,
                  cancel_timeout_s=native.timeout_s + 5 if native else 15)
    runner = await _serve_proxy(proxy, proxy_port)
    # Start the common sampler before initial controller actions; evaluation
    # energy remains restricted to t_start below, setup costs stay separate.
    sampler = gpus.sampler(interval_s=0.1)
    sampler.start()
    stop = asyncio.Event()
    ctl_task = asyncio.create_task(ctl.run(stop))
    try:
        await asyncio.sleep(2.0)
        load = LoadClient(f"http://127.0.0.1:{proxy_port}", sampling_seed=sampling_seed)
        if warmup:
            await load.replay(warmup, progress_every_s=1e9)
            while any(l.inflight_seqs for l in router.loads.values()):
                await asyncio.sleep(0.2)
        await asyncio.sleep(1.0)
        t_start = time.time()
        outcomes = await load.replay_detached(trace)
        t_load_end = (t_start + observation_duration_s if observation_duration_s is not None
                      else t_start + trace[-1].arrival_s if trace else time.time())
        if observation_duration_s is not None:
            await asyncio.sleep(max(0., t_load_end-time.time()))
        t_requests_done = time.time()
        # Requests are all finished (replay awaits them); keep sampling briefly so the tail is captured.
        await asyncio.sleep(1.0)
        stop.set()
        await ctl_task
        terminal_native = {}
        if native:
            for iid, role in ctl.roles.items():
                if role != 'off':
                    terminal_native[iid] = await native.drain(iid, tail_timeout_s)
            (out_dir / 'native-cleanup.json').write_text(json.dumps(terminal_native, indent=2) + '\n')
        t_done = time.time()
    finally:
        stop.set()
        if not ctl_task.done():
            ctl_task.cancel()
        await asyncio.gather(ctl_task, return_exceptions=True)
        sampler.stop()
        await runner.cleanup()
    dump_outcomes(outcomes, out_dir / "outcomes.jsonl")
    (out_dir / 'routes.jsonl').write_text(''.join(json.dumps(dict(
        request_id=r.request_id, input_tokens=r.input_tokens, path=r.path,
        submitted_s=r.submitted_s, m_pressure=r.route_pressure, reason=r.route_reason,
        prefill_instance=r.prefill_instance, decode_instance=r.decode_instance,
        tp=r.tp, pp=r.pp, pool_id=r.pool_id, generation=r.generation, profile_key=r.profile_key,
        terminal_state=r.terminal_state, last_token_s=r.last_token_s, route_estimate=r.route_estimate)) + '\n'
        for r in router.records))
    if isinstance(router, ResidentRouter):
        (out_dir / 'resident-routes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in router.selector.routes))
    total_j, total_w = window_energy(sampler.samples, t_start, t_done)
    win_j, win_w = window_energy(sampler.samples, t_start, t_load_end)
    per_gpu = {}
    if sampler.samples:
        for i, g in enumerate(gpus.gpus):
            per_gpu[str(g)] = sum(w[i] for _, w in sampler.samples) / len(sampler.samples)
    att = slo_attainment(outcomes, slo.ttft_s, slo.tpot_s)
    (out_dir / "power.jsonl").write_text("\n".join(json.dumps([t, w]) for t, w in sampler.samples) + "\n")
    if sampler.utilization_samples:
        util_text = "\n".join(json.dumps([t, u]) for t, u in sampler.utilization_samples) + "\n"
        (out_dir / "util.jsonl").write_text(util_text)
    if sampler.frequency_samples:
        freq_text = "\n".join(json.dumps([t, f]) for t, f in sampler.frequency_samples) + "\n"
        (out_dir / "freq.jsonl").write_text(freq_text)
    power_stats = series_stats([sum(w) for _, w in sampler.samples])
    util_stats = gpu_series_stats(sampler.utilization_samples, gpus.gpus)
    freq_stats = gpu_series_stats(sampler.frequency_samples, gpus.gpus)
    window_s = max(t_load_end - t_start, 1e-9)
    good_req_s = att["joint_slo_requests"] / window_s
    good_tok_s = att["joint_output_tokens"] / window_s
    metering = {"source": getattr(sampler, "power_source", "unknown"),
                "metadata": getattr(sampler, "power_metadata", {}),
                "error": getattr(sampler, "error", None), "interval_s": 0.1,
                "power_samples": len(sampler.samples), "utilization_samples": len(sampler.utilization_samples),
                "frequency_samples": len(sampler.frequency_samples)}
    (out_dir / "metering.json").write_text(json.dumps(metering, indent=2, default=str))
    transitions = measure_transitions(ctl.transition_events, sampler.samples, gpus.gpus,
                                     sampler_error=getattr(sampler, 'error', None),
                                     power_source=getattr(sampler, 'power_source', 'unknown'))
    (out_dir / 'transition-measurements.json').write_text(json.dumps(transitions, indent=2) + '\n')
    metering_summary = {k: v for k, v in metering.items() if k != "metadata"}
    return dict(slo=att, energy_j=total_j, mean_power_w=total_w, peak_power_w=power_stats["max"],
                power=power_stats, duration_s=t_done - t_start,
                window_energy_j=win_j, window_mean_power_w=win_w, window_s=t_load_end - t_start,
                tail_s=t_done - t_load_end, request_tail_s=t_requests_done - t_load_end, per_gpu_mean_w=per_gpu,
                j_per_request=total_j / max(att["succeeded"], 1), j_per_token=total_j / max(att["output_tokens"], 1),
                j_per_goodput_request=total_j / max(att["joint_slo_requests"], 1),
                j_per_goodput_token=total_j / max(att["joint_output_tokens"], 1),
                goodput_request_s=good_req_s, goodput_token_s=good_tok_s,
                success_request_s=att["succeeded"] / window_s,
                utilization=util_stats, frequency=freq_stats, metering=metering_summary,
                controller=ctl.summary(), final_roles=dict(ctl.roles), power_samples=len(sampler.samples),
                quarantined_instances=sorted(getattr(router, 'quarantined', ())),
                transition_measurements='transition-measurements.json',
                native_cleanup_complete=bool(native and terminal_native))


def run_point(model_name: str, gpus: Sequence[int], tp: int, policy_name: str, profile_path: Path,
              trace: list[Request], slo: SLO, out_dir: Path, warmup: Optional[list[Request]] = None,
              kv_connector: Optional[str] = "P2pNcclConnector", proxy_port: Optional[int] = None, period_s: float = 10.0,
              tail_timeout_s: float = 300.0, trace_meta: Optional[dict] = None,
              fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0,
              base_port: int = 8100, sampling_seed: Optional[int] = None, *,
              tp_mode: Optional[str] = None,
              topology_profiles: Optional[Mapping[tuple[int, int], Path]] = None,
              resident_pools: Sequence[ResidentPool] = (),
              pool_fixed_plans: Optional[dict[str, Plan]] = None,
              joint_resident: bool = False,
              incremental_energy_path: Optional[Path] = None,
                 transition_catalog_path: Optional[Path] = None,
                 capacity_floor_path: Optional[Path] = None,
                 transition_qualified_only: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = get_policy(policy_name)
    if (joint_resident or incremental_energy_path or transition_catalog_path or capacity_floor_path) and not policy.name.startswith('pdblend'):
        raise ValueError('progressive optimization controls require the PDBlend policy')
    if transition_qualified_only and transition_catalog_path is None:
        raise ValueError('qualified-only transitions require an explicit catalog')
    runtime = None
    if tp_mode is not None:
        try:
            if tp_mode == 'slow_reshard_tp':
                raise UnsupportedTPMode('slow_reshard_tp requires a GPU-qualified native transaction backend')
            runtime = prepare_tp_runtime(
                model_name=model_name, gpus=gpus, fixed_tp=tp, mode=tp_mode,
                profiles=topology_profiles or {(tp, 1): profile_path}, policy=policy,
                forecast=offline_forecast(trace), slo=slo, requests=[*trace, *(warmup or [])],
                base_port=base_port, kv_connector=kv_connector, resident_pools=resident_pools)
        except UnsupportedTPMode as exc:
            result = dict(status=exc.status, reason=str(exc), tp_mode=tp_mode,
                          model=model_name, formal_eligible=False, hardware_executed=False)
            (out_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
            return result
        model, specs, profile_path = runtime.model, runtime.specs, runtime.profile_path
        tp = specs[0].tp if len({s.tp for s in specs}) == 1 else None
        (out_dir / 'tp-runtime.json').write_text(json.dumps(runtime.metadata, indent=2) + '\n')
    else:
        if topology_profiles or resident_pools or pool_fixed_plans:
            raise ValueError('topology profiles/pools require an explicit TP mode')
        loaded = load_profile(profile_path, system='pdblend',
                              model_id=ModelRegistry().get(model_name).model_id, tp=tp, pp=1)
        model = loaded.model
        (out_dir / 'profile-selection.json').write_text(json.dumps(loaded.manifest_fields(), indent=2) + '\n')
        specs = make_specs(model_name, gpus, tp=tp, base_port=base_port,
                           kv_connector=kv_connector if policy.allow_pd else None)
    if (joint_resident or incremental_energy_path is not None) and not (runtime and runtime.pool_models):
        raise ValueError('joint/energy routing requires explicit resident pools')
    if joint_resident and (pool_fixed_plans or policy.dynamic_m_floor):
        raise ValueError('joint resident optimization requires ordinary periodic inner plans')
    if policy.name.startswith('pdblend'):
        specs = [replace(spec, native_control=True) for spec in specs]
        if runtime is not None:
            runtime.specs = specs
            runtime.metadata.update(native_control=True, engine_entrypoint='pdblend_runtime.serve')
            (out_dir / 'tp-runtime.json').write_text(json.dumps(runtime.metadata, indent=2) + '\n')
    # Reject missing runtime components and unbound optimization artifacts before
    # acquiring/resetting GPUs or starting any engine process.
    from pdblend.profile.query.runtime import require_planner_components
    from pdblend.planner.capacity import load_capacity_floors, select_artifact
    from pdblend.planner.transitions import TransitionCatalog
    preflight_models = runtime.pool_models.values() if runtime and runtime.pool_models else (model,)
    for checked in preflight_models:
        require_planner_components(checked, allow_pd=policy.allow_pd, allow_dvfs=policy.allow_dvfs)
        if capacity_floor_path is not None:
            load_capacity_floors(capacity_floor_path, model=checked)
        if transition_catalog_path is not None:
            TransitionCatalog.load(select_artifact(transition_catalog_path, checked), model=checked,
                                   qualified_only=transition_qualified_only)
    meter = Gpus(list(gpus))
    meter.reset_all()
    proxy_port = proxy_port or 8000 + min(gpus)  # concurrent fleets on disjoint GPUs share the host network
    started = time.time()
    try:
        with Fleet(specs, out_dir / "logs") as fleet:
            if runtime and runtime.pool_models:
                # One resident lifecycle, stagger cold loads to avoid disk/CPU
                # contention; all pools serve concurrently after health checks.
                startup = {}
                for iid, instance in fleet.instances.items():
                    instance.start()
                    startup[iid] = instance.wait_ready()
            else:
                startup = fleet.start_all()
            result = asyncio.run(_point(fleet, meter, model, policy, slo, trace, warmup or [], out_dir, proxy_port,
                                        period_s, tail_timeout_s, fixed_plan, min_warm_s, sampling_seed,
                                        initial_plan=runtime.selected_plan if runtime else None,
                                        pool_models=runtime.pool_models if runtime else None,
                                        pool_fixed_plans=pool_fixed_plans, joint_resident=joint_resident,
                                        resident_pools=resident_pools, incremental_energy_path=incremental_energy_path,
                                        transition_catalog_path=transition_catalog_path, capacity_floor_path=capacity_floor_path,
                                        transition_qualified_only=transition_qualified_only))
            result.update(fleet_events=fleet.events(), startup_s=startup)
    finally:
        meter.reset_all()
    result.update(model=model_name, gpus=list(gpus), tp=tp, policy=asdict(policy), profile=str(profile_path),
                  profile_key=model.profile_key,
                  calibration_identity=getattr(model, 'calibration_identity', {}),
                  calibration_coverage=getattr(model, 'calibration_coverage', {}),
                  calibration_qualification=getattr(model, 'calibration_qualification', {}),
                  fixed_plan=None if fixed_plan is None else dict(counts=fixed_plan.counts, f_P=fixed_plan.f_P,
                                                                  f_D=fixed_plan.f_D, f_M=fixed_plan.f_M, tau=fixed_plan.tau,
                                                                  detail=fixed_plan.detail),
                  slo_ttft_s=slo.ttft_s, slo_tpot_s=slo.tpot_s, trace=trace_summary(trace), trace_meta=trace_meta or {},
                  wall_s=time.time() - started, requests=len(trace))
    if runtime:
        result.update(tp_mode=runtime.mode, topology=runtime.metadata, formal_eligible=False)
    if fixed_plan and fixed_plan.detail.get('mechanism_forced_roles'):
        result.update(scope='mechanism_probe', policy_decision=False, formal_eligible=False)
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    tmp.replace(out_dir / "summary.json")
    return result


def make_warmup(records: list[dict], n: int = 8, seed: int = 2701) -> list[Request]:
    reqs = poisson_trace(records, rate_rps=2.0, duration_s=n / 2.0, seed=seed, source="warmup")[:n]
    for r in reqs:
        r.max_tokens = min(r.max_tokens, 32)
    return reqs
