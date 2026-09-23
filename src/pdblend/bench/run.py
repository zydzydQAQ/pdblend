"""One benchmark point: fleet + proxy + controller + open-loop load with 8-GPU power metering."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
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
from ..proxy.router import ResidentRouter, Router
from ..proxy.server import Proxy
from .client import LoadClient, Request, dump_outcomes, nearest_rank, poisson_trace, slo_attainment, trace_summary
from .metering import Gpus
from .tp_runtime import UnsupportedTPMode, _covers, prepare_tp_runtime


def offline_forecast(trace: list[Request]) -> Forecast:
    s = trace_summary(trace)
    return Forecast(s["mean_rps"], 0.0, s["input_mean"], s["input_p95"], s["output_mean"], 0,
                    tuple(r.input_tokens for r in trace), tuple(r.max_tokens for r in trace))


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
                     fixed_plan=None, min_warm_s=20.0, initial_plan=None):
    cfg = policy.planner_config(PlannerConfig(slots=len(fleet.instances), slo=slo, freqs=model.freqs))
    cfg.pressure_controls = policy.dynamic_m_floor
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
                     transition_cooldown_s=policy.transition_cooldown_s)
    return ctl


class _PoolControllers:
    def __init__(self, controllers):
        self.controllers = controllers

    @property
    def roles(self):
        return {iid: role for controller in self.controllers.values() for iid, role in controller.roles.items()}

    async def run(self, stop):
        tasks = [asyncio.create_task(controller.run(stop)) for controller in self.controllers.values()]
        try:
            await asyncio.gather(*tasks)
        finally:
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    def summary(self):
        return {'mode': 'resident_hetero_tp', 'pools': {pool: ctl.summary() for pool, ctl in self.controllers.items()}}


async def _point(fleet: Fleet, gpus: Gpus, model: PerfModel, policy: Policy, slo: SLO, trace: list[Request],
                 warmup: list[Request], out_dir: Path, proxy_port: int, period_s: float, tail_timeout_s: float,
                 fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0,
                 sampling_seed: Optional[int] = None, *, initial_plan: Optional[Plan] = None,
                 pool_models: Optional[dict[str, PerfModel]] = None,
                 pool_fixed_plans: Optional[dict[str, Plan]] = None,
                 planning_trace: Optional[list[Request]] = None,
                 observation_duration_s: Optional[float] = None) -> dict:
    selection_trace = trace if planning_trace is None else planning_trace
    if not selection_trace:
        raise ValueError('nonempty calibration/tuning bootstrap trace required')
    if observation_duration_s is not None and (observation_duration_s <= 0
            or any(r.arrival_s >= observation_duration_s for r in trace)):
        raise ValueError('observation duration must contain all evaluation arrivals')
    urls = {iid: inst.spec.base_url for iid, inst in fleet.instances.items()}
    specs = [inst.spec for inst in fleet.instances.values()]
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
                (pool_fixed_plans or {}).get(pool_id), min_warm_s)
            pool_routers[pool_id] = sub_router
        router = ResidentRouter(pool_routers, pool_models)
        router.frequency_provider = lambda iid: controllers[router._owners[iid]].freqs.get(iid)
        ctl = _PoolControllers(controllers)
    else:
        router = (EcoRouter(list(urls), model, slo) if policy.name == 'ecoserve'
                  else Router(list(urls), instance_metadata=metadata))
        ctl = _make_controller(fleet, router, gpus, model, policy, slo, selection_trace, out_dir, period_s,
                               fixed_plan, min_warm_s, initial_plan)
    proxy = Proxy(urls, router, transfer=transfer)
    runner = await _serve_proxy(proxy, proxy_port)
    stop = asyncio.Event()
    ctl_task = asyncio.create_task(ctl.run(stop))
    await asyncio.sleep(2.0)
    load = LoadClient(f"http://127.0.0.1:{proxy_port}", sampling_seed=sampling_seed)
    if warmup:
        await load.replay(warmup, progress_every_s=1e9)
        while any(l.inflight_seqs for l in router.loads.values()):
            await asyncio.sleep(0.2)
    await asyncio.sleep(1.0)
    sampler = gpus.sampler(interval_s=0.1)
    sampler.start()
    t_start = time.time()
    outcomes = await load.replay_detached(trace)
    t_load_end = (t_start + observation_duration_s if observation_duration_s is not None
                  else t_start + trace[-1].arrival_s if trace else time.time())
    if observation_duration_s is not None:
        await asyncio.sleep(max(0., t_load_end-time.time()))
    t_done = time.time()
    # Requests are all finished (replay awaits them); keep sampling briefly so the tail is captured.
    await asyncio.sleep(1.0)
    sampler.stop()
    stop.set()
    await ctl_task
    await runner.cleanup()
    dump_outcomes(outcomes, out_dir / "outcomes.jsonl")
    (out_dir / 'routes.jsonl').write_text(''.join(json.dumps(dict(
        request_id=r.request_id, input_tokens=r.input_tokens, path=r.path,
        submitted_s=r.submitted_s, m_pressure=r.route_pressure, reason=r.route_reason,
        prefill_instance=r.prefill_instance, decode_instance=r.decode_instance,
        tp=r.tp, pp=r.pp, pool_id=r.pool_id, generation=r.generation, profile_key=r.profile_key)) + '\n'
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
    metering_summary = {k: v for k, v in metering.items() if k != "metadata"}
    return dict(slo=att, energy_j=total_j, mean_power_w=total_w, peak_power_w=power_stats["max"],
                power=power_stats, duration_s=t_done - t_start,
                window_energy_j=win_j, window_mean_power_w=win_w, window_s=t_load_end - t_start,
                tail_s=t_done - t_load_end, per_gpu_mean_w=per_gpu,
                j_per_request=total_j / max(att["succeeded"], 1), j_per_token=total_j / max(att["output_tokens"], 1),
                j_per_goodput_request=total_j / max(att["joint_slo_requests"], 1),
                j_per_goodput_token=total_j / max(att["joint_output_tokens"], 1),
                goodput_request_s=good_req_s, goodput_token_s=good_tok_s,
                success_request_s=att["succeeded"] / window_s,
                utilization=util_stats, frequency=freq_stats, metering=metering_summary,
                controller=ctl.summary(), final_roles=dict(ctl.roles), power_samples=len(sampler.samples),
                quarantined_instances=sorted(getattr(router, 'quarantined', ())))


def run_point(model_name: str, gpus: Sequence[int], tp: int, policy_name: str, profile_path: Path,
              trace: list[Request], slo: SLO, out_dir: Path, warmup: Optional[list[Request]] = None,
              kv_connector: Optional[str] = "P2pNcclConnector", proxy_port: Optional[int] = None, period_s: float = 10.0,
              tail_timeout_s: float = 300.0, trace_meta: Optional[dict] = None,
              fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0,
              base_port: int = 8100, sampling_seed: Optional[int] = None, *,
              tp_mode: Optional[str] = None,
              topology_profiles: Optional[Mapping[tuple[int, int], Path]] = None,
              resident_pools: Sequence[ResidentPool] = (),
              pool_fixed_plans: Optional[dict[str, Plan]] = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = get_policy(policy_name)
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
        model = PerfModel.load(profile_path)
        specs = make_specs(model_name, gpus, tp=tp, base_port=base_port,
                           kv_connector=kv_connector if policy.allow_pd else None)
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
                                        pool_fixed_plans=pool_fixed_plans))
            result.update(fleet_events=fleet.events(), startup_s=startup)
    finally:
        meter.reset_all()
    result.update(model=model_name, gpus=list(gpus), tp=tp, policy=asdict(policy), profile=str(profile_path),
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
