"""One benchmark point: fleet + proxy + controller + open-loop load with 8-GPU power metering."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from aiohttp import web

from ..control.controller import Controller
from ..control.forecast import Forecast, Forecaster
from ..control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from ..control.policies import Policy, get_policy
from ..control.policies.baselines import EcoRouter, build_control
from ..control.shield import Shield
from ..engine.client import PDTransfer
from ..engine.launcher import Fleet, make_specs
from ..profile.model import PerfModel
from ..proxy.router import Router
from ..proxy.server import Proxy
from .client import LoadClient, Request, dump_outcomes, nearest_rank, poisson_trace, slo_attainment, trace_summary
from .metering import Gpus


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


async def _point(fleet: Fleet, gpus: Gpus, model: PerfModel, policy: Policy, slo: SLO, trace: list[Request],
                 warmup: list[Request], out_dir: Path, proxy_port: int, period_s: float, tail_timeout_s: float,
                 fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0) -> dict:
    urls = {iid: inst.spec.base_url for iid, inst in fleet.instances.items()}
    specs = [inst.spec for inst in fleet.instances.values()]
    transfer = PDTransfer(specs[0].kv_connector, {s.instance_id: s.zmq_address for s in specs})
    router = EcoRouter(list(urls), model, slo) if policy.name == "ecoserve" else Router(list(urls))
    proxy = Proxy(urls, router, transfer=transfer)
    runner = await _serve_proxy(proxy, proxy_port)
    cfg = policy.planner_config(PlannerConfig(slots=len(urls), slo=slo, freqs=model.freqs))
    planner = PoolPlanner(model, cfg)
    freeze = policy.freeze or fixed_plan is not None
    initial = fixed_plan or (planner.plan(offline_forecast(trace)) if (policy.freeze or policy.warm_start) else None)
    if policy.ported and fixed_plan is None:
        planner, initial, freeze, ported_period = build_control(policy.name, planner, router, offline_forecast(trace),
                                                                [r.arrival_s for r in trace])
        period_s = ported_period or period_s
    ctl = Controller(fleet, router, gpus, planner, Shield(slo) if policy.shield else None, Forecaster(),
                     period_s=period_s, log_path=out_dir / "controller.jsonl", initial_plan=initial, freeze=freeze,
                     hold_initial=policy.warm_start, min_warm_s=min_warm_s)
    stop = asyncio.Event()
    ctl_task = asyncio.create_task(ctl.run(stop))
    await asyncio.sleep(2.0)
    load = LoadClient(f"http://127.0.0.1:{proxy_port}")
    if warmup:
        await load.replay(warmup, progress_every_s=1e9)
        while any(l.inflight_seqs for l in router.loads.values()):
            await asyncio.sleep(0.2)
    await asyncio.sleep(1.0)
    sampler = gpus.sampler(interval_s=0.1)
    sampler.start()
    t_start = time.time()
    outcomes = await load.replay_detached(trace)
    t_load_end = t_start + trace[-1].arrival_s if trace else time.time()
    t_done = time.time()
    # Requests are all finished (replay awaits them); keep sampling briefly so the tail is captured.
    await asyncio.sleep(1.0)
    sampler.stop()
    stop.set()
    await ctl_task
    await runner.cleanup()
    dump_outcomes(outcomes, out_dir / "outcomes.jsonl")
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
        (out_dir / "utilization.jsonl").write_text(util_text)
        (out_dir / "util.jsonl").write_text(util_text)
    if sampler.frequency_samples:
        freq_text = "\n".join(json.dumps([t, f]) for t, f in sampler.frequency_samples) + "\n"
        (out_dir / "frequency.jsonl").write_text(freq_text)
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
    return dict(slo=att, energy_j=total_j, mean_power_w=total_w, peak_power_w=power_stats["max"],
                power=power_stats, duration_s=t_done - t_start,
                window_energy_j=win_j, window_mean_power_w=win_w, window_s=t_load_end - t_start,
                tail_s=t_done - t_load_end, per_gpu_mean_w=per_gpu,
                j_per_request=total_j / max(att["succeeded"], 1), j_per_token=total_j / max(att["output_tokens"], 1),
                j_per_goodput_request=total_j / max(att["joint_slo_requests"], 1),
                j_per_goodput_token=total_j / max(att["joint_output_tokens"], 1),
                goodput_request_s=good_req_s, goodput_token_s=good_tok_s,
                success_request_s=att["succeeded"] / window_s,
                utilization=util_stats, frequency=freq_stats, metering=metering,
                controller=ctl.summary(), final_roles=dict(ctl.roles), power_samples=len(sampler.samples))


def run_point(model_name: str, gpus: Sequence[int], tp: int, policy_name: str, profile_path: Path,
              trace: list[Request], slo: SLO, out_dir: Path, warmup: Optional[list[Request]] = None,
              kv_connector: Optional[str] = "P2pNcclConnector", proxy_port: Optional[int] = None, period_s: float = 10.0,
              tail_timeout_s: float = 300.0, trace_meta: Optional[dict] = None,
              fixed_plan: Optional[Plan] = None, min_warm_s: float = 20.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = get_policy(policy_name)
    model = PerfModel.load(profile_path)
    specs = make_specs(model_name, gpus, tp=tp, kv_connector=kv_connector if policy.allow_pd else None)
    meter = Gpus(list(gpus))
    meter.reset_all()
    proxy_port = proxy_port or 8000 + min(gpus)  # concurrent fleets on disjoint GPUs share the host network
    started = time.time()
    with Fleet(specs, out_dir / "logs") as fleet:
        startup = fleet.start_all()
        result = asyncio.run(_point(fleet, meter, model, policy, slo, trace, warmup or [], out_dir, proxy_port,
                                    period_s, tail_timeout_s, fixed_plan, min_warm_s))
        result.update(fleet_events=fleet.events(), startup_s=startup)
    meter.reset_all()
    result.update(model=model_name, gpus=list(gpus), tp=tp, policy=asdict(policy), profile=str(profile_path),
                  fixed_plan=None if fixed_plan is None else dict(counts=fixed_plan.counts, f_P=fixed_plan.f_P,
                                                                  f_D=fixed_plan.f_D, f_M=fixed_plan.f_M, tau=fixed_plan.tau),
                  slo_ttft_s=slo.ttft_s, slo_tpot_s=slo.tpot_s, trace=trace_summary(trace), trace_meta=trace_meta or {},
                  wall_s=time.time() - started, requests=len(trace))
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    tmp.replace(out_dir / "summary.json")
    return result


def make_warmup(records: list[dict], n: int = 8, seed: int = 2701) -> list[Request]:
    reqs = poisson_trace(records, rate_rps=2.0, duration_s=n / 2.0, seed=seed, source="warmup")[:n]
    for r in reqs:
        r.max_tokens = min(r.max_tokens, 32)
    return reqs
