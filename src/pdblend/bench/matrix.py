"""Benchmark points from plain dicts: one point (shared by the CLI) or a JSON matrix run unattended."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from ..control.forecast import Forecast
from ..control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from ..control.policies import get_policy
from ..control.policies.baselines import capacity_rps
from ..profile.model import PerfModel
from . import client as bc
from .run import make_warmup, run_point

DEFAULTS = dict(model="Qwen2.5-7B-Instruct", gpus="0,1,2,3,4,5,6,7", tp=1, policy="pdblend", profile=None,
                corpus="datasets/prepared/2026-09-13-7b-v1", dataset="sharegpt", split="evaluation",
                rate=5.0, duration=100.0, stages="", cv=1.5, azure="", azure_offset=0.0, azure_peak=5.0,
                seed=701, scale=None, connector="P2pNcclConnector", period=10.0, layout="", clocks="P=2520,D=2520,M=2520",
                tau=0, out=None)


def parse_kv(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (kv.split("=") for kv in text.split(","))} if text else {}


def build_trace(a: dict, records: list[dict]):
    meta = {}
    if a["azure"]:
        rows = bc.read_azure_window(Path("datasets/raw/azure-llm-2024") / f"AzureLLMInferenceTrace_{a['azure']}_1week.csv",
                                    float(a["azure_offset"]), float(a["duration"]))
        trace, meta = bc.azure_trace(rows, records, float(a["azure_peak"]), int(a["seed"]))
    elif a["stages"]:
        stages = [(float(d), float(s)) for d, s in (p.split(":") for p in a["stages"].split(","))]
        trace = bc.staged_trace(records, float(a["rate"]), stages, float(a["cv"]), int(a["seed"]), a["dataset"])
    else:
        trace = bc.poisson_trace(records, float(a["rate"]), float(a["duration"]), int(a["seed"]), a["dataset"])
    source = a["azure"] and f"azure-{a['azure']}" or (a["stages"] and "staged") or "poisson"
    return trace, dict(meta, dataset=a["dataset"], seed=a["seed"], scale=a["scale"], source=source)


def corpus_forecast(records: list[dict], rate_rps: float) -> Forecast:
    inputs = sorted(r["input_tokens"] for r in records)
    outputs = [r["output_tokens"] for r in records]
    return Forecast(rate_rps, 0.0, sum(inputs) / len(inputs), inputs[int(0.95 * (len(inputs) - 1))],
                    sum(outputs) / len(outputs), 0, tuple(inputs), tuple(outputs))


def layout_capacity(profile: Path, corpus: Path, dataset: str, layout: str, clocks: str, tau: int,
                    split: str = "evaluation") -> float:
    """Max Poisson rate at which the planner's SLO model still admits the given fixed layout."""
    planner, counts, fc = _planner(profile, corpus, dataset, layout, split)
    c = parse_kv(clocks)
    return capacity_rps(planner, fc, counts, c.get("P", 2520), c.get("D", 2520), c.get("M", 2520), tau,
                        upper=256.0, eps=0.05)


def _planner(profile: Path, corpus: Path, dataset: str, layout: str, split: str = "evaluation"):
    model = PerfModel.load(profile)
    counts = parse_kv(layout)
    planner = PoolPlanner(model, PlannerConfig(slots=sum(counts.values()), slo=SLO(*bc.SLOS[dataset]), freqs=model.freqs))
    return planner, counts, corpus_forecast(bc.load_split(corpus, dataset, split), 1.0)


def auto_clocks(profile: Path, corpus: Path, dataset: str, layout: str, rate_rps: float, tau: int,
                split: str = "evaluation") -> dict[str, int]:
    """Lowest-power clocks the planner model admits for a fixed layout at this rate (max clocks if none)."""
    planner, counts, fc = _planner(profile, corpus, dataset, layout, split)
    fc = replace(fc, rate_rps=rate_rps)
    freqs = list(planner.model.freqs)
    best, best_power = None, float("inf")
    for f_P in [f for f in freqs if f >= 2100] or freqs[-1:]:
        for f_D in freqs:
            for f_M in freqs:
                plan = planner.evaluate(counts, f_P, f_D, f_M, tau, fc)
                if plan is not None and plan.power_w < best_power:
                    best, best_power = dict(P=f_P, D=f_D, M=f_M), plan.power_w
    return best or dict(P=freqs[-1], D=freqs[-1], M=freqs[-1])


def m3_spec(profile: Path, gpus: str, root: Path, corpus: Path = Path("datasets/prepared/2026-09-13-7b-v1"),
            datasets=("alpaca", "sharegpt", "longbench"), scales=(0.25, 0.5, 0.75, 1.0),
            layouts=("M=4", "P=1,D=3", "P=2,D=2"), duration: float = 120.0, tau: int = 0) -> dict:
    """Co-located vs disaggregated crossover on 4 GPUs: each layout at max clocks and at planner-chosen clocks."""
    n = sum(parse_kv(layouts[0]).values())
    points, caps = [], {}
    for ds in datasets:
        caps[ds] = layout_capacity(profile, corpus, ds, f"M={n}", "M=2520", tau)
        for sc in scales:
            rate = round(caps[ds] * sc, 3)
            for lay in layouts:
                label = "+".join(f"{v}{k}" for k, v in parse_kv(lay).items())
                auto = auto_clocks(profile, corpus, ds, lay, rate, tau)
                for tag, clocks in (("max", "P=2520,D=2520,M=2520"), ("auto", ",".join(f"{k}={v}" for k, v in auto.items()))):
                    points.append(dict(name=f"{ds}-x{sc:g}-{label}-{tag}", dataset=ds, rate=rate, scale=sc, layout=lay,
                                       clocks=clocks, tau=tau))
    spec = dict(root=str(root), capacity_rps=caps,
                defaults=dict(policy="manual", profile=str(profile), gpus=gpus, duration=duration, corpus=str(corpus)),
                points=points)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


CORE_POLICIES = ("mixed", "mixed_dvfs", "mixed_dvfs_park", "static_best", "pdblend")
PORTED_POLICIES = ("distserve_static", "dynamollm", "ecoserve")
ABLATIONS = ("pdblend_no_park", "pdblend_no_pd", "pdblend_no_shield", "pdblend_fixed_pools")


def eval_spec(profile: Path, gpus: str, root: Path, corpus: Path = Path("datasets/prepared/2026-09-13-7b-v1"),
              model: str = "Qwen2.5-7B-Instruct", tp: int = 1,
              datasets=("alpaca", "sharegpt", "longbench"), scales=(0.25, 0.5, 0.75),
              core=CORE_POLICIES, ported=PORTED_POLICIES, ablations=ABLATIONS,
              reduced_datasets=("sharegpt", "longbench"), reduced_scale: float = 0.5,
              duration: float = 300.0, stages: str = "300:0.25,300:0.75,300:0.5,300:1.0",
              azure=("conv", "code"), azure_duration: float = 1800.0, azure_peak_scale: float = 0.75,
              azure_policies=("mixed", "mixed_dvfs_park", "pdblend"), seed: int = 701,
              ported_scales=None, reduced_core=(), ported_datasets=None) -> dict:
    """P4 evaluation matrix on one fleet: controlled Poisson, ported baselines, ablations, staged and Azure traces.

    Rates are fractions of the planner's max-clock all-mixed capacity so every dataset sees the same relative load."""
    n = len(str(gpus).split(","))
    points, caps, groups = [], {}, {}

    def add(group: str, **p):
        points.append(p)
        groups[group] = groups.get(group, 0) + 1

    for ds in datasets:
        caps[ds] = layout_capacity(profile, corpus, ds, f"M={n}", "M=2520", 0)
    for ds in datasets:
        for sc in scales:
            rate = round(caps[ds] * sc, 3)
            for pol in core:
                add("controlled", name=f"{ds}-x{sc:g}-{pol}", dataset=ds, rate=rate, scale=sc, policy=pol)
    for ds in (ported_datasets or reduced_datasets):
        for sc in (tuple(ported_scales) if ported_scales is not None else (reduced_scale,)):
            rate = round(caps[ds] * sc, 3)
            for pol in ported:
                add("ported", name=f"{ds}-x{sc:g}-{pol}", dataset=ds, rate=rate, scale=sc, policy=pol)
    for ds in reduced_datasets:
        rate = round(caps[ds] * reduced_scale, 3)
        for pol in reduced_core:
            add("reduced_core", name=f"{ds}-x{reduced_scale:g}-{pol}", dataset=ds, rate=rate, scale=reduced_scale,
                policy=pol)
        for pol in ablations:
            add("ablation", name=f"{ds}-x{reduced_scale:g}-{pol}", dataset=ds, rate=rate, scale=reduced_scale, policy=pol)
    if stages:
        total = sum(float(d) for d, _ in (p.split(":") for p in stages.split(",")))
        for ds in reduced_datasets:
            for pol in core:
                add("staged", name=f"{ds}-staged-{pol}", dataset=ds, rate=caps[ds], stages=stages, duration=total,
                    policy=pol)
    for az in azure:
        for ds in reduced_datasets:
            peak = round(caps[ds] * azure_peak_scale, 3)
            for pol in azure_policies:
                add("azure", name=f"{ds}-azure-{az}-{pol}", dataset=ds, azure=az, azure_peak=peak, scale=azure_peak_scale,
                    duration=azure_duration, policy=pol)
    spec = dict(root=str(root), capacity_rps=caps, groups=groups,
                defaults=dict(model=model, tp=tp, profile=str(profile), gpus=gpus, duration=duration, corpus=str(corpus),
                              seed=seed),
                points=points)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def bench_point(args: dict) -> dict:
    a = dict(DEFAULTS, **{k: v for k, v in args.items() if v is not None})
    records = bc.load_split(Path(a["corpus"]), a["dataset"], a["split"])
    trace, meta = build_trace(a, records)
    fixed = None
    if a["layout"]:
        c = parse_kv(a["clocks"])
        fixed = Plan(parse_kv(a["layout"]), c.get("P", 2520), c.get("D", 2520), c.get("M", 2520), int(a["tau"]),
                     0.0, 0.0, 0.0, dict(manual=True))
    gpus = [int(g) for g in str(a["gpus"]).split(",")]
    policy = get_policy(a["policy"])
    warmup = make_warmup(records)
    if policy.history_s > 0:
        # Unmeasured pre-window history (replayed before metering starts): the observed-history
        # load template for history-driven policies, standing in for the paper's past-week data.
        history = bc.poisson_trace(records, float(a["rate"]), policy.history_s,
                                   seed=int(a["seed"]) + 90001, source="history")
        warmup = sorted(history + warmup, key=lambda r: r.arrival_s)
    return run_point(a["model"], gpus, int(a["tp"]), a["policy"], Path(a["profile"]), trace, SLO(*bc.SLOS[a["dataset"]]),
                     Path(a["out"]), warmup, a["connector"], period_s=float(a["period"]),
                     fixed_plan=fixed, trace_meta=meta)


def brief(result: dict) -> dict:
    s = result["slo"]
    return dict(policy=result["policy"]["name"], joint_slo_rate=s["joint_slo_rate"], success_rate=s["success_rate"],
                ttft_p90=s["ttft_p90"], tpot_p90=s["tpot_p90"], energy_j=result["energy_j"],
                window_energy_j=result["window_energy_j"], mean_power_w=result["mean_power_w"],
                j_per_request=result["j_per_request"], final_roles=result["final_roles"])


def run_matrix(spec_path: Path, only: str = "", dry: bool = False, shard: str = "", gpus: str = "") -> list[dict]:
    """spec: {"root": dir, "defaults": {...}, "points": [{"name": ..., <overrides>}, ...]}; skips finished points.

    shard="i/n" takes points[i::n] so several runners can share a spec on disjoint GPU groups (gpus override)."""
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec.get("root", Path(spec_path).with_suffix("")))
    points = spec["points"]
    status_name = "matrix-status.json"
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        points = points[i::n]
        status_name = f"matrix-status-{i}of{n}.json"
    done = []
    for point in points:
        name = point["name"]
        if only and only not in name:
            continue
        out = root / name
        if (out / "summary.json").exists():
            done.append(dict(name=name, status="skipped"))
            continue
        args = dict(spec.get("defaults", {}), **{k: v for k, v in point.items() if k != "name"}, out=str(out))
        if gpus:
            args["gpus"] = gpus
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps({k: args[k] for k in ('policy', 'dataset', 'rate', 'layout', 'clocks') if k in args})}",
              flush=True)
        if dry:
            done.append(dict(name=name, status="dry", args=args))
            continue
        try:
            result = bench_point(args)
            done.append(dict(name=name, status="ok", **brief(result)))
            print(json.dumps(done[-1]), flush=True)
        except Exception as exc:  # keep the matrix going; the point is re-run on the next invocation
            (out / "error.txt").parent.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(repr(exc))
            done.append(dict(name=name, status="error", error=repr(exc)))
            print(json.dumps(done[-1]), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / status_name).write_text(json.dumps(done, indent=1, default=str))
    return done
