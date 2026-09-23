"""Benchmark points from plain dicts: one point (shared by the CLI) or a JSON matrix run unattended."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from ..control.forecast import Forecast
from ..control.planner import SLO, Plan, PlannerConfig, PoolPlanner
from ..control.policies import get_policy
from ..control.policies.baselines import capacity_rps
from ..profile.model import PerfModel
from ..seed_config import SINGLE_SEED, SEED_POLICY, seed_metadata
from . import client as bc
from .run import make_warmup, run_point


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Stable digest for the exact corpus tree used by a point."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(bytes.fromhex(_sha256(path)))
    return h.hexdigest()


def _trace_sha256(trace: list[bc.Request]) -> str:
    h = hashlib.sha256()
    for r in trace:
        h.update(json.dumps([r.idx, r.arrival_s, r.input_tokens, r.max_tokens],
                            separators=(",", ":"), allow_nan=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def _env_identity(name: str, default: str = "unknown") -> str:
    value = os.environ.get(name)
    return value if value else default


def _write_evidence(out: Path, args: dict, result: dict, trace: list[bc.Request], corpus: Path) -> None:
    """Write a self-contained attestation for one active-workspace matrix point.

    The matrix runner is executed inside a pinned container with read-only source
    and profile inputs.  The launcher supplies the corresponding hashes and
    hardware identity through environment variables; missing values remain
    explicitly ``unknown`` and therefore cannot pass strict pairing.
    """
    profile = Path(args["profile"])
    summary_path = out / "summary.json"
    identity = {
        "dataset": args["dataset"], "rate": float(args["rate"]), "seed": int(args["seed"]),
        **seed_metadata((int(args["seed"]),)),
        "duration": float(args["duration"]), "trace_window_s": float(result["window_s"]),
        "requests": int(result["requests"]), "model": args["model"], "tp": int(args["tp"]),
        "gpus": str(args["gpus"]), "policy": result["policy"]["name"],
        "trace_sha256": _trace_sha256(trace), "corpus_sha256": _tree_sha256(corpus),
        "profile_sha256": _sha256(profile),
        "profile_system": args.get("profile_system", args.get("policy", "unknown")),
        "profile_key": args.get("profile_key", {}),
        "source_sha256": _env_identity("PDBLEND_SOURCE_SHA256"),
        "image": _env_identity("PDBLEND_IMAGE_ID"),
        "hardware": _env_identity("PDBLEND_HARDWARE_UUIDS"),
        "clock_protocol": _env_identity("PDBLEND_CLOCK_PROTOCOL", "nvidia-smi-lock-clock-v1"),
        "energy_protocol": _env_identity("PDBLEND_ENERGY_PROTOCOL", "nvml-0.1s-trapezoid-v1"),
    }
    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "evidence.json":
            artifacts[path.name] = _sha256(path)
    evidence = {
        "status": "complete", "returncode": 0, "inputs_unchanged": True,
        **seed_metadata((int(args["seed"]),)),
        "identity": identity,
        "identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest(),
        "artifacts": artifacts,
        "trace_meta": result.get("trace_meta", {}),
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))

DEFAULTS = dict(model="Qwen2.5-7B-Instruct", gpus="0,1,2,3,4,5,6,7", tp=1, policy="pdblend", profile=None,
                corpus="datasets/prepared/2026-09-13-7b-v1", dataset="sharegpt", split="evaluation",
                rate=5.0, duration=100.0, stages="", cv=1.5, azure="", azure_offset=0.0, azure_peak=5.0,
                seed=SINGLE_SEED, scale=None, connector="P2pNcclConnector", period=10.0, layout="", clocks="P=2520,D=2520,M=2520",
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
    return trace, dict(meta, dataset=a["dataset"], seed=a["seed"], scale=a["scale"], source=source,
                       **seed_metadata((int(a["seed"]),)))


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
    spec = dict(root=str(root), capacity_rps=caps, **seed_metadata(),
                defaults=dict(policy="manual", profile=str(profile), gpus=gpus, duration=duration, corpus=str(corpus),
                              seed=SINGLE_SEED),
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
              azure_policies=("mixed", "mixed_dvfs_park", "pdblend"), seed: int = SINGLE_SEED,
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
    spec = dict(root=str(root), capacity_rps=caps, groups=groups, **seed_metadata((seed,)),
                defaults=dict(model=model, tp=tp, profile=str(profile), gpus=gpus, duration=duration, corpus=str(corpus),
                              seed=seed),
                points=points)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def bench_point(args: dict) -> dict:
    if args.get('runner') == 'independent_native_dispatch':
        raise ValueError('active five-system points require independent_dispatch; legacy matrix policies are forbidden')
    a = dict(DEFAULTS, **{k: v for k, v in args.items() if v is not None})
    profile_by_system = a.get("profile_by_system", {}) or {}
    selected_profile = profile_by_system.get(a["policy"]) or profile_by_system.get(a.get("system", ""))
    if selected_profile:
        if isinstance(selected_profile, dict):
            a["profile_key"] = selected_profile.get("profile_key", {})
            selected_profile = selected_profile.get("path")
        a["profile"] = selected_profile
    a["profile_system"] = a.get("policy", "pdblend")
    if not a.get("profile"):
        raise ValueError(f"independent profile required for {a['policy']}")
    records = bc.load_split(Path(a["corpus"]), a["dataset"], a["split"])
    trace, meta = build_trace(a, records)
    meta = dict(meta, trace_sha256=_trace_sha256(trace), corpus_sha256=_tree_sha256(Path(a["corpus"])))
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
    optimization = {}
    optional = ("tp_mode", "joint_resident", "incremental_energy_path", "transition_catalog_path",
                "capacity_floor_path", "transition_qualified_only")
    if any(a.get(key) for key in (*optional, "topology_profiles", "resident_pools")):
        if not a["policy"].startswith("pdblend"):
            raise ValueError("PDBlend optimization/TP options cannot alter an independent baseline")
        optimization = {key: a[key] for key in optional if a.get(key) is not None}
        def json_option(value):
            if isinstance(value, (str, Path)):
                path = Path(value)
                return json.loads(path.read_text()), path.parent
            return value, Path.cwd()
        if a.get("topology_profiles"):
            profiles, root = json_option(a["topology_profiles"])
            optimization["topology_profiles"] = {tuple(int(n) for n in key.replace("tp", "").replace("pp", "").split("-")):
                                                   root / value for key, value in profiles.items()}
        if a.get("resident_pools"):
            from pdblend.planner.topology import ResidentPool, Topology
            pools, _ = json_option(a["resident_pools"])
            optimization["resident_pools"] = tuple(ResidentPool(**dict(pool, topology=Topology(**pool["topology"])))
                                                    for pool in pools)
    result = run_point(a["model"], gpus, int(a["tp"]), a["policy"], Path(a["profile"]), trace,
                       SLO(*bc.SLOS[a["dataset"]]), Path(a["out"]), warmup, a["connector"],
                       period_s=float(a["period"]), fixed_plan=fixed, trace_meta=meta,
                       sampling_seed=int(a["seed"]), **optimization)
    # Kept in memory for the attestation writer; run_point has already written
    # the public summary without serialising request objects.
    result["_evidence_trace"] = trace
    return result


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
        args = dict(spec.get("defaults", {}), **{k: v for k, v in point.items() if k != "name"})
        active_seed_policy = spec.get("seed_policy") == SEED_POLICY or spec.get("formal")
        if active_seed_policy:
            if int(args.get("seed", SINGLE_SEED)) != SINGLE_SEED:
                raise ValueError(f"{name}: active campaign requires {SEED_POLICY}")
        out = root / name
        if (out / "summary.json").exists():
            # A summary without the matching attestation is not resumable.  It
            # may be an old/preliminary run and must be audited or moved aside
            # explicitly rather than silently becoming paired evidence.
            evidence_path = out / "evidence.json"
            try:
                evidence = json.loads(evidence_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                evidence = {}
            if evidence.get("status") != "complete" or evidence.get("returncode") != 0:
                raise RuntimeError(f"refusing to skip {name}: summary exists without evidence.json")
            if active_seed_policy:
                identity = evidence.get("identity", {})
                if (identity.get("seed") != SINGLE_SEED
                        or evidence.get("seed_policy", SEED_POLICY) != SEED_POLICY
                        or identity.get("seed_policy", SEED_POLICY) != SEED_POLICY):
                    raise RuntimeError(f"refusing to skip {name}: evidence violates {SEED_POLICY}")
            done.append(dict(name=name, status="skipped"))
            continue
        args["out"] = str(out)
        if spec.get("formal"):
            pol = args.get("policy", args.get("system", ""))
            profiles = args.get("profile_by_system", {}) or {}
            if pol in {"distserve_static", "dynamollm", "ecoserve"} and pol not in profiles:
                done.append(dict(name=name, status="inconclusive", error="independent profile missing"))
                continue
        if gpus:
            args["gpus"] = gpus
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {json.dumps({k: args[k] for k in ('policy', 'dataset', 'rate', 'layout', 'clocks') if k in args})}",
              flush=True)
        if dry:
            done.append(dict(name=name, status="dry", args=args))
            continue
        try:
            result = bench_point(args)
            _write_evidence(out, args, result, result.pop("_evidence_trace"), Path(args["corpus"]))
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
