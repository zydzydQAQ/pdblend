"""Isolated, frozen v3 experiments and paired three-seed evidence (CPU only).

CLI: ``python -m ecopadg.serving.evaluation_v3 {plan,manifest,report} --help``.
Plans bind actual files, full baseline proofs, held-out corpora and every cell.
Reports never drop missing/failed formal-domain points or select winning seeds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from .evidence import REQUIRED_MECHANISMS
from ecopadg.measure.power import instant_power_verified

MODELS = ("7b", "14b", "32b")
DATASETS = ("alpaca", "sharegpt", "longbench")
BASELINES = ("mixed", "distserve", "ecoserve", "dynamollm")
SYSTEMS = ("pdblend",) + BASELINES
SEEDS = (101, 202, 303)
Q, TTFT, TPOT = .9, 5., .1
MIXED_GRID = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95, 1., 1.1, 1.2)
BANDS = ("low", "medium", "high")


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _seeds(seeds):
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int for s in seeds):
        raise ValueError("exactly three distinct integer seeds required")
    return tuple(seeds)


def round_alpha(round_number):
    if type(round_number) is not int or round_number < 1:
        raise ValueError("formal round must be a positive integer")
    return .05 / (round_number * (round_number + 1))


def log_ratio_upper(values, round_number):
    """One-sided exact t(df=2) upper limit on three paired log ratios.

    Round alpha spending sums to .05. All 12 model/baseline comparisons and
    both endpoints must pass: this is a predeclared intersection-union test,
    not a claim of simultaneous coverage for all displayed intervals.
    """
    if len(values) != 3 or any(not math.isfinite(x) for x in values):
        raise ValueError("three finite independent seed-level log ratios required")
    alpha = round_alpha(round_number)
    t = (1 - 2 * alpha) / math.sqrt(2 * alpha * (1 - alpha))
    mean = statistics.mean(values)
    upper = mean + t * statistics.stdev(values) / math.sqrt(3)
    return dict(alpha=alpha, df=2, mean_log_ratio=mean, upper_log_ratio=upper,
                ratio=math.exp(mean), upper_ratio=math.exp(upper) if upper < 709 else None,
                passed=upper < 0, seed_log_ratios=list(values))


def build_rate_grid(capacities):
    if set(capacities) != set(SYSTEMS) or any(not _positive(v) for v in capacities.values()):
        raise ValueError("positive independently confirmed capacities for all five systems required")
    lo, hi = min(capacities.values()), max(capacities.values())
    rates = [capacities["mixed"] * x for x in MIXED_GRID]
    rates += [v * x for v in capacities.values() for x in (.9, 1., 1.1)]
    rates += [lo * x for x in (.05, .15, .45, .75)] + [1.2 * hi]
    unique = []
    for value in sorted(rates):
        if not unique or not math.isclose(value, unique[-1], rel_tol=1e-12, abs_tol=0):
            unique.append(value)
    result = []
    for rate in unique:
        rho = rate / lo
        band = next((b for b, upper in zip(BANDS, (.3, .6, .9)) if rho <= upper + 1e-12), None)
        result.append(dict(rate=rate, relative_min_capacity=rho, load=band or "sweep",
                           main_domain=band is not None))
    return result


def _records(payload, dataset):
    if payload.get("dataset") != dataset:
        raise ValueError("corpus dataset identity mismatch")
    formal = payload.get("formal_pool", payload.get("formal"))
    if isinstance(formal, dict):
        formal = [r for batch in formal.values() for r in batch]
    if not isinstance(formal, list) or not formal:
        raise ValueError("explicit nonempty held-out formal pool required")
    forbidden = {r["request_shape_sha256"] for split in ("calibration", "development")
                 for r in payload.get(split, [])}
    forbidden_prompts = {_digest(r["prompt"]) for split in ("calibration", "development")
                         for r in payload.get(split, [])}
    unique = {}
    for r in formal:
        key, prompt = r.get("request_shape_sha256"), r.get("prompt")
        if (not _sha(key) or not isinstance(prompt, list) or not prompt
                or any(type(t) is not int or t < 0 for t in prompt)
                or r.get("input_tokens") != len(prompt)
                or type(r.get("output_tokens")) is not int or r["output_tokens"] < 1):
            raise ValueError("formal records require verified token shapes and prescribed output work")
        if key in forbidden or _digest(prompt) in forbidden_prompts:
            raise ValueError("formal corpus overlaps calibration/development")
        selected = {k: r[k] for k in ("prompt", "input_tokens", "output_tokens", "request_shape_sha256")}
        if key in unique and unique[key] != selected:
            raise ValueError("one source identity has conflicting token work")
        unique[key] = selected
    return list(unique.values())


def _rng(seed, stream):
    return random.Random(int(_digest([seed, stream]), 16))


def _trace(records, arrivals, *, dataset, model, seed, round_number, load, resample):
    return dict(schema=3, evaluation_protocol="evaluation-v3", dataset=dataset, model=model,
                split="formal", load=load, seed=seed, round=round_number,
                duration_s=arrivals[-1] - arrivals[0], process="open_loop_poisson",
                requests=[dict(arrival_s=t, prompt_len=r["input_tokens"], output_len=r["output_tokens"])
                          for r, t in zip(records, arrivals)],
                prompts=[r["prompt"] for r in records],
                source_shapes=[r["request_shape_sha256"] for r in records],
                resampling="with_replacement_from_heldout_pool" if resample else "without_replacement",
                slo_ttft_s=TTFT, slo_tpot_s=TPOT, slo_attainment_target=Q)


def generate_static_trace(records, rate, seed, *, dataset, model, round_number=1,
                          load="sweep", resample=False, min_requests=1000, min_duration_s=300):
    if (not records or not _positive(rate) or min_requests < 1000 or min_duration_s < 300
            or dataset not in DATASETS or model not in MODELS):
        raise ValueError("static trace requires >=1000 requests, >=300s and explicit model/dataset")
    round_alpha(round_number)
    arrivals_rng = _rng(seed, ["arrival", model, dataset, rate, round_number])
    select_rng = _rng(seed, ["selection", model, dataset, rate, round_number])
    available = list(records)
    select_rng.shuffle(available)
    chosen, arrivals = [], []
    at = 0.
    while len(chosen) < min_requests or at < min_duration_s:
        if chosen:
            at += arrivals_rng.expovariate(rate)
        if resample:
            r = select_rng.choice(records)
        elif available:
            r = available.pop()
        else:
            raise ValueError("held-out pool exhausted; explicitly authorize resampling or supply more examples")
        chosen.append(r)
        arrivals.append(at)
    result = _trace(chosen, arrivals, dataset=dataset, model=model, seed=seed,
                    round_number=round_number, load=load, resample=resample)
    result.update(rate=rate, stopping_rule="both request count >=1000 and arrival span >=300s",
                  n_unique_shapes=len(set(result["source_shapes"])))
    return result


def generate_dynamic_trace(corpora, capacities, seed, *, model, round_number=1, resample=False):
    if (set(corpora) != set(DATASETS) or set(capacities) != set(DATASETS)
            or any(not corpora[d] or not _positive(capacities[d]) for d in DATASETS)):
        raise ValueError("three corpora and positive common capacities required")
    phases = [( .15, (.6, .3, .1)), (.45, (.6, .3, .1)), (.85, (.6, .3, .1)),
              ( .85, (.1, .2, .7)), (.45, (.2, .6, .2)), (.15, (.6, .3, .1))]
    arrivals_rng, select_rng = _rng(seed, [model, round_number, "dynamic_arrival"]), _rng(seed, [model, round_number, "dynamic_selection"])
    available = {d: list(rows) for d, rows in corpora.items()}
    for rows in available.values():
        select_rng.shuffle(rows)
    chosen, arrivals, labels, metadata = [], [], [], []
    for index, (fraction, weights) in enumerate(phases):
        start, end = index * 900., (index + 1) * 900.
        rate = fraction / sum(w / capacities[d] for d, w in zip(DATASETS, weights))
        at = start if index == 0 else start + arrivals_rng.expovariate(rate)
        while at < end:
            dataset = select_rng.choices(DATASETS, weights=weights)[0]
            if resample:
                record = select_rng.choice(corpora[dataset])
            elif available[dataset]:
                record = available[dataset].pop()
            else:
                raise ValueError("dynamic formal pool exhausted; resampling must be explicit")
            chosen.append(record); arrivals.append(at); labels.append(dataset)
            at += arrivals_rng.expovariate(rate)
        metadata.append(dict(start_s=start, end_s=end, capacity_fraction=fraction,
                             length_mix=list(weights), rate=rate))
    # This explicit boundary request records a full 90-minute observation span.
    # It is the sole deterministic endpoint; it is disclosed in the artifact.
    dataset = "alpaca"
    if resample:
        record = select_rng.choice(corpora[dataset])
    elif available[dataset]:
        record = available[dataset].pop()
    else:
        raise ValueError("dynamic endpoint requires one more held-out request")
    chosen.append(record); arrivals.append(5400.); labels.append(dataset)
    result = _trace(chosen, arrivals, dataset="dynamic", model=model, seed=seed,
                    round_number=round_number, load="changing", resample=resample)
    result.update(phases=metadata, source_datasets=labels, endpoint_request=True,
                  capacity_model="harmonic mixture of independently confirmed five-system minimum capacities")
    return result


def _freeze_paths(paths):
    if not isinstance(paths, list) or not paths:
        raise ValueError("nonempty explicit source/model/profile file groups required")
    return {str(Path(p).resolve()): digest_file(p) for p in paths}


def _check_files(files):
    return [p for p, h in files.items() if not Path(p).is_file() or digest_file(p) != h]


def _mechanisms(path):
    registry = _read(path)
    files = {str(Path(path).resolve()): digest_file(path)}
    for system in BASELINES:
        for mechanism in REQUIRED_MECHANISMS[system]:
            proof = registry.get(system, {}).get(mechanism, {})
            artifact = proof.get("artifact")
            if proof.get("passed") is not True or not artifact or digest_file(artifact) != proof.get("sha256"):
                raise ValueError(f"full baseline proof missing: {system}/{mechanism}")
            files[str(Path(artifact).resolve())] = proof["sha256"]
    return files


def _config(path, system):
    cfg = _read(path)
    strategy = cfg.get("strategy", "")
    if (strategy != system and not (system == "pdblend" and strategy.startswith("pdblend-"))):
        raise ValueError("config strategy differs from full system identity (resident Dynamo is not full)")
    if (cfg.get("evaluation_protocol") != "evaluation-v3" or cfg.get("slo_ttft_s") != TTFT
            or cfg.get("slo_tpot_s") != TPOT or cfg.get("slo_attainment_target") != Q
            or cfg.get("power_mode") != "instant"):
        raise ValueError("v3 config must explicitly freeze q=.9, TTFT=5, TPOT=.1 and instant power")
    return cfg


def create_plan(spec, out_dir):
    """Materialize a full plan and paired traces from an explicit JSON spec.

    spec: round, seeds, resample_heldout, models[7b/14b/32b]. Each model has
    source_files/model_files/profile_files (nonempty lists), engine_image,
    configs[system], corpora[dataset], mechanisms (registry path), and
    capacity_proofs[dataset] (JSON mapping each of five systems to a confirmed
    capacity record: capacity_rps, split, confirmed, slo_attainment,
    work_complete, stable_queue, open_loop_verified).
    """
    seeds = _seeds(spec.get("seeds", SEEDS))
    round_number = spec.get("round", 1)
    round_alpha(round_number)
    if set(spec.get("models", {})) != set(MODELS):
        raise ValueError("all three models 7b/14b/32b required")
    if type(spec.get("resample_heldout", False)) is not bool:
        raise ValueError("resample_heldout must be an explicit boolean")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "traces").mkdir()
    plan = dict(schema=3, evaluation_protocol="evaluation-v3", round=round_number,
                seeds=list(seeds), q=Q, slo_ttft_s=TTFT, slo_tpot_s=TPOT,
                models={}, cells=[], artifacts={}, alpha=round_alpha(round_number),
                weighting="equal datasets; equal three minimum-capacity load bands; equal points within band",
                resample_heldout=spec.get("resample_heldout", False))
    for model in MODELS:
        src = spec["models"][model]
        if set(src["configs"]) != set(SYSTEMS) or set(src["corpora"]) != set(DATASETS):
            raise ValueError("five frozen system configs and three corpora required per model")
        groups = {name: _freeze_paths(src[name + "_files"]) for name in ("source", "model", "profile")}
        groups["source"][str(Path(__file__).resolve())] = digest_file(__file__)
        image = src["engine_image"]
        if not isinstance(image, str) or not image.startswith("sha256:") or not _sha(image[7:]):
            raise ValueError("immutable engine image sha256 required")
        identities = {name + "_sha256": _digest(files) for name, files in groups.items()}
        identities["engine_image"] = image
        for files in groups.values():
            plan["artifacts"].update(files)
        mechanism_files = _mechanisms(src["mechanisms"])
        plan["artifacts"].update(mechanism_files)
        configs = {}
        for system, path in src["configs"].items():
            _config(path, system)
            absolute = str(Path(path).resolve())
            configs[system] = dict(path=absolute, sha256=digest_file(path))
            plan["artifacts"][absolute] = configs[system]["sha256"]
        corpora, capacities = {}, {}
        for dataset in DATASETS:
            path = str(Path(src["corpora"][dataset]).resolve())
            corpora[dataset] = _records(_read(path), dataset)
            plan["artifacts"][path] = digest_file(path)
            proof_path = str(Path(src["capacity_proofs"][dataset]).resolve())
            proofs = _read(proof_path)
            if set(proofs) != set(SYSTEMS):
                raise ValueError("capacity proof must include PDB and all four baselines")
            for proof in proofs.values():
                if (not _positive(proof.get("capacity_rps")) or proof.get("split") != "calibration"
                        or not Q <= proof.get("slo_attainment", -1) <= 1
                        or any(proof.get(k) is not True for k in
                               ("confirmed", "work_complete", "stable_queue", "open_loop_verified"))):
                    raise ValueError("unconfirmed, unstable or infeasible calibration capacity")
            capacities[dataset] = {s: proofs[s]["capacity_rps"] for s in SYSTEMS}
            plan["artifacts"][proof_path] = digest_file(proof_path)
        dependencies = set(mechanism_files)
        dependencies.update(p for files in groups.values() for p in files)
        dependencies.update(v["path"] for v in configs.values())
        dependencies.update(str(Path(p).resolve()) for p in src["corpora"].values())
        dependencies.update(str(Path(p).resolve()) for p in src["capacity_proofs"].values())
        plan["models"][model] = dict(identities=identities, groups=groups, configs=configs, capacities=capacities,
            dependencies=sorted(dependencies), mechanisms=str(Path(src["mechanisms"]).resolve()))
        for dataset in DATASETS + ("dynamic",):
            grid = build_rate_grid(capacities[dataset]) if dataset != "dynamic" else [dict(load="changing", main_domain=False)]
            for index, point in enumerate(grid):
                for seed in seeds:
                    cell_id = f"{model}-{dataset}-{index:03d}-{seed}"
                    if dataset == "dynamic":
                        trace = generate_dynamic_trace(corpora, {d: min(capacities[d].values()) for d in DATASETS},
                            seed, model=model, round_number=round_number, resample=plan["resample_heldout"])
                    else:
                        trace = generate_static_trace(corpora[dataset], point["rate"], seed,
                            dataset=dataset, model=model, round_number=round_number, load=point["load"],
                            resample=plan["resample_heldout"])
                    trace["cell_id"] = cell_id
                    path = out / "traces" / (cell_id + ".json")
                    _write(path, trace)
                    plan["artifacts"][str(path)] = digest_file(path)
                    plan["cells"].append(dict(point, cell_id=cell_id, model=model, dataset=dataset, seed=seed,
                        split="formal", trace_path=str(path), trace_sha256=digest_file(path),
                        n_requests=len(trace["requests"]), trace_duration_s=trace["duration_s"],
                        expected_generated_tokens=sum(r["output_len"] for r in trace["requests"])))
    _write(out / "plan.json", plan)
    return plan


def _validate_plan(plan):
    if (plan.get("schema") != 3 or plan.get("evaluation_protocol") != "evaluation-v3"
            or plan.get("q") != Q or plan.get("slo_ttft_s") != TTFT or plan.get("slo_tpot_s") != TPOT
            or set(plan.get("models", {})) != set(MODELS)):
        raise ValueError("invalid v3 plan identity")
    seeds = _seeds(plan["seeds"])
    if plan.get("alpha") != round_alpha(plan["round"]):
        raise ValueError("round alpha differs from spending protocol")
    expected = set()
    for model in MODELS:
        info = plan["models"][model]
        if set(info.get("configs", {})) != set(SYSTEMS):
            raise ValueError("missing frozen system config")
        for system, cfg in info["configs"].items():
            if plan.get("artifacts", {}).get(cfg["path"]) != cfg["sha256"]:
                raise ValueError("system config is not in the artifact freeze")
        if not info.get("dependencies") or info.get("mechanisms") not in info["dependencies"]:
            raise ValueError("missing full-baseline mechanism freeze")
        for dataset in DATASETS:
            for index, point in enumerate(build_rate_grid(info["capacities"][dataset])):
                for seed in seeds:
                    expected.add((f"{model}-{dataset}-{index:03d}-{seed}", model, dataset, seed,
                                  point["rate"], point["main_domain"], point["load"]))
        for seed in seeds:
            expected.add((f"{model}-dynamic-000-{seed}", model, "dynamic", seed, None, False, "changing"))
        for kind in ("source", "model", "profile"):
            files = info.get("groups", {}).get(kind, {})
            if (not files or info["identities"].get(kind + "_sha256") != _digest(files)
                    or any(plan.get("artifacts", {}).get(p) != h for p, h in files.items())):
                raise ValueError("missing or inconsistent source/model/profile freeze")
    actual = [(c["cell_id"], c["model"], c["dataset"], c["seed"], c.get("rate"),
               c["main_domain"], c["load"]) for c in plan["cells"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("plan omits or alters prespecified rate/seed/domain cells")


def _verify_trace(plan, cell):
    path = cell["trace_path"]
    if digest_file(path) != cell["trace_sha256"] or plan["artifacts"].get(path) != cell["trace_sha256"]:
        raise ValueError("trace bytes differ from frozen plan")
    trace = _read(path)
    if any(trace.get(k) != cell[k] for k in ("cell_id", "model", "dataset", "seed", "load", "split")):
        raise ValueError("trace evaluation identity mismatch")
    requests = trace.get("requests", [])
    if not requests or len(trace.get("prompts", [])) != len(requests) or len(trace.get("source_shapes", [])) != len(requests):
        raise ValueError("trace missing explicit work or source shapes")
    arrivals = [r.get("arrival_s") for r in requests]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in arrivals)
            or arrivals[0] != 0 or any(b <= a for a, b in zip(arrivals, arrivals[1:]))):
        raise ValueError("trace must preserve strict open-loop arrival times")
    if (trace.get("schema") != 3 or trace.get("round") != plan["round"]
            or trace.get("process") != "open_loop_poisson"
            or trace.get("duration_s") != arrivals[-1]
            or cell["n_requests"] != len(requests) or cell["trace_duration_s"] != arrivals[-1]):
        raise ValueError("trace size/duration/round does not match plan")
    for r, prompt, shape in zip(requests, trace["prompts"], trace["source_shapes"]):
        if (not isinstance(prompt, list) or not prompt or r.get("prompt_len") != len(prompt)
                or type(r.get("output_len")) is not int or r["output_len"] < 1 or not _sha(shape)):
            raise ValueError("invalid prescribed token work")
    if cell["dataset"] == "dynamic":
        if arrivals[-1] != 5400:
            raise ValueError("dynamic trace must span 90 minutes")
    elif len(requests) < 1000 or arrivals[-1] < 300 or trace.get("rate") != cell["rate"]:
        raise ValueError("static trace requires >=1000 requests and >=300 seconds")
    work = sum(r["output_len"] for r in requests)
    if cell["expected_generated_tokens"] != work:
        raise ValueError("planned work differs from actual trace")
    return trace


def verify_cell(plan_path, cell_id, system, config_path, trace_path):
    """Call before and after live execution; return summary identity fields.

    Actual engine image and loaded-source checks remain the runtime's job.
    Hashes of every frozen local artifact are checked here; no summary claim
    can replace checking the trace itself.
    """
    plan = _read(plan_path)
    _validate_plan(plan)
    cells = [c for c in plan["cells"] if c["cell_id"] == cell_id]
    if len(cells) != 1 or system not in SYSTEMS:
        raise ValueError("unknown cell/system")
    cell = cells[0]
    info = plan["models"][cell["model"]]
    files = {p: plan["artifacts"][p] for p in info["dependencies"] + [cell["trace_path"]]}
    if _check_files(files):
        raise ValueError("frozen source/model/profile/config/trace artifacts changed")
    _mechanisms(info["mechanisms"])
    expected = info["configs"][system]
    if (str(Path(config_path).resolve()) != expected["path"] or digest_file(config_path) != expected["sha256"]
            or str(Path(trace_path).resolve()) != cell["trace_path"]):
        raise ValueError("config or trace is not the one frozen for this cell")
    _config(config_path, system)
    _verify_trace(plan, cell)
    return dict(evaluation_schema=3, plan_sha256=digest_file(plan_path), cell_id=cell_id,
                model=cell["model"], dataset=cell["dataset"], seed=cell["seed"], load=cell["load"],
                split="formal", round=plan["round"], system=system,
                config_sha256=expected["sha256"], trace_sha256=cell["trace_sha256"],
                slo_attainment_target=Q, slo_ttft_s=TTFT, slo_tpot_s=TPOT, **info["identities"])


def summaries_manifest(plan_path, paths):
    return dict(schema=3, plan_sha256=digest_file(plan_path), summaries=[
        dict(path=str(Path(p).resolve()), sha256=digest_file(p)) for p in paths])


def freeze_historical_baselines(csv_paths, description_path, out_dir):
    """Index existing observations without changing, rerunning or replacing any.

    Missing original remote artifacts remain explicitly missing; embedded
    summaries in the existing reference JSON retain their actual status.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = _freeze_paths([str(description_path)] + [str(p) for p in csv_paths])
    entries, seen, references = [], set(), {}
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            model, dataset, system = row["model"].lower(), row["dataset"], row["baseline"]
            key = (model, dataset, system)
            if key in seen:
                raise ValueError("historical index must not silently select one of duplicate baselines")
            seen.add(key)
            ref_path = row["reference"]
            if ref_path not in references:
                references[ref_path] = _read(ref_path)
                sources[str(Path(ref_path).resolve())] = digest_file(ref_path)
            matches = [by_system[system] for point, by_system in references[ref_path]["entries"].items()
                       if point.split("/")[0] == dataset and system in by_system]
            if len(matches) != 1:
                raise ValueError("historical point is missing or ambiguous in its original reference")
            ref = matches[0]
            summary = ref["summary"]
            energy, slo = float(row["baseline_energy_j_raw"]), float(row["baseline_slo"])
            completed, expected = int(row["baseline_completed"]), int(row["baseline_expected"])
            if (energy != summary.get("energy_j") or slo != summary.get("slo_attainment")
                    or completed != summary.get("completed") or expected != summary.get("n_expected")):
                raise ValueError("CSV observation conflicts with embedded historical summary")
            artifacts, rate = [], None
            for path, claimed in ref.get("artifacts", {}).items():
                actual = digest_file(path) if Path(path).is_file() else None
                status = "verified_local" if actual == claimed else "missing_local" if actual is None else "changed_local"
                artifacts.append(dict(path=path, recorded_sha256=claimed, observed_sha256=actual, status=status))
                if actual == claimed and claimed == ref.get("trace_sha256"):
                    trace = _read(path)
                    rate = trace.get("rate")
            complete = row["baseline_work_complete"].lower() == "true"
            good_float = slo * expected
            good = round(good_float) if math.isclose(good_float, round(good_float), abs_tol=1e-9) else None
            limitations = ["single historical run; no independent three-seed formal interval",
                          "no new baseline runs authorized; missing artifacts are not backfilled by measurement"]
            if system == "dynamollm-resident":
                limitations.append("resident mechanism only; not a full DynamoLLM baseline")
            if not complete:
                limitations.append("incomplete prescribed work; cannot claim same-workload energy saving")
            if any(a["status"] == "missing_local" for a in artifacts):
                limitations.append("some original artifacts are unavailable locally; recorded hashes are not live verification")
            if any(a["status"] == "changed_local" for a in artifacts):
                limitations.append("some local artifacts differ from historical hashes; do not overwrite the historical identity")
            entries.append(dict(model=model, dataset=dataset, system=system, historical_run_id=ref["id"],
                reference_path=ref_path, reference_sha256=digest_file(ref_path), csv_path=str(Path(csv_path).resolve()),
                csv_sha256=digest_file(csv_path), trace_sha256=ref.get("trace_sha256"), seed=summary.get("seed"),
                rate_rps=rate, rate_status="verified_trace" if rate is not None else "unavailable_local_trace",
                model_name=ref.get("model"), slo_thresholds_s=ref.get("slo"),
                comparison_q=Q, slo_feasible_at_comparison_q=slo >= Q, work_complete=complete,
                completed=completed, n_expected=expected, good_requests_derived=good,
                energy_j=energy, slo_attainment=slo,
                energy_per_good_request_j=energy/good if good else None,
                expected_generated_tokens=ref.get("expected_output_tokens"),
                summary=summary, raw_artifacts=artifacts, limitations=limitations,
                formal_eligible=False, evidence_class="fixed_historical_development_reference"))
    required = {(m, d, "dynamollm-resident" if s == "dynamollm" and m in ("7b", "32b") else s)
                for m in MODELS for d in DATASETS for s in BASELINES}
    if seen != required:
        raise ValueError("requires all 36 historical observations, including failures and resident labels")
    manifest = dict(schema=3, mode="historical_baselines", formal_eligible=False,
        policy="read-only fixed references; no baseline rerun, replacement, or formal promotion",
        sources=sources, entries=entries,
        inventory=dict(observations=len(entries), work_complete=sum(e["work_complete"] for e in entries),
            locally_missing_artifact_references=sum(a["status"] == "missing_local" for e in entries for a in e["raw_artifacts"]),
            changed_artifact_references=sum(a["status"] == "changed_local" for e in entries for a in e["raw_artifacts"])))
    _write(out / "manifest.json", manifest)
    return manifest


def historical_report(candidates, manifest):
    """Compare only PDB candidates to fixed old references, never a formal pass.

    candidates is a list of {path, model}; the explicit model label is needed
    for legacy summaries that did not record a canonical 7b/14b/32b identity.
    Exact trace/work/power matches permit descriptive energy ratios only.
    """
    if isinstance(manifest, (str, Path)):
        manifest = _read(manifest)
    if (manifest.get("schema") != 3 or manifest.get("mode") != "historical_baselines"
            or not manifest.get("sources") or _check_files(manifest["sources"])):
        raise ValueError("fixed historical index sources changed or are unavailable")
    # Bind indexed observations back to the original reference, so editing a
    # cached energy/completion value cannot manufacture a historical gain.
    for base in manifest.get("entries", []):
        path = str(Path(base["reference_path"]).resolve())
        if manifest["sources"].get(path) != base["reference_sha256"]:
            raise ValueError("historical reference is not bound by the frozen source index")
        ref = _read(path)
        matches = [entry for systems in ref.get("entries", {}).values() for entry in systems.values()
                   if entry.get("id") == base["historical_run_id"]]
        if (len(matches) != 1 or matches[0].get("summary") != base["summary"]
                or base["energy_j"] != base["summary"].get("energy_j")
                or base["n_expected"] != base["summary"].get("n_expected")
                or base["trace_sha256"] != matches[0].get("trace_sha256")
                or base["slo_thresholds_s"] != matches[0].get("slo")):
            raise ValueError("historical index observation differs from its immutable original reference")
    comparisons = []
    for candidate in candidates:
        path, model = candidate["path"], candidate["model"].lower()
        row = _read(path)
        if model not in MODELS or not row.get("system", row.get("variant", "")).startswith("pdblend"):
            raise ValueError("historical-report accepts only explicitly identified PDB candidates")
        for base in manifest["entries"]:
            if (model, row.get("dataset")) != (base["model"], base["dataset"]):
                continue
            b, why = base["summary"], []
            if not base["work_complete"] or row.get("completed") != base["n_expected"] or row.get("n_expected") != base["n_expected"]:
                why.append("prescribed request work incomplete or different")
            if (row.get("generated_tokens") != base["expected_generated_tokens"]
                    or row.get("expected_generated_tokens") != base["expected_generated_tokens"]):
                why.append("prescribed output work incomplete or different")
            if not _sha(row.get("trace_sha256")) or row.get("trace_sha256") != base["trace_sha256"]:
                why.append("trace content/rate/arrival/seed is different or unverified")
            if any(row.get(k) != b.get(k) for k in ("measurement_schema", "gpu_count", "power_mode", "power_source_id", "power_field_id")):
                why.append("measurement protocol or GPU energy boundary differs")
            if not instant_power_verified(row) or row.get("validity") != "ok" or row.get("incomplete_drain", False):
                why.append("candidate measurement/work is invalid")
            if [row.get("slo_ttft_s"), row.get("slo_tpot_s")] != base["slo_thresholds_s"]:
                why.append("candidate latency SLO identity unavailable or differs")
            if not _positive(row.get("energy_j")):
                why.append("candidate lacks measured positive energy")
            comparisons.append(dict(model=model, dataset=base["dataset"], baseline=base["system"],
                candidate_path=str(Path(path).resolve()), candidate_sha256=digest_file(path),
                historical_run_id=base["historical_run_id"], reference_sha256=base["reference_sha256"],
                evidence_class="historical_descriptive_comparison" if not why else "historical_not_comparable",
                comparable=not why, reasons=why, baseline_energy_j=base["energy_j"],
                candidate_energy_j=row.get("energy_j"),
                energy_ratio=row["energy_j"]/base["energy_j"] if not why else None,
                candidate_slo_attainment=row.get("slo_attainment"), baseline_slo_attainment=base["slo_attainment"],
                formal_eligible=False, limitations=base["limitations"]))
    return dict(verdict="development_historical_only", formal_eligible=False, comparisons=comparisons,
                policy="frozen old baselines; no baseline rerun and no same-protocol formal claim")


def evaluate(plan_path, manifest):
    """Audit a frozen list of summary files; return all failures and 12 tests."""
    plan = _read(plan_path)
    plan_hash = digest_file(plan_path)
    reasons, invalid, grouped = [], [], defaultdict(list)
    try:
        _validate_plan(plan)
        changed = _check_files(plan["artifacts"])
        if changed:
            reasons.append("frozen artifacts changed: " + ", ".join(changed))
        for info in plan["models"].values():
            _mechanisms(info["mechanisms"])
            for system, cfg in info["configs"].items():
                _config(cfg["path"], system)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return dict(verdict="evidence_insufficient", reasons=[str(exc)], comparisons={})
    if isinstance(manifest, (str, Path)):
        manifest = _read(manifest)
    if manifest.get("schema") != 3 or manifest.get("plan_sha256") != plan_hash:
        reasons.append("summaries manifest does not bind this plan")
    seen_paths = set()
    for entry in manifest.get("summaries", []):
        try:
            path = str(Path(entry["path"]).resolve())
            if path in seen_paths or digest_file(path) != entry["sha256"]:
                raise ValueError("duplicate or changed summary file")
            seen_paths.add(path)
            row = _read(path)
            grouped[(row.get("cell_id"), row.get("system"))].append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append("summary manifest: " + str(exc))
    expected_keys = {(c["cell_id"], s) for c in plan["cells"] for s in SYSTEMS}
    if set(grouped) != expected_keys or any(len(rows) != 1 for rows in grouped.values()):
        reasons.append("missing, duplicate or unexpected summary cells; no formal failure may be omitted")
    checked = {}
    for cell in plan["cells"]:
        try:
            _verify_trace(plan, cell)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reasons.append(cell["cell_id"] + ": " + str(exc))
            continue
        for system in SYSTEMS:
            rows = grouped.get((cell["cell_id"], system), [])
            if len(rows) != 1:
                continue
            row = rows[0]
            info = plan["models"][cell["model"]]
            identity = dict(evaluation_schema=3, plan_sha256=plan_hash,
                cell_id=cell["cell_id"], model=cell["model"], dataset=cell["dataset"], seed=cell["seed"],
                load=cell["load"], split="formal", round=plan["round"], system=system,
                config_sha256=info["configs"][system]["sha256"], trace_sha256=cell["trace_sha256"],
                slo_attainment_target=Q, slo_ttft_s=TTFT, slo_tpot_s=TPOT, **info["identities"])
            error = None
            if any(row.get(k) != v for k, v in identity.items()):
                error = "source/model/profile/config/trace/system identity mismatch"
            elif row.get("variant", "").startswith("dynamollm-resident"):
                error = "resident Dynamo cannot pass the full baseline contract"
            elif row.get("n_expected") != cell["n_requests"] or row.get("expected_generated_tokens") != cell["expected_generated_tokens"]:
                error = "summary work differs from frozen trace"
            elif (row.get("measurement_valid") is not True or row.get("formal_eligible") is not True
                  or row.get("measurement_schema") not in (2, 3) or not instant_power_verified(row)
                  or row.get("gpu_count") != 8 or not _positive(row.get("energy_j"))
                  or row.get("incomplete_drain", False)):
                error = "measurement/provenance/drain invalid; overload results must still be measurable"
            elif cell["main_domain"]:
                good = row.get("good_requests")
                if (row.get("measurement_valid") is not True or row.get("work_complete") is not True
                        or row.get("validity") != "ok" or row.get("formal_eligible") is not True
                        or row.get("measurement_schema") not in (2, 3) or not instant_power_verified(row)
                        or row.get("gpu_count") != 8 or not _positive(row.get("energy_j"))
                        or row.get("completed") != cell["n_requests"]
                        or row.get("generated_tokens") != cell["expected_generated_tokens"]
                        or type(good) is not int or not 0 < good <= cell["n_requests"]
                        or not math.isclose(row.get("slo_attainment", -1), good / cell["n_requests"], abs_tol=1e-12)
                        or good / cell["n_requests"] < Q or row.get("incomplete_drain", False)):
                    error = "formal-domain measurement/work/SLO failure (retained, never dropped)"
            if error:
                invalid.append(dict(cell_id=cell["cell_id"], system=system, reason=error))
            else:
                checked[(cell["cell_id"], system)] = row
    comparisons = {}
    for model in MODELS:
        main = [c for c in plan["cells"] if c["model"] == model and c["main_domain"]]
        for baseline in BASELINES:
            values = {"energy_j": [], "energy_per_good_request_j": []}
            complete = True
            for seed in plan["seeds"]:
                buckets = defaultdict(lambda: {k: [] for k in values})
                for cell in main:
                    if cell["seed"] != seed:
                        continue
                    a, b = checked.get((cell["cell_id"], "pdblend")), checked.get((cell["cell_id"], baseline))
                    if a is None or b is None:
                        complete = False
                        continue
                    log_e = math.log(a["energy_j"] / b["energy_j"])
                    bucket = buckets[(cell["dataset"], cell["load"])]
                    bucket["energy_j"].append(log_e)
                    bucket["energy_per_good_request_j"].append(log_e + math.log(b["good_requests"] / a["good_requests"]))
                if set(buckets) != {(d, band) for d in DATASETS for band in BANDS}:
                    complete = False
                for metric in values:
                    if len(buckets) == 9:
                        values[metric].append(statistics.mean(statistics.mean(v[metric]) for v in buckets.values()))
            metrics = {k: log_ratio_upper(v, plan["round"]) for k, v in values.items()} if complete else {}
            comparisons[f"{model}/{baseline}"] = dict(complete=complete, metrics=metrics,
                passed=complete and all(m["passed"] for m in metrics.values()))
    if invalid:
        reasons.append("one or more frozen cells failed identity or formal-domain gates")
    return dict(verdict="evidence_insufficient" if reasons else
                "target_achieved" if all(c["passed"] for c in comparisons.values()) else "target_not_achieved",
                reasons=reasons, invalid_cells=invalid, comparisons=comparisons,
                n_expected_summaries=len(expected_keys), n_supplied_summaries=sum(map(len, grouped.values())),
                round=plan["round"], alpha=round_alpha(plan["round"]),
                inference="all 12 comparisons and both endpoints must pass; three independent paired seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="freeze spec and generate all paired open-loop traces")
    p.add_argument("--spec", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("manifest", help="freeze the explicit list of all summary files")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report", help="audit all expected cells and compute prespecified comparisons")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("freeze-historical", help="index existing baseline CSV/reference files without rerunning")
    p.add_argument("--csv", nargs="+", type=Path, required=True)
    p.add_argument("--description", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("historical-report", help="descriptive PDB-only comparison to fixed historical references")
    p.add_argument("--candidates", type=Path, required=True, help="JSON list of {path, model}")
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = create_plan(_read(args.spec), args.out)
        print(json.dumps(dict(plan=str(args.out / "plan.json"), cells=len(plan["cells"]),
                              runs=len(plan["cells"]) * len(SYSTEMS))))
    elif args.command == "manifest":
        _write(args.out, summaries_manifest(args.plan, args.summaries))
    elif args.command == "freeze-historical":
        frozen = freeze_historical_baselines(args.csv, args.description, args.out)
        print(json.dumps(frozen["inventory"]))
    elif args.command == "historical-report":
        report = historical_report(_read(args.candidates), args.manifest)
        _write(args.out, report)
        print(json.dumps(dict(verdict=report["verdict"], comparisons=len(report["comparisons"])) ))
    else:
        report = evaluate(args.plan, args.manifest)
        _write(args.out, report)
        print(json.dumps(dict(verdict=report["verdict"], reasons=report["reasons"])))


if __name__ == "__main__":
    main()
