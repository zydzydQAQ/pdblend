"""Explicit-input scalability reports; no recursive result discovery or simulation claims."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .audit import audit_run
from .statistics import backlog_stability, capacity_interval, paired_ratio_interval
from .protocol import FORMAL_SEEDS, FORMAL_SCALES, SYSTEMS

CPU_SEEDS = (101, 202, 303, 404, 505)
CPU_SCALES = (4, 8, 16, 32, 64, 128)
CPU_MODES = ("planner", "concurrent")
CPU_LAYOUTS = ("mixed", "selective", "pd")
CPU_ACTIVE_MODES = ("per_instance4", "total32")


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _complete(seeds, expected=FORMAL_SEEDS):
    return len(seeds) == len(expected) and set(seeds) == set(expected)


def _mean_interval(values):
    """Mean of seed-level statistics; never pool P99s as request observations."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(estimate=None, ci95_low=None, ci95_high=None)
    result = dict(estimate=float(values.mean()), ci95_low=None, ci95_high=None)
    if len(values) >= 2:
        indices = np.random.default_rng(20260910).integers(0, len(values), (5000, len(values)))
        means = values[indices].mean(axis=1)
        result.update(ci95_low=float(np.quantile(means, .025)), ci95_high=float(np.quantile(means, .975)))
    return result


def _target(identifier, estimate, threshold, direction, complete, *, ci_low=None, ci_high=None, **context):
    known = _finite(estimate)
    satisfied = known and (estimate >= threshold if direction == ">=" else estimate <= threshold)
    status = "pass" if complete and satisfied else "fail" if complete and known else "incomplete"
    ci_support = "unavailable"
    if ci_low is not None and ci_high is not None:
        supported = ci_low >= threshold if direction == ">=" else ci_high <= threshold
        rejected = ci_high < threshold if direction == ">=" else ci_low > threshold
        ci_support = "supports" if supported else "contradicts" if rejected else "overlaps"
    return dict(target=identifier, status=status, estimate=estimate, threshold=threshold,
                direction=direction, five_seed_complete=complete, ci95_low=ci_low, ci95_high=ci_high,
                ci_support=ci_support, criterion="predeclared point estimate; confidence interval reported separately", **context)


def _identity(row):
    required = ("dataset", "model", "host_id", "profile_sha256", "source_config_sha256")
    if any(not row.get(key) for key in required) or not row.get("source_hashes"):
        raise ValueError("GPU comparison requires dataset/model/host/source/profile identity")
    frozen = tuple(sorted(row["source_hashes"].items()))
    return tuple(row[key] for key in required), frozen


def _write_csv(path, rows, default_fields):
    fields = list(dict.fromkeys(default_fields + [key for row in rows for key in row]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _capacity(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("stage") == "capacity" and row.get("formal_eligible"):
            groups[(row["dataset"], row["system"], row["n_gpus"], row["seed"])].append(row)
    capacities = [dict(dataset=k[0], system=k[1], n_gpus=k[2], seed=k[3], **capacity_interval(v))
                  for k, v in sorted(groups.items())]
    efficiencies = []
    systems = sorted({(r["dataset"], r["system"]) for r in capacities})
    for dataset, system in systems:
        for small, large in ((3, 6), (4, 8)):
            left = {r["seed"]: r for r in capacities if r["dataset"] == dataset and r["system"] == system
                    and r["n_gpus"] == small and r["bracket_complete"] and not r["inconsistent"]}
            right = {r["seed"]: r for r in capacities if r["dataset"] == dataset and r["system"] == system
                     and r["n_gpus"] == large and r["bracket_complete"] and not r["inconsistent"]}
            seeds = sorted(left.keys() & right.keys())
            if not seeds:
                continue
            stats = paired_ratio_interval([right[s]["capacity_lower_rps"] for s in seeds],
                                          [left[s]["capacity_lower_rps"] for s in seeds], scale=large / small)
            low = sum(right[s]["capacity_lower_rps"] for s in seeds) / (large / small * sum(left[s]["capacity_upper_rps"] for s in seeds))
            high = sum(right[s]["capacity_upper_rps"] for s in seeds) / (large / small * sum(left[s]["capacity_lower_rps"] for s in seeds))
            efficiencies.append(dict(dataset=dataset, system=system, small_gpus=small, large_gpus=large,
                seeds=seeds, five_seed_complete=_complete(seeds),
                metric="scaling of maximum tested passing arrival rate C_lower",
                bracket_efficiency_low=low, bracket_efficiency_high=high,
                excluded_unpaired_seeds=sorted(left.keys() ^ right.keys()), **stats))
    return capacities, efficiencies


def _weak(rows):
    # Every pair has the same offered sequence, arrival clock offsets, load and SLO.
    groups = defaultdict(dict)
    for row in rows:
        if row.get("stage") == "weak" and row.get("formal_eligible") and row.get("measurement_valid"):
            key = (row["dataset"], row["n_gpus"], row["seed"], row["rate_rps"], row["trace_sha256"],
                   row.get("slo_ttft_s"), row.get("slo_tpot_s"))
            if row["system"] in groups[key]:
                raise ValueError("duplicate weak-load system/seed/trace; select explicit manifests")
            groups[key][row["system"]] = row
    pairs = defaultdict(list)
    for key, systems in groups.items():
        pdb = [name for name in systems if name.startswith("pdblend")]
        if len(pdb) > 1:
            raise ValueError("ambiguous PDBlend weak-load comparison")
        if not pdb:
            continue
        ours = systems[pdb[0]]
        for name, baseline in systems.items():
            if name == pdb[0]:
                continue
            a, b = ours.get("joules_per_good_request"), baseline.get("joules_per_good_request")
            if a is None or b is None or b <= 0:
                continue
            pairs[(key[0], key[1], pdb[0], name, key[3])].append((key[2], a, b, key[4]))
    comparisons = []
    for key, values in sorted(pairs.items()):
        if len({v[0] for v in values}) != len(values):
            raise ValueError("multiple weak traces for one seed require separate report inputs")
        comparisons.append(dict(dataset=key[0], n_gpus=key[1], system=key[2], baseline=key[3],
            rate_rps=key[4], seeds=[v[0] for v in values], five_seed_complete=_complete([v[0] for v in values]),
            trace_sha256=[v[3] for v in values], metric="allocated joules per good request ratio at identical offered load",
            **paired_ratio_interval([v[1] for v in values], [v[2] for v in values])))
    return comparisons


def _weak_scaling(rows):
    """Same-system, paired-seed energy inflation at the same offered rate/GPU."""
    groups = defaultdict(dict)
    for row in rows:
        if not (row.get("stage") == "weak" and row.get("formal_eligible") and row.get("measurement_valid")):
            continue
        q = row.get("q")
        if not _finite(q) or q <= 0 or not math.isclose(row["rate_rps"] / row["n_gpus"], q, rel_tol=1e-10):
            continue
        # Across N, the traces have different lengths; compare the frozen pool,
        # content seed and arrival seed instead of requiring equal trace hashes.
        generation = row.get("trace_generation_identity")
        if not generation:
            continue
        key = (row["dataset"], row["system"], q, generation, row.get("slo_ttft_s"), row.get("slo_tpot_s"))
        cell = (row["n_gpus"], row["seed"])
        if cell in groups[key]:
            raise ValueError("duplicate weak scale/seed at one declared per-GPU load")
        groups[key][cell] = row
    result = []
    for key, cells in sorted(groups.items()):
        for small, large in ((3, 6), (4, 8)):
            seeds = sorted({seed for n, seed in cells if n == small} & {seed for n, seed in cells if n == large})
            seeds = [seed for seed in seeds if _finite(cells[small, seed].get("joules_per_good_request"))
                     and cells[small, seed]["joules_per_good_request"] > 0
                     and _finite(cells[large, seed].get("joules_per_good_request"))]
            if not seeds:
                continue
            stats = paired_ratio_interval([cells[large, s]["joules_per_good_request"] for s in seeds],
                                          [cells[small, s]["joules_per_good_request"] for s in seeds])
            result.append(dict(dataset=key[0], system=key[1], q=key[2], trace_generation_identity=key[3],
                small_gpus=small, large_gpus=large, seeds=seeds, five_seed_complete=_complete(seeds),
                metric="same-system joules/good-request ratio at identical offered rate per GPU",
                energy_inflation=stats["estimate"] - 1.,
                inflation_ci95_low=stats["ci95_low"] - 1. if stats["ci95_low"] is not None else None,
                inflation_ci95_high=stats["ci95_high"] - 1. if stats["ci95_high"] is not None else None, **stats))
    return result


def _cpu_identity(row):
    source = row.get("source_hashes")
    inputs = row.get("provenance", {}).get("input_hashes")
    cpu = row.get("cpu", {})
    if not source or not inputs or cpu.get("physical_core_count") != 2 or cpu.get("planning_workers") != 1:
        raise ValueError("formal CPU comparison requires frozen sources/profiles and fixed two-core/one-worker identity")
    parameters = row.get("parameters", {})
    fixed = {key: parameters.get(key) for key in ("planner_calls", "concurrent_seconds", "arrivals_per_instance",
        "token_updates_per_instance", "telemetry_period", "frequency_period", "role_period", "stale_retries",
        "input_tokens", "output_tokens", "frequencies")}
    common = json.dumps(dict(source=source,
        inputs={k: v.get("sha256") for k, v in inputs.items() if k != "role_costs"},
        cpu=cpu, fixed_parameters=fixed), sort_keys=True)
    # Pure admission planning does not load resident-role transition costs;
    # concurrent replay must keep its additional role evidence consistent.
    role_identity = inputs.get("role_costs", {}).get("sha256") if row.get("mode") == "concurrent" else None
    return common, role_identity


def _cpu_metrics(row):
    planner = row.get("planner", {})
    values = dict(planning_p99_ms=planner.get("latency", {}).get("p99_ms"),
        candidate_p99_ms=planner.get("candidate_latency", {}).get("p99_ms"),
        root_candidate_p99_ms=planner.get("first_candidate_latency", {}).get("p99_ms"),
        worker_wait_p99_ms=planner.get("worker_wait", {}).get("p99_ms"),
        budget_fallback_ratio=planner.get("budget_fallback_ratio"), no_candidate_ratio=planner.get("no_candidate_ratio"))
    # Pure planning never supplies values for concurrent execution behavior.
    concurrent = row.get("mode") == "concurrent"
    values.update(control_p99_ms=row.get("control_latency", {}).get("p99_ms") if concurrent else None,
        successful_commit_throughput_rps=row.get("successful_commit_throughput_rps") if concurrent else None,
        nominal_arrival_rate_rps=row.get("input_arrival_rate_rps") if concurrent else None,
        terminal_commit_throughput_rps=row.get("successful_commit_throughput_including_drain_rps") if concurrent else None,
        stale_retry_ratio=row.get("stale_retry_ratio") if concurrent else None,
        event_loop_lag_p99_ms=row.get("event_loop_lag", {}).get("p99_ms") if concurrent else None,
        backlog_slope_rps=row.get("audited_backlog_stability", {}).get("slope_rps") if concurrent else None)
    counters = row.get("counts", {}) if concurrent else {}
    arrived, terminal = counters.get("arrived"), counters.get("successful_commits")
    window = row.get("arrival_window_s") if concurrent else None
    values["actual_offered_rps"] = arrived / window if _finite(arrived) and _finite(window) and window > 0 else None
    values["terminal_commit_ratio"] = terminal / arrived if _finite(terminal) and _finite(arrived) and arrived > 0 else None
    values["pending_at_window_end"] = counters.get("pending_at_window_end")
    values["drain_timeout"] = counters.get("drain_timeout")
    return {key: value if _finite(value) else None for key, value in values.items()}


def _cpu_aggregate(rows):
    groups = defaultdict(list)
    seen = set()
    identity = None
    mode_identities = {}
    for row in rows:
        if not row.get("formal_evidence"):
            continue
        if (row.get("mode") not in CPU_MODES or row.get("layout") not in CPU_LAYOUTS
                or row.get("active_mode") not in CPU_ACTIVE_MODES
                or not isinstance(row.get("n_instances"), int) or row["n_instances"] <= 0):
            raise ValueError("formal CPU rows require explicit mode/layout/active-mode/instance count")
        current = _cpu_identity(row)
        if identity is not None and current[0] != identity:
            raise ValueError("cannot mix CPU source/profile/hardware/workload configuration")
        if row["mode"] in mode_identities and current != mode_identities[row["mode"]]:
            raise ValueError("cannot mix CPU role-transition evidence within one mode")
        identity = current[0]
        mode_identities[row["mode"]] = current
        key = (row.get("mode"), row.get("layout"), row.get("active_mode"), row.get("n_instances"), row.get("seed"))
        if key in seen:
            raise ValueError("duplicate formal CPU mode/layout/active-mode/N/seed")
        seen.add(key)
        if not row.get("measurement_valid") or not row.get("protocol_sampling_complete"):
            continue
        if row.get("mode") == "planner" and row.get("planner", {}).get("latency", {}).get("count", 0) < 1000:
            continue
        if row.get("mode") == "concurrent" and row.get("arrival_window_s", 0) < 300:
            continue
        groups[key[:4]].append(row)
    result = []
    for (mode, layout, active_mode, n), group in sorted(groups.items()):
        by_metric = defaultdict(list)
        for row in group:
            for metric, value in _cpu_metrics(row).items():
                if value is not None:
                    by_metric[metric].append((row["seed"], value))
        # Long-form output also records absent concurrent metrics as null rows.
        for metric in _cpu_metrics(group[0]):
            values = by_metric[metric]
            seeds = [seed for seed, _ in values]
            result.append(dict(mode=mode, layout=layout, active_mode=active_mode, n_instances=n,
                metric=metric, seeds=seeds, seed_count=len(seeds), five_seed_complete=_complete(seeds, CPU_SEEDS),
                statistic="mean of per-seed metrics; P99 metrics are not pooled request percentiles",
                **_mean_interval([value for _, value in values])))
    return result


def _gpu_metrics(rows, capacities):
    """Seed-level SLO/KV metrics: common weak load, or each seed's passing C_lower."""
    lower = {(r["dataset"], r["system"], r["n_gpus"], r["seed"]): r["capacity_lower_rps"]
             for r in capacities if r["bracket_complete"] and not r["inconsistent"]}
    result = []
    seen = set()
    for row in rows:
        if not row.get("formal_eligible") or not row.get("measurement_valid"):
            continue
        key = (row["dataset"], row["system"], row["n_gpus"], row["seed"])
        if row["stage"] == "capacity":
            if row.get("rate_rps") != lower.get(key) or not row.get("capacity_pass"):
                continue
            stage = "capacity_at_C_lower"
        elif row["stage"] == "weak":
            stage = "weak"
        else:
            continue
        unique = (*key, stage, row.get("q"))
        if unique in seen:
            raise ValueError("duplicate selected GPU seed-level performance observation")
        seen.add(unique)
        offered = row.get("offered_requests")
        transfer = row.get("kv_transfer_bytes")
        transfer_s, transfer_count = row.get("kv_transfer_s"), row.get("kv_transfer_count")
        routes = row.get("route_counts")
        route_total = sum(routes.values()) if routes else 0
        result.append(dict(dataset=key[0], system=key[1], n_gpus=key[2], seed=key[3], stage=stage,
            q=row.get("q"), rate_rps=row.get("rate_rps"), trace_sha256=row.get("trace_sha256"),
            ttft_p99_s=row.get("ttft_s", {}).get("p99"), tpot_p99_s=row.get("tpot_s", {}).get("p99"),
            slo_ttft_s=row.get("slo_ttft_s"), slo_tpot_s=row.get("slo_tpot_s"),
            slo_attainment=row.get("slo_attainment"), goodput_rps=row.get("goodput_rps"),
            token_itl_p99_s=row.get("token_itl_s", {}).get("p99"),
            energy_allocated_j=row.get("energy_allocated_j"), energy_node8_j=row.get("energy_node8_j"),
            joules_per_good_request=row.get("joules_per_good_request"), kv_transfer_bytes=transfer,
            kv_transfer_s=transfer_s, kv_send_s=row.get("kv_send_s"), kv_receive_s=row.get("kv_receive_s"),
            kv_transfer_count=transfer_count,
            kv_observed_span_mean_s=(transfer_s / transfer_count if _finite(transfer_s) and transfer_count else None),
            kv_bytes_per_offered_request=transfer / offered if _finite(transfer) and offered else None,
            pd_route_fraction=routes.get("pd", 0) / route_total if route_total else None))
    return result


def _targets(capacities, efficiencies, weak_scaling, gpu_metrics, cpu_aggregate):
    targets = []
    capacity_keys = {(r["dataset"], r["system"], r["n_gpus"], r["seed"]) for r in capacities
                     if r["bracket_complete"] and not r["inconsistent"]}
    expected_capacity = {(dataset, system, n, seed) for dataset, scales in FORMAL_SCALES.items()
                         for system in SYSTEMS for n in scales for seed in FORMAL_SEEDS}
    missing_capacity = sorted(expected_capacity - capacity_keys)
    targets.append(dict(target="gpu_capacity_matrix_coverage", status="incomplete" if missing_capacity else "pass",
        completed_cells=len(expected_capacity & capacity_keys), expected_cells=len(expected_capacity),
        missing_cells=missing_capacity, criterion="all predeclared per-seed capacity brackets complete within 5%"))
    expected_pairs = [(dataset, system, small, large) for dataset, scales in FORMAL_SCALES.items()
                      for system in SYSTEMS for small, large in ((3, 6), (4, 8)) if small in scales and large in scales]
    for kind, values, threshold, direction in (("capacity_efficiency", efficiencies, .8, ">="),
                                               ("weak_energy_inflation", weak_scaling, .10, "<=")):
        existing = {(r["dataset"], r["system"], r["small_gpus"], r["large_gpus"]) for r in values}
        for row in values:
            energy = kind == "weak_energy_inflation"
            targets.append(_target(kind, row["energy_inflation"] if energy else row["estimate"], threshold,
                direction, row["five_seed_complete"],
                ci_low=row.get("inflation_ci95_low") if energy else row.get("ci95_low"),
                ci_high=row.get("inflation_ci95_high") if energy else row.get("ci95_high"),
                dataset=row["dataset"], system=row["system"], small_gpus=row["small_gpus"], large_gpus=row["large_gpus"],
                seeds=row["seeds"], q=row.get("q")))
        for dataset, system, small, large in expected_pairs:
            if (dataset, system, small, large) not in existing:
                targets.append(_target(kind, None, threshold, direction, False,
                    dataset=dataset, system=system, small_gpus=small, large_gpus=large, seeds=[]))
    weak_groups = defaultdict(list)
    for row in gpu_metrics:
        if row["stage"] == "weak" and _finite(row.get("slo_attainment")):
            weak_groups[row["dataset"], row["system"], row["n_gpus"], row.get("q")].append(row)
    for (dataset, system, n, q), rows in weak_groups.items():
        statistics = _mean_interval([r["slo_attainment"] for r in rows])
        seeds = [r["seed"] for r in rows]
        targets.append(_target("weak_joint_slo_attainment", statistics["estimate"], .90, ">=", _complete(seeds),
            ci_low=statistics["ci95_low"], ci_high=statistics["ci95_high"], dataset=dataset, system=system,
            n_gpus=n, q=q, seeds=seeds))
    weak_keys = {(r[0], r[1], r[2]) for r in weak_groups}
    for dataset, scales in FORMAL_SCALES.items():
        for system in SYSTEMS:
            for n in scales:
                if (dataset, system, n) not in weak_keys:
                    targets.append(_target("weak_joint_slo_attainment", None, .90, ">=", False,
                                           dataset=dataset, system=system, n_gpus=n, seeds=[]))
    by_cpu_cell = defaultdict(dict)
    for row in cpu_aggregate:
        by_cpu_cell[row["mode"], row["layout"], row["active_mode"], row["n_instances"]][row["metric"]] = row
    completed_cpu_cells = 0
    for mode in CPU_MODES:
        for layout in CPU_LAYOUTS:
            for active_mode in CPU_ACTIVE_MODES:
                completed_scales = []
                for n in CPU_SCALES:
                    metrics = by_cpu_cell[mode, layout, active_mode, n]
                    metric = "planning_p99_ms" if mode == "planner" else "control_p99_ms"
                    latency = metrics.get(metric, {})
                    if latency.get("five_seed_complete"):
                        completed_scales.append(n)
                    targets.append(_target("cpu_planning_p99" if mode == "planner" else "cpu_control_p99",
                        latency.get("estimate"), 50., "<=", latency.get("five_seed_complete", False),
                        ci_low=latency.get("ci95_low"), ci_high=latency.get("ci95_high"), mode=mode,
                        layout=layout, active_mode=active_mode, n_instances=n, seeds=latency.get("seeds", [])))
                    if mode == "concurrent":
                        parts = []
                        for metric, name, threshold, direction in (
                                ("terminal_commit_ratio", "cpu_commit_meets_actual_arrivals", 1., ">="),
                                ("backlog_slope_rps", "cpu_backlog_no_growth", 0., "<=")):
                            row = metrics.get(metric, {})
                            target = _target(name, row.get("estimate"), threshold, direction,
                                row.get("five_seed_complete", False), ci_low=row.get("ci95_low"), ci_high=row.get("ci95_high"),
                                mode=mode, layout=layout, active_mode=active_mode, n_instances=n, seeds=row.get("seeds", []))
                            targets.append(target)
                            parts.append(target["status"])
                        parts.append(targets[-3]["status"])
                        status = "incomplete" if "incomplete" in parts else "pass" if all(s == "pass" for s in parts) else "fail"
                        targets.append(dict(target="cpu_full_control", status=status, mode=mode, layout=layout,
                            active_mode=active_mode, n_instances=n,
                            criterion="all: P99 <= 50 ms; terminal successful commits / actual offered arrivals >= 1; second-half backlog trend <= 0"))
                completed_cpu_cells += len(completed_scales) * len(CPU_SEEDS)
                targets.append(dict(target="cpu_submatrix_coverage", status="pass" if tuple(completed_scales) == CPU_SCALES else "incomplete",
                    mode=mode, layout=layout, active_mode=active_mode, completed_scales=completed_scales,
                    completed_cells=len(completed_scales) * len(CPU_SEEDS), expected_cells=len(CPU_SCALES) * len(CPU_SEEDS),
                    criterion="six N values, all five prescribed seeds; planner >= 1000 calls or concurrent >= 300 seconds"))
    expected_cpu = len(CPU_MODES) * len(CPU_LAYOUTS) * len(CPU_ACTIVE_MODES) * len(CPU_SCALES) * len(CPU_SEEDS)
    targets.append(dict(target="cpu_full_matrix_coverage", status="pass" if completed_cpu_cells == expected_cpu else "incomplete",
                        completed_cells=completed_cpu_cells, expected_cells=expected_cpu))
    return targets


def _plot(out, name, rows, draw):
    if not rows:
        # Rebuilding a report with fewer inputs must not retain an earlier
        # positive curve for an experiment that is now absent.
        for suffix in ("svg", "pdf"):
            (out / f"{name}.{suffix}").unlink(missing_ok=True)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    draw(ax, rows)
    paths = []
    for suffix in ("svg", "pdf"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _capacity_plot(ax, rows):
    groups = defaultdict(list)
    for row in rows:
        if row["capacity_lower_rps"] is not None and not row["inconsistent"]:
            groups[(row["dataset"], row["system"])].append(row)
    for (dataset, system), values in groups.items():
        artist = ax.scatter([v["n_gpus"] for v in values], [v["capacity_lower_rps"] for v in values],
                            label=f"{dataset}/{system}", alpha=.7)
        color = artist.get_facecolor()[0]
        for small, large in ((3, 6), (4, 8)):
            left = {v["seed"]: v for v in values if v["n_gpus"] == small}
            right = {v["seed"]: v for v in values if v["n_gpus"] == large}
            seeds = left.keys() & right.keys()
            if seeds:
                base = sum(left[s]["capacity_lower_rps"] for s in seeds) / len(seeds)
                ax.plot([small, large], [base, base * large / small], linestyle="--", linewidth=1,
                        color=color, label=f"{dataset}/{system} ideal {small}→{large}")
    ax.set(xlabel="Allocated GPUs", ylabel="Maximum tested passing arrival rate (requests/s)")
    if groups:
        ax.legend(fontsize=8)
    ax.grid(alpha=.2)


def _ratio_plot(ax, rows, kind):
    labels = [f"{r['dataset']}/{r['system']}\n{r['small_gpus']}→{r['large_gpus']}" if kind in ("capacity", "inflation")
              else f"{r['dataset']}/{r['baseline']}\n{r['n_gpus']} GPUs" for r in rows]
    for index, row in enumerate(rows):
        low, high = row.get("ci95_low"), row.get("ci95_high")
        error = ([[max(0., row["estimate"] - low)], [max(0., high - row["estimate"])]]) if low is not None else None
        ax.errorbar([index], [row["estimate"]], yerr=error, fmt="o", capsize=3)
    ax.axhline(1., color="gray", linestyle="--", linewidth=1)
    if kind in ("capacity", "inflation"):
        ax.axhline(.8 if kind == "capacity" else 1.1, color="red", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Capacity scaling efficiency" if kind == "capacity" else
                  "Joules/good-request at doubled N / original N" if kind == "inflation" else
                  "PDBlend / baseline joules per good request")
    ax.grid(axis="y", alpha=.2)


def _control_plot(ax, rows, metric="planning_p99_ms", ylabel="Mean per-seed planning P99 (ms)"):
    groups = defaultdict(list)
    for row in rows:
        if row["metric"] == metric and row["estimate"] is not None:
            groups[row["mode"], row["layout"], row["active_mode"]].append(row)
    for label, values in groups.items():
        values.sort(key=lambda row: row["n_instances"])
        x, y = [v["n_instances"] for v in values], [v["estimate"] for v in values]
        ax.plot(x, y, "o-", label="/".join(label))
        for row in values:
            if row["ci95_low"] is not None:
                ax.vlines(row["n_instances"], row["ci95_low"], row["ci95_high"], linewidth=1)
    if metric in ("planning_p99_ms", "control_p99_ms"):
        ax.axhline(50., linestyle=":", color="red", linewidth=1)
    ax.set(xlabel="Virtual controller instances", ylabel=ylabel)
    if groups:
        ax.legend()
    ax.grid(alpha=.2)


def _gpu_metric_plot(ax, rows, metric, ylabel):
    grouped = defaultdict(list)
    for row in rows:
        if _finite(row.get(metric)):
            grouped[row["dataset"], row["system"], row["stage"], row.get("q")].append(row)
    for key, values in grouped.items():
        per_n = defaultdict(list)
        for row in values:
            per_n[row["n_gpus"]].append(row[metric])
        points = [(n, _mean_interval(v)) for n, v in sorted(per_n.items())]
        ax.plot([n for n, _ in points], [v["estimate"] for _, v in points], "o-",
                label=f"{key[0]}/{key[1]}/{key[2]}" + (f" q={key[3]:.3g}" if key[3] is not None else ""))
        for n, stats in points:
            if stats["ci95_low"] is not None:
                ax.vlines(n, stats["ci95_low"], stats["ci95_high"], linewidth=1)
    if metric == "slo_attainment":
        ax.axhline(.90, color="red", linestyle=":", linewidth=1)
    ax.set(xlabel="Allocated GPUs", ylabel=ylabel)
    if grouped:
        ax.legend(fontsize=7)
    ax.grid(alpha=.2)


def _read_cpu_summary(summary_path, evidence_hashes):
    value = json.loads(summary_path.read_text())
    evidence_hashes[str(summary_path)] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    if value.get("scope") != "control_plane_replay":
        raise ValueError("CPU index must reference control-plane replay summaries")
    if value.get("mode") == "concurrent":
        samples_path = summary_path.parent / "samples.jsonl"
        if samples_path.is_file():
            # Reduce 100-Hz observations to independent time bins before the
            # trend/HAC calculation; retain the raw file hash and provenance.
            bins = defaultdict(list)
            with samples_path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("kind") == "backlog" and _finite(row.get("at_s")) and _finite(row.get("pending")):
                        bins[int(row["at_s"])].append((row["at_s"], row["pending"]))
            backlog = [dict(t_s=sum(x[0] for x in pairs) / len(pairs), pending=sum(x[1] for x in pairs) / len(pairs))
                       for _, pairs in sorted(bins.items())]
            end = value.get("arrival_window_s")
            if _finite(end) and end >= 300:
                result = backlog_stability(backlog, end, window_s=end / 2)
                if result.get("valid") and abs(result["slope_rps"]) < 1e-12:
                    result["slope_rps"] = 0.
                value["audited_backlog_stability"] = result
            evidence_hashes[str(samples_path)] = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    return value


def build_report(listpaths, out):
    """Audit explicitly selected manifest files, write four tables and supported figures.

    GPU datasets may differ, but each dataset must retain one host/model/source/
    profile identity. A mixture raises an error instead of silently averaging.
    CPU inputs must declare ``scope=control_plane`` and reference their own rows.
    """
    if not isinstance(listpaths, (list, tuple)):
        raise ValueError("explicit list of manifest paths required")
    paths = [Path(path).resolve() for path in listpaths]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate report input")
    out = Path(out).resolve()
    gpu, cpu, identities, evidence_hashes = [], [], {}, {}
    for path in paths:
        if path.is_dir():
            raise ValueError("report inputs must be explicit manifest paths")
        manifest = json.loads(path.read_text())
        scope = manifest.get("scope", "gpu_serving")
        if scope == "gpu_serving":
            row = audit_run(path)
            trace_path = path.parent / "trace.json"
            if trace_path.is_file():
                trace = json.loads(trace_path.read_text())
                if trace.get("pool_sha256") and trace.get("content_seed") is not None:
                    row["trace_generation_identity"] = json.dumps(dict(pool_sha256=trace["pool_sha256"],
                        content_seed=trace["content_seed"], arrival_process=trace.get("arrival_process"),
                        content_sampling=trace.get("content_sampling")), sort_keys=True)
                if trace.get("arrival_seed", row.get("seed")) != row.get("seed"):
                    raise ValueError("trace arrival seed differs from audited manifest")
            # Invalid inputs still appear in audited output but cannot create curves.
            if row.get("formal_eligible"):
                identity = _identity(row)
                dataset = row["dataset"]
                if dataset in identities and identities[dataset] != identity:
                    raise ValueError("cannot mix host/model/source/config/profile within dataset " + dataset)
                identities[dataset] = identity
            gpu.append(row)
        elif scope in ("control_plane", "control_plane_replay"):
            values = manifest.get("rows")
            if isinstance(manifest.get("cells"), list):
                values = []
                for cell in manifest["cells"]:
                    summary_path = Path(cell["summary"])
                    if not summary_path.is_absolute():
                        summary_path = path.parent / summary_path
                    value = _read_cpu_summary(summary_path, evidence_hashes)
                    values.append(value)
            if values is None and manifest.get("results_path"):
                result_path = (path.parent / manifest["results_path"]).resolve()
                values = json.loads(result_path.read_text())
                values = values.get("rows", values) if isinstance(values, dict) else values
            if not isinstance(values, list):
                raise ValueError("control-plane manifest requires rows or results_path")
            for value in values:
                normalized = dict(value, scope="control_plane_replay", input_manifest=str(path))
                milliseconds = value.get("planner", {}).get("latency", {}).get("p99_ms")
                if milliseconds is not None:
                    normalized["planning_p99_s"] = milliseconds / 1000.
                cpu.append(normalized)
        else:
            raise ValueError("unsupported report evidence scope: " + str(scope))
    capacities, efficiencies = _capacity(gpu)
    weak = _weak(gpu)
    energy_scaling = _weak_scaling(gpu)
    gpu_metrics = _gpu_metrics(gpu, capacities)
    cpu_aggregate = _cpu_aggregate(cpu)
    targets = _targets(capacities, efficiencies, energy_scaling, gpu_metrics, cpu_aggregate)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audited_runs.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2, allow_nan=False))
    _write_csv(out / "capacity.csv", capacities, ["dataset", "system", "n_gpus", "seed", "capacity_lower_rps", "capacity_upper_rps"])
    _write_csv(out / "scaling_efficiency.csv", efficiencies, ["dataset", "system", "small_gpus", "large_gpus", "estimate", "ci95_low", "ci95_high"])
    _write_csv(out / "weak_energy.csv", weak, ["dataset", "system", "baseline", "n_gpus", "estimate", "ci95_low", "ci95_high"])
    _write_csv(out / "control_plane.csv", cpu, ["scope", "n_instances", "layout", "planning_p99_s"])
    _write_csv(out / "weak_energy_scaling.csv", energy_scaling,
               ["dataset", "system", "small_gpus", "large_gpus", "q", "energy_inflation", "inflation_ci95_low", "inflation_ci95_high"])
    _write_csv(out / "gpu_performance.csv", gpu_metrics,
               ["dataset", "system", "n_gpus", "seed", "stage", "ttft_p99_s", "tpot_p99_s", "slo_attainment", "kv_observed_span_mean_s"])
    _write_csv(out / "control_plane_aggregate.csv", cpu_aggregate,
               ["mode", "layout", "active_mode", "n_instances", "metric", "seed_count", "estimate", "ci95_low", "ci95_high"])
    _write_csv(out / "targets.csv", targets, ["target", "status", "estimate", "threshold", "five_seed_complete"])
    (out / "targets.json").write_text(json.dumps(targets, ensure_ascii=False, indent=2, allow_nan=False))
    plots = []
    drawable_capacity = [r for r in capacities if r["capacity_lower_rps"] is not None and not r["inconsistent"]]
    plots += _plot(out, "capacity", drawable_capacity, _capacity_plot)
    plots += _plot(out, "scaling_efficiency", efficiencies, lambda ax, rows: _ratio_plot(ax, rows, "capacity"))
    plots += _plot(out, "weak_energy", weak, lambda ax, rows: _ratio_plot(ax, rows, "energy"))
    plots += _plot(out, "weak_energy_scaling", energy_scaling, lambda ax, rows: _ratio_plot(ax, rows, "inflation"))
    for name, metric, label in (("gpu_ttft", "ttft_p99_s", "Mean per-seed TTFT P99 (s)"),
            ("gpu_tpot", "tpot_p99_s", "Mean per-seed mean-TPOT P99 (s)"),
            ("gpu_attainment", "slo_attainment", "Mean per-seed joint SLO attainment"),
            ("gpu_goodput", "goodput_rps", "Good requests / full arrival-plus-drain second"),
            ("gpu_kv_span", "kv_observed_span_mean_s", "Engine-observed KV span per transfer (s); includes wait/import"),
            ("gpu_kv_bytes", "kv_bytes_per_offered_request", "Measured KV bytes / offered request")):
        usable = [r for r in gpu_metrics if _finite(r.get(metric))]
        plots += _plot(out, name, usable, lambda ax, rows, metric=metric, label=label: _gpu_metric_plot(ax, rows, metric, label))
    for name, metric, label in (("control_plane", "planning_p99_ms", "Mean per-seed planning P99 (ms)"),
            ("control_commit_latency", "control_p99_ms", "Mean per-seed full-control P99 (ms)"),
            ("control_commit_throughput", "successful_commit_throughput_rps", "Successful commits / arrival-window second"),
            ("control_budget_fallback", "budget_fallback_ratio", "Mean per-seed feasible greedy fallback fraction"),
            ("control_stale_retries", "stale_retry_ratio", "Mean per-seed stale retry fraction"),
            ("control_no_candidate", "no_candidate_ratio", "Mean per-seed no-candidate fraction")):
        usable = [r for r in cpu_aggregate if r["metric"] == metric and r["estimate"] is not None]
        plots += _plot(out, name, usable, lambda ax, rows, metric=metric, label=label: _control_plot(ax, rows, metric, label))
    valid = sum(r.get("measurement_valid", False) and r.get("formal_eligible", False) for r in gpu)
    missing = []
    if not capacities:
        missing.append("尚无正式容量测量；不能给出 GPU 可扩展性结论。")
    if capacities and not all(r["bracket_complete"] for r in capacities):
        missing.append("部分容量上下界未闭合到 5%，这些 seed 不参与扩展效率。")
    if any(not r["five_seed_complete"] for r in efficiencies + weak):
        missing.append("部分对比不足 5 个配对 seed，不能标为完成正式验收。")
    if not weak:
        missing.append("尚无同到达率、同请求序列的正式能效配对。")
    if not energy_scaling:
        missing.append("尚无同系统、同 seed、同每卡到达率 q 的跨规模能耗增幅配对。")
    cpu_subsets = [t for t in targets if t["target"] == "cpu_submatrix_coverage"]
    complete_subsets = [t for t in cpu_subsets if t["status"] == "pass"]
    cpu_full = next(t for t in targets if t["target"] == "cpu_full_matrix_coverage")
    if cpu_full["status"] != "pass":
        missing.append(f"CPU 完整矩阵尚未完成：完成 {len(complete_subsets)}/12 个正式子矩阵；纯 planner 结果不能验收并发控制面。")
    failed = [t for t in targets if t["status"] == "fail"]
    if failed:
        missing.append(f"已有 {len(failed)} 个完整测量的预注册目标未通过；详见 targets.csv，不能据此宣称全部可扩展。")
    text = ["# PDBlend 可扩展性实验报告", "",
            f"显式输入 {len(paths)} 个文件；GPU 原始测量 {len(gpu)} 组，其中有效且正式 {valid} 组；控制面记录 {len(cpu)} 条。", "",
            "容量 C 为联合 TTFT/平均 TPOT 达标率至少 90%、且到达窗口末 300 秒队列增长率单侧 95% 上界不超过 1% 到达率的最高已测通过率。每个 seed 保留通过下界和失败上界，不将试探点或工程失败作为有效界限。", "",
            "队列使用 OLS 趋势及 30 秒滞后的 Newey–West/Bartlett 稳健协方差；采样缺失时不通过。吞吐 G 的时间分母包含完整到达窗口及实际 drain，拒绝、超时和未完成请求均留在达标率分母。", "",
            "扩展效率仅比较 3→6 与 4→8，同 seed 成对重采样 5000 次。点估计使用最高已测通过容量下界，bootstrap 区间只描述 seed 差异；容量二分上下界导致的额外区间另列，不冒充精确容量。", "",
            "能效使用同规模、同到达率、同 seed、同完整 trace 的策略配对。E_alloc 按实际分配 GPU 独立积分，包含闲置及失败工作的能耗；E_node8 对全部 8 卡积分，不按卡数比例折算。", "",
            "跨规模能效单独比较 3→6、4→8：同系统、同 seed、同每卡到达率 q，并核对同一请求池与内容生成配置。单位达标请求能耗增幅目标为 ≤10%；弱扩展联合达标率目标为 ≥90%，容量扩展效率目标为 ≥0.8。容量图的虚线仅为从已测小规模点出发的理想比例参考。", "",
            "CPU 按 mode/layout/active_mode 分组，先计算每个 seed 的统计量，再聚合 seed 和 bootstrap 区间；每组每个 N 只有一个聚合点，不把多个 seed 的 P99 连成规模曲线，也不把这些值称为合并请求后的 P99。规划 P99 与完整控制 P99 目标均为 ≤50 ms；纯 planner 只验收前者。并发完整控制还须终态成功提交数覆盖实际到达数、后半段队列无持续增长。名义 Poisson 到达率、实际到达率、窗口内提交吞吐与包含有界 drain 的终态完成比例分列；不要求最后到达的请求都在窗口边界前提交，也不把有限样本的实际到达率必须超过名义 λ 作为验收条件。缺少实际 arrived/successful_commits 计数则此目标 incomplete。回退率与过期重试率仅披露，无预注册合格阈值。", "",
            "targets 使用预注册点估计判定 pass/fail；不足规定五个 seed 或缺测则 incomplete。95% CI 与 ci_support 单独报告，不将事后添加的 CI 下界条件替代预注册判据。CPU 队列趋势从 samples.jsonl 的一秒平均值重算后半段 OLS/HAC，不凭两个端点推断持续增长。", "",
            "KV 时长若可得，是引擎观测的 send/receive 覆盖跨度，包含等待和导入；不能解释为纯网络时延。缺少字节或时长测量时对应图不生成，数据保留空值。", "",
            "CPU 图仅描述虚拟实例上的实际控制面计算/回放，不能证明对应数量的真实 GPU、跨机 KV 传输或端到端 SLO。未测项目留空，不生成正向占位曲线。", "", "## 证据状态", ""]
    text.extend("- " + warning for warning in missing)
    for subset in complete_subsets:
        text.append(f"- 已完成 CPU 子矩阵：{subset['mode']}/{subset['layout']}/{subset['active_mode']}，6 个规模 × 5 个 seed；这表示采样完整性，不等于性能目标通过。")
    for target in failed:
        label = "/".join(str(target[k]) for k in ("mode", "layout", "active_mode", "n_instances", "dataset", "system", "small_gpus", "large_gpus") if k in target)
        text.append(f"- 未通过：{target['target']} {label}，点估计 {target.get('estimate')}，阈值 {target.get('direction', '')} {target.get('threshold')}。")
    if not missing:
        text.append("已有所列测量及配对统计；结论仅覆盖原始清单中的模型、主机、profile 和规模。")
    text.extend(["", "## 输出", "", "- capacity.csv：每个 seed 的容量区间及失败/缺失状态。",
                 "- scaling_efficiency.csv：成对扩展效率及置信区间。", "- weak_energy.csv：同负载请求能耗比。",
                 "- control_plane.csv：独立控制面证据。", "- audited_runs.json：逐请求复核、所有无效原因及功率来源。", ""])
    text.extend(["- weak_energy_scaling.csv：同每卡 q 的成对能耗增幅及 CI。",
                 "- gpu_performance.csv：弱扩展或各 seed 容量下界处的 TTFT/TPOT/达标率/实测 KV 指标。",
                 "- control_plane_aggregate.csv：按 mode/layout/active_mode/N 聚合的 seed 均值与 CI。",
                 "- targets.csv / targets.json：预注册目标状态、五 seed 完整性与未测矩阵。", ""])
    (out / "REPORT.md").write_text("\n".join(text))
    provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = dict(report_path=str(out / "REPORT.md"), input_manifest_hashes=provenance,
                  gpu_runs=len(gpu), formal_valid_runs=valid, capacity_rows=len(capacities),
                  efficiency_rows=len(efficiencies), weak_energy_rows=len(weak), control_plane_rows=len(cpu),
                  weak_energy_scaling_rows=len(energy_scaling), gpu_performance_rows=len(gpu_metrics),
                  control_plane_aggregate_rows=len(cpu_aggregate), referenced_evidence_hashes=evidence_hashes,
                  target_status_counts={status: sum(t["status"] == status for t in targets) for status in ("pass", "fail", "incomplete")},
                  cpu_complete_submatrices=[dict(mode=t["mode"], layout=t["layout"], active_mode=t["active_mode"]) for t in complete_subsets],
                  cpu_full_matrix_complete=cpu_full["status"] == "pass", plots=plots, warnings=missing)
    (out / "report_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="*", type=Path, default=[],
                        help="explicit GPU manifest files and/or CPU index.json files")
    parser.add_argument("--manifest-list", type=Path, help="JSON array of explicit manifest paths; may combine with --manifests")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    paths = list(args.manifests)
    if args.manifest_list:
        listed = json.loads(args.manifest_list.read_text())
        if not isinstance(listed, list) or any(not isinstance(path, str) for path in listed):
            parser.error("--manifest-list must contain a JSON string array")
        paths.extend(Path(path) if Path(path).is_absolute() else args.manifest_list.parent / path for path in listed)
    print(json.dumps(build_report(paths, args.out), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
