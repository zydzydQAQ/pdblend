#!/usr/bin/env python3
"""Plot an already selected, frozen comparison snapshot without selecting runs.

Usage: /home/pdblend/.venv/bin/python scripts/plot_cohort_energy.py --input-dir DIR
DIR must contain points.json (an array) and snapshot.json. The script writes only
figures into DIR; it never modifies the selection, metrics, or source receipts.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


MODELS = ("7B", "14B", "32B")
DATASETS = ("alpaca", "sharegpt", "longbench")
DATASET_NAMES = dict(zip(DATASETS, ("Alpaca", "ShareGPT", "LongBench")))
SERIES = ("mixed", "ecoserve", "distserve", "dynamollm", "pd_previous", "pd_current")
NAMES = {
    "mixed": "Mixed", "ecoserve": "EcoServe", "distserve": "DistServe",
    "dynamollm": "DynamoLLM", "pd_previous": "PDblend: previous round",
    "pd_current": "PDblend: current round",
}
COLORS = {
    "mixed": "#536B87", "ecoserve": "#B97906", "distserve": "#C35C88",
    "dynamollm": "#8172B2", "pd_previous": "#48A59A", "pd_current": "#006A62",
}
MARKERS = dict(zip(SERIES, ("s", "^", "D", "v", "o", "*")))
METRICS = (
    ("total_energy_kj", "energy-vs-rate", "GPU energy vs rate | service + tail", "GPU energy (kJ)"),
    ("slo_attainment_pct", "slo-attainment-vs-rate", "SLO attainment vs rate", "SLO attainment (%)"),
    ("success_pct", "success-rate-vs-rate", "Request success rate vs rate", "Successful requests (%)"),
)


def numeric(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number or null: {value!r}")
    return float(value)


def load_snapshot(directory):
    points = json.loads((directory / "points.json").read_text())
    snapshot = json.loads((directory / "snapshot.json").read_text())
    if not isinstance(points, list) or not points:
        raise ValueError("points.json must be a nonempty array")
    seen, rates, revisions = set(), {}, defaultdict(set)
    for row in points:
        series, model, dataset = row["series"], row["model"], row["dataset"]
        if series not in SERIES or model not in MODELS or dataset not in DATASETS:
            raise ValueError(f"Unknown plot identity: {series}, {model}, {dataset}")
        scale = numeric(row["rate_scale"], "rate_scale")
        rate = numeric(row["offered_rps"], "offered_rps")
        if scale is None or rate is None or scale <= 0 or rate <= 0:
            raise ValueError("rate_scale and offered_rps must be positive")
        key = (model, dataset, round(scale, 12))
        identity = (series, *key)
        if identity in seen:
            raise ValueError(f"Duplicate selection; deduplicate upstream: {identity}")
        seen.add(identity)
        if key in rates and not math.isclose(rate, rates[key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Different offered rates for the same panel/scale: {key}")
        rates[key] = rate
        for field in ("all_requests_successful", "slo_pass", "energy_rank_eligible"):
            if not isinstance(row[field], bool):
                raise ValueError(f"{field} must be a JSON boolean")
        for field in ("total_energy_kj", "service_energy_kj", "tail_energy_kj",
                      "slo_attainment_pct", "success_pct"):
            value = numeric(row[field], field)
            if value is not None and (value < 0 or field.endswith("_pct") and value > 100):
                raise ValueError(f"Invalid {field}: {value}")
        total, service, tail = (row[f] for f in ("total_energy_kj", "service_energy_kj", "tail_energy_kj"))
        if total is not None and (service is None or tail is None
                                 or not math.isclose(total, service + tail, rel_tol=1e-7, abs_tol=1e-5)):
            raise ValueError(f"Total energy must equal complete service + tail: {identity}")
        if row["slo_pass"] and not row["all_requests_successful"]:
            raise ValueError(f"Hard SLO pass cannot include unsuccessful requests: {identity}")
        if series.startswith("pd_"):
            revisions[series].add(row["revision"])
    if any(len(values) != 1 for values in revisions.values()):
        raise ValueError("Each PDblend series must contain exactly one revision")
    return points, snapshot


def panel_rows(points, model, dataset):
    """Use the union of measured rates so absent intermediate points stay gaps."""
    rows = [r for r in points if r["model"] == model and r["dataset"] == dataset]
    grid = sorted({(round(float(r["rate_scale"]), 12), float(r["offered_rps"])) for r in rows})
    index = {(r["series"], round(float(r["rate_scale"]), 12)): r for r in rows}
    return grid, index


def plot_metric(points, snapshot, metric, title, ylabel):
    energy = metric == "total_energy_kj"
    present = [s for s in SERIES if any(r["series"] == s for r in points)]
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 11.3), sharey="row")
    fig.subplots_adjust(left=.075, right=.985, bottom=.155 if energy else .115,
                        top=.825, hspace=.43, wspace=.14)
    fig.suptitle(title, x=.075, y=.984, ha="left", fontsize=22, fontweight="bold")
    stamp = str(snapshot.get("source_modified_at") or snapshot.get("captured_at") or "unspecified")
    current = sum(r["series"] == "pd_current" for r in points)
    fig.text(.075, .947, f"Qwen2.5 | source snapshot: {stamp} | current PDblend: {current} observed points",
             fontsize=10.5, color="#536777")
    handles = [Line2D([], [], color=COLORS[s], marker=MARKERS[s],
                      linestyle="--" if s == "pd_previous" else "-",
                      linewidth=2.5 if s == "pd_current" else 1.8,
                      markersize=11 if s == "pd_current" else 6, label=NAMES[s]) for s in present]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .922),
               ncol=3, frameon=False, fontsize=10.7, handlelength=2.6, columnspacing=2.8)

    for i, model in enumerate(MODELS):
        energies = [r[metric] for r in points if r["model"] == model and r[metric] is not None]
        upper = (max(energies, default=100) * 1.18) if energy else 106
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            grid, index = panel_rows(points, model, dataset)
            xs = [rate for _, rate in grid]
            missing = []
            for series in present:
                rows = [index.get((series, scale)) for scale, _ in grid]
                ys = [r[metric] if r is not None and r[metric] is not None else math.nan for r in rows]
                current_series = series == "pd_current"
                ax.plot(xs, ys, color=COLORS[series], marker=MARKERS[series],
                        markersize=12 if current_series else 5.2,
                        markeredgecolor="white" if current_series else COLORS[series],
                        markeredgewidth=.8 if current_series else 1,
                        linestyle="--" if series == "pd_previous" else "-",
                        linewidth=2.4 if current_series else 1.7,
                        zorder=10 if current_series else 4)
                if energy:
                    missing_n = sum(r is not None and r[metric] is None for r in rows)
                    if missing_n:
                        missing.append(f"{NAMES[series].replace('PDblend: ', 'PD ')} {missing_n}")
                    for x, y, row in zip(xs, ys, rows):
                        if row is None or not math.isfinite(y):
                            continue
                        if not row["all_requests_successful"]:
                            ax.scatter([x], [y], marker="x", s=64, linewidths=1.7,
                                       color="#D12D35", zorder=20)
                        elif not row["slo_pass"]:
                            ax.scatter([x], [y], marker="o", s=76, linewidths=1.3,
                                       facecolors="none", edgecolors="#1F2933", zorder=19)
            ax.set_title(f"{model} / {DATASET_NAMES[dataset]}", loc="left", fontsize=12,
                         fontweight="bold", pad=9)
            ax.set_ylim(0, upper)
            if xs:
                padding = (max(xs) - min(xs)) * .07 or max(xs) * .1
                ax.set_xlim(min(xs) - padding, max(xs) + padding)
                ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(labelsize=9.5)
            ax.set_xlabel("Rate (requests/s)", fontsize=10, labelpad=5)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=10.5)
            ax.grid(axis="y", color="#E6EBEE", linewidth=.8)
            ax.set_axisbelow(True)
            if not energy:
                ax.yaxis.set_major_locator(MultipleLocator(20))
            if metric == "slo_attainment_pct":
                ax.axhline(90, color="#7B8791", linestyle=(0, (3, 3)), linewidth=1, zorder=1)
            if missing:
                ax.text(.015, .97, "NA: " + "; ".join(missing), transform=ax.transAxes,
                        va="top", fontsize=8.4, color="#8A4545",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=2))
            if not xs:
                ax.text(.5, .5, "No observations", transform=ax.transAxes, ha="center", color="#82909A")

    if energy:
        quality_handles = [
            Line2D([], [], color="#D12D35", marker="x", linestyle="none", markersize=8,
                   markeredgewidth=1.7, label="Failed / unfinished requests: not a feasible energy candidate"),
            Line2D([], [], color="#1F2933", marker="o", linestyle="none", markersize=8,
                   markerfacecolor="none", label="All requests successful, hard SLO not met"),
        ]
        fig.legend(handles=quality_handles, loc="lower left", bbox_to_anchor=(.067, .073),
                   ncol=1, frameon=False, fontsize=9.5, handlelength=1.4, labelspacing=.55)
        note = "Eight-GPU energy: service + tail, including in-window wakes. Excludes pre-service setup, between-window reset and host. NA = missing."
    elif metric == "slo_attainment_pct":
        note = "Joint attainment = successful requests meeting TTFT and TPOT limits / all offered requests. Dashed guide: 90%; hard SLO also requires both P99s."
    else:
        note = "Success rate = successfully completed requests / all offered requests. All requests must succeed for hard SLO feasibility."
    fig.text(.075, .046, note, fontsize=9.1, color="#536777")
    fig.text(.075, .025, "Previous and current PDblend revisions are separate. Missing rate points break lines. Single observations; formal qualification remains incomplete.",
             fontsize=9.1, color="#536777")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    points, snapshot = load_snapshot(directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#ADB8C0", "axes.labelcolor": "#23313C",
                         "text.color": "#23313C", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "svg.fonttype": "none"})
    with PdfPages(directory / "comparison.pdf") as pdf:
        pdf.infodict().update(Title="Energy, SLO attainment and request success vs rate",
                              Subject="Frozen single-observation comparison; service plus tail energy")
        for metric, filename, title, ylabel in METRICS:
            fig = plot_metric(points, snapshot, metric, title, ylabel)
            fig.savefig(directory / f"{filename}.png", dpi=200)
            fig.savefig(directory / f"{filename}.svg")
            pdf.savefig(fig)
            plt.close(fig)
            print(directory / f"{filename}.png")
    print(directory / "comparison.pdf")


if __name__ == "__main__":
    main()
