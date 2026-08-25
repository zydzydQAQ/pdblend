"""Generate reproducible PNG figures for the selective-PD pilot report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = [
    "colocated",
    "PD(A100-P,V100-D)",
    "PD(V100-P,A100-D)",
]
ARCH_LABELS = {
    "colocated": "colocated",
    "PD(A100-P,V100-D)": "PD(A100-P,V100-D)",
    "PD(V100-P,A100-D)": "PD(V100-P,A100-D)",
}
ARCH_COLORS = {
    "colocated": "#4C78A8",
    "PD(A100-P,V100-D)": "#F58518",
    "PD(V100-P,A100-D)": "#54A24B",
}
ARCH_MARKERS = {
    "colocated": "o",
    "PD(A100-P,V100-D)": "s",
    "PD(V100-P,A100-D)": "^",
}
CASE_ORDER = [
    "short_low_max",
    "short_medium_max",
    "prefill_low_max",
    "prefill_medium_max",
    "prefill_medium_eco",
    "balanced_low_max",
    "balanced_medium_max",
    "decode_low_max",
    "decode_medium_max",
    "decode_medium_eco",
]
CASE_LABELS = {
    case: case.replace("_medium_", "\nmed\n").replace("_low_", "\nlow\n").replace(
        "_max", "max"
    ).replace("_eco", "eco")
    for case in CASE_ORDER
}
WORKLOAD_ORDER = ["short", "prefill", "balanced", "decode"]
LOAD_ORDER = ["low", "medium"]


def _float(value: str) -> float:
    return float(value) if value else math.nan


def load_aggregated(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "repeats",
        "mean_energy_per_request_j",
        "energy_ci95_low_j",
        "energy_ci95_high_j",
        "mean_energy_per_output_token_j",
        "mean_p90_ttft_ms",
        "mean_p90_tbt_ms",
        "mean_request_throughput_rps",
        "failed_requests",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = _float(row[key])
    return rows


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.close(fig)


def _phase_points(
    rows: Iterable[dict[str, Any]], suite: str, phase: str
) -> list[dict[str, Any]]:
    points = [
        row
        for row in rows
        if row["suite"] == suite and row["case"].startswith(f"{phase}_f")
    ]
    return sorted(points, key=lambda row: int(re.search(r"_f(\d+)", row["case"]).group(1)))


def plot_phase(rows: list[dict[str, Any]], suite: str, gpu: str, output_dir: Path) -> None:
    prefill = _phase_points(rows, suite, "prefill")
    decode = _phase_points(rows, suite, "decode")
    pf_freq = [int(re.search(r"_f(\d+)", row["case"]).group(1)) for row in prefill]
    dec_freq = [int(re.search(r"_f(\d+)", row["case"]).group(1)) for row in decode]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    panels = [
        (
            axes[0, 0],
            pf_freq,
            [row["mean_energy_per_request_j"] for row in prefill],
            "Prefill energy",
            "Energy (J / request)",
            ARCH_COLORS["colocated"],
        ),
        (
            axes[0, 1],
            pf_freq,
            [row["mean_p90_ttft_ms"] for row in prefill],
            "Prefill latency",
            "p90 TTFT (ms)",
            ARCH_COLORS["PD(A100-P,V100-D)"],
        ),
        (
            axes[1, 0],
            dec_freq,
            [row["mean_energy_per_request_j"] for row in decode],
            "Decode energy",
            "Energy (J / request)",
            ARCH_COLORS["colocated"],
        ),
        (
            axes[1, 1],
            dec_freq,
            [row["mean_p90_tbt_ms"] for row in decode],
            "Decode latency",
            "p90 TBT (ms)",
            ARCH_COLORS["PD(A100-P,V100-D)"],
        ),
    ]
    for ax, x, y, title, ylabel, color in panels:
        ax.plot(x, y, marker="o", linewidth=2, color=color)
        ax.set_title(f"{gpu}: {title}")
        ax.set_xlabel("SM clock (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        for px, py in zip(x, y):
            ax.annotate(f"{py:.1f}", (px, py), xytext=(0, 7), textcoords="offset points", ha="center")
        ax.margins(y=0.16)
    fig.suptitle(f"{gpu} phase-level DVFS sensitivity", fontsize=15, y=1.01)
    save(fig, output_dir, f"phase_{gpu.lower()}.png")


def _crossover_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["architecture"] in ARCHITECTURES]


def _lookup(
    rows: Iterable[dict[str, Any]], architecture: str, case: str
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["architecture"] == architecture and row["case"] == case
        ),
        None,
    )


def grouped_bars(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    filename: str,
    *,
    log_y: bool = False,
    error_bars: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(CASE_ORDER))
    width = 0.25
    for index, architecture in enumerate(ARCHITECTURES):
        selected = [_lookup(rows, architecture, case) for case in CASE_ORDER]
        values = [row[metric] if row is not None else math.nan for row in selected]
        kwargs: dict[str, Any] = {}
        if error_bars:
            lows = [
                row[metric] - row["energy_ci95_low_j"] if row is not None else 0
                for row in selected
            ]
            highs = [
                row["energy_ci95_high_j"] - row[metric] if row is not None else 0
                for row in selected
            ]
            kwargs.update(yerr=np.array([lows, highs]), capsize=2, error_kw={"elinewidth": 1})
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=ARCH_LABELS[architecture],
            color=ARCH_COLORS[architecture],
            **kwargs,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Workload / load / clock policy")
    ax.set_xticks(x, [CASE_LABELS[case] for case in CASE_ORDER])
    if log_y:
        ax.set_yscale("log")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, output_dir, filename)


def plot_energy_delta(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(CASE_ORDER))
    width = 0.36
    pd_arches = ARCHITECTURES[1:]
    for index, architecture in enumerate(pd_arches):
        values = []
        for case in CASE_ORDER:
            colocated = _lookup(rows, "colocated", case)
            pd_row = _lookup(rows, architecture, case)
            values.append(
                100
                * (
                    pd_row["mean_energy_per_request_j"]
                    / colocated["mean_energy_per_request_j"]
                    - 1
                )
            )
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=f"{ARCH_LABELS[architecture]} vs colocated",
            color=ARCH_COLORS[architecture],
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("PD energy relative to colocated (negative means lower GPU energy)")
    ax.set_ylabel("Energy difference (%)")
    ax.set_xlabel("Workload / load / clock policy")
    ax.set_xticks(x, [CASE_LABELS[case] for case in CASE_ORDER])
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, output_dir, "crossover_energy_delta.png")


def plot_pareto(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture]
        ax.scatter(
            [row["mean_p90_ttft_ms"] for row in selected],
            [row["mean_energy_per_request_j"] for row in selected],
            s=65,
            alpha=0.9,
            color=ARCH_COLORS[architecture],
            marker=ARCH_MARKERS[architecture],
            label=ARCH_LABELS[architecture],
        )
    ax.set_xscale("log")
    ax.set_title("Energy–TTFT trade-off across equal-GPU architectures")
    ax.set_xlabel("p90 TTFT (ms, log scale)")
    ax.set_ylabel("GPU energy (J / request)")
    ax.legend()
    save(fig, output_dir, "crossover_pareto_energy_ttft.png")


def _cell_label(workload: str, load: str) -> str:
    return f"{workload}\n{'med' if load == 'medium' else load}"


def plot_slo_attainment(summary: dict[str, Any], output_dir: Path) -> None:
    cells = [(workload, load) for workload in WORKLOAD_ORDER for load in LOAD_ORDER]
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True, layout="constrained"
    )
    for ax, slo in zip(axes, ["tight", "loose"]):
        matrix = np.full((len(ARCHITECTURES), len(cells)), np.nan)
        for col, (workload, load) in enumerate(cells):
            record = next(
                item
                for item in summary["selective"]
                if item["workload"] == workload
                and item["load"] == load
                and item["slo"] == slo
            )
            for row_index, architecture in enumerate(ARCHITECTURES):
                candidates = [
                    candidate
                    for candidate in record["candidates"]
                    if candidate["architecture"] == architecture
                ]
                matrix[row_index, col] = max(
                    min(candidate["ttft_attainment"], candidate["tbt_attainment"])
                    for candidate in candidates
                )
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_title(
            f"{slo.title()} SLO: best minimum of TTFT/TBT attainment per architecture"
        )
        ax.set_yticks(range(len(ARCHITECTURES)), [ARCH_LABELS[item] for item in ARCHITECTURES])
        for row_index in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row_index, col]
                color = "white" if value < 0.35 else "black"
                ax.text(col, row_index, f"{100 * value:.0f}%", ha="center", va="center", color=color)
                if value >= 0.9:
                    ax.add_patch(
                        plt.Rectangle(
                            (col - 0.48, row_index - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="black",
                            linewidth=1.5,
                        )
                    )
    axes[-1].set_xticks(
        range(len(cells)), [_cell_label(workload, load) for workload, load in cells]
    )
    axes[-1].set_xlabel("Workload / load")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Attainment ratio")
    fig.suptitle(
        "90% dual-SLO feasibility (outlined cells are feasible)",
        fontsize=15,
        y=1.06,
    )
    save(fig, output_dir, "slo_attainment.png")


def plot_slo_winners(summary: dict[str, Any], output_dir: Path) -> None:
    cells = [(workload, load) for workload in WORKLOAD_ORDER for load in LOAD_ORDER]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(cells))
    width = 0.36
    for index, slo in enumerate(["tight", "loose"]):
        records = [
            next(
                item
                for item in summary["selective"]
                if item["workload"] == workload
                and item["load"] == load
                and item["slo"] == slo
            )
            for workload, load in cells
        ]
        ax.bar(
            x + (index - 0.5) * width,
            [item["winner_energy_per_request_j"] for item in records],
            width,
            label=f"{slo} SLO winner",
            color=["#4C78A8", "#72B7B2"][index],
        )
    ax.set_title("Energy of the feasible oracle winner under dual SLOs")
    ax.set_ylabel("GPU energy (J / request)")
    ax.set_xlabel("Workload / load")
    ax.set_xticks(x, [_cell_label(workload, load) for workload, load in cells])
    ax.legend()
    save(fig, output_dir, "slo_winner_energy.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregated",
        type=Path,
        default=Path("results/experiments/summary/aggregated.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/experiments/summary/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
    )
    args = parser.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_aggregated(args.aggregated)
    summary = load_summary(args.summary)
    crossover = _crossover_rows(rows)

    plot_phase(rows, "phase_a100", "A100", args.output_dir)
    plot_phase(rows, "phase_v100", "V100", args.output_dir)
    grouped_bars(
        crossover,
        "mean_energy_per_request_j",
        "GPU energy (J / request)",
        "Equal-GPU architecture crossover: energy per request (95% CI)",
        args.output_dir,
        "crossover_energy.png",
        error_bars=True,
    )
    plot_energy_delta(crossover, args.output_dir)
    grouped_bars(
        crossover,
        "mean_energy_per_output_token_j",
        "GPU energy (J / output token)",
        "Equal-GPU architecture crossover: energy per output token",
        args.output_dir,
        "crossover_energy_per_token.png",
    )
    grouped_bars(
        crossover,
        "mean_p90_ttft_ms",
        "p90 TTFT (ms, log scale)",
        "Equal-GPU architecture crossover: p90 TTFT",
        args.output_dir,
        "crossover_ttft.png",
        log_y=True,
    )
    grouped_bars(
        crossover,
        "mean_p90_tbt_ms",
        "p90 TBT (ms)",
        "Equal-GPU architecture crossover: p90 TBT",
        args.output_dir,
        "crossover_tbt.png",
    )
    grouped_bars(
        crossover,
        "mean_request_throughput_rps",
        "Request throughput (request / s)",
        "Equal-GPU architecture crossover: request throughput",
        args.output_dir,
        "crossover_throughput.png",
    )
    plot_pareto(crossover, args.output_dir)
    plot_slo_attainment(summary, args.output_dir)
    plot_slo_winners(summary, args.output_dir)


if __name__ == "__main__":
    main()
