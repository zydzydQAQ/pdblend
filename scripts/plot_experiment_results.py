#!/usr/bin/env python3
"""Aggregate latest experiment suite_summary.json files and emit paper figures."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "figures"

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

CASE_SHORT = {
    "short_low_max": "short\nlow",
    "short_medium_max": "short\nmed",
    "prefill_low_max": "prefill\nlow",
    "prefill_medium_max": "prefill\nmed",
    "prefill_medium_eco": "prefill\neco",
    "balanced_low_max": "balanced\nlow",
    "balanced_medium_max": "balanced\nmed",
    "decode_low_max": "decode\nlow",
    "decode_medium_max": "decode\nmed",
    "decode_medium_eco": "decode\neco",
}


def latest_suite(name: str) -> Path | None:
    dirs = sorted(
        [p for p in EXP.iterdir() if p.is_dir() and f"-{name}-" in p.name],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs:
        summary = d / "suite_summary.json"
        if summary.exists() and summary.stat().st_size > 200:
            return summary
    return None


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs)


def aggregate(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row["case"]].append(row)
    out = {}
    for case, items in grouped.items():
        j_req = [x["energy_per_successful_request_j"] for x in items]
        j_tok = [x["energy_per_output_token_j"] for x in items]
        ttft = [x["latency"]["p90_ttft_ms"] for x in items]
        tbt_raw = [x["latency"]["p90_tbt_ms"] for x in items]
        tbt = [v for v in tbt_raw if v is not None]
        power = []
        for x in items:
            per = x["energy"]["per_gpu"]
            power.append(sum(g["mean_power_w"] for g in per.values()))
        clocks = items[0].get("clocks_mhz", {})
        mhz = next(iter(clocks.values())) if clocks else None
        out[case] = {
            "n": len(items),
            "mhz": mhz,
            "j_req": mean(j_req),
            "j_tok": mean(j_tok),
            "p90_ttft": mean(ttft),
            "p90_tbt": mean(tbt) if tbt else None,
            "mean_power_w": mean(power) if power else None,
        }
    return out


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def plot_phase(phase: dict[str, dict], title_gpu: str, fname: str, clocks: list[int]):
    prefill = [f"prefill_f{c}" for c in clocks]
    decode = [f"decode_f{c}" for c in clocks]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    ax = axes[0]
    ax.plot(clocks, [phase[c]["j_req"] for c in prefill], marker="o", label="prefill")
    ax.plot(clocks, [phase[c]["j_req"] for c in decode], marker="s", label="decode")
    ax.set_xlabel("SM clock (MHz)")
    ax.set_ylabel("Energy (J / request)")
    ax.set_title(f"{title_gpu}: J/request vs frequency")
    ax.legend()

    ax = axes[1]
    ax.plot(clocks, [phase[c]["p90_ttft"] for c in prefill], marker="o", label="prefill p90 TTFT")
    ax.plot(clocks, [phase[c]["p90_ttft"] for c in decode], marker="s", label="decode p90 TTFT")
    ax.set_xlabel("SM clock (MHz)")
    ax.set_ylabel("p90 TTFT (ms)")
    ax.set_title(f"{title_gpu}: p90 TTFT vs frequency")
    ax.legend()

    ax = axes[2]
    tbt = [phase[c]["p90_tbt"] for c in decode]
    ax.plot(clocks, tbt, marker="s", color="C1", label="decode p90 TBT")
    ax.set_xlabel("SM clock (MHz)")
    ax.set_ylabel("p90 TBT (ms)")
    ax.set_title(f"{title_gpu}: decode p90 TBT vs frequency")
    ax.legend()
    fig.tight_layout()
    save(fig, fname)


def grouped_bars(ax, categories, series: dict[str, list[float]], ylabel: str, title: str, log=False):
    n = len(categories)
    k = len(series)
    width = 0.8 / k
    x = list(range(n))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (k - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def main():
    style()
    latest = {name: latest_suite(name) for name in [
        "phase_a100",
        "phase_v100",
        "crossover_colocated",
        "crossover_a100_prefill",
        "crossover_v100_prefill",
    ]}
    missing = [k for k, v in latest.items() if v is None]
    if missing:
        raise SystemExit(f"missing suites: {missing}")

    phase_a100 = aggregate(latest["phase_a100"])
    phase_v100 = aggregate(latest["phase_v100"])
    coloc = aggregate(latest["crossover_colocated"])
    pd_a = aggregate(latest["crossover_a100_prefill"])
    pd_v = aggregate(latest["crossover_v100_prefill"])

    print("using")
    for k, v in latest.items():
        print(f"  {k}: {v.parent.name}")

    plot_phase(phase_a100, "A100", "phase_a100.png", [615, 930, 1410])
    plot_phase(phase_v100, "V100", "phase_v100.png", [600, 900, 1380])

    cats = [CASE_SHORT[c] for c in CASE_ORDER]
    energy = {
        "colocated": [coloc[c]["j_req"] for c in CASE_ORDER],
        "PD(A100-P,V100-D)": [pd_a[c]["j_req"] for c in CASE_ORDER],
        "PD(V100-P,A100-D)": [pd_v[c]["j_req"] for c in CASE_ORDER],
    }
    ttft = {
        "colocated": [coloc[c]["p90_ttft"] for c in CASE_ORDER],
        "PD(A100-P,V100-D)": [pd_a[c]["p90_ttft"] for c in CASE_ORDER],
        "PD(V100-P,A100-D)": [pd_v[c]["p90_ttft"] for c in CASE_ORDER],
    }
    tbt = {
        "colocated": [coloc[c]["p90_tbt"] for c in CASE_ORDER],
        "PD(A100-P,V100-D)": [pd_a[c]["p90_tbt"] for c in CASE_ORDER],
        "PD(V100-P,A100-D)": [pd_v[c]["p90_tbt"] for c in CASE_ORDER],
    }

    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    grouped_bars(ax, cats, energy, "Energy (J / request)", "Equal-GPU architecture crossover: energy per request")
    fig.tight_layout()
    save(fig, "crossover_energy.png")

    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    grouped_bars(ax, cats, ttft, "p90 TTFT (ms)", "Equal-GPU architecture crossover: p90 TTFT", log=True)
    fig.tight_layout()
    save(fig, "crossover_ttft.png")

    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    grouped_bars(ax, cats, tbt, "p90 TBT (ms)", "Equal-GPU architecture crossover: p90 TBT")
    fig.tight_layout()
    save(fig, "crossover_tbt.png")

    # Energy delta vs colocated (negative = PD cheaper)
    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    delta = {
        "PD(A100-P,V100-D) vs colocated": [
            100.0 * (pd_a[c]["j_req"] - coloc[c]["j_req"]) / coloc[c]["j_req"] for c in CASE_ORDER
        ],
        "PD(V100-P,A100-D) vs colocated": [
            100.0 * (pd_v[c]["j_req"] - coloc[c]["j_req"]) / coloc[c]["j_req"] for c in CASE_ORDER
        ],
    }
    grouped_bars(
        ax,
        cats,
        delta,
        "Energy delta vs colocated (%)",
        "PD energy relative to colocated (negative = PD uses less energy)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    save(fig, "crossover_energy_delta.png")

    # Pareto-style scatter: energy vs TTFT, one point per case, marker by arch
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    series = [
        ("colocated", coloc, "o"),
        ("PD(A100-P,V100-D)", pd_a, "s"),
        ("PD(V100-P,A100-D)", pd_v, "^"),
    ]
    for name, data, marker in series:
        xs = [data[c]["p90_ttft"] for c in CASE_ORDER]
        ys = [data[c]["j_req"] for c in CASE_ORDER]
        ax.scatter(xs, ys, marker=marker, label=name, s=42)
    ax.set_xscale("log")
    ax.set_xlabel("p90 TTFT (ms, log)")
    ax.set_ylabel("Energy (J / request)")
    ax.set_title("Energy–TTFT tradeoff across architectures")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    save(fig, "crossover_pareto_energy_ttft.png")

    summary = {
        "sources": {k: str(v.parent.name) for k, v in latest.items()},
        "phase_a100": phase_a100,
        "phase_v100": phase_v100,
        "crossover": {
            "colocated": coloc,
            "PD(A100-P, V100-D)": pd_a,
            "PD(V100-P, A100-D)": pd_v,
        },
    }
    (OUT / "plot_data.json").write_text(json.dumps(summary, indent=2))
    print("wrote", OUT / "plot_data.json")


if __name__ == "__main__":
    main()
