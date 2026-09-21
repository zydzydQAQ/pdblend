"""Motivation figures: M1 phase/frequency asymmetry and M2 static power from profile raw.json; M3 crossover heat-map."""
from __future__ import annotations

import json
from pathlib import Path

from .report import load_points


def m1_rows(raw: dict, prefill_tokens: int = 2048, batch: int = 32, ctx: int = 1024) -> list[dict]:
    """Per frequency: prefill J/token and ms for one prompt, decode J/token and ms/step for one batch."""
    rows = []
    for f in raw["freqs"]:
        p = next((r for r in raw["prefill"] if r["freq_mhz"] == f and r["input_tokens"] == prefill_tokens), None)
        d = next((r for r in raw["decode"] if r["freq_mhz"] == f and r["batch"] == batch and r["context_tokens"] == ctx), None)
        if not p or not d or d["power_w"] is None:
            continue
        rows.append(dict(freq_mhz=f,
                         prefill_ms=p["seconds"] * 1e3, prefill_j_per_token=p["seconds"] * p["power_w"] / prefill_tokens,
                         decode_ms_per_step=d["step_seconds"] * 1e3, decode_j_per_token=d["step_seconds"] * d["power_w"] / batch,
                         prefill_power_w=p["power_w"], decode_power_w=d["power_w"]))
    return rows


def m2_rows(raw: dict) -> list[dict]:
    st = raw["static"]
    order = ["off", "parked", "active_idle_reset"] + [f"active_idle@{f}" for f in raw["freqs"]]
    return [dict(state=k, power_w=st[k]["power_w"], wake_s=st[k].get("wake_s")) for k in order if k in st]


def m3_rows(root: Path, slo_floor: float = 0.9) -> list[dict]:
    """One row per (dataset, scale, layout label): energy per request and feasibility."""
    rows = []
    for s in load_points(root):
        meta, fp = s.get("trace_meta", {}), s.get("fixed_plan") or {}
        counts = fp.get("counts", {})
        layout = "+".join(f"{v}{k}" for k, v in counts.items() if v and k in ("P", "D", "M")) or s["policy"]["name"]
        rows.append(dict(dataset=meta.get("dataset"), scale=meta.get("scale"), mean_rps=round(s["trace"]["mean_rps"], 2),
                         layout=layout, f_P=fp.get("f_P"), f_D=fp.get("f_D"), f_M=fp.get("f_M"),
                         joint_slo_rate=s["slo"]["joint_slo_rate"], feasible=s["slo"]["joint_slo_rate"] >= slo_floor,
                         j_per_request=s["j_per_request"], mean_power_w=s["mean_power_w"],
                         ttft_p90=s["slo"]["ttft_p90"], tpot_p90=s["slo"]["tpot_p90"], dir=s["_dir"]))
    return rows


def m3_winners(rows: list[dict]) -> list[dict]:
    """Best feasible configuration per (dataset, scale), and the best co-located one for the saving ratio."""
    out = []
    keys = sorted({(r["dataset"], r["scale"]) for r in rows}, key=lambda k: (k[0], k[1] or 0))
    for ds, sc in keys:
        cell = [r for r in rows if (r["dataset"], r["scale"]) == (ds, sc) and r["feasible"]]
        if not cell:
            out.append(dict(dataset=ds, scale=sc, winner=None))
            continue
        best = min(cell, key=lambda r: r["j_per_request"])
        colocated = [r for r in cell if r["layout"].endswith("M") and "+" not in r["layout"]]
        base = min(colocated, key=lambda r: r["j_per_request"]) if colocated else None
        out.append(dict(dataset=ds, scale=sc, winner=best["layout"], winner_f=(best["f_P"], best["f_D"], best["f_M"]),
                        winner_j_per_request=best["j_per_request"],
                        colocated_best_j_per_request=base["j_per_request"] if base else None,
                        saving_vs_colocated=(1 - best["j_per_request"] / base["j_per_request"]) if base else None))
    return out


def plot_m1_m2(raw_path: Path, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = json.loads(Path(raw_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    m1, m2 = m1_rows(raw), m2_rows(raw)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    f = [r["freq_mhz"] for r in m1]
    ax[0].plot(f, [r["prefill_j_per_token"] * 1e3 for r in m1], "o-", label="prefill (2048 tok)")
    ax[0].plot(f, [r["decode_j_per_token"] * 1e3 for r in m1], "s-", label="decode (B=32, ctx 1024)")
    ax[0].set_xlabel("SM clock (MHz)"); ax[0].set_ylabel("mJ / token"); ax[0].legend(); ax[0].set_title("energy per token")
    ax[1].plot(f, [r["prefill_ms"] / m1[-1]["prefill_ms"] for r in m1], "o-", label="prefill latency")
    ax[1].plot(f, [r["decode_ms_per_step"] / m1[-1]["decode_ms_per_step"] for r in m1], "s-", label="decode step")
    ax[1].set_xlabel("SM clock (MHz)"); ax[1].set_ylabel("latency / latency@max clock"); ax[1].legend(); ax[1].set_title("slowdown")
    fig.tight_layout(); fig.savefig(out_dir / "m1_phase_frequency.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar([r["state"].replace("active_idle@", "idle@").replace("active_idle_reset", "idle@reset") for r in m2],
           [r["power_w"] for r in m2])
    ax.set_ylabel("W per GPU"); ax.set_title("static power by state (L20)"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out_dir / "m2_static_power.png", dpi=160); plt.close(fig)
    summary = dict(m1=m1, m2=m2)
    (out_dir / "m1_m2.json").write_text(json.dumps(summary, indent=1))
    return summary


def plot_m3(root: Path, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = m3_rows(root)
    winners = m3_winners(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    layouts = sorted({r["layout"] for r in rows})
    scales = sorted({r["scale"] for r in rows if r["scale"] is not None})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        grid = [[None] * len(scales) for _ in layouts]
        for i, lay in enumerate(layouts):
            for j, sc in enumerate(scales):
                cell = [r for r in rows if (r["dataset"], r["layout"], r["scale"]) == (ds, lay, sc) and r["feasible"]]
                if cell:
                    grid[i][j] = min(r["j_per_request"] for r in cell)
        vals = [[v if v is not None else float("nan") for v in row] for row in grid]
        im = ax.imshow(vals, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(scales))); ax.set_xticklabels([f"{s:g}" for s in scales])
        ax.set_yticks(range(len(layouts))); ax.set_yticklabels(layouts)
        ax.set_title(f"{ds}: best feasible J/req"); ax.set_xlabel("load (x co-located capacity)")
        for i in range(len(layouts)):
            for j in range(len(scales)):
                ax.text(j, i, "x" if grid[i][j] is None else f"{grid[i][j]:.0f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out_dir / "m3_crossover.png", dpi=160); plt.close(fig)
    (out_dir / "m3.json").write_text(json.dumps(dict(rows=rows, winners=winners), indent=1))
    return dict(rows=rows, winners=winners)
