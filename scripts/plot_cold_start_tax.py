#!/usr/bin/env python3
"""O4 暖启动验证图：sharegpt-x0.5-pdblend 功率时间序列 × 计划段着色。

数据源：results/v2/eval-7b-v2/sharegpt-x0.5-pdblend/{power,controller}.jsonl
输出：当前工程 results/v2/figs/o4_cold_start_tax.png
同时打印分段实测表（去头 3s），与 OPTIMIZATION-LOG §分段实测 核对。
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN = PROJECT_ROOT / "results/v2/eval-7b-v2/sharegpt-x0.5-pdblend"
OUT = PROJECT_ROOT / "results/v2/figs/o4_cold_start_tax.png"


def open_result(path: Path):
    resolved = path if path.exists() else Path(str(path) + ".gz")
    return gzip.open(resolved, "rt") if resolved.name.endswith(".gz") else resolved.open()

with open_result(RUN / "power.jsonl") as fh:
    power = [json.loads(l) for l in fh]
t0 = power[0][0]
ts = np.array([p[0] - t0 for p in power])
ws = np.array([sum(p[1]) for p in power])

plans = []
with open_result(RUN / "controller.jsonl") as fh:
    for line in fh:
        e = json.loads(line)
        if e.get("kind") != "plan":
            continue
        counts = e.get("counts", {})
        fmap = {"P": e.get("f_P"), "D": e.get("f_D"), "M": e.get("f_M")}
        freqs = {fmap[r] for r in ("P", "D", "M") if counts.get(r)}
        parts = [f"{counts[r]}{r}" for r in ("P", "D", "M") if counts.get(r)]
        if counts.get("L1"):
            parts.append(f"{counts['L1']}L1")
        if counts.get("off"):
            parts.append(f"{counts['off']}off")
        label = "+".join(parts) or "?"
        if len(freqs) == 1:
            label += f"@{next(iter(freqs))}"
        plans.append({"t": e["t"] - t0, "label": label, "pred_w": e.get("power_w", 0.0),
                      "cold": e.get("cold_start", False)})

# 合并连续相同布局段
segs = []
for p in plans:
    if segs and segs[-1]["label"] == p["label"]:
        segs[-1]["pred_w"] = p["pred_w"]
    else:
        segs.append(dict(p))
for i, s in enumerate(segs):
    s["t1"] = segs[i + 1]["t"] if i + 1 < len(segs) else ts[-1]
    m = (ts >= s["t"] + 3.0) & (ts < s["t1"])
    s["meas_w"] = float(ws[m].mean()) if m.any() else float("nan")

print(f"{'布局':<22}{'段长(s)':>8}{'预测W':>8}{'实测W(去头3s)':>14}")
for s in segs:
    print(f"{s['label']:<22}{s['t1']-s['t']:>8.1f}{s['pred_w']:>8.0f}{s['meas_w']:>14.0f}")

fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=160)
palette = plt.cm.tab10.colors
seen = {}
for s in segs:
    key = "cold" if s["cold"] else ("warm" if s["label"].startswith("5M") else s["label"])
    if key == "cold":
        color, alpha = "#d62728", 0.16
    elif key == "warm":
        color, alpha = "#2ca02c", 0.22
    else:
        idx = seen.setdefault(s["label"], len(seen))
        color, alpha = palette[idx % 10], 0.10
    ax.axvspan(s["t"], s["t1"], color=color, alpha=alpha, lw=0)
    ax.axvline(s["t"], color="gray", lw=0.5, ls=":", alpha=0.7)

ax.plot(ts, ws, color="black", lw=1.0)

# 段标签（布局名 + 实测均值），错层避免重叠
ymax = ws.max() * 1.12
ax.set_ylim(0, ymax)
for i, s in enumerate(segs):
    mid = (s["t"] + s["t1"]) / 2
    y = ymax * (0.97 if i % 2 == 0 else 0.86)
    name = "cold-start\n" + s["label"] if s["cold"] else s["label"]
    ax.text(mid, y, name, ha="center", va="top", fontsize=7.5,
            color="#444444" if not s["cold"] else "#d62728")

# 暖计划段标注：预测 vs 实测
warm = [s for s in segs if s["label"].startswith("5M")]
if warm:
    s = warm[-1]
    ax.annotate(f"warm plan M5+3L1@2100\nmeasured {s['meas_w']:.0f} W (predicted {s['pred_w']:.0f} W)",
                xy=((s["t"] + s["t1"]) / 2, s["meas_w"]), xytext=(s["t1"] + 18, 700),
                fontsize=8, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
# 冷启动税标注
cold = [s for s in segs if s["cold"]]
if cold:
    s = cold[0]
    ax.annotate(f"cold-start tax: {s['meas_w']:.0f} W for {s['t1']-s['t']:.0f}s",
                xy=(s["t1"] / 2, s["meas_w"]), xytext=(s["t1"] + 25, s["meas_w"] + 60),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

ax.set_xlabel("time since run start (s)")
ax.set_ylabel("node power (W), sum of 8 GPUs")
ax.set_title("sharegpt x0.5 pdblend (warm-start validation): power trace with plan segments\n"
             f"{len(plans)} plan switches in {ts[-1]:.0f}s", fontsize=10)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"saved: {OUT}")
