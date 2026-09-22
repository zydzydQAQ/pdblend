#!/usr/bin/env python3
"""论文版五层架构图（Design Overview）。

蓝本：docs/2026-09-21_pdblend技术原理.md §3 ASCII 图。
输出：当前工程 results/v2/figs/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

fig, ax = plt.subplots(figsize=(10, 6.0), dpi=160)
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

def box(x, y, w, h, title, lines, fc="#f7f7f7", ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + (h - 5) / 2, "\n".join(lines), ha="center", va="center",
            fontsize=8.2, color="#222222")

def arrow(x1, y1, x2, y2, color="#333333", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.2, linestyle=ls))

def elbow(points, color, ls="-"):
    """折线 + 末端箭头。points 至少两个。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1] + [xs[-2] + (xs[-1] - xs[-2]) * 0.92] if len(xs) > 1 else xs,
            ys[:-1] + [ys[-2] + (ys[-1] - ys[-2]) * 0.92] if len(ys) > 1 else ys,
            color=color, lw=1.2, ls=ls, solid_capstyle="round")
    arrow(*points[-2], *points[-1], color=color, ls=ls)

# 控制层（顶部）
box(24, 47, 56, 13, "Controller  (control/)",
    ["slow loop ~10 s:  Forecaster (dual EWMA) → PoolPlanner,",
     "enumerate min-power layout in SLO-feasible region",
     "fast loop 1 s:  escalate-only SLO Shield + hold-down floor"],
    fc="#e8f0fe", ec="#1a73e8")

# 模型层（左下，Proxy 正下方）
box(24, 2, 30, 13, "PerfModel  (profile/)",
    ["affine perf / power models",
     "~150-point grid, <1 h per model",
     "residual gate < 15%"],
    fc="#fef7e0", ec="#b06000")

# 客户端
box(1, 25, 15, 10, "Client", ["open-loop", "Poisson replay"], fc="#f1f3f4")

# 代理层
box(24, 21, 30, 17, "Proxy  (proxy/)",
    ["OpenAI-compatible endpoint",
     "role-table routing + τ split",
     "per-request TTFT / TPOT records",
     "byte-level SSE passthrough"],
    fc="#e6f4ea", ec="#188038")

# 引擎层
box(62, 21, 37, 17, "Fleet  (engine/)   8 × vLLM V1",
    ["one instance per GPU, kv_role = kv_both",
     "+ overlay patches (KV transfer fixes)",
     "roles as routing labels: P / D / M / parked / off",
     "per-pool SM frequency; park keeps weights resident"],
    fc="#fce8e6", ec="#d93025")

# 请求流
arrow(16.8, 29.5, 23.2, 29.5, color="#188038")
arrow(54.8, 29.5, 61.2, 29.5, color="#188038")
ax.text(20, 31.2, "requests", ha="center", fontsize=7.8, color="#188038")
ax.text(58, 31.2, "route", ha="center", fontsize=7.8, color="#188038")

# 观测流：Proxy → Controller（x=34 向上）
arrow(34, 38.8, 34, 46.2, color="#1a73e8")
ax.text(33, 42.5, "RequestRecord\n(arrive / first_token / finish)", ha="right",
        fontsize=7.8, color="#1a73e8")

# 控制流：Controller → Proxy（x=44 向下，虚线）
arrow(44, 46.2, 44, 38.8, color="#1a73e8", ls="--")
ax.text(45, 42.5, "routing table / τ rewrite\n(free, ms)", ha="left",
        fontsize=7.8, color="#1a73e8")

# 控制流：Controller → Fleet（x=72 向下，虚线）
arrow(72, 46.2, 72, 38.8, color="#d93025", ls="--")
ax.text(73.5, 42.5, "NVML: lock freq / park / wake\n(100 ms / 28 ms / 32 s)", ha="left",
        fontsize=7.8, color="#d93025")

# 模型供给：PerfModel 右边 → x=58.5 通道向上 → Controller 底
elbow([(54.5, 8.5), (58.5, 8.5), (58.5, 46.2)], color="#b06000")
ax.text(57.5, 18.5, "fitted models\nfeed planner", ha="right", fontsize=7.8, color="#b06000")

# 离线 profile：Fleet 底 → 向下 → 向左 → PerfModel 右边（虚线）
elbow([(66, 20.2), (66, 5.5), (54.8, 5.5)], color="#b06000", ls=":")
ax.text(67.5, 12, "offline\nprofiling", ha="left", fontsize=7.5, color="#b06000")

ax.set_title("PDblend architecture: roles as routing labels, slow-loop planner + fast-loop shield",
             fontsize=11)
fig.tight_layout()
out = str(PROJECT_ROOT / "results/v2/figs/architecture.png")
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(f"saved: {out}")
