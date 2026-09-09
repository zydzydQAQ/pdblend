# -*- coding: utf-8 -*-
"""全局周期调度器:监督负载,在分区空间中选最小能耗可行分区,含 GPU 迁移。

决策流程(每周期 T_g):
  1. 对每个候选分区预测 (可行, attainment, 功率/能耗);
  2. target = baseline_attainment - slo_non_inferior_pp(同负载非劣);
  3. 选 min 能耗 s.t. 预测 attainment >= target 且队列稳定;
  4. 迁移需过三关:最小驻留(partition_min_dwell_s)、迟滞(hysteresis)、
     迁移成本摊销((ΔP×驻留窗) >= migration_cost_j)。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ecopadg.types import Partition, SystemConfig


@dataclass
class Prediction:
    """分区预测结果。

    power_w:稳态功率。energy_j_per_s:gross-energy 代理
    (P / SLO-goodput × λ),禁止再静默写成 power_w。
    """
    attainable: bool = True
    attainment: float = 0.0
    power_w: float = 0.0
    energy_j_per_s: float = 0.0
    energy_j_per_req: float = 0.0
    goodput_rps: float = 0.0
    calibrated: bool = True
    rho: float = float("nan")


@dataclass
class GlobalState:
    """全局调度器观测状态。"""
    partition: Partition
    lam_hat: float = 0.0
    baseline_attainment: float = 0.5
    last_switch: float = -1e9
    extra: dict = field(default_factory=dict)


@dataclass
class Action:
    """全局调度器动作。"""
    migrate: bool
    partition: Partition
    reasons: List[str] = field(default_factory=list)
    projected_savings_w: float = float("nan")
    restart_required: bool = False


def partition_requires_rematerialization(
    current: Partition, target: Partition
) -> bool:
    """Return whether a partition change cannot be park/unpark only."""
    current_counts = (
        current.n_mixed, current.n_prefill, current.n_decode
    )
    target_counts = (target.n_mixed, target.n_prefill, target.n_decode)
    deltas = tuple(
        target_value - current_value
        for current_value, target_value in zip(
            current_counts, target_counts
        )
    )
    if any(value > 0 for value in deltas) and any(
        value < 0 for value in deltas
    ):
        return True
    role_shapes = (
        (current.n_mixed, current.tp_mixed, current.pp_mixed,
         target.n_mixed, target.tp_mixed, target.pp_mixed),
        (current.n_prefill, current.tp_prefill, current.pp_prefill,
         target.n_prefill, target.tp_prefill, target.pp_prefill),
        (current.n_decode, current.tp_decode, current.pp_decode,
         target.n_decode, target.tp_decode, target.pp_decode),
    )
    return any(
        old_n > 0 and new_n > 0 and (old_tp, old_pp) != (new_tp, new_pp)
        for old_n, old_tp, old_pp, new_n, new_tp, new_pp in role_shapes
    )


def build_partition_space(tp_options: dict, gpu_count: int,
                          pp_options: Optional[dict] = None) -> List[Partition]:
    """由角色 TP(/PP) 选项生成分区空间。

    tp_options = {"mixed": [...], "prefill": [...], "decode": [...]}
    pp 缺省为 1,保持旧测试;传入 pp_options 则搜非对称并行。
    """
    out: List[Partition] = []
    from itertools import product
    pps = pp_options or {}
    for tm, tp, td, pm, pp, pd in product(
            tp_options.get("mixed", [1]),
            tp_options.get("prefill", [1]),
            tp_options.get("decode", [1]),
            pps.get("mixed", [1]),
            pps.get("prefill", [1]),
            pps.get("decode", [1])):
        for nm in range(0, gpu_count // max(tm * pm, 1) + 1):
            for np_ in range(0, gpu_count // max(tp * pp, 1) + 1):
                for nd in range(0, gpu_count // max(td * pd, 1) + 1):
                    used = nm * tm * pm + np_ * tp * pp + nd * td * pd
                    if used > gpu_count:
                        continue
                    if np_ > 0 and nd == 0:
                        continue
                    if nd > 0 and np_ == 0:
                        continue
                    if used == 0:
                        continue
                    out.append(Partition(
                        n_mixed=nm, n_prefill=np_, n_decode=nd,
                        tp_mixed=tm, tp_prefill=tp, tp_decode=td,
                        pp_mixed=pm, pp_prefill=pp, pp_decode=pd,
                        gpus_total=gpu_count))
    seen = set()
    uniq = []
    for p in out:
        k = (p.n_mixed, p.n_prefill, p.n_decode,
             p.tp_mixed if p.n_mixed else 0,
             p.tp_prefill if p.n_prefill else 0,
             p.tp_decode if p.n_decode else 0,
             p.pp_mixed if p.n_mixed else 0,
             p.pp_prefill if p.n_prefill else 0,
             p.pp_decode if p.n_decode else 0)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


class GlobalScheduler:
    """周期调度器主体;predictor 可注入(默认见 ApproxPredictor 语义,由调用方
    提供 OpModel 支撑的实现;此处接受 callable)。"""

    def __init__(self, space: Sequence[Partition],
                 predictor: Callable[[Partition, float, GlobalState], Prediction],
                 config: SystemConfig, clock=time.time):
        self.space = list(space)
        self.predictor = predictor
        self.cfg = config
        self._clock = clock

    @staticmethod
    def _pkey(p: Partition) -> tuple:
        """语义键:数量为 0 的角色 TP 归一为 0(避免等价分区键漂移)。"""
        return (p.n_mixed, p.n_prefill, p.n_decode,
                p.tp_mixed if p.n_mixed else 0,
                p.tp_prefill if p.n_prefill else 0,
                p.tp_decode if p.n_decode else 0,
                getattr(p, "pp_mixed", 1) if p.n_mixed else 0,
                getattr(p, "pp_prefill", 1) if p.n_prefill else 0,
                getattr(p, "pp_decode", 1) if p.n_decode else 0)

    def _target(self, st: GlobalState) -> float:
        """同负载:att >= att_mixed − δ。不用绝对 0.9。"""
        delta = getattr(self.cfg, "slo_non_inferior_pp", None)
        if delta is None:
            delta = self.cfg.attainment_tolerance_pp
        return float(st.baseline_attainment) - float(delta)

    def step(self, state: GlobalState, now: float) -> Action:
        target = self._target(state)
        preds = {}
        candidates = []
        rejected = []
        for p in self.space:
            pr = self.predictor(p, state.lam_hat, state)
            preds[self._pkey(p)] = pr
            if pr.attainable and pr.attainment >= target:
                candidates.append(p)
            else:
                rejected.append((p, pr))
        if not candidates:
            return Action(migrate=False, partition=state.partition,
                          reasons=["no-feasible-candidate", "max-freq"])
        best = min(candidates, key=lambda p: preds[self._pkey(p)].energy_j_per_s)
        # 冷启动/空窗 λ̂=0:更小分区永远更省,会把 2×1P1D park 成 1 段
        # (32B n=500 r=3.4:前 30s migrate→1 pair,TTFT p99=120s)。
        if float(state.lam_hat) <= 1e-9 and (
                best.total_gpus() < state.partition.total_gpus()
                or best.pairs() < state.partition.pairs()
                or best.n_mixed < state.partition.n_mixed):
            return Action(migrate=False, partition=state.partition,
                          reasons=["no-downscale-cold"])
        cur_e = preds.get(self._pkey(state.partition))
        # SLO 紧急通道(Gate 1 语义):当前分区不可行/低于目标 → 容量优先,
        # 绕过 dwell/迟滞/成本三关立即迁移(节能是次级目标)。
        urgent = (cur_e is None or not cur_e.attainable
                  or cur_e.attainment < target)
        if urgent and self._pkey(best) != self._pkey(state.partition):
            savings = (
                float(cur_e.energy_j_per_s)
                - float(preds[self._pkey(best)].energy_j_per_s)
                if cur_e is not None else float("nan")
            )
            return Action(migrate=True, partition=best,
                          reasons=["urgent-slo"],
                          projected_savings_w=savings,
                          restart_required=partition_requires_rematerialization(
                              state.partition, best))
        if self._pkey(best) == self._pkey(state.partition):
            reasons = ["already-min-energy"]
            for p, pr in rejected:
                if not pr.attainable:
                    reasons.append("unattainable:%s" % p)
                else:
                    reasons.append("target:%s" % p)
            return Action(migrate=False, partition=state.partition,
                          reasons=reasons[:3])
        best_e = preds[self._pkey(best)].energy_j_per_s
        cur = cur_e.energy_j_per_s if cur_e is not None else best_e
        savings = float(cur) - float(best_e)
        restart_required = partition_requires_rematerialization(
            state.partition, best)
        if now - state.last_switch < self.cfg.partition_min_dwell_s:
            return Action(migrate=False, partition=state.partition,
                          reasons=["dwell"])
        gain = (cur - best_e) / cur if cur > 0 else 0.0
        if gain < self.cfg.hysteresis:
            return Action(migrate=False, partition=state.partition,
                          reasons=["hysteresis:gain=%.3f" % gain])
        # Park/unpark and physical role restart have separate cost models.
        # Restart cost is measured and gated by TopologyCoordinator; applying
        # migration_cost_j here as well would double-charge it.
        amortized = savings * self.cfg.partition_min_dwell_s
        if (not restart_required
                and (not math.isfinite(amortized)
                     or amortized < self.cfg.migration_cost_j)):
            return Action(migrate=False, partition=state.partition,
                          reasons=["cost"])
        # 缩容护栏:降配(实例数减少)须在 λ̂×(1+headroom) 下仍可行。
        # λ̂ EMA 在爬坡/回落沿滞后 + 窗口统计漂移会让容量估计虚高,
        # 无余量缩容会造成 park 抖动(日巡实测 t=360/510/661 反复横跳)。
        if best.total_gpus() < state.partition.total_gpus():
            hr = getattr(self.cfg, "downscale_headroom", 0.3)
            pr_hr = self.predictor(best, state.lam_hat * (1.0 + hr), state)
            if not (pr_hr.attainable and pr_hr.attainment >= target):
                return Action(migrate=False, partition=state.partition,
                              reasons=["downscale-headroom"])
        return Action(
            migrate=True,
            partition=best,
            reasons=["ok"],
            projected_savings_w=savings,
            restart_required=restart_required,
        )
