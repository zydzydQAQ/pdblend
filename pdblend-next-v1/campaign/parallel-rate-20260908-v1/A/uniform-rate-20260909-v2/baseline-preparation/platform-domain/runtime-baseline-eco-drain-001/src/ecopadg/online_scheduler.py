# -*- coding: utf-8 -*-
"""池内在线调度器:M̃1 窗口状态机 + decode slack 选频。

M̃1 是 phase-biased mixed,不是严格 temporal PaDG:
  - 只改变请求进入引擎的时刻和整卡时钟;
  - 引擎内仍是 chunked prefill + continuous decode,相位重叠率不可约束到 0;
  - prefill 窗锁最高频并释放积压;decode 窗扣住准入并按 J/token 选频;
  - 过载或 force_continuous 退化为连续混批;λ̂ 回落后可回到窗口(非吸收态)。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from ecopadg.types import (
    CTX_REF, DECODE_B_CAP, OUT_REF, RHO_TPUT, TOKEN_BUDGET, SystemConfig,
)


MODE_PREFILL_WINDOW = "prefill_window"
MODE_DECODE_WINDOW = "decode_window"
MODE_CONTINUOUS = "continuous"
# 钉满频不必等 8 个 SLO 样本;相对盾仍用更长窗口。
DECODE_GATE_MIN_SAMPLES = 2
# 相对 −1pp 用短窗,避免等满 64 样本才钉(r=24/28 漏)。
DECODE_RELATIVE_WINDOW = 8
DECODE_SLACK_MARGIN = 0.35


def decode_slack_blocks_dvfs(slack_s, tpot_s: float,
                             margin: float = DECODE_SLACK_MARGIN) -> bool:
    """队列 decode slack 将尽 → 立刻关 D-DVFS。slack 未知则不据此钉频。"""
    if slack_s is None:
        return False
    try:
        slack = float(slack_s)
        tpot = float(tpot_s)
    except (TypeError, ValueError):
        return False
    if slack != slack:
        return False
    if slack <= 0.0:
        return True
    if tpot == tpot and tpot > 0.0 and slack < tpot * float(margin):
        return True
    return False


def decode_occupancy_blocks_dvfs(
        concurrent, n_inst,
        b_cap: int = DECODE_B_CAP,
        alpha: float = 0.50) -> bool:
    """Decode 占用过 α → 立刻关 D-DVFS。

    concurrent 是 λ·T_decode(b=1) 或观测 inflight,不是未校准组批 μ_D。
    容量按 n_inst·DECODE_B_CAP,避免把顺序 μ_D 当成过载去改布局。
    """
    if concurrent is None or n_inst is None:
        return False
    try:
        load = float(concurrent)
        inst = float(n_inst)
        cap = float(b_cap)
        gate = float(alpha)
    except (TypeError, ValueError):
        return False
    if load != load or inst != inst or cap != cap or gate != gate:
        return False
    if load <= 0.0 or inst <= 0.0 or cap <= 0.0:
        return False
    return (load / (inst * cap)) + 1e-12 >= gate


def decode_window_blocks_dvfs(outcomes, *,
                              min_samples: int = DECODE_GATE_MIN_SAMPLES,
                              tau: float = 0.90,
                              baseline: float = float("nan"),
                              trusted: bool = False,
                              delta: float = 0.01) -> bool:
    """少样本窗口已低于 τ 或 baseline−δ → 钉 2520。"""
    buf = list(outcomes or [])
    if len(buf) < int(min_samples):
        return False
    try:
        att = sum(float(x) for x in buf) / float(len(buf))
    except (TypeError, ValueError):
        return False
    if att + 1e-12 < float(tau):
        return True
    if trusted:
        try:
            base = float(baseline)
        except (TypeError, ValueError):
            base = float("nan")
        if base == base and att + 1e-12 < base - float(delta):
            return True
    return False


def select_decode_freq(opmodel, b_hat: int, ctx_hat: float, slo_tpot_s: float,
                       freq_candidates: Sequence[int], margin: float = 0.35,
                       min_tok_per_s: float = 0.0, energy_fn=None,
                       budget_ms_override: Optional[float] = None,
                       demand_tok_per_s: float = 0.0,
                       b_cap: int = 0, prefer_mhz: int = 0) -> int:
    """decode 选频:在同时满足
      (a) iter(b_use,f,ctx) <= 预算(静态 S_TPOT*(1-margin) 或 slack 覆盖);
      (b) b/iter_s(f) >= min_tok_per_s            —— 固定批量吞吐下限;
      (c) demand_tok_per_s>0 时按均衡批量判定:continuous batching 下
          需求升高 → b 自涨,均衡批 b_eq = demand × iter_s(b̂,f),
          可行 = iter(b_use=max(b̂,b_eq), f) <= 预算 且 b_eq <= b_cap。
          (b) 的固定 b 口径在低 b 时会误判"任何频率都不够"→ 永远满频,
          D 池丢失全部 DVFS 空间(日巡实测 D 恒 2520 的根因)。
    的档中取每 token 能耗最低者(E1b 曲线为 U 型,能量最优未必最低频);
    全部不可行时回退最高频(保 SLO 优先于省电)。"""
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin 必须在 [0,1)")
    budget_ms = (budget_ms_override if budget_ms_override is not None
                 else slo_tpot_s * 1000.0 * (1.0 - margin))
    cost = energy_fn if energy_fn is not None else opmodel.dyn_j_per_token
    feasible: List[Tuple[int, float]] = []
    for f in freq_candidates:
        it = opmodel.iter_time_ms(int(b_hat), int(f), float(ctx_hat))
        b_use = int(b_hat)
        if demand_tok_per_s > 0:
            b_eq = demand_tok_per_s * it / 1000.0
            if b_cap > 0 and b_eq > b_cap:
                continue  # 均衡批超过准入上限:该频率吞吐不可持续
            b_use = max(b_use, int(b_eq + 0.999))
            it = opmodel.iter_time_ms(b_use, int(f), float(ctx_hat))
        if it > budget_ms:
            continue
        if min_tok_per_s > 0 and int(b_hat) / max(it / 1000.0, 1e-9) \
                < min_tok_per_s:
            continue
        feasible.append((int(f), float(cost(b_use, int(f)))))
    if not feasible:
        return max(int(f) for f in freq_candidates)
    prefer = int(prefer_mhz or 0)
    if prefer > 0:
        for freq, _energy in feasible:
            if int(freq) == prefer:
                return prefer
    return min(feasible, key=lambda x: x[1])[0]


@dataclass
class Decision:
    """调度器每步决策。"""
    mode: str
    freq_mhz: int
    released: List[int] = field(default_factory=list)
    window_end_s: float = 0.0


@dataclass
class _Buffered:
    rid: int
    arrival_s: float
    prompt_len: int
    output_len: int

class PaDGWindowScheduler:
    """mixed 池在线调度器(时间窗分离)。纯逻辑,时钟/模型可注入。"""

    def __init__(self, opmodel, config: SystemConfig, clock=time.time,
                 mixed_capacity_req_s: float = 10.0,
                 prefill_token_budget: int = TOKEN_BUDGET,
                 lam_alpha: float = 0.3):
        self.opmodel = opmodel
        self.cfg = config
        self._clock = clock
        self.capacity = float(mixed_capacity_req_s)
        self.prefill_token_budget = prefill_token_budget
        self.lam_alpha = lam_alpha
        self.buffer: Deque[_Buffered] = deque()
        self.active: Dict[int, _Buffered] = {}
        self._tok: Dict[int, Tuple[float, int]] = {}  # rid → (首 token 时刻, n)
        self.mode: str = MODE_PREFILL_WINDOW
        self.freq: int = int(config.prefill_freq)
        self.window_end_s: float = 0.0
        self.lam_hat: float = 0.0
        self._last_step: Optional[float] = None
        self._arrivals_since: int = 0

    def submit(self, rid: int, arrival_s: float, prompt_len: int,
               output_len: int) -> None:
        if self._last_step is None:
            self._last_step = float(arrival_s)  # 基准时刻取首个到达
        self.buffer.append(_Buffered(rid=rid, arrival_s=float(arrival_s),
                                     prompt_len=int(prompt_len),
                                     output_len=int(output_len)))
        self._arrivals_since += 1

    def _update_rate(self, now: float) -> None:
        if self._last_step is None:
            self._last_step = now
            return
        dt = now - self._last_step
        if dt > 0:
            inst = self._arrivals_since / dt
            self.lam_hat = (self.lam_alpha * inst
                            + (1.0 - self.lam_alpha) * self.lam_hat)
        self._arrivals_since = 0
        self._last_step = now

    def _overload(self) -> bool:
        return self.lam_hat >= 0.95 * self.capacity

    def _buffered_tokens(self) -> int:
        return sum(b.prompt_len for b in self.buffer)

    def _predicted_prefill_s(self) -> float:
        # 窗内积压 token 合计:容量口径,传批 Σtok 而非单条 L
        toks = self._buffered_tokens()
        return max(1e-3, self.opmodel.prefill_time_ms(toks) / 1000.0)

    def _b_hat(self) -> int:
        return max(len(self.active), len(self.buffer), 1)

    def _ctx_hat(self) -> float:
        if self.active:
            return float(sum(b.prompt_len + b.output_len / 2.0
                             for b in self.active.values()) / len(self.active))
        if self.buffer:
            return float(sum(b.prompt_len + b.output_len / 2.0
                             for b in self.buffer) / len(self.buffer))
        return CTX_REF

    def _out_hat(self) -> float:
        pool = list(self.active.values()) or list(self.buffer)
        if not pool:
            return OUT_REF
        return float(sum(b.output_len for b in pool) / len(pool))

    def _decode_freq(self, now: Optional[float] = None) -> int:
        # 吞吐下限:λ̂ × L̂o token/s 的需求按 ρ 预算 0.7(实测膝点 0.7245)
        # 折算成必须的服务速率
        min_tok = self.lam_hat * self._out_hat() / RHO_TPUT
        slo = self.cfg.slo.tpot_s
        budget_ms = slo * 1000.0 * (1.0 - self.cfg.tpot_margin)  # 静态回退
        if now is not None:
            s = self.pool_slack(now)
            if s is not None:
                # slack 驱动:最紧请求盈余 s → 下一步预算 (s+TPOT)×0.8,
                # 上限 3×TPOT 防单步过长(EcoServe 记账思想)
                budget_ms = min((s + slo) * 0.8, 3.0 * slo) * 1000.0
        demand = self.lam_hat * self._out_hat() / RHO_TPUT
        floor = int(getattr(self.cfg, "decode_floor_mhz", 0) or 0)
        cands = tuple(f for f in self.cfg.freq_candidates
                      if floor <= 0 or int(f) >= floor)
        if not cands:
            cands = (max(int(f) for f in self.cfg.freq_candidates),)
        prefer = int(getattr(self.cfg, "lookup_decode_freq", 0) or 0)
        return select_decode_freq(self.opmodel, self._b_hat(), self._ctx_hat(),
                                  self.cfg.slo.tpot_s, cands,
                                  margin=self.cfg.tpot_margin,
                                  min_tok_per_s=min_tok,
                                  budget_ms_override=budget_ms,
                                  demand_tok_per_s=demand,
                                  b_cap=DECODE_B_CAP,
                                  prefer_mhz=prefer)

    def _release(self, now: float, max_release: Optional[int],
                 token_budget: Optional[int] = None) -> List[int]:
        out: List[int] = []
        toks = 0
        budget = (token_budget if token_budget is not None
                  else self.prefill_token_budget)
        while self.buffer and (max_release is None or len(out) < max_release):
            b = self.buffer[0]
            if toks + b.prompt_len > budget and out:
                break
            self.buffer.popleft()
            out.append(b.rid)
            toks += b.prompt_len
            self.active[b.rid] = b
        return out

    def on_token(self, rid: int, now: float) -> None:
        """逐 token 回调(EcoServe 式 slack 记账):记首 token 时刻与产出数。"""
        rid = int(rid)
        first, n = self._tok.get(rid, (float(now), 0))
        self._tok[rid] = (first, n + 1)

    def pool_slack(self, now: float) -> Optional[float]:
        """池内最紧请求的节奏盈余(s):slack_i = first_i + n_i×TPOT − now。

        正 = 领先 SLO 节奏(可容忍更慢迭代),负 = 落后(需提速)。
        无产出请求(TTFT 阶段,预算宽)不计。
        """
        slo = self.cfg.slo.tpot_s
        vals = [first + n * slo - now
                for rid, (first, n) in self._tok.items()
                if rid in self.active and n > 0]
        return min(vals) if vals else None

    def complete(self, rid: int) -> None:
        """请求完成(引擎侧流出结束):从 active 移除,供 b_hat 估计。"""
        self.active.pop(int(rid), None)
        self._tok.pop(int(rid), None)

    def _leftover_att_ok(self) -> bool:
        """leftover prefill 会否把预测 att 压到 att_mixed 以下。"""
        from ecopadg.constraints import att_would_drop
        if not self.buffer:
            return True
        leftover_s = self._predicted_prefill_s()
        if leftover_s > self.cfg.slo.ttft_s:
            return False
        frac = leftover_s / max(self.cfg.slo.ttft_s, 1e-9)
        if frac <= 0.8:
            return True
        att_hat = max(0.0, 1.0 - frac)
        att_mixed = float(getattr(self.cfg, "baseline_att", float("nan")) or 1.0)
        if att_mixed != att_mixed:
            att_mixed = float(getattr(self.cfg, "target_attainment", 1.0))
        return not att_would_drop(
            att_hat, att_mixed,
            getattr(self.cfg, "slo_non_inferior_pp", 0.01))

    def _stay_continuous(self) -> bool:
        if getattr(self.cfg, "force_continuous", False):
            return True
        if self.cfg.slo.ttft_s < 2.0:
            return True
        return self._overload()

    def step(self, now: float, max_release: Optional[int] = None) -> Decision:
        """周期决策:返回 (mode, freq, released)。"""
        self._update_rate(now)
        if self._stay_continuous():
            self.mode = MODE_CONTINUOUS
            # leftover 威胁同负载 att:禁止再降频
            if not self._leftover_att_ok():
                self.freq = max(int(f) for f in self.cfg.freq_candidates)
            else:
                self.freq = self._decode_freq(now)
            rel = self._release(now, max_release, token_budget=None)
            return Decision(mode=self.mode, freq_mhz=self.freq, released=rel,
                            window_end_s=now + self.cfg.global_period_s)
        if self.mode == MODE_CONTINUOUS:
            self.mode = MODE_PREFILL_WINDOW
            self.window_end_s = now
        if self.mode == MODE_DECODE_WINDOW:
            if self.buffer:
                self.mode = MODE_PREFILL_WINDOW
            else:
                self.freq = self._decode_freq(now)
                return Decision(mode=self.mode, freq_mhz=self.freq,
                                released=[], window_end_s=self.window_end_s)
        if self.mode == MODE_PREFILL_WINDOW:
            if now >= self.window_end_s:
                if not self.buffer:
                    if not self._leftover_att_ok():
                        self.freq = int(self.cfg.prefill_freq)
                        rel = self._release(now, max_release)
                        return Decision(
                            mode=self.mode, freq_mhz=self.freq, released=rel,
                            window_end_s=now + self.cfg.window_min_s)
                    self.mode = MODE_DECODE_WINDOW
                    self.freq = self._decode_freq(now)
                    self.window_end_s = now + self.cfg.window_max_s
                    return Decision(mode=self.mode, freq_mhz=self.freq,
                                    released=[], window_end_s=self.window_end_s)
                pre_s = max(self.cfg.window_min_s, self._predicted_prefill_s())
                self.window_end_s = now + min(pre_s, self.cfg.window_max_s)
            self.freq = int(self.cfg.prefill_freq)
            rel = self._release(now, max_release)
            return Decision(mode=self.mode, freq_mhz=self.freq, released=rel,
                            window_end_s=self.window_end_s)
        self.mode = MODE_CONTINUOUS
        self.freq = self._decode_freq(now)
        rel = self._release(now, max_release, token_budget=None)
        return Decision(mode=self.mode, freq_mhz=self.freq, released=rel,
                        window_end_s=now + self.cfg.global_period_s)
