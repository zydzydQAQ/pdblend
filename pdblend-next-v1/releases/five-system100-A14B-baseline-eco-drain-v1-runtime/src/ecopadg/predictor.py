# -*- coding: utf-8 -*-
"""predictor:Partition × λ̂ → Prediction(可行/attainment/功率)解析预测(M4)。

设计原则:预测器的职责是正确排序候选分区,绝对数值由 PE1 校准:
  - 容量:mixed 实例 = 流水化 prefill+decode 服务时间;段 = min(P 速率, D 速率);
  - attainment:ρ = λ/μ 的分段线性膝形(rho_knee 前≈att_high,之后线性跌落),
    膝点与上限由 PE1 实测校准;
  - 功率:逐池 busy 分数 ×(相位功率 - 驻留)+ 驻留 + 空卡 idle(整机口径,
    E0/P5:空卡 ~35W、驻留(载权重)~76W/卡、卡间加性成立)。
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Sequence, Tuple

from ecopadg.global_scheduler import Prediction
from ecopadg.online_scheduler import select_decode_freq
from ecopadg.types import B_REF, CTX_REF, OUT_REF, TOKEN_BUDGET, Partition, SloSpec


def prefill_token_pack(prompt_len: float,
                       token_budget: int = TOKEN_BUDGET) -> Tuple[int, int]:
    """DistServe FCFS token 批:(Σtok, n_req)。

    单条 L>=B:独占整条(pack=L, n_req=1)。否则 n_req=floor(B/L),
    pack=n_req*L。容量路径把 pack 传给 prefill_time_ms;单请求 TTFT
    仍传该条 L。
    """
    length = max(int(round(float(prompt_len))), 1)
    budget = max(int(token_budget), 1)
    if length >= budget:
        return length, 1
    n_req = max(budget // length, 1)
    return n_req * length, n_req


@dataclass
class WorkloadStats:
    """负载画像(滑窗统计或 trace 先验)。"""
    prompt_len: float = 600.0
    output_len: float = 200.0
    frac_long: float = 0.3        # prompt >= pd_prompt_threshold 的比例


class ApproxPredictor:
    """OpModel 支撑的解析预测器(GlobalScheduler predictor 接口)。"""

    def __init__(self, opmodel, stats: WorkloadStats, slo: SloSpec,
                 freq_candidates: Sequence[int],
                 gpu_idle_w: float = 35.0,
                 pd_prompt_threshold: int = 512,
                 b_ref: int = B_REF,
                 rho_knee: float = 0.7,
                 att_high: float = 0.98,
                 rho_max: float = 0.85,
                 att_target: float = 0.9,
                 tpot_margin: float = 0.35,
                 mixed_scale: float = 1.0,
                 seg_scale: float = 1.0,
                 att_high_mixed: Optional[float] = None,
                 att_high_seg: Optional[float] = None,
                 prefill_freq: Optional[int] = None,
                 token_budget: int = TOKEN_BUDGET):
        self.opmodel = opmodel
        self.stats = stats
        self.slo = slo
        self.freqs = tuple(int(f) for f in freq_candidates)
        self.gpu_idle_w = float(gpu_idle_w)
        self.pd_prompt_threshold = int(pd_prompt_threshold)
        self.b_ref = int(b_ref)
        self.rho_knee = float(rho_knee)     # PE1 校准点
        self.att_high = float(att_high)     # PE1 校准点
        self.rho_max = float(rho_max)       # 可行上限(利用率护栏)
        self.att_target = float(att_target)  # 可行下限(预测 attainment)
        self.tpot_margin = float(tpot_margin)
        # PE1 校准系数:μ_实测/μ_模型(解析模型只保证排序,绝对容量靠实测校准)
        self.mixed_scale = float(mixed_scale)
        self.seg_scale = float(seg_scale)
        # attainment 结构性天花板(按池类型,PE1 实测):膝形模型假设低 ρ 即
        # att_high,但 longbench 类 prefill 重负载下 mixed 池存在与 ρ 无关的
        # 干扰上限(实测 ~0.895 < 0.9),不校准会误选注定不达标的 mixed 布局
        self.att_high_mixed = float(att_high_mixed
                                    if att_high_mixed is not None
                                    else self.att_high)
        self.att_high_seg = float(att_high_seg if att_high_seg is not None
                                  else self.att_high)
        self.prefill_freq = int(prefill_freq or max(self.freqs))
        self.token_budget = int(token_budget)

    # ------------------------------------------------------------------
    # 分级容量(req/s)
    # ------------------------------------------------------------------

    def _decode_freq(self) -> int:
        """运行时 slack 选频(功率预测用)。容量口径一律用最高频:
        DVFS 是可行域内的省能手段,不得压低容量/可行性判断本身。"""
        return int(select_decode_freq(
            self.opmodel, self.b_ref,
            self.stats.prompt_len + self.stats.output_len / 2.0,
            self.slo.tpot_s, self.freqs, margin=self.tpot_margin))

    def _pe1_scale(self, freq: int, kind: str) -> float:
        """PE1 μ 定标只乘校准频率(档顶)。降频候选用未放大的解析 μ。"""
        cal = int(max(self.freqs)) if self.freqs else 2520
        if int(freq) < cal:
            return 1.0
        return self.mixed_scale if kind == "mixed" else self.seg_scale

    def rate_mixed(self, n: int, freq: Optional[int] = None,
                   prefill_freq: Optional[int] = None) -> float:
        """mixed 实例池速率。缺省满频(容量/过载护栏);planner 传入分区频。"""
        if n <= 0:
            return 0.0
        f = int(freq if freq is not None else max(self.freqs))
        fp = int(prefill_freq if prefill_freq is not None
                 else self.prefill_freq)
        ctx = self.stats.prompt_len + self.stats.output_len / 2.0
        it_ms = self.opmodel.iter_time_ms(self.b_ref, f, ctx)
        # mixed 单请求墙钟:TTFT 用该条 L,不是 P 池组批 Σtok
        pre_ms = self.opmodel.prefill_time_ms(int(self.stats.prompt_len), fp)
        per_req_s = (pre_ms + self.stats.output_len * it_ms) / 1000.0
        return self._pe1_scale(f, "mixed") * n * self.b_ref / max(
            per_req_s, 1e-9)

    def stage_rate_prefill(self, n: int, freq: Optional[int] = None) -> float:
        """P 池速率:每实例按 TOKEN_BUDGET 组批。

        Σtok_pack, n_req = prefill_token_pack(L̄, B);
        μ = n × n_req / T(Σtok_pack, f)。单请求 TTFT 不走本函数。
        满频乘 seg_scale;降频不套 PE1(避免 r=3.4 把 1500 当成满载 μ)。
        """
        if n <= 0:
            return 0.0
        fp = int(freq if freq is not None else self.prefill_freq)
        pack, n_req = prefill_token_pack(self.stats.prompt_len,
                                         self.token_budget)
        pre_s = self.opmodel.prefill_time_ms(int(pack), fp) / 1000.0
        return self._pe1_scale(fp, "seg") * n * n_req / max(pre_s, 1e-9)

    def stage_rate_decode(self, n: int, freq: Optional[int] = None) -> float:
        """D 池速率。缺省满频;planner 传入分区 decode 频。"""
        if n <= 0:
            return 0.0
        f = int(freq if freq is not None else max(self.freqs))
        ctx = self.stats.prompt_len + self.stats.output_len / 2.0
        it_s = self.opmodel.iter_time_ms(self.b_ref, f, ctx) / 1000.0
        per_req_s = self.stats.output_len * it_s
        return self._pe1_scale(f, "seg") * n * self.b_ref / max(
            per_req_s, 1e-9)

    def rate_pd(self, n_p: int, n_d: int,
                freq_p: Optional[int] = None,
                freq_d: Optional[int] = None) -> float:
        """两阶段吞吐:min(μ_P, μ_D),允许 n_P ≠ n_D。"""
        return min(self.stage_rate_prefill(n_p, freq=freq_p),
                   self.stage_rate_decode(n_d, freq=freq_d))

    def rate_segments(self, pairs: int) -> float:
        return self.rate_pd(pairs, pairs)

    def capacity(self, p: Partition) -> float:
        """用分区频率估 μ;与 predict() 一致。"""
        fm = int(getattr(p, "freq_mixed", 0) or 0) or None
        fp = int(getattr(p, "freq_prefill", 0) or 0) or None
        fd = int(getattr(p, "freq_decode", 0) or 0) or None
        return (self.rate_mixed(p.n_mixed, freq=fm, prefill_freq=fm)
                + self.rate_pd(p.n_prefill, p.n_decode, freq_p=fp, freq_d=fd))

    # ------------------------------------------------------------------
    # 负载切分:仅当 mixed 与 PD 同时存在(A9 双池)才按长度分流
    # ------------------------------------------------------------------

    def split_lambda(self, p: Partition, lam: float) -> Tuple[float, float]:
        """(λ_mixed, λ_pd):按 frac_long 切分;单池承接全部。"""
        has_m = p.n_mixed > 0
        has_pd = p.pairs() > 0
        if has_m and has_pd:
            lam_pd = lam * self.stats.frac_long
            return lam - lam_pd, lam_pd
        if has_m:
            return lam, 0.0
        if has_pd:
            return 0.0, lam
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # attainment 膝形近似
    # ------------------------------------------------------------------

    def _attainment(self, rho: float, high: Optional[float] = None) -> float:
        h = self.att_high if high is None else float(high)
        if rho <= self.rho_knee:
            return h
        if rho >= 1.0:
            return 0.0
        return h * (1.0 - rho) / (1.0 - self.rho_knee)

    def rho_at_target(self) -> float:
        """att(ρ)=att_target 的利用率(膝形反解):μ_S90 = ρ_t × μ_cap。"""
        if self.att_target >= self.att_high:
            return self.rho_knee
        return 1.0 - self.att_target * (1.0 - self.rho_knee) / self.att_high

    # ------------------------------------------------------------------
    # 功率(整机口径:忙时相位功率 + 驻留 + 空卡 idle)
    # ------------------------------------------------------------------

    def _phase_power(self, phase: str, freq: int, tp: int) -> float:
        """每实例相位功率(W);无实测表时用驻留+经验增量兜底。"""
        p = None
        power_fn = getattr(self.opmodel, "power_w", None)
        if power_fn is not None:
            try:
                v = float(power_fn(phase, freq))
                if v == v:  # not NaN
                    p = v
            except (ValueError, TypeError):
                p = None
        if p is None:
            res = self._residency_w()
            # 兜底:忙时较驻留增量 ~55W/卡(P5 满载 130 vs 驻留 76)
            p = res + 55.0 * tp
        return p

    def _residency_w(self) -> float:
        fn = getattr(self.opmodel, "residency_w", None)
        if fn is None:
            return 76.0
        try:
            v = float(fn())
            return v if v == v and v > 0 else 76.0
        except (ValueError, TypeError):
            return 76.0

    def _busy_occ(self, lam_inst: float, t_req_s: float) -> float:
        """decode 存在性占空(M/G/∞ 占用率):busy = 1 − e^(−λ_inst×T_req)。

        continuous batching 下只要实例上有 ≥1 活跃请求就整卡忙碌;
        旧口径 busy=ρ 系统性低估功率(实测 4×mixed r1.2 1969W,
        并发 ~2.9/实例 → busy≈0.94,而 ρ 口径只给 0.22)。"""
        c = max(lam_inst, 0.0) * max(t_req_s, 0.0)
        return 1.0 - math.exp(-c)

    def _t_req_mixed(self, freq: Optional[int] = None,
                     prefill_freq: Optional[int] = None) -> float:
        """mixed 单请求墙钟时长(s):prefill + Lo 次迭代。"""
        f = int(freq if freq is not None else self._decode_freq())
        fp = int(prefill_freq if prefill_freq is not None
                 else self.prefill_freq)
        ctx = self.stats.prompt_len + self.stats.output_len / 2.0
        it_s = self.opmodel.iter_time_ms(self.b_ref, f, ctx) / 1000.0
        # 单请求 TTFT(该条 L),与容量路径的批 Σtok 分开
        pre_s = self.opmodel.prefill_time_ms(int(self.stats.prompt_len),
                                             fp) / 1000.0
        return pre_s + self.stats.output_len * it_s

    def _t_req_decode(self, freq: Optional[int] = None) -> float:
        f = int(freq if freq is not None else self._decode_freq())
        ctx = self.stats.prompt_len + self.stats.output_len / 2.0
        it_s = self.opmodel.iter_time_ms(self.b_ref, f, ctx) / 1000.0
        return self.stats.output_len * it_s

    def _pool_power(self, n_inst: int, tp: int, phase: str, freq: int,
                    busy: float) -> float:
        if n_inst <= 0:
            return 0.0
        busy = min(max(busy, 0.0), 1.0)
        p_busy = self._phase_power(phase, freq, tp)
        p_idle = self._residency_w()
        return n_inst * (busy * p_busy + (1.0 - busy) * p_idle)

    # ------------------------------------------------------------------
    # GlobalScheduler predictor 接口
    # ------------------------------------------------------------------

    def predict(self, p: Partition, lam_hat: float, state=None) -> Prediction:
        cap = self.capacity(p)
        idle = self.gpu_idle_w * p.gpus_total
        if cap <= 0:
            return Prediction(attainable=False, attainment=0.0,
                              power_w=idle, energy_j_per_s=idle,
                              energy_j_per_req=float("inf"), goodput_rps=0.0)
        lam_m, lam_pd = self.split_lambda(p, float(lam_hat))
        f_m = int(getattr(p, "freq_mixed", 0) or max(self.freqs))
        f_p = int(getattr(p, "freq_prefill", 0) or self.prefill_freq)
        f_d = int(getattr(p, "freq_decode", 0) or max(self.freqs))
        mu_m = self.rate_mixed(p.n_mixed, freq=f_m, prefill_freq=f_m)
        mu_p = self.stage_rate_prefill(p.n_prefill, freq=f_p)
        mu_d = self.stage_rate_decode(p.n_decode, freq=f_d)
        mu_pd = min(mu_p, mu_d) if (mu_p > 0 or mu_d > 0) else 0.0
        rho_m = lam_m / mu_m if mu_m > 0 else (1.0 if lam_m > 0 else 0.0)
        rho_p = lam_pd / mu_p if mu_p > 0 else (1.0 if lam_pd > 0 else 0.0)
        rho_d = lam_pd / mu_d if mu_d > 0 else (1.0 if lam_pd > 0 else 0.0)
        rho_pd = max(rho_p, rho_d)
        rho = max(rho_m, rho_pd)
        lam_tot = max(lam_m + lam_pd, 1e-9)
        att_pd = min(self._attainment(rho_p, self.att_high_seg),
                     self._attainment(rho_d, self.att_high_seg))
        pre_s = self.opmodel.prefill_time_ms(
            int(self.stats.prompt_len),
            f_p if p.n_prefill else f_m) / 1000.0
        # 单请求 TTFT 硬约束:组批排队另由 ρ 膝形覆盖
        if pre_s > float(self.slo.ttft_s):
            if p.n_prefill:
                att_pd = 0.0
            else:
                att_pd = att_pd
                # mixed 单请求已破 TTFT
        att = (self._attainment(rho_m, self.att_high_mixed) * lam_m
               + att_pd * lam_pd) / lam_tot
        if pre_s > float(self.slo.ttft_s) and not p.n_prefill:
            att = 0.0
        attainable = (rho < self.rho_max) and pre_s <= float(self.slo.ttft_s)
        f_dec = f_d
        power = 0.0
        if p.n_mixed > 0:
            share_p = (min(1.0, lam_m * pre_s / p.n_mixed)
                       if lam_m > 0 else 0.0)
            p_busy_m = (share_p
                        * self._phase_power("prefill", f_m, p.tp_mixed)
                        + (1.0 - share_p)
                        * self._phase_power("decode", f_m, p.tp_mixed))
            busy_m = self._busy_occ(
                lam_m / p.n_mixed, self._t_req_mixed(freq=f_m, prefill_freq=f_m))
            power += p.n_mixed * (busy_m * p_busy_m
                                  + (1.0 - busy_m) * self._residency_w())
        mu_p_raw = (self.stage_rate_prefill(p.n_prefill, freq=f_p)
                    if p.n_prefill > 0 else 0.0)
        # 占空用未放大 μ,避免 PE1 scale 把 busy 压低
        busy_scale = self._pe1_scale(f_p, "seg")
        mu_p_occ = mu_p_raw / busy_scale if busy_scale > 0 else mu_p_raw
        busy_p = (min(1.0, lam_pd / mu_p_occ)
                  if mu_p_occ > 0 and lam_pd > 0 else 0.0)
        power += self._pool_power(p.n_prefill, p.tp_prefill, "prefill",
                                  f_p, busy_p)
        busy_d = (self._busy_occ(lam_pd / p.n_decode,
                                 self._t_req_decode(freq=f_d))
                  if p.n_decode > 0 else 0.0)
        power += self._pool_power(p.n_decode, p.tp_decode, "decode",
                                  f_dec, busy_d)
        unused = p.gpus_total - p.total_gpus()
        power += max(unused, 0) * self.gpu_idle_w
        # SLO-goodput 代理:P×T_makespan ≡ P / goodput(每达标请求焦耳)
        goodput = min(float(lam_hat), cap) * float(att)
        e_per_req = float(power) / max(goodput, 1e-9)
        energy_rate = e_per_req * max(float(lam_hat), 1e-9)
        return Prediction(attainable=attainable, attainment=float(att),
                          power_w=float(power),
                          energy_j_per_s=float(energy_rate),
                          energy_j_per_req=float(e_per_req),
                          goodput_rps=float(goodput),
                          rho=float(rho))

    __call__ = predict


@dataclass
class CalibrationResult:
    """PE1 定标结果。缺 100ms μ_S90 时不静默回退 200ms。"""
    mixed_scale: float
    seg_scale: float
    att_high_mixed: float
    att_high_seg: float
    calibrated: bool = True
    note: str = ""
    rank_spearman: Optional[float] = None

    def __iter__(self):
        yield self.mixed_scale
        yield self.seg_scale
        yield self.att_high_mixed
        yield self.att_high_seg


def calibration_scales(pred: "ApproxPredictor", grid: dict, model_key: str,
                       n_mixed_pe1: int, n_pairs_pe1: int
                       ) -> CalibrationResult:
    """由 PE1 rate_grid.json 反推容量校准系数与 attainment 天花板。

    只读 100ms 的 μ_S90。缺测标 uncalibrated,scale=1,禁止静默用 200ms 档。
    另报 mixed vs ds_pd 的保序 Spearman(两点同序=+1,反序=−1)。
    """
    rho_t = max(pred.rho_at_target(), 1e-6)
    g = grid.get(model_key, {}) if grid else {}

    def _mu100(sysn):
        rec = g.get(sysn) or {}
        try:
            v = float(rec.get("mu_s90"))
            if v == v and v > 0:
                return v
        except (TypeError, ValueError):
            pass
        return None

    def _ceiling(sysn, default):
        pts = (g.get(sysn) or {}).get("points") or []
        vals = []
        for p in pts:
            try:
                v = float(p.get("slo"))
                if v == v:
                    vals.append(v)
            except (TypeError, ValueError):
                continue
        return min(max(vals), default) if vals else default

    notes = []
    mixed_scale = 1.0
    mu_m = _mu100("mixed")
    base_m = pred.rate_mixed(n_mixed_pe1)
    if mu_m is not None and base_m > 0:
        mixed_scale = (mu_m / rho_t) / base_m
    else:
        notes.append("uncalibrated-missing-mu_s90-mixed")
    seg_scale = 1.0
    mu_s = _mu100("ds_pd")
    base_s = pred.rate_pd(n_pairs_pe1, n_pairs_pe1)
    if mu_s is not None and base_s > 0:
        seg_scale = (mu_s / rho_t) / base_s
    else:
        notes.append("uncalibrated-missing-mu_s90-ds_pd")
        if mu_m is not None:
            seg_scale = mixed_scale
            notes.append("seg-scale-copied-from-mixed")
    att_m = _ceiling("mixed", pred.att_high)
    att_s = _ceiling("ds_pd", pred.att_high)
    spearman = None
    if mu_m is not None and mu_s is not None and base_m > 0 and base_s > 0:
        da, db = base_m - base_s, mu_m - mu_s
        if da == 0 or db == 0:
            spearman = 0.0
            notes.append("rank-unknown-tied-mu")
        else:
            spearman = 1.0 if da * db > 0 else -1.0
        if spearman < 0:
            notes.append("rank-inversion")
    calibrated = not any(n.startswith("uncalibrated") for n in notes)
    return CalibrationResult(
        mixed_scale, seg_scale, att_m, att_s,
        calibrated=calibrated, note=";".join(notes) or "ok",
        rank_spearman=spearman)


def default_pe1_grid_path(ws: Optional[str] = None) -> str:
    """与控制器同一口径:script/bench/results/pe1_capacity/rate_grid.json。"""
    if os.environ.get("PDBLEND_PE1_GRID"):
        return str(os.environ["PDBLEND_PE1_GRID"])
    root = ws or os.environ.get("PDBLEND_WS", "/workspace")
    return os.path.join(root, "pdblend", "script", "bench",
                        "results", "pe1_capacity", "rate_grid.json")


def evaluate_capacity_trust(pred: "ApproxPredictor", model_key: str,
                            gpu_count: int, tp: int,
                            grid_path: Optional[str] = None,
                            freq_trusted: bool = True):
    """读 PE1 网格并判定容量模型是否可信。缺文件 → (False, None)。

    不可信时必须保持 trusted=False:未校准预测 att 常为 0,若打开
    att-then-min-J,低载会误选 mixed、丢掉空间 PD。
    """
    path = grid_path or default_pe1_grid_path()
    if not os.path.isfile(path):
        return False, None
    try:
        with open(path, encoding="utf-8") as fh:
            grid = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False, None
    n_mixed = max(int(gpu_count) // max(int(tp), 1), 1)
    n_pairs = max(int(gpu_count) // max(2 * int(tp), 1), 1)
    cres = calibration_scales(
        pred, grid, model_key, n_mixed, n_pairs)
    trusted = capacity_model_trustworthy(
        bool(cres.calibrated), cres.rank_spearman, freq_trusted)
    return trusted, cres


def capacity_model_trustworthy(pe1_ok: bool,
                               rank_spearman: Optional[float] = None,
                               freq_trusted: bool = True) -> bool:
    """缺 μ_S90、保序未知/反序或频率不可分 → 容量/att 排序不可信。

    14B 合成表 + 无 PE1 膝点时必须为 False,禁止用预测 att=0 否决 PD。
    """
    if not pe1_ok:
        return False
    if rank_spearman is None:
        return False
    try:
        rs = float(rank_spearman)
    except (TypeError, ValueError):
        return False
    if rs != rs or rs <= 0:
        return False
    return bool(freq_trusted)


@dataclass
class _Arrival:
    t: float
    prompt_len: int
    output_len: int


class LoadMonitor:
    """多时间尺度负载监测:1s 驱动选频/unpark,30s 驱动 park。"""

    def __init__(self, window_s: float = 30.0,
                 fast_window_s: float = 1.0,
                 pd_prompt_threshold: int = 512):
        self.window_s = float(window_s)
        self.fast_window_s = float(fast_window_s)
        self.pd_prompt_threshold = int(pd_prompt_threshold)
        self._buf: Deque[_Arrival] = deque()

    def on_arrival(self, t: float, prompt_len: int, output_len: int) -> None:
        self._buf.append(_Arrival(float(t), int(prompt_len), int(output_len)))

    def _count(self, now: float, window_s: float) -> int:
        lo = now - window_s
        n = 0
        for a in reversed(self._buf):
            if a.t < lo:
                break
            n += 1
        while self._buf and self._buf[0].t < now - self.window_s:
            self._buf.popleft()
        return n

    def rate(self, now: float, window_s: Optional[float] = None) -> float:
        w = float(self.window_s if window_s is None else window_s)
        n = self._count(now, w)
        return n / max(w, 1e-9)

    def rate_fast(self, now: float) -> float:
        return self.rate(now, self.fast_window_s)

    def reset(self) -> None:
        """档间隔离:清滑窗,避免上一档 λ/CV 泄漏。"""
        self._buf.clear()

    def arrival_cv(self, now: float) -> float:
        """当前长窗到达间隔的变异系数。不足 3 个到达 → 0。"""
        self._count(now, self.window_s)
        ts = [a.t for a in self._buf if a.t >= now - self.window_s]
        if len(ts) < 3:
            return 0.0
        gaps = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        mean = sum(gaps) / len(gaps)
        if mean <= 1e-12:
            return 0.0
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return math.sqrt(var) / mean

    def stats(self, now: float,
              default: Optional[WorkloadStats] = None) -> WorkloadStats:
        self._count(now, self.window_s)
        if not self._buf:
            return default or WorkloadStats()
        n = len(self._buf)
        return WorkloadStats(
            prompt_len=sum(a.prompt_len for a in self._buf) / n,
            output_len=sum(a.output_len for a in self._buf) / n,
            frac_long=sum(1 for a in self._buf
                          if a.prompt_len >= self.pd_prompt_threshold) / n)
