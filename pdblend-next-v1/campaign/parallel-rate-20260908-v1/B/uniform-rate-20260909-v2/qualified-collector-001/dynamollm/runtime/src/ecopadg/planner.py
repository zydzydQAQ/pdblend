# -*- coding: utf-8 -*-
"""同负载 planner:对给定 λ 过滤 att≥att_mixed−δ 后取 min gross J。

默认目标 min_j_s.t. slo_non_inferior。max-rate 仅 --mode capacity-envelope。
合并原 placement 枚举与 choose_partition 打分,禁止两套优化函数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from ecopadg.metrics import slo_non_inferior
from ecopadg.placement import bisect_rate, distserve_configs
from ecopadg.types import (
    DECODE_B_CAP, DEFAULT_FREQS, ENERGY_RANK_MAX_J_PER_S, GOODPUT_ATT_GATE,
    LOAD_CAP_PER_INST, PLANNER_MODE_CAPACITY, PLANNER_MODE_ISO_LOAD,
    PREFILL_PIN_MHZ, SHRINK_ATT_MARGIN_PP, SHRINK_RHO_MAX,
    SLO_NON_INFERIOR_PP, TOKEN_BUDGET, Partition,
)

LAYOUT_MIXED = "mixed"
LAYOUT_SPATIAL_PD = "spatial_pd"

# 不可信路径:λ < α·μ_P 且 λ < α·μ_D 且单请求 prefill ≤ S_TTFT 才起纯空间 PD。
# 同一 α,两个瓶颈 AND。μ_D 非有限 → fail-closed 回 mixed,不猜。
# 形状感知:长 prefill 早关、短请求由 decode 占用晚关。
# SLO 只进 slack/盾,不进 lookup,不决定布局。
# 不可信 A9:仅当 frac_long∈[L,H] 且分流后 2×mixed 与 1P1D 都装得下。
# α 低于 predictor.rho_knee(0.7),给排队留余量。
# 默认路径满卡 {1×TP8, winner PD, 可选 A9};ScaleInst 只留给 dynamollm。
PD_PREFILL_RHO_GUARD = 0.50
HYBRID_FRAC_LONG_LO = 0.15
HYBRID_FRAC_LONG_HI = 0.85
# 混长检测键:跨 trace 同一 token 门槛,不是该 trace 的 mean pin。
# mean pin 会把单峰短/长分布都切成 ~0.5,误开 A9。
# 路由器分流仍用 default_pd_threshold(mean pin)。
HYBRID_LONG_PROMPT = 512


def mix_frac_long(prompt_lens) -> float:
    """prompt >= HYBRID_LONG_PROMPT 的比例。Alpaca→0, LongBench→1。"""
    pins = list(prompt_lens or [])
    n = max(len(pins), 1)
    return sum(1 for pin in pins if float(pin) >= HYBRID_LONG_PROMPT) / float(n)


@dataclass
class PlanScore:
    partition: Partition
    attainment: float
    energy_j_per_s: float
    energy_j_per_req: float
    power_w: float
    attainable: bool
    accepted: bool
    reason: str = ""


def _guard_num(value: float):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return round(x, 4)


@dataclass
class CapacityGuard:
    """不可信启动:预填 ρ 护栏,不用预测 att。"""
    mu_prefill: float = float("nan")
    mu_mixed: float = float("nan")
    mu_decode: float = float("nan")
    rho_prefill: float = float("nan")
    rho_decode: float = float("nan")
    alpha: float = PD_PREFILL_RHO_GUARD
    pre_s: float = float("nan")
    pd_ok: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(
            mu_prefill=_guard_num(self.mu_prefill),
            mu_mixed=_guard_num(self.mu_mixed),
            mu_decode=_guard_num(self.mu_decode),
            rho_prefill=_guard_num(self.rho_prefill),
            rho_decode=_guard_num(self.rho_decode),
            alpha=float(self.alpha),
            pre_s=_guard_num(self.pre_s),
            pd_ok=bool(self.pd_ok),
            reason=str(self.reason or ""),
        )


@dataclass
class HybridGuard:
    """不可信 A9:长短分流后两池都装得下。PD 半池同样 AND μ_D。"""
    frac_long: float = float("nan")
    lam_mixed: float = float("nan")
    lam_pd: float = float("nan")
    mu_mixed: float = float("nan")
    mu_prefill: float = float("nan")
    mu_decode: float = float("nan")
    rho_mixed: float = float("nan")
    rho_prefill: float = float("nan")
    rho_decode: float = float("nan")
    alpha: float = PD_PREFILL_RHO_GUARD
    ok: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(
            frac_long=_guard_num(self.frac_long),
            lam_mixed=_guard_num(self.lam_mixed),
            lam_pd=_guard_num(self.lam_pd),
            mu_mixed=_guard_num(self.mu_mixed),
            mu_prefill=_guard_num(self.mu_prefill),
            mu_decode=_guard_num(self.mu_decode),
            rho_mixed=_guard_num(self.rho_mixed),
            rho_prefill=_guard_num(self.rho_prefill),
            rho_decode=_guard_num(self.rho_decode),
            alpha=float(self.alpha),
            ok=bool(self.ok),
            reason=str(self.reason or ""),
        )


@dataclass
class PlanResult:
    mode: str
    chosen: Optional[Partition]
    scores: List[PlanScore] = field(default_factory=list)
    att_mixed: float = float("nan")
    fallback: str = ""
    guard: Optional[CapacityGuard] = None
    hybrid_guard: Optional[HybridGuard] = None
    used_gpus: int = 8
    unused_gpus: int = 0
    freq0: int = 2520


def enumerate_partitions(
        gpu_count: int,
        tps: Sequence[int] = (1, 2, 4),
        pps: Sequence[int] = (1, 2),
        freqs: Sequence[int] = (2520, 1500, 900),
        token_budgets: Sequence[int] = (TOKEN_BUDGET,),
        decode_batches: Sequence[int] = (DECODE_B_CAP,),
        include_pp: bool = True) -> List[Partition]:
    """mixed ∪ 分离配置。pp 默认参与;放不下的组合被 GPU 预算滤掉。"""
    pps = tuple(pps) if include_pp else (1,)
    out: List[Partition] = []
    for tm, pm, nm, fm, tb in product(tps, pps, range(0, gpu_count + 1),
                                      freqs, token_budgets):
        used_m = nm * tm * pm
        if used_m > gpu_count:
            continue
        if nm > 0:
            out.append(Partition(
                n_mixed=nm, tp_mixed=tm, pp_mixed=pm,
                freq_mixed=int(fm), token_budget_mixed=int(tb),
                gpus_total=gpu_count))
        remain = gpu_count - used_m
        if remain < 2:
            continue
        for tp, pp, td, pd, np_, nd, fp, fd, ptok, db in product(
                tps, pps, tps, pps,
                range(1, remain + 1), range(1, remain + 1),
                freqs, freqs, token_budgets, decode_batches):
            used = used_m + np_ * tp * pp + nd * td * pd
            if used > gpu_count:
                continue
            out.append(Partition(
                n_mixed=nm, tp_mixed=tm, pp_mixed=pm,
                freq_mixed=int(fm), token_budget_mixed=int(tb),
                n_prefill=np_, tp_prefill=tp, pp_prefill=pp,
                n_decode=nd, tp_decode=td, pp_decode=pd,
                freq_prefill=int(fp), freq_decode=int(fd),
                prefill_max_tokens=int(ptok), decode_max_batch=int(db),
                gpus_total=gpu_count))
    seen, uniq = set(), []
    for p in out:
        if p.n_mixed == 0 and p.n_prefill == 0:
            continue
        k = (p.n_mixed, p.n_prefill, p.n_decode,
             p.tp_mixed if p.n_mixed else 0,
             p.pp_mixed if p.n_mixed else 0,
             p.tp_prefill if p.n_prefill else 0,
             p.pp_prefill if p.n_prefill else 0,
             p.tp_decode if p.n_decode else 0,
             p.pp_decode if p.n_decode else 0,
             p.freq_mixed, p.freq_prefill, p.freq_decode,
             p.token_budget_mixed, p.prefill_max_tokens, p.decode_max_batch)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


A9_TP = 2


def default_mixed_tp(model_key: str = "14b") -> int:
    """共置默认一律 1×TP8(7B/14B/32B/72B)。"""
    del model_key
    return 8


def _mixed_full_freq(gpu_count: int, tp: int = 2) -> Partition:
    n = max(gpu_count // max(int(tp), 1), 1)
    return Partition(n_mixed=n, tp_mixed=int(tp), pp_mixed=1,
                     freq_mixed=2520, gpus_total=gpu_count)


def _full_clock(p: Partition, mhz: int = 2520) -> bool:
    if p.n_mixed and int(p.freq_mixed) < mhz:
        return False
    if p.n_prefill and int(p.freq_prefill) < mhz:
        return False
    if p.n_decode and int(p.freq_decode) < mhz:
        return False
    return True


def trusted_space(space: Sequence[Partition], gpu_count: int,
                  trusted: bool) -> List[Partition]:
    """不可信:只用满卡且各角色 2520,禁止缩容/深降频。"""
    space = list(space)
    if trusted:
        return space
    return [p for p in space
            if p.total_gpus() == int(gpu_count) and _full_clock(p)]


def _role_key(p: Partition) -> Tuple:
    return (int(p.n_mixed), int(p.n_prefill), int(p.n_decode),
            int(p.tp_mixed) if p.n_mixed else 0,
            int(p.tp_prefill) if p.n_prefill else 0,
            int(p.tp_decode) if p.n_decode else 0)


def same_role_layout(a: Partition, b: Partition) -> bool:
    return _role_key(a) == _role_key(b)


def is_shrink(p: Partition, gpu_count: int) -> bool:
    """少开卡:占用 GPU 少于预算。"""
    return int(p.total_gpus()) < int(gpu_count)


def shrink_allowed(att: float, att_mixed: float, rho: float,
                   margin: float = SHRINK_ATT_MARGIN_PP,
                   rho_max: float = SHRINK_RHO_MAX) -> bool:
    """缩容须 pred_att ≥ att_mixed + 3pp 且 ρ < 0.4。缺 ρ 则不许。"""
    if att != att or att_mixed != att_mixed:
        return False
    if rho != rho:
        return False
    if float(att) + 1e-12 < float(att_mixed) + float(margin):
        return False
    return float(rho) < float(rho_max)


def pin_prefill(p: Partition, mhz: int = PREFILL_PIN_MHZ) -> Partition:
    """有 P 池则 freq_prefill 不低于满频。"""
    if p.n_prefill <= 0 or int(p.freq_prefill) >= int(mhz):
        return p
    return replace(p, freq_prefill=int(mhz))


def _single_pair_budget(gpu_count: int, tp: int) -> bool:
    """8 卡 TP4 只装下一对 P/D,高载应偏 mixed。"""
    return int(gpu_count) // max(2 * max(int(tp), 1), 1) < 2


def _pred_fields(pr) -> Tuple[float, float, float, float, bool]:
    return (
        float(getattr(pr, "attainment", 0.0)),
        float(getattr(pr, "energy_j_per_s", 0.0)),
        float(getattr(pr, "energy_j_per_req", 0.0)),
        float(getattr(pr, "power_w", 0.0)),
        bool(getattr(pr, "attainable", True)),
    )


def _choose_untrusted_mixed_or_pd(
        predictor, lam: float, att_mixed: float, inherit: Partition,
        gpu_count: int, tp_fallback: int, delta: float,
        pin_prefill_mhz: int, shrink_rho_max: float,
        space: Sequence[Partition],
        layout_decision: Optional[str] = None,
        layout_reason: str = "") -> PlanResult:
    """不可信:启动只起满配 mixed。空间 PD 由线上控制面决定,不查表。"""
    del inherit, delta, shrink_rho_max, layout_decision
    mixed = _mixed_full_freq(gpu_count, tp_fallback)
    scores: List[PlanScore] = []
    seen = set()
    for p in (mixed,) + tuple(space):
        key = _role_key(p)
        if key in seen:
            continue
        seen.add(key)
        pr = predictor(p, float(lam), None)
        att, e_js, e_req, pw, attainable = _pred_fields(pr)
        ok = p.pairs() == 0 and p.n_mixed == mixed.n_mixed
        scores.append(PlanScore(
            partition=p, attainment=att, energy_j_per_s=e_js,
            energy_j_per_req=e_req, power_w=pw, attainable=attainable,
            accepted=ok,
            reason="boot-mixed" if ok else "online-not-startup"))
    chosen = mixed
    if pin_prefill_mhz:
        chosen = pin_prefill(chosen, pin_prefill_mhz)
    return PlanResult(mode=PLANNER_MODE_ISO_LOAD, chosen=chosen,
                      scores=scores, att_mixed=float(att_mixed),
                      fallback=layout_reason or "boot-mixed")


def _finite_pos(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return x == x and x > 0.0


def estimate_mu_prefill(predictor, n_p: int,
                        freq: Optional[int] = None) -> Optional[float]:
    """P 池 req/s。缺 stage_rate_prefill 则无法证明 PD 安全。"""
    if int(n_p) <= 0:
        return 0.0
    fn = getattr(predictor, "stage_rate_prefill", None)
    if not callable(fn):
        return None
    try:
        if freq is None:
            val = fn(int(n_p))
        else:
            val = fn(int(n_p), freq=int(freq))
    except TypeError:
        try:
            val = fn(int(n_p))
        except (TypeError, ValueError, AttributeError):
            return None
    except (ValueError, AttributeError):
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out or out < 0.0:
        return None
    return out


def estimate_mu_mixed(predictor, n_m: int,
                      freq: Optional[int] = None) -> Optional[float]:
    if int(n_m) <= 0:
        return 0.0
    fn = getattr(predictor, "rate_mixed", None)
    if not callable(fn):
        return None
    try:
        if freq is None:
            val = fn(int(n_m))
        else:
            val = fn(int(n_m), freq=int(freq))
    except TypeError:
        try:
            val = fn(int(n_m))
        except (TypeError, ValueError, AttributeError):
            return None
    except (ValueError, AttributeError):
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out or out < 0.0:
        return None
    return out


def estimate_mu_prefill_single(predictor, n_p: int,
                               freq: Optional[int] = None,
                               tp: int = 2) -> Optional[float]:
    """护栏用的单请求 μ_P。缺表时回退组批 μ。"""
    if int(n_p) <= 0:
        return 0.0
    om = getattr(predictor, "opmodel", None)
    stats = getattr(predictor, "stats", None)
    mhz = int(freq if freq is not None else PREFILL_PIN_MHZ)
    if om is not None and stats is not None:
        try:
            from ecopadg.profiler import mu_prefill
            plen = max(int(round(float(getattr(stats, "prompt_len", 0) or 0))), 1)
            return float(mu_prefill(om, plen, int(n_p), mhz, int(tp)))
        except (TypeError, ValueError, AttributeError):
            pass
    return estimate_mu_prefill(predictor, n_p, freq)


def estimate_prefill_s(predictor, freq: int, tp: int = 2) -> Optional[float]:
    om = getattr(predictor, "opmodel", None)
    stats = getattr(predictor, "stats", None)
    if om is None or stats is None:
        return None
    try:
        from ecopadg.profiler import prefill_s
        plen = max(int(round(float(getattr(stats, "prompt_len", 0) or 0))), 1)
        return float(prefill_s(om, plen, int(freq), int(tp)))
    except (TypeError, ValueError, AttributeError):
        return None


def default_pd_threshold(prompt_len: float) -> int:
    """A9 长短分流键:mean pin,不是写死的 512。"""
    try:
        pin = float(prompt_len)
    except (TypeError, ValueError):
        return 1
    if pin != pin or pin <= 0.0:
        return 1
    return max(1, int(round(pin)))


def estimate_mu_decode(predictor, n_d: int,
                       freq: Optional[int] = None) -> Optional[float]:
    """D 池 req/s。缺 stage_rate_decode 且无 OpModel 则 μ_D 未知。"""
    if int(n_d) <= 0:
        return 0.0
    fn = getattr(predictor, "stage_rate_decode", None)
    if not callable(fn):
        return None
    try:
        if freq is None:
            val = fn(int(n_d))
        else:
            val = fn(int(n_d), freq=int(freq))
    except TypeError:
        try:
            val = fn(int(n_d))
        except (TypeError, ValueError, AttributeError):
            return None
    except (ValueError, AttributeError):
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out or out < 0.0:
        return None
    return out


def estimate_decode_s(predictor, freq: int, tp: int = 2) -> Optional[float]:
    """单请求 decode 墙钟:pout × iter(b=1, f, ctx)。护栏与 DVFS 共用。"""
    om = getattr(predictor, "opmodel", None)
    stats = getattr(predictor, "stats", None)
    if om is None or stats is None:
        return None
    try:
        from ecopadg.profiler import decode_s
        plen = max(float(getattr(stats, "prompt_len", 0) or 0), 1.0)
        pout = max(int(round(float(getattr(stats, "output_len", 0) or 0))), 1)
        return float(decode_s(om, plen, pout, int(freq), int(tp), 1))
    except (TypeError, ValueError, AttributeError):
        return None


def estimate_mu_decode_single(predictor, n_d: int,
                              freq: Optional[int] = None,
                              tp: int = 2) -> Optional[float]:
    """护栏 μ_D:n_D × LOAD_CAP_PER_INST / T_dec(pout,b=1)。

    顺序单请求墙钟不是 req/s。乘已有并发上限才和 λ、α 同一量纲。
    无 OpModel 时回退 stage_rate_decode。非有限则调用方 fail-closed。
    """
    if int(n_d) <= 0:
        return 0.0
    mhz = int(freq if freq is not None else PREFILL_PIN_MHZ)
    dec_s = estimate_decode_s(predictor, mhz, tp)
    if dec_s is not None and dec_s > 0.0:
        return float(n_d) * float(LOAD_CAP_PER_INST) / float(dec_s)
    return estimate_mu_decode(predictor, n_d, freq)


def _frac_long_of(predictor, frac_long: Optional[float]) -> float:
    if frac_long is not None:
        try:
            value = float(frac_long)
        except (TypeError, ValueError):
            value = float("nan")
        else:
            if value == value:
                return value
    stats = getattr(predictor, "stats", None)
    if stats is None:
        return float("nan")
    try:
        value = float(getattr(stats, "frac_long", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value


def _slo_ttft_s(predictor) -> float:
    slo = getattr(predictor, "slo", None)
    if slo is None:
        return 0.0
    try:
        return float(getattr(slo, "ttft_s", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def hybrid_capacity_ok(
        predictor, rate: float, part: Partition, frac_long: float,
        alpha: float = PD_PREFILL_RHO_GUARD) -> HybridGuard:
    """A9 当且仅当长短都在且分流后 2×mixed / 1P 都装得下。"""
    fl = float(frac_long) if frac_long == frac_long else float("nan")
    guard = HybridGuard(frac_long=fl, alpha=float(alpha))
    if fl != fl or fl + 1e-12 < HYBRID_FRAC_LONG_LO \
            or fl - 1e-12 > HYBRID_FRAC_LONG_HI:
        guard.reason = "hybrid-frac-long-out-of-band"
        return guard
    lam = max(float(rate), 0.0)
    guard.lam_mixed = lam * (1.0 - fl)
    guard.lam_pd = lam * fl
    mu_m = estimate_mu_mixed(predictor, int(part.n_mixed), 2520)
    fp = int(getattr(part, "freq_prefill", 0) or PREFILL_PIN_MHZ)
    fd = int(getattr(part, "freq_decode", 0) or PREFILL_PIN_MHZ)
    tp_p = int(getattr(part, "tp_prefill", 0) or 2)
    tp_d = int(getattr(part, "tp_decode", 0) or 2)
    mu_p = estimate_mu_prefill_single(
        predictor, int(part.n_prefill), fp, tp_p)
    mu_d = estimate_mu_decode_single(
        predictor, int(part.n_decode), fd, tp_d)
    guard.mu_mixed = float(mu_m) if mu_m is not None else float("nan")
    guard.mu_prefill = float(mu_p) if mu_p is not None else float("nan")
    guard.mu_decode = float(mu_d) if mu_d is not None else float("nan")
    if _finite_pos(mu_m):
        guard.rho_mixed = guard.lam_mixed / float(mu_m)
    if _finite_pos(mu_p):
        guard.rho_prefill = guard.lam_pd / float(mu_p)
    if _finite_pos(mu_d):
        guard.rho_decode = guard.lam_pd / float(mu_d)
    pre_s = estimate_prefill_s(predictor, fp, tp_p)
    ttft = _slo_ttft_s(predictor)
    if pre_s is not None and ttft > 0.0 and float(pre_s) > ttft:
        guard.reason = "hybrid-ttft"
        return guard
    if not _finite_pos(mu_m) or not _finite_pos(mu_p) or not _finite_pos(mu_d):
        guard.reason = "hybrid-mu-unknown"
        return guard
    if guard.lam_mixed + 1e-12 >= float(alpha) * float(mu_m):
        guard.reason = "hybrid-mixed-rho-guard"
        return guard
    if guard.lam_pd + 1e-12 >= float(alpha) * float(mu_p):
        guard.reason = "hybrid-pd-rho-guard"
        return guard
    if guard.lam_pd + 1e-12 >= float(alpha) * float(mu_d):
        guard.reason = "hybrid-pd-decode-rho-guard"
        return guard
    guard.ok = True
    guard.reason = "hybrid-capacity-ok"
    return guard


def spatial_pd_capacity_ok(
        predictor, rate: float, part: Partition,
        alpha: float = PD_PREFILL_RHO_GUARD) -> CapacityGuard:
    """纯空间 PD:λ < α·μ_P 且 λ < α·μ_D 且单请求 prefill ≤ S_TTFT。"""
    fp = int(getattr(part, "freq_prefill", 0) or PREFILL_PIN_MHZ)
    fd = int(getattr(part, "freq_decode", 0) or PREFILL_PIN_MHZ)
    tp_p = int(getattr(part, "tp_prefill", 0) or 2)
    tp_d = int(getattr(part, "tp_decode", 0) or 2)
    mu_p = estimate_mu_prefill_single(
        predictor, int(part.n_prefill), fp, tp_p)
    mu_d = estimate_mu_decode_single(
        predictor, int(part.n_decode), fd, tp_d)
    tp = int(getattr(part, "tp_prefill", 0) or getattr(part, "tp_decode", 0)
             or 2)
    mixed = _mixed_full_freq(int(getattr(part, "gpus_total", 0) or 8), tp)
    mu_m = estimate_mu_mixed(predictor, int(mixed.n_mixed), 2520)
    pre_s = estimate_prefill_s(predictor, fp, tp_p)
    slo = getattr(predictor, "slo", None)
    ttft = 0.0
    if slo is not None:
        try:
            ttft = float(getattr(slo, "ttft_s", 0) or 0)
        except (TypeError, ValueError):
            ttft = 0.0
    rho = float("nan")
    rho_d = float("nan")
    if _finite_pos(mu_p):
        rho = float(rate) / float(mu_p)
    if _finite_pos(mu_d):
        rho_d = float(rate) / float(mu_d)
    pd_ok = False
    reason = "capacity-mu-unknown-mixed"
    if pre_s is not None and ttft > 0.0 and float(pre_s) > ttft:
        reason = "capacity-ttft-mixed"
    elif not _finite_pos(mu_p):
        reason = "capacity-mu-unknown-mixed"
    elif not _finite_pos(mu_d):
        reason = "capacity-mu-d-unknown-mixed"
    elif float(rate) + 1e-12 >= float(alpha) * float(mu_p):
        reason = "capacity-rho-guard-mixed"
    elif float(rate) + 1e-12 >= float(alpha) * float(mu_d):
        reason = "capacity-decode-rho-guard-mixed"
    else:
        pd_ok = True
        reason = "capacity-rho-ok"
    return CapacityGuard(
        mu_prefill=float(mu_p) if mu_p is not None else float("nan"),
        mu_mixed=float(mu_m) if mu_m is not None else float("nan"),
        mu_decode=float(mu_d) if mu_d is not None else float("nan"),
        rho_prefill=rho, rho_decode=rho_d, alpha=float(alpha),
        pre_s=float(pre_s) if pre_s is not None else float("nan"),
        pd_ok=pd_ok, reason=reason)


def lookup_to_partition(cfg: dict, gpu_count: int = 8,
                        tp: int = 2) -> Partition:
    """查表 layout → 8 卡 4×mixed 或 2×1P1D。忽略 tp=1 与表内 freq。

    启动时钟一律满频;L0 用能量最优 + goodput 门,不把查表 f* 写成 prefer。
    """
    layout = str((cfg or {}).get("layout") or LAYOUT_MIXED)
    mhz = int(PREFILL_PIN_MHZ)
    if layout == LAYOUT_SPATIAL_PD:
        part = spatial_pd_partition(gpu_count, tp)
        return replace(part, freq_prefill=mhz, freq_decode=mhz)
    mixed = _mixed_full_freq(gpu_count, tp)
    return replace(mixed, freq_mixed=mhz, freq_prefill=mhz, freq_decode=mhz)


def choose_boot_lookup(
        predictor, rate: float, att_mixed: float,
        lookup_cfg: Optional[dict],
        gpu_count: int = 8, tp: int = 2,
        trusted: bool = False,
        mixed_tp: Optional[int] = None,
        spatial: Optional[Partition] = None) -> PlanResult:
    """有查表键则用其 layout; 预填 ρ 护栏禁止不安全的纯 PD。无键走 boot-slo。"""
    if not lookup_cfg:
        result = choose_boot_slo_layout(
            predictor, rate, att_mixed, gpu_count=gpu_count, tp=tp,
            trusted=trusted, mixed_tp=mixed_tp, spatial=spatial)
        result.mode = "boot-lookup"
        result.fallback = "boot-lookup-fallback-%s" % (
            result.fallback or "slo")
        return result
    mix_tp = int(mixed_tp if mixed_tp is not None else tp)
    layout = str(lookup_cfg.get("layout") or LAYOUT_MIXED)
    guard = None
    if layout == LAYOUT_SPATIAL_PD:
        spatial = lookup_to_partition(lookup_cfg, gpu_count, tp)
        guard = spatial_pd_capacity_ok(predictor, rate, spatial)
        if not guard.pd_ok:
            chosen = lookup_to_partition(
                dict(lookup_cfg, layout=LAYOUT_MIXED), gpu_count, mix_tp)
            fallback = "boot-lookup-capacity-mixed"
        else:
            chosen = spatial
            fallback = "boot-lookup-spatial-pd"
    else:
        chosen = lookup_to_partition(lookup_cfg, gpu_count, mix_tp)
        fallback = "boot-lookup-mixed"
    chosen = pin_prefill(chosen, PREFILL_PIN_MHZ)
    return PlanResult(mode="boot-lookup", chosen=chosen,
                      att_mixed=float(att_mixed), fallback=fallback,
                      guard=guard)


def spatial_pd_partition(gpu_count: int = 8, tp: int = 2) -> Partition:
    """满卡空间 1P1D。14B/32B TP2 → 2 段；72B TP4 → 1 段。不查表。"""
    pairs = int(gpu_count) // (2 * int(tp))
    if pairs < 1:
        raise ValueError(
            "spatial PD needs at least 2*tp GPUs, got gpus=%s tp=%s"
            % (gpu_count, tp))
    return Partition(
        n_mixed=0,
        n_prefill=pairs,
        n_decode=pairs,
        tp_prefill=int(tp),
        tp_decode=int(tp),
        freq_prefill=int(PREFILL_PIN_MHZ),
        freq_decode=2520,
        gpus_total=int(gpu_count))


def hybrid_selective_partition(gpu_count: int = 8,
                               tp: int = 2) -> Optional[Partition]:
    """A9:2×mixed + 1×1P1D。仅 8 卡 TP2(2×2+2+2=8)。"""
    if int(gpu_count) < 8 or int(tp) != 2:
        return None
    return Partition(
        n_mixed=2, n_prefill=1, n_decode=1,
        tp_mixed=2, tp_prefill=2, tp_decode=2,
        freq_mixed=2520, freq_prefill=2520, freq_decode=2520,
        gpus_total=int(gpu_count))


def layout_kind(p: Partition) -> str:
    """启动候选标签:mixed / spatial-pd / hybrid-a9。"""
    if int(p.n_mixed) > 0 and p.pairs() > 0:
        return "hybrid-a9"
    if p.pairs() > 0:
        return "spatial-pd"
    return "mixed"


def energy_rankable(att: float, e_js: float) -> bool:
    """att→0 时 energy_j_per_s 会炸到 1e13,禁止用来排名。"""
    return (math.isfinite(float(e_js))
            and 0.0 < float(e_js) < ENERGY_RANK_MAX_J_PER_S
            and float(att) > 1e-3)


_KIND_TIE = {"spatial-pd": 0, "hybrid-a9": 1, "mixed": 2}


def _stats_pin_pout(predictor) -> Tuple[int, int]:
    stats = getattr(predictor, "stats", None)
    pin = getattr(stats, "prompt_len", 512.0) if stats is not None else 512.0
    pout = getattr(stats, "output_len", 200.0) if stats is not None else 200.0
    try:
        pin_i = max(int(round(float(pin))), 1)
    except (TypeError, ValueError):
        pin_i = 512
    try:
        pout_i = max(int(round(float(pout))), 1)
    except (TypeError, ValueError):
        pout_i = 200
    return pin_i, pout_i


def opmodel_taxed_proxy_j(predictor, part: Partition, rate: float):
    """OpModel 带驻留/KV 税的路径 J。无模型返回 None。"""
    om = getattr(predictor, "opmodel", None)
    if om is None:
        return None
    try:
        from ecopadg.constraints import path_gross_j
    except ImportError:
        return None
    pin, pout = _stats_pin_pout(predictor)
    kind = layout_kind(part)
    path = "mixed" if kind == "mixed" else "pd"
    try:
        res_w = float(om.residency_w(PREFILL_PIN_MHZ))
    except (TypeError, ValueError, AttributeError):
        res_w = 0.0
    if res_w != res_w or res_w < 0.0:
        res_w = 0.0
    extra = 0
    if path == "pd":
        extra = max(int(getattr(part, "n_prefill", 0) or 0)
                    * int(getattr(part, "tp_prefill", 0) or 2), 2)
    try:
        return float(path_gross_j(
            om, path, pin, pout, max(float(rate), 0.0),
            freq_prefill=PREFILL_PIN_MHZ,
            freq_decode=900 if path == "pd" else 1500,
            residency_w=res_w, extra_gpus=extra))
    except (TypeError, ValueError, AttributeError):
        return None


def untrusted_rank_j(predictor, part: Partition, rate: float,
                     att: float, e_js: float) -> float:
    """不可信排名 J:有 OpModel 用 taxed 代理;否则用可排名预测;再否则 mixed 便宜。"""
    taxed = opmodel_taxed_proxy_j(predictor, part, rate)
    if taxed is not None and math.isfinite(taxed) and taxed > 0.0:
        return float(taxed)
    if energy_rankable(att, e_js):
        return float(e_js)
    return 1.0 if layout_kind(part) == "mixed" else ENERGY_RANK_MAX_J_PER_S


def _invoke_pred(predictor, part: Partition, rate: float):
    pred_fn = getattr(predictor, "predict", None)
    if callable(pred_fn):
        return pred_fn(part, max(float(rate), 0.0))
    return predictor(part, max(float(rate), 0.0), None)


def choose_boot_slo_layout(
        predictor, rate: float, att_mixed: float,
        gpu_count: int = 8, tp: int = 2,
        trusted: bool = False,
        frac_long: Optional[float] = None,
        mixed_tp: Optional[int] = None,
        spatial: Optional[Partition] = None,
        profile: Optional[dict] = None,
        n: int = 200,
        dataset: str = "sharegpt") -> PlanResult:
    """满卡一次门控:1×TP8 mixed / DistServe 赢家 PD / 可选 A9。

    不可信:ρ 门只作可行性;可行集内 min 代理 J(OpModel taxed,否则预测)。
    不缩 used、不走 ScaleInst。ScaleInst 只在 dynamollm / compose_boot_slo_layout。
    trusted 仍是满卡三候选 att≥0.90 再 min J。A9 仅 8 卡 TP2。
    """
    del profile, n, dataset
    mix_tp = int(mixed_tp if mixed_tp is not None else 8)
    if float(rate) <= 0.0:
        fallback = pin_prefill(_mixed_full_freq(gpu_count, mix_tp), PREFILL_PIN_MHZ)
        return PlanResult(
            mode="boot-slo-layout", chosen=fallback,
            att_mixed=float(att_mixed), fallback="rate-nonpositive",
            used_gpus=int(gpu_count), unused_gpus=0, freq0=2520)
    mixed = _mixed_full_freq(gpu_count, mix_tp)
    named: List[Tuple[str, Partition]] = [("mixed", mixed)]
    if spatial is not None:
        named.append(("spatial-pd", spatial))
    else:
        try:
            named.append(("spatial-pd", spatial_pd_partition(gpu_count, tp)))
        except ValueError:
            pass
    hybrid = hybrid_selective_partition(gpu_count, A9_TP)
    if hybrid is not None:
        named.append(("hybrid-a9", hybrid))

    fl = _frac_long_of(predictor, frac_long)
    scores: List[PlanScore] = []
    by_name: Dict[str, PlanScore] = {}
    guard: Optional[CapacityGuard] = None
    hybrid_guard: Optional[HybridGuard] = None
    for name, part in named:
        part = pin_prefill(part, PREFILL_PIN_MHZ)
        pr = _invoke_pred(predictor, part, rate)
        att, e_js, e_req, pw, attainable = _pred_fields(pr)
        att_ok = att + 1e-12 >= GOODPUT_ATT_GATE
        e_ok = energy_rankable(att, e_js)
        if name == "spatial-pd" and not trusted:
            guard = spatial_pd_capacity_ok(predictor, rate, part)
            if guard.pd_ok:
                ok, reason = True, "capacity-rho-ok"
            else:
                ok, reason = False, guard.reason
        elif trusted:
            if att_ok and e_ok:
                ok, reason = True, "att-then-j"
            elif not att_ok:
                ok, reason = False, "pred-att-below-0.90"
            else:
                ok, reason = False, "energy-unrankable"
        elif name == "hybrid-a9":
            hybrid_guard = hybrid_capacity_ok(predictor, rate, part, fl)
            if hybrid_guard.ok:
                ok, reason = True, hybrid_guard.reason
            else:
                ok, reason = False, hybrid_guard.reason
        else:
            ok, reason = True, "boot-mixed-fallback"
        sc = PlanScore(
            partition=part, attainment=att, energy_j_per_s=e_js,
            energy_j_per_req=e_req, power_w=pw, attainable=attainable,
            accepted=ok, reason=reason)
        scores.append(sc)
        by_name[name] = sc

    accepted = [s for s in scores if s.accepted]
    mixed_score = by_name["mixed"]
    if trusted and accepted:
        best = min(accepted, key=lambda s: (s.energy_j_per_s, -s.attainment))
        fallback = "trusted-min-j-s.t.-att-0.90"
    elif (not trusted) and accepted:
        ranked = []
        for score in accepted:
            proxy = untrusted_rank_j(
                predictor, score.partition, rate,
                score.attainment, score.energy_j_per_s)
            ranked.append((proxy, score))
        best_proxy, best = min(
            ranked,
            key=lambda item: (
                item[0],
                _KIND_TIE.get(layout_kind(item[1].partition), 9),
                -item[1].attainment))
        del best_proxy
        fallback = "untrusted-min-j-%s" % layout_kind(best.partition)
        if (layout_kind(best.partition) == "mixed"
                and guard is not None and not guard.pd_ok):
            fallback = guard.reason
    else:
        best = mixed_score
        fallback = "boot-mixed-fallback"

    chosen = pin_prefill(best.partition, PREFILL_PIN_MHZ)
    # pdblend L0 一律能量档起步;mixed_dvfs 对照不走这条。
    freq0 = 900
    return PlanResult(mode="boot-slo-layout", chosen=chosen,
                      scores=scores, att_mixed=float(att_mixed),
                      fallback=fallback, guard=guard,
                      hybrid_guard=hybrid_guard,
                      used_gpus=int(gpu_count), unused_gpus=0,
                      freq0=int(freq0))


def choose_iso_load(predictor, lam: float, att_mixed: float,
                    space: Optional[Sequence[Partition]] = None,
                    gpu_count: int = 8, tp_fallback: int = 2,
                    delta: float = SLO_NON_INFERIOR_PP,
                    mode: str = PLANNER_MODE_ISO_LOAD,
                    trusted: bool = True,
                    capacity_trusted: bool = True,
                    inherit: Optional[Partition] = None,
                    energy_cap_j_per_s: Optional[float] = None,
                    pin_prefill_mhz: int = PREFILL_PIN_MHZ,
                    shrink_att_margin: float = SHRINK_ATT_MARGIN_PP,
                    shrink_rho_max: float = SHRINK_RHO_MAX,
                    layout_decision: Optional[str] = None,
                    layout_reason: str = "",
                    model_key: str = "",
                    dataset: str = "",
                    arrival: str = "",
                    frac_long: Optional[float] = None) -> PlanResult:
    """对指定 λ 选 min J 且 att 非劣于 mixed。可行集空则回退 mixed@2520。

    capacity_trusted=False:启动只起满配 mixed,不查 inherit 表。
    energy_cap_j_per_s:可信搜索时拒绝预测能耗高于 ds_pd 布局的候选。
    pin_prefill_mhz:搜索空间丢掉 P 降频;0 关闭。
    缩容须 pred_att ≥ att_mixed+shrink_att_margin 且 ρ < shrink_rho_max。
    """
    del model_key, dataset, arrival, frac_long, layout_decision, layout_reason
    if mode == PLANNER_MODE_CAPACITY:
        raise ValueError("capacity-envelope 请用 capacity_envelope()")
    space = list(space) if space is not None else enumerate_partitions(
        gpu_count, tps=(tp_fallback,), pps=(1,), include_pp=False)
    space = trusted_space(space, gpu_count, trusted)
    if pin_prefill_mhz:
        space = [p for p in space
                 if p.n_prefill == 0
                 or int(p.freq_prefill) >= int(pin_prefill_mhz)]
    if inherit is not None and pin_prefill_mhz:
        inherit = pin_prefill(inherit, pin_prefill_mhz)
    if not (trusted and capacity_trusted):
        return _choose_untrusted_mixed_or_pd(
            predictor, lam, att_mixed, inherit or _mixed_full_freq(
                gpu_count, tp_fallback),
            gpu_count, tp_fallback,
            delta, pin_prefill_mhz, shrink_rho_max, space,
            layout_reason="boot-mixed")
    scores = []
    accepted: List[PlanScore] = []
    for p in space:
        pr = predictor(p, float(lam), None)
        att = float(getattr(pr, "attainment", 0.0))
        e_js = float(getattr(pr, "energy_j_per_s", 0.0))
        pw = float(getattr(pr, "power_w", 0.0))
        rho = float(getattr(pr, "rho", float("nan")))
        model_ok = (bool(getattr(pr, "attainable", True))
                    and slo_non_inferior(att, att_mixed, delta))
        inherit_hit = inherit is not None and same_role_layout(p, inherit)
        full_pd = (p.pairs() > 0 and p.total_gpus() == int(gpu_count)
                   and _full_clock(p))
        if not capacity_trusted and (inherit_hit or full_pd):
            ok, reason = True, "capacity-untrusted-keep-pd"
        elif model_ok:
            ok, reason = True, ""
        else:
            ok, reason = False, "slo-or-queue"
        if ok and is_shrink(p, gpu_count):
            if not shrink_allowed(att, att_mixed, rho,
                                  margin=shrink_att_margin,
                                  rho_max=shrink_rho_max):
                ok, reason = False, "shrink-margin"
        if (ok and capacity_trusted and energy_cap_j_per_s is not None
                and e_js == e_js
                and e_js > float(energy_cap_j_per_s) + 1e-9
                and not inherit_hit):
            ok, reason = False, "energy-above-distserve"
        sc = PlanScore(
            partition=p, attainment=att, energy_j_per_s=e_js,
            energy_j_per_req=float(getattr(pr, "energy_j_per_req", 0.0)),
            power_w=pw,
            attainable=bool(getattr(pr, "attainable", True)),
            accepted=ok, reason=reason)
        scores.append(sc)
        if ok:
            accepted.append(sc)
    if accepted:
        if capacity_trusted:
            best = min(accepted, key=lambda s: (s.energy_j_per_s, -s.attainment))
        else:
            best = min(accepted, key=lambda s: (s.power_w, -s.attainment))
        note = "" if trusted else "untrusted-full-gpu-2520"
        chosen = best.partition
        if pin_prefill_mhz:
            chosen = pin_prefill(chosen, pin_prefill_mhz)
        return PlanResult(mode=PLANNER_MODE_ISO_LOAD, chosen=chosen,
                          scores=scores, att_mixed=float(att_mixed),
                          fallback=note)
    fb = _mixed_full_freq(gpu_count, tp_fallback)
    if pin_prefill_mhz:
        fb = pin_prefill(fb, pin_prefill_mhz)
    return PlanResult(mode=PLANNER_MODE_ISO_LOAD, chosen=fb, scores=scores,
                      att_mixed=float(att_mixed), fallback="mixed@2520")


def choose_m0_tuned(predictor, lam: float, att_mixed: float,
                    gpu_count: int = 8, tp: int = 2,
                    freqs: Sequence[int] = (2520, 2100, 1800, 1500, 1200, 900),
                    token_budgets: Sequence[int] = (4096, TOKEN_BUDGET),
                    max_seqs: Sequence[int] = (8, 16, 32),
                    delta: float = SLO_NON_INFERIOR_PP) -> PlanResult:
    """mixed-only 消融:扫 token_budget × max_seqs × freq,约束 att≥mixed@2520。

    不是外部 QoS baseline,只报「只调 batch+DVFS 相对默认 mixed 能省多少」。
    """
    n = max(int(gpu_count) // max(int(tp), 1), 1)
    space = [
        Partition(n_mixed=n, tp_mixed=int(tp), pp_mixed=1,
                  freq_mixed=int(f), token_budget_mixed=int(tb),
                  decode_max_batch=int(ms), gpus_total=gpu_count)
        for f, tb, ms in product(freqs, token_budgets, max_seqs)
    ]
    r = choose_iso_load(predictor, lam, att_mixed, space=space,
                        gpu_count=gpu_count, tp_fallback=tp, delta=delta)
    r.mode = "m0-tuned"
    return r


def capacity_envelope(profile: dict, model: str, workload_lengths,
                      gpu_count: int, tps: Sequence[int], pps: Sequence[int],
                      ttft_target_ms: float, tpot_target_ms: float,
                      seed: int = 0, n: int = 200) -> dict:
    """附录:DistServe 式 max per-GPU rate,不参与选配。"""
    cfgs = distserve_configs(tps, pps, gpu_count)
    rates = {}
    for cfg in cfgs:
        _, tp_p, pp_p, tp_d, pp_d = cfg
        rates[cfg] = bisect_rate(
            profile, model, workload_lengths, tp_p, pp_p, tp_d, pp_d,
            transfer_ms_per_kv_token=0.0, seed=seed, N=n,
            ttft_target_ms=ttft_target_ms, tpot_target_ms=tpot_target_ms)
    return dict(mode=PLANNER_MODE_CAPACITY, rates={str(k): v for k, v in rates.items()})
