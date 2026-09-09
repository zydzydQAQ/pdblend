# -*- coding: utf-8 -*-
"""EcoServe Algorithm 1 约束 + 同负载 SLO 非劣路径检查。

padg_scheduler 的仿真扩展从这里 import,不再维护第二份 Algorithm 1。
线上 SelectiveRouter 用 pool_slo_ok / path_gross_j。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ecopadg.types import KV_XFER_S_PER_TOK, SLO_NON_INFERIOR_PP

PrefillTimeFn = Callable[[int], float]
IterTimeFn = Callable[[int, int], float]
EnergyCostFn = Callable[[int, int], float]
KvBytesFn = Callable[[int, int], float]


@dataclass
class RequestRecord:
    """Scoreboard 中一条在批请求。"""
    rid: int
    arrival_time: float
    output_len: int
    first_token_time: float
    input_len: int = 0
    kv_bytes: float = 0.0
    deadline: float = float("inf")

    def remaining_tokens(self, now: float, iter_time: float) -> int:
        if iter_time <= 0:
            raise ValueError("iter_time 必须为正")
        elapsed_tokens = int((now - self.first_token_time) / iter_time)
        return max(0, self.output_len - elapsed_tokens)


@dataclass
class InstanceState:
    """一个实例的状态快照。"""
    idx: int
    t_switch: float
    pending_prefills: List[RequestRecord] = field(default_factory=list)
    existed_decodes: List[RequestRecord] = field(default_factory=list)
    used_kv_bytes: float = 0.0
    kv_capacity_bytes: float = float("inf")
    in_prefill_window: bool = True
    freq_mhz: Optional[int] = None


def _call_itf(itf: IterTimeFn, b: int, f: int, ctx: float) -> float:
    try:
        return itf(b, f, ctx)  # type: ignore[call-arg]
    except TypeError:
        return itf(b, f)


def constraint_reason(inst: InstanceState, req: RequestRecord,
                      s_ttft: float, s_tpot: float,
                      prefill_time_fn: PrefillTimeFn,
                      now: float) -> Optional[str]:
    """Algorithm 1: TTFT / TPOT slack / KV。None=通过。"""
    pending = inst.pending_prefills + [req]
    t_total = sum(prefill_time_fn(r.input_len) for r in pending)
    if t_total > s_ttft:
        return "ttft"
    if inst.existed_decodes:
        saved = [r.output_len * s_tpot - (now - r.first_token_time)
                 for r in inst.existed_decodes]
        if sum(saved) / len(saved) < t_total:
            return "tpot"
    if inst.used_kv_bytes + req.kv_bytes > inst.kv_capacity_bytes:
        return "kv"
    return None


def check_constraints(
    instances: List[InstanceState],
    req: RequestRecord,
    s_ttft: float,
    s_tpot: float,
    prefill_time_fn: PrefillTimeFn,
    now: float,
) -> Optional[int]:
    for inst in instances:
        if constraint_reason(inst, req, s_ttft, s_tpot, prefill_time_fn,
                             now) is None:
            return inst.idx
    return None


def energy_constraint_reason(
        inst: InstanceState, req: RequestRecord, s_ttft: float, s_tpot: float,
        prefill_time_fn: PrefillTimeFn, iter_time_fn: IterTimeFn,
        freq_candidates: List[int], now: float, lam: float) -> Optional[str]:
    pending = inst.pending_prefills + [req]
    t_total = sum(prefill_time_fn(r.input_len) for r in pending)
    if t_total > s_ttft:
        return "ttft"
    if inst.used_kv_bytes + req.kv_bytes > inst.kv_capacity_bytes:
        return "kv"
    if not inst.existed_decodes:
        return None
    b = len(inst.existed_decodes)
    f_max = max(freq_candidates)
    budget = sum(max(0.0, r.output_len * s_tpot - (now - r.first_token_time))
                 for r in inst.existed_decodes)
    for f in sorted(freq_candidates):
        ok = True
        for r in inst.existed_decodes:
            ctx = r.input_len + r.output_len / 2.0
            rem = r.remaining_tokens(now, _call_itf(iter_time_fn, b, f, ctx))
            used = now - r.first_token_time
            if used + rem * _call_itf(iter_time_fn, b, f, ctx) \
                    > r.output_len * s_tpot:
                ok = False
                break
        if not ok:
            continue
        if t_total > min(r.output_len * s_tpot - (now - r.first_token_time)
                         for r in inst.existed_decodes):
            continue
        dslow = 0.0
        for r in inst.existed_decodes:
            ctx = r.input_len + r.output_len / 2.0
            rem_max = max(
                0.0,
                r.output_len
                - (now - r.first_token_time) / _call_itf(iter_time_fn, b, f_max, ctx))
            dslow += rem_max * (_call_itf(iter_time_fn, b, f, ctx)
                                - _call_itf(iter_time_fn, b, f_max, ctx))
        if dslow <= lam * budget:
            return None
    return "freq"


@dataclass
class PoolView:
    """路由用的池快照(不必构造完整 InstanceState)。"""
    name: str
    pending_prefill_s: float = 0.0
    tpot_slack_s: float = float("inf")
    free_blocks: int = 10 ** 9
    need_blocks: int = 0
    load: float = 0.0
    residency_w: float = 0.0
    extra_gpus: int = 0


def pool_slo_ok(view: PoolView, need_prefill_s: float, *,
                s_ttft: float, need_blocks: int = 0) -> bool:
    """EcoServe 风格:排队 prefill + 本请求 ≤ TTFT,且 KV/slack 够。"""
    if view.pending_prefill_s + need_prefill_s > s_ttft:
        return False
    if need_prefill_s > view.tpot_slack_s:
        return False
    if need_blocks and need_blocks > view.free_blocks:
        return False
    return True


def path_gross_j(opmodel, path: str, prompt_len: int, output_len: int,
                 load: float, *,
                 freq_prefill: int = 2520, freq_decode: int = 1500,
                 batch: int = 8,
                 kv_xfer_s_per_tok: float = KV_XFER_S_PER_TOK,
                 residency_w: float = 0.0,
                 extra_gpus: int = 0) -> float:
    """同负载路径能耗代理:执行 J + 排队等待驻留 + KV。"""
    plen = max(int(prompt_len), 1)
    olen = max(int(output_len), 1)
    # 单请求 TTFT / 路径能耗:传该条 L,不是 P 池组批 Σtok
    t_pre = float(opmodel.prefill_time_ms(plen, freq_prefill)) / 1000.0
    t_dec = (float(opmodel.iter_time_ms(batch, freq_decode)) / 1000.0) * olen
    wait = max(float(load), 0.0) * t_pre
    e_pre = float(opmodel.prefill_dyn_j_per_token(freq_prefill)) * plen
    e_dec = float(opmodel.dyn_j_per_token(batch, freq_decode)) / 1000.0 * olen
    e_kv = 0.0
    if path == "pd":
        e_kv = kv_xfer_s_per_tok * plen * max(residency_w, 1.0)
    e_res = (wait + t_pre + t_dec) * max(residency_w, 0.0)
    e_extra = extra_gpus * 35.0 * (t_pre + t_dec)
    return e_pre + e_dec + e_kv + e_res + e_extra


def att_would_drop(att_hat: float, att_mixed: float,
                   delta: float = SLO_NON_INFERIOR_PP) -> bool:
    return float(att_hat) + 1e-12 < float(att_mixed) - float(delta)
