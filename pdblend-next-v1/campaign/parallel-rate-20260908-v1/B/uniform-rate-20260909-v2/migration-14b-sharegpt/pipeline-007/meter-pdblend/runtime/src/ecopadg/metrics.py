# -*- coding: utf-8 -*-
"""指标:成功/SLO attainment(P50/P90/P99 分桶)、能效统计与逐请求归因。

成功定义:请求同时满足 TTFT SLO 与 TPOT SLO(EcoServe 口径)。
能效:整系统能耗(梯形积分,含驻留)+ 模型归因逐请求能耗(校准后报残差)。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ecopadg.types import ENERGY_PAD_S, SLO_NON_INFERIOR_PP, SloSpec
from ecopadg.measure.power import trapezoid_energy, trapezoid_mean_power

PERCENTILES = (50, 90, 99)


def percentiles(vals: Sequence[float], ps: Sequence[int] = PERCENTILES) -> Dict[int, float]:
    """np.percentile linear 口径(与 EcoServe bench 客户端一致)。"""
    if not vals:
        return {int(p): float("nan") for p in ps}
    arr = np.asarray([float(v) for v in vals if v == v], dtype=float)
    if arr.size == 0:
        return {int(p): float("nan") for p in ps}
    return {int(p): float(np.percentile(arr, p)) for p in ps}


def slo_attainment(rows: Sequence[dict], slo: SloSpec, *, n_expected: int = 0) -> float:
    """达标率 = 同时满足 TTFT/TPOT SLO 的请求占比。"""
    n = max(len(rows), int(n_expected))
    if not n:
        return float("nan")
    ok = sum(1 for r in rows if classify_row_slo(r, slo))
    return ok / n


def classify_row_slo(r: dict, slo: SloSpec) -> bool:
    if row_has_error(r):
        return False
    ttft = r.get("ttft_s")
    tpot = r.get("tpot_s")
    if ttft is None or tpot is None or ttft == "" or tpot == "":
        # Legacy attribution-only rows have no latency fields. Measured rows
        # must always be reclassified against the supplied SLO, not a CSV flag.
        return ("ttft_s" not in r and "tpot_s" not in r
                and str(r.get("slo_ok", "")).lower() in ("1", "true"))
    try:
        return (0 <= float(ttft) < slo.ttft_s
                and 0 <= float(tpot) < slo.tpot_s)
    except (ValueError, TypeError):
        return False


def row_has_error(r: dict) -> bool:
    """controller / HTTP 错误或空流:不得记为成功。"""
    if "success" in r and str(r["success"]).lower() in ("0", "false", "no"):
        return True
    for k in ("error", "controller_error", "http_error"):
        v = r.get(k)
        if v not in (None, "", "0", 0, False):
            return True
    if str(r.get("empty_stream", "")).lower() in ("1", "true", "yes"):
        return True
    return False


def emitted_output_tokens(r: dict) -> int:
    """真实发出 token;禁止用 trace 目标长度冒充。"""
    for k in ("emitted_tokens", "generated_tokens", "output_tokens",
              "n_out", "completion_tokens"):
        v = r.get(k)
        if v not in (None, ""):
            try:
                return max(int(v), 0)
            except (TypeError, ValueError):
                continue
    try:
        return max(int(r.get("output_len") or 0), 0)
    except (TypeError, ValueError):
        return 0


def emitted_input_tokens(r: dict) -> int:
    for k in ("prompt_tokens", "input_tokens", "n_in"):
        v = r.get(k)
        if v not in (None, ""):
            try:
                return max(int(v), 0)
            except (TypeError, ValueError):
                continue
    try:
        return max(int(r.get("prompt_len") or 0), 0)
    except (TypeError, ValueError):
        return 0


def clip_power_window(
        rows: Sequence[Tuple[float, Sequence[float]]],
        t_start: Optional[float], t_end: Optional[float],
        pad_s: float = ENERGY_PAD_S
) -> List[Tuple[float, Sequence[float]]]:
    """Clip at interpolated boundaries; never substitute an unrelated window."""
    if not rows or t_start is None or t_end is None:
        return list(rows)
    lo, hi = float(t_start) - float(pad_s), float(t_end) + float(pad_s)
    if hi <= lo or lo < rows[0][0] or hi > rows[-1][0]:
        raise ValueError("power samples do not cover the observation window")
    def at(t):
        for (a, wa), (b, wb) in zip(rows, rows[1:]):
            if a <= t <= b and b > a:
                if len(wa) != len(wb):
                    raise ValueError("GPU count changed in power samples")
                q = (t - a) / (b - a)
                return (t, [float(x) + q * (float(y) - float(x))
                            for x, y in zip(wa, wb)])
        raise ValueError("non-monotonic power timestamps")
    return [at(lo)] + [(t, w) for t, w in rows if lo < t < hi] + [at(hi)]


def bench_time_bounds(bench_rows: Sequence[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Actual observation bounds; duration-only rows cannot define a window."""
    starts, ends = [], []
    for r in bench_rows:
        for k in ("arrival_s", "start_s", "t_arrive"):
            v = r.get(k)
            if v not in (None, "") and math.isfinite(_as_float(v)):
                starts.append(float(v))
                break
        for k in ("finish_s", "end_s", "complete_s", "t_done"):
            v = r.get(k)
            if v not in (None, "") and math.isfinite(_as_float(v)):
                ends.append(float(v))
                break
        lat = r.get("latency_s")
        arr = r.get("arrival_s")
        if (math.isfinite(_as_float(lat)) and _as_float(lat) >= 0
                and math.isfinite(_as_float(arr))):
            ends.append(float(arr) + float(lat))
    if not starts or not ends:
        return None, None
    t0 = min(starts)
    t1 = max(max(ends), max(starts))
    return t0, t1


def classify_run_validity(bench_rows: Sequence[dict], *,
                          n_expected: int = 0,
                          controller_errors: int = 0,
                          error_frac_invalid: float = 0.05) -> str:
    """invalid_run / ok。controller 大面积错误不得当性能点。"""
    n = len(bench_rows)
    errs = int(controller_errors) + sum(1 for r in bench_rows if row_has_error(r))
    if not n or (n_expected > 0 and n != int(n_expected)):
        return "invalid_run"
    ids = [str(r.get("request_id", r.get("idx", r.get("rid"))))
           for r in bench_rows if any(k in r for k in ("request_id", "idx", "rid"))]
    if len(ids) != len(set(ids)):
        return "invalid_run"
    if n and errs / n >= error_frac_invalid:
        return "invalid_run"
    if n_expected > 0 and errs >= int(0.05 * n_expected) and errs >= 5:
        return "invalid_run"
    return "ok"


def slo_non_inferior(att_ours: float, att_baseline: float,
                     delta: float = SLO_NON_INFERIOR_PP) -> bool:
    if att_ours != att_ours or att_baseline != att_baseline:
        return False
    return float(att_ours) + 1e-12 >= float(att_baseline) - float(delta)


def att_delta_vs_baseline(att_ours: float, att_baseline: float) -> float:
    return float(att_ours) - float(att_baseline)


def iso_load_key(row: dict) -> tuple:
    """同负载单元键:数据集/到达率/seed。n 与 GPU 在组内校验必须一致。"""
    return (str(row.get("dataset") or ""),
            _as_float(row.get("rate"), float("nan")),
            str(row.get("seed") or "0"))


def align_iso_load(cells: Sequence[dict],
                   baseline_system: str = "mixed",
                   delta: float = SLO_NON_INFERIOR_PP) -> List[dict]:
    """按同负载键对齐;写 att_delta_vs_baseline 与 validity。

    禁止跨 rate 摘点。n 或 GPU 不一致的组标 invalid_load。
    """
    by: Dict[tuple, List[dict]] = {}
    for r in cells:
        by.setdefault(iso_load_key(r), []).append(dict(r))
    out: List[dict] = []
    for key, rs in sorted(by.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        base = next((x for x in rs
                     if str(x.get("system", "")) == baseline_system), None)
        att_b = _as_float(base.get("slo_attainment")) if base else float("nan")
        base_n = (int(_as_float(base.get("n_requests") or base.get("n"), 0))
                  if base is not None else None)
        base_g = (int(_as_float(base.get("gpu_count") or base.get("gpus"), 0))
                  if base is not None else None)
        ns = {int(_as_float(x.get("n_requests") or x.get("n"), 0)) for x in rs}
        gpus = {int(_as_float(x.get("gpu_count") or x.get("gpus"), 0))
                for x in rs}
        group_load_ok = (len(ns) <= 1 and len(gpus) <= 1)
        for x in rs:
            row = dict(x)
            att = _as_float(row.get("slo_attainment"))
            row["att_baseline"] = att_b
            row["att_delta_vs_baseline"] = (
                att_delta_vs_baseline(att, att_b) if att == att and att_b == att_b
                else float("nan"))
            validity = str(row.get("validity") or "ok")
            row_n = int(_as_float(row.get("n_requests") or row.get("n"), 0))
            row_g = int(_as_float(row.get("gpu_count") or row.get("gpus"), 0))
            if base is not None:
                load_ok = (row_n == base_n and row_g == base_g)
            else:
                load_ok = group_load_ok
            if not load_ok:
                validity = "invalid_load"
            elif str(row.get("system", "")) != baseline_system and base is not None:
                if validity == "ok" and not slo_non_inferior(att, att_b, delta):
                    validity = "invalid_slo"
            row["validity"] = validity
            row["iso_load_ok"] = validity == "ok"
            out.append(row)
    return out


def latency_bucket_attainment(rows: Sequence[dict], slo: SloSpec) -> dict:
    """按端到端 latency 分桶 [0,P50) [P50,P90) [P90,P99) [P99,∞) 的达标率。"""
    lats = [float(r["latency_s"]) for r in rows
            if r.get("latency_s") not in (None, "")]
    if lats:
        p = percentiles(lats, ps=(50, 90, 99))
    else:
        p = {50: float("nan"), 90: float("nan"), 99: float("nan")}
    edges = [p[50], p[90], p[99]]
    buckets = [dict(lo=0.0, hi=edges[0], n=0, attainment=float("nan")),
               dict(lo=edges[0], hi=edges[1], n=0, attainment=float("nan")),
               dict(lo=edges[1], hi=edges[2], n=0, attainment=float("nan")),
               dict(lo=edges[2], hi=float("inf"), n=0, attainment=float("nan"))]
    for r in rows:
        lat = r.get("latency_s")
        if lat in (None, ""):
            continue
        lat = float(lat)
        for i, b in enumerate(buckets):
            if (lat >= b["lo"] or i == 0) and lat < b["hi"]:
                b["n"] += 1
                if "ok" not in b:
                    b["ok"] = 0
                if classify_row_slo(r, slo):
                    b["ok"] += 1
                break
    for b in buckets:
        if b["n"] > 0:
            b["attainment"] = b.get("ok", 0) / b["n"]
    return dict(buckets=buckets, edges=edges)


def summarize_bench(rows: Sequence[dict], slo: SloSpec, *, n_expected: int = 0) -> dict:
    """run 级汇总:完成数、TTFT/TPOT 分位、attainment、吞吐。"""
    ttfts = [float(r["ttft_s"]) for r in rows
             if r.get("ttft_s") not in (None, "")]
    tpots = [float(r["tpot_s"]) for r in rows
             if r.get("tpot_s") not in (None, "")]
    p_ttft = percentiles(ttfts)
    p_tpot = percentiles(tpots)
    in_toks = sum(emitted_input_tokens(r) for r in rows)
    out_toks = sum(emitted_output_tokens(r) for r in rows)
    t0, t1 = bench_time_bounds(rows)
    dur = t1 - t0 if t0 is not None and t1 is not None else float("nan")
    completed = sum(not row_has_error(r) for r in rows)
    ttft_avg = float(np.mean(ttfts)) if ttfts else float("nan")
    tpot_avg = float(np.mean(tpots)) if tpots else float("nan")
    return dict(
        completed=completed,
        n_expected=max(len(rows), int(n_expected)),
        slo_attainment=slo_attainment(rows, slo, n_expected=n_expected),
        ttft_avg_s=ttft_avg, tpot_avg_s=tpot_avg,
        ttft_p50_s=p_ttft[50], ttft_p90_s=p_ttft[90], ttft_p99_s=p_ttft[99],
        tpot_p50_s=p_tpot[50], tpot_p90_s=p_tpot[90], tpot_p99_s=p_tpot[99],
        input_tokens=in_toks, output_tokens=out_toks,
        duration_s=dur,
        req_throughput=(completed / dur if dur == dur and dur > 0
                        else float("nan")),
        goodput_rps=(sum(classify_row_slo(r, slo) for r in rows) / dur
                     if dur == dur and dur > 0 else float("nan")),
        output_tok_throughput=(out_toks / dur if dur == dur and dur > 0
                               else float("nan")),
        buckets=latency_bucket_attainment(rows, slo),
    )


def gpu_busy_frac(rows: Sequence[Tuple[float, Sequence[float]]],
                  idle_w: float = 35.0) -> float:
    """8 卡平均非空闲时间占比:功率 > idle_w 的样本比例再对卡取均值。"""
    if not rows:
        return float("nan")
    n_gpu = max((len(ws) for _t, ws in rows), default=0)
    if n_gpu <= 0:
        return float("nan")
    busy = [0] * n_gpu
    total = [0] * n_gpu
    thresh = float(idle_w)
    for _t, ws in rows:
        for i, w in enumerate(ws):
            if i >= n_gpu:
                break
            try:
                val = float(w)
            except (TypeError, ValueError):
                continue
            if val != val:
                continue
            total[i] += 1
            if val > thresh:
                busy[i] += 1
    fracs = [busy[i] / float(total[i]) for i in range(n_gpu) if total[i]]
    if not fracs:
        return float("nan")
    return float(sum(fracs) / len(fracs))


def energy_from_power_rows(rows: Sequence[Tuple[float, Sequence[float]]]) -> dict:
    """功率行 [(t_s, [每卡 W...])] → 总能耗/平均功率(梯形积分,Σ卡)。"""
    total = trapezoid_energy(list(rows))
    mean = trapezoid_mean_power(list(rows))
    return dict(total_j=total, mean_w=mean)


def attribute_energy(bench_rows: Sequence[dict], opmodel,
                     freq_prefill: int, freq_decode: int,
                     batch_hat: int) -> List[float]:
    """模型归因逐请求动态能耗(J):prefill_tok×e_pre + output_tok×e_dec。"""
    e_pre = float(opmodel.prefill_dyn_j_per_token(freq_prefill))
    e_dec = float(opmodel.dyn_j_per_token(batch_hat, freq_decode)) / 1000.0
    out = []
    for r in bench_rows:
        j = (emitted_input_tokens(r) * e_pre
             + emitted_output_tokens(r) * e_dec)
        out.append(float(j))
    return out


_DEFAULT_SLO = SloSpec()


def _as_float(v, default: float = float("nan")) -> float:
    try:
        if v in (None, "", "nan"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def caliber_b_working_points(cells: Sequence[dict],
                             min_attainment: float = 0.9,
                             rate_key: str = "rate",
                             att_key: str = "slo_attainment") -> List[dict]:
    """附录:每系统最大可行速率。不是同负载主口径,禁止作发表门槛。

    cells 需含 system / rate / slo_attainment,以及可选能耗字段。
    无可行点时 feasible=False,取 attainment 最高者作参考(不标为工作点)。
    """
    by: Dict[str, List[dict]] = {}
    for r in cells:
        by.setdefault(str(r.get("system", "")), []).append(dict(r))
    out: List[dict] = []
    for sysn, rs in sorted(by.items()):
        feas = [x for x in rs
                if _as_float(x.get(att_key)) >= min_attainment]
        if feas:
            best = max(feas, key=lambda x: _as_float(x.get(rate_key), -1.0))
            row = dict(best)
            row.update(system=sysn, feasible=True,
                       min_attainment=min_attainment)
            out.append(row)
            continue
        if not rs:
            continue
        ref = max(rs, key=lambda x: (_as_float(x.get(att_key), -1.0),
                                     _as_float(x.get(rate_key), -1.0)))
        row = dict(ref)
        row.update(system=sysn, feasible=False,
                   min_attainment=min_attainment)
        out.append(row)
    return out


def energy_summary(power_rows: Sequence[Tuple[float, Sequence[float]]],
                   bench_rows: Sequence[dict], opmodel,
                   freq_prefill: int, freq_decode: int,
                   batch_hat: int,
                   residency_w: Optional[float] = None,
                   slo: Optional[SloSpec] = None,
                   utilization_rows: Sequence[Tuple[float, Sequence[float]]] = ()) -> dict:
    """能效汇总:总能耗、J/req、J/token、逐请求归因 P50/90/99、残差、goodput/W。"""
    slo = slo or _DEFAULT_SLO
    e_pow = energy_from_power_rows(power_rows)
    total_j = e_pow["total_j"]
    mean_w = e_pow["mean_w"]
    n = len(bench_rows)
    out_toks = sum(emitted_output_tokens(r) for r in bench_rows)
    in_toks = sum(emitted_input_tokens(r) for r in bench_rows)
    per_req = attribute_energy(bench_rows, opmodel, freq_prefill,
                               freq_decode, batch_hat)
    predicted = float(np.sum(per_req)) if per_req else 0.0
    res = float(residency_w) if residency_w is not None else 0.0
    dur = (power_rows[-1][0] - power_rows[0][0]) if len(power_rows) > 1 else 0.0
    resid_j = total_j - predicted - res * dur
    calibration = (predicted / max(total_j - res * dur, 1e-9)
                   if total_j > 0 else float("nan"))
    attained = [per_req[i] for i, r in enumerate(bench_rows)
                if classify_row_slo(r, slo)]
    attained_n = sum(1 for r in bench_rows if classify_row_slo(r, slo))
    if attained:
        p_attained = percentiles(attained)
    else:
        p_attained = {50: float("nan"), 90: float("nan"), 99: float("nan")}
    p_all = percentiles(per_req)
    slo_goodput = (attained_n / dur) if dur > 0 else float("nan")
    goodput_per_w = (slo_goodput / mean_w
                     if mean_w and mean_w == mean_w and mean_w > 0
                     else float("nan"))
    idle = float(residency_w) if residency_w is not None and residency_w == residency_w else 35.0
    if idle <= 0:
        idle = 35.0
    busy_frac = gpu_busy_frac(power_rows, idle)
    gpu_util = float("nan")
    if len(utilization_rows) >= 2:
        n_gpu = len(utilization_rows[0][1])
        if n_gpu and all(len(ws) == n_gpu and all(
                math.isfinite(float(v)) and 0 <= float(v) <= 100 for v in ws)
                for _, ws in utilization_rows):
            gpu_util = trapezoid_mean_power(utilization_rows) / (n_gpu * 100.0)
    return dict(
        total_j=total_j,
        mean_w=mean_w,
        duration_s=dur,
        n_requests=n,
        attained_n=attained_n,
        j_per_req=(total_j / n if n else float("nan")),
        j_per_output_token=(total_j / out_toks if out_toks else float("nan")),
        j_per_input_token=(total_j / in_toks if in_toks else float("nan")),
        j_per_total_token=(total_j / (in_toks + out_toks)
                           if (in_toks + out_toks) else float("nan")),
        j_per_total_gross=(total_j / (in_toks + out_toks)
                           if (in_toks + out_toks) else float("nan")),
        emitted_output_tokens=out_toks,
        emitted_input_tokens=in_toks,
        slo_goodput=slo_goodput,
        goodput_per_w=goodput_per_w,
        gpu_util=gpu_util,
        power_busy_frac=busy_frac,
        predicted_dyn_j=predicted,
        residency_j=res * dur,
        residual_j=resid_j,
        calibration=calibration,
        per_req_j=per_req,
        all_per_req_p=dict(p50_j=p_all[50], p90_j=p_all[90], p99_j=p_all[99]),
        attained_per_req_j=dict(p50_j=p_attained[50], p90_j=p_attained[90],
                                p99_j=p_attained[99]),
    )
