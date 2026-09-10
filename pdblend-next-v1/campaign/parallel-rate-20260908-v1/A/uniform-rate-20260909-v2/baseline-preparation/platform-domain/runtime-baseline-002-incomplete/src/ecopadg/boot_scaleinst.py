# -*- coding: utf-8 -*-
"""DynamoLLM / 诊断用 ScaleInst 组合。不是 pdblend 默认启动路径。

pdblend 默认走 planner.choose_boot_slo_layout 的满卡三候选。
本模块只给 dynamollm 对照或显式 compose_boot_slo_layout 诊断旗标。
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from ecopadg.dynamo_planner import (
    GPU_COUNT, PDBLEND_RHO_MAX, PDBLEND_TTFT_FRAC, dataset_shape,
    enumerate_packings, pdblend_profile, predict_packing,
)

# 等墙钟能量模型把 4 卡低频判得比 1×TP8 更省;本节点 n=200 ShareGPT
# 实测 used≥4 的 shrink 节点 J 都高于 mixed。只留 1/2 卡收缩。
MINJ_SHRINK_USED_MAX = 3
from ecopadg.planner import (
    A9_TP, CapacityGuard, HybridGuard, PlanResult, PlanScore,
    _frac_long_of, _slo_ttft_s, hybrid_capacity_ok, hybrid_selective_partition,
    layout_kind, pin_prefill, spatial_pd_capacity_ok, spatial_pd_partition,
    _mixed_full_freq, _role_key,
)
from ecopadg.types import PREFILL_PIN_MHZ, Partition, USER_DATASET_SLO, normalize_dataset


def _slo_tpot_s(predictor) -> float:
    slo = getattr(predictor, "slo", None)
    if slo is None:
        return 0.15
    try:
        return float(getattr(slo, "tpot_s", 0) or 0.15)
    except (TypeError, ValueError):
        return 0.15


def node_energy_j(used: int, freq_mhz: int, rate: float, n: int,
                  profile: dict, layout: str = "mixed") -> float:
    from ecopadg.profiler import node_energy_j as _node_j
    return _node_j(
        int(used), int(freq_mhz), float(rate), int(n), profile,
        gpu_count=int(GPU_COUNT), layout=str(layout))


def spatial_pd_on_used(
        used: int, spatial: Optional[Partition],
        gpu_count: int = 8) -> List[Partition]:
    out: List[Partition] = []
    seen = set()
    if (int(used) == int(gpu_count) and spatial is not None
            and int(spatial.total_gpus()) == int(gpu_count)):
        seen.add(_role_key(spatial))
        out.append(pin_prefill(spatial, PREFILL_PIN_MHZ))
    for role_tp in (1, 2, 4):
        try:
            part = spatial_pd_partition(int(used), int(role_tp))
        except ValueError:
            continue
        part = replace(part, gpus_total=int(gpu_count))
        key = _role_key(part)
        if key in seen:
            continue
        seen.add(key)
        out.append(pin_prefill(part, PREFILL_PIN_MHZ))
    return out


def _score(part: Partition, rate: float, n: int, energy_j: float,
           accepted: bool, reason: str) -> PlanScore:
    cell_s = float(n) / max(float(rate), 1e-6)
    e_js = float(energy_j) / max(cell_s, 1e-6)
    return PlanScore(
        partition=part, attainment=0.97,
        energy_j_per_s=e_js, energy_j_per_req=e_js / max(float(rate), 1e-6),
        power_w=e_js, attainable=True, accepted=bool(accepted),
        reason=str(reason))


def compose_boot_slo_layout(
        predictor, rate: float, att_mixed: float,
        gpu_count: int = 8,
        frac_long: Optional[float] = None,
        mixed_tp: int = 8,
        spatial: Optional[Partition] = None,
        profile: Optional[dict] = None,
        n: int = 200,
        dataset: str = "sharegpt") -> PlanResult:
    fallback_mixed = pin_prefill(
        _mixed_full_freq(int(gpu_count), int(mixed_tp)), PREFILL_PIN_MHZ)
    if float(rate) <= 0.0:
        return PlanResult(
            mode="boot-slo-layout", chosen=fallback_mixed,
            att_mixed=float(att_mixed), fallback="rate-nonpositive",
            used_gpus=int(gpu_count), unused_gpus=0, freq0=2520)

    prof = profile if profile is not None else pdblend_profile(
        dataset=str(dataset))
    rho_max = float(prof.get("rho_max") or PDBLEND_RHO_MAX)
    raw_frac = float(prof.get("ttft_frac") or PDBLEND_TTFT_FRAC)
    ttft_frac = None if raw_frac >= 1.0 - 1e-9 else raw_frac
    ds = normalize_dataset(dataset)
    default_slo = USER_DATASET_SLO.get(ds, USER_DATASET_SLO["sharegpt"])
    slo_ttft = _slo_ttft_s(predictor) or float(default_slo[0])
    slo_tpot = _slo_tpot_s(predictor) or float(default_slo[1])
    pin, pout = dataset_shape(ds)
    fl = _frac_long_of(predictor, frac_long)
    scores: List[PlanScore] = []
    guard: Optional[CapacityGuard] = None
    hybrid_guard: Optional[HybridGuard] = None

    for packing in enumerate_packings():
        sc = predict_packing(
            packing, prof, rate=float(rate), n=int(n), pin=pin, pout=pout,
            slo_ttft_s=float(slo_ttft), slo_tpot_s=float(slo_tpot),
            rho_max=rho_max, ttft_frac=ttft_frac)
        part = Partition(
            n_mixed=int(packing.n_replica), tp_mixed=int(packing.tp),
            pp_mixed=1, freq_mixed=int(packing.freq_mhz),
            gpus_total=int(gpu_count))
        always = (
            packing.n_replica == 1 and packing.tp == 8
            and packing.freq_mhz == 2520)
        ok = bool(sc.slo_ok or always)
        reason = "scaleinst-slo-ok" if sc.slo_ok else (
            "boot-mixed-fallback" if always else sc.reason)
        scores.append(_score(part, rate, n, sc.energy_j, ok, reason))

    last_pd: Optional[CapacityGuard] = None
    for used in sorted({int(s.partition.total_gpus()) for s in scores}):
        for part in spatial_pd_on_used(used, spatial, gpu_count):
            g = spatial_pd_capacity_ok(predictor, rate, part)
            last_pd = g
            energy = node_energy_j(
                part.total_gpus(), 2520, rate, n, prof,
                layout=layout_kind(part))
            scores.append(_score(
                part, rate, n, energy, g.pd_ok,
                g.reason if not g.pd_ok else "capacity-rho-ok"))
    guard = last_pd

    hybrid = hybrid_selective_partition(gpu_count, A9_TP)
    if hybrid is not None:
        hybrid = pin_prefill(hybrid, PREFILL_PIN_MHZ)
        hybrid_guard = hybrid_capacity_ok(predictor, rate, hybrid, fl)
        energy = node_energy_j(
            gpu_count, 2520, rate, n, prof, layout="hybrid-a9")
        scores.append(_score(
            hybrid, rate, n, energy, hybrid_guard.ok, hybrid_guard.reason))

    accepted = [s for s in scores if s.accepted]
    if not accepted:
        chosen = fallback_mixed
        fallback = "boot-mixed-fallback"
    else:
        best = min(accepted, key=lambda s: (
            s.energy_j_per_s, s.partition.total_gpus(),
            -int(s.partition.freq_mixed or 0)))
        chosen = pin_prefill(best.partition, PREFILL_PIN_MHZ)
        kind = layout_kind(chosen)
        if kind == "hybrid-a9":
            fallback = "untrusted-hybrid-a9"
        elif kind == "spatial-pd":
            fallback = "untrusted-capacity-spatial-pd"
        elif chosen.total_gpus() < int(gpu_count):
            fallback = "untrusted-scaleinst-mixed"
        elif (int(chosen.n_mixed) == 1 and int(chosen.tp_mixed) == 8
              and int(chosen.freq_mixed) == 2520
              and best.reason == "boot-mixed-fallback"):
            fallback = (guard.reason if guard is not None and not guard.pd_ok
                        else "boot-mixed-fallback")
        else:
            fallback = "untrusted-scaleinst-mixed"

    used = int(chosen.total_gpus()) if chosen is not None else int(gpu_count)
    is_mixed_tp8 = (
        chosen is not None
        and int(chosen.n_mixed) == 1
        and int(chosen.tp_mixed) == int(mixed_tp)
        and int(chosen.freq_mixed or 0) == 2520
        and chosen.pairs() == 0)
    if used > MINJ_SHRINK_USED_MAX and not is_mixed_tp8:
        chosen = fallback_mixed
        fallback = ("minj-fullnode-mixed" if used >= int(gpu_count)
                    else "minj-vs-mixed")
        used = int(gpu_count)
    freq0 = 2520
    if chosen is not None and int(chosen.n_mixed) > 0 and chosen.pairs() == 0:
        freq0 = int(chosen.freq_mixed or 2520)
    elif chosen is not None and chosen.pairs() > 0:
        # PD/A9 满频起步,地板留 900 让 ScaleFreq 能降 D。
        freq0 = 900
    return PlanResult(
        mode="boot-slo-layout", chosen=chosen, scores=scores,
        att_mixed=float(att_mixed), fallback=fallback, guard=guard,
        hybrid_guard=hybrid_guard, used_gpus=used,
        unused_gpus=max(int(gpu_count) - used, 0), freq0=int(freq0))
