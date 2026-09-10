# -*- coding: utf-8 -*-
"""DistServe 式 ds_pd 布局:(pp_cross, tp_p, pp_p, tp_d, pp_d),满频 2520。

配置空间对齐 simdistserve/benchmarks/search_configs.py::get_distserve_configs。
8 卡等资源只留 GPU 数恰好为 8 的点。选配只认实测(att 达标的最高可持续率,
并列取 2×TP2 PP1)。
"""
from __future__ import annotations

import json
import os
from itertools import product
from typing import List, Optional, Sequence

from ecopadg.planner import same_role_layout
from ecopadg.types import Partition

# Qwen2.5:TP 整除头数,PP 整除层数。度数限制在 1/2/4/8 以便单机拉起。
MODEL_PARALLEL = {
    "7b": dict(layers=28, heads=28),
    "14b": dict(layers=48, heads=40),
    "32b": dict(layers=64, heads=40),
    "72b": dict(layers=80, heads=64),
}
PARALLEL_DEGREES = (1, 2, 4, 8)


def _degrees(n: int) -> List[int]:
    return [d for d in PARALLEL_DEGREES if int(n) % d == 0]


def enumerate_ds_candidates(model_key: str = "14b",
                            gpu_count: int = 8) -> List[dict]:
    """DistServe 五元组,卡数必须恰好 gpu_count。"""
    spec = MODEL_PARALLEL.get(str(model_key).lower()) or MODEL_PARALLEL["14b"]
    tps = _degrees(spec["heads"])
    pps = _degrees(spec["layers"])
    out: List[dict] = []
    seen = set()
    for pp_cross, tp_p, pp_p, tp_d, pp_d in product(
            PARALLEL_DEGREES, tps, pps, tps, pps):
        gpus = pp_cross * (tp_p * pp_p + tp_d * pp_d)
        if gpus != int(gpu_count):
            continue
        if pp_cross * pp_p not in pps or pp_cross * pp_d not in pps:
            continue
        key = (pp_cross, tp_p, pp_p, tp_d, pp_d)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(
            pp_cross=pp_cross, tp_prefill=tp_p, pp_prefill=pp_p,
            tp_decode=tp_d, pp_decode=pp_d, n_segments=pp_cross))
    out.sort(key=lambda d: (
        int(d["pp_prefill"] == 1 and d["pp_decode"] == 1),
        -int(d["pp_cross"]), int(d["tp_prefill"]), int(d["pp_prefill"]),
        int(d["tp_decode"]), int(d["pp_decode"])))
    return out


# 现行实测只搜 PP=1。层切 PP 仍可由 enumerate_ds_candidates 列出,不进本波。
CANDIDATES_14B: List[dict] = [
    dict(pp_cross=2, tp_prefill=2, tp_decode=2, pp_prefill=1, pp_decode=1),
    dict(pp_cross=1, tp_prefill=4, tp_decode=4, pp_prefill=1, pp_decode=1),
    dict(pp_cross=4, tp_prefill=1, tp_decode=1, pp_prefill=1, pp_decode=1),
]
CANDIDATES_72B: List[dict] = [
    dict(pp_cross=1, tp_prefill=4, tp_decode=4, pp_prefill=1, pp_decode=1),
]

DEFAULT_14B = dict(
    pp_cross=2, tp_prefill=2, tp_decode=2, pp_prefill=1, pp_decode=1,
    n_segments=2, freq=2520, source="default-2x-tp2",
    note="synthetic-E1b-not-used; 2xTP2 holds ShareGPT r=4 att=0.975")

DEFAULT_72B = dict(
    pp_cross=1, tp_prefill=4, tp_decode=4, pp_prefill=1, pp_decode=1,
    n_segments=1, freq=2520, source="default-1x-tp4",
    note="8-GPU L20 budget only fits one TP4 prefill/decode pair")


def placement_gpus(d: dict) -> int:
    pp = int(d.get("pp_cross") or 1)
    return pp * (int(d.get("tp_prefill") or 1) * int(d.get("pp_prefill") or 1)
                 + int(d.get("tp_decode") or 1) * int(d.get("pp_decode") or 1))


def validate_runtime_layout(model_key: str, n_mixed: int, n_segments: int,
                            tp_mixed: int, tp_prefill: int, tp_decode: int,
                            gpu_count: int = 8,
                            pp_mixed: int = 1,
                            pp_prefill: int = 1,
                            pp_decode: int = 1,
                            allow_partial: bool = False) -> int:
    """Reject over/under-budget layouts. 世界卡数 = TP×PP。

    默认必须恰好 gpu_count。allow_partial 时允许 1..gpu_count
    (pdblend ScaleInst 少加载);used=0 或超过预算仍拒绝。
    """
    del model_key
    used = (int(n_mixed) * int(tp_mixed) * max(int(pp_mixed), 1)
            + int(n_segments) * (
                int(tp_prefill) * max(int(pp_prefill), 1)
                + int(tp_decode) * max(int(pp_decode), 1)))
    limit = int(gpu_count)
    ok = (1 <= used <= limit) if allow_partial else (used == limit)
    if not ok:
        raise ValueError(
            "非法布局: mixed=%d×TP%dxPP%d + segments=%d×(TP%dxPP%d+TP%dxPP%d)"
            " = %d GPU，不是 %s"
            % (n_mixed, tp_mixed, pp_mixed, n_segments,
               tp_prefill, pp_prefill, tp_decode, pp_decode,
               used, ("1..%d" % limit if allow_partial else str(limit))))
    return used


def partition_from_placement(d: dict, gpu_count: int = 8) -> Partition:
    """pp_cross 份 1P1D,P/D 实例数相等(DistServe 搜索空间)。"""
    n = int(d.get("pp_cross") or d.get("n_segments") or 2)
    tp_p = int(d.get("tp_prefill") or 2)
    tp_d = int(d.get("tp_decode") or 2)
    return Partition(
        n_mixed=0, n_prefill=n, n_decode=n,
        tp_prefill=tp_p, tp_decode=tp_d,
        pp_prefill=int(d.get("pp_prefill") or 1),
        pp_decode=int(d.get("pp_decode") or 1),
        freq_prefill=int(d.get("freq") or 2520),
        freq_decode=int(d.get("freq") or 2520),
        gpus_total=int(gpu_count))


def load_ds_placement(path: str) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        return None
    if "pp_cross" not in d and "n_segments" not in d:
        return None
    return d


def _cand_key(d: dict) -> tuple:
    return (int(d.get("pp_cross") or 0),
            int(d.get("tp_prefill") or 0),
            int(d.get("tp_decode") or 0),
            int(d.get("pp_prefill") or 1),
            int(d.get("pp_decode") or 1))


def pick_measured(candidates: Sequence[dict],
                  measurements: Sequence[dict],
                  att_min: float = 0.90,
                  default: Optional[dict] = None) -> dict:
    """达标者取最高可持续率;并列且含 2×TP2 则取 2×TP2。"""
    by_key = {}
    for m in measurements:
        try:
            att = float(m.get("slo_attainment", 0.0))
            rate = float(m.get("rate", 0.0))
        except (TypeError, ValueError):
            continue
        if att != att or att + 1e-12 < att_min:
            continue
        k = _cand_key(m)
        prev = by_key.get(k)
        if prev is None or rate > prev["rate"]:
            by_key[k] = dict(m, rate=rate, slo_attainment=att)
    ok = []
    for c in candidates:
        hit = by_key.get(_cand_key(c))
        if hit is not None:
            row = dict(c)
            row["max_feasible_rate"] = hit["rate"]
            row["measured_att"] = hit["slo_attainment"]
            ok.append(row)
    if not ok:
        return dict(default or DEFAULT_14B)
    best_rate = max(x["max_feasible_rate"] for x in ok)
    tied = [x for x in ok if x["max_feasible_rate"] >= best_rate - 1e-9]
    prefer = [x for x in tied if _cand_key(x) == (2, 2, 2, 1, 1)]
    win = prefer[0] if prefer else tied[0]
    out = dict(win)
    out["n_segments"] = int(out["pp_cross"])
    out["freq"] = 2520
    out["source"] = "measured-iso-load"
    return out


def default_placement_path(ws: str, model_key: str) -> str:
    return os.path.join(ws, "pdblend", "script", "bench", "results",
                        "ds_placement_%s.json" % model_key)


def is_inherit_layout(current: Partition, place: Optional[dict],
                      gpu_count: int = 8) -> bool:
    if not place:
        return False
    return same_role_layout(current, partition_from_placement(place, gpu_count))
