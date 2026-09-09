# -*- coding: utf-8 -*-
"""DynamoLLM warehouse planner: boot ScaleInst + ScaleShard, online ScaleFreq.

Paper (HPCA'25 §IV) uses 30 min scale-in, 5 min reshard, and ~5 s ScaleFreq.
An n=200 cell cannot run those epochs, and L20 remat is ~55 s / 41 kJ.
This module does the paper search once at boot:

  enumerate homogeneous k×TP (k·TP ≤ 8) × freq
  keep rows that meet TTFT/TPOT under the cell's known Poisson rate
  minimize execution + empty-idle energy on unused GPUs

No MultiPool, no BERT length predictor, no mid-cell remat.
``source=opmodel`` when E1b tables load; otherwise a deterministic builtin
14B colocated table. The chosen (N, TP, f) is what the cell actually runs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

from ecopadg.profiler import SIZE_BUCKETS
from ecopadg.types import DECODE_B_CAP, USER_DATASET_SLO, normalize_dataset

_DS_SIZE_BUCKET = {
    "alpaca": "short", "sharegpt": "medium", "longbench": "long",
}

GPU_COUNT = 8
IDLE_EMPTY_W = 35.0
LOADED_IDLE_W = 76.0
PEAK_GPU_W = 130.0
RHO_MAX = 0.90
# pdblend 绝对 min J:SLO 只作 Dynamo 同款模型过滤,不保证实测 att。
PDBLEND_RHO_MAX = 0.90
PDBLEND_TTFT_FRAC = 1.0
PDBLEND_PROFILE_PURPOSE = "pdblend-boot-scaleinst"
# E1b decode mean_w 是 TP2 两卡之和(400–500 W),不能当单卡。
PDBLEND_GPU_W_MIN = 70.0
PDBLEND_GPU_W_MAX = 140.0
# r=2 1×TP1 格子 power.csv:空卡 gpu1–7 均值 ~66.9 W(未加载 L20)。
PDBLEND_IDLE_EMPTY_W = 67.0
PDBLEND_IDLE_SOURCE = "measured-unused-l20-r2-n200"
PDBLEND_ATT_GATE_PP = 0.01
DECODE_BATCH = 16
FREQS = (900, 1500, 2100, 2520)
# Homogeneous packings only (paper ScaleShard on one pool).
PACKINGS: Tuple[Tuple[int, int], ...] = (
    (1, 1), (2, 1), (4, 1), (8, 1),
    (1, 2), (2, 2), (4, 2),
    (1, 4), (2, 4),
    (1, 8),
)
PROFILE_SOURCE_OPMODEL = "opmodel"
PROFILE_SOURCE_BUILTIN = "builtin-14b"


def dataset_shape(dataset: str) -> Tuple[int, int]:
    """(pin, pout) from the Dynamo-style size bucket, not a 9-way SS…LL split."""
    key = normalize_dataset(dataset)
    bucket = _DS_SIZE_BUCKET.get(key, "medium")
    pin, pout = SIZE_BUCKETS[bucket]
    return int(pin), int(pout)


def tp_prefill_scale(tp: int, base_tp: int = 2) -> float:
    """Prefill wall time relative to the E1b base TP. Compute-bound + comm."""
    t = max(int(tp), 1)
    b = max(int(base_tp), 1)
    comm = 1.0 + 0.08 * max(t - 1, 0)
    return (float(b) / float(t)) * comm


def tp_decode_scale(tp: int, base_tp: int = 2) -> float:
    """Decode step time relative to the E1b base TP. Weakly TP-sensitive."""
    t = max(int(tp), 1)
    b = max(int(base_tp), 1)
    return (float(b) / float(t)) ** 0.20


def gpu_power_w(freq_mhz: int, loaded_idle_w: float = LOADED_IDLE_W,
                peak_w: float = PEAK_GPU_W) -> float:
    frac = min(max(float(freq_mhz) / 2520.0, 0.0), 1.0)
    return float(loaded_idle_w) + (float(peak_w) - float(loaded_idle_w)) * (
        frac ** 1.5)


def builtin_profile(model_key: str = "14b") -> dict:
    """Deterministic colocated 14B-TP2-shaped table. Not a measured sweep."""
    freqs = {}
    # TP2 @ 2520: ~90 ms / 512 tok, ~23 ms iter @ b=8.
    for freq in FREQS:
        slow = 2520.0 / float(freq)
        freqs[str(int(freq))] = dict(
            prefill_s_per_tok=0.000176 * slow,
            iter_s_b8=0.023 * (0.65 + 0.35 * slow),
            gpu_w=gpu_power_w(int(freq)),
        )
    return dict(
        model=str(model_key),
        source=PROFILE_SOURCE_BUILTIN,
        base_tp=2,
        idle_empty_w=IDLE_EMPTY_W,
        loaded_idle_w=LOADED_IDLE_W,
        freqs=freqs,
    )


def synthesize_profile_from_opmodel(opmodel, model_key: str = "14b",
                                    pin: Optional[int] = None,
                                    dataset: Optional[str] = None) -> dict:
    """Scale the E1b TP2 OpModel across freq. Prefill 用数据集/给定 pin,不写死 512。"""
    if pin is None:
        pin, _pout = dataset_shape(dataset or "sharegpt")
    pin = max(int(pin), 1)
    freqs = {}
    for freq in FREQS:
        try:
            pre_ms = float(opmodel.prefill_time_ms(int(pin), int(freq)))
            iter_ms = float(opmodel.iter_time_ms(8, int(freq)))
        except (TypeError, ValueError):
            continue
        try:
            board = float(opmodel.power_w("decode", int(freq)))
        except (TypeError, ValueError, AttributeError):
            board = float("nan")
        if board != board or board <= 0.0:
            board = gpu_power_w(int(freq))
        freqs[str(int(freq))] = dict(
            prefill_s_per_tok=max(pre_ms / 1000.0 / float(pin), 1e-6),
            iter_s_b8=max(iter_ms / 1000.0, 1e-4),
            gpu_w=float(board),
        )
    if not freqs:
        raise ValueError("opmodel produced no freq rows")
    return dict(
        model=str(model_key),
        source=PROFILE_SOURCE_OPMODEL,
        base_tp=2,
        shape_pin=int(pin),
        shape_dataset=str(dataset or ""),
        idle_empty_w=IDLE_EMPTY_W,
        loaded_idle_w=LOADED_IDLE_W,
        freqs=freqs,
    )


def load_or_build_profile(model_key: str = "14b",
                          profile_path: str = "",
                          opmodel=None,
                          dataset: Optional[str] = None,
                          pin: Optional[int] = None) -> dict:
    if profile_path:
        with open(profile_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "freqs" not in data:
            raise ValueError("invalid dynamo profile: %s" % profile_path)
        return data
    if opmodel is not None:
        try:
            return synthesize_profile_from_opmodel(
                opmodel, model_key, pin=pin, dataset=dataset)
        except ValueError:
            pass
    try:
        from ecopadg.opmodel import load_opmodel
        om = load_opmodel(model_key)
    except Exception:
        om = None
    if om is not None:
        try:
            return synthesize_profile_from_opmodel(
                om, model_key, pin=pin, dataset=dataset)
        except ValueError:
            pass
    return builtin_profile(model_key)


def per_gpu_decode_w(board: float, base_tp: int, freq_mhz: int) -> float:
    """单卡忙功率。两卡之和 / TP 后仍越界则回退 L20 76–130 公式。"""
    curve = gpu_power_w(int(freq_mhz))
    try:
        raw = float(board)
    except (TypeError, ValueError):
        return curve
    if raw != raw or raw <= 0.0:
        return curve
    per = raw / max(int(base_tp), 1)
    if PDBLEND_GPU_W_MIN - 1e-9 <= per <= PDBLEND_GPU_W_MAX + 1e-9:
        return float(per)
    return curve


def apply_pdblend_power(profile: dict) -> dict:
    """把 E1b 两卡功率收成单卡,闲卡用未加载 L20 实测。"""
    out = dict(profile)
    base_tp = int(out.get("base_tp") or 2)
    freqs = {}
    for key, row in (out.get("freqs") or {}).items():
        item = dict(row)
        item["gpu_w"] = per_gpu_decode_w(
            item.get("gpu_w"), base_tp, int(key))
        freqs[str(key)] = item
    out["freqs"] = freqs
    try:
        idle = float(out.get("idle_empty_w") or 0.0)
    except (TypeError, ValueError):
        idle = 0.0
    if idle < 50.0 or idle > 100.0:
        out["idle_empty_w"] = float(PDBLEND_IDLE_EMPTY_W)
        out["idle_empty_source"] = PDBLEND_IDLE_SOURCE
    out["power_unit"] = "per-gpu"
    return out


def pdblend_profile(model_key: str = "14b",
                    profile_path: str = "",
                    opmodel=None,
                    dataset: str = "",
                    pin: Optional[int] = None) -> dict:
    """E1b 时延 + 单卡功率。ρ/TTFT 与 Dynamo 同款。禁止挂 lookup.json。"""
    if profile_path and "lookup.json" in os.path.basename(profile_path):
        raise ValueError("pdblend profile must not read lookup.json")
    data = load_or_build_profile(
        model_key, profile_path=profile_path, opmodel=opmodel,
        dataset=dataset, pin=pin)
    if data.get("lookup") not in (None, False, {}, ""):
        raise ValueError("pdblend profile must not carry a lookup table")
    out = apply_pdblend_power(dict(data))
    out["purpose"] = PDBLEND_PROFILE_PURPOSE
    out["rho_max"] = float(PDBLEND_RHO_MAX)
    out["ttft_frac"] = float(PDBLEND_TTFT_FRAC)
    out["lookup"] = False
    return out


def tp1_row_from_measured(
        ttft_s: float, tpot_s: float, att: float, *,
        pin: int = 512, mixed_att_ref: float = 0.97,
        source: str = "n80-1x-tp1") -> dict:
    """n=80 1×TP1 实测 → TP1 行。比 1×TP8 差 >1 pp 则 att_gate。"""
    ref = float(mixed_att_ref)
    measured = float(att)
    gated = (ref == ref and measured == measured
             and measured + 1e-12 < ref - PDBLEND_ATT_GATE_PP)
    return dict(
        prefill_s_per_tok=max(float(ttft_s) / max(float(pin), 1.0), 1e-6),
        iter_s_b8=max(float(tpot_s), 1e-4),
        att=measured,
        mixed_att_ref=ref,
        slo_ok=not gated,
        att_gate=gated,
        source=str(source),
    )


def apply_tp1_overlay(profile: dict, tp1_rows: dict) -> dict:
    """写入 freqs 旁的 tp1 覆盖,predict_packing 对 TP=1 优先用它。"""
    out = dict(profile)
    merged = dict(out.get("tp1") or {})
    for key, row in (tp1_rows or {}).items():
        merged[str(int(key))] = dict(row)
    out["tp1"] = merged
    return out


def write_profile(profile: dict, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
        fh.write("\n")


@dataclass(frozen=True)
class DynamoPacking:
    n_replica: int
    tp: int
    freq_mhz: int

    @property
    def used_gpus(self) -> int:
        return int(self.n_replica) * int(self.tp)

    @property
    def unused_gpus(self) -> int:
        return GPU_COUNT - self.used_gpus


@dataclass
class DynamoScore:
    packing: DynamoPacking
    ttft_s: float
    tpot_s: float
    rho: float
    mu: float
    energy_j: float
    active_w: float
    idle_w: float
    slo_ok: bool
    reason: str = ""


@dataclass
class DynamoPlan:
    n_replica: int
    tp: int
    freq0: int
    unused_gpus: int
    used_gpus: int
    source: str
    reason: str
    energy_j: float
    ttft_s: float
    tpot_s: float
    rho: float
    slo_ok: bool
    scores: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["implementation"] = "dynamollm-scaleinst-shard-freq"
        payload["mid_cell_remat"] = False
        payload["multipool"] = False
        payload["gpu_count"] = GPU_COUNT
        return payload

    def shell_exports(self) -> str:
        return (
            "MIXED_TP=%d\nMIXED_N_REPLICA=%d\nFREQ0=%d\nUNUSED_GPUS=%d\n"
            % (int(self.tp), int(self.n_replica), int(self.freq0),
               int(self.unused_gpus))
        )


def enumerate_packings(
        freqs: Sequence[int] = FREQS,
        packings: Sequence[Tuple[int, int]] = PACKINGS,
) -> List[DynamoPacking]:
    out: List[DynamoPacking] = []
    for n_replica, tp in packings:
        used = int(n_replica) * int(tp)
        if used < 1 or used > GPU_COUNT:
            continue
        for freq in freqs:
            out.append(DynamoPacking(
                n_replica=int(n_replica), tp=int(tp), freq_mhz=int(freq)))
    return out


def _freq_row(profile: dict, freq_mhz: int) -> dict:
    freqs = profile.get("freqs") or {}
    key = str(int(freq_mhz))
    if key in freqs:
        return dict(freqs[key])
    # nearest
    keys = sorted(int(k) for k in freqs)
    if not keys:
        raise ValueError("profile has no freqs")
    nearest = min(keys, key=lambda f: abs(f - int(freq_mhz)))
    return dict(freqs[str(nearest)])


def predict_packing(packing: DynamoPacking, profile: dict, *,
                    rate: float, n: int, pin: int, pout: int,
                    slo_ttft_s: float, slo_tpot_s: float,
                    batch: int = DECODE_BATCH,
                    rho_max: Optional[float] = None,
                    ttft_frac: Optional[float] = None,
                    use_tp1: bool = False) -> DynamoScore:
    row = _freq_row(profile, packing.freq_mhz)
    base_tp = int(profile.get("base_tp") or 2)
    idle_empty = float(profile.get("idle_empty_w") or IDLE_EMPTY_W)
    tp1 = (profile.get("tp1") or {}).get(str(int(packing.freq_mhz))) or {}
    apply_tp1 = bool(use_tp1) and int(packing.tp) == 1 and tp1
    if apply_tp1 and tp1.get("prefill_s_per_tok") and tp1.get("iter_s_b8"):
        t_pre = float(tp1["prefill_s_per_tok"]) * float(pin)
        t_iter = float(tp1["iter_s_b8"])
    else:
        t_pre = (float(row["prefill_s_per_tok"]) * float(pin)
                 * tp_prefill_scale(packing.tp, base_tp))
        t_iter = float(row["iter_s_b8"]) * tp_decode_scale(packing.tp, base_tp)
    b = max(1, min(int(batch), int(DECODE_B_CAP)))
    mu_pre = packing.n_replica / max(t_pre, 1e-6)
    mu_dec = packing.n_replica * b / max(t_iter * float(pout), 1e-6)
    mu = min(mu_pre, mu_dec)
    rho = float(rate) / mu if mu > 0 else float("inf")
    cell_s = float(n) / max(float(rate), 1e-6)
    gpu_w = float(row.get("gpu_w") or gpu_power_w(packing.freq_mhz))
    active_w = packing.used_gpus * gpu_w
    idle_w = packing.unused_gpus * idle_empty
    energy_j = (active_w + idle_w) * cell_s
    reasons = []
    rho_lim = float(RHO_MAX if rho_max is None else rho_max)
    ttft_lim = float(slo_ttft_s)
    if ttft_frac is not None:
        ttft_lim = ttft_lim * max(float(ttft_frac), 0.0)
    if rho >= rho_lim:
        reasons.append("rho")
    if t_pre > ttft_lim:
        reasons.append("ttft")
    if t_iter > float(slo_tpot_s):
        reasons.append("tpot")
    if apply_tp1 and tp1.get("slo_ok") is False:
        reasons.append("att_gate")
    return DynamoScore(
        packing=packing,
        ttft_s=t_pre,
        tpot_s=t_iter,
        rho=rho,
        mu=mu,
        energy_j=energy_j,
        active_w=active_w,
        idle_w=idle_w,
        slo_ok=not reasons,
        reason=",".join(reasons) or "ok",
    )


def choose_dynamo_plan(
        rate: float,
        n: int = 200,
        dataset: str = "sharegpt",
        slo_ttft_s: Optional[float] = None,
        slo_tpot_s: Optional[float] = None,
        profile: Optional[dict] = None,
        model_key: str = "14b",
) -> DynamoPlan:
    if float(rate) <= 0.0:
        return DynamoPlan(
            n_replica=1, tp=8, freq0=2520, unused_gpus=0, used_gpus=8,
            source="rate-missing-fallback-tp8",
            reason="rate-nonpositive",
            energy_j=float("nan"), ttft_s=float("nan"), tpot_s=float("nan"),
            rho=float("nan"), slo_ok=False)
    ds = normalize_dataset(dataset)
    if slo_ttft_s is None or slo_tpot_s is None:
        ttft, tpot = USER_DATASET_SLO.get(ds, USER_DATASET_SLO["sharegpt"])
        slo_ttft_s = float(slo_ttft_s if slo_ttft_s is not None else ttft)
        slo_tpot_s = float(slo_tpot_s if slo_tpot_s is not None else tpot)
    prof = profile if profile is not None else load_or_build_profile(
        model_key, dataset=ds)
    pin, pout = dataset_shape(ds)
    scores = [
        predict_packing(
            packing, prof, rate=float(rate), n=int(n), pin=pin, pout=pout,
            slo_ttft_s=float(slo_ttft_s), slo_tpot_s=float(slo_tpot_s))
        for packing in enumerate_packings()
    ]
    feasible = [s for s in scores if s.slo_ok]
    if feasible:
        best = min(feasible, key=lambda s: (
            s.energy_j, s.packing.used_gpus, -s.packing.freq_mhz))
        reason = "min-j-slo-ok"
        slo_ok = True
    else:
        # Paper provisions for peak when nothing cheaper holds SLO.
        full = [s for s in scores
                if s.packing.used_gpus == GPU_COUNT
                and s.packing.freq_mhz == 2520]
        best = min(full or scores, key=lambda s: (s.rho, s.energy_j))
        reason = "fallback-peak-" + (best.reason or "infeasible")
        slo_ok = False
    plan = DynamoPlan(
        n_replica=best.packing.n_replica,
        tp=best.packing.tp,
        freq0=best.packing.freq_mhz,
        unused_gpus=best.packing.unused_gpus,
        used_gpus=best.packing.used_gpus,
        source=str(prof.get("source") or PROFILE_SOURCE_BUILTIN),
        reason=reason,
        energy_j=best.energy_j,
        ttft_s=best.ttft_s,
        tpot_s=best.tpot_s,
        rho=best.rho,
        slo_ok=slo_ok,
        scores=[
            dict(n=s.packing.n_replica, tp=s.packing.tp,
                 freq=s.packing.freq_mhz, used=s.packing.used_gpus,
                 unused=s.packing.unused_gpus, rho=round(s.rho, 4),
                 ttft_s=round(s.ttft_s, 4), tpot_s=round(s.tpot_s, 4),
                 energy_j=round(s.energy_j, 2), slo_ok=s.slo_ok,
                 reason=s.reason)
            for s in scores
        ],
    )
    return plan


def write_plan(plan: DynamoPlan, path: str,
               profile: Optional[dict] = None) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = plan.as_dict()
    if profile is not None:
        payload["profile_source"] = profile.get("source")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DynamoLLM boot planner")
    ap.add_argument("--model-key", default="14b")
    ap.add_argument("--dataset", default="sharegpt")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--slo-ttft", type=float, default=None)
    ap.add_argument("--slo-tpot", type=float, default=None)
    ap.add_argument("--profile", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--profile-out", default="")
    ap.add_argument("--exports", action="store_true",
                    help="print MIXED_TP / MIXED_N_REPLICA / FREQ0 for eval")
    args = ap.parse_args(argv)
    profile = load_or_build_profile(
        args.model_key, profile_path=args.profile or "",
        dataset=args.dataset)
    if args.profile_out:
        write_profile(profile, args.profile_out)
    plan = choose_dynamo_plan(
        rate=args.rate, n=args.n, dataset=args.dataset,
        slo_ttft_s=args.slo_ttft, slo_tpot_s=args.slo_tpot,
        profile=profile, model_key=args.model_key)
    if args.out:
        write_plan(plan, args.out, profile=profile)
        if args.profile_out == "" and args.out:
            write_profile(
                profile,
                os.path.join(os.path.dirname(args.out) or ".",
                             "dynamo_profile.json"))
    if args.exports:
        print(plan.shell_exports(), end="")
    else:
        print(json.dumps(plan.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
