# -*- coding: utf-8 -*-
"""离线 2-候选启动布局打分。报告 only，不接入 choose_partition / run_cell。

候选只有满配 mixed@2520 与满配空间 1P1D。不可信则默认 1P1D。
不读 regime_map / inherit，不搜 n_P≠n_D，不写 energy_summary.csv。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from ecopadg.metrics import slo_non_inferior
from ecopadg.opmodel import (
    default_table_dir, e1b_dir_synthetic, freq_model_trustworthy, load_opmodel,
)
from ecopadg.planner import pin_prefill, spatial_pd_partition
from ecopadg.predictor import ApproxPredictor, WorkloadStats
from ecopadg.profiler import conservative_kv_ttft_s
from ecopadg.types import (
    PREFILL_PIN_MHZ, SLO_NON_INFERIOR_PP, Partition, SloSpec,
    USER_DATASET_SLO, freqs_for_model, normalize_dataset,
)

REPORT_KIND = "offline_boot_layout_score"
CHOSEN_SPATIAL = "spatial-pd"
CHOSEN_MIXED = "mixed"
REASON_UNTRUSTED = "untrusted-default-spatial-pd"
REASON_MIXED_WINS = "trusted-mixed-noninferior-lower-j"
REASON_SPATIAL = "trusted-spatial-pd"


@dataclass
class CandidateScore:
    name: str
    n_mixed: int
    n_prefill: int
    n_decode: int
    attainment: float
    energy_j_per_s: float
    power_w: float
    ttft_s: float
    kv_ttft_s: float
    ttft_ok: bool
    accepted: bool
    reason: str = ""


@dataclass
class LayoutScoreResult:
    report_kind: str = REPORT_KIND
    report_only: bool = True
    online_system_result: bool = False
    model_key: str = ""
    dataset: str = ""
    rate: float = 0.0
    chosen: str = CHOSEN_SPATIAL
    reason: str = REASON_UNTRUSTED
    trusted: bool = False
    synthetic: bool = True
    freq_trusted: bool = False
    capacity_trusted: bool = False
    att_mixed: float = float("nan")
    scores: List[CandidateScore] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["report_only"] = True
        payload["online_system_result"] = False
        return payload


def _mixed_full(gpu_count: int, tp: int) -> Partition:
    n = max(int(gpu_count) // max(int(tp), 1), 1)
    return Partition(n_mixed=n, tp_mixed=int(tp), pp_mixed=1,
                     freq_mixed=2520, gpus_total=int(gpu_count))


def _pd_ttft_s(predictor: ApproxPredictor, prompt_len: float,
               cross_numa: bool) -> tuple:
    pre_s = predictor.opmodel.prefill_time_ms(
        int(max(prompt_len, 1)), PREFILL_PIN_MHZ) / 1000.0
    kv_s, _trust = conservative_kv_ttft_s(prompt_len, cross_numa)
    return float(pre_s), float(kv_s)


def score_boot_layout(
        predictor: ApproxPredictor,
        lam: float,
        *,
        model_key: str = "14b",
        dataset: str = "sharegpt",
        gpu_count: int = 8,
        tp: int = 2,
        synthetic: bool = True,
        freq_trusted: bool = False,
        capacity_trusted: bool = False,
        cross_numa: bool = False,
        delta: float = SLO_NON_INFERIOR_PP) -> LayoutScoreResult:
    """对 mixed@2520 与空间 1P1D 打分。不可信 → 1P1D。"""
    ds = normalize_dataset(dataset)
    mixed_p = _mixed_full(gpu_count, tp)
    spatial_p = pin_prefill(spatial_pd_partition(gpu_count, tp), PREFILL_PIN_MHZ)
    rate = max(float(lam), 0.0)
    pred_m = predictor.predict(mixed_p, rate)
    pred_s = predictor.predict(spatial_p, rate)
    prompt = float(predictor.stats.prompt_len)
    pre_s, kv_s = _pd_ttft_s(predictor, prompt, cross_numa)
    ttft_lim = float(predictor.slo.ttft_s)
    mixed_ttft = predictor.opmodel.prefill_time_ms(
        int(max(prompt, 1)), PREFILL_PIN_MHZ) / 1000.0
    pd_ttft_ok = (pre_s + kv_s) <= ttft_lim
    mixed_ttft_ok = mixed_ttft <= ttft_lim
    att_mixed = float(pred_m.attainment)
    mixed_ok = (
        bool(pred_m.attainable) and mixed_ttft_ok
        and slo_non_inferior(pred_m.attainment, att_mixed, delta))
    spatial_ok = bool(pred_s.attainable) and pd_ttft_ok
    scores = [
        CandidateScore(
            name=CHOSEN_MIXED, n_mixed=mixed_p.n_mixed,
            n_prefill=0, n_decode=0,
            attainment=float(pred_m.attainment),
            energy_j_per_s=float(pred_m.energy_j_per_s),
            power_w=float(pred_m.power_w),
            ttft_s=float(mixed_ttft), kv_ttft_s=0.0,
            ttft_ok=mixed_ttft_ok, accepted=mixed_ok,
            reason="compare-only" if not mixed_ok else ""),
        CandidateScore(
            name=CHOSEN_SPATIAL, n_mixed=0,
            n_prefill=spatial_p.n_prefill, n_decode=spatial_p.n_decode,
            attainment=float(pred_s.attainment),
            energy_j_per_s=float(pred_s.energy_j_per_s),
            power_w=float(pred_s.power_w),
            ttft_s=float(pre_s + kv_s), kv_ttft_s=float(kv_s),
            ttft_ok=pd_ttft_ok, accepted=spatial_ok,
            reason="" if spatial_ok else "ttft-or-capacity"),
    ]
    trusted = (not bool(synthetic)) and bool(freq_trusted) and bool(
        capacity_trusted)
    if not trusted:
        chosen, reason = CHOSEN_SPATIAL, REASON_UNTRUSTED
    elif (mixed_ok
          and slo_non_inferior(pred_m.attainment, pred_s.attainment, delta)
          and float(pred_m.energy_j_per_s) < float(pred_s.energy_j_per_s)):
        chosen, reason = CHOSEN_MIXED, REASON_MIXED_WINS
    else:
        chosen, reason = CHOSEN_SPATIAL, REASON_SPATIAL
    return LayoutScoreResult(
        model_key=str(model_key), dataset=ds, rate=rate,
        chosen=chosen, reason=reason, trusted=trusted,
        synthetic=bool(synthetic), freq_trusted=bool(freq_trusted),
        capacity_trusted=bool(capacity_trusted),
        att_mixed=att_mixed, scores=scores)


def score_from_tables(
        model_key: str,
        stats: WorkloadStats,
        lam: float,
        *,
        dataset: str = "sharegpt",
        gpu_count: int = 8,
        tables: Optional[str] = None,
        capacity_trusted: bool = False,
        cross_numa: bool = False) -> LayoutScoreResult:
    key = str(model_key).lower()
    base = tables or default_table_dir(key)
    om = load_opmodel(base)
    if om is None:
        raise ValueError("缺操作点表: %s" % base)
    ds = normalize_dataset(dataset)
    if ds not in USER_DATASET_SLO:
        raise KeyError("未知数据集 SLO: %s" % dataset)
    ttft, tpot = USER_DATASET_SLO[ds]
    freqs = freqs_for_model(key)
    pred = ApproxPredictor(
        opmodel=om, stats=stats,
        slo=SloSpec(ttft_s=ttft, tpot_s=tpot),
        freq_candidates=freqs)
    tp = 4 if key == "72b" else 2
    return score_boot_layout(
        pred, lam, model_key=key, dataset=ds, gpu_count=gpu_count, tp=tp,
        synthetic=e1b_dir_synthetic(base),
        freq_trusted=freq_model_trustworthy(om, freqs),
        capacity_trusted=bool(capacity_trusted),
        cross_numa=cross_numa)


def render_markdown(result: LayoutScoreResult) -> str:
    lines = [
        "# Offline boot layout score (report only)",
        "",
        "Not an online result. Default `pdblend` still boots spatial PD.",
        "",
        "- model: `%s`" % result.model_key,
        "- dataset: `%s`" % result.dataset,
        "- rate: %s" % result.rate,
        "- trusted: %s (synthetic=%s freq_trusted=%s capacity_trusted=%s)" % (
            result.trusted, result.synthetic, result.freq_trusted,
            result.capacity_trusted),
        "- chosen: **%s** (`%s`)" % (result.chosen, result.reason),
        "",
        "| candidate | att | J/s | TTFT+KV s | ttft_ok |",
        "|---|---:|---:|---:|---|",
    ]
    for sc in result.scores:
        lines.append(
            "| %s | %.3f | %.1f | %.4f | %s |" % (
                sc.name, sc.attainment, sc.energy_j_per_s, sc.ttft_s,
                sc.ttft_ok))
    lines.append("")
    return "\n".join(lines)
