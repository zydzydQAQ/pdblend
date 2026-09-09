# -*- coding: utf-8 -*-
"""profiler:离线工作点采集计划 + 表分析 + DistServe JSON 导出入口。

离线协议复用 P1/E1b(p1b_fine_sweep):freq × batch × prompt_len 扫描,
PowerSampler 20ms 采样、prefix cache 关(make_llm)、settle 1.5s、
首测丢弃、finally 解锁频率;E8 子协议采 ctx 斜率(pl∈{128,2048})。
GPU 相关执行在 M1 由脚本驱动(见 pdblend/script/),本模块提供纯逻辑:
扫描单元生成、表加载、JSON 导出、迁移成本记录。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: F401

from ecopadg.opmodel import OpModel, load_opmodel as _load_opmodel
from ecopadg.types import (
    B_REF, DEFAULT_FREQS, INSTANCE_KINDS, KV_TTFT_S_CROSS_NUMA,
    KV_TTFT_S_SAME_NUMA, PREFILL_PIN_MHZ, SLO_TIER_USER, TOKEN_BUDGET,
    TRUSTED_KV_TOKENS, USER_DATASET_SLO, normalize_dataset,
)

# DynamoLLM 式请求形状桶(对齐用户档数据集,不是 9 档 SS…LL)
SIZE_BUCKETS = {
    "short": (128, 64),
    "medium": (512, 200),
    "long": (2048, 200),
}
# PD 版 Table I:input×output 交叉,不是对角桶。pout 无第三档(用户档无 ≥350)。
PIN_AXIS = (128, 512, 2048)
POUT_AXIS = (64, 200)
PIN_TAG = {128: "S", 512: "M", 2048: "L"}
POUT_TAG = {64: "S", 200: "M"}
# 第二份权重驻留加子(L20 已加载 idle);只进 energy_j_taxed,不进 energy_j。
LOADED_IDLE_W = 76.0
GPUS_PER_TP2 = 2
# Table I TP 维。基表是 TP2;其它度数走 dynamo 公式,source=opmodel+tp_scale。
MIXED_TPS = (1, 2, 4)
PD_TP_PAIRS = ((1, 1), (2, 1), (2, 2), (4, 2))
LAYOUT_MIXED = "mixed"
LAYOUT_SPATIAL_PD = "spatial_pd"
LAYOUTS = (LAYOUT_MIXED, LAYOUT_SPATIAL_PD)

# E5 多卡切换实测缺省(api 24-73ms、settle 0.4-0.7s、瞬态 ≤20J);
# drain/角色切换/实例启停未测则 None/NaN = 禁止该层。
DEFAULT_SWITCH_COST: Dict[str, Optional[float]] = dict(
    freq_api_s=0.073, freq_settle_s=0.7, freq_transient_j=20.0,
    drain_s=None, role_switch_s=None, role_switch_cost_j=None,
    instance_start_s=None, instance_stop_s=None, instance_switch_j=None,
    horizon_l0_s=5.0, horizon_l1_s=300.0, horizon_l2_s=1800.0,
    hysteresis=0.15,
)
LAYER_L0 = "freq"
LAYER_L1 = "tp"
LAYER_L2 = "layout"
LAYERS = (LAYER_L0, LAYER_L1, LAYER_L2)
_DS_SIZE_BUCKET = {
    "alpaca": "short", "sharegpt": "medium", "longbench": "long",
}


def build_sweep_cells(freqs: Sequence[int] = DEFAULT_FREQS,
                      batches: Sequence[int] = (1, 2, 4, 8, 16, 32),
                      prompt_lens: Sequence[int] = (128, 256, 512, 1024, 2048),
                      repeats: int = 2) -> List[dict]:
    """P1/E1b 协议扫描单元(确定性顺序:先 freq 后 batch 后 plen)。"""
    cells = []
    for f in freqs:
        for b in batches:
            for pl in prompt_lens:
                cells.append(dict(freq_mhz=int(f), batch=int(b),
                                  prompt_len=int(pl), repeats=int(repeats)))
    return cells


def build_ctx_cells(freqs: Sequence[int] = (2520, 1500, 1050),
                    batches: Sequence[int] = (1, 8, 32),
                    plens: Sequence[int] = (128, 2048),
                    repeats: int = 2) -> List[dict]:
    """E8 子协议:ctx 斜率采集单元(pl 128 vs 2048 差分)。"""
    cells = []
    for f in freqs:
        for b in batches:
            for pl in plens:
                cells.append(dict(freq_mhz=int(f), batch=int(b),
                                  prompt_len=int(pl), repeats=int(repeats)))
    return cells


def conservative_kv_ttft_s(prompt_len: float,
                           cross_numa: bool = False) -> Tuple[float, str]:
    """B5 可信行外推的 PD KV→TTFT 约束。短于 4096 记 0 / untrusted。"""
    try:
        length = float(prompt_len)
    except (TypeError, ValueError):
        return 0.0, "untrusted"
    if length != length or length < float(TRUSTED_KV_TOKENS):
        return 0.0, "untrusted"
    base = KV_TTFT_S_CROSS_NUMA if cross_numa else KV_TTFT_S_SAME_NUMA
    return float(base) * (length / float(TRUSTED_KV_TOKENS)), "b5-4096"


def kv_ttft_constraint_table(
        prompt_lens: Sequence[int] = (512, 2048, 4096, 8192),
        default_cross_numa: bool = False) -> dict:
    """统一 profile 的 KV 约束列。不进能耗目标。"""
    by_len = {}
    for plen in prompt_lens:
        ttft_s, trust = conservative_kv_ttft_s(plen, default_cross_numa)
        by_len[str(int(plen))] = dict(ttft_s=float(ttft_s), trust=trust)
    return dict(
        trusted_tokens=int(TRUSTED_KV_TOKENS),
        same_numa_s_at_trusted=float(KV_TTFT_S_SAME_NUMA),
        cross_numa_s_at_trusted=float(KV_TTFT_S_CROSS_NUMA),
        default_placement="cross_numa" if default_cross_numa else "same_numa",
        energy_objective=False,
        by_prompt_len=by_len,
    )


def load_opmodel(latency_csv: str, power_csv: str,
                 ctx_latency_csv: str = "") -> OpModel:
    """加载操作点表(复用 E1b 存量或新采集)。"""
    return OpModel.from_tables(latency_csv, power_csv, ctx_latency_csv)


def load_opmodel_dir(base: str):
    """目录入口,转发到 opmodel.load_opmodel。"""
    return _load_opmodel(base)


def export_placement_profile(opmodel: OpModel, model_name: str, tp: int,
                             out_path: str) -> str:
    """把 OpModel 导出为 DistServe time_estimator 同构画像 JSON。"""
    opmodel.export_distserve_json(model_name=model_name, tp=tp,
                                  out_path=out_path)
    return out_path


def _try(fn, *args, **kwargs):
    """能耗/功率接口可选(无 power 表时缺省 None)。"""
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, AttributeError, TypeError):
        return None


def export_unified_profile(opmodel, model_name: str, model_key: str, tp: int,
                           out_path: str,
                           freq_candidates: Sequence[int] = DEFAULT_FREQS,
                           batches: Sequence[int] = (1, 2, 4, 8, 16, 32),
                           prompt_lens: Sequence[int] = (128, 256, 512,
                                                         1024, 2048),
                           ctx_buckets: Sequence[int] = (272, 1024,
                                                         2048, 4096),
                           slo_tpot_s: Optional[float] = None,
                           tpot_margin: float = 0.35,
                           f_star_tolerance: float = 0.05,
                           switch_cost: Optional[dict] = None) -> dict:
    """统一 profile JSON:三池调度器 + DVFS 的唯一共享输入。

    内容:
      - 延迟表:decode_iter_ms[f][b]、prefill_ms[f][plen]
        (plen 是单请求 TTFT;容量路径把批 Σtok 传入 prefill_time_ms);
      - 能耗表:decode_dyn_mj_per_tok[f][b]、prefill_dyn_j_per_tok、residency_w;
      - 决策点(可复用的能耗调频点):
          f_star_mhz     decode 饱和频率(iter 相对最高频增幅 <= tolerance 的最低档);
          energy_opt_decode_freq[b]  逐 batch 每 token 能耗最低频率;
          decode_freq_slack[b][ctx]  slack 选频表(select_decode_freq 语义);
          prefill_freq   prefill 窗频率(高频:能效 + TTFT 双优,E1b/M1-C);
      - switch_cost:E5 缺省 + PE2 实测覆盖(drain/角色切换/实例启停)。
    """
    from ecopadg.online_scheduler import select_decode_freq

    if slo_tpot_s is None:
        slo_tpot_s = float(USER_DATASET_SLO["sharegpt"][1])
    freqs = sorted(int(f) for f in freq_candidates)
    f_max = max(freqs)

    iter_ms: Dict[str, Dict[str, float]] = {}
    dyn_mj: Dict[str, Dict[str, float]] = {}
    prefill_ms: Dict[str, Dict[str, float]] = {}
    for f in freqs:
        iter_ms[str(f)] = {str(b): float(opmodel.iter_time_ms(b, f))
                           for b in batches}
        row = {str(b): _try(opmodel.dyn_j_per_token, b, f) for b in batches}
        if all(v is not None for v in row.values()):
            dyn_mj[str(f)] = row
        prefill_ms[str(f)] = {str(pl): float(opmodel.prefill_time_ms(pl, f))
                              for pl in prompt_lens}

    # f*:decode 饱和频率(以 b=8 为参考;无平台时回落最高频)
    b_ref = 8 if 8 in set(int(b) for b in batches) else max(batches)
    base_iter = float(opmodel.iter_time_ms(b_ref, f_max))
    f_star = f_max
    for f in freqs:
        if float(opmodel.iter_time_ms(b_ref, f)) \
                <= base_iter * (1.0 + f_star_tolerance):
            f_star = f
            break

    energy_opt: Dict[str, int] = {}
    if dyn_mj:
        for b in batches:
            costs = [(f, dyn_mj[str(f)][str(b)]) for f in freqs
                     if str(f) in dyn_mj]
            if costs:
                energy_opt[str(b)] = int(min(costs, key=lambda x: x[1])[0])

    def _slack_table(tpot_s: float) -> Dict[str, Dict[str, int]]:
        table: Dict[str, Dict[str, int]] = {}
        for b in batches:
            row2: Dict[str, int] = {}
            for ctx in ctx_buckets:
                row2[str(ctx)] = int(select_decode_freq(
                    opmodel, int(b), float(ctx), float(tpot_s), freqs,
                    margin=tpot_margin))
            table[str(b)] = row2
        return table

    slack_by_ds = {
        name: _slack_table(tpot)
        for name, (_ttft, tpot) in USER_DATASET_SLO.items()
    }
    slack_table = _slack_table(slo_tpot_s)

    sc = dict(DEFAULT_SWITCH_COST)
    if switch_cost:
        sc.update(switch_cost)

    prof = dict(
        model=model_name, model_key=model_key, tp=int(tp),
        freq_candidates=freqs,
        slo_tpot_s=float(slo_tpot_s), tpot_margin=float(tpot_margin),
        slo_tier=SLO_TIER_USER,
        decode_iter_ms=iter_ms,
        decode_dyn_mj_per_tok=dyn_mj or None,
        prefill_ms=prefill_ms,
        prefill_dyn_j_per_tok=_try(getattr(opmodel, "prefill_dyn_j_per_token",
                                           None) or (lambda *_: None), f_max),
        residency_w=_try(getattr(opmodel, "residency_w",
                                 None) or (lambda *_: None)),
        decision=dict(
            f_star_mhz=int(f_star),
            energy_opt_decode_freq=energy_opt,
            decode_freq_slack=slack_table,
            decode_freq_slack_by_dataset=slack_by_ds,
            prefill_freq=int(f_max),
        ),
        constraints=dict(kv_ttft=kv_ttft_constraint_table()),
        switch_cost=sc,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1, ensure_ascii=False)
    return prof


def estimate_max_num_tokens(hbm_gb: float, model_gb: float,
                            kv_bytes_per_token: float,
                            util: float = 0.85) -> int:
    """DistServe 式 batching 上界:可用显存 / 每 token KV。"""
    free = max(float(hbm_gb) * float(util) - float(model_gb), 0.0) * (1024 ** 3)
    if kv_bytes_per_token <= 0:
        return 0
    return int(free / float(kv_bytes_per_token))


def build_instance_sweep_cells(kind: str,
                               freqs: Sequence[int] = DEFAULT_FREQS,
                               batches: Sequence[int] = (1, 2, 4, 8, 16, 32),
                               prompt_lens: Sequence[int] = (128, 512, 2048),
                               ctxs: Sequence[int] = (256, 1024, 2048),
                               token_budgets: Sequence[int] = (TOKEN_BUDGET,),
                               repeats: int = 2) -> List[dict]:
    """三类 instance 的扫描单元。"""
    if kind not in INSTANCE_KINDS:
        raise ValueError("未知 instance kind: %s" % kind)
    cells = []
    for f in freqs:
        for b in batches:
            for pl in prompt_lens:
                for ctx in ctxs:
                    for tb in token_budgets:
                        cells.append(dict(
                            kind=kind, freq_mhz=int(f), batch=int(b),
                            prompt_len=int(pl), ctx=int(ctx),
                            token_budget=int(tb), repeats=int(repeats)))
    return cells


def export_instance_profile(opmodel, kind: str, model_name: str,
                            tp: int, pp: int, out_path: str,
                            freq_candidates: Sequence[int] = DEFAULT_FREQS,
                            batches: Sequence[int] = (1, 2, 4, 8, 16, 32),
                            prompt_lens: Sequence[int] = (128, 512, 2048),
                            max_num_tokens: Optional[int] = None) -> dict:
    """单类 instance 画像(延迟 + 能耗 + batching 上界)。"""
    unified = export_unified_profile(
        opmodel, model_name=model_name, model_key=kind, tp=tp,
        out_path=out_path, freq_candidates=freq_candidates,
        batches=batches, prompt_lens=prompt_lens)
    unified["kind"] = kind
    unified["pp"] = int(pp)
    unified["max_num_tokens"] = (int(max_num_tokens)
                                 if max_num_tokens is not None else None)
    unified["batching"] = dict(
        prefill_max_tokens=int(max_num_tokens or TOKEN_BUDGET),
        decode_max_batch=max(batches),
        max_num_batched_tokens=int(max_num_tokens or TOKEN_BUDGET),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=1, ensure_ascii=False)
    return unified


def export_three_way_profile(opmodel, model_name: str, model_key: str,
                             tp: int, out_dir: str,
                             pp: int = 1,
                             max_num_tokens: Optional[int] = None) -> dict:
    """mixed / prefill / decode 三份统一画像。"""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for kind in INSTANCE_KINDS:
        path = os.path.join(out_dir, "profile_%s_%s_tp%d_pp%d.json"
                            % (model_key, kind, tp, pp))
        out[kind] = export_instance_profile(
            opmodel, kind, model_name, tp, pp, path,
            max_num_tokens=max_num_tokens)
    index = os.path.join(out_dir, "profile_%s_three_way.json" % model_key)
    with open(index, "w", encoding="utf-8") as f:
        json.dump({k: v.get("model_key") for k, v in out.items()},
                  f, indent=1)
    return out


def load_unified_profile(path: str) -> dict:
    """读取统一 profile JSON。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify_size_bucket(prompt_len: float, output_len: float) -> str:
    """最近邻桶:short/medium/long。"""
    try:
        pin, pout = float(prompt_len), float(output_len)
    except (TypeError, ValueError):
        return "medium"
    best, best_d = "medium", None
    for name, (tin, tout) in SIZE_BUCKETS.items():
        d = (pin - tin) ** 2 + (pout - tout) ** 2
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


def _row_slo_ok(row: dict, ttft_s: float, tpot_s: float) -> bool:
    try:
        ttft = float(row.get("ttft_s") if row.get("ttft_s") not in (None, "")
                     else float(row.get("ttft_ms") or 0) / 1000.0)
        tpot = float(row.get("tpot_s") if row.get("tpot_s") not in (None, "")
                     else float(row.get("tpot_ms") or 0) / 1000.0)
    except (TypeError, ValueError):
        return False
    return ttft < float(ttft_s) and tpot < float(tpot_s)


def _row_energy(row: dict) -> float:
    for k in ("energy_j", "total_j", "j"):
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float("inf")


def _row_batch(row: dict) -> int:
    try:
        return int(float(row.get("batch") or 1))
    except (TypeError, ValueError):
        return 1


def _row_ttft_s(row: dict) -> float:
    try:
        if row.get("ttft_s") not in (None, ""):
            return float(row["ttft_s"])
        return float(row.get("ttft_ms") or 0) / 1000.0
    except (TypeError, ValueError):
        return float("inf")


def _row_tpot_s(row: dict) -> float:
    try:
        if row.get("tpot_s") not in (None, ""):
            return float(row["tpot_s"])
        return float(row.get("tpot_ms") or 0) / 1000.0
    except (TypeError, ValueError):
        return float("inf")


def _config_key(row: dict) -> Tuple[str, int, int, int]:
    layout = str(row.get("layout") or LAYOUT_MIXED)
    try:
        tp = int(float(row.get("tp") or row.get("tp_p") or 2))
    except (TypeError, ValueError):
        tp = 2
    try:
        freq_p = int(float(row.get("freq_p") or row.get("freq_mhz") or 2520))
    except (TypeError, ValueError):
        freq_p = 2520
    try:
        freq_d = int(float(row.get("freq_d") or row.get("freq_mhz") or freq_p))
    except (TypeError, ValueError):
        freq_d = freq_p
    return (layout, tp, freq_p, freq_d)


def _average_rows(rows: Sequence[dict]) -> dict:
    """同一操作点的 repeats 取均值后再比 J。"""
    head = dict(rows[0])
    n = float(len(rows))
    head["energy_j"] = sum(_row_energy(r) for r in rows) / n
    head["ttft_s"] = sum(_row_ttft_s(r) for r in rows) / n
    head["tpot_s"] = sum(_row_tpot_s(r) for r in rows) / n
    head["ttft_ms"] = head["ttft_s"] * 1000.0
    head["tpot_ms"] = head["tpot_s"] * 1000.0
    head["repeat"] = "mean"
    return head


def lookup_config(rows: Sequence[dict], size_bucket: str, model: str,
                  slo_name: str = "sharegpt",
                  ttft_s: Optional[float] = None,
                  tpot_s: Optional[float] = None,
                  batch: Optional[int] = 1) -> Optional[dict]:
    """在可行集里取 min J 的 {layout, tp, freq_p, freq_d}。

    SLO 是过滤不是扫描轴。默认只看 batch=1(K=1 主协议)。
    无可行格返回 None。
    """
    ds = normalize_dataset(slo_name)
    if ttft_s is None or tpot_s is None:
        if ds not in USER_DATASET_SLO:
            raise KeyError("未知 SLO: %s" % slo_name)
        def_ttft, def_tpot = USER_DATASET_SLO[ds]
        if ttft_s is None:
            ttft_s = def_ttft
        if tpot_s is None:
            tpot_s = def_tpot
    feasible = feasible_configs(
        rows, size_bucket, model, slo_name,
        ttft_s=ttft_s, tpot_s=tpot_s, batch=batch)
    if not feasible:
        return None
    best = min(feasible, key=_row_energy)
    layout, tp, freq_p, freq_d = _config_key(best)
    return dict(
        size_bucket=str(size_bucket), model=str(model).lower(), slo=ds,
        slo_ttft_s=float(ttft_s), slo_tpot_s=float(tpot_s),
        layout=layout, tp=tp,
        freq_p=freq_p, freq_d=freq_d,
        energy_j=_row_energy(best),
        ttft_s=_row_ttft_s(best),
        tpot_s=_row_tpot_s(best),
        batch=int(batch) if batch is not None else _row_batch(best),
    )


def build_lookup_table(rows: Sequence[dict], model: str = "14b",
                       slo_names: Optional[Sequence[str]] = None) -> dict:
    """(size_bucket, model, slo) → 配置。无可行格的键不写。"""
    names = list(slo_names) if slo_names is not None else list(USER_DATASET_SLO)
    table: Dict[str, dict] = {}
    for bucket in SIZE_BUCKETS:
        for slo in names:
            cfg = lookup_config(rows, bucket, model, slo)
            if cfg is None:
                continue
            cfg8 = lookup_config(rows, bucket, model, slo, batch=8)
            cfg["freq_d_b1"] = int(cfg["freq_d"])
            cfg["freq_d_b8"] = int(cfg8["freq_d"]) if cfg8 else int(cfg["freq_d"])
            table["%s|%s|%s" % (bucket, str(model).lower(),
                                normalize_dataset(slo))] = cfg
    return dict(model=str(model).lower(), keys=table)


def attach_lookup(profile: dict, rows: Sequence[dict]) -> dict:
    """把 lookup 字典挂到统一 profile 上。"""
    model = str(profile.get("model_key") or profile.get("model") or "14b")
    profile["lookup"] = build_lookup_table(rows, model=model)
    profile["switch_cost"] = load_switch_cost()
    return profile


def _finite(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return x == x and x >= 0.0


def load_switch_cost(path: str = "", ws: Optional[str] = None) -> dict:
    """E5 缺省 + 实测 JSON 覆盖。未出现的启停/改布局键保持 None。

    不读 32B pe2_switch_cost,也不把任意 summary.json 当 14B 代价。
    """
    out = dict(DEFAULT_SWITCH_COST)
    root = ws or os.environ.get("PDBLEND_WS", "/root/workspace")
    env = os.environ.get("PDBLEND_SWITCH_COST")
    candidates = []
    if path:
        candidates.append(path)
    if env:
        candidates.append(env)
    nm = os.path.join(root, "pdblend", "new-motivations", "results")
    if os.path.isdir(nm):
        dated = sorted(
            (p for p in os.listdir(nm) if p.endswith("-switch-cost")),
            reverse=True)
        for stamp in dated:
            candidates.append(os.path.join(nm, stamp, "switch_cost.json"))
    for cand in candidates:
        if not cand or not os.path.isfile(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        payload = data.get("switch_cost")
        if not isinstance(payload, dict):
            payload = data
        if not any(k in payload for k in (
                "freq_transient_j", "freq_settle_s", "role_switch_cost_j",
                "instance_switch_j", "instance_start_s")):
            continue
        for key in DEFAULT_SWITCH_COST:
            if key in payload:
                out[key] = payload[key]
        break
    return out


def default_lookup_path(model: str = "14b", ws: Optional[str] = None) -> str:
    root = ws or os.environ.get("PDBLEND_WS", "/root/workspace")
    env = os.environ.get("PDBLEND_LOOKUP")
    if env:
        return env
    nm = os.path.join(root, "pdblend", "new-motivations", "results")
    stable = os.path.join(nm, "lookup-opmodel-14b", "lookup.json")
    if os.path.isfile(stable):
        return stable
    if os.path.isdir(nm):
        dated = sorted(
            (p for p in os.listdir(nm) if p.endswith("-profile-14b")),
            reverse=True)
        for stamp in dated:
            path = os.path.join(nm, stamp, "lookup.json")
            if os.path.isfile(path):
                return path
    return os.path.join(nm, "lookup.json")


def load_lookup_table(path: str = "", model: str = "14b",
                      ws: Optional[str] = None) -> dict:
    """只读显式路径。空路径不自动发现 lookup.json(禁止 900-prefer 暗线)。"""
    del ws
    cand = str(path or "").strip()
    if cand and os.path.isfile(cand):
        with open(cand, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("keys", {})
            data.setdefault("model", str(model).lower())
            return data
    return dict(model=str(model).lower(), keys={})


def dataset_size_bucket(dataset: str) -> str:
    return _DS_SIZE_BUCKET.get(normalize_dataset(dataset), "medium")


def lookup_freq_d(table: dict, size_bucket: str, model: str,
                  slo_name: str) -> Optional[int]:
    keys = (table or {}).get("keys") or {}
    key = "%s|%s|%s" % (size_bucket, str(model).lower(),
                        normalize_dataset(slo_name))
    cfg = keys.get(key)
    if not cfg:
        return None
    try:
        return int(cfg.get("freq_d") or cfg.get("freq_mhz") or 0) or None
    except (TypeError, ValueError):
        return None


def interp_lookup_freq(table: dict, size_bucket: str, model: str,
                       slo_name: str, batch: float = 1.0) -> Optional[int]:
    """batch∈[1,8] 在 freq_d_b1 与 freq_d_b8 之间插值,再落到两档之一。"""
    keys = (table or {}).get("keys") or {}
    key = "%s|%s|%s" % (size_bucket, str(model).lower(),
                        normalize_dataset(slo_name))
    cfg = keys.get(key)
    if not cfg:
        return None
    try:
        f1 = int(cfg.get("freq_d_b1") or cfg.get("freq_d") or 0)
        f8 = int(cfg.get("freq_d_b8") or f1 or 0)
    except (TypeError, ValueError):
        return None
    if f1 <= 0:
        return None
    if f8 <= 0:
        f8 = f1
    try:
        b = float(batch)
    except (TypeError, ValueError):
        b = 1.0
    if b != b:
        b = 1.0
    if b <= 1.0:
        raw = float(f1)
    elif b >= 8.0:
        raw = float(f8)
    else:
        raw = float(f1) + (b - 1.0) / 7.0 * (float(f8) - float(f1))
    return int(min((f1, f8), key=lambda f: (abs(float(f) - raw), -int(f))))


def _try_model_energy(opmodel, kind: str, *args) -> Optional[float]:
    try:
        if kind == "prefill":
            return float(opmodel.prefill_dyn_j_per_token(*args))
        if kind == "decode":
            return float(opmodel.dyn_j_per_token(*args)) / 1000.0
    except (ValueError, AttributeError, TypeError):
        return None
    return None


def synthesize_lookup_rows(opmodel, model: str = "14b",
                           tps: Sequence[int] = (1, 2),
                           freqs: Sequence[int] = (2520, 1500, 900),
                           batches: Sequence[int] = (1, 8)) -> List[dict]:
    """用 14B OpModel 合成 (layout, tp, freq, size, batch) 行。source=opmodel。

    tp=1 与 tp=2 共用 TP2 表,只作查表轴,启动仍映射到 4×TP2 / 2×1P1D。
    """
    rows: List[dict] = []
    model_key = str(model).lower()
    for layout in LAYOUTS:
        for tp in tps:
            for name, (pin, pout) in SIZE_BUCKETS.items():
                ctx = float(pin) + float(pout) / 2.0
                for batch in batches:
                    pairs = ((int(f), int(f)) for f in freqs)
                    if layout == LAYOUT_SPATIAL_PD:
                        pairs = ((2520, int(fd)) for fd in freqs)
                    for freq_p, freq_d in pairs:
                        try:
                            ttft_s = float(opmodel.prefill_time_ms(
                                int(pin), int(freq_p))) / 1000.0
                            tpot_s = float(opmodel.iter_time_ms(
                                int(batch), int(freq_d), ctx)) / 1000.0
                        except (ValueError, AttributeError, TypeError):
                            continue
                        e_pre = _try_model_energy(opmodel, "prefill", int(freq_p))
                        e_dec = _try_model_energy(
                            opmodel, "decode", int(batch), int(freq_d))
                        energy = float("inf")
                        if e_pre is not None and e_dec is not None:
                            energy = float(pin) * e_pre + float(pout) * e_dec
                        rows.append(dict(
                            model=model_key, layout=layout, tp=int(tp),
                            freq_p=int(freq_p), freq_d=int(freq_d),
                            freq_mhz=int(freq_d), size_bucket=name,
                            batch=int(batch), prompt_len=int(pin),
                            gen_len=int(pout), output_len=int(pout),
                            ttft_s=ttft_s, tpot_s=tpot_s,
                            ttft_ms=ttft_s * 1000.0, tpot_ms=tpot_s * 1000.0,
                            energy_j=energy, source="opmodel",
                        ))
    return rows


def shape_name(pin: int, pout: int) -> str:
    """SS/SM/MS/MM/LS/LM。缺档回退 S/M/L 最近邻。"""
    def _tag(value: int, cuts: Sequence[int], tags: dict) -> str:
        if int(value) in tags:
            return tags[int(value)]
        nearest = min(cuts, key=lambda c: abs(int(c) - int(value)))
        return tags[int(nearest)]
    return _tag(pin, PIN_AXIS, PIN_TAG) + _tag(pout, POUT_AXIS, POUT_TAG)


def prefill_s(opmodel, pin: float, freq: int, tp: int = 2) -> float:
    """单请求 prefill 墙钟(秒)。基表 TP2,其它度数乘时间缩放。"""
    scale_p, _ = _tp_time_scales(int(tp), int(tp))
    return (float(opmodel.prefill_time_ms(int(pin), int(freq))) / 1000.0
            * scale_p)


def decode_s(opmodel, pin: float, pout: float, freq: int,
             tp: int = 2, batch: int = 1) -> float:
    """单请求 decode 墙钟(秒):pout × iter(b, f, ctx)。"""
    ctx = float(pin) + float(pout) / 2.0
    _, scale_d = _tp_time_scales(2, int(tp))
    it_s = (float(opmodel.iter_time_ms(int(batch), int(freq), ctx)) / 1000.0
            * scale_d)
    return float(pout) * it_s


def mu_prefill(opmodel, pin: float, n: int = 1, freq: int = 2520,
               tp: int = 2) -> float:
    """护栏 μ_P:n / T_pre(pin,f,TP)。单请求,不组批。"""
    t = prefill_s(opmodel, pin, freq, tp)
    return float(n) / max(float(t), 1e-9)


def mu_decode(opmodel, pin: float, pout: float, n: int = 1,
              freq: int = 2520, tp: int = 2, batch: int = 1) -> float:
    """护栏 μ_D:n / T_dec(pout,b=1)。缺表由调用方 fail-closed。"""
    t = decode_s(opmodel, pin, pout, freq, tp, batch)
    return float(n) / max(float(t), 1e-9)


def node_energy_j(used: int, freq_mhz: int, rate: float, n: int,
                  profile: dict, gpu_count: int = 8,
                  layout: str = LAYOUT_MIXED) -> float:
    """节点 J:used×gpu_w + unused×空闲。PD/A9 的 used 已含第二份卡。

    gpu_w 含 loaded idle。闲卡只付 idle_empty,不付 76 W。
    layout 只作记账标签,不另加一份驻留(避免和 used 双计)。
    """
    from ecopadg.dynamo_planner import (
        GPU_COUNT, IDLE_EMPTY_W, _freq_row, gpu_power_w,
    )
    del layout
    row = _freq_row(profile, int(freq_mhz))
    gpu_w = float(row.get("gpu_w") or gpu_power_w(int(freq_mhz)))
    empty = float(profile.get("idle_empty_w") or IDLE_EMPTY_W)
    used_n = max(int(used), 0)
    limit = int(gpu_count) if gpu_count else int(GPU_COUNT)
    unused = max(limit - used_n, 0)
    cell_s = float(n) / max(float(rate), 1e-6)
    return (used_n * gpu_w + unused * empty) * cell_s


def prefill_s(opmodel, pin: float, freq: int, tp: int = 2) -> float:
    """单请求 prefill 墙钟。门与 ScaleInst 共用,按 mean pin,不走组批。

    E1b 基表是 TP2;仅 tp≠2 才套时间缩放。
    """
    raw = float(opmodel.prefill_time_ms(int(pin), int(freq))) / 1000.0
    if int(tp) == 2:
        return raw
    scale_p, _ = _tp_time_scales(int(tp), 2)
    return raw * scale_p


def decode_s(opmodel, pin: float, pout: float, freq: int,
             tp: int = 2, batch: int = 1) -> float:
    """单请求 decode 墙钟:pout × iter(b, f, ctx)。b=1 给护栏。"""
    ctx = float(pin) + float(pout) / 2.0
    it_s = float(opmodel.iter_time_ms(int(batch), int(freq), ctx)) / 1000.0
    if int(tp) != 2:
        _, scale_d = _tp_time_scales(2, int(tp))
        it_s *= scale_d
    if it_s <= 0.0:
        raise ValueError("non-positive iter time")
    return float(pout) * it_s


def mu_prefill(opmodel, pin: float, n: int = 1, freq: int = 2520,
               tp: int = 2) -> float:
    """n 个 P 实例的 req/s。"""
    return float(n) / max(prefill_s(opmodel, pin, freq, tp), 1e-9)


def mu_decode(opmodel, pin: float, pout: float, n: int = 1,
              freq: int = 2520, tp: int = 2, batch: int = 1) -> float:
    """n 个 D 实例的 req/s。"""
    return float(n) / max(
        decode_s(opmodel, pin, pout, freq, tp, batch), 1e-9)


def node_energy_j(used: int, freq_mhz: int, rate: float, n: int,
                  profile: dict, gpu_count: int = 8,
                  layout: str = LAYOUT_MIXED) -> float:
    """节点焦耳:used×gpu_w + unused×空闲 + PD/A9 第二份 loaded idle。

    mixed:第二份为 0。spatial_pd/hybrid:再加 used×loaded_idle×墙钟的一半
    (两摊各一半卡在对侧空闲时仍加载)。不进 lookup。
    """
    from ecopadg.dynamo_planner import (
        GPU_COUNT, IDLE_EMPTY_W, _freq_row, gpu_power_w,
    )
    row = _freq_row(profile, int(freq_mhz))
    gpu_w = float(row.get("gpu_w") or gpu_power_w(int(freq_mhz)))
    empty = float(profile.get("idle_empty_w") or IDLE_EMPTY_W)
    loaded = float(profile.get("loaded_idle_w") or LOADED_IDLE_W)
    used_n = max(int(used), 0)
    limit = int(gpu_count) if gpu_count else int(GPU_COUNT)
    unused = max(limit - used_n, 0)
    cell_s = float(n) / max(float(rate), 1e-6)
    busy_w = used_n * gpu_w
    idle_w = unused * empty
    kind = str(layout or LAYOUT_MIXED)
    tax_w = 0.0
    if kind in (LAYOUT_SPATIAL_PD, "hybrid-a9"):
        tax_w = 0.5 * used_n * loaded
    return (busy_w + idle_w + tax_w) * cell_s


def _mu_prefill(opmodel, pin: float, freq_p: int, n: int = 1) -> float:
    from ecopadg.predictor import prefill_token_pack
    pack, n_req = prefill_token_pack(pin, TOKEN_BUDGET)
    pre_s = float(opmodel.prefill_time_ms(int(pack), int(freq_p))) / 1000.0
    return float(n) * float(n_req) / max(pre_s, 1e-9)


def _mu_decode(opmodel, pin: float, pout: float, freq_d: int,
               batch: int = B_REF, n: int = 1) -> float:
    ctx = float(pin) + float(pout) / 2.0
    it_s = float(opmodel.iter_time_ms(
        int(batch), int(freq_d), ctx)) / 1000.0
    per_req_s = float(pout) * it_s
    return float(n) * float(batch) / max(per_req_s, 1e-9)


def _tp_time_scales(tp_p: int, tp_d: int, base_tp: int = 2) -> Tuple[float, float]:
    from ecopadg.dynamo_planner import tp_decode_scale, tp_prefill_scale
    return (float(tp_prefill_scale(int(tp_p), base_tp)),
            float(tp_decode_scale(int(tp_d), base_tp)))


def synthesize_pd_table_i_rows(
        opmodel, model: str = "14b",
        pins: Sequence[int] = PIN_AXIS,
        pouts: Sequence[int] = POUT_AXIS,
        freqs: Sequence[int] = (2520, 1500, 900),
        batches: Sequence[int] = (1, 8),
        loaded_idle_w: float = LOADED_IDLE_W,
        mixed_tps: Sequence[int] = (2,),
        pd_tp_pairs: Sequence[Tuple[int, int]] = ((2, 2),)) -> List[dict]:
    """PD 版 Table I:pin×pout 交叉查询。默认只 TP2。source=opmodel。

    energy_j = 相表动态焦耳 × (卡数/2) × 时间缩放。
    energy_j_taxed 再加 used_gpus × loaded_idle × 墙钟。
    KV 只进 PD ttft。不上 lookup、不进 boot-slo-layout。
    """
    rows: List[dict] = []
    model_key = str(model).lower()
    f_pin = int(PREFILL_PIN_MHZ)
    for pin in pins:
        for pout in pouts:
            name = shape_name(int(pin), int(pout))
            ctx = float(pin) + float(pout) / 2.0
            kv_s, kv_trust = conservative_kv_ttft_s(pin, False)
            for batch in batches:
                jobs = []
                for tp in mixed_tps:
                    jobs.append((LAYOUT_MIXED, int(tp), int(tp)))
                for tp_p, tp_d in pd_tp_pairs:
                    jobs.append((LAYOUT_SPATIAL_PD, int(tp_p), int(tp_d)))
                for layout, tp_p, tp_d in jobs:
                    scale_p, scale_d = _tp_time_scales(tp_p, tp_d)
                    used = (int(tp_p) if layout == LAYOUT_MIXED
                            else int(tp_p) + int(tp_d))
                    src = ("opmodel" if tp_p == 2 and tp_d == 2
                           else "opmodel+tp_scale")
                    pairs = ((int(f), int(f)) for f in freqs)
                    if layout == LAYOUT_SPATIAL_PD:
                        pairs = ((f_pin, int(fd)) for fd in freqs)
                    for freq_p, freq_d in pairs:
                        try:
                            t_pre = float(opmodel.prefill_time_ms(
                                int(pin), int(freq_p))) / 1000.0 * scale_p
                            t_iter = float(opmodel.iter_time_ms(
                                int(batch), int(freq_d), ctx)) / 1000.0 * scale_d
                        except (ValueError, AttributeError, TypeError):
                            continue
                        ttft_s = t_pre
                        if layout == LAYOUT_SPATIAL_PD:
                            ttft_s = t_pre + float(kv_s)
                        e_pre = _try_model_energy(
                            opmodel, "prefill", int(freq_p))
                        e_dec = _try_model_energy(
                            opmodel, "decode", int(batch), int(freq_d))
                        e_pre_j = float("nan")
                        e_dec_j = float("nan")
                        if e_pre is not None:
                            e_pre_j = (float(pin) * e_pre
                                       * (float(tp_p) / 2.0) * scale_p)
                        if e_dec is not None:
                            e_dec_j = (float(pout) * e_dec
                                       * (float(tp_d) / 2.0) * scale_d)
                        if e_pre is not None and e_dec is not None:
                            energy = e_pre_j + e_dec_j
                        else:
                            energy = float("inf")
                        t_e2e = t_pre + float(pout) * t_iter
                        tax_j = float(used) * float(loaded_idle_w) * t_e2e
                        taxed = (energy + tax_j
                                 if energy == energy and energy < float("inf")
                                 else float("inf"))
                        try:
                            mu_p = _mu_prefill(opmodel, pin, freq_p) / scale_p
                            mu_d = _mu_decode(
                                opmodel, pin, pout, freq_d,
                                batch=int(batch)) / scale_d
                        except (ValueError, AttributeError, TypeError):
                            mu_p, mu_d = float("nan"), float("nan")
                        if mu_p == mu_p and mu_d == mu_d:
                            bottleneck = ("prefill" if mu_p < mu_d
                                          else "decode")
                            mu = min(mu_p, mu_d)
                        else:
                            bottleneck, mu = "unknown", float("nan")
                        e_sum = e_pre_j + e_dec_j
                        e_pre_frac = (e_pre_j / e_sum
                                      if e_sum == e_sum and e_sum > 0
                                      else float("nan"))
                        rows.append(dict(
                            model=model_key, layout=layout, tp=int(tp_p),
                            tp_p=int(tp_p), tp_d=int(tp_d),
                            freq_p=int(freq_p), freq_d=int(freq_d),
                            freq_mhz=int(freq_d),
                            shape=name, size_bucket=name,
                            batch=int(batch), prompt_len=int(pin),
                            gen_len=int(pout), output_len=int(pout),
                            ctx=ctx, ttft_s=ttft_s, tpot_s=t_iter,
                            ttft_ms=ttft_s * 1000.0,
                            tpot_ms=t_iter * 1000.0,
                            e_pre_j=e_pre_j, e_dec_j=e_dec_j,
                            e_pre_frac=e_pre_frac,
                            energy_j=energy, energy_j_taxed=taxed,
                            residency_tax_j=tax_j,
                            used_gpus=int(used),
                            kv_ttft_s=float(kv_s) if layout == LAYOUT_SPATIAL_PD
                            else 0.0,
                            kv_trust=kv_trust if layout == LAYOUT_SPATIAL_PD
                            else "n/a",
                            mu_p=mu_p, mu_d=mu_d, mu=mu,
                            bottleneck=bottleneck,
                            source=src,
                        ))
    return rows


def summarize_pd_table_i(rows: Sequence[dict],
                         slo_name: str = "sharegpt",
                         batch: int = 8) -> List[dict]:
    """每个形状:mixed@2520 对照 vs 可行 PD 里 min energy_j。"""
    ds = normalize_dataset(slo_name)
    if ds not in USER_DATASET_SLO:
        raise KeyError("未知 SLO: %s" % slo_name)
    ttft_s, tpot_s = USER_DATASET_SLO[ds]
    by_shape: Dict[str, List[dict]] = {}
    for row in rows:
        if int(_row_batch(row)) != int(batch):
            continue
        by_shape.setdefault(str(row.get("shape") or ""), []).append(dict(row))
    out = []
    for shape in sorted(by_shape):
        group = by_shape[shape]
        mixed = [r for r in group
                 if r.get("layout") == LAYOUT_MIXED
                 and int(r.get("freq_p") or 0) == 2520
                 and int(r.get("freq_d") or 0) == 2520]
        pd_ok = [r for r in group
                 if r.get("layout") == LAYOUT_SPATIAL_PD
                 and _row_slo_ok(r, ttft_s, tpot_s)]
        base = mixed[0] if mixed else None
        best = min(pd_ok, key=_row_energy) if pd_ok else None
        item = dict(
            shape=shape, slo=ds, batch=int(batch),
            pin=int((base or best or {}).get("prompt_len") or 0),
            pout=int((base or best or {}).get("output_len") or 0),
            mixed_j=((base or {}).get("energy_j")),
            mixed_e_pre_frac=((base or {}).get("e_pre_frac")),
            pd_j=((best or {}).get("energy_j")),
            pd_j_taxed=((best or {}).get("energy_j_taxed")),
            pd_freq_d=((best or {}).get("freq_d")),
            pd_e_pre_frac=((best or {}).get("e_pre_frac")),
            pd_bottleneck=((best or {}).get("bottleneck")),
            pd_mu_p=((best or {}).get("mu_p")),
            pd_mu_d=((best or {}).get("mu_d")),
            kv_trust=((best or {}).get("kv_trust")),
        )
        if base and best and _finite(base.get("energy_j")) \
                and _finite(best.get("energy_j")) \
                and float(base["energy_j"]) > 0:
            item["dJ_pct"] = 100.0 * (
                1.0 - float(best["energy_j"]) / float(base["energy_j"]))
            if _finite(best.get("energy_j_taxed")):
                item["dJ_taxed_pct"] = 100.0 * (
                    1.0 - float(best["energy_j_taxed"])
                    / float(base["energy_j"]))
        else:
            item["dJ_pct"] = None
            item["dJ_taxed_pct"] = None
        out.append(item)
    return out


def _best_slo_row(rows: Sequence[dict], ttft_s: float,
                  tpot_s: float) -> Optional[dict]:
    ok = [r for r in rows if _row_slo_ok(r, ttft_s, tpot_s)]
    if not ok:
        return None
    return min(ok, key=_row_energy)


def summarize_pd_table_i_tp(rows: Sequence[dict],
                            slo_name: str = "sharegpt",
                            batch: int = 8) -> List[dict]:
    """每个形状:可行 mixed min J vs 可行 PD min J(含 TP 维)。"""
    ds = normalize_dataset(slo_name)
    if ds not in USER_DATASET_SLO:
        raise KeyError("未知 SLO: %s" % slo_name)
    ttft_s, tpot_s = USER_DATASET_SLO[ds]
    by_shape: Dict[str, List[dict]] = {}
    for row in rows:
        if int(_row_batch(row)) != int(batch):
            continue
        by_shape.setdefault(str(row.get("shape") or ""), []).append(dict(row))
    out = []
    for shape in sorted(by_shape):
        group = by_shape[shape]
        mixed = _best_slo_row(
            [r for r in group if r.get("layout") == LAYOUT_MIXED],
            ttft_s, tpot_s)
        pd = _best_slo_row(
            [r for r in group if r.get("layout") == LAYOUT_SPATIAL_PD],
            ttft_s, tpot_s)
        same = {}
        for tp in MIXED_TPS:
            m_tp = _best_slo_row(
                [r for r in group if r.get("layout") == LAYOUT_MIXED
                 and int(r.get("tp_p") or 0) == int(tp)
                 and int(r.get("tp_d") or 0) == int(tp)],
                ttft_s, tpot_s)
            p_tp = _best_slo_row(
                [r for r in group if r.get("layout") == LAYOUT_SPATIAL_PD
                 and int(r.get("tp_p") or 0) == int(tp)
                 and int(r.get("tp_d") or 0) == int(tp)],
                ttft_s, tpot_s)
            if m_tp is None and p_tp is None:
                continue
            same[str(int(tp))] = dict(
                mixed_freq=((m_tp or {}).get("freq_d")),
                mixed_j=((m_tp or {}).get("energy_j")),
                pd_freq_p=((p_tp or {}).get("freq_p")),
                pd_freq_d=((p_tp or {}).get("freq_d")),
                pd_j=((p_tp or {}).get("energy_j")),
                freq_mismatch=bool(
                    m_tp and p_tp and (
                        int(m_tp.get("freq_d") or 0)
                        != int(p_tp.get("freq_d") or 0)
                        or int(p_tp.get("freq_p") or 0)
                        != int(p_tp.get("freq_d") or 0))),
            )
        item = dict(
            shape=shape, slo=ds, batch=int(batch),
            pin=int((mixed or pd or {}).get("prompt_len") or 0),
            pout=int((mixed or pd or {}).get("output_len") or 0),
            mixed_tp=((mixed or {}).get("tp_p")),
            mixed_freq=((mixed or {}).get("freq_d")),
            mixed_j=((mixed or {}).get("energy_j")),
            pd_tp_p=((pd or {}).get("tp_p")),
            pd_tp_d=((pd or {}).get("tp_d")),
            pd_freq_p=((pd or {}).get("freq_p")),
            pd_freq_d=((pd or {}).get("freq_d")),
            pd_j=((pd or {}).get("energy_j")),
            pd_j_taxed=((pd or {}).get("energy_j_taxed")),
            pd_e_pre_frac=((pd or {}).get("e_pre_frac")),
            pd_bottleneck=((pd or {}).get("bottleneck")),
            tp_mismatch=bool(
                mixed and pd
                and (int(mixed.get("tp_p") or 0) != int(pd.get("tp_p") or 0)
                     or int(mixed.get("tp_d") or 0) != int(pd.get("tp_d") or 0))),
            same_tp=same,
        )
        if mixed and pd and _finite(mixed.get("energy_j")) \
                and _finite(pd.get("energy_j")) \
                and float(mixed["energy_j"]) > 0:
            item["dJ_pct"] = 100.0 * (
                1.0 - float(pd["energy_j"]) / float(mixed["energy_j"]))
        else:
            item["dJ_pct"] = None
        out.append(item)
    return out


def reconfig_layer(current: dict, target: dict) -> str:
    """current → target 落在哪一层:freq / tp / layout。"""
    cur_l = str(current.get("layout") or LAYOUT_MIXED)
    tgt_l = str(target.get("layout") or LAYOUT_MIXED)
    if cur_l != tgt_l:
        return LAYER_L2
    try:
        cur_tp = int(float(current.get("tp") or current.get("tp_p") or 2))
        tgt_tp = int(float(target.get("tp") or target.get("tp_p") or 2))
    except (TypeError, ValueError):
        cur_tp, tgt_tp = 2, 2
    if cur_tp != tgt_tp:
        return LAYER_L1
    return LAYER_L0


def layer_enabled(costs: Optional[dict], layer: str) -> bool:
    """未测的 L1/L2 关闭;L0 切频永远可用。"""
    sc = costs or DEFAULT_SWITCH_COST
    if layer == LAYER_L0:
        return True
    if layer == LAYER_L1:
        return _finite(sc.get("instance_switch_j")) \
            or _finite(sc.get("instance_start_s"))
    if layer == LAYER_L2:
        return _finite(sc.get("role_switch_cost_j"))
    return False


def rematerialization_open(enabled: bool, costs: Optional[dict] = None) -> bool:
    """--enable-rematerialization 且 L2 代价已测才开 TopologyCoordinator。"""
    return bool(enabled) and layer_enabled(costs, LAYER_L2)


def switch_cost_j(current: dict, target: dict,
                  costs: Optional[dict] = None) -> Optional[float]:
    """current → target 的焦耳代价。None = 该层未测,禁止切换。

    同配置为 0。切频用 E5 瞬态;改 TP/布局必须有实测。
    """
    sc = costs or DEFAULT_SWITCH_COST
    layer = reconfig_layer(current, target)
    if _config_key(current) == _config_key(target):
        return 0.0
    if layer == LAYER_L0:
        return float(sc.get("freq_transient_j") or 0.0)
    if layer == LAYER_L1:
        if _finite(sc.get("instance_switch_j")):
            return float(sc["instance_switch_j"])
        return None
    if _finite(sc.get("role_switch_cost_j")):
        return float(sc["role_switch_cost_j"])
    return None


def _horizon_s(layer: str, costs: Optional[dict] = None) -> float:
    sc = costs or DEFAULT_SWITCH_COST
    keys = {LAYER_L0: "horizon_l0_s", LAYER_L1: "horizon_l1_s",
            LAYER_L2: "horizon_l2_s"}
    raw = sc.get(keys.get(layer, "horizon_l0_s"))
    if _finite(raw) and float(raw) > 0:
        return float(raw)
    return {LAYER_L0: 5.0, LAYER_L1: 300.0, LAYER_L2: 1800.0}[layer]


def reconfig_score(cfg: dict, current: dict,
                   costs: Optional[dict] = None) -> Optional[float]:
    """J + switch_cost / H。不可达返回 None。"""
    cost = switch_cost_j(current, cfg, costs)
    if cost is None:
        return None
    layer = reconfig_layer(current, cfg)
    return float(_row_energy(cfg)) + float(cost) / _horizon_s(layer, costs)


def choose_reconfig(current: dict, candidates: Sequence[dict],
                    costs: Optional[dict] = None,
                    hysteresis: Optional[float] = None) -> Optional[dict]:
    """在可达候选里取 min(J + cost/H);先频后 TP 后 layout。

    未测 L1/L2 的候选丢弃。相对当前增益 < hysteresis 则维持。
    """
    sc = costs or DEFAULT_SWITCH_COST
    hyst = float(hysteresis if hysteresis is not None
                 else sc.get("hysteresis") or 0.15)
    reachable: List[Tuple[float, str, dict]] = []
    for raw in candidates:
        cfg = dict(raw)
        score = reconfig_score(cfg, current, sc)
        if score is None:
            continue
        layer = reconfig_layer(current, cfg)
        reachable.append((score, layer, cfg))
    if not reachable:
        return None
    # 同 score 时优先更便宜的层:freq < tp < layout
    rank = {LAYER_L0: 0, LAYER_L1: 1, LAYER_L2: 2}
    best_score, best_layer, best = min(
        reachable, key=lambda x: (x[0], rank.get(x[1], 9)))
    stay = next((item for item in reachable
                 if _config_key(item[2]) == _config_key(current)), None)
    if stay is not None:
        stay_score = stay[0]
        if stay_score > 0:
            gain = (stay_score - best_score) / stay_score
            if gain < hyst:
                out = dict(stay[2])
                out["reconfig_layer"] = LAYER_L0
                out["reconfig_score"] = stay_score
                out["reconfig_reason"] = "hysteresis"
                return out
    out = dict(best)
    out["reconfig_layer"] = best_layer
    out["reconfig_score"] = best_score
    out["reconfig_reason"] = "ok"
    return out


def feasible_configs(rows: Sequence[dict], size_bucket: str, model: str,
                     slo_name: str = "sharegpt",
                     ttft_s: Optional[float] = None,
                     tpot_s: Optional[float] = None,
                     batch: Optional[int] = 1) -> List[dict]:
    """lookup 可行集(已按操作点均值),供摊还选用。"""
    ds = normalize_dataset(slo_name)
    if ttft_s is None or tpot_s is None:
        if ds not in USER_DATASET_SLO:
            raise KeyError("未知 SLO: %s" % slo_name)
        def_ttft, def_tpot = USER_DATASET_SLO[ds]
        if ttft_s is None:
            ttft_s = def_ttft
        if tpot_s is None:
            tpot_s = def_tpot
    want = str(size_bucket)
    model_key = str(model).lower()
    grouped: Dict[Tuple[str, int, int, int], List[dict]] = {}
    for row in rows:
        if str(row.get("size_bucket") or classify_size_bucket(
                row.get("prompt_len") or 0, row.get("gen_len")
                or row.get("output_len") or 0)) != want:
            continue
        if batch is not None and _row_batch(row) != int(batch):
            continue
        row_model = str(row.get("model") or row.get("model_key") or "").lower()
        if row_model and row_model not in (model_key, model_key.replace("b", "")):
            if model_key not in row_model and row_model not in model_key:
                continue
        if not _row_slo_ok(row, float(ttft_s), float(tpot_s)):
            continue
        grouped.setdefault(_config_key(row), []).append(dict(row))
    out = []
    for key, rs in grouped.items():
        avg = _average_rows(rs)
        layout, tp, freq_p, freq_d = key
        avg.update(layout=layout, tp=tp, freq_p=freq_p, freq_d=freq_d,
                   size_bucket=want, model=model_key, slo=ds)
        out.append(avg)
    return out


def lookup_reconfig(rows: Sequence[dict], current: dict,
                    size_bucket: str, model: str,
                    slo_name: str = "sharegpt",
                    costs: Optional[dict] = None,
                    **kwargs) -> Optional[dict]:
    """查表可行集 + 摊还:能只改频就不改 TP/布局。"""
    cands = feasible_configs(rows, size_bucket, model, slo_name, **kwargs)
    return choose_reconfig(current, cands, costs=costs)


class MigrationCost:
    """实例迁移(启停)成本记录:由 M1 实测填充(启动时长 + 迁移能耗)。"""

    def __init__(self, model: str, tp: int, start_s: float = 0.0,
                 stop_s: float = 0.0, energy_j: float = 0.0):
        self.model = model
        self.tp = tp
        self.start_s = float(start_s)
        self.stop_s = float(stop_s)
        self.energy_j = float(energy_j)

    def as_dict(self) -> dict:
        return dict(model=self.model, tp=self.tp, start_s=self.start_s,
                    stop_s=self.stop_s, energy_j=self.energy_j)
