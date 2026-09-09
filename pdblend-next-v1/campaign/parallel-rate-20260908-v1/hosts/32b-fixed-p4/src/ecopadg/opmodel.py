# -*- coding: utf-8 -*-
"""OpModel:实测操作点表 + 预测 + DistServe profiler JSON 导出。"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ecopadg.measure.perfmodel import PerfModel, _interp1d


class HybridIterLUT:
    """hybrid iteration LUT:(prefill_tokens, n_decode, kv_tokens, freq)
    → p50/p90 时延与板功耗。缺表时调用方回退 E1b 二维插值。"""

    def __init__(self, rows: List[dict]):
        self.rows = list(rows)

    @classmethod
    def from_csv(cls, path: str) -> "HybridIterLUT":
        with open(path, encoding="utf-8") as f:
            return cls(list(csv.DictReader(f)))

    def query(self, prefill_tokens: float, n_decode: int, kv_tokens: float,
              freq_mhz: int) -> Tuple[float, float, float]:
        if not self.rows:
            raise ValueError("empty hybrid LUT")
        best, best_d = None, None
        for r in self.rows:
            d = ((float(r["prefill_tokens"]) - prefill_tokens) ** 2
                 + (float(r["n_decode"]) - n_decode) ** 2
                 + (float(r.get("kv_tokens") or 0) - kv_tokens) ** 2
                 + ((float(r["freq_mhz"]) - freq_mhz) / 100.0) ** 2)
            if best_d is None or d < best_d:
                best_d, best = d, r
        p50 = float(best["iter_p50_ms"])
        p90 = float(best.get("iter_p90_ms") or p50)
        w = float(best.get("board_w") or 0.0)
        return p50, p90, w


class OpModel(PerfModel):
    """加载 p1b 表;提供能量/驻留/导出接口(继承 iter/prefill 插值)。"""

    def __init__(self, latency_csv: str, power_csv: str = "",
                 ctx_latency_csv: str = ""):
        super().__init__(latency_csv, power_csv, ctx_latency_csv)
        self.hybrid_lut: Optional[HybridIterLUT] = None

    @staticmethod
    def from_tables(latency_csv: str, power_csv: str,
                    ctx_latency_csv: str = "",
                    hybrid_csv: str = "") -> "OpModel":
        om = OpModel(latency_csv, power_csv or "", ctx_latency_csv)
        om._load_power_extras(power_csv)
        if hybrid_csv and os.path.exists(hybrid_csv):
            om.hybrid_lut = HybridIterLUT.from_csv(hybrid_csv)
        return om

    def iter_time_ms(self, batch: int, freq_mhz: int,
                     ctx_tokens: float = 272.0) -> float:
        """decode 一步。有 hybrid LUT 时走统一 (0, b, kv, f) 查询。"""
        if self.hybrid_lut is not None:
            p50, _, _ = self.hybrid_lut.query(
                0.0, int(batch), float(ctx_tokens), int(freq_mhz))
            return p50
        return super().iter_time_ms(int(batch), int(freq_mhz),
                                    float(ctx_tokens))

    def hybrid_iter_ms(self, prefill_tokens: float, n_decode: int,
                       kv_tokens: float, freq_mhz: int,
                       quantile: str = "p50") -> float:
        """混相一步时延。有 LUT 用 LUT;否则 E1b decode iter + prefill 份额。

        prefill_tokens 是该步 Σtok(容量口径),不是请求条数 K。
        """
        if self.hybrid_lut is not None:
            p50, p90, _ = self.hybrid_lut.query(
                prefill_tokens, n_decode, kv_tokens, freq_mhz)
            return p90 if quantile == "p90" else p50
        it = super().iter_time_ms(max(int(n_decode), 1), int(freq_mhz),
                                  float(kv_tokens))
        if prefill_tokens > 0:
            pre = self.prefill_time_ms(int(prefill_tokens), int(freq_mhz))
            return max(it, pre / max(float(n_decode), 1.0))
        return it

    def _load_power_extras(self, power_csv: str) -> None:
        if not power_csv or not os.path.exists(power_csv):
            self._prefill_j_per_tok: Dict[int, List[float]] = {}
            self._idle_w: Dict[int, List[float]] = {}
            self._mean_w: Dict[Tuple[str, int], List[float]] = {}
            return
        pre: Dict[int, List[float]] = defaultdict(list)
        idle: Dict[int, List[float]] = defaultdict(list)
        mw: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        with open(power_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                freq = int(r["freq_mhz"])
                phase = r["phase"]
                try:
                    mean_w = float(r["mean_w"])
                except (TypeError, ValueError):
                    continue
                mw[(phase, freq)].append(mean_w)
                if phase == "idle":
                    idle[freq].append(mean_w)
                elif phase == "prefill":
                    try:
                        e = float(r["energy_j"])
                        batch = int(r["batch"])
                        plen = int(r["prompt_len"])
                        reps = max(int(r.get("repeat") or 1), 1)
                    except (TypeError, ValueError, KeyError):
                        continue
                    if batch > 0 and plen > 0:
                        # 探针行 energy_j 为 reps 次重复之和(P1B 协议)
                        pre[freq].append(e / (batch * plen * reps))
        self._prefill_j_per_tok = {k: v for k, v in pre.items() if v}
        self._idle_w = {k: v for k, v in idle.items() if v}
        self._mean_w = {k: v for k, v in mw.items() if v}

    def prefill_dyn_j_per_token(self, freq_mhz: int) -> float:
        """prefill 每 token 动态能耗(J/token),按 freq 插值。"""
        if not self._prefill_j_per_tok:
            raise ValueError("未加载 prefill 能耗(需 p1b_power.csv)")
        pts = [(f, float(np.mean(vs)))
               for f, vs in sorted(self._prefill_j_per_tok.items())]
        return _interp1d(pts, float(freq_mhz))

    def residency_w(self, freq_mhz: Optional[int] = None) -> float:
        """实例驻留功率(W,idle 行均值;默认取最高频档)。"""
        if not self._idle_w:
            return 0.0
        if freq_mhz is None:
            freq_mhz = max(self._idle_w)
        pts = [(f, float(np.mean(vs)))
               for f, vs in sorted(self._idle_w.items())]
        return _interp1d(pts, float(freq_mhz))

    def power_w(self, phase: str, freq_mhz: int) -> float:
        """某 phase 某频率的平均功率(W);无数据返回 NaN。"""
        key = (phase, freq_mhz)
        if key in self._mean_w:
            return float(np.mean(self._mean_w[key]))
        pts = [(k[1], float(np.mean(vs)))
               for k, vs in self._mean_w.items() if k[0] == phase]
        if not pts:
            return float("nan")
        return _interp1d(sorted(pts), float(freq_mhz))

    def export_distserve_json(self, model_name: str, tp: int,
                              out_path: str, freq_mhz: int = 2520) -> None:
        """导出与 DistServe time_estimator 同构的画像 JSON。

        prefill: [a,b,c] → delay_ms = a + b*Σtok + c*Σtok²(指定 freq 档拟合)
        decode:  分小/大批两段 [a,b,c] → delay_ms = a + b*Σgen + c*bs
        """
        a_p, b_p, c_p = self._fit_prefill_coeffs(freq_mhz)
        thr = self._fit_decode_threshold(freq_mhz)
        small = self._fit_decode_coeffs(upper=thr, freq_mhz=freq_mhz)
        large = self._fit_decode_coeffs(lower=thr + 1, freq_mhz=freq_mhz)
        if large == (0.0, 0.0, 0.0):
            large = tuple(small)  # 大批段样本不足(如 batch 只到 8),回退小批系数
        payload = {
            model_name: {
                str(tp): {
                    "decoding_large_small_bs_threshold": thr,
                    "prefill": [a_p, b_p, c_p],
                    "decoding_smallbs": list(small),
                    "decoding_largebs": list(large),
                }
            }
        }
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    def _prefill_cells(self, freq_mhz: int = 2520) -> List[Tuple[float, float, float]]:
        """(Σtok, Σtok², ttft_ms) @指定频率。"""
        out = []
        for f, table in self.prefill_table.items():
            if f != freq_mhz:
                continue
            for plen, v in table.items():
                out.append((float(plen), float(plen) ** 2, float(v)))
        return out

    def _fit_prefill_coeffs(self, freq_mhz: int = 2520) -> Tuple[float, float, float]:
        cells = self._prefill_cells(freq_mhz)
        if not cells:
            return (0.0, 0.0, 0.0)
        x = np.array([[1.0, s, s2] for s, s2, _ in cells])
        y = np.array([v for _, _, v in cells])
        a, b, c = np.linalg.lstsq(x, y, rcond=None)[0]
        return (max(float(a), 0.0), max(float(b), 0.0), max(float(c), 0.0))

    def _fit_decode_threshold(self, freq_mhz: int = 2520) -> int:
        bs = sorted(self.iter_table.get(freq_mhz, {}))
        if not bs:
            return 95
        if len(bs) < 4:
            return max(bs)
        return int(np.median(bs))

    def _fit_decode_coeffs(self, lower: Optional[int] = None,
                           upper: Optional[int] = None,
                           freq_mhz: int = 2520) -> Tuple[float, float, float]:
        xs, ys = [], []
        for b, v in self.iter_table.get(freq_mhz, {}).items():
            if lower is not None and b < lower:
                continue
            if upper is not None and b > upper:
                continue
            xs.append(float(b))
            ys.append(float(v))
        if not xs:
            return (0.0, 0.0, 0.0)
        if len(xs) < 2:
            return (max(ys[0], 0.0), 0.0, 0.0)
        slope, a = np.polyfit(xs, ys, 1)  # polyfit 降幂序:[斜率, 截距]
        half = max(float(slope), 0.0) / 2.0
        return (max(float(a), 0.0), half, half)


def default_table_dir(model_key: str, ws: Optional[str] = None) -> str:
    """14b/32b/72b 操作点表目录。优先 new-motivations,兼容旧 motivations 路径。"""
    root = ws or os.environ.get("PDBLEND_WS", "/root/workspace")
    key = str(model_key).lower()
    aliases = {
        "14b": ("e1b_14b_tp2", "E1b_14B_tp2"),
        "14": ("e1b_14b_tp2", "E1b_14B_tp2"),
        "32b": ("e1b_32b_tp2", "E1b_32B"),
        "32": ("e1b_32b_tp2", "E1b_32B"),
        "72b": ("e1b_72b_tp4", "E1b_72b_tp4"),
        "72": ("e1b_72b_tp4", "E1b_72b_tp4"),
    }
    if key not in aliases:
        raise KeyError("未知 model_key: %s" % model_key)
    names = aliases[key]
    env = os.environ.get("PDBLEND_TABLE_DIR")
    if env:
        return env
    candidates = []
    # Accept both workspace root (/root/workspace) and pdblend root.
    prefixes = (
        os.path.join(root, "pdblend"),
        root,
    )
    for prefix in prefixes:
        for name in names:
            candidates.append(os.path.join(
                prefix, "new-motivations", "results", name))
    nm_dirs = [
        os.path.join(root, "pdblend", "new-motivations", "results"),
        os.path.join(root, "new-motivations", "results"),
    ]
    for nm in nm_dirs:
        if not os.path.isdir(nm):
            continue
        dated = sorted(
            (p for p in os.listdir(nm) if p.endswith("-profile-14b")
             or p.endswith("-e1b-14b-probe")),
            reverse=True)
        for stamp in dated:
            for name in names:
                candidates.append(os.path.join(nm, stamp, name))
    preferred, legacy = [], []
    for path in candidates:
        if ("/motivations/results/" in path.replace("\\", "/")
                and "/new-motivations/" not in path.replace("\\", "/")):
            legacy.append(path)
        else:
            preferred.append(path)
    for prefix in prefixes:
        legacy.append(os.path.join(
            prefix, "motivations", "results", names[-1]))
    if key in ("72b", "72"):
        for prefix in prefixes:
            legacy.append(os.path.join(
                prefix, "motivations", "results", "m1", "E1b_72b_tp4"))
    for path in preferred:
        if os.path.isfile(os.path.join(path, "p1b_latency.csv")):
            return path
    for path in legacy:
        if os.path.isfile(os.path.join(path, "p1b_latency.csv")):
            return path
    return (preferred or legacy or [""])[0]


def load_opmodel_from_dir(base: str) -> Optional[OpModel]:
    """从 E1b 目录加载 OpModel;缺表返回 None。"""
    if not base:
        return None
    lat = os.path.join(base, "p1b_latency.csv")
    pw = os.path.join(base, "p1b_power.csv")
    ctx = os.path.join(base, "e8_ctx.csv")
    hy = os.path.join(base, "hybrid_iter.csv")
    if os.path.exists(lat) and os.path.exists(pw):
        return OpModel.from_tables(
            lat, pw,
            ctx if os.path.exists(ctx) else "",
            hy if os.path.exists(hy) else "")
    return None


def load_opmodel(source) -> Optional[OpModel]:
    """统一入口:目录路径、目录列表或 model_key。"""
    if source is None:
        return None
    if isinstance(source, OpModel):
        return source
    if isinstance(source, (list, tuple)):
        for item in source:
            om = load_opmodel(item)
            if om is not None:
                return om
        return None
    path = str(source)
    if os.path.isdir(path):
        return load_opmodel_from_dir(path)
    try:
        return load_opmodel_from_dir(default_table_dir(path))
    except KeyError:
        return None


# 控制器/CLI 共用的模型表(禁止再复制 MODEL_CFG.tables)
MODEL_TABLES = {
    "14b": dict(tp=2, key="14b"),
    "32b": dict(tp=2, key="32b"),
    "72b": dict(tp=4, key="72b"),
}

_RUNTIME_EXTRA = {
    "14b": dict(tp=2, capacity_req_s=4.0, gpus=[0, 1],
                fallback_decode_freq=1500),
    "32b": dict(tp=2, capacity_req_s=2.0, gpus=[0, 1],
                fallback_decode_freq=1500),
    "72b": dict(tp=4, capacity_req_s=0.9, gpus=[0, 1, 2, 3],
                fallback_decode_freq=2520),
}


def e1b_dir_synthetic(base: str) -> bool:
    """E1b_14B_tp2 等合成表目录:有 SYNTHETIC 标记则禁止 DistServe 仿真选配。"""
    if not base:
        return False
    return os.path.isfile(os.path.join(str(base), "SYNTHETIC"))


def freq_model_trustworthy(opmodel, freqs=None, batch: int = 8,
                           pe1_calibrated: bool = False) -> bool:
    """decode iter 随频率可分或已有 PE1 μ_S90 才打开深 DVFS。

    14B-TP2 合成表三档 iter 几乎相同(≈21ms),比值 < 1.15 → 不可信。
    """
    if pe1_calibrated:
        return True
    if opmodel is None:
        return False
    from ecopadg.types import FREQ_TRUST_RATIO
    cands = tuple(int(f) for f in (freqs or (2520, 900)))
    if len(cands) < 2:
        return True
    fmin, fmax = min(cands), max(cands)
    try:
        t_lo = float(opmodel.iter_time_ms(int(batch), int(fmin)))
        t_hi = float(opmodel.iter_time_ms(int(batch), int(fmax)))
    except (TypeError, ValueError, AttributeError):
        return False
    if t_hi <= 0 or t_lo != t_lo or t_hi != t_hi:
        return False
    return (t_lo / t_hi) >= float(FREQ_TRUST_RATIO)


def model_runtime_cfg(model_key: str, ws: Optional[str] = None) -> dict:
    """控制器唯一模型配置:表路径 + TP + 容量先验。"""
    key = str(model_key).lower()
    if key not in _RUNTIME_EXTRA:
        raise KeyError("未知 model_key: %s" % model_key)
    cfg = dict(_RUNTIME_EXTRA[key])
    cfg["key"] = key
    cfg["tables"] = default_table_dir(key, ws)
    return cfg
