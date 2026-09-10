# -*- coding: utf-8 -*-
"""E1b 操作点表:decode iter / prefill TTFT / 动态 mJ/token 插值。"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _interp1d(pts: List[Tuple[float, float]], x: float) -> float:
    """分段线性;x 落在外侧时钳到端点。"""
    if not pts:
        raise ValueError("空插值点")
    ordered = sorted((float(a), float(b)) for a, b in pts)
    if x <= ordered[0][0]:
        return ordered[0][1]
    if x >= ordered[-1][0]:
        return ordered[-1][1]
    for (x0, y0), (x1, y1) in zip(ordered, ordered[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ordered[-1][1]


def _f(row: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if v in (None, ""):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


class PerfModel:
    """读 p1b_latency.csv / p1b_power.csv;缺列则对应查询失败。"""

    def __init__(self, latency_csv: str, power_csv: str = "",
                 ctx_latency_csv: str = ""):
        self.iter_table: Dict[int, Dict[int, float]] = {}
        self.prefill_table: Dict[int, Dict[int, float]] = {}
        self._dyn_mj: Dict[int, Dict[int, float]] = {}
        self._ctx_slope: Dict[int, float] = {}
        if latency_csv and os.path.isfile(latency_csv):
            self._load_latency(latency_csv)
        if power_csv and os.path.isfile(power_csv):
            self._load_power(power_csv)
        if ctx_latency_csv and os.path.isfile(ctx_latency_csv):
            self._load_ctx(ctx_latency_csv)

    def _load_latency(self, path: str) -> None:
        iters: Dict[int, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        pres: Dict[int, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                freq = _f(row, "freq_mhz")
                if freq is None:
                    continue
                fi = int(freq)
                phase = str(row.get("phase") or "").strip().lower()
                batch = int(_f(row, "batch") or 1)
                plen = int(_f(row, "prompt_len", "tokens") or 0)
                metric = str(row.get("metric") or "").strip().lower()
                val = _f(row, "value_ms", "value")
                it = _f(row, "iter_ms", "tpot_ms")
                ttft = _f(row, "ttft_corrected_ms", "ttft_ms", "prefill_ms")
                if it is None and metric in ("iter_ms", "tpot_ms", "decode_ms"):
                    it = val
                if ttft is None and metric in (
                        "ttft_corrected_ms", "ttft_ms", "prefill_ms"):
                    ttft = val
                if phase in ("decode", "iter", "") and it is not None:
                    if phase == "" and ttft is not None and batch <= 1 and plen:
                        pres[fi][plen].append(ttft)
                    else:
                        iters[fi][batch].append(it)
                if phase in ("prefill", "prompt") and ttft is not None:
                    key = plen or batch
                    pres[fi][key].append(ttft)
                elif phase == "" and ttft is not None and it is None:
                    pres[fi][plen or batch].append(ttft)
        self.iter_table = {f: {b: sum(vs) / len(vs) for b, vs in row.items()}
                           for f, row in iters.items()}
        self.prefill_table = {f: {p: sum(vs) / len(vs) for p, vs in row.items()}
                              for f, row in pres.items()}

    def _load_power(self, path: str) -> None:
        dyn: Dict[int, Dict[int, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                freq = _f(row, "freq_mhz")
                if freq is None:
                    continue
                phase = str(row.get("phase") or "").strip().lower()
                if phase not in ("decode", "iter", ""):
                    continue
                batch = int(_f(row, "batch") or 1)
                jpt = _f(row, "j_per_token_dynamic", "dyn_j_per_token")
                if jpt is None:
                    continue
                # 表里是 J/tok;接口返回 mJ/tok
                dyn[int(freq)][batch].append(jpt * 1000.0)
        self._dyn_mj = {f: {b: sum(vs) / len(vs) for b, vs in row.items()}
                        for f, row in dyn.items()}

    def _load_ctx(self, path: str) -> None:
        # e8: 每频一条斜率 ms/token
        by_f: Dict[int, List[float]] = defaultdict(list)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                freq = _f(row, "freq_mhz")
                slope = _f(row, "slope_ms_per_tok", "ctx_slope")
                if freq is None or slope is None:
                    continue
                by_f[int(freq)].append(slope)
        self._ctx_slope = {f: sum(vs) / len(vs) for f, vs in by_f.items()}

    def iter_time_ms(self, batch: int, freq_mhz: int,
                     ctx_tokens: float = 272.0) -> float:
        if not self.iter_table:
            raise ValueError("无 decode 表")
        base = self._grid2(self.iter_table, int(freq_mhz), int(batch))
        if self._ctx_slope:
            slope = _interp1d(
                [(float(f), float(s)) for f, s in self._ctx_slope.items()],
                float(freq_mhz))
            base = base + max(float(ctx_tokens) - 272.0, 0.0) * slope
        return float(base)

    def prefill_time_ms(self, tokens: int, freq_mhz: int = 2520) -> float:
        if not self.prefill_table:
            raise ValueError("无 prefill 表")
        return float(self._grid2(self.prefill_table, int(freq_mhz), int(tokens)))

    def dyn_j_per_token(self, batch: int, freq_mhz: int) -> float:
        """decode 动态能耗,mJ/token。"""
        if not self._dyn_mj:
            raise ValueError("无 decode 能耗表")
        return float(self._grid2(self._dyn_mj, int(freq_mhz), int(batch)))

    @staticmethod
    def _grid2(table: Dict[int, Dict[int, float]],
               x: int, y: int) -> float:
        """先在每条 x 上对 y 插值,再对 x 插值。"""
        xs = sorted(table)
        if not xs:
            raise ValueError("空表")
        col: List[Tuple[float, float]] = []
        for xv in xs:
            ys = sorted(table[xv])
            if not ys:
                continue
            pts = [(float(k), float(table[xv][k])) for k in ys]
            col.append((float(xv), _interp1d(pts, float(y))))
        return _interp1d(col, float(x))
