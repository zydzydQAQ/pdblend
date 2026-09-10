# -*- coding: utf-8 -*-
"""Poisson/Gamma 到达 trace 生成 —— 原样对齐 DistServe 基准 harness。

来源(DistServe,Apache-2.0):
  simdistserve/base/workload.py::get_poisson_interarrival / get_gamma_interarrival
    shape = 1/cv², scale = cv²/rate(秒);poisson 即 cv=1;
  evaluation/2-benchmark-serving/2-benchmark-serving.py::sample_requests
    random.sample 无放回;请求 i 的绝对到达 = Σ_{j<i} interval(j),首请求 t=0。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ecopadg.measure.datasets import Dataset

# (arrival_s, prompt_len, output_len)
TraceRow = Tuple[float, int, int]


@dataclass
class Trace:
    meta: Dict = field(default_factory=dict)
    rows: List[TraceRow] = field(default_factory=list)


def gamma_intervals(n: int, rate: float, cv: float, seed: int) -> List[float]:
    """DistServe get_gamma_interarrival 原样:n 个间隔(秒)。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if rate <= 0:
        raise ValueError("rate 必须为正")
    if cv <= 0:
        raise ValueError("cv 必须为正")
    if seed is not None:
        np.random.seed(seed)
    shape = 1.0 / (cv * cv)
    scale = cv * cv / rate
    return [float(x) for x in np.random.gamma(shape, scale, size=n)]


def sample_lengths(ds_path: str, n: int, seed: int) -> List[Tuple[int, int]]:
    """DistServe sample_requests 同款:random.sample 无放回 → [(plen, olen)]。"""
    rng = random.Random(seed)
    ds: Dataset = Dataset.load(ds_path)
    if n > len(ds.reqs):
        raise ValueError("n=%d 大于数据集大小 (%d)" % (n, len(ds.reqs)))
    return [(r.prompt_len, r.output_len) for r in rng.sample(ds.reqs, n)]


def build_trace(ds_path: str, n: int, rate: float, cv: float = 1.0,
                seed: int = 0, process: str = "poisson") -> Trace:
    """生成一次到达 trace:首请求 t=0,第 i 请求 t=Σ_{j<i} interval(j)。"""
    if process == "poisson":
        cv = 1.0
    elif process == "gamma":
        pass
    else:
        raise ValueError("process 只支持 poisson/gamma")
    lengths = sample_lengths(ds_path, n, seed)
    intervals = gamma_intervals(n, rate, cv, seed)
    arrivals = [0.0]
    for itv in intervals[:-1]:
        arrivals.append(arrivals[-1] + itv)
    rows: List[TraceRow] = [
        (round(arrivals[i], 6), lengths[i][0], lengths[i][1])
        for i in range(n)
    ]
    meta = dict(dataset=ds_path.split("/")[-1].replace(".ds", ""),
                process=process, rate=rate, cv=cv, seed=seed, n=n,
                unit="s", source="DistServe: workload.py::get_gamma_interarrival")
    return Trace(meta=meta, rows=rows)


def trace_interval_stats(rows: Sequence[TraceRow]) -> Dict[str, float]:
    """到达间隔统计(均值/变异系数/个数)。"""
    if len(rows) < 2:
        return dict(mean_s=float("nan"), cv=float("nan"), n_intervals=0)
    ivs = [rows[i][0] - rows[i - 1][0] for i in range(1, len(rows))]
    mean = float(np.mean(ivs))
    std = float(np.std(ivs))
    return dict(mean_s=mean, cv=(std / mean if mean > 0 else float("nan")),
                n_intervals=len(ivs))
