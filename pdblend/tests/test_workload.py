# -*- coding: utf-8 -*-
"""workload:DistServe 算法 trace 生成(确定性、无放回采样)。"""
from __future__ import annotations

import os
import tempfile

from ecopadg.workload import (
    build_trace, gamma_intervals, sample_lengths, trace_interval_stats,
)
from ecopadg.measure.datasets import Dataset, TestRequest


def _make_ds(path: str, n: int = 50, seed: int = 0):
    reqs = []
    for i in range(n):
        reqs.append(TestRequest(prompt="p%d" % i, prompt_len=100 + i,
                                output_len=10 + i))
    Dataset(dataset_name="syn", reqs=reqs).dump(path)
    return path


def test_gamma_intervals_deterministic():
    a = gamma_intervals(10, rate=2.0, cv=1.0, seed=7)
    b = gamma_intervals(10, rate=2.0, cv=1.0, seed=7)
    assert a == b
    c = gamma_intervals(10, rate=2.0, cv=1.0, seed=8)
    assert a != c


def test_gamma_intervals_statistics():
    xs = gamma_intervals(5000, rate=2.0, cv=1.0, seed=0)
    mean = sum(xs) / len(xs)
    assert abs(mean - 0.5) / 0.5 < 0.05  # 1/rate
    xs2 = gamma_intervals(5000, rate=2.0, cv=3.0, seed=0)
    mean2 = sum(xs2) / len(xs2)
    var2 = sum((x - mean2) ** 2 for x in xs2) / len(xs2)
    cv2 = var2 ** 0.5 / mean2
    assert abs(cv2 - 3.0) / 3.0 < 0.10


def test_sample_lengths_without_replacement(tmp_path):
    ds = _make_ds(str(tmp_path / "syn.ds"), n=50)
    a = sample_lengths(ds, 10, seed=1)
    assert len(a) == 10
    assert len(set(a)) == 10  # 无放回 -> 10 条互不相同
    b = sample_lengths(ds, 10, seed=1)
    assert a == b  # 同 seed 可复现


def test_build_trace_shape_and_order(tmp_path):
    ds = _make_ds(str(tmp_path / "syn.ds"), n=50)
    tr = build_trace(ds, n=8, rate=2.0, cv=1.0, seed=3)
    assert len(tr.rows) == 8
    assert tr.rows[0][0] == 0.0  # 首请求 t=0(DistServe 语义)
    arrs = [r[0] for r in tr.rows]
    assert all(a <= b for a, b in zip(arrs, arrs[1:]))
    assert tr.meta["process"] == "poisson"
    assert tr.meta["rate"] == 2.0


def test_build_trace_gamma_meta(tmp_path):
    ds = _make_ds(str(tmp_path / "syn.ds"), n=50)
    tr = build_trace(ds, n=8, rate=2.0, cv=2.0, seed=3, process="gamma")
    assert tr.meta["cv"] == 2.0
    assert tr.meta["process"] == "gamma"


def test_trace_interval_stats(tmp_path):
    ds = _make_ds(str(tmp_path / "syn.ds"), n=50)
    tr = build_trace(ds, n=40, rate=2.0, cv=1.0, seed=0)
    st = trace_interval_stats(tr.rows)
    assert abs(st["mean_s"] - 0.5) / 0.5 < 0.2
    assert st["n_intervals"] == 39
