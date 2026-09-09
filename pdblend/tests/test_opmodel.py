# -*- coding: utf-8 -*-
"""opmodel:E1b_32B 表加载、预测、DistServe profiler JSON 导出。"""
from __future__ import annotations

import json
import os

import pytest

from ecopadg.opmodel import OpModel


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(
        os.environ.get("PDBLEND_RESULTS", "/workspace/pdblend/motivations/results/E1b_32B"),
        "p1b_latency.csv")),
    reason="缺少 E1b_32B 表")


@pytest.fixture(scope="module")
def om(e1b_tables):
    lat, pw = e1b_tables
    return OpModel.from_tables(lat, pw)


def test_iter_reproduces_measured(om):
    # p1b_latency.csv: decode b=1 @300MHz iter_ms=99.861(重复均值)
    v = om.iter_time_ms(batch=1, freq_mhz=300)
    assert abs(v - 99.86) / 99.86 < 0.05


def test_iter_monotonic_in_freq(om):
    hi = om.iter_time_ms(batch=4, freq_mhz=2520)
    lo = om.iter_time_ms(batch=4, freq_mhz=300)
    assert lo >= hi  # 低频更慢


def test_prefill_reproduces_measured(om):
    # prefill pl=96 @300MHz ttft_corrected_ms=226.16/227.06
    v = om.prefill_time_ms(tokens=96, freq_mhz=300)
    assert abs(v - 226.6) / 226.6 < 0.10


def test_decode_energy_table(om):
    # decode b=1 @300 j_per_token_dynamic=20.25 J/tok -> 20250 mJ
    mj = om.dyn_j_per_token(batch=1, freq_mhz=300)
    assert abs(mj - 20250.0) / 20250.0 < 0.10


def test_residency_and_prefill_energy(om):
    assert om.residency_w() > 0.0
    e = om.prefill_dyn_j_per_token(freq_mhz=2520)
    assert e > 0.0


def test_export_distserve_json_shape(tmp_path, om):
    out = str(tmp_path / "prof.json")
    om.export_distserve_json(model_name="Qwen2.5-32B-Instruct", tp=2,
                             out_path=out)
    with open(out) as f:
        d = json.load(f)
    assert "Qwen2.5-32B-Instruct" in d
    tp2 = d["Qwen2.5-32B-Instruct"]["2"]
    assert "decoding_large_small_bs_threshold" in tp2
    assert len(tp2["prefill"]) == 3
    assert len(tp2["decoding_smallbs"]) == 3
    assert len(tp2["decoding_largebs"]) == 3
    # 系数非负
    assert all(x >= 0 for x in tp2["prefill"])


def test_export_json_formula_reproduces(tmp_path, om):
    out = str(tmp_path / "prof.json")
    om.export_distserve_json(model_name="m", tp=2, out_path=out)
    with open(out) as f:
        d = json.load(f)["m"]["2"]
    # decode 拟合应在实测点的合理误差内(b=8 @2520)
    a, b, c = d["decoding_smallbs"]
    est = a + b * 8 + c * 8
    meas = om.iter_time_ms(batch=8, freq_mhz=2520)
    assert abs(est - meas) / meas < 0.25
