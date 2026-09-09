# -*- coding: utf-8 -*-
"""metrics:SLO attainment(P50/90/99 分桶)、能效统计、逐请求归因。"""
from __future__ import annotations

import pytest

from ecopadg.metrics import (
    align_iso_load, attribute_energy, caliber_b_working_points,
    classify_run_validity, clip_power_window, emitted_output_tokens,
    energy_from_power_rows, energy_summary, gpu_busy_frac,
    latency_bucket_attainment,
    percentiles, slo_attainment, slo_non_inferior, summarize_bench,
)
from ecopadg.types import SloSpec
from tests.conftest import SynthOpModel


SLO = SloSpec(ttft_s=5.0, tpot_s=0.1)


def test_percentiles_known():
    import math
    v = [float(i) for i in range(1, 101)]  # 1..100
    p = percentiles(v, ps=(50, 90, 99))
    # np.percentile linear:idx=(p/100)*(n-1) 线性插值
    assert p[50] == pytest.approx(50.5)
    assert p[90] == pytest.approx(90.1)
    assert p[99] == pytest.approx(99.01)
    pe = percentiles([], ps=(50, 90, 99))
    assert all(math.isnan(pe[k]) for k in (50, 90, 99))


def _rows():
    # 4 条:第 4 条超 SLO(慢尾)
    return [
        dict(rid=0, success=True, latency_s=1.0, ttft_s=0.5, tpot_s=0.05,
             prompt_len=100, output_len=100, slo_ok=True),
        dict(rid=1, success=True, latency_s=2.0, ttft_s=0.6, tpot_s=0.06,
             prompt_len=100, output_len=100, slo_ok=True),
        dict(rid=2, success=True, latency_s=3.0, ttft_s=0.7, tpot_s=0.07,
             prompt_len=100, output_len=100, slo_ok=True),
        dict(rid=3, success=True, latency_s=10.0, ttft_s=6.0, tpot_s=0.2,
             prompt_len=100, output_len=100, slo_ok=False),
    ]


def test_slo_attainment():
    assert slo_attainment(_rows(), SLO) == pytest.approx(0.75)


def test_latency_bucket_attainment():
    st = latency_bucket_attainment(_rows(), SLO)
    assert sum(b["n"] for b in st["buckets"]) == 4
    # 最慢桶(≥P99)attainment 为 0,最快桶为 1
    assert st["buckets"][0]["attainment"] == 1.0
    assert st["buckets"][-1]["attainment"] == 0.0


def test_summarize_bench():
    s = summarize_bench(_rows(), SLO)
    assert s["completed"] == 4
    assert s["slo_attainment"] == pytest.approx(0.75)
    # np.percentile linear: [0.5,0.6,0.7,6.0] 的 P50 = 0.65
    assert s["ttft_p50_s"] == pytest.approx(0.65)
    assert s["ttft_avg_s"] == pytest.approx(sum([0.5, 0.6, 0.7, 6.0]) / 4.0)
    assert "tpot_p99_s" in s


def test_gpu_busy_frac_idle_threshold():
    rows = [
        (0.0, [30.0, 80.0]),
        (1.0, [30.0, 80.0]),
        (2.0, [80.0, 80.0]),
    ]
    # gpu0: 1/3 busy, gpu1: 3/3 → mean 2/3
    assert gpu_busy_frac(rows, idle_w=35.0) == pytest.approx(2.0 / 3.0)


def test_energy_from_power_rows():
    rows = [(float(i), [100.0, 100.0]) for i in range(11)]  # 2 卡 ×100W ×10s
    e = energy_from_power_rows(rows)
    assert e["total_j"] == pytest.approx(2000.0)
    assert e["mean_w"] == pytest.approx(200.0)


def test_energy_trapezoid_ramp():
    rows = [(0.0, [0.0]), (10.0, [100.0])]  # 线性爬升 → 积分 500J
    assert energy_from_power_rows(rows)["total_j"] == pytest.approx(500.0)


def test_attribute_energy_and_summary():
    om = SynthOpModel()
    bench = [
        dict(rid=0, prompt_len=100, output_len=10, slo_ok=True),
        dict(rid=1, prompt_len=200, output_len=20, slo_ok=False),
    ]
    e = attribute_energy(bench, om, freq_prefill=2520, freq_decode=1500,
                         batch_hat=2)
    assert len(e) == 2
    # 请求0:100×2e-3 + 10×1.002 = 0.2 + 10.02 = 10.22 J
    assert e[0] == pytest.approx(10.22, abs=0.01)
    assert e[1] > e[0]
    power_rows = [(0.0, [140.0]), (100.0, [140.0])]
    s = energy_summary(power_rows, bench, om, freq_prefill=2520,
                       freq_decode=1500, batch_hat=2)
    assert s["total_j"] == pytest.approx(14000.0)
    assert s["predicted_dyn_j"] > 0
    assert s["j_per_output_token"] > 0
    # 达标请求逐请求能耗 P50/90/99
    assert "p50_j" in s["attained_per_req_j"]
    assert s["attained_per_req_j"]["p50_j"] == pytest.approx(10.22, abs=0.01)
    assert s["attained_n"] == 1
    assert s["mean_w"] == pytest.approx(140.0)
    assert s["goodput_per_w"] > 0
    import math
    assert math.isnan(s["gpu_util"])
    assert s["power_busy_frac"] == pytest.approx(1.0)


def test_caliber_b_picks_max_feasible_rate():
    cells = [
        dict(system="mixed", rate=1.0, slo_attainment=0.95, j_per_req=10),
        dict(system="mixed", rate=2.0, slo_attainment=0.91, j_per_req=12),
        dict(system="mixed", rate=4.0, slo_attainment=0.50, j_per_req=8),
        dict(system="ds_pd", rate=1.0, slo_attainment=0.80, j_per_req=9),
        dict(system="ecospd", rate=1.0, slo_attainment=0.99, j_per_req=7),
        dict(system="ecospd", rate=2.0, slo_attainment=0.92, j_per_req=11),
    ]
    pts = caliber_b_working_points(cells, min_attainment=0.9)
    by = {p["system"]: p for p in pts}
    assert by["mixed"]["feasible"] is True
    assert by["mixed"]["rate"] == 2.0
    assert by["mixed"]["j_per_req"] == 12
    assert by["ds_pd"]["feasible"] is False
    assert by["ecospd"]["rate"] == 2.0
    strict = caliber_b_working_points(cells, min_attainment=0.99)
    bys = {p["system"]: p for p in strict}
    assert bys["ecospd"]["rate"] == 1.0
    assert bys["mixed"]["feasible"] is False


def test_emitted_tokens_prefer_generated():
    r = dict(output_len=200, generated_tokens=12)
    assert emitted_output_tokens(r) == 12


def test_error_row_fails_slo():
    rows = [dict(ttft_s=0.1, tpot_s=0.01, error="controller")]
    assert slo_attainment(rows, SLO) == 0.0


def test_slo_non_inferior_delta():
    assert slo_non_inferior(0.956, 0.960, 0.01)
    assert not slo_non_inferior(0.82, 0.996, 0.01)


def test_classify_run_validity_errors():
    rows = [dict(error="x") for _ in range(10)]
    assert classify_run_validity(rows, n_expected=10) == "invalid_run"
    ok = [dict() for _ in range(10)]
    assert classify_run_validity(ok, n_expected=10) == "ok"


def test_clip_power_window():
    rows = [(float(i), [10.0]) for i in range(20)]
    clipped = clip_power_window(rows, 5.0, 8.0, pad_s=2.0)
    assert clipped[0][0] == 3.0
    assert clipped[-1][0] == 10.0


def test_align_iso_load_marks_slo_and_n_mismatch():
    cells = [
        dict(dataset="sg", system="mixed", rate=1.2, n_requests=500,
             seed="0", gpu_count=8, slo_attainment=0.978, total_j=100),
        dict(dataset="sg", system="ecospd", rate=1.2, n_requests=500,
             seed="0", gpu_count=8, slo_attainment=0.626, total_j=80),
        dict(dataset="sg", system="ds_pd", rate=1.2, n_requests=200,
             seed="0", gpu_count=8, slo_attainment=0.98, total_j=90),
    ]
    out = align_iso_load(cells)
    by = {(r["system"], r["n_requests"]): r for r in out}
    assert by[("ecospd", 500)]["validity"] == "invalid_slo"
    assert by[("mixed", 500)]["iso_load_ok"] is True
    assert by[("ds_pd", 200)]["validity"] == "invalid_load"
