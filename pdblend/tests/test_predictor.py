# -*- coding: utf-8 -*-
"""predictor:Partition × λ̂ → (可行, attainment, 功率) 解析预测(测试先行,M4)。

预测器只需正确"排序"候选分区(容量单调、能耗单调、过载不可行),
绝对精度由 PE1 实测校准(rho_knee / 功率标定)。
"""
from __future__ import annotations

import pytest

from ecopadg.predictor import (
    ApproxPredictor, LoadMonitor, WorkloadStats, prefill_token_pack,
    evaluate_capacity_trust, default_pe1_grid_path,
)
from ecopadg.types import TOKEN_BUDGET, Partition, SloSpec


@pytest.fixture
def stats():
    # ShareGPT 量级:平均 prompt 600、输出 200
    return WorkloadStats(prompt_len=600, output_len=200)


@pytest.fixture
def pred(synth_opmodel, stats):
    return ApproxPredictor(
        opmodel=synth_opmodel, stats=stats, slo=SloSpec(),
        freq_candidates=(2520, 1800, 1500, 1200, 900, 600),
        gpu_idle_w=35.0, pd_prompt_threshold=512)


def P(nm=0, np_=0, nd=0, tp=2, total=8):
    return Partition(n_mixed=nm, n_prefill=np_, n_decode=nd,
                     tp_mixed=tp, tp_prefill=tp, tp_decode=tp,
                     gpus_total=total)


def test_capacity_scales_with_instances(pred):
    lam = 1.0
    p2 = pred.predict(P(nm=2), lam)
    p4 = pred.predict(P(nm=4), lam)
    assert p4.attainment >= p2.attainment
    assert pred.capacity(P(nm=4)) > pred.capacity(P(nm=2))


def test_overload_unattainable(pred):
    mu = pred.capacity(P(nm=1))
    pr = pred.predict(P(nm=1), lam_hat=mu * 2.0)
    assert not pr.attainable


def test_attainment_monotonic_in_lambda(pred):
    part = P(nm=2)
    a_low = pred.predict(part, 0.2).attainment
    a_high = pred.predict(part, pred.capacity(part) * 0.95).attainment
    assert a_low >= a_high


def test_power_fewer_instances_cheaper_at_low_load(pred):
    lam = 0.2
    e2 = pred.predict(P(nm=2), lam).energy_j_per_s
    e4 = pred.predict(P(nm=4), lam).energy_j_per_s
    assert e2 < e4  # 低负载下收缩实例省驻留功率


def test_power_counts_unused_gpus_idle(pred):
    # 2×TP2 只用 4 卡,其余 4 卡按空卡 idle 计入(整机 8 卡口径)
    pr = pred.predict(P(nm=2), 0.2)
    assert pr.power_w >= 4 * 35.0


def test_energy_proxy_is_not_silent_power(pred):
    # ρ 过膝后 att<1:energy_j_per_s = P/att ≠ P。禁止再写成 power_w。
    part = P(nm=2)
    mu = pred.capacity(part)
    pr = pred.predict(part, mu * 0.8)
    assert pr.power_w > 0 and pr.attainment < 1.0
    assert pr.energy_j_per_s == pytest.approx(pr.power_w / pr.attainment,
                                              rel=1e-6)
    assert pr.energy_j_per_s != pytest.approx(pr.power_w, rel=1e-3)


def test_asymmetric_pd_capacity(pred):
    mu11 = pred.rate_pd(1, 1)
    mu12 = pred.rate_pd(1, 2)
    mu21 = pred.rate_pd(2, 1)
    assert mu12 >= mu11 and mu21 >= mu11
    assert pred.capacity(P(np_=1, nd=2)) == pytest.approx(mu12)


def test_prefill_token_pack_fills_budget():
    pack, n_req = prefill_token_pack(600, 8192)
    assert n_req == 8192 // 600
    assert pack == n_req * 600
    pack_big, n_big = prefill_token_pack(9000, 8192)
    assert n_big == 1 and pack_big == 9000


def test_stage_rate_prefill_uses_token_budget(pred):
    # Synth: T(ms)=50+0.05*Σtok。L=600,B=8192 → 13 req / T(7800)=440ms
    pack, n_req = prefill_token_pack(pred.stats.prompt_len, pred.token_budget)
    assert pack == 7800 and n_req == 13
    t_s = (50.0 + 0.05 * pack) / 1000.0
    assert pred.stage_rate_prefill(1) == pytest.approx(n_req / t_s, rel=1e-6)
    assert pred.stage_rate_prefill(2) == pytest.approx(2 * n_req / t_s, rel=1e-6)
    # 单条超预算:独占,退回 1/T(L)
    pred.stats.prompt_len = 9000
    t_solo = (50.0 + 0.05 * 9000) / 1000.0
    assert pred.stage_rate_prefill(1) == pytest.approx(1.0 / t_solo, rel=1e-6)
    pred.stats.prompt_len = 600


def test_rate_mixed_ttft_uses_single_prompt(pred):
    # mixed 容量仍用该条 L 的 TTFT,不把 TOKEN_BUDGET 组批算进去
    it_ms = pred.opmodel.iter_time_ms(pred.b_ref, max(pred.freqs),
                                      pred.stats.prompt_len
                                      + pred.stats.output_len / 2.0)
    pre_ms = pred.opmodel.prefill_time_ms(600, max(pred.freqs))
    per_req = (pre_ms + pred.stats.output_len * it_ms) / 1000.0
    assert pred.rate_mixed(1) == pytest.approx(pred.b_ref / per_req, rel=1e-6)
    assert pred.token_budget == TOKEN_BUDGET


def test_segment_capacity_min_of_stages(pred, stats):
    # 1P1D 段容量 = min(prefill 速率, decode 速率)
    mu_seg = pred.capacity(P(np_=1, nd=1))
    mu_p = pred.stage_rate_prefill(1)
    mu_d = pred.stage_rate_decode(1)
    assert mu_seg == pytest.approx(min(mu_p, mu_d), rel=1e-6)


def test_hybrid_splits_load_by_prompt_threshold(pred):
    # 混合分区:长 prompt 走 PD,短走 mixed;frac_long 影响两侧负载
    # (λ=0.4 使两侧 ρ 都低于 rho_max:段容量在 Synth 下 ≈0.5 req/s)
    pred.stats.frac_long = 0.5
    pr = pred.predict(P(nm=2, np_=1, nd=1), 0.4)
    assert pr.attainable
    lam_m, lam_pd = pred.split_lambda(P(nm=2, np_=1, nd=1), 0.4)
    assert lam_m == pytest.approx(0.2)
    assert lam_pd == pytest.approx(0.2)


def test_empty_partition_unattainable(pred):
    pr = pred.predict(P(), 0.5)
    assert not pr.attainable


def test_predict_respects_partition_freq(pred):
    p_hi = Partition(n_mixed=2, tp_mixed=2, freq_mixed=2520, gpus_total=8)
    p_lo = Partition(n_mixed=2, tp_mixed=2, freq_mixed=900, gpus_total=8)
    hi = pred.predict(p_hi, 0.3)
    lo = pred.predict(p_lo, 0.3)
    assert lo.attainment <= hi.attainment + 1e-9


# ---------------------------------------------------------------------------
# 功率模型:相位混合 + busy 分池口径(可控 Fake 模型,精确断言)
# ---------------------------------------------------------------------------

class _FakePowerOpModel:
    """iter/prefill 恒定;功率相位非对称:prefill 500W、decode 200W、驻留 100W。"""

    def iter_time_ms(self, batch, freq_mhz, ctx_tokens=272.0):
        return 50.0

    def prefill_time_ms(self, tokens, freq_mhz=2520):
        return 200.0  # 0.2s / 请求

    def dyn_j_per_token(self, batch, freq_mhz):
        return 1.0

    def power_w(self, phase, freq_mhz):
        return 500.0 if phase == "prefill" else 200.0

    def residency_w(self):
        return 100.0


@pytest.fixture
def fpred(stats):
    return ApproxPredictor(
        opmodel=_FakePowerOpModel(), stats=stats, slo=SloSpec(),
        freq_candidates=(2520,), gpu_idle_w=35.0, rho_knee=0.7)


def test_mixed_power_blends_prefill_phase(fpred):
    # λ_m=1.0、n_mixed=2:share_p = 1.0×0.2/2 = 0.1
    # T_req = 0.2 + 200×0.05 = 10.2s → busy = 1−e^(−0.5×10.2) ≈ 0.9939
    # p_busy = 0.1×500 + 0.9×200 = 230
    import math
    pr = fpred.predict(P(nm=2), 1.0)
    exp_inst = (1.0 * 0.2 / 2) * 500.0 + (1 - 1.0 * 0.2 / 2) * 200.0
    busy = 1.0 - math.exp(-(1.0 / 2) * 10.2)
    expected = 2 * (busy * exp_inst + (1 - busy) * 100.0) + 4 * 35.0
    assert pr.power_w == pytest.approx(expected, rel=1e-6)


def test_prefill_pool_duty_cycle_power(fpred):
    # 段布局:P 池 duty = λ_pd / μ_pack;D 池 busy = 1−e^(−λ_inst×T_dec)
    import math
    fpred.stats.frac_long = 1.0  # 全部走 PD
    lam = 0.5
    pr = fpred.predict(P(np_=1, nd=1), lam)
    mu_p = fpred.stage_rate_prefill(1)
    busy_p = min(1.0, lam / mu_p)
    busy_d = 1.0 - math.exp(-lam * (200 * 0.05))         # T_dec = 10s
    expected = (busy_p * 500.0 + (1 - busy_p) * 100.0) \
        + (busy_d * 200.0 + (1 - busy_d) * 100.0) + 4 * 35.0
    assert pr.power_w == pytest.approx(expected, rel=1e-6)


def test_busy_saturates_at_high_load(fpred):
    # 占用率随 λ 饱和于 1:高负载区功率趋平(相同布局)
    mu = fpred.rate_mixed(4)
    p_mid = fpred.predict(P(nm=4), mu * 0.7).power_w
    p_high = fpred.predict(P(nm=4), mu * 0.85).power_w
    assert p_high >= p_mid
    assert p_high - p_mid < 0.1 * p_mid


def test_capacity_model_trustworthy_lite_14b():
    from ecopadg.predictor import capacity_model_trustworthy
    assert capacity_model_trustworthy(False, None, False) is False
    assert capacity_model_trustworthy(True, 0.0, True) is False
    assert capacity_model_trustworthy(True, None, True) is False


def test_pe1_scale_skipped_on_downclock(fpred):
    """满频乘 scale;降频用裸解析 μ,避免把 1500 当成满载容量。"""
    hot = ApproxPredictor(
        opmodel=fpred.opmodel, stats=fpred.stats, slo=SloSpec(),
        freq_candidates=(2520, 1500), mixed_scale=4.0, seg_scale=7.0)
    mu_hi = hot.stage_rate_prefill(1, freq=2520)
    mu_lo = hot.stage_rate_prefill(1, freq=1500)
    raw_hi = fpred.stage_rate_prefill(1, freq=2520)
    assert mu_hi == pytest.approx(7.0 * raw_hi, rel=1e-6)
    assert mu_lo < mu_hi / 3.0


def test_calibration_missing_100ms_is_uncalibrated(fpred):
    # 缺 100ms μ_S90:scale=1 并标 uncalibrated,禁止静默用 200ms 档
    from ecopadg.predictor import calibration_scales
    grid = {"32b": {"mixed": {"mu_s90": None, "mu_s90_tpot200": 0.5},
                    "ds_pd": {"mu_s90": None, "mu_s90_tpot200": 0.5}}}
    cres = calibration_scales(fpred, grid, "32b", 4, 2)
    assert cres.mixed_scale == 1.0
    assert cres.seg_scale == 1.0
    assert cres.calibrated is False
    assert "uncalibrated" in cres.note


def test_calibration_rank_spearman_and_scales(fpred):
    from ecopadg.predictor import calibration_scales
    grid = {"32b": {"mixed": {"mu_s90": 2.0, "points": [{"slo": 0.97}]},
                    "ds_pd": {"mu_s90": 1.0, "points": [{"slo": 0.96}]}}}
    cres = calibration_scales(fpred, grid, "32b", 4, 2)
    assert cres.calibrated is True
    assert cres.mixed_scale != 1.0
    assert cres.rank_spearman in (1.0, -1.0, 0.0)


def test_calibration_att_ceiling_from_points(fpred):
    # attainment 天花板取 PE1 实测最大值。绝对 0.9 不再当可行门槛:
    # 低 ρ 只要队列稳定即可行,SLO 非劣在 planner 里对 att_mixed 比较。
    from ecopadg.predictor import calibration_scales
    grid = {"32b": {
        "mixed": {"mu_s90": None, "mu_s90_tpot200": 0.5,
                  "points": [{"rate": 0.5, "slo": 0.895},
                             {"rate": 1.0, "slo": 0.85}]},
        "ds_pd": {"mu_s90": None, "mu_s90_tpot200": 0.5,
                  "points": [{"rate": 0.5, "slo": 0.895}]}}}
    m, s, att_m, att_s = calibration_scales(fpred, grid, "32b", 4, 2)
    assert att_m == pytest.approx(0.895)
    assert att_s == pytest.approx(0.895)
    p2 = ApproxPredictor(
        opmodel=fpred.opmodel, stats=fpred.stats, slo=SloSpec(),
        freq_candidates=(2520,), mixed_scale=m, seg_scale=s,
        att_high_mixed=att_m, att_high_seg=att_s)
    pr = p2.predict(P(nm=2), 0.05)   # 极低负载:ρ 远小于膝点
    assert pr.attainment == pytest.approx(0.895)
    assert pr.attainable


# ---------------------------------------------------------------------------
# LoadMonitor:滑窗到达率 + 长度统计
# ---------------------------------------------------------------------------

def test_load_monitor_rate():
    lm = LoadMonitor(window_s=10.0)
    for i in range(20):
        lm.on_arrival(t=i * 0.5, prompt_len=100, output_len=50)  # 2 req/s
    assert lm.rate(now=10.0) == pytest.approx(2.0, rel=0.15)


def test_load_monitor_stats_and_frac_long():
    lm = LoadMonitor(window_s=100.0, pd_prompt_threshold=512)
    lm.on_arrival(0.0, prompt_len=100, output_len=10)
    lm.on_arrival(1.0, prompt_len=1000, output_len=10)
    st = lm.stats(now=2.0)
    assert st.prompt_len == pytest.approx(550.0)
    assert st.frac_long == pytest.approx(0.5)


def test_load_monitor_window_expiry():
    lm = LoadMonitor(window_s=5.0)
    lm.on_arrival(0.0, 100, 10)
    lm.on_arrival(10.0, 100, 10)
    assert lm.rate(now=10.0) == pytest.approx(1.0 / 5.0, rel=0.2)


def test_load_monitor_rate_fast():
    lm = LoadMonitor(window_s=30.0, fast_window_s=1.0)
    for i in range(10):
        lm.on_arrival(t=9.0 + i * 0.1, prompt_len=100, output_len=10)
    assert lm.rate_fast(now=10.0) == pytest.approx(10.0, rel=0.2)
    assert lm.rate(now=10.0) < lm.rate_fast(now=10.0)


def test_load_monitor_arrival_cv_and_reset():
    lm = LoadMonitor(window_s=30.0)
    for i in range(8):
        lm.on_arrival(t=float(i), prompt_len=100, output_len=10)
    cv = lm.arrival_cv(now=8.0)
    assert cv == pytest.approx(0.0, abs=0.05)
    lm.reset()
    assert lm.rate(now=8.0) == 0.0
    assert lm.arrival_cv(now=8.0) == 0.0
    # 长短间隔交替 → CV > 0
    t = 0.0
    for gap in (0.2, 2.0, 0.2, 2.0, 0.2, 2.0):
        t += gap
        lm.on_arrival(t=t, prompt_len=100, output_len=10)
    assert lm.arrival_cv(now=t) > 0.5


def test_evaluate_capacity_trust_missing_grid(pred, tmp_path):
    missing = tmp_path / "no_rate_grid.json"
    trusted, cres = evaluate_capacity_trust(
        pred, "14b", 8, 2, grid_path=str(missing), freq_trusted=True)
    assert trusted is False
    assert cres is None


def test_evaluate_capacity_trust_calibrated_grid(fpred, tmp_path):
    grid = {
        "14b": {
            "mixed": {"mu_s90": 2.0, "points": [{"slo": 0.97}]},
            "ds_pd": {"mu_s90": 1.0, "points": [{"slo": 0.96}]},
        }
    }
    path = tmp_path / "rate_grid.json"
    path.write_text(__import__("json").dumps(grid), encoding="utf-8")
    trusted, cres = evaluate_capacity_trust(
        fpred, "14b", 8, 2, grid_path=str(path), freq_trusted=True)
    assert cres is not None
    assert cres.calibrated is True
    assert trusted is (cres.rank_spearman is not None
                       and cres.rank_spearman > 0)


def test_default_pe1_grid_path_respects_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.json"
    monkeypatch.setenv("PDBLEND_PE1_GRID", str(custom))
    assert default_pe1_grid_path() == str(custom)
