# -*- coding: utf-8 -*-
"""online_scheduler:PaDG 窗口状态机 + slack 选频。"""
from __future__ import annotations

from ecopadg.online_scheduler import (
    MODE_CONTINUOUS, MODE_DECODE_WINDOW, MODE_PREFILL_WINDOW,
    PaDGWindowScheduler, decode_occupancy_blocks_dvfs,
    decode_slack_blocks_dvfs,
    decode_window_blocks_dvfs, select_decode_freq,
)
from tests.conftest import FakeClock, SynthOpModel


FREQS = (2520, 2100, 1800, 1500, 1200, 1050, 900, 600)


def test_select_decode_freq_prefer_if_feasible(synth_opmodel):
    f = select_decode_freq(synth_opmodel, b_hat=1, ctx_hat=272.0,
                           slo_tpot_s=0.1, freq_candidates=FREQS,
                           margin=0.15, prefer_mhz=1800)
    assert f == 1800


def test_select_decode_freq_prefer_ignored_if_infeasible(synth_opmodel):
    f = select_decode_freq(synth_opmodel, b_hat=16, ctx_hat=272.0,
                           slo_tpot_s=0.095, freq_candidates=FREQS,
                           margin=0.15, prefer_mhz=900)
    assert f == 2520


def test_padg_uses_lookup_decode_freq(synth_opmodel, fake_clock):
    from ecopadg.types import SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       lookup_decode_freq=1800, window_min_s=3.0,
                       window_max_s=10.0)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(20.0)
    d0 = s.step(now=fake_clock())
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    d1 = s.step(now=fake_clock())
    assert d1.mode == MODE_DECODE_WINDOW
    assert d1.freq_mhz == 1800


def test_select_decode_freq_energy_min_feasible(synth_opmodel):
    # U 型能耗:1500 最优;iter 随频率下降变慢。b=1:iter@1500=60ms,
    # SLO 100ms×(1-0.15)=85ms → 1500 可行;900:iter=67.7ms 也可行,
    # 但能耗高于 1500 → 选 1500
    f = select_decode_freq(synth_opmodel, b_hat=1, ctx_hat=272.0,
                           slo_tpot_s=0.1, freq_candidates=FREQS,
                           margin=0.15)
    assert f == 1500


def test_select_decode_freq_margin_respected(synth_opmodel):
    # b=16:iter@2520=72ms、@2100=84ms;SLO 95ms×(1-0.15)=80.75ms
    # → 仅 2520 可行(margin 被遵守)
    f = select_decode_freq(synth_opmodel, b_hat=16, ctx_hat=272.0,
                           slo_tpot_s=0.095, freq_candidates=FREQS,
                           margin=0.15)
    assert f == 2520


def test_decode_floor_blocks_low_freq(synth_opmodel, fake_clock):
    from ecopadg.types import SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       decode_floor_mhz=2520, window_min_s=3.0,
                       window_max_s=10.0)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(20.0)
    d0 = s.step(now=fake_clock())
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    d1 = s.step(now=fake_clock())
    assert d1.freq_mhz == 2520


def test_select_decode_freq_fallback_max(synth_opmodel):
    # SLO 极紧:连 2520 都不满足 → 回退最高频
    f = select_decode_freq(synth_opmodel, b_hat=16, ctx_hat=272.0,
                           slo_tpot_s=0.05, freq_candidates=FREQS,
                           margin=0.15)
    assert f == 2520


def test_select_decode_freq_throughput_floor(synth_opmodel):
    # 吞吐下限:b=2 时 throughput(f)=2/iter(f):2520→44.6、2100→39.0 tok/s。
    # min_tok=40 → 仅 2520 可持续(纯 slack 会选 1500,见首个用例)
    f = select_decode_freq(synth_opmodel, b_hat=2, ctx_hat=272.0,
                           slo_tpot_s=0.2, freq_candidates=FREQS,
                           margin=0.15, min_tok_per_s=40.0)
    assert f == 2520


def _mk_sched(opmodel, clock, window_min_s=3.0, overload_mu=10.0):
    from ecopadg.types import SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       window_min_s=window_min_s, window_max_s=10.0)
    return PaDGWindowScheduler(opmodel=opmodel, config=cfg, clock=clock,
                               mixed_capacity_req_s=overload_mu)


def test_window_cycle_prefill_then_decode(synth_opmodel, fake_clock):
    s = _mk_sched(synth_opmodel, fake_clock)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    s.submit(1, arrival_s=0.1, prompt_len=300, output_len=100)
    # 低到达率(λ̂≈0.1×0.3=0.03):吞吐下限不约束,decode 窗应选能耗最优档
    fake_clock.advance(20.0)
    d0 = s.step(now=fake_clock())
    assert d0.mode == MODE_PREFILL_WINDOW
    assert d0.freq_mhz == 2520
    assert d0.released == [0, 1]  # FIFO 释放全部积压
    # prefill 窗结束 → decode 窗,选低频(1500)
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    d1 = s.step(now=fake_clock())
    assert d1.mode == MODE_DECODE_WINDOW
    assert d1.freq_mhz == 1500


def test_window_decode_freq_floors_on_demand(synth_opmodel, fake_clock):
    # 高到达率:λ̂ 大 → min_tok=λ̂×L̂o/0.6 超过低频吞吐 → decode 窗不降频
    s = _mk_sched(synth_opmodel, fake_clock)
    for i in range(8):
        s.submit(i, arrival_s=i * 0.1, prompt_len=200, output_len=100)
    fake_clock.advance(1.0)
    d0 = s.step(now=fake_clock())         # prefill 窗释放
    assert d0.mode == MODE_PREFILL_WINDOW
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    d1 = s.step(now=fake_clock())
    assert d1.mode == MODE_DECODE_WINDOW
    assert d1.freq_mhz == 2520


def test_overload_degrades_to_continuous(synth_opmodel, fake_clock):
    s = _mk_sched(synth_opmodel, fake_clock, overload_mu=0.5)  # 容量极小
    for i in range(20):
        s.submit(i, arrival_s=float(i) * 0.05, prompt_len=100, output_len=10)
    d = s.step(now=1.0)
    assert d.mode == MODE_CONTINUOUS


def test_continuous_recovers_when_load_drops(synth_opmodel, fake_clock):
    # continuous 不是吸收态:λ̂ 回落后回到 prefill 窗
    s = _mk_sched(synth_opmodel, fake_clock, overload_mu=0.5)
    for i in range(20):
        s.submit(i, arrival_s=float(i) * 0.05, prompt_len=100, output_len=10)
    assert s.step(now=1.0).mode == MODE_CONTINUOUS
    fake_clock.advance(1.0)
    for t in range(2, 40):
        s.step(now=float(t))
        fake_clock.advance(1.0)
    d = s.step(now=fake_clock())
    assert d.mode != MODE_CONTINUOUS


def test_force_continuous_stays_open(synth_opmodel, fake_clock):
    from ecopadg.types import SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       force_continuous=True)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(20.0)
    d = s.step(now=fake_clock())
    assert d.mode == MODE_CONTINUOUS
    assert d.released == [0]


def test_decode_floor_disables_downclock(synth_opmodel, fake_clock):
    from ecopadg.types import SystemConfig
    cfg = SystemConfig(model="72b", freq_candidates=FREQS,
                       decode_floor_mhz=2520)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(20.0)
    d0 = s.step(now=fake_clock())
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    d1 = s.step(now=fake_clock())
    assert d1.mode == MODE_DECODE_WINDOW
    assert d1.freq_mhz == 2520


def test_fifo_release_respects_limit(synth_opmodel, fake_clock):
    s = _mk_sched(synth_opmodel, fake_clock)
    for i in range(6):
        s.submit(i, arrival_s=0.0, prompt_len=100, output_len=10)
    d = s.step(now=0.5, max_release=2)
    assert d.released == [0, 1]


def test_tight_ttft_bypasses_windowing(synth_opmodel, fake_clock):
    """TTFT 预算 <2s 时直通连续混批:窗口化准入延迟(中位 ~0.5s)在紧
    TTFT 下占比过大(alpaca 1s SLO 实测贴线),应立即释放不开窗。"""
    from ecopadg.types import SloSpec, SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       slo=SloSpec(ttft_s=1.0, tpot_s=0.1),
                       window_min_s=3.0, window_max_s=10.0)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(0.1)
    d = s.step(now=fake_clock())
    assert d.mode == MODE_CONTINUOUS      # 不进 prefill 窗
    assert d.released == [0]              # 立即释放


def test_decode_freq_follows_request_slack(synth_opmodel, fake_clock):
    """EcoServe 式 slack 记账:请求领先节奏 → 允许更低频;落后 → 顶满频。

    SLO_TPOT=0.09:静态 margin 0.35 预算 58.5ms 只允许 ≥2100
    (iter@1500=64.6ms);slack 充足时预算放大 → 选能耗最优 1500。
    """
    from ecopadg.types import SloSpec, SystemConfig
    cfg = SystemConfig(model="syn", freq_candidates=FREQS,
                       slo=SloSpec(ttft_s=5.0, tpot_s=0.09),
                       window_min_s=3.0, window_max_s=10.0)
    s = PaDGWindowScheduler(opmodel=synth_opmodel, config=cfg,
                            clock=fake_clock, mixed_capacity_req_s=10.0)
    s.submit(0, arrival_s=0.0, prompt_len=200, output_len=100)
    fake_clock.advance(20.0)
    d0 = s.step(now=fake_clock())            # prefill 窗释放 rid0
    assert d0.released == [0]
    fake_clock.advance(d0.window_end_s - fake_clock() + 0.001)
    # 领先:0.2s 内喂 10 个 token(授权 10×0.09=0.9s)→ slack≈+0.7s
    t0 = fake_clock()
    for k in range(10):
        s.on_token(0, t0 + k * 0.02)
    fake_clock.advance(0.2)
    d1 = s.step(now=fake_clock())
    assert d1.mode == MODE_DECODE_WINDOW
    assert d1.freq_mhz == 1500
    # 落后:2s 不产 token → slack 转负 → 回退最高频
    fake_clock.advance(2.0)
    d2 = s.step(now=fake_clock())
    assert d2.freq_mhz == 2520


def test_decode_occupancy_blocks_dvfs_uses_batch_cap():
    assert decode_occupancy_blocks_dvfs(None, 2) is False
    assert decode_occupancy_blocks_dvfs(20, 2) is False
    assert decode_occupancy_blocks_dvfs(48, 2) is True
    assert decode_occupancy_blocks_dvfs(47, 2) is False


def test_decode_slack_blocks_dvfs_immediate():
    assert decode_slack_blocks_dvfs(None, 0.1) is False
    assert decode_slack_blocks_dvfs(-0.01, 0.1) is True
    assert decode_slack_blocks_dvfs(0.01, 0.1) is True
    assert decode_slack_blocks_dvfs(0.2, 0.1) is False


