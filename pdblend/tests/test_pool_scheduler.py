# -*- coding: utf-8 -*-
"""pool_scheduler:P 池 FCFS / D 池 slack 选频 / selective 路由(测试先行,M3)。

批处理统一约定:
  - prefill 池:FCFS + token 预算派发(实例内 vLLM 按 token 批执行);
  - decode 池:continuous batching(vLLM 原生)+ 逐实例 slack 选频 DVFS;
  - mixed 池:sarathi 式混批(vLLM chunked prefill)+ PaDG 窗口(复用现有)。
"""
from __future__ import annotations

import pytest

from ecopadg.pool_scheduler import (
    DecodePoolScheduler, PrefillPoolScheduler, SelectiveRouter,
    normalize_pd_sched_policy, pool_saturated,
)


class _Clock:
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
# PrefillPoolScheduler:FCFS + token 预算 + least-loaded
# ---------------------------------------------------------------------------

@pytest.fixture
def psched(synth_opmodel):
    return PrefillPoolScheduler(opmodel=synth_opmodel, prefill_freq=2520,
                                idle_freq=600, token_budget=4096)


def test_prefill_fcfs_order(psched):
    psched.submit(1, prompt_len=100)
    psched.submit(2, prompt_len=100)
    psched.submit(3, prompt_len=100)
    out = psched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in out] == [1, 2, 3]
    assert all(name == "p0" for _, name in out)


def test_prefill_token_budget_holds_tail(psched):
    psched.submit(1, prompt_len=3000)
    psched.submit(2, prompt_len=3000)
    out = psched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    # 3000+3000 > 4096:第二条留队(FCFS 不越序)
    assert [rid for rid, _ in out] == [1]
    assert psched.queued() == 1
    # 完成后可继续派发
    psched.on_dispatched_done(1, "p0")
    out = psched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in out] == [2]


def test_prefill_least_loaded_routing(psched):
    for rid in (1, 2):
        psched.submit(rid, prompt_len=100)
    out = psched.dispatch(instances=["p0", "p1"],
                          inflight_tokens={"p0": 2000, "p1": 0})
    # 先派最空的 p1;派发计入其 inflight 后二者均衡
    assert out[0][1] == "p1"
    assert len(out) == 2


def test_prefill_freq_busy_vs_idle(psched):
    assert psched.pool_freq(busy=True) == 2520
    assert psched.pool_freq(busy=False) == 600


def test_prefill_single_oversize_request_dispatches(psched):
    # 单条超预算请求不可饿死:独占预算派发
    psched.submit(1, prompt_len=9999)
    out = psched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in out] == [1]


def test_prefill_pack_skips_infeasible_head(synth_opmodel):
    clock = _Clock(0.0)
    sched = PrefillPoolScheduler(
        opmodel=synth_opmodel, token_budget=4096, policy="pack",
        pack_starve_s=0.2, clock=clock)
    sched.submit(1, prompt_len=3000)
    sched.submit(2, prompt_len=500)
    sched.submit(3, prompt_len=500)
    out = sched.dispatch(instances=["p0"], inflight_tokens={"p0": 1500})
    assert [rid for rid, _ in out] == [2, 3]
    assert sched.queued() == 1
    assert sched.queued_prompt_lens() == [3000]
    assert sched.stats.hol_skips == 2


def test_prefill_pack_starvation_forces_head(synth_opmodel):
    clock = _Clock(0.0)
    sched = PrefillPoolScheduler(
        opmodel=synth_opmodel, token_budget=4096, policy="pack",
        pack_starve_s=0.2, clock=clock)
    sched.submit(1, prompt_len=3000)
    sched.submit(2, prompt_len=500)
    sched.dispatch(instances=["p0"], inflight_tokens={"p0": 1500})
    assert sched.queued() == 1
    clock.t = 0.21
    out = sched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in out] == [1]


def test_prefill_sjf_shortest_first(synth_opmodel):
    clock = _Clock(0.0)
    sched = PrefillPoolScheduler(
        opmodel=synth_opmodel, token_budget=2500, policy="sjf",
        sjf_starve_s=0.5, clock=clock)
    sched.submit(1, prompt_len=3000)
    sched.submit(2, prompt_len=100)
    sched.submit(3, prompt_len=200)
    out = sched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in out] == [2, 3]
    assert sched.queued_prompt_lens() == [3000]


def test_prefill_sjf_starvation_promotes_long(synth_opmodel):
    clock = _Clock(0.0)
    sched = PrefillPoolScheduler(
        opmodel=synth_opmodel, token_budget=2500, policy="sjf",
        sjf_starve_s=0.5, clock=clock)
    sched.submit(1, prompt_len=3000)
    sched.submit(2, prompt_len=100)
    first = sched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert [rid for rid, _ in first] == [2]
    assert sched.queued() == 1
    sched.on_dispatched_done(2, "p0")
    clock.t = 0.51
    sched.submit(3, prompt_len=200)
    out = sched.dispatch(instances=["p0"], inflight_tokens={"p0": 0})
    assert out[0][0] == 1


def test_unknown_policy_rejected(synth_opmodel):
    with pytest.raises(ValueError):
        normalize_pd_sched_policy("srpt")
    with pytest.raises(ValueError):
        PrefillPoolScheduler(opmodel=synth_opmodel, policy="srpt")


def test_prefill_busy_tracks_prefill_phase_only(psched):
    # busy 仅覆盖 prefill 阶段(排队或派发未完成),decode 流阶段不算:
    # 否则段布局下 P 池被整请求生命周期锁在高频,永不空闲降频
    assert not psched.busy()
    psched.submit(1, prompt_len=256)
    assert psched.busy()                     # 排队中
    psched.dispatch(instances=["p0"])
    assert psched.busy()                     # prefill 在途
    psched.on_dispatched_done(1, "p0", 256)
    assert not psched.busy()                 # prefill 完成 → 空闲(即使请求仍在 decode)


# ---------------------------------------------------------------------------
# DecodePoolScheduler:桥接准入 + 逐实例 slack 选频
# ---------------------------------------------------------------------------

@pytest.fixture
def dsched(synth_opmodel):
    return DecodePoolScheduler(
        opmodel=synth_opmodel, slo_tpot_s=0.1, tpot_margin=0.15,
        freq_candidates=(2520, 1800, 1500, 1200, 900, 600),
        idle_freq=600, max_blocks=100, waiting_threshold=0.5, block_size=16)


def test_decode_hold_waits_for_batch_or_timeout(synth_opmodel):
    clock = _Clock(100.0)
    sched = DecodePoolScheduler(
        opmodel=synth_opmodel, slo_tpot_s=0.1, tpot_margin=0.15,
        freq_candidates=(2520, 1800, 1500, 1200, 900, 600),
        idle_freq=600, max_blocks=100, waiting_threshold=0.5, block_size=16,
        clock=clock, policy="hold", hold_ms=50, hold_b=4)
    sched.on_prefill_done(1, prompt_len=160, output_len=100)
    assert sched.try_admit(instances=["d0"]) == []
    for rid in (2, 3, 4):
        sched.on_prefill_done(rid, prompt_len=160, output_len=100)
    admitted = sched.try_admit(instances=["d0"])
    assert [rid for rid, _ in admitted] == [1, 2, 3, 4]

    clock2 = _Clock(200.0)
    lone = DecodePoolScheduler(
        opmodel=synth_opmodel, slo_tpot_s=0.1, tpot_margin=0.15,
        freq_candidates=(2520, 1800, 1500),
        idle_freq=600, max_blocks=100, waiting_threshold=0.5, block_size=16,
        clock=clock2, policy="hold", hold_ms=50, hold_b=4)
    lone.on_prefill_done(9, prompt_len=160, output_len=100)
    assert lone.try_admit(instances=["d0"]) == []
    clock2.t = 200.05
    assert [rid for rid, _ in lone.try_admit(instances=["d0"])] == [9]


def test_decode_admission_and_freq(dsched):
    dsched.on_prefill_done(1, prompt_len=160, output_len=100)
    admitted = dsched.try_admit(instances=["d0"])
    assert admitted == [(1, "d0")]
    freqs = dsched.pool_freqs(instances=["d0"])
    # Synth:可行域内能量最优 1500
    assert freqs["d0"] == 1500


def test_decode_admission_gate_blocks(dsched):
    # 塞满 waiting blocks(threshold 0.5 × 100 blocks = 50 块上限)
    dsched.on_prefill_done(1, prompt_len=16 * 50, output_len=10)   # 50 块
    dsched.on_prefill_done(2, prompt_len=16 * 10, output_len=10)   # 10 块
    admitted = dsched.try_admit(instances=["d0"])
    # 第 1 条进入后 waiting=50,不再 < 50,第 2 条被挡(FCFS 停排)
    assert [r for r, _ in admitted] == [1]
    # 完成释放后第 2 条可进
    dsched.complete(1, "d0")
    admitted = dsched.try_admit(instances=["d0"])
    assert [r for r, _ in admitted] == [2]


def test_decode_least_loaded_across_instances(dsched):
    for rid in (1, 2, 3, 4):
        dsched.on_prefill_done(rid, prompt_len=64, output_len=64)
    admitted = dsched.try_admit(instances=["d0", "d1"])
    assert len(admitted) == 4
    by_inst = {}
    for rid, name in admitted:
        by_inst.setdefault(name, []).append(rid)
    assert len(by_inst["d0"]) == 2 and len(by_inst["d1"]) == 2


def test_decode_idle_freq(dsched):
    freqs = dsched.pool_freqs(instances=["d0"])
    assert freqs["d0"] == 600  # 空闲降至最低档


def test_decode_cancel_removes_bridge_ghost(dsched):
    # 准入超时的请求必须从桥队列移除,否则泵稍后"准入"幽灵进活跃集,
    # 无人 complete → 活跃集永久泄漏 → 门锁死(r3.4 复测级联雪崩根因)
    dsched.on_prefill_done(1, prompt_len=160, output_len=100)
    dsched.cancel(1, "d0")                    # 超时路径:未准入即取消
    assert dsched.try_admit(instances=["d0"]) == []   # 桥里没有幽灵
    assert dsched.bridge_depth() == 0
    # 已准入后取消:活跃集也应清空
    dsched.on_prefill_done(2, prompt_len=160, output_len=100)
    dsched.try_admit(instances=["d0"])
    dsched.cancel(2, "d0")
    assert dsched._b_hat("d0") == 0


def test_decode_freq_follows_slack(dsched):
    # slack 记账:领先节奏 → 预算放大 → 能量最优档;落后 → 顶满频
    dsched.on_prefill_done(1, prompt_len=160, output_len=100)
    dsched.try_admit(instances=["d0"])
    for k in range(10):                      # 0.2s 产 10 token(授权 1.0s)
        dsched.on_token(1, "d0", 100.0 + 0.02 * k)
    assert dsched.pool_freqs(["d0"], now=100.2)["d0"] == 1500
    # 1.3s 无产出:slack = 100+1.0−101.5 = −0.5 → 无档可行 → 回退 2520
    assert dsched.pool_freqs(["d0"], now=101.5)["d0"] == 2520


def test_decode_freq_honors_lookup_prefer(dsched):
    dsched.lookup_decode_freq = 1800
    dsched.on_prefill_done(1, prompt_len=160, output_len=100)
    dsched.try_admit(instances=["d0"])
    for k in range(10):
        dsched.on_token(1, "d0", 100.0 + 0.02 * k)
    assert dsched.pool_freqs(["d0"], now=100.2)["d0"] == 1800


def test_decode_freq_tracks_batch(dsched, synth_opmodel):
    # 大 batch → iter 变慢 → 可行域缩小:准入门控把 batch 截到 13
    # (50 块阈值 / 4 块每请求),iter(13,f)<=85ms 仅 {1800,2520} 可行,
    # 能量 U 型 → 取 1800(而非小 batch 时的 1500)
    for rid in range(24):
        dsched.on_prefill_done(rid, prompt_len=64, output_len=64)
    admitted = dsched.try_admit(instances=["d0"])
    b = len(admitted)
    f = dsched.pool_freqs(instances=["d0"])["d0"]
    assert synth_opmodel.iter_time_ms(b, f) <= 100.0 * 0.85
    assert f == 1800


# ---------------------------------------------------------------------------
# SelectiveRouter:长 prompt 走 P→D,短走 mixed;单池退化;过载外溢
# ---------------------------------------------------------------------------

def test_router_only_mixed(dsched):
    r = SelectiveRouter(pd_prompt_threshold=512)
    assert r.route(prompt_len=2000, has_mixed=True, has_pd=False,
                   mixed_load=0.0, pd_load=0.0) == "mixed"


def test_router_only_pd():
    r = SelectiveRouter(pd_prompt_threshold=512)
    assert r.route(prompt_len=10, has_mixed=False, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "pd"


def test_router_selective_by_prompt_len():
    r = SelectiveRouter(pd_prompt_threshold=512)
    assert r.route(prompt_len=1024, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "pd"
    assert r.route(prompt_len=128, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "mixed"


def test_router_spills_on_overload():
    r = SelectiveRouter(pd_prompt_threshold=512, load_spill=0.9)
    # 短请求本该走 mixed,但 mixed 过载且 pd 空闲 → 外溢
    assert r.route(prompt_len=128, has_mixed=True, has_pd=True,
                   mixed_load=0.95, pd_load=0.1) == "pd"
    # 双双过载 → 保持首选(不来回倒)
    assert r.route(prompt_len=128, has_mixed=True, has_pd=True,
                   mixed_load=0.95, pd_load=0.95) == "mixed"


def test_router_fallback_threshold_is_caller_mean_pin():
    r = SelectiveRouter(pd_prompt_threshold=720)
    assert r.route(prompt_len=719, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "mixed"
    assert r.route(prompt_len=720, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "pd"


def test_router_both_infeasible_flags_no_further_dvfs(synth_opmodel):
    r = SelectiveRouter(pd_prompt_threshold=512, opmodel=synth_opmodel,
                        slo_ttft_s=5.0)
    path = r.route(prompt_len=128, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0,
                   mixed_pending_s=10.0, pd_pending_s=10.0)
    assert path == "mixed"
    assert r.last_both_infeasible is True


def test_router_no_pool_raises():
    r = SelectiveRouter()
    with pytest.raises(RuntimeError):
        r.route(prompt_len=1, has_mixed=False, has_pd=False,
                mixed_load=0.0, pd_load=0.0)


def test_router_cost_based_prefers_cheaper_path(synth_opmodel):
    r = SelectiveRouter(pd_prompt_threshold=512, opmodel=synth_opmodel,
                        cost_based=True, kv_xfer_s_per_tok=1.0)
    # 巨大 KV 传输代价 → 即使 prompt 很长也走 mixed
    assert r.route(prompt_len=2000, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0) == "mixed"


def test_router_slo_constraint_drops_infeasible_pd(synth_opmodel):
    r = SelectiveRouter(pd_prompt_threshold=512, opmodel=synth_opmodel,
                        cost_based=True, slo_ttft_s=5.0)
    assert r.route(prompt_len=128, has_mixed=True, has_pd=True,
                   mixed_load=0.0, pd_load=0.0,
                   mixed_pending_s=0.0, pd_pending_s=10.0) == "mixed"


def test_pool_saturated_thresholds():
    assert not pool_saturated(mean_inflight=8.0, queued=0, n_active=1, b_ref=8)
    assert pool_saturated(mean_inflight=24.0, queued=0, n_active=1, b_ref=8)
    assert pool_saturated(mean_inflight=0.0, queued=4, n_active=1, b_ref=8)
