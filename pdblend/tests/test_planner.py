# -*- coding: utf-8 -*-
"""planner:同负载 min J s.t. SLO 非劣。"""
from __future__ import annotations

from ecopadg.global_scheduler import Prediction
from ecopadg.planner import (
    default_mixed_tp,
    choose_boot_slo_layout, choose_iso_load, choose_m0_tuned,
    enumerate_partitions,
)
from ecopadg.types import Partition


class _Pred:
    def __call__(self, p, lam, state=None):
        # 全 mixed 贵;带 PD 的便宜但 att 看 n_prefill
        att = 0.96 if p.n_mixed >= 1 else 0.70
        if p.n_mixed >= 2:
            att = 0.97
        e = 1000.0 if p.pairs() == 0 else 700.0
        if p.n_mixed == 0 and p.pairs() >= 1:
            e = 700.0
        return Prediction(attainable=True, attainment=att,
                          power_w=e, energy_j_per_s=e)


def test_enumerate_has_mixed_and_disagg():
    space = enumerate_partitions(8, tps=(2,), pps=(1,), freqs=(2520,),
                                 include_pp=False)
    assert any(p.n_mixed > 0 and p.pairs() == 0 for p in space)
    assert any(p.pairs() > 0 for p in space)


def test_choose_rejects_slo_worse_than_mixed():
    space = [
        Partition(n_mixed=4, tp_mixed=2, gpus_total=8),
        Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                  gpus_total=8),
    ]
    r = choose_iso_load(_Pred(), lam=1.2, att_mixed=0.97, space=space,
                        gpu_count=8, tp_fallback=2)
    assert r.chosen is not None
    assert r.chosen.n_mixed == 4  # PD att 0.70 被滤掉;2×mixed 是缩容另测
    assert r.fallback == ""


def test_choose_min_j_when_slo_ok():
    class PredOk:
        def __call__(self, p, lam, state=None):
            e = 900.0 if p.n_mixed else 600.0
            return Prediction(attainable=True, attainment=0.96,
                              power_w=e, energy_j_per_s=e)

    space = [
        Partition(n_mixed=2, tp_mixed=2, gpus_total=8),
        Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                  gpus_total=8),
    ]
    r = choose_iso_load(PredOk(), lam=1.2, att_mixed=0.95, space=space,
                        gpu_count=8)
    assert r.chosen.pairs() == 2


def test_empty_feasible_falls_back_mixed():
    class PredBad:
        def __call__(self, p, lam, state=None):
            return Prediction(attainable=True, attainment=0.1,
                              power_w=1.0, energy_j_per_s=1.0)

    r = choose_iso_load(PredBad(), lam=1.0, att_mixed=0.99, space=[
        Partition(n_prefill=1, n_decode=1, tp_prefill=2, tp_decode=2,
                  gpus_total=8)], gpu_count=8, tp_fallback=2)
    assert r.fallback == "mixed@2520"
    assert r.chosen.n_mixed >= 1
    assert r.chosen.freq_mixed == 2520


def test_m0_tuned_stays_mixed():
    class Pred:
        def __call__(self, p, lam, state=None):
            e = 200.0 + p.freq_mixed * 0.2
            att = 0.96 if p.freq_mixed >= 1500 else 0.50
            return Prediction(attainable=True, attainment=att,
                              power_w=e, energy_j_per_s=e)

    r = choose_m0_tuned(Pred(), lam=1.0, att_mixed=0.95,
                        gpu_count=8, tp=2, freqs=(2520, 1500, 900))
    assert r.mode == "m0-tuned"
    assert r.chosen.n_mixed >= 1
    assert r.chosen.pairs() == 0
    assert r.chosen.freq_mixed == 1500


def test_untrusted_keeps_full_gpu_and_2520():
    class Pred:
        def __call__(self, p, lam, state=None):
            if p.n_mixed == 2:
                e = 100.0
            elif p.pairs() >= 2:
                e = 600.0
            else:
                e = 800.0
            if int(getattr(p, "freq_mixed", 2520) or 2520) < 2500:
                e = min(e, 80.0)
            return Prediction(attainable=True, attainment=0.98,
                              power_w=e, energy_j_per_s=e)

    space = [
        Partition(n_mixed=2, tp_mixed=2, freq_mixed=900, gpus_total=8),
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=900, gpus_total=8),
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
        Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                  freq_prefill=2520, freq_decode=2520, gpus_total=8),
    ]
    r = choose_iso_load(Pred(), lam=2.0, att_mixed=0.98, space=space,
                        gpu_count=8, trusted=False)
    assert r.chosen is not None
    assert r.chosen.total_gpus() == 8
    assert r.chosen.n_mixed == 4
    assert r.chosen.pairs() == 0
    assert r.fallback == "boot-mixed"


def test_trusted_still_allows_partial():
    class Pred:
        def __call__(self, p, lam, state=None):
            e = 100.0 if p.n_mixed == 2 else 800.0
            return Prediction(attainable=True, attainment=0.98,
                              power_w=e, energy_j_per_s=e)

    space = [
        Partition(n_mixed=2, tp_mixed=2, freq_mixed=900, gpus_total=8),
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
    ]
    r = choose_iso_load(Pred(), lam=2.0, att_mixed=0.98, space=space,
                        gpu_count=8, trusted=True)
    # 2×mixed 是缩容且 att 无 +3pp / 无 ρ → 回满卡
    assert r.chosen.n_mixed == 4
    assert any(s.reason == "shrink-margin" for s in r.scores)


def test_pin_prefill_drops_1500():
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import PREFILL_PIN_MHZ

    class Pred:
        def __call__(self, p, lam, state=None):
            e = 100.0 if int(p.freq_prefill) < 2000 else 800.0
            return Prediction(attainable=True, attainment=0.98,
                              power_w=e, energy_j_per_s=e, rho=0.2)

    cheap_p = Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                        freq_prefill=1500, freq_decode=2520, gpus_total=8)
    full = Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                     freq_prefill=2520, freq_decode=2520, gpus_total=8)
    r = choose_iso_load(Pred(), lam=2.0, att_mixed=0.95, space=[cheap_p, full],
                        gpu_count=8, trusted=True, pin_prefill_mhz=PREFILL_PIN_MHZ)
    assert r.chosen.freq_prefill >= 2520


def test_shrink_allowed_with_margin_and_low_rho():
    class Pred:
        def __call__(self, p, lam, state=None):
            if p.n_mixed == 2:
                return Prediction(attainable=True, attainment=0.99,
                                  power_w=100.0, energy_j_per_s=100.0,
                                  rho=0.2)
            return Prediction(attainable=True, attainment=0.95,
                              power_w=800.0, energy_j_per_s=800.0, rho=0.5)

    space = [
        Partition(n_mixed=2, tp_mixed=2, freq_mixed=1800, gpus_total=8),
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
    ]
    r = choose_iso_load(Pred(), lam=1.2, att_mixed=0.95, space=space,
                        gpu_count=8, trusted=True)
    assert r.chosen.n_mixed == 2


def test_hybrid_selective_partition_8gpu_tp2():
    from ecopadg.planner import hybrid_selective_partition
    p = hybrid_selective_partition(8, 2)
    assert p is not None
    assert p.n_mixed == 2 and p.pairs() == 1
    assert p.total_gpus() == 8
    assert hybrid_selective_partition(8, 4) is None


class _CapPred(_Pred):
    """μ_P=μ_D=20 → α=0.50 时 r<10 才允许 PD。"""

    def stage_rate_prefill(self, n, freq=None):
        del freq
        return 20.0 if int(n) > 0 else 0.0

    def stage_rate_decode(self, n, freq=None):
        del freq
        return 20.0 if int(n) > 0 else 0.0

    def rate_mixed(self, n, freq=None, prefill_freq=None):
        del freq, prefill_freq
        return 15.0 if int(n) > 0 else 0.0


class _SharePred(_CapPred):
    """ShareGPT 切点:μ_P=15 → r=6 PD, r=8 mixed。"""

    def stage_rate_prefill(self, n, freq=None):
        del freq
        return 15.0 if int(n) > 0 else 0.0

    def stage_rate_decode(self, n, freq=None):
        del freq
        return 40.0 if int(n) > 0 else 0.0


class _AlpacaPred(_CapPred):
    """短 prompt decode bound:μ_P=40, μ_D=28 → r=12 PD, r=16 mixed。"""

    def stage_rate_prefill(self, n, freq=None):
        del freq
        return 40.0 if int(n) > 0 else 0.0

    def stage_rate_decode(self, n, freq=None):
        del freq
        return 28.0 if int(n) > 0 else 0.0


class _LongPred(_CapPred):
    """长 prefill:μ_P=5 → r=2 PD, r=3 mixed。"""

    def stage_rate_prefill(self, n, freq=None):
        del freq
        return 5.0 if int(n) > 0 else 0.0

    def stage_rate_decode(self, n, freq=None):
        del freq
        return 20.0 if int(n) > 0 else 0.0


def test_boot_slo_untrusted_high_rho_forbids_spatial_pd():
    r = choose_boot_slo_layout(_CapPred(), rate=12.0, att_mixed=0.97,
                               trusted=False)
    assert r.chosen is not None
    assert r.chosen.n_mixed > 0
    assert r.chosen.pairs() == 0
    spatial = next(s for s in r.scores
                   if s.partition.n_mixed == 0 and s.partition.pairs() > 0)
    assert spatial.accepted is False
    assert spatial.reason == "capacity-rho-guard-mixed"
    assert r.guard is not None
    assert r.guard.pd_ok is False


def test_boot_slo_untrusted_low_rate_fullnode_spatial_pd():
    r = choose_boot_slo_layout(_CapPred(), rate=2.0, att_mixed=0.97,
                               trusted=False, frac_long=0.0)
    assert r.chosen is not None
    assert r.chosen.pairs() > 0
    assert r.chosen.n_mixed == 0
    assert r.used_gpus == 8
    assert r.unused_gpus == 0
    assert r.freq0 == 900
    assert r.fallback == "untrusted-min-j-spatial-pd"


def test_boot_slo_untrusted_long_prompt_mid_rate_mixed():
    class LongPred(_CapPred):
        def stage_rate_prefill(self, n, freq=None):
            del freq
            return 5.0 if int(n) > 0 else 0.0

    r = choose_boot_slo_layout(LongPred(), rate=4.0, att_mixed=0.90,
                               trusted=False, frac_long=0.0)
    assert r.chosen.n_mixed > 0
    assert r.chosen.pairs() == 0
    spatial = [s for s in r.scores
               if s.partition.n_mixed == 0 and s.partition.pairs() > 0]
    assert spatial and all(s.accepted is False for s in spatial)


def test_boot_slo_untrusted_unknown_mu_fails_closed_mixed():
    r = choose_boot_slo_layout(_Pred(), rate=2.0, att_mixed=0.97,
                               trusted=False)
    assert r.chosen is not None
    assert r.chosen.n_mixed > 0
    assert r.unused_gpus == 0
    spatial = [s for s in r.scores
               if s.partition.n_mixed == 0 and s.partition.pairs() > 0]
    assert spatial and all(
        s.reason == "capacity-mu-unknown-mixed" for s in spatial)


def test_boot_slo_untrusted_unknown_mu_d_fails_closed_mixed():
    class PrefillOnly(_CapPred):
        def stage_rate_decode(self, n, freq=None):
            del n, freq
            return None

    r = choose_boot_slo_layout(PrefillOnly(), rate=2.0, att_mixed=0.97,
                               trusted=False, frac_long=0.0)
    assert r.chosen.pairs() == 0
    spatial = next(s for s in r.scores
                   if s.partition.n_mixed == 0 and s.partition.pairs() > 0)
    assert spatial.accepted is False
    assert spatial.reason == "capacity-mu-d-unknown-mixed"


def test_boot_slo_sharegpt_cut_stays_pd_at_6_mixed_at_8():
    low = choose_boot_slo_layout(_SharePred(), rate=6.0, att_mixed=0.97,
                                 trusted=False, frac_long=0.0)
    assert low.fallback == "untrusted-min-j-spatial-pd"
    high = choose_boot_slo_layout(_SharePred(), rate=8.0, att_mixed=0.97,
                                  trusted=False, frac_long=0.0)
    assert high.chosen.pairs() == 0
    assert high.guard.reason == "capacity-rho-guard-mixed"


def test_boot_slo_alpaca_cut_pd_at_12_mixed_at_16():
    low = choose_boot_slo_layout(_AlpacaPred(), rate=12.0, att_mixed=0.99,
                                 trusted=False, frac_long=0.0)
    assert low.fallback == "untrusted-min-j-spatial-pd"
    high = choose_boot_slo_layout(_AlpacaPred(), rate=16.0, att_mixed=0.99,
                                  trusted=False, frac_long=0.0)
    assert high.chosen.pairs() == 0
    assert high.guard.reason == "capacity-decode-rho-guard-mixed"


def test_boot_slo_longbench_cut_pd_at_2_mixed_at_3():
    low = choose_boot_slo_layout(_LongPred(), rate=2.0, att_mixed=0.90,
                                 trusted=False, frac_long=1.0)
    assert low.fallback == "untrusted-min-j-spatial-pd"
    high = choose_boot_slo_layout(_LongPred(), rate=3.0, att_mixed=0.90,
                                  trusted=False, frac_long=1.0)
    assert high.chosen.pairs() == 0
    assert high.guard.reason == "capacity-rho-guard-mixed"


def test_boot_slo_trusted_att_then_min_j():
    r = choose_boot_slo_layout(_Pred(), rate=12.0, att_mixed=0.97,
                               trusted=True)
    # mixed att 0.97 J=1000; PD att 0.70 丢掉; A9 att 0.97 J=700 → A9
    assert r.chosen is not None
    assert r.chosen.n_mixed == 2
    assert r.chosen.pairs() == 1
    assert r.fallback == "trusted-min-j-s.t.-att-0.90"


def test_boot_slo_rejects_exploded_energy():
    class PredBoom:
        def __call__(self, p, lam, state=None):
            att = 0.96 if p.n_mixed else 0.0
            e = 1e13 if p.pairs() else 900.0
            return Prediction(attainable=True, attainment=att,
                              power_w=1300.0, energy_j_per_s=e)

    r = choose_boot_slo_layout(PredBoom(), rate=12.0, att_mixed=0.97,
                               trusted=False)
    assert r.chosen.n_mixed > 0
    assert r.chosen.pairs() == 0
    hybrid = next(s for s in r.scores
                  if s.partition.n_mixed > 0 and s.partition.pairs() > 0)
    assert hybrid.accepted is False
    assert hybrid.reason == "hybrid-frac-long-out-of-band"


def test_default_pd_threshold_is_mean_pin():
    from ecopadg.planner import default_pd_threshold
    assert default_pd_threshold(720.4) == 720
    assert default_pd_threshold(17.2) == 17
    assert default_pd_threshold(0) == 1


def test_mix_frac_long_uses_cross_trace_token_key():
    from ecopadg.planner import mix_frac_long
    assert mix_frac_long([17, 16, 20]) == 0.0
    assert mix_frac_long([1900, 2000]) == 1.0
    assert mix_frac_long([200, 900]) == 0.5


def test_boot_slo_untrusted_a9_is_candidate_only_on_full_node():
    from ecopadg.planner import layout_kind
    r = choose_boot_slo_layout(_CapPred(), rate=6.0, att_mixed=0.97,
                               trusted=False, frac_long=0.4)
    assert r.hybrid_guard is not None
    assert r.hybrid_guard.ok is True
    a9 = [s for s in r.scores
          if layout_kind(s.partition) == "hybrid-a9"]
    assert a9 and all(s.partition.total_gpus() == 8 for s in a9)
    assert r.unused_gpus == 0
    assert r.used_gpus == 8
    # I1 主语是满卡空间 PD;A9 只在 PD 被否时才上。
    assert layout_kind(r.chosen) == "spatial-pd"


def test_boot_slo_untrusted_a9_only_when_pd_rejected():
    from ecopadg.planner import layout_kind
    r = choose_boot_slo_layout(_CapPred(), rate=10.0, att_mixed=0.97,
                               trusted=False, frac_long=0.4)
    assert r.guard is not None and r.guard.pd_ok is False
    assert r.hybrid_guard is not None and r.hybrid_guard.ok is True
    assert layout_kind(r.chosen) == "hybrid-a9"
    assert r.fallback == "untrusted-min-j-hybrid-a9"


def test_boot_slo_untrusted_a9_off_when_all_short():
    from ecopadg.planner import layout_kind
    r = choose_boot_slo_layout(_CapPred(), rate=6.0, att_mixed=0.97,
                               trusted=False, frac_long=0.0)
    hybrid = next(s for s in r.scores
                  if s.partition.n_mixed > 0 and s.partition.pairs() > 0)
    assert hybrid.accepted is False
    assert hybrid.reason == "hybrid-frac-long-out-of-band"
    assert layout_kind(r.chosen) != "hybrid-a9"


def test_boot_slo_untrusted_a9_off_when_split_overloads_mixed():
    r = choose_boot_slo_layout(_CapPred(), rate=20.0, att_mixed=0.97,
                               trusted=False, frac_long=0.4)
    # λ_m=12 ≥ 0.5*15 → A9 否; packing 被拒才回 1×TP8
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.chosen.pairs() == 0
    assert r.hybrid_guard is not None
    assert r.hybrid_guard.ok is False
    assert r.hybrid_guard.reason == "hybrid-mixed-rho-guard"
    assert r.fallback in (
        "capacity-rho-guard-mixed", "boot-mixed-fallback")


def test_default_mixed_tp_is_tp8_for_all_models():
    assert default_mixed_tp("7b") == 8
    assert default_mixed_tp("14b") == 8
    assert default_mixed_tp("32b") == 8
    assert default_mixed_tp("72b") == 8


def test_boot_slo_mixed_tp8_when_pd_rejected():
    r = choose_boot_slo_layout(
        _CapPred(), rate=20.0, att_mixed=0.97, trusted=False,
        frac_long=0.4, mixed_tp=8)
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.chosen.pairs() == 0


def test_boot_slo_a9_stays_tp2_even_with_mixed_tp8():
    from ecopadg.planner import layout_kind
    r = choose_boot_slo_layout(
        _CapPred(), rate=6.0, att_mixed=0.97, trusted=False,
        frac_long=0.4, mixed_tp=8)
    a9 = next(s for s in r.scores
              if layout_kind(s.partition) == "hybrid-a9")
    assert a9.partition.tp_mixed == 2
    assert a9.partition.tp_prefill == 2
    assert a9.partition.pairs() == 1
    assert a9.partition.total_gpus() == 8


def test_boot_slo_untrusted_min_j_skips_costly_pd():
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import layout_kind

    class CostlyPD(_CapPred):
        def __call__(self, p, lam, state=None):
            e = 2000.0 if p.pairs() else 800.0
            return Prediction(attainable=True, attainment=0.97,
                              power_w=e, energy_j_per_s=e)

    r = choose_boot_slo_layout(
        CostlyPD(), rate=2.0, att_mixed=0.97, trusted=False, frac_long=0.0)
    assert layout_kind(r.chosen) == "mixed"
    assert r.fallback == "untrusted-min-j-mixed"
    assert r.freq0 == 900


def test_boot_slo_uses_spatial_placement_not_hardcoded_tp2():
    spatial = Partition(
        n_mixed=0, n_prefill=4, n_decode=4,
        tp_prefill=1, tp_decode=1, gpus_total=8)
    r = choose_boot_slo_layout(
        _CapPred(), rate=2.0, att_mixed=0.97, trusted=False,
        frac_long=0.0, mixed_tp=8, spatial=spatial)
    assert r.chosen.n_prefill == 4
    assert r.chosen.tp_prefill == 1
    assert r.unused_gpus == 0
