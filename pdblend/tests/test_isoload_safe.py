# -*- coding: utf-8 -*-
"""IsoLoad-Safe:频率可信度、14b 频档、nan baseline 不把门槛写成 1.0。"""
from __future__ import annotations

import pytest

from ecopadg.opmodel import freq_model_trustworthy
from ecopadg.types import freqs_for_model


class _FlatOM:
    def iter_time_ms(self, batch, freq_mhz, ctx_tokens=272.0):
        return 21.3


class _ScaledOM:
    def iter_time_ms(self, batch, freq_mhz, ctx_tokens=272.0):
        return 21.3 * (2520.0 / max(float(freq_mhz), 1.0))


def test_flat_iter_table_untrusted():
    assert freq_model_trustworthy(_FlatOM(), (2520, 1500, 900)) is False


def test_scaled_iter_or_pe1_trusted():
    assert freq_model_trustworthy(_ScaledOM(), (2520, 900)) is True
    assert freq_model_trustworthy(_FlatOM(), (2520, 900),
                                  pe1_calibrated=True) is True


def test_freqs_14b_drops_600():
    f = freqs_for_model("14b")
    assert 600 not in f
    assert min(f) == 900
    assert max(f) == 2520


def test_runtime_space_untrusted_drops_partial():
    from ecopadg.global_scheduler import build_partition_space
    from ecopadg.planner import trusted_space
    raw = [p for p in build_partition_space(
        {"mixed": [2], "prefill": [2], "decode": [2]}, 8)
        if p.n_mixed <= 4 and p.n_prefill == p.n_decode]
    safe = trusted_space(raw, 8, trusted=False)
    assert safe
    assert all(p.total_gpus() == 8 for p in safe)
    assert not any(p.n_mixed == 1 and p.pairs() == 0 for p in safe)


def test_capacity_untrusted_without_mu_s90():
    from ecopadg.predictor import capacity_model_trustworthy
    assert capacity_model_trustworthy(False, None, False) is False
    assert capacity_model_trustworthy(True, None, True) is False
    assert capacity_model_trustworthy(True, -1.0, True) is False
    assert capacity_model_trustworthy(True, 1.0, True) is True
    assert capacity_model_trustworthy(True, 0.0, True) is False


def test_capacity_untrusted_boots_mixed_when_att_zero():
    """不可信:预测否决 PD 也不 inherit,启动只起 mixed。"""
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    class Pred:
        def __call__(self, p, lam, state=None):
            if p.pairs() > 0:
                return Prediction(attainable=False, attainment=0.0,
                                  power_w=1982.0, energy_j_per_s=1e12)
            return Prediction(attainable=True, attainment=0.98,
                              power_w=2744.0, energy_j_per_s=2800.0)

    space = [
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
        Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                  freq_prefill=2520, freq_decode=2520, gpus_total=8),
    ]
    r = choose_iso_load(Pred(), lam=4.0, att_mixed=0.98, space=space,
                        gpu_count=8, trusted=False, capacity_trusted=False)
    assert r.chosen is not None
    assert r.chosen.pairs() == 0
    assert r.chosen.n_mixed == 4
    assert r.fallback == "boot-mixed"


def test_untrusted_ignores_inherit_even_if_cheaper():
    from ecopadg.ds_placement import partition_from_placement
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    inherit = partition_from_placement(
        dict(pp_cross=2, tp_prefill=2, tp_decode=2), 8)

    class Pred:
        def __call__(self, p, lam, state=None):
            pd = p.pairs() > 0
            return Prediction(attainable=True, attainment=0.98,
                              power_w=400.0 if pd else 900.0,
                              energy_j_per_s=400.0 if pd else 900.0,
                              rho=0.2)

    r = choose_iso_load(Pred(), lam=4.0, att_mixed=0.98, space=[
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
        inherit,
    ], gpu_count=8, trusted=False, capacity_trusted=False, inherit=inherit,
        layout_decision="inherit", layout_reason="oracle-inherit")
    assert r.chosen.pairs() == 0
    assert r.chosen.n_mixed == 4
    assert r.fallback == "boot-mixed"


def test_untrusted_keeps_mixed_when_pd_worse():
    from ecopadg.ds_placement import DEFAULT_72B, partition_from_placement
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    inherit = partition_from_placement(DEFAULT_72B, 8)

    class Pred:
        def __call__(self, p, lam, state=None):
            if p.pairs() > 0:
                return Prediction(attainable=True, attainment=0.90,
                                  power_w=400.0, energy_j_per_s=400.0,
                                  rho=0.8)
            return Prediction(attainable=True, attainment=1.0,
                              power_w=800.0, energy_j_per_s=800.0, rho=0.3)

    r = choose_iso_load(Pred(), lam=2.0, att_mixed=1.0, space=[
        Partition(n_mixed=2, tp_mixed=4, freq_mixed=2520, gpus_total=8),
        inherit,
    ], gpu_count=8, tp_fallback=4, trusted=False, capacity_trusted=False,
        inherit=inherit)
    assert r.chosen.n_mixed == 2
    assert r.chosen.pairs() == 0
    assert r.fallback == "boot-mixed"


def test_72b_untrusted_boots_mixed():
    from ecopadg.ds_placement import DEFAULT_72B, partition_from_placement
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    inherit = partition_from_placement(DEFAULT_72B, 8)

    class Pred:
        def __call__(self, p, lam, state=None):
            return Prediction(attainable=True, attainment=0.97,
                              power_w=400.0 if p.pairs() else 800.0,
                              energy_j_per_s=400.0 if p.pairs() else 800.0,
                              rho=0.7)

    r = choose_iso_load(Pred(), lam=1.5, att_mixed=0.97, space=[
        Partition(n_mixed=2, tp_mixed=4, freq_mixed=2520, gpus_total=8),
        inherit,
    ], gpu_count=8, tp_fallback=4, trusted=False, capacity_trusted=False,
        inherit=inherit)
    assert r.chosen.pairs() == 0
    assert r.chosen.n_mixed == 2
    assert r.fallback == "boot-mixed"


def test_trusted_energy_cap_rejects_above_dspd():
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    class Pred:
        def __call__(self, p, lam, state=None):
            e = 500.0 if p.n_mixed == 4 else 400.0
            return Prediction(attainable=True, attainment=0.98,
                              power_w=e, energy_j_per_s=e)

    mixed = Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8)
    pd = Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                   freq_prefill=2520, freq_decode=2520, gpus_total=8)
    r = choose_iso_load(Pred(), lam=2.0, att_mixed=0.98, space=[mixed, pd],
                        gpu_count=8, trusted=True, capacity_trusted=True,
                        inherit=pd, energy_cap_j_per_s=400.0)
    assert r.chosen.pairs() == 2


def test_ds_candidates_include_layer_pp_and_old_three():
    from ecopadg.ds_placement import (
        CANDIDATES_14B, enumerate_ds_candidates, placement_gpus,
        validate_runtime_layout,
    )

    keys = {
        (c["pp_cross"], c["tp_prefill"], c["pp_prefill"],
         c["tp_decode"], c["pp_decode"])
        for c in enumerate_ds_candidates("14b")
    }
    for old in ((2, 2, 1, 2, 1), (1, 4, 1, 4, 1), (4, 1, 1, 1, 1)):
        assert old in keys
    for layer_pp in ((1, 2, 2, 2, 2), (1, 1, 4, 1, 4), (2, 1, 2, 1, 2)):
        assert layer_pp in keys
    assert len(keys) == 14
    assert len(CANDIDATES_14B) == 3
    for c in CANDIDATES_14B:
        assert placement_gpus(c) == 8
        assert int(c.get("pp_prefill") or 1) == 1
        assert int(c.get("pp_decode") or 1) == 1
    assert validate_runtime_layout(
        "14b", 0, 1, 8, 2, 2, pp_prefill=2, pp_decode=2) == 8


def test_pick_measured_prefers_2xtp2_on_tie():
    from ecopadg.ds_placement import CANDIDATES_14B, pick_measured
    meas = [
        dict(pp_cross=2, tp_prefill=2, tp_decode=2, rate=4.0,
             slo_attainment=0.975),
        dict(pp_cross=1, tp_prefill=4, tp_decode=4, rate=4.0,
             slo_attainment=0.975),
    ]
    w = pick_measured(CANDIDATES_14B, meas)
    assert w["pp_cross"] == 2
    assert w["tp_prefill"] == 2


def test_72b_default_is_single_tp4_pair():
    from ecopadg.ds_placement import (
        CANDIDATES_72B, DEFAULT_72B, pick_measured,
        validate_runtime_layout,
    )

    chosen = pick_measured(CANDIDATES_72B, [], default=DEFAULT_72B)
    assert chosen["n_segments"] == 1
    assert chosen["tp_prefill"] == chosen["tp_decode"] == 4
    assert validate_runtime_layout("72b", 0, 1, 4, 4, 4) == 8


def test_72b_pd_follows_ds_placement_gpu_budget():
    from ecopadg.ds_placement import validate_runtime_layout

    assert validate_runtime_layout("72b", 0, 2, 8, 2, 2) == 8
    assert validate_runtime_layout("7b", 1, 0, 8, 0, 0) == 8
    with pytest.raises(ValueError, match="非法布局"):
        validate_runtime_layout("72b", 0, 2, 8, 2, 1)


def test_e1b_14b_not_synthetic_after_probe_commit():
    import os

    import pytest

    from ecopadg.opmodel import (
        default_table_dir, e1b_dir_synthetic, freq_model_trustworthy,
        load_opmodel,
    )
    from ecopadg.types import FREQS_14B
    base = default_table_dir("14b")
    pd_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    via_pd = default_table_dir("14b", pd_root)
    assert os.path.isfile(os.path.join(via_pd, "p1b_latency.csv"))
    assert e1b_dir_synthetic(base) is False
    assert e1b_dir_synthetic(default_table_dir("32b")) is False
    if not os.path.isfile(os.path.join(base, "p1b_latency.csv")):
        pytest.skip("14B E1b 表尚未重采: %s" % base)
    om = load_opmodel(base)
    assert freq_model_trustworthy(om, FREQS_14B) is True


def test_nan_baseline_uses_predict_not_one():
    """控制器缺省不得把 target 写成 0.99(att_high=0.98 会全灭)。"""
    from ecopadg.global_scheduler import Prediction
    from ecopadg.planner import choose_iso_load
    from ecopadg.types import Partition

    class Pred:
        def __call__(self, p, lam, state=None):
            return Prediction(attainable=True, attainment=0.98,
                              power_w=800.0, energy_j_per_s=800.0)

    # 模拟 controller:nan → predict(mixed@2520)=0.98,门槛 0.97
    att_mixed = 0.98
    r = choose_iso_load(Pred(), lam=2.0, att_mixed=att_mixed, space=[
        Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8),
        Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                  gpus_total=8),
    ], gpu_count=8, trusted=False)
    assert r.chosen is not None
    assert r.chosen.pairs() == 0
    assert r.fallback == "boot-mixed"
    assert not r.fallback.startswith("mixed@")
