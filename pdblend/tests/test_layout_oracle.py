# -*- coding: utf-8 -*-
"""在线启动:boot mixed;容量崖只认现场 λ/μ。无 inherit 表。"""
from __future__ import annotations

import os

from ecopadg.ds_placement import partition_from_placement
from ecopadg.global_scheduler import Prediction
from ecopadg.layout_oracle import cliff_lock_wanted
from ecopadg.planner import choose_iso_load
from ecopadg.types import Partition


class _Pred:
    def __call__(self, p, lam, state=None):
        pd = p.pairs() > 0
        return Prediction(attainable=not pd, attainment=0.0 if pd else 0.98,
                          power_w=400.0 if pd else 900.0,
                          energy_j_per_s=1e12 if pd else 900.0, rho=0.9)


def _pd():
    return Partition(n_prefill=2, n_decode=2, tp_prefill=2, tp_decode=2,
                     freq_prefill=2520, freq_decode=2520, gpus_total=8)


def _mixed():
    return Partition(n_mixed=4, tp_mixed=2, freq_mixed=2520, gpus_total=8)


def _assert_boot_mixed(r):
    assert r.chosen.pairs() == 0
    assert r.chosen.n_mixed == 4
    assert r.fallback == "boot-mixed"


def test_untrusted_high_rate_boots_mixed_not_inherit():
    inherit = _pd()
    r = choose_iso_load(
        _Pred(), lam=6.0, att_mixed=0.975, space=[_mixed(), inherit],
        gpu_count=8, trusted=False, capacity_trusted=False, inherit=inherit,
        model_key="14b", dataset="sharegpt", arrival="poisson")
    _assert_boot_mixed(r)


def test_untrusted_gamma_boots_mixed():
    inherit = _pd()
    r = choose_iso_load(
        _Pred(), lam=4.0, att_mixed=1.0, space=[_mixed(), inherit],
        gpu_count=8, trusted=False, capacity_trusted=False, inherit=inherit,
        model_key="14b", dataset="alpaca", arrival="gamma")
    _assert_boot_mixed(r)


def test_untrusted_midload_boots_mixed():
    inherit = _pd()
    r = choose_iso_load(
        _Pred(), lam=2.4, att_mixed=0.975, space=[_mixed(), inherit],
        gpu_count=8, trusted=False, capacity_trusted=False, inherit=inherit,
        model_key="32b", dataset="sharegpt", arrival="poisson")
    _assert_boot_mixed(r)


def test_untrusted_without_inherit_boots_mixed():
    inherit = partition_from_placement(
        dict(pp_cross=2, tp_prefill=2, tp_decode=2), 8)
    r = choose_iso_load(
        _Pred(), lam=2.0, att_mixed=0.98, space=[_mixed(), inherit],
        gpu_count=8, trusted=False, capacity_trusted=False, inherit=inherit)
    _assert_boot_mixed(r)


def test_cliff_lock_wanted():
    assert cliff_lock_wanted(5.0, 4.0) is True
    assert cliff_lock_wanted(2.0, 4.0) is False
    assert cliff_lock_wanted(1.0, 10.0, static_cliff=True) is True
    assert cliff_lock_wanted(0.0, 0.0, slo_trip=True) is True
    assert cliff_lock_wanted(float("nan"), 4.0) is False


def test_c_fix_wave_reuses_baselines_and_reruns_ecospd():
    import sys
    here = os.path.join(os.path.dirname(__file__),
                        "..", "new-results", "scripts")
    sys.path.insert(0, os.path.abspath(here))
    from asplos_matrix import C_FIX_KEYS, wave_cells
    cells = wave_cells("c_fix")
    assert len(C_FIX_KEYS) == 8
    assert len(cells) == 24
    assert sum(1 for c in cells if c["system"] == "ecospd") == 8
    assert sum(1 for c in cells if c["system"] == "mixed") == 8
    assert sum(1 for c in cells if c["system"] == "ds_pd") == 8


def test_p2_14b_wave_is_control_plane_not_oracle():
    import sys
    here = os.path.join(os.path.dirname(__file__),
                        "..", "new-results", "scripts")
    sys.path.insert(0, os.path.abspath(here))
    from asplos_matrix import RATES_DENSE_14B, wave_cells
    cells = wave_cells("p2_14b")
    assert cells
    assert all(c["model"] == "14b" for c in cells)
    systems = {c["system"] for c in cells}
    assert systems == {"mixed", "mixed_dvfs", "ds_pd", "pdblend"}
    sg = [c["rate"] for c in cells
          if c["dataset"] == "sharegpt" and c["process"] == "poisson"
          and c["system"] == "mixed"]
    assert sg == RATES_DENSE_14B[("14b", "sharegpt")]
    assert len(cells) == sum(len(rs) * 2 * 4 for rs in RATES_DENSE_14B.values())


def test_strict_smoke_wave_compares_gating_to_engine_purity():
    import sys
    here = os.path.join(os.path.dirname(__file__),
                        "..", "new-results", "scripts")
    sys.path.insert(0, os.path.abspath(here))
    from asplos_matrix import wave_cells
    cells = wave_cells("strict_smoke")
    assert len(cells) == 8
    assert {c["system"] for c in cells} == {
        "mixed", "mtilde1", "strict_padg_v0", "strict_padg"}
    assert {c["model"] for c in cells} == {"14b", "32b"}
