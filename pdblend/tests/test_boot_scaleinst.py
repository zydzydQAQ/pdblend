# -*- coding: utf-8 -*-
"""pdblend 开机 ScaleInst 组合:used / A9 / μ_D / FREQ0。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecopadg.boot_scaleinst import compose_boot_slo_layout, spatial_pd_on_used
from ecopadg.dynamo_planner import (
    PDBLEND_PROFILE_PURPOSE, builtin_profile, pdblend_profile, write_profile,
)
from ecopadg.planner import (
    choose_boot_slo_layout, layout_kind, spatial_pd_capacity_ok,
    spatial_pd_partition,
)
from ecopadg.types import Partition

ROOT = Path(__file__).resolve().parents[1]


class _CapPred:
    def stage_rate_prefill(self, n, freq=None):
        del freq
        return 20.0 if int(n) > 0 else 0.0

    def stage_rate_decode(self, n, freq=None):
        del freq
        return 20.0 if int(n) > 0 else 0.0

    def rate_mixed(self, n, freq=None, prefill_freq=None):
        del freq, prefill_freq
        return 15.0 if int(n) > 0 else 0.0

    def __call__(self, p, lam, state=None):
        del p, lam, state
        from ecopadg.global_scheduler import Prediction
        return Prediction(attainable=True, attainment=0.97,
                          power_w=800.0, energy_j_per_s=800.0)


def test_rate_nonpositive_falls_back_to_tp8():
    r = choose_boot_slo_layout(
        _CapPred(), rate=0.0, att_mixed=0.97, trusted=False)
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.used_gpus == 8
    assert r.unused_gpus == 0
    assert r.freq0 == 2520
    assert r.fallback == "rate-nonpositive"


def test_choose_boot_slo_layout_never_shrinks():
    r = choose_boot_slo_layout(
        _CapPred(), rate=2.0, att_mixed=0.97, trusted=False, frac_long=0.0)
    assert r.used_gpus == 8
    assert r.unused_gpus == 0
    assert r.fallback != "untrusted-scaleinst-mixed"
    assert "scaleinst" not in (r.fallback or "")


def test_scaleinst_compose_still_leaves_gpus_idle():
    r = compose_boot_slo_layout(
        _CapPred(), rate=2.0, att_mixed=0.97, frac_long=0.0)
    assert r.unused_gpus > 0
    assert r.used_gpus + r.unused_gpus == 8
    assert r.fallback == "untrusted-scaleinst-mixed"


def test_a9_never_selected_on_partial_used():
    r = compose_boot_slo_layout(
        _CapPred(), rate=2.0, att_mixed=0.97, frac_long=0.4)
    assert r.unused_gpus > 0
    assert layout_kind(r.chosen) != "hybrid-a9"
    a9 = [s for s in r.scores if layout_kind(s.partition) == "hybrid-a9"]
    assert a9 and all(s.partition.total_gpus() == 8 for s in a9)


def test_packing_rejected_falls_back_to_tp8():
    r = choose_boot_slo_layout(
        _CapPred(), rate=20.0, att_mixed=0.97, trusted=False, frac_long=0.4,
        mixed_tp=8)
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.unused_gpus == 0
    assert r.freq0 == 900
    assert r.fallback in (
        "capacity-rho-guard-mixed", "boot-mixed-fallback")


def test_four_gpu_shrink_is_forced_to_mixed_tp8():
    r = compose_boot_slo_layout(
        _CapPred(), rate=6.0, att_mixed=0.97, frac_long=0.0)
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.unused_gpus == 0
    assert r.freq0 == 2520
    assert r.fallback == "minj-vs-mixed"


def test_fullnode_winner_is_forced_to_mixed_tp8():
    r = choose_boot_slo_layout(
        _CapPred(), rate=20.0, att_mixed=0.97, trusted=False, frac_long=0.0)
    assert r.chosen.n_mixed == 1
    assert r.chosen.tp_mixed == 8
    assert r.unused_gpus == 0
    assert r.freq0 == 900
    assert r.fallback in (
        "capacity-rho-guard-mixed", "boot-mixed-fallback")
    assert int(r.chosen.freq_mixed or 2520) == 2520


def test_mu_d_unknown_fails_closed():
    class NoDecode(_CapPred):
        def stage_rate_decode(self, n, freq=None):
            del n, freq
            return None

    part = spatial_pd_partition(8, 2)
    guard = spatial_pd_capacity_ok(NoDecode(), 2.0, part)
    assert guard.pd_ok is False
    assert guard.reason == "capacity-mu-d-unknown-mixed"


def test_mu_d_and_gate_allows_when_both_sides_slack():
    part = spatial_pd_partition(8, 2)
    guard = spatial_pd_capacity_ok(_CapPred(), 2.0, part)
    assert guard.pd_ok is True
    assert guard.reason == "capacity-rho-ok"


def test_mu_d_and_gate_vetoes_decode_bound():
    class DecodeBound(_CapPred):
        def stage_rate_decode(self, n, freq=None):
            del freq
            return 8.0 if int(n) > 0 else 0.0

    part = spatial_pd_partition(8, 2)
    guard = spatial_pd_capacity_ok(DecodeBound(), 6.0, part)
    assert guard.pd_ok is False
    assert guard.reason == "capacity-decode-rho-guard-mixed"


def test_spatial_pd_on_used_includes_distserve_winner_only_at_eight():
    spatial = Partition(
        n_mixed=0, n_prefill=4, n_decode=4,
        tp_prefill=1, tp_decode=1, gpus_total=8)
    eight = spatial_pd_on_used(8, spatial, 8)
    assert any(p.n_prefill == 4 and p.tp_prefill == 1 for p in eight)
    four = spatial_pd_on_used(4, spatial, 8)
    assert all(p.total_gpus() == 4 for p in four)
    assert not any(p.n_prefill == 4 for p in four)


def test_pdblend_profile_rejects_lookup_json(tmp_path):
    path = tmp_path / "lookup.json"
    path.write_text('{"freqs": {"2520": {}}, "lookup": {"x": 1}}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="lookup.json"):
        pdblend_profile("14b", profile_path=str(path))
    other = tmp_path / "profile.json"
    write_profile(dict(builtin_profile(), lookup={"k": 1}), str(other))
    with pytest.raises(ValueError, match="lookup table"):
        pdblend_profile("14b", profile_path=str(other))


def test_pdblend_profile_14b_json_is_not_lookup():
    path = ROOT / "script" / "bench" / "pdblend_profile_14b.json"
    assert path.is_file()
    prof = pdblend_profile("14b", profile_path=str(path))
    assert prof["lookup"] is False
    assert prof["purpose"] == PDBLEND_PROFILE_PURPOSE
    assert abs(float(prof["rho_max"]) - 0.90) < 1e-9
    assert abs(float(prof["ttft_frac"]) - 1.0) < 1e-9
    assert "lookup" not in prof.get("freqs", {})
    raw = json.loads(path.read_text(encoding="utf-8"))
    for row in list(raw["freqs"].values()) + list(prof["freqs"].values()):
        assert float(row["gpu_w"]) < 200.0
        assert 70.0 <= float(row["gpu_w"]) <= 140.0
    assert 50.0 <= float(raw["idle_empty_w"]) <= 100.0


def test_run_cell_applies_freq0_as_floor():
    text = (ROOT / "script" / "bench" / "run_cell.sh").read_text(
        encoding="utf-8")
    assert "lock_freq_floor" in text
    assert "ALLOW_PARTIAL" in text
    assert 'nvidia-smi -lgc "${lo},2520"' in text
    assert "lock_freq_floor \"$FREQ0\"" in text
    assert "allow_partial" in text
    dyna = text.split("dynamollm)")[2].split(";;")[0]
    assert "lock_freq_floor" in dyna
    pdb = text.split("pdblend|ecospd)")[1].split("assert_containers_alive")[0]
    assert "lock_freq_floor" in pdb
    assert "feature_gate=\"boot-slo-fullnode-no-mid-cell-remat\"" in text


def test_validate_runtime_layout_allow_partial():
    from ecopadg.ds_placement import validate_runtime_layout
    assert validate_runtime_layout(
        "14b", 2, 0, 1, 0, 0, allow_partial=True) == 2
    with pytest.raises(ValueError):
        validate_runtime_layout("14b", 0, 0, 1, 0, 0, allow_partial=True)
    with pytest.raises(ValueError):
        validate_runtime_layout("14b", 2, 0, 1, 0, 0, allow_partial=False)
    with pytest.raises(ValueError):
        validate_runtime_layout(
            "14b", 9, 0, 1, 0, 0, gpu_count=8, allow_partial=True)
