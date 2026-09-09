# -*- coding: utf-8 -*-
from __future__ import annotations

from ecopadg.dynamo_planner import (
    FREQS, GPU_COUNT, PACKINGS, DynamoPacking, apply_tp1_overlay,
    builtin_profile, choose_dynamo_plan, enumerate_packings,
    load_or_build_profile, pdblend_profile, predict_packing,
    synthesize_profile_from_opmodel, tp1_row_from_measured, write_profile,
)


def test_enumerate_packings_are_homogeneous_and_fit_eight_gpus():
    rows = enumerate_packings()
    keys = {(p.n_replica, p.tp) for p in rows}
    assert keys == set(PACKINGS)
    assert {p.freq_mhz for p in rows} == set(FREQS)
    assert all(1 <= p.used_gpus <= GPU_COUNT for p in rows)
    assert all(p.unused_gpus == GPU_COUNT - p.used_gpus for p in rows)


def test_low_rate_scaleinst_leaves_gpus_idle():
    plan = choose_dynamo_plan(
        rate=2.0, n=200, dataset="sharegpt", profile=builtin_profile())
    assert plan.unused_gpus >= 4
    assert plan.used_gpus == plan.n_replica * plan.tp
    assert plan.used_gpus + plan.unused_gpus == 8
    assert plan.slo_ok
    assert plan.source == "builtin-14b"
    assert plan.tp in (1, 2, 4, 8)


def test_high_rate_uses_more_gpus_than_low_rate():
    low = choose_dynamo_plan(
        rate=2.0, n=200, dataset="sharegpt", profile=builtin_profile())
    high = choose_dynamo_plan(
        rate=10.0, n=200, dataset="sharegpt", profile=builtin_profile())
    assert high.used_gpus > low.used_gpus
    assert high.unused_gpus < low.unused_gpus


def test_energy_includes_empty_idle_cards():
    plan = choose_dynamo_plan(
        rate=2.0, n=200, dataset="sharegpt", profile=builtin_profile())
    idle_rows = [s for s in plan.scores
                 if s["n"] == plan.n_replica and s["tp"] == plan.tp
                 and s["freq"] == plan.freq0]
    assert idle_rows
    assert abs(plan.energy_j - idle_rows[0]["energy_j"]) < 1.0
    assert plan.energy_j > 0
    assert any(s["unused"] > 0 and s["slo_ok"] for s in plan.scores)


def test_write_profile_marks_source(tmp_path):
    path = tmp_path / "dynamo_profile.json"
    write_profile(builtin_profile("14b"), str(path))
    loaded = load_or_build_profile("14b", profile_path=str(path))
    assert loaded["source"] == "builtin-14b"
    assert "2520" in loaded["freqs"]


def test_nonpositive_rate_falls_back_to_tp8():
    plan = choose_dynamo_plan(
        rate=0.0, n=200, dataset="sharegpt", profile=builtin_profile())
    assert plan.n_replica == 1 and plan.tp == 8
    assert plan.unused_gpus == 0
    assert plan.source == "rate-missing-fallback-tp8"


def test_pdblend_profile_adds_iso_slo_margin(tmp_path):
    path = tmp_path / "p.json"
    write_profile(builtin_profile("14b"), str(path))
    prof = pdblend_profile("14b", profile_path=str(path))
    assert prof["lookup"] is False
    assert prof["purpose"] == "pdblend-boot-scaleinst"
    assert abs(float(prof["rho_max"]) - 0.90) < 1e-9
    assert abs(float(prof["ttft_frac"]) - 1.0) < 1e-9
    for row in prof["freqs"].values():
        assert 70.0 <= float(row["gpu_w"]) <= 140.0
        assert float(row["gpu_w"]) < 200.0
    assert 50.0 <= float(prof["idle_empty_w"]) <= 100.0


def test_pdblend_profile_rewrites_two_card_probe_watts(tmp_path):
    src = dict(builtin_profile("14b"))
    src["freqs"] = {k: dict(v, gpu_w=500.0) for k, v in src["freqs"].items()}
    src["idle_empty_w"] = 35.0
    path = tmp_path / "p.json"
    write_profile(src, str(path))
    prof = pdblend_profile("14b", profile_path=str(path))
    for row in prof["freqs"].values():
        assert float(row["gpu_w"]) < 200.0
        assert 70.0 <= float(row["gpu_w"]) <= 140.0
    assert abs(float(prof["idle_empty_w"]) - 67.0) < 1e-6
    assert prof["power_unit"] == "per-gpu"


def test_tp1_att_gate_rejects_packing():
    prof = apply_tp1_overlay(builtin_profile("14b"), {
        "2100": tp1_row_from_measured(
            0.8, 0.04, 0.925, mixed_att_ref=0.97),
    })
    assert prof["tp1"]["2100"]["slo_ok"] is False
    sc = predict_packing(
        DynamoPacking(1, 1, 2100), prof,
        rate=2.0, n=200, pin=512, pout=200,
        slo_ttft_s=5.0, slo_tpot_s=0.15,
        rho_max=0.90, use_tp1=True)
    assert sc.slo_ok is False
    assert "att_gate" in sc.reason
    skipped = predict_packing(
        DynamoPacking(1, 1, 2100), prof,
        rate=2.0, n=200, pin=512, pout=200,
        slo_ttft_s=5.0, slo_tpot_s=0.15, rho_max=0.90)
    assert "att_gate" not in skipped.reason
    ok = tp1_row_from_measured(0.4, 0.03, 0.97, mixed_att_ref=0.97)
    assert ok["slo_ok"] is True


def test_pdblend_profile_keeps_tp1_overlay():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "script" / "bench" / "pdblend_profile_14b.json"
    if not path.is_file():
        return
    raw = __import__("json").loads(path.read_text(encoding="utf-8"))
    if "tp1" not in raw:
        return
    prof = pdblend_profile("14b", profile_path=str(path))
    assert "2100" in prof["tp1"]
    assert prof["tp1"]["2100"]["slo_ok"] is False
    assert float(prof["freqs"]["2520"]["gpu_w"]) < 200.0


def test_shell_exports_match_plan():
    plan = choose_dynamo_plan(
        rate=2.0, n=200, dataset="sharegpt", profile=builtin_profile())
    text = plan.shell_exports()
    assert "MIXED_TP=%d" % plan.tp in text
    assert "MIXED_N_REPLICA=%d" % plan.n_replica in text
    assert "FREQ0=%d" % plan.freq0 in text
    assert "UNUSED_GPUS=%d" % plan.unused_gpus in text


def test_synthesize_profile_uses_dataset_pin(synth_opmodel):
    alp = synthesize_profile_from_opmodel(
        synth_opmodel, "14b", dataset="alpaca")
    lng = synthesize_profile_from_opmodel(
        synth_opmodel, "14b", dataset="longbench")
    assert alp["shape_pin"] == 128
    assert lng["shape_pin"] == 2048
    a = float(alp["freqs"]["2520"]["prefill_s_per_tok"])
    b = float(lng["freqs"]["2520"]["prefill_s_per_tok"])
    assert a != b
