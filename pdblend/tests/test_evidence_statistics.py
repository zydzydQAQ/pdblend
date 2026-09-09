# -*- coding: utf-8 -*-
"""CPU-only evidence-tier planning and paired-statistics tests."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "new-results" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from aggregate_iso_load import (  # noqa: E402
    aggregate_pairs,
    bootstrap_mean_ci,
    normalize_row,
    pair_seed_rows,
)
from aggregate_ablation import ablation_pairs, aggregate_ablation  # noqa: E402
from asplos_matrix import (  # noqa: E402
    SYSTEMS_PROOF_REGIME_14B,
    SYSTEMS_FREQUENCY_SMOKE,
    SYSTEMS_PDBLEND_SPATIAL_SMOKE,
    RATES_PDBLEND_SPATIAL_14B,
    SAT_PROBE_14B,
    RATES_SAT_N200_14B,
    RATES_SLO_RETEST_14B,
    RATES_KNEE_SHIELD_14B,
    RATES_TWO_BASE_LOW_N200_14B,
    RATES_TWO_BASE_ALPACA_N200_14B,
    RATES_TWO_BASE_LONGBENCH_N200_14B,
    RATES_LOWLOAD_N200_14B,
    RATES_LOOKUP_N200_14B,
    RATES_RHO_GUARD_SHAREGPT_14B,
    RATES_RHO_GUARD_ALPACA_14B,
    RATES_MIXED_DVFS_SHAREGPT_14B,
    RATES_MIXED_DVFS_LONGBENCH_14B,
    RATES_A9_SHAREGPT_14B,
    RATES_A9_ALPACA_14B,
    RATES_DS_PLACEMENT_14B,
    SYSTEMS_DS_PLACEMENT_14B,
    RATES_DS_PP_SMOKE_14B,
    SYSTEMS_DS_PP_SMOKE_14B,
    RATES_TP8_SHAREGPT_14B,
    RATES_TP8_ALPACA_14B,
    SYSTEMS_TP8_14B,
    RATES_REPRO_BASE_14B,
    SYSTEMS_REPRO_BASE_14B,
    RATES_DIR_SG_14B,
    RATES_DIR_ALPACA_14B,
    RATES_DIR_LB_14B,
    RATES_DIR_KNEE_14B,
    SYSTEMS_DIR_FULLNODE_14B,
    SYSTEMS_DIR_KNEE_14B,
    RATES_JMIN_SG_14B,
    RATES_JMIN_PROBE_14B,
    SYSTEMS_JMIN_SG_14B,
    SYSTEMS_PUB_FIGAB_14B,
    RATES_PUB_FIGB_14B,
    RATES_PUB_FIGA_14B,
    assert_dir_fullnode_ready,
    placement_for_system,
    SYSTEMS_TWO_BASE_14B,
    KEEP_REL,
    find_keep_cell,
    prior_mixed_att,
    SYSTEMS_GOLD_CONFIRM,
    SYSTEM_LADDER,
    build_plan,
    cell_tag,
    cells_from_line,
    dest_for,
    effective_reuse_policy,
    parse_seeds,
    run_cell_system,
    update_manifest_status,
    validate_cell_provenance,
    wave_cells,
)
from build_regime_map import build_mode_map  # noqa: E402
from export_envelope import measured_envelope, predicted_overlay  # noqa: E402
from offline_oracle import replay_oracle  # noqa: E402
from validate_gold_confirm import (  # noqa: E402
    validate_campaign,
    validate_strict_sched,
    verdict_from_aggregates,
)


def _row(seed, system, *, n=500, rate=2.0, ttft=5.0, tpot=0.15,
         total_j=100.0, attainment=0.98, validity="ok", **extra):
    row = dict(
        model="14b", dataset="sharegpt", process="poisson",
        rate=rate, n=n, seed=str(seed), gpu_count=8,
        slo_ttft_s=ttft, slo_tpot_s=tpot, system=system,
        total_j=total_j, slo_attainment=attainment, validity=validity,
        publication_eligible=True, diagnostic=False, report_only=False,
    )
    row.update(extra)
    return row


def test_tier_wave_counts_n_and_full_system_ladder():
    smoke = wave_cells("smoke")
    confirm = wave_cells("confirm")
    publication = wave_cells("publication")
    assert len(smoke) == 2 * len(SYSTEM_LADDER) == 20
    assert len(confirm) == 2 * 3 * 2 * len(SYSTEM_LADDER) == 120
    assert len(publication) == 2 * 3 * 3 * 2 * len(SYSTEM_LADDER) == 360
    assert {cell["n"] for cell in smoke} == {80}
    assert {cell["n"] for cell in confirm} == {500}
    assert {cell["n"] for cell in publication} == {500}
    assert {cell["system"] for cell in smoke} == set(SYSTEM_LADDER)
    with pytest.raises(ValueError):
        wave_cells("confirm", n=499)
    with pytest.raises(ValueError):
        wave_cells("publication", n=80)


def test_frequency_smoke_is_exact_seeded_eight_cell_wave(tmp_path):
    cells = wave_cells("frequency_smoke", seeds="7")
    assert len(cells) == 8
    assert {cell["n"] for cell in cells} == {80}
    assert {cell["tier"] for cell in cells} == {"smoke"}
    assert {cell["seed"] for cell in cells} == {"7"}
    assert {cell["system"] for cell in cells} == set(
        SYSTEMS_FREQUENCY_SMOKE)
    assert {
        (cell["model"], cell["rate"])
        for cell in cells
    } == {("14b", "2"), ("32b", "2.4")}
    assert all(cell["include_seed"] for cell in cells)
    assert all(
        "_seed7" in Path(dest_for(str(tmp_path), cell)).name
        for cell in cells)
    with pytest.raises(ValueError, match="exactly one seed"):
        wave_cells("frequency_smoke", seeds="0,1")
    with pytest.raises(ValueError, match="n=80"):
        wave_cells("frequency_smoke", n=81, seeds="0")


def test_gold_confirm_is_exact_24_cell_ordered_evidence_wave(tmp_path):
    cells = wave_cells("gold_confirm", seeds="0,1,2")
    assert len(cells) == 24
    assert {cell["n"] for cell in cells} == {500}
    assert {cell["model"] for cell in cells} == {"14b", "32b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {cell["system"] for cell in cells} == set(SYSTEMS_GOLD_CONFIRM)
    assert all(cell["include_seed"] for cell in cells)
    assert all(cell.get("publication_eligible", True) for cell in cells)
    destinations = [dest_for(str(tmp_path), cell) for cell in cells]
    assert len(destinations) == len(set(destinations))
    assert all("_seed" in Path(path).name for path in destinations)
    for model in ("14b", "32b"):
        model_cells = [cell for cell in cells if cell["model"] == model]
        assert [cell["system"] for cell in model_cells[::3]] == list(
            SYSTEMS_GOLD_CONFIRM)
        for seed in ("0", "1", "2"):
            systems = [
                cell["system"] for cell in model_cells
                if cell["seed"] == seed
            ]
            assert systems == list(SYSTEMS_GOLD_CONFIRM)


def test_proof_regime_14b_is_fixed_24_cell_seed0_wave():
    cells = wave_cells("proof_regime_14b", seeds="0")
    assert len(cells) == 24
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["n"] for cell in cells} == {80}
    assert {cell["system"] for cell in cells} == set(
        SYSTEMS_PROOF_REGIME_14B)
    assert {
        (cell["dataset"], cell["process"], str(cell["rate"]))
        for cell in cells
    } == {
        ("sharegpt", "gamma", "4"),
        ("sharegpt", "gamma", "6"),
        ("alpaca", "poisson", "4"),
        ("alpaca", "poisson", "8"),
        ("longbench", "poisson", "1"),
        ("longbench", "poisson", "2"),
    }
    assert run_cell_system("strict_pd_dvfs") == "ds_pd_dvfs"
    with pytest.raises(ValueError):
        wave_cells("proof_regime_14b", seeds="0,1")


def test_survivor_wave_is_exact_four_cell_transition_diagnostic():
    cells = wave_cells("survivor_14b", seeds="0")
    assert len(cells) == 4
    assert [(str(c["rate"]), c["system"]) for c in cells] == [
        ("1", "mixed"), ("1", "mixed_survivor2"),
        ("2", "mixed"), ("2", "mixed_survivor2"),
    ]
    survivor = [c for c in cells if c["system"] == "mixed_survivor2"]
    assert all(c["diagnostic"] for c in survivor)
    assert all(not c["publication_eligible"] for c in survivor)


def test_joint_temporal_proof_is_exact_two_baseline_seed0_wave():
    cells = wave_cells("proof_joint_temporal_14b", seeds="0")
    assert len(cells) == 18
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["n"] for cell in cells} == {80}
    assert {cell["system"] for cell in cells} == {
        "mixed", "strict_pd", "joint_temporal"}
    assert "mixed_dvfs" not in {cell["system"] for cell in cells}
    assert {
        (cell["dataset"], cell["process"], str(cell["rate"]))
        for cell in cells
    } == {
        ("sharegpt", "gamma", "4"),
        ("sharegpt", "gamma", "6"),
        ("alpaca", "poisson", "4"),
        ("alpaca", "poisson", "8"),
        ("longbench", "poisson", "1"),
        ("longbench", "poisson", "2"),
    }
    with pytest.raises(ValueError):
        wave_cells("proof_joint_temporal_14b", seeds="0,1")


def test_regime_map_requires_both_mode_families():
    rows = []
    for rate, energies in (
            (1.0, (100.0, 80.0, 90.0, 85.0)),
            (2.0, (100.0, 95.0, 70.0, 65.0))):
        for system, energy in zip(
                SYSTEMS_PROOF_REGIME_14B, energies):
            rows.append(_row(
                "0", system, total_j=energy, attainment=0.98, rate=rate,
                model="14b", dataset="sharegpt", process="poisson",
                n=80))
    mapped, summary = build_mode_map(rows)
    assert len(mapped) == 2
    assert summary["mixed_favorable_count"] == 1
    assert summary["pd_favorable_count"] == 1
    assert summary["dynamic_possible"] is True


def test_gold_confirm_strict_telemetry_gate(tmp_path):
    path = tmp_path / "sched.csv"
    path.write_text(
        "execution_mode,telemetry_available,omega_hat\n"
        "strict-padg,0,0\n"
        "strict-padg,4,0\n",
        encoding="utf-8")
    ok, issues, info = validate_strict_sched(path)
    assert ok is True
    assert issues == []
    assert info["max_telemetry_available"] == 4
    path.write_text(
        "execution_mode,telemetry_available,omega_hat\n"
        "strict-padg,4,0.01\n",
        encoding="utf-8")
    ok, issues, _ = validate_strict_sched(path)
    assert ok is False
    assert "strict-omega-nonzero" in issues


def test_gold_confirm_verdict_requires_all_four_ci_rows():
    rows = []
    for model in ("14b", "32b"):
        for system in ("strict_pd", "pdblend"):
            rows.append({
                "model": model,
                "system": system,
                "baseline_system": (
                    "mixed" if system == "strict_pd" else "mixed_dvfs"),
                "comparison_kind": (
                    "component" if system == "strict_pd" else "target"),
                "n_seeds": 3,
                "ci_available": True,
                "publication_valid": not (
                    model == "32b" and system == "strict_pd"),
                "publication_reason": (
                    "attainment-lower-ci-below-minus-1pp"
                    if model == "32b" and system == "strict_pd"
                    else "passes-ci-gates"),
                "mean_energy_saving_pct": 8.0,
                "energy_saving_ci_lower_pct": 6.0,
                "mean_attainment_delta_pp": -0.2,
                "attainment_delta_ci_lower_pp": -0.8,
            })
    verdicts = verdict_from_aggregates(rows)
    assert len(verdicts) == 4
    assert sum(item["go"] for item in verdicts) == 3
    failed = next(item for item in verdicts if not item["go"])
    assert failed["model"] == "32b"
    assert failed["system"] == "strict_pd"


def test_seed_expansion_tags_and_legacy_path_compatibility(tmp_path):
    assert parse_seeds("0,1,1,02") == ("0", "1", "2")
    cells = wave_cells("confirm", seeds="0,1,2")
    assert len(cells) == 3 * 120
    assert {cell["seed"] for cell in cells} == {"0", "1", "2"}
    assert all(cell["include_seed"] for cell in cells)
    destinations = [dest_for(str(tmp_path), cell) for cell in cells]
    assert len(destinations) == len(set(destinations))
    assert all("_seed" in Path(path).name for path in destinations)
    assert cell_tag(
        "sharegpt", "14b", "mixed", "poisson", "2", 80
    ) == "sharegpt_14b_mixed_poisson_r2_n80"
    legacy = wave_cells("p1", seeds="0")
    assert all(not cell["include_seed"] for cell in legacy)
    assert all("_seed" not in Path(dest_for(str(tmp_path), cell)).name
               for cell in legacy)
    explicit = cells_from_line(
        "14b,sharegpt,2,80,5,0.15,poisson,mixed,0 "
        "14b,sharegpt,2,80,5,0.15,poisson,mixed,1",
        "p1", "0")
    assert all(cell["include_seed"] for cell in explicit)
    assert len({dest_for(str(tmp_path), cell) for cell in explicit}) == 2


def test_publication_reuse_is_forced_off():
    assert effective_reuse_policy("publication", "off") == "off"
    with pytest.raises(ValueError, match="publication"):
        effective_reuse_policy("publication", "hash")
    assert effective_reuse_policy("confirm", "hash") == "hash"


def test_dynamic_failure_and_ablation_waves_cover_required_axes():
    for wave in ("dynamic", "failure_envelope"):
        cells = wave_cells(wave)
        assert {cell["model"] for cell in cells} == {"14b", "32b"}
        assert {cell["process"] for cell in cells} == {
            "poisson", "gamma", "diurnal"}
        assert min(cell["n"] for cell in cells) >= 500
    ablations = wave_cells("ablation")
    assert {cell["system"] for cell in ablations} == {
        "mixed", "pdblend", "no_park", "no_dvfs", "no_roll"}
    assert {
        tuple(cell.get("ablation_flags") or ())
        for cell in ablations if cell["system"].startswith("no_")
    } == {("no_park",), ("no_dvfs",), ("no_roll",)}


def test_alias_mapping_keeps_diagnostics_honest():
    assert run_cell_system("ecospd_hybrid") == "ecospd"
    assert run_cell_system("strict_pd") == "ds_pd"
    assert run_cell_system("strict_pd_pp4tp1") == "ds_pd"
    assert run_cell_system("strict_pd_pp2tp2") == "ds_pd"
    assert run_cell_system("strict_pd_pp1tp4") == "ds_pd"
    assert run_cell_system("strict_pd_dvfs") == "ds_pd_dvfs"
    assert run_cell_system("strict_padg") == "strict_padg"
    assert run_cell_system("ds_pd_dvfs") == "ds_pd_dvfs"
    assert run_cell_system("pdblend_mpc") == "pdblend_mpc"
    runner = (ROOT / "script" / "bench" / "run_cell.sh").read_text(
        encoding="utf-8")
    assert "ds_pd|ds_pd_dvfs)" in runner
    assert "TP=8; N_REPLICA=1" in runner
    assert "Qwen2.5-7B-Instruct" in runner
    assert "缺 ds_placement" in runner
    assert "--ds-placement" in runner
    assert "tp_mixed" in runner
    assert "--mode boot-slo-layout" in runner
    assert "--mode boot-lookup" not in runner
    assert "--force-continuous --no-park" in runner
    assert "--goodput-gate" in runner
    assert runner.count("--goodput-gate") == 1
    assert 'PYTHONPATH="$PWD/src:$WS' in runner
    assert 'rm -f "$CD/power.csv"' in runner
    assert 'power.csv 采样过少' in runner
    # Select the legacy execution branch after the new explicit-runtime alias.
    pdblend = runner.rsplit("pdblend|ecospd)", 1)[1].split("assert_containers_alive")[0]
    ctrl = pdblend.split("start_proxy_retry")[-1]
    assert "--goodput-gate" in ctrl
    assert "--lookup" not in ctrl
    assert "start_ecospd_only_mixed --force-continuous" in runner
    helper = runner.split("start_ecospd_only_mixed()")[1].split(
        "CONTAINERS=()")[0]
    assert "--goodput-gate" not in helper
    assert 'runtime["phase_control"] = (' in runner
    sys_line = next(ln for ln in runner.splitlines() if ln.startswith("SYSTEM="))
    assert "dynamollm" in sys_line and "ecoserve" in sys_line
    for dead in ("joint_temporal", "pdblend_mpc", "pdblend_p2",
                 "mixed_survivor2", "no_park", "no_dvfs", "no_roll"):
        assert dead not in sys_line
        assert ("%s)" % dead) not in runner
        assert ("%s|" % dead) not in runner
    matrix_runner = (
        ROOT / "new-results" / "scripts"
        / "run_iso_load_asplos_matrix.sh"
    ).read_text(encoding="utf-8")
    assert "from asplos_matrix import run_cell_system" in matrix_runner
    assert (
        "frequency_smoke|pdblend_spatial_smoke|pdblend_spatial_rates_14b|"
        "pdblend_sat_probe_14b|pdblend_sat_n200_14b|pdblend_n200_rates_14b|"
        "pdblend_slo_retest_14b|pdblend_n200_slo_ladder_14b|"
        "pdblend_n200_two_base_14b|pdblend_n200_two_base_alpaca_14b|"
        "pdblend_n200_two_base_longbench_14b|pdblend_knee_shield_14b|"
        "pdblend_n200_lowload_14b|pdblend_n200_lookup_14b|"
        "pdblend_n200_rho_guard_14b|pdblend_n200_mixed_dvfs_14b|"
        "pdblend_n200_a9_14b|ds_placement_n200_14b|ds_placement_pp_smoke_14b|pdblend_n200_tp8_14b|pdblend_n200_sg_repro_base_14b|pdblend_n200_sg_scaleinst_14b|pdblend_n200_dir_fullnode_14b|pdblend_n200_jmin_probe_14b|pdblend_n200_jmin_sg_14b|pdblend_pub_figab_14b|"
        "proof_regime_14b|survivor_14b|"
        "proof_joint_temporal_14b|gold_confirm"
        in matrix_runner)


def test_exact_pairing_selects_lowest_j_candidate_and_ci_passes():
    rows = []
    for seed in range(3):
        rows.extend([
            _row(seed, "mixed", total_j=120.0, attainment=0.99),
            _row(seed, "mixed_dvfs", total_j=100.0, attainment=0.985),
            _row(seed, "strict_pd", total_j=95.0, attainment=0.982),
            _row(seed, "pdblend", total_j=90.0, attainment=0.981),
        ])
    pairs, rejected = pair_seed_rows(rows)
    assert not rejected
    assert len(pairs) == 9
    assert {
        (pair["system"], pair["comparison_kind"], pair["baseline_system"])
        for pair in pairs
    } == {
        ("mixed_dvfs", "component", "mixed"),
        ("strict_pd", "component", "mixed"),
        ("pdblend", "target", "strict_pd"),
    }
    result = next(
        row for row in aggregate_pairs(
            pairs, resamples=500, bootstrap_seed=7)
        if row["system"] == "pdblend")
    assert result["n_seeds"] == 3
    assert result["ci_available"] is True
    assert result["comparison_kind"] == "target"
    assert result["baseline_system"] == "strict_pd"
    assert result["energy_saving_ci_lower"] == pytest.approx(5.0 / 95.0)
    assert result["attainment_delta_ci_lower_pp"] == pytest.approx(-0.9)
    assert result["publication_valid"] is True


def test_pdblend_spatial_smoke_is_exact_four_cell_14b_wave():
    cells = wave_cells("pdblend_spatial_smoke", seeds="0")
    assert len(cells) == 4
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["n"] for cell in cells} == {80}
    assert [cell["system"] for cell in cells] == list(
        SYSTEMS_PDBLEND_SPATIAL_SMOKE)
    assert all(cell["dataset"] == "sharegpt" for cell in cells)
    assert all(str(cell["rate"]) == "2" for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_spatial_smoke", seeds="0,1")


def test_pdblend_spatial_rates_14b_is_exact_thirty_two_cell_wave():
    cells = wave_cells("pdblend_spatial_rates_14b", seeds="0")
    assert len(cells) == 32
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["n"] for cell in cells} == {80}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {str(cell["rate"]) for cell in cells} == set(
        RATES_PDBLEND_SPATIAL_14B)
    assert "2" not in {str(cell["rate"]) for cell in cells}
    assert [cell["system"] for cell in cells[:4]] == list(
        SYSTEMS_PDBLEND_SPATIAL_SMOKE)
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_spatial_rates_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_spatial_rates_14b", n=81, seeds="0")


def test_pdblend_sat_probe_14b_is_exact_four_cell_mixed_wave():
    cells = wave_cells("pdblend_sat_probe_14b", seeds="0")
    assert len(cells) == 4
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, n, "mixed") for rate, n in SAT_PROBE_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_sat_probe_14b", seeds="0,1")


def test_pdblend_sat_n200_14b_is_exact_eight_cell_mixed_wave():
    cells = wave_cells("pdblend_sat_n200_14b", seeds="0")
    assert len(cells) == 8
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, "mixed") for rate in RATES_SAT_N200_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_sat_n200_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_sat_n200_14b", n=80, seeds="0")


def test_pdblend_n200_rates_14b_is_exact_thirty_two_cell_wave():
    cells = wave_cells("pdblend_n200_rates_14b", seeds="0")
    assert len(cells) == 32
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["n"] for cell in cells} == {200}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {str(cell["rate"]) for cell in cells} == set(RATES_SAT_N200_14B)
    assert [cell["system"] for cell in cells[:4]] == list(
        SYSTEMS_PDBLEND_SPATIAL_SMOKE)
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_rates_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_rates_14b", n=80, seeds="0")


def test_pdblend_slo_retest_14b_is_exact_three_cell_pdblend_wave():
    cells = wave_cells("pdblend_slo_retest_14b", seeds="0")
    assert len(cells) == 3
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, "pdblend") for rate in RATES_SLO_RETEST_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_slo_retest_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_slo_retest_14b", n=80, seeds="0")


def test_pdblend_n200_slo_ladder_14b_is_exact_eight_cell_pdblend_wave():
    cells = wave_cells("pdblend_n200_slo_ladder_14b", seeds="0")
    assert len(cells) == 8
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, "pdblend") for rate in RATES_SAT_N200_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_slo_ladder_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_slo_ladder_14b", n=80, seeds="0")


def test_pdblend_n200_two_base_14b_is_exact_fifteen_cell_wave():
    cells = wave_cells("pdblend_n200_two_base_14b", seeds="0")
    assert len(cells) == 15
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, system)
        for rate in RATES_TWO_BASE_LOW_N200_14B
        for system in SYSTEMS_TWO_BASE_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_14b", n=80, seeds="0")


def test_pdblend_n200_two_base_alpaca_14b_is_exact_twenty_seven_cell_wave():
    cells = wave_cells("pdblend_n200_two_base_alpaca_14b", seeds="0")
    assert len(cells) == 27
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, system)
        for rate in RATES_TWO_BASE_ALPACA_N200_14B
        for system in SYSTEMS_TWO_BASE_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"alpaca"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {(cell["ttft"], cell["tpot"]) for cell in cells} == {(1.0, 0.10)}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_alpaca_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_alpaca_14b", n=80, seeds="0")


def test_pdblend_n200_two_base_longbench_14b_is_exact_twenty_seven_cell_wave():
    cells = wave_cells("pdblend_n200_two_base_longbench_14b", seeds="0")
    assert len(cells) == 27
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, system)
        for rate in RATES_TWO_BASE_LONGBENCH_N200_14B
        for system in SYSTEMS_TWO_BASE_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"longbench"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {(cell["ttft"], cell["tpot"]) for cell in cells} == {(15.0, 0.20)}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_longbench_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_two_base_longbench_14b", n=80, seeds="0")


def test_pdblend_knee_shield_14b_is_exact_three_cell_pdblend_wave():
    cells = wave_cells("pdblend_knee_shield_14b", seeds="0")
    assert len(cells) == 3
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, "pdblend") for rate in RATES_KNEE_SHIELD_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_knee_shield_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_knee_shield_14b", n=80, seeds="0")


def test_pdblend_n200_lowload_14b_is_exact_sixteen_cell_wave():
    cells = wave_cells("pdblend_n200_lowload_14b", seeds="0")
    assert len(cells) == 16
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, system)
        for rate in RATES_LOWLOAD_N200_14B
        for system in SYSTEMS_PDBLEND_SPATIAL_SMOKE]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_lowload_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_lowload_14b", n=80, seeds="0")


def test_pdblend_n200_lookup_14b_is_exact_thirty_nine_cell_wave():
    cells = wave_cells("pdblend_n200_lookup_14b", seeds="0")
    assert len(cells) == 39
    assert [(str(cell["rate"]), cell["n"], cell["system"]) for cell in cells] == [
        (rate, 200, system)
        for rate in RATES_LOOKUP_N200_14B
        for system in SYSTEMS_TWO_BASE_14B]
    assert {cell["model"] for cell in cells} == {"14b"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {(cell["ttft"], cell["tpot"]) for cell in cells} == {(5.0, 0.15)}
    assert "mixed_dvfs" not in {cell["system"] for cell in cells}
    assert all(cell.get("include_seed") for cell in cells)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_lookup_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_lookup_14b", n=80, seeds="0")


def test_pdblend_n200_rho_guard_14b_is_exact_thirty_six_cell_wave():
    cells = wave_cells("pdblend_n200_rho_guard_14b", seeds="0")
    want = (len(RATES_RHO_GUARD_SHAREGPT_14B)
            + len(RATES_RHO_GUARD_ALPACA_14B)
            + len(RATES_TWO_BASE_LONGBENCH_N200_14B)
            * len(SYSTEMS_TWO_BASE_14B))
    assert len(cells) == want == 36
    assert {cell["system"] for cell in cells if cell["dataset"] == "sharegpt"
            } == {"pdblend"}
    assert {cell["system"] for cell in cells if cell["dataset"] == "alpaca"
            } == {"pdblend"}
    assert {cell["system"] for cell in cells if cell["dataset"] == "longbench"
            } == set(SYSTEMS_TWO_BASE_14B)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_rho_guard_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_rho_guard_14b", n=80, seeds="0")


def test_pdblend_n200_a9_14b_is_exact_seven_cell_wave():
    cells = wave_cells("pdblend_n200_a9_14b", seeds="0")
    want = len(RATES_A9_SHAREGPT_14B) + len(RATES_A9_ALPACA_14B)
    assert len(cells) == want == 7
    assert {cell["system"] for cell in cells} == {"pdblend"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt", "alpaca"}
    assert {str(cell["rate"]) for cell in cells
            if cell["dataset"] == "sharegpt"} == set(RATES_A9_SHAREGPT_14B)
    assert {str(cell["rate"]) for cell in cells
            if cell["dataset"] == "alpaca"} == set(RATES_A9_ALPACA_14B)
    assert "longbench" not in {cell["dataset"] for cell in cells}
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_a9_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_a9_14b", n=80, seeds="0")


def test_ds_placement_n200_14b_is_exact_twelve_cell_wave():
    cells = wave_cells("ds_placement_n200_14b", seeds="0")
    want = len(RATES_DS_PLACEMENT_14B) * len(SYSTEMS_DS_PLACEMENT_14B)
    assert len(cells) == want == 12
    assert {cell["system"] for cell in cells} == set(SYSTEMS_DS_PLACEMENT_14B)
    assert {str(cell["rate"]) for cell in cells} == set(RATES_DS_PLACEMENT_14B)
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {run_cell_system(cell["system"]) for cell in cells} == {"ds_pd"}
    assert placement_for_system("strict_pd_pp4tp1")["pp_cross"] == 4
    assert placement_for_system("strict_pd_pp2tp2")["tp_prefill"] == 2
    assert placement_for_system("strict_pd_pp1tp4")["tp_prefill"] == 4
    assert placement_for_system("mixed") is None
    with pytest.raises(ValueError):
        wave_cells("ds_placement_n200_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("ds_placement_n200_14b", n=80, seeds="0")


def test_ds_placement_pp_smoke_14b_is_exact_three_cell_wave():
    cells = wave_cells("ds_placement_pp_smoke_14b", seeds="0")
    want = len(RATES_DS_PP_SMOKE_14B) * len(SYSTEMS_DS_PP_SMOKE_14B)
    assert len(cells) == want == 3
    assert {cell["system"] for cell in cells} == set(SYSTEMS_DS_PP_SMOKE_14B)
    assert {int(cell["n"]) for cell in cells} == {80}
    assert {run_cell_system(cell["system"]) for cell in cells} == {"ds_pd"}
    assert placement_for_system("strict_pd_tp2pp2")["pp_prefill"] == 2
    assert placement_for_system("strict_pd_tp1pp4")["pp_prefill"] == 4
    assert placement_for_system("strict_pd_x2_tp1pp2")["pp_cross"] == 2
    with pytest.raises(ValueError):
        wave_cells("ds_placement_pp_smoke_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("ds_placement_pp_smoke_14b", n=200, seeds="0")


def test_pdblend_n200_tp8_14b_is_exact_twenty_eight_cell_wave():
    cells = wave_cells("pdblend_n200_tp8_14b", seeds="0")
    want = ((len(RATES_TP8_SHAREGPT_14B) + len(RATES_TP8_ALPACA_14B))
            * len(SYSTEMS_TP8_14B))
    assert len(cells) == want == 28
    assert {cell["system"] for cell in cells} == set(SYSTEMS_TP8_14B)
    assert {cell["dataset"] for cell in cells} == {"sharegpt", "alpaca"}
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_tp8_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_tp8_14b", n=80, seeds="0")


def test_pdblend_n200_sg_repro_base_14b_is_exact_ten_cell_wave():
    cells = wave_cells("pdblend_n200_sg_repro_base_14b", seeds="0")
    want = len(RATES_REPRO_BASE_14B) * len(SYSTEMS_REPRO_BASE_14B)
    assert len(cells) == want == 10
    assert [cell["system"] for cell in cells] == [
        system for _rate in RATES_REPRO_BASE_14B
        for system in SYSTEMS_REPRO_BASE_14B]
    assert {cell["system"] for cell in cells} == {"dynamollm", "ecoserve"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {cell["process"] for cell in cells} == {"poisson"}
    assert {int(cell["n"]) for cell in cells} == {200}
    assert {str(cell["rate"]) for cell in cells} == set(RATES_REPRO_BASE_14B)
    assert {run_cell_system(cell["system"]) for cell in cells} == {
        "dynamollm", "ecoserve"}
    runner = (ROOT / "script" / "bench" / "run_cell.sh").read_text(
        encoding="utf-8")
    assert "dynamollm)" in runner
    assert "ecoserve)" in runner
    assert "TP=$TP × N_REPLICA=$N_REPLICA != 8,拒绝" in runner
    assert 'if [ "$SYSTEM" = dynamollm ]; then' in runner
    assert "8×TP1 MacroInstance" in runner
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_sg_repro_base_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_sg_repro_base_14b", n=80, seeds="0")


def test_pdblend_n200_sg_scaleinst_14b_is_exact_five_cell_wave():
    cells = wave_cells("pdblend_n200_sg_scaleinst_14b", seeds="0")
    assert len(cells) == len(RATES_REPRO_BASE_14B) == 5
    assert [cell["system"] for cell in cells] == ["pdblend"] * 5
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {int(cell["n"]) for cell in cells} == {200}
    assert {str(cell["rate"]) for cell in cells} == set(RATES_REPRO_BASE_14B)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_sg_scaleinst_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_sg_scaleinst_14b", n=80, seeds="0")


def test_pdblend_n200_dir_fullnode_14b_is_exact_four_column_wave():
    cells = wave_cells("pdblend_n200_dir_fullnode_14b", seeds="0")
    want = (len(RATES_DIR_SG_14B) * len(SYSTEMS_DIR_FULLNODE_14B)
            + len(RATES_DIR_ALPACA_14B) * len(SYSTEMS_DIR_FULLNODE_14B)
            + len(RATES_DIR_LB_14B) * len(SYSTEMS_DIR_FULLNODE_14B)
            + len(RATES_DIR_KNEE_14B) * len(SYSTEMS_DIR_KNEE_14B))
    assert len(cells) == want == 58
    assert {int(cell["n"]) for cell in cells} == {200}
    assert {cell["system"] for cell in cells} <= set(
        SYSTEMS_DIR_FULLNODE_14B)
    assert {cell["dataset"] for cell in cells} == {
        "sharegpt", "alpaca", "longbench"}
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_dir_fullnode_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_dir_fullnode_14b", n=80, seeds="0")


def test_pdblend_n200_jmin_sg_14b_is_five_system_full_rate_wave():
    probe = wave_cells("pdblend_n200_jmin_probe_14b", seeds="0")
    assert len(probe) == len(RATES_JMIN_PROBE_14B) * len(SYSTEMS_JMIN_SG_14B)
    assert {str(c["rate"]) for c in probe} == set(RATES_JMIN_PROBE_14B)
    cells = wave_cells("pdblend_n200_jmin_sg_14b", seeds="0")
    want = len(RATES_JMIN_SG_14B) * len(SYSTEMS_JMIN_SG_14B)
    assert len(cells) == want == 65
    assert {cell["system"] for cell in cells} == set(SYSTEMS_JMIN_SG_14B)
    assert {str(cell["rate"]) for cell in cells} == set(RATES_JMIN_SG_14B)
    assert {cell["dataset"] for cell in cells} == {"sharegpt"}
    assert {int(cell["n"]) for cell in cells} == {200}
    ecoserve = [c for c in cells if c["system"] == "ecoserve"]
    assert ecoserve
    assert all(c.get("publication_eligible") is False for c in ecoserve)
    assert all(c.get("implementation") == "ecoserve-macro-8x-tp1-vllm092"
               for c in ecoserve)
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_jmin_sg_14b", seeds="0,1")
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_jmin_sg_14b", n=80, seeds="0")


def test_pdblend_pub_figab_14b_requires_three_seeds_and_n500():
    cells = wave_cells("pdblend_pub_figab_14b", seeds="0,1,2")
    want = (len(RATES_PUB_FIGB_14B) + len(RATES_PUB_FIGA_14B)) * len(
        SYSTEMS_PUB_FIGAB_14B) * 3
    assert len(cells) == want == 48
    assert {int(cell["n"]) for cell in cells} == {500}
    assert {cell["seed"] for cell in cells} == {"0", "1", "2"}
    assert {cell["system"] for cell in cells} == set(SYSTEMS_PUB_FIGAB_14B)
    with pytest.raises(ValueError, match="3 seeds"):
        wave_cells("pdblend_pub_figab_14b", seeds="0")
    with pytest.raises(ValueError, match="n>=500"):
        wave_cells("pdblend_pub_figab_14b", n=200, seeds="0,1,2")


def test_pub_wave_blocked_until_dir_verdict_green(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "asplos_matrix.pdblend_root", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="blocked"):
        assert_dir_fullnode_ready()
    stamp = tmp_path / "new-results" / "results"
    stamp.mkdir(parents=True)
    bad = stamp / "DIR_FULLNODE_VERDICT.json"
    bad.write_text(json.dumps({
        "dir_green": False, "publication_ready": False,
        "holes": ["sharegpt r=6 want spatial-pd"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not green"):
        assert_dir_fullnode_ready()
    bad.write_text(json.dumps({
        "dir_green": True, "publication_ready": True, "holes": [],
    }), encoding="utf-8")
    assert_dir_fullnode_ready()


def test_pdblend_n200_mixed_dvfs_14b_is_exact_six_cell_wave():
    cells = wave_cells("pdblend_n200_mixed_dvfs_14b", seeds="0")
    want = (len(RATES_MIXED_DVFS_SHAREGPT_14B)
            + len(RATES_MIXED_DVFS_LONGBENCH_14B))
    assert len(cells) == want == 6
    assert {cell["system"] for cell in cells} == {"mixed_dvfs"}
    assert {cell["dataset"] for cell in cells} == {"sharegpt", "longbench"}
    with pytest.raises(ValueError):
        wave_cells("pdblend_n200_mixed_dvfs_14b", seeds="0,1")


def test_rho_guard_metrics_read_failed_stamp_when_csv_exists(tmp_path):
    from write_rho_guard_verdict import _metrics
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "FAILED").write_text("invalid_provenance\n", encoding="utf-8")
    (cell / "slo_summary.csv").write_text(
        "slo_attainment\n0.9\n", encoding="utf-8")
    (cell / "energy_summary.csv").write_text(
        "total_j\n1234.5\n", encoding="utf-8")
    att, joule = _metrics(str(cell))
    assert att == pytest.approx(0.9)
    assert joule == pytest.approx(1234.5)


def test_runner_allows_source_mismatch_on_rho_guard_and_mixed_dvfs():
    runner = (
        ROOT / "new-results" / "scripts"
        / "run_iso_load_asplos_matrix.sh"
    ).read_text(encoding="utf-8")
    assert 'WAVE" = pdblend_n200_rho_guard_14b' in runner
    assert 'WAVE" = pdblend_n200_mixed_dvfs_14b' in runner
    assert 'WAVE" = pdblend_n200_a9_14b' in runner
    assert 'WAVE" = pdblend_n200_dir_fullnode_14b' in runner
    assert 'WAVE" = pdblend_n200_jmin_sg_14b' in runner
    assert 'WAVE" = pdblend_pub_figab_14b' in runner
    assert "write_a9_verdict.py" in runner
    assert "write_dir_fullnode_verdict.py" in runner
    assert "assert_publication_ready.py" in runner
    assert "write_pub_figab_bounds.py" in runner
    assert "collect_ds_placement.py" in runner
    assert "placement_for_system" in runner
    assert "DONE_EXTRA=(--allow-source-mismatch)" in runner


def test_find_keep_cell_reposts_baselines_not_pdblend():
    assert any("sat-n200-14b" in rel for rel in KEEP_REL)
    assert any("n200-two-base-14b" in rel for rel in KEEP_REL)
    assert any("n200-sg-tp8-14b" in rel for rel in KEEP_REL)
    assert find_keep_cell(
        "sharegpt", "14b", "pdblend", "poisson", "2", 200, seed="0") == ""
    mixed = find_keep_cell(
        "sharegpt", "14b", "mixed", "poisson", "2", 200, seed="0")
    assert mixed
    assert mixed.endswith("sharegpt_14b_mixed_poisson_r2_n200_seed0")
    high = find_keep_cell(
        "sharegpt", "14b", "mixed", "poisson", "12", 200, seed="0")
    assert high
    assert high.endswith("sharegpt_14b_mixed_poisson_r12_n200_seed0")
    assert any(tok in high for tok in (
        "two-base-14b", "sat-n200-14b", "lookup-layout-14b"))


def test_prior_mixed_att_reads_frozen_n200_csv():
    att, src = prior_mixed_att("12", n=200, seed="0")
    assert src.endswith("slo_summary.csv")
    assert float(att) == pytest.approx(0.975)
    att20, _ = prior_mixed_att("20", n=200, seed="0")
    att32, _ = prior_mixed_att("32", n=200, seed="0")
    assert float(att20) == pytest.approx(0.970)
    assert float(att32) == pytest.approx(0.905)
    att2, src2 = prior_mixed_att("2", n=200, seed="0")
    assert "20260904-pdblend-n200-two-base-14b" in src2
    assert float(att2) == pytest.approx(0.975)
    att_a, src_a = prior_mixed_att(
        "12", n=200, seed="0", dataset="alpaca")
    assert "two-base-alpaca-14b" in src_a
    assert float(att_a) == pytest.approx(0.99)


def test_strict_pd_dvfs_is_not_an_energy_baseline():
    pairs, rejected = pair_seed_rows([
        _row(0, "mixed", total_j=100.0, attainment=0.99),
        _row(0, "mixed_dvfs", total_j=90.0, attainment=0.985),
        _row(0, "strict_pd", total_j=80.0, attainment=0.985),
        _row(0, "strict_pd_dvfs", total_j=50.0, attainment=0.985),
        _row(0, "pdblend", total_j=52.0, attainment=0.985),
    ])
    assert not rejected
    target = next(
        pair for pair in pairs
        if pair["comparison_kind"] == "target" and pair["system"] == "pdblend")
    assert target["baseline_system"] == "strict_pd"
    assert target["energy_baseline_candidates"] == "mixed_dvfs;strict_pd"


def test_energy_baseline_selection_is_per_seed_and_canonicalizes_ds_pd():
    rows = []
    for seed, dvfs_j, strict_j in ((0, 80.0, 90.0), (1, 95.0, 70.0)):
        rows.extend([
            _row(seed, "mixed", total_j=100.0, attainment=0.99),
            _row(seed, "mixed_dvfs", total_j=dvfs_j, attainment=0.985),
            _row(seed, "ds_pd", total_j=strict_j, attainment=0.985),
            _row(seed, "pdblend", total_j=60.0, attainment=0.985),
        ])
    pairs, rejected = pair_seed_rows(rows)
    assert not rejected
    targets = {
        pair["seed"]: pair for pair in pairs
        if pair["comparison_kind"] == "target"
    }
    assert targets["0"]["baseline_system"] == "mixed_dvfs"
    assert targets["1"]["baseline_system"] == "strict_pd"
    assert all(
        pair["energy_baseline_candidates"]
        == "mixed_dvfs;strict_pd"
        for pair in targets.values())
    assert {
        pair["system"] for pair in pairs
        if pair["comparison_kind"] == "component"
    } == {"mixed_dvfs", "strict_pd"}
    assert normalize_row(_row(0, "ds_pd"))["system"] == "strict_pd"


def test_invalid_strict_pd_uses_explicit_mixed_fallback():
    pairs, rejected = pair_seed_rows([
        _row(0, "mixed", total_j=100.0, attainment=0.99),
        _row(0, "strict_pd", total_j=60.0, attainment=0.95,
             validity="invalid_slo"),
        _row(0, "pdblend", total_j=70.0, attainment=0.985),
    ])
    assert not rejected
    target = next(
        pair for pair in pairs if pair["comparison_kind"] == "target")
    assert target["baseline_system"] == "mixed"
    assert target["energy_baseline_fallback"] is True
    assert target["fallback_reason"] == "no-valid-energy-candidate"
    assert target["baseline_selection_reason"].startswith("fallback-mixed")
    aggregate = aggregate_pairs([target], resamples=10)[0]
    assert aggregate["comparison_kind"] == "target"
    assert aggregate["baseline_system"] == "mixed"
    assert aggregate["baseline_system_by_seed"] == "0:mixed"
    assert aggregate["energy_baseline_fallback_count"] == 1
    strict_component = next(
        pair for pair in pairs if pair["system"] == "strict_pd")
    assert strict_component["comparison_kind"] == "component"
    assert strict_component["validity"] == "invalid_slo"


def test_target_slo_delta_and_ci_gate_are_anchored_to_mixed():
    rows = []
    for seed in range(3):
        rows.extend([
            _row(seed, "mixed", total_j=120.0, attainment=0.99),
            _row(seed, "mixed_dvfs", total_j=100.0, attainment=0.98),
            _row(seed, "pdblend", total_j=90.0, attainment=0.975),
        ])
    pairs, rejected = pair_seed_rows(rows)
    assert not rejected
    target_pairs = [
        pair for pair in pairs if pair["comparison_kind"] == "target"
    ]
    assert all(
        pair["attainment_delta"] == pytest.approx(-0.015)
        for pair in target_pairs)
    assert all(
        pair["attainment_delta_vs_energy_baseline"]
        == pytest.approx(-0.005)
        for pair in target_pairs)
    result = next(
        row for row in aggregate_pairs(target_pairs, resamples=200)
        if row["system"] == "pdblend")
    assert result["mean_attainment_delta_pp"] == pytest.approx(-1.5)
    assert result[
        "mean_attainment_delta_vs_energy_baseline_pp"
    ] == pytest.approx(-0.5)
    assert result["publication_valid"] is False
    assert "attainment-lower-ci" in result["publication_reason"]


def test_less_than_three_seeds_never_gets_ci_claim():
    rows = []
    for seed in range(2):
        rows.extend([
            _row(seed, "mixed", total_j=100.0),
            _row(seed, "pdblend", total_j=80.0),
        ])
    pairs, rejected = pair_seed_rows(rows)
    assert not rejected
    result = aggregate_pairs(pairs, resamples=100)[0]
    assert result["n_seeds"] == 2
    assert result["ci_available"] is False
    assert result["energy_saving_ci_lower"] is None
    assert result["attainment_delta_ci_lower"] is None
    assert result["evidence_status"] == "insufficient_seeds"
    assert result["publication_valid"] is False


def test_ci_gates_reject_energy_or_attainment_lower_bound():
    energy_rows = []
    attainment_rows = []
    for seed in range(3):
        energy_rows.extend([
            _row(seed, "mixed", total_j=100.0),
            _row(seed, "pdblend", total_j=97.0),
        ])
        attainment_rows.extend([
            _row(seed, "mixed", total_j=100.0, attainment=0.98),
            _row(seed, "pdblend", total_j=90.0, attainment=0.96),
        ])
    energy = aggregate_pairs(
        pair_seed_rows(energy_rows)[0], resamples=200)[0]
    attainment = aggregate_pairs(
        pair_seed_rows(attainment_rows)[0], resamples=200)[0]
    assert energy["publication_valid"] is False
    assert "energy-lower-ci" in energy["publication_reason"]
    assert attainment["publication_valid"] is False
    assert "attainment-lower-ci" in attainment["publication_reason"]


def test_bootstrap_is_deterministic():
    first = bootstrap_mean_ci([0.05, 0.08, 0.12, 0.2], 300, seed=11)
    second = bootstrap_mean_ci([0.05, 0.08, 0.12, 0.2], 300, seed=11)
    assert first == second


@pytest.mark.parametrize("field,value", [
    ("n", 501),
    ("rate", 2.1),
    ("gpu_count", 4),
    ("slo_tpot_s", 0.2),
])
def test_load_mismatch_is_rejected_not_cross_paired(field, value):
    baseline = _row(0, "mixed")
    target = _row(0, "pdblend")
    target[field] = value
    pairs, rejected = pair_seed_rows([baseline, target])
    assert pairs == []
    assert any(item["reason"] == "no-exact-valid-baseline"
               for item in rejected)


def test_report_only_oracle_is_excluded_from_online_pairing():
    rows = [
        _row(0, "mixed"),
        _row(0, "offline_oracle_report_only", total_j=1.0,
             report_only=True),
        _row(0, "pdblend", total_j=90.0),
    ]
    pairs, rejected = pair_seed_rows(rows)
    assert [pair["system"] for pair in pairs] == ["pdblend"]
    assert any(item["reason"] == "report-only-not-online"
               for item in rejected)
    reports, _ = replay_oracle(rows)
    assert reports[0]["record_type"] == "offline_oracle_replay"
    assert reports[0]["report_only"] is True
    assert reports[0]["online_system_result"] is False
    assert "system" not in reports[0]


def test_measured_envelope_uses_max_feasible_rate_and_separates_prediction():
    measured = [
        _row(0, "pdblend", rate=1.0, attainment=0.99),
        _row(0, "pdblend", rate=2.0, attainment=0.91),
        _row(0, "pdblend", rate=3.0, attainment=0.89),
    ]
    envelope = measured_envelope(measured, min_attainment=0.9)
    assert len(envelope) == 1
    assert envelope[0]["source_kind"] == "measured"
    assert envelope[0]["max_feasible_rate"] == pytest.approx(2.0)
    overlay = predicted_overlay([
        dict(model="14b", dataset="sharegpt", system="pdblend",
             predicted_max_feasible_rate=2.5, attainable=True)
    ])
    assert overlay[0]["source_kind"] == "predicted"
    assert overlay[0]["overlay_only"] is True
    assert "max_feasible_rate" not in overlay[0]


def test_ablation_aggregation_is_exact_and_seed_aware():
    rows = []
    for seed in range(3):
        rows.extend([
            _row(seed, "pdblend", total_j=80.0),
            _row(seed, "no_park", total_j=88.0, attainment=0.97),
        ])
    pairs = ablation_pairs(rows)
    result = aggregate_ablation(pairs, resamples=100)[0]
    assert result["ablation"] == "no_park"
    assert result["n_seeds"] == 3
    assert result["mean_energy_penalty_vs_full"] == pytest.approx(0.1)
    assert result["ci_available"] is True


def _provenance_for(cell, source_sha="source", image_id="image"):
    identity_cell = {
        "system": run_cell_system(cell["system"]),
        "model_key": cell["model"],
        "dataset": cell["dataset"],
        "process": cell["process"],
        "rate": float(cell["rate"]),
        "n": int(cell["n"]),
        "seed": str(cell["seed"]),
        "gpu_count": 8,
        "slo_ttft_s": float(cell["ttft"]),
        "slo_tpot_s": float(cell["tpot"]),
    }
    identity = {
        "source_tree_sha256": source_sha,
        "trace_sha256": "trace",
        "profile_sha256": "profile",
        "model_sha256": "model",
        "image_id": image_id,
        "cell": identity_cell,
    }
    record_id = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": 1,
        "record_id": record_id,
        "identity": identity,
    }


def test_strict_pd_plan_label_and_execution_provenance_are_distinct(tmp_path):
    plan = build_plan(
        "frequency_smoke", str(tmp_path), "5", reuse_policy="off")
    cell = next(
        item for item in plan["cells"] if item["system"] == "strict_pd")
    record = _provenance_for(cell)
    manifest = {
        "source": {"tree": {"sha256": "source"}},
        "images": [{"id": "image"}],
    }
    valid, reason = validate_cell_provenance(record, cell, manifest)
    assert valid is True
    assert reason == ""
    assert cell["system"] == "strict_pd"
    assert "_strict_pd_" in cell["tag"]
    assert record["identity"]["cell"]["system"] == "ds_pd"
    assert record["identity"]["cell"]["seed"] == "5"


def test_gold_campaign_accepts_spatial_strict_without_sched_and_rejects_swap(
        tmp_path):
    root = tmp_path / "gold"
    root.mkdir()
    plan = build_plan(
        "gold_confirm", str(root), "0,1,2", reuse_policy="off")
    for cell in plan["cells"]:
        cell["status"] = "done"
        destination = Path(cell["destination"])
        destination.mkdir(parents=True)
        (destination / "energy_summary.csv").write_text(
            "total_j\n100\n", encoding="utf-8")
        (destination / "slo_summary.csv").write_text(
            "slo_attainment\n0.99\n", encoding="utf-8")
        (destination / "provenance.json").write_text(
            json.dumps(_provenance_for(cell)), encoding="utf-8")
    manifest = dict(plan)
    manifest.update(
        source={"tree": {"sha256": "source"}},
        images=[{"id": "image"}],
    )
    manifest_path = root / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    issues, cells, structural = validate_campaign(root)
    assert issues == []
    assert len(cells) == 24
    assert structural["strict_telemetry"] == {}

    wrong = json.loads(json.dumps(manifest))
    swapped = next(
        cell for cell in wrong["cells"]
        if cell["system"] == "strict_pd")
    swapped["system"] = "strict_padg"
    manifest_path.write_text(json.dumps(wrong), encoding="utf-8")
    wrong_issues, _, _ = validate_campaign(root)
    assert "wrong-formal-composition" in wrong_issues


def test_manifest_status_transitions_and_fail_closed_provenance(tmp_path):
    plan = build_plan("smoke", str(tmp_path), "0", reuse_policy="off")
    cell = plan["cells"][0]
    destination = Path(cell["destination"])
    destination.mkdir(parents=True)
    (destination / "energy_summary.csv").write_text(
        "total_j\n100\n", encoding="utf-8")
    (destination / "provenance.json").write_text(
        json.dumps(_provenance_for(cell)), encoding="utf-8")
    manifest_path = tmp_path / "matrix_manifest.json"
    manifest = {
        "source": {"tree": {"sha256": "source"}},
        "images": [{"id": "image"}],
        "cells": [cell],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    update_manifest_status(str(manifest_path), cell["tag"], "running")
    updated = update_manifest_status(
        str(manifest_path), cell["tag"], "done",
        provenance_path=str(destination / "provenance.json"))
    assert updated["status"] == "done"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["status_counts"]["done"] == 1
    assert len(saved["cells"][0]["provenance"]["record_id"]) == 64

    bad_plan = build_plan("smoke", str(tmp_path / "bad"), "0")
    bad_cell = bad_plan["cells"][0]
    bad_dest = Path(bad_cell["destination"])
    bad_dest.mkdir(parents=True)
    (bad_dest / "energy_summary.csv").write_text(
        "total_j\n100\n", encoding="utf-8")
    bad_record = _provenance_for(bad_cell, source_sha="wrong")
    bad_prov = bad_dest / "provenance.json"
    bad_prov.write_text(json.dumps(bad_record), encoding="utf-8")
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({
        "source": {"tree": {"sha256": "source"}},
        "images": [{"id": "image"}],
        "cells": [bad_cell],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="source-provenance-mismatch"):
        update_manifest_status(
            str(bad_manifest), bad_cell["tag"], "done",
            provenance_path=str(bad_prov))
    assert json.loads(
        bad_manifest.read_text(encoding="utf-8")
    )["cells"][0]["status"] == "pending"


def test_manifest_reused_status_requires_and_records_source(tmp_path):
    plan = build_plan("smoke", str(tmp_path / "out"), "0")
    cell = plan["cells"][1]
    destination = Path(cell["destination"])
    destination.mkdir(parents=True)
    (destination / "energy_summary.csv").write_text(
        "total_j\n100\n", encoding="utf-8")
    provenance = destination / "provenance.json"
    provenance.write_text(
        json.dumps(_provenance_for(cell)), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "source": {"tree": {"sha256": "source"}},
        "images": [{"id": "image"}],
        "cells": [cell],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="reuse_source"):
        update_manifest_status(
            str(manifest_path), cell["tag"], "reused",
            provenance_path=str(provenance))
    reused_from = tmp_path / "prior-campaign" / cell["tag"]
    updated = update_manifest_status(
        str(manifest_path), cell["tag"], "reused",
        provenance_path=str(provenance), reuse_source=str(reused_from))
    assert updated["status"] == "reused"
    assert updated["reuse_source"] == str(reused_from.resolve())


def test_matrix_runner_prevents_concurrent_campaign_writers():
    runner = (
        ROOT / "new-results" / "scripts"
        / "run_iso_load_asplos_matrix.sh"
    ).read_text(encoding="utf-8")
    assert 'exec 9>"$OUT/.matrix.lock"' in runner
    assert "flock -n 9" in runner
