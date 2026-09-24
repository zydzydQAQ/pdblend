from dataclasses import replace
import json

import pytest

from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
from pdblend.planner.transitions import KIND, TransitionCatalog, build_catalog_entry, identity, signature
from synthetic import fc, synthetic_model


def receipts(tmp_path, *, paired=True, omit_publish=False):
    model = synthetic_model()
    source = Plan({"M": 1}, 2520, 2520, 2520, 0, 100, 1, .1)
    target = replace(source, f_M=1800)
    phases = [dict(instance="one", operation=operation, started_s=start, finished_s=end,
                   gpus=[0], status="passed", energy_j=100)
              for operation, start, end in [("clock_set", 1, 2), ("validate", 1.5, 3),
                                            ("publish", 3, 4)] if not (omit_publish and operation == "publish")]
    binding = dict(identity=identity(model), source=signature(source), target=signature(target))
    raw = dict(schema=1, phases=phases, energy_complete=True, measured_union_energy_j=150,
               incremental_energy_j=None, power_source="synthetic-meter", binding=binding, target_verified=True)
    if paired:
        raw["pairing"] = dict(pair_id="trial", workload_sha256="same-trace", role="transition", uncertainty_j=2)
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps(raw))
    reference = None
    if paired:
        reference = tmp_path / "control.json"
        control = dict(raw, measured_union_energy_j=120,
                       pairing=dict(raw["pairing"], role="counterfactual", uncertainty_j=3))
        reference.write_text(json.dumps(control))
    return model, source, target, measured, reference


def load(tmp_path, model, entry, *, qualified_only=False):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(dict(kind=KIND, identity=identity(model), entries=[entry])))
    return TransitionCatalog.load(path, model=model, qualified_only=qualified_only)


def test_paired_measurement_uses_extra_energy_plus_uncertainty_and_critical_path(tmp_path):
    model, source, target, raw, reference = receipts(tmp_path)
    entry = build_catalog_entry(raw, source, target, model, counterfactual_path=reference)
    assert entry["qualified"]
    assert entry["critical_path_s"] == 3  # overlapping phases are not added
    assert entry["incremental_energy_j"] == 30
    assert entry["planning_energy_j"] == 35
    catalog = load(tmp_path, model, entry)
    planner = PoolPlanner(model, PlannerConfig(1, SLO(5, .15), transition_estimator=catalog))
    assert planner.switch_energy_j(source, target) == 35
    assert target.detail["transition_cost"]["qualified"]
    assert len(target.detail["transition_cost"]["evidence"]) == 2


def test_gross_energy_is_never_used_as_incremental_penalty(tmp_path):
    model, source, target, raw, _ = receipts(tmp_path, paired=False)
    entry = build_catalog_entry(raw, source, target, model)
    assert entry["incremental_energy_j"] is None and not entry["qualified"]
    planner = PoolPlanner(model, PlannerConfig(1, SLO(5, .15), transition_estimator=load(tmp_path, model, entry)))
    assert planner.switch_energy_j(source, target) == model.freq_switch_s * model.static_power_w("active_idle", 2520)
    assert target.detail["transition_cost"]["fallback"] == "legacy_wake_and_clock_estimate"
    assert not target.detail["transition_cost"]["qualified"]


def test_missing_phase_and_mismatched_counterfactual_cannot_qualify(tmp_path):
    model, source, target, raw, reference = receipts(tmp_path, omit_publish=True)
    entry = build_catalog_entry(raw, source, target, model, counterfactual_path=reference)
    assert not entry["qualified"] and any("publish" in reason for reason in entry["reasons"])
    data = json.loads(reference.read_text())
    data["pairing"]["workload_sha256"] = "other-workload"
    reference.write_text(json.dumps(data))
    entry = build_catalog_entry(raw, source, target, model, counterfactual_path=reference)
    assert entry["incremental_energy_j"] is None


def test_catalog_reaudits_evidence_and_binds_exact_profile_and_transition(tmp_path):
    model, source, target, raw, reference = receipts(tmp_path)
    entry = build_catalog_entry(raw, source, target, model, counterfactual_path=reference)
    catalog = load(tmp_path, model, entry)
    assert catalog(source, replace(target, f_M=1500))["incremental_energy_j"] is None
    assert catalog(source, replace(target, profile_key='{"other":"profile"}'))["incremental_energy_j"] is None
    with pytest.raises(ValueError, match="identity"):
        load(tmp_path, replace(model, tp=2), entry)
    forged = dict(entry, planning_energy_j=1)
    with pytest.raises(ValueError, match="reproduce"):
        load(tmp_path, model, forged)
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        load(tmp_path, model, entry)


def test_qualified_only_skips_missing_transition_but_allows_staying_put(tmp_path):
    model, source, target, raw, _ = receipts(tmp_path, paired=False)
    entry = build_catalog_entry(raw, source, target, model)
    planner = PoolPlanner(model, PlannerConfig(1, SLO(5, .15),
        transition_estimator=load(tmp_path, model, entry, qualified_only=True)))
    assert planner.switch_energy_j(source, target) == float("inf")
    assert planner.switch_energy_j(source, replace(source)) == 0
    kept = planner.plan(fc(.1), source)
    assert kept.f_M == source.f_M
