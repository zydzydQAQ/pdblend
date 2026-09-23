import json
from pathlib import Path

import pytest

from pdblend_baselines.cpu import MODEL_GEOMETRY, run_contracts, topology_report, verify_migration
from pdblend_baselines.dynamollm.predictor import calibration_examples, output_class, split_examples
from .independence_contract import assert_independent_imports


def test_migrated_core_sources_are_exact_and_independent():
    assert verify_migration()["verified_files"] >= 20
    import pdblend_baselines
    root = Path(pdblend_baselines.__file__).parent
    for path in root.rglob("*.py"):
        assert_independent_imports(path.read_text(), path.relative_to(root).as_posix())


@pytest.mark.parametrize("module,symbol", [
    ("pdblend.control.planner", "Plan"),
    ("pdblend.control.policies", "get_policy"),
    ("pdblend.profile.model", "PerfModel"),
    ("pdblend.profile.decode_fit", "fit_candidate"),
    ("pdblend.profile.profiler", "Profiler"),
])
def test_measurement_permission_cannot_import_policy_or_fitted_models(module, symbol):
    for path in ("dynamollm/policy.py", "dynamollm/profile_epochs.py", "distserve/stage_collect.py"):
        with pytest.raises(AssertionError):
            assert_independent_imports(f"from {module} import {symbol}", path)


def test_measurement_permissions_cannot_expand_by_wildcard_or_importer():
    with pytest.raises(AssertionError):
        assert_independent_imports("from pdblend.profile.sampling_epochs import *", "dynamollm/profile_epochs.py")
    with pytest.raises(AssertionError):
        assert_independent_imports("import pdblend.profile.sampling_epochs as p", "dynamollm/profile_epochs.py")
    with pytest.raises(AssertionError):
        assert_independent_imports("from pdblend.engine.launcher import Fleet", "dynamollm/policy.py")


@pytest.mark.parametrize("model", MODEL_GEOMETRY)
def test_all_three_model_search_spaces_obey_geometry_and_eight_gpu_budget(model):
    report = topology_report(model)
    g = MODEL_GEOMETRY[model]
    assert report["configuration_count"] > 0
    for cross, tp, pp, td, pd in report["configurations"]:
        assert tp in g["allowed_tps"] and td in g["allowed_tps"]
        assert g["layers"] % (cross * pp) == g["layers"] % (cross * pd) == 0
        assert g["attention_heads"] % tp == g["attention_heads"] % td == 0
        assert cross * (tp * pp + td * pd) <= 8
    assert report["hardware_qualified"] is False
    if model == "7b":
        assert report["configuration_count"] == 38


def test_predictor_calibration_split_cannot_leak_evaluation_or_duplicate_prompts():
    rows = [{"text": f"prompt-{i}", "output_tokens": 99 + i} for i in range(10)]
    examples = calibration_examples({"calibration": rows + [dict(rows[0], output_tokens=400)]})
    train, holdout = split_examples(examples)
    assert {text for text, _ in train}.isdisjoint(text for text, _ in holdout)
    assert len(train) + len(holdout) == 11
    assert [output_class(n) for n in (99, 100, 349, 350)] == [0, 1, 1, 2]
    with pytest.raises(ValueError, match="non-calibration"):
        calibration_examples({"calibration": [dict(rows[0], split="evaluation")]})


def test_cpu_contract_entry_runs_simpy_milp_and_keeps_qualification_false():
    report = run_contracts()
    assert report["status"] == "passed"
    assert {m["model"] for m in report["models"]} == {"7b", "14b", "32b"}
    assert report["measurement"] == "synthetic_cpu_contract"
    assert report["hardware_qualified"] is False
    assert report["energy_comparable"] is False
    assert report["contracts"]["dynamollm"]["periods_due_at_1800"] == (
        "ScaleInst", "ScaleShard", "ScaleFreq")
    json.dumps(report)
