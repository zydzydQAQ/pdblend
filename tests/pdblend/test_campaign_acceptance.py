from pdblend.bench.campaign_acceptance import (
    accept_campaign,
    pair_records,
    validate_evidence,
)
from pdblend.profile.identity import (
    ProfileKey,
    evidence_binding,
    require_profile_provenance,
    sha256_value,
)
import pytest


def evidence(system="a", seed=701, *, model="m", trace="trace", slo=1.0,
             tp=1, pp=1, profile="profile-a", status="complete"):
    identity = dict(system=system, model=model, tp=tp, pp=pp, role="mixed", profile_sha256=profile,
                    dataset="sharegpt", rate=1.0, duration=300.0,
                    trace_sha256=trace, corpus_sha256="corpus", image="image",
                    source_sha256="source", hardware="gpu-set", clock_protocol="clock",
                    energy_protocol="nvml-board-v1", engine_revision="engine",
                    vllm="vllm", torch="torch", cuda="cuda", seed=seed,
                    slo_target={"ttft_s": 5.0, "tpot_s": .15})
    return dict(status=status, returncode=0, inputs_unchanged=True,
                identity=identity, identity_sha256=sha256_value(identity),
                metrics={"joint_slo_rate": slo, "success_rate": slo, "j_per_token": 1.0})


def test_pair_allows_independent_tp_and_profile_but_requires_common_identity():
    left = evidence("a", tp=1, profile="profile-a")
    right = evidence("b", tp=2, profile="profile-b")
    right["metrics"]["j_per_token"] = 2.0
    assert pair_records(left, right)["status"] == "passed"
    right["identity"]["seed"] = 1701
    assert pair_records(left, right)["status"] == "inconclusive"


def test_native_pair_binds_shared_meter_and_engine_but_allows_independent_policy_source():
    left, right = evidence('pdblend'), evidence('distserve', tp=2)
    right['metrics']['j_per_token'] = 2.
    for row in (left,right):
        row['identity'].update(identity_protocol='independent_native_v2',
            source_sha256='policy-'+row['identity']['system'],
            runtime_source_sha256='runtime', measurement_source_sha256='sampler',
            model_hash='same-weights', tokenizer_hash='same-tokenizer')
        row['identity_sha256'] = sha256_value(row['identity'])
    assert pair_records(left,right)['status'] == 'passed'
    right['identity']['measurement_source_sha256'] = 'another-meter'
    right['identity_sha256'] = sha256_value(right['identity'])
    assert pair_records(left,right)['status'] == 'inconclusive'


def test_pair_rejects_runtime_or_slo_target_mismatch():
    left = evidence("a")
    right = evidence("b", tp=2, profile="profile-b")
    right["metrics"]["j_per_token"] = 2.0
    right["identity"]["vllm"] = "different-runtime"
    right["identity_sha256"] = sha256_value(right["identity"])
    assert pair_records(left, right)["status"] == "inconclusive"
    right["identity"]["vllm"] = "vllm"
    right["identity"]["slo_target"] = {"ttft_s": 2.0, "tpot_s": .15}
    right["identity_sha256"] = sha256_value(right["identity"])
    assert pair_records(left, right)["status"] == "inconclusive"


def test_bad_slo_is_measured_failure_and_missing_execution_is_inconclusive():
    bad = validate_evidence(evidence(slo=.2))
    assert bad["status"] == "measured_failure"
    missing = evidence(status=None)
    assert validate_evidence(missing)["status"] == "inconclusive"


def test_campaign_groups_by_workload_and_requires_paired_seed_701():
    rows = [evidence(system=system, seed=seed, tp=1 if system == "a" else 2,
                     profile=f"p-{system}") for system in ("a", "b") for seed in (701,)]
    rows[-1]["identity"]["trace_sha256"] = "different-workload"
    result = accept_campaign(rows, systems=("a", "b"))
    assert result["status"] == "inconclusive"
    assert result["formal_eligible"] is False


def test_campaign_reports_slo_failure_separately_from_measurement_completeness():
    rows = [evidence(system=system, seed=seed, slo=.5 if system == "b" else 1.0,
                     tp=1 if system == "a" else 2, profile=f"p-{system}")
            for system in ("a", "b") for seed in (701,)]
    result = accept_campaign(rows, systems=("a", "b"))
    assert result["status"] == "failed"
    assert result["complete"] is True
    assert all(row["status"] != "inconclusive" for row in result["rows"])


def test_campaign_single_seed_is_complete_without_historical_seeds():
    rows = [evidence(system=system) for system in ("a", "b")]
    result = accept_campaign(rows, systems=("a", "b"))
    assert result["status"] == "passed"
    assert result["complete"] is True
    assert result["formal_eligible"] is True
    assert result["seeds"] == [701]
    assert result["single_seed"] is True
    assert result["seed_policy"] == "single_seed_701"


def test_campaign_single_seed_still_requires_every_system_once():
    one = evidence(system="a")
    for rows in ([one], [one, one, evidence(system="b")]):
        result = accept_campaign(rows, systems=("a", "b"))
        assert result["status"] == "inconclusive"
        assert result["formal_eligible"] is False


def test_campaign_rejects_other_seed_as_active_evidence():
    rows = [evidence(system=system, seed=1701) for system in ("a", "b")]
    result = accept_campaign(rows, systems=("a", "b"))
    assert result["status"] == "inconclusive"
    assert all("seed701:missing_evidence" in reason for reason in result["reasons"])
    assert validate_evidence(rows[0])["status"] == "inconclusive"


def test_campaign_rejects_conflicting_seed_marker():
    row = evidence()
    row["seed_policy"] = "three_seed"
    assert "invalid_seed_policy_marker" in validate_evidence(row)["reasons"]


def test_strict_profile_binds_environment_and_separate_sample_holdout():
    key = ProfileKey("pdblend", "m", "engine", "hw", 2, role="decode")
    bindings = {
        "sample": evidence_binding({"sample.raw": "sample-file-sha"}, kind="sample"),
        "holdout": evidence_binding({"holdout.raw": "holdout-file-sha"}, kind="holdout"),
    }
    raw = {
        "profile_key": key.as_dict(),
        "model_hash": "model-hash",
        "tokenizer_hash": "tokenizer-hash",
        "verification_receipt": "model-verification.json",
        "provenance": {"environment": {
            "image_digest": "image", "source_hash": "source", "vllm": "vllm",
            "torch": "torch", "cuda": "cuda", "gpu_uuids": ["GPU-a"],
        }, "hardware": {"hardware_id": "hw", "meter_scope": "gpu_board"}},
        "evidence_bindings": bindings,
    }
    raw["identity_sha256"] = sha256_value(raw)
    assert require_profile_provenance(raw, expected=key)["system"] == "pdblend"
    raw["evidence_bindings"]["holdout"]["files"]["holdout.raw"] = "tampered"
    try:
        require_profile_provenance(raw, expected=key)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("tampered holdout was accepted")


def test_nvml_board_power_cannot_be_claimed_as_whole_host():
    row = evidence()
    row["identity"]["meter_scope"] = "whole_host"
    assert validate_evidence(row)["status"] == "inconclusive"


@pytest.mark.parametrize(("layer", "field", "reason"), [
    ("record", "formal_eligible", "development_only"),
    ("record", "energy_comparable", "energy_not_comparable"),
    ("record", "hardware_executed", "no_gpu_execution"),
    ("summary", "formal_eligible", "development_only"),
    ("summary", "energy_comparable", "energy_not_comparable"),
    ("summary", "hardware_executed", "no_gpu_execution"),
])
def test_explicit_nonformal_execution_flags_are_inconclusive(layer, field, reason):
    row = evidence()
    row.setdefault("summary", {})[field] = False if layer == "summary" else row.get(field)
    if layer == "record":
        row[field] = False
    checked = validate_evidence(row)
    assert checked["status"] == "inconclusive"
    assert reason in checked["reasons"]
    result = accept_campaign([row, evidence(system="b")], systems=("a", "b"))
    assert result["formal_eligible"] is False
