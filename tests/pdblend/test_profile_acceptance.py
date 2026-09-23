import json
from pathlib import Path

import pytest

from pdblend.profile.acceptance import (m2_gate, relative_error, validate_parallel_interference,
                                        validate_parallel_layout)
from pdblend.profile.merge import merge_raw


def test_relative_error_uses_observed_denominator():
    assert relative_error(11, 10) == pytest.approx(.1)
    assert relative_error(9, 10) == pytest.approx(.1)


def test_m2_gate_requires_active_seed_and_tails():
    rows = [dict(seed=701, complete=True, joint_slo_rate=1.0, ttft_p99=1.0,
                 tpot_p99=.1, power_error=.01)]
    assert m2_gate(rows)["passed"]
    rows[0]["tpot_p99"] = .151
    assert not m2_gate(rows)["passed"]


def test_merge_rejects_overlapping_frequency_shards(tmp_path):
    base = dict(schema=2, model="m", tp=1, freqs=[2100], kv_capacity_tokens=1,
                kv_bytes_per_token=1, config=dict(decode_repeats=3, decode_settle_s=2,
                decode_measure_s=5, decode_batches=[1]), environment=dict(
                image_digest="i", source_hash="s", vllm="v", torch="t", cuda="c",
                python="p", gpu_uuids=["GPU-a"]), prefill=[], decode=[], mixed=[],
                static={}, transfer=[], freq_switch_s=[])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(base)); base["environment"]["gpu_uuids"] = ["GPU-b"]; b.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="overlapping"):
        merge_raw([a, b], tmp_path / "out")


def test_parallel_layout_validation_checks_ownership_and_concurrency():
    layout = {
        "instances": [{"instance_id": "i0", "gpus": [2, 3], "tp": 2, "pp": 1}],
        "gpus": [2, 3],
    }
    raw = {
        "parallel_layout": layout,
        "concurrency": {"decode_max_batch": 4, "mixed_background_max_batch": 4,
                         "prefill_inflight": 1, "transfer_inflight": 1},
        "prefill": [{"concurrency": 1, "parallel_layout": layout}],
        "decode": [{"batch": 4, "concurrency": 4, "parallel_layout": layout}],
        "mixed": [{"batch": 4, "concurrency": 4, "parallel_layout": layout}],
        "transfer": [{"concurrency": 1, "parallel_layout": layout}],
    }
    assert validate_parallel_layout(raw)["passed"]
    raw["decode"][0]["concurrency"] = 1
    result = validate_parallel_layout(raw)
    assert not result["passed"]
    assert any("decode[0]" in failure for failure in result["failures"])


def test_legacy_profile_without_parallel_metadata_is_skipped():
    result = validate_parallel_layout({"decode": [{"batch": 8}]})
    assert result == {"passed": True, "skipped": True, "failures": []}


def test_parallel_interference_acceptance_requires_bounded_receipt(tmp_path):
    evidence = tmp_path / "parallel.json"
    evidence.write_text('{"samples": true}\n')
    receipt = {
        "complete": True,
        "point": {"freq_mhz": 2100, "batch": 8, "context_tokens": 1024},
        "isolated": [{"step_seconds": 1.0, "power_w": 100.0}],
        "parallel": [{"step_seconds": 1.04, "power_w": 96.0}],
        "validation": {"passed": True},
        "samples_file": evidence.name,
        "samples_sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
    }
    raw = {"measured_mode": "parallel", "parallel_interference": receipt}
    result = validate_parallel_interference(raw, tmp_path)
    assert result["passed"] and not result["formal_eligible"]
    receipt["parallel"][0]["step_seconds"] = 1.2
    assert not validate_parallel_interference(raw, tmp_path)["passed"]


def test_serial_fallback_receipt_is_explicitly_reasoned():
    assert validate_parallel_interference({
        "measured_mode": "serial_fallback",
        "parallel_interference": {"complete": False, "error": "timing exceeded 5%"},
    })["passed"]


def test_external_receipt_requires_coordinator_marked_cross_job_evidence(tmp_path):
    evidence = tmp_path / "external.json"
    evidence.write_text(json.dumps({"complete": True, "cross_job": False,
                                    "cohort_id": None, "members": ["a", "b"],
                                    "overlapping_windows": True}))
    receipt = {"complete": True, "measured_mode": "parallel",
               "point": {"freq_mhz": 2100, "batch": 8, "context_tokens": 1024},
               "isolated": [{"step_seconds": 1, "power_w": 100}],
               "parallel": [{"step_seconds": 1.01, "power_w": 101}],
               "validation": {"passed": True}, "samples_file": evidence.name,
               "samples_sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest()}
    result = validate_parallel_interference({"external_interference": receipt}, tmp_path)
    assert not result["passed"]
    assert not result["formal_eligible"]


def test_full_host_receipt_can_qualify_without_peer_jobs(tmp_path):
    evidence = tmp_path / "local.json"
    evidence.write_text(json.dumps({"samples": True}))
    lease = tmp_path / "lease.json"
    lease.write_text(json.dumps({"allocated": [f"GPU-{i}" for i in range(8)]}))
    layout = {"complete": True, "measured_mode": "parallel",
              "point": {"freq_mhz": 2100, "batch": 8, "context_tokens": 1024},
              "isolated": [{"step_seconds": 1, "power_w": 100}],
              "parallel": [{"step_seconds": 1.01, "power_w": 101}],
              "validation": {"passed": True}, "samples_file": evidence.name,
              "samples_sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest()}
    raw = {"parallel_interference": layout,
           "concurrency_environment": {
               "physical_gpu_uuids": [f"GPU-{i}" for i in range(8)],
               "allocated_gpu_uuids": [f"GPU-{i}" for i in range(8)],
               "peer_jobs": [], "lease_manifest_file": lease.name,
               "lease_manifest_sha256": __import__("hashlib").sha256(lease.read_bytes()).hexdigest()}}
    result = validate_parallel_interference(raw, tmp_path)
    assert result["passed"] and result["formal_eligible"]


def test_full_host_gate_accepts_worker_concurrency_environment_schema(tmp_path):
    evidence = tmp_path / "local.json"
    evidence.write_text(json.dumps({"samples": True}))
    worker_env = tmp_path / "concurrency-environment.json"
    worker_env.write_text(json.dumps({
        "schema": 1,
        "allocated_gpu_uuids": [f"GPU-{i}" for i in range(8)],
        "inventory": [{"uuid": f"GPU-{i}", "index": i, "pids": [100 + i]} for i in range(8)],
        "peer_snapshots": [{"at_s": 1.0, "peers": []}],
    }))
    receipt = {"complete": True, "measured_mode": "parallel",
               "point": {"freq_mhz": 2100, "batch": 8, "context_tokens": 1024},
               "isolated": [{"step_seconds": 1, "power_w": 100}],
               "parallel": [{"step_seconds": 1.01, "power_w": 101}],
               "validation": {"passed": True}, "samples_file": evidence.name,
               "samples_sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest()}
    env_hash = __import__("hashlib").sha256(worker_env.read_bytes()).hexdigest()
    raw = {"parallel_interference": receipt,
           "concurrency_environment": {
               "samples_file": worker_env.name, "samples_sha256": env_hash,
               **json.loads(worker_env.read_text())}}
    result = validate_parallel_interference(raw, tmp_path)
    assert result["passed"] and result["formal_eligible"]


def test_partial_host_layout_is_safe_and_diagnostic_only(tmp_path):
    raw = {"parallel_interference": {
        "complete": True, "measured_mode": "parallel",
        "point": {"freq_mhz": 2100, "batch": 8, "context_tokens": 1024},
        "isolated": [{"step_seconds": 1, "power_w": 100}],
        "parallel": [{"step_seconds": 1.01, "power_w": 101}],
        "validation": {"passed": True}},
        "concurrency_environment": {
            "physical_gpu_uuids": ["GPU-0"] * 4,
            "allocated_gpu_uuids": ["GPU-0"] * 4,
            "peer_jobs": []}}
    result = validate_parallel_interference(raw, tmp_path)
    assert not result["formal_eligible"]
