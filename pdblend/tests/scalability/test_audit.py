import csv
import hashlib
import json

import pytest

from ecopadg.measure.backends import INSTANT_POWER_SOURCE_ID
from ecopadg.scalability.audit import audit_run


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_fixture(tmp_path, n=10):
    frozen = tmp_path / "source.py"
    frozen.write_text("# frozen test source\n")
    source = dict(mode="instant", source_id=INSTANT_POWER_SOURCE_ID, field_id=186, scope_id=0)
    manifest = dict(scope="gpu_serving", system="pdblend-joint", dataset="sharegpt", n_gpus=3,
        allocated_gpu_ids=[0, 2, 5], seed=701, rate_rps=1., stage="capacity", arrival_window_s=600,
        model="test-model", host_id="test-host", profile_sha256="test-profile", source_config_sha256="test-config",
        source_hashes={str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()},
        slo_ttft_s=.2, slo_tpot_s=.05, formal_eligible=True)
    measurement = dict(start_s=1000., arrival_end_s=1600., end_s=1602., initial_quiescent=True,
        terminal_quiescent=True, hardware_qualification_verified=True, source_freeze_verified=True, declared_formal=True)
    requests, outputs, rows = [], [], []
    for index in range(n):
        request = dict(arrival_s=index * 50., prompt_len=3, output_len=2, timeout_s=120.)
        raw = dict(request_id=str(index), arrival_s=1000 + index * 50., finish_s=1000.2 + index * 50.,
            success=True, error="", input_tokens=3, generated_tokens=2, token_ids=[42, 43],
            token_count_source="server_usage", ttft=.1, latency=.11, token_itl=[.01], token_events_exact=True,
            declared_timeout_s=120.)
        row = dict(request_id=str(index), arrival_s=raw["arrival_s"], finish_s=raw["finish_s"],
            prompt_len=3, output_len=2, success=1, error="", input_tokens=3, generated_tokens=2,
            token_count_source="server_usage", token_ids_verified=1, ttft_s=.1, tpot_s=.01,
            output_token_sha256=hashlib.sha256(json.dumps(raw["token_ids"]).encode()).hexdigest(),
            http_status="", admission_rejection="")
        requests.append(request)
        outputs.append(raw)
        rows.append(row)
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "measurement.json", measurement)
    write_json(tmp_path / "trace.json", dict(requests=requests, prompts=[[1, 2, 3]] * n))
    manifest["trace_sha256"] = hashlib.sha256((tmp_path / "trace.json").read_bytes()).hexdigest()
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "outputs.json", outputs)
    write_json(tmp_path / "power_source.json", source)
    write_csv(tmp_path / "bench.csv", rows)
    powers, metadata = [], []
    for timestamp in range(999, 1604):
        powers.append(dict(t_s=timestamp, **{f"gpu{g}_w": 10 * (g + 1) for g in range(8)},
                           **{f"gpu{g}_util_pct": 50 for g in range(8)}))
        metadata.append(dict(t_s=timestamp, gpus=list(range(8)), **{k: [v] * 8 for k, v in source.items()},
            value_type=[1] * 8, return_code=[0] * 8, nvml_timestamp_us=[timestamp * 1000000] * 8,
            nvml_latency_us=[0] * 8, read_started_s=[timestamp] * 8, read_finished_s=[timestamp] * 8))
    write_csv(tmp_path / "power.csv", powers)
    (tmp_path / "power_metadata.jsonl").write_text("\n".join(json.dumps(v) for v in metadata))
    (tmp_path / "backlog.jsonl").write_text("\n".join(json.dumps(dict(t_s=t, pending=0)) for t in range(1300, 1601)))
    (tmp_path / "control.jsonl").write_text("")
    return manifest, measurement, outputs, rows


def test_independent_energy_full_window_and_complete_output(tmp_path):
    run_fixture(tmp_path)
    result = audit_run(tmp_path)
    assert result["measurement_valid"], result["audit_errors"]
    assert result["capacity_pass"]
    assert result["energy_allocated_j"] == pytest.approx(602 * 100)
    assert result["energy_node8_j"] == pytest.approx(602 * 360)
    assert result["energy_allocated_j"] != result["energy_node8_j"] * 3 / 8
    assert result["goodput_rps"] == pytest.approx(10 / 602)
    assert result["token_itl_count"] == 10
    assert result["route_counts"] is None


def test_all_offered_requests_include_classified_rejection_and_timeout(tmp_path):
    _, _, outputs, rows = run_fixture(tmp_path)
    for index in (0, 1):
        outputs[index].update(success=False, generated_tokens=0, input_tokens=0, token_ids=[],
                              token_count_source="missing", ttft=None, latency=.2, token_itl=[])
        rows[index].update(success=0, generated_tokens=0, input_tokens=0, token_count_source="missing",
                           ttft_s=None, tpot_s=None, output_token_sha256="")
    outputs[0].update(http_status=429, admission_rejection="admission_queue_full", error="HTTP 429")
    rows[0].update(http_status=429, admission_rejection="admission_queue_full", error="HTTP 429")
    outputs[1]["error"] = rows[1]["error"] = "TimeoutError: request deadline"
    outputs[1]["finish_s"] = rows[1]["finish_s"] = outputs[1]["arrival_s"] + 120.
    write_json(tmp_path / "outputs.json", outputs)
    write_csv(tmp_path / "bench.csv", rows)
    result = audit_run(tmp_path)
    assert result["measurement_valid"], result["audit_errors"]
    assert result["slo_attainment"] == .8 and not result["capacity_pass"]
    assert result["classification_counts"]["request_timeout"] == 1
    assert result["classification_counts"]["capacity_rejection"] == 1
    assert result["energy_allocated_j"] == 60200


def test_missing_ids_and_unknown_failures_never_pass(tmp_path):
    _, _, outputs, rows = run_fixture(tmp_path)
    outputs[0]["token_ids"] = [42]
    write_json(tmp_path / "outputs.json", outputs)
    assert not audit_run(tmp_path)["measurement_valid"]
    (tmp_path / "outputs.json").unlink()
    assert not audit_run(tmp_path)["measurement_valid"]


def test_slo_equality_is_not_strict_pass_and_csv_flag_is_ignored(tmp_path):
    manifest, _, _, _ = run_fixture(tmp_path)
    manifest["slo_ttft_s"] = .1
    write_json(tmp_path / "manifest.json", manifest)
    result = audit_run(tmp_path)
    assert result["measurement_valid"] and result["slo_attainment"] == 0


def test_missing_power_provenance_and_modified_freeze_are_invalid(tmp_path):
    run_fixture(tmp_path)
    (tmp_path / "power_metadata.jsonl").unlink()
    (tmp_path / "source.py").write_text("changed")
    result = audit_run(tmp_path)
    assert not result["measurement_valid"]
    assert any("source freeze mismatch" in error for error in result["audit_errors"])


def test_diagnostic_run_cannot_be_formal_capacity(tmp_path):
    manifest, measurement, _, _ = run_fixture(tmp_path)
    manifest.update(stage="diagnostic", formal_eligible=False)
    measurement["declared_formal"] = False
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "measurement.json", measurement)
    result = audit_run(tmp_path)
    assert result["measurement_valid"] and not result["formal_eligible"] and not result["capacity_pass"]


def test_qualified_pilot_can_pass_but_never_be_formal(tmp_path):
    manifest, measurement, _, _ = run_fixture(tmp_path)
    manifest.update(stage="pilot", formal_eligible=False)
    measurement["declared_formal"] = False
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "measurement.json", measurement)
    result = audit_run(tmp_path)
    assert result["measurement_valid"] and result["capacity_pass"] and not result["formal_eligible"]


def test_arrival_shift_and_cleanup_error_are_invalid(tmp_path):
    _, measurement, outputs, rows = run_fixture(tmp_path)
    outputs[0]["arrival_s"] = rows[0]["arrival_s"] = 1000.1
    write_json(tmp_path / "outputs.json", outputs)
    write_csv(tmp_path / "bench.csv", rows)
    measurement["cleanup_errors"] = ["native drain missing"]
    write_json(tmp_path / "measurement.json", measurement)
    result = audit_run(tmp_path)
    assert not result["measurement_valid"]
    assert any("boundary" in error for error in result["audit_errors"])
    assert any("cleanup_errors" in error for error in result["audit_errors"])


def test_route_counts_exclude_warmup_and_untimestamped_events(tmp_path):
    run_fixture(tmp_path)
    events = [dict(kind="admission", at_s=t, plan=dict(routes=[dict(prefill_id="p", decode_id="d")]))
              for t in (900., 1000., 1602., 1700.)]
    events.append(dict(kind="admission", plan=dict(routes=[dict(prefill_id="m", decode_id="m")])))
    (tmp_path / "control.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    result = audit_run(tmp_path)
    assert result["measurement_valid"] and result["route_counts"] == {"pd": 2}
