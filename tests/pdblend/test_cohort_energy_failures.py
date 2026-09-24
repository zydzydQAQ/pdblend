import gzip
import json

import pytest

from scripts.cohort_energy_failures import analyze_failures


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def journal(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_eco_cohort_receipt_attributes_only_matching_failed_requests(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "native-result.json", {"outcomes": [
        {"request_id": "ecoserve-701-0", "ok": False},
        {"request_id": "ecoserve-701-1", "ok": False},
    ]})
    journal(run / "events.jsonl.gz", [
        {"kind": "eco_comparison_cohort_cancel_begin", "reason": "cohort_timeout",
         "error": "TimeoutError()", "pending_requests": ["ecoserve-701-1", "ecoserve-701-2"]},
    ])
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    result = analyze_failures(tmp_path, [
        {"idx": 0, "successful": False, "error": "native_output_validation_failed"},
        {"idx": 1, "successful": False, "error": "native_output_validation_failed"},
        {"idx": 2, "successful": True},
    ])
    assert result["counts"]["cohort_timeout_cancelled"] == 1
    assert result["counts"]["invalid_output"] == 1
    assert sum(result["counts"].values()) == 2
    assert result["timeout_requests"] == result["cancelled_requests"] == 1
    detail = next(row for row in result["failure_details"] if row["idx"] == 1)
    assert detail["raw_error"] is None
    assert "eco_comparison_cohort_cancel_begin" in detail["cause_source"]
    assert all(path.startswith(str(tmp_path)) for path in result["source_paths"])
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


def test_generic_timeout_never_assumed_predictor_and_nested_native_error(tmp_path):
    write_json(tmp_path / "outcomes.json", [
        {"request_id": "dynamo-701-0", "error": "TimeoutError()"},
        {"request_id": "distserve-701-1", "result": {"error": "request deadline exceeded"}},
        {"request_id": "dynamo-701-2", "error": "Dynamo request rejected; retry elsewhere"},
        {"request_id": "dynamo-701-3", "error": "TimeoutError()"},
        {"request_id": "dynamo-701-4", "error": "cancelled"},
        {"request_id": "dynamo-701-5", "result": {"error": "predictor timed out"}},
    ])
    journal(tmp_path / "events.jsonl.gz", [
        {"event": "dynamo_predictor_timeout", "request_id": "dynamo-701-3", "error": "TimeoutError()"},
    ])
    result = analyze_failures(tmp_path, [
        {"idx": index, "successful": False, "error": "native_output_validation_failed"}
        for index in range(6)
    ])
    expected = ["timeout_unknown", "native_timeout", "native_reject", "predictor_timeout", "cancelled", "predictor_timeout"]
    assert [row["category"] for row in result["failure_details"]] == expected
    assert result["timeout_requests"] == 4
    assert result["rejected_requests"] == result["cancelled_requests"] == 1
    assert sum(result["counts"].values()) == 6


def test_unresolved_is_distinct_from_tail_completion_and_unknown(tmp_path):
    result = analyze_failures(tmp_path, [
        {"idx": 0, "successful": False, "error": "missing_outcome"},
        {"idx": 1, "successful": True, "pending_at_window_end": True},
        {"idx": 2, "successful": False, "error": "unexplained native failure", "pending_at_window_end": True},
    ])
    assert result["counts"]["unresolved"] == 1
    assert result["counts"]["unknown"] == 1
    assert result["unresolved_requests"] == 1
    assert len(result["failure_details"]) == 2
    assert result["source_paths"] == []


def test_explicit_request_id_and_no_numeric_suffix_guessing(tmp_path):
    (tmp_path / "outcomes.jsonl").write_text(
        json.dumps({"request_id": "custom-key", "error": "cancelled"}) + "\n" +
        json.dumps({"request_id": "foreign-8", "error": "TimeoutError()"}) + "\n")
    result = analyze_failures(tmp_path, [
        {"idx": 7, "request_id": "custom-key", "successful": False},
        {"idx": 8, "successful": False},
    ])
    assert result["counts"]["cancelled"] == result["counts"]["unknown"] == 1
    assert result["timeout_requests"] == 0


def test_conflicting_identity_and_duplicate_canonical_indices_fail_closed(tmp_path):
    rows = [{"idx": 0, "successful": False}]
    with pytest.raises(ValueError, match="unique"):
        analyze_failures(tmp_path, rows * 2)
    write_json(tmp_path / "outcomes.json", [{"idx": 0, "request_id": "dynamo-701-1"}])
    with pytest.raises(ValueError, match="conflicting"):
        analyze_failures(tmp_path, rows)


def test_success_only_fast_path_does_not_read_native_files(tmp_path):
    (tmp_path / "events.jsonl.gz").write_bytes(b"not a gzip file")
    result = analyze_failures(tmp_path, [{"idx": 0, "successful": True}])
    assert sum(result["counts"].values()) == 0
    assert result["source_paths"] == result["failure_details"] == []


def test_cancel_begin_does_not_prove_a_missing_outcome_was_resolved(tmp_path):
    journal(tmp_path / "events.jsonl.gz", [
        {"kind": "eco_comparison_cohort_cancel_begin", "reason": "cohort_timeout",
         "pending_requests": ["ecoserve-701-0"]},
    ])
    result = analyze_failures(tmp_path, [
        {"idx": 0, "successful": False, "error": "missing_outcome"},
    ])
    assert result["counts"]["cohort_timeout_cancelled"] == 1
    assert result["unresolved_requests"] == 1
