#!/usr/bin/env python3
"""Append receipt-bound analysis annotations after the sole writer stops."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile

SCHEMA = "pdblend-analysis-row-annotation-registry/v1"
BINDING = ("receipt_path", "receipt_sha256", "point_id", "point_sha256",
           "system", "run_id", "revision")
COLUMNS = (
    "writer_startup_overlap_status", "writer_startup_overlap_s",
    "writer_startup_interval_start_s", "writer_startup_interval_end_s",
    "writer_startup_interval_semantics", "writer_startup_causal_impact",
    "writer_startup_diagnostic_path", "writer_startup_diagnostic_sha256",
)
SEMANTICS = "writer_launch_to_first_csv_publish_not_measured_cpu_busy_time"
ENERGY_SCHEMA = "pdblend-energy-gap-estimate-annotation-proposal/v1"
IO_SCHEMA = "pdblend-concurrent-audit-io-annotation-proposal/v1"
RECOVERED_SCHEMA = "pdblend-failed-native-recovery-annotation/v2"
AUXILIARY_SCHEMA = "pdblend-auxiliary-observer-annotation/v1"
AUXILIARY_COLUMNS = (
    "auxiliary_status", "auxiliary_energy_service_j", "auxiliary_energy_tail_j",
    "auxiliary_energy_service_tail_j", "auxiliary_service_mean_power_w",
    "auxiliary_gpu_count", "auxiliary_gpu_uuids_json", "auxiliary_gpu_util_mean_pct",
    "auxiliary_service_power_coverage_fraction", "auxiliary_service_power_min_gpu_coverage_fraction",
    "auxiliary_service_power_max_gap_s", "auxiliary_tail_power_coverage_fraction",
    "auxiliary_tail_power_min_gpu_coverage_fraction", "auxiliary_tail_power_max_gap_s",
    "auxiliary_service_util_coverage_fraction", "auxiliary_service_util_max_gap_s",
    "auxiliary_source_sha256", "auxiliary_method_sha256", "auxiliary_artifact_path",
    "auxiliary_artifact_sha256", "auxiliary_used_for_ranking", "auxiliary_unavailable_reason",
)
ENERGY_COLUMNS = (
    "analysis_energy_estimate_status", "analysis_energy_service_trapezoid_j",
    "analysis_energy_tail_trapezoid_j", "analysis_energy_service_tail_trapezoid_j",
    "analysis_energy_service_gap_contribution_j", "analysis_energy_tail_gap_contribution_j",
    "analysis_energy_service_coverage_fraction", "analysis_energy_tail_coverage_fraction",
    "analysis_energy_service_max_gap_s", "analysis_energy_tail_max_gap_s",
    "analysis_energy_estimate_boundaries_bracketed", "analysis_energy_estimate_method",
    "analysis_energy_estimate_eligible_for_ranking", "analysis_energy_estimate_diagnostic_path",
    "analysis_energy_estimate_diagnostic_sha256",
)
IO_COLUMNS = (
    "concurrent_audit_io_status", "concurrent_audit_io_input_bytes",
    "concurrent_audit_io_command_elapsed_s", "concurrent_audit_io_overlap_s",
    "concurrent_audit_io_causal_impact", "concurrent_audit_io_elapsed_semantics",
    "concurrent_audit_io_diagnostic_path", "concurrent_audit_io_diagnostic_sha256",
)
RECOVERED_STATS = ("p50_s", "p90_s", "p95_s", "p99_s", "mean_s", "max_s", "samples")
RECOVERED_COLUMNS = (
    "recovered_status", "recovered_offered_requests", "recovered_native_successful_requests",
    "recovered_native_failed_requests", "recovered_joint_slo_requests", "recovered_joint_slo_rate",
    "recovered_native_slo_pass", "recovered_latency_sample_scope",
    *("recovered_" + name + "_" + statistic for name in ("ttft", "tpot") for statistic in RECOVERED_STATS),
    "recovered_service_start_s", "recovered_service_end_s", "recovered_service_energy_j",
    "recovered_service_mean_power_w", "recovered_service_power_coverage_fraction",
    "recovered_service_minimum_gpu_coverage_fraction", "recovered_service_max_gap_s",
    "recovered_service_gpu_util_mean_pct", "recovered_service_util_coverage_fraction",
    "recovered_tail_status", "recovered_tail_energy_j", "recovered_used_for_ranking",
    "recovered_native_goodput_request_s_lower_bound", "recovered_native_cohort_goodput_request_s",
    "recovered_native_cohort_goodput_token_s", "recovered_goodput_scope",
    "recovered_client_pre_dispatch_delay_p99_s", "recovered_client_peak_pre_dispatch_outstanding",
    "recovered_client_send_queue_status", "recovered_source_manifest_path", "recovered_source_manifest_sha256",
    "recovered_native_completion_path", "recovered_native_completion_sha256",
    "recovered_power_manifest_path", "recovered_power_manifest_review_sha256",
    "recovered_power_raw_path", "recovered_power_raw_original_manifest_sha256",
    "recovered_power_binding_scope", "recovered_power_method_path", "recovered_power_method_sha256",
    "recovered_native_review_path", "recovered_native_review_sha256",
    "recovered_power_review_path", "recovered_power_review_sha256",
    *(f"recovered_gpu{i}_{field}" for i in range(8) for field in
      ("uuid", "util_mean_pct", "util_peak_pct", "util_coverage_fraction", "util_max_gap_s", "util_status")),
    "recovered_native_goodput_token_s_lower_bound",
    "recovered_native_window_good_output_tokens_lower_bound", "recovered_native_window_goodput_status",
    "recovered_scalar_requests_path", "recovered_scalar_requests_sha256",
)
RECOVERED_BINDING_SCOPE = "raw_SHA_from_original_session_manifest; manifest_SHA_first_bound_by_independent_review_not_failed_receipt"
REGISTRY_COLUMNS = {SCHEMA: COLUMNS, ENERGY_SCHEMA: ENERGY_COLUMNS, IO_SCHEMA: IO_COLUMNS,
                    RECOVERED_SCHEMA: RECOVERED_COLUMNS, AUXILIARY_SCHEMA: AUXILIARY_COLUMNS}


def annotation_defaults(schema):
    columns = REGISTRY_COLUMNS[schema]
    return {k: ("not_scheduled" if schema == AUXILIARY_SCHEMA else "not_annotated")
            if k == columns[0] else "" for k in columns}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_json(path, expected):
    data = Path(path).read_bytes()
    if sha(data) != expected:
        raise ValueError(f"artifact SHA mismatch: {path}")
    return json.loads(data)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integrate_projection(projection, start, end):
    rows = projection["selected"]
    samples, metadata = rows["samples"], rows["power_metadata"]
    if not samples or len(samples) != len(metadata) or not all(finite(v) for v in (start, end)) or end <= start:
        raise ValueError("invalid energy projection")
    series = [[] for _ in range(8)]
    for sample, meta in zip(samples, metadata):
        values, times = sample[1], meta["read_finished_s"]
        if (len(values) != 8 or len(times) != 8 or not all(finite(v) and v >= 0 for v in values)
                or not all(finite(t) for t in times)):
            raise ValueError("energy projection requires eight finite device readings")
        for gpu in range(8):
            series[gpu].append((times[gpu], values[gpu]))
    strict_total = estimate_total = 0.0
    for device in series:
        if (device[0][0] > start or device[-1][0] < end
                or any(z[0] <= a[0] for a, z in zip(device, device[1:]))):
            raise ValueError("energy projection does not bracket each GPU phase monotonically")
        for (a, pa), (z, pz) in zip(device, device[1:]):
            left, right = max(a, start), min(z, end)
            if right <= left:
                continue
            pl = pa + (pz - pa) * (left - a) / (z - a)
            pr = pa + (pz - pa) * (right - a) / (z - a)
            area = (pl + pr) * 0.5 * (right - left)
            estimate_total += area
            if z - a <= 1.0:
                strict_total += area
    return strict_total, estimate_total


def validate_energy(annotation, receipt):
    binding, columns = annotation["binding"], annotation["columns"]
    diagnostic = checked_json(columns[ENERGY_COLUMNS[-2]], columns[ENERGY_COLUMNS[-1]])
    if (diagnostic.get("schema") != "pdblend-sampling-gap-energy-analysis/v1"
            or diagnostic.get("used_for_ranking") is not False
            or diagnostic.get("receipt") != {"path": binding["receipt_path"], "sha256": binding["receipt_sha256"]}):
        raise ValueError("invalid energy estimate diagnostic binding or ranking scope")
    meter_ref, projection_ref = diagnostic["metering"], diagnostic["projection"]
    meter = checked_json(meter_ref["path"], meter_ref["sha256"])
    projection = checked_json(projection_ref["path"], projection_ref["sha256"])
    source = diagnostic["source_journal"]
    directory = Path(binding["receipt_path"]).parent
    if (Path(source["path"]) != directory / "run/power.samples.jsonl.gz"
            or source["sha256"] != receipt["artifacts"]["run/power.samples.jsonl.gz"]
            or projection.get("source_sha256") != source["sha256"]
            or Path(meter_ref["path"]) != directory / "run/comparison-metering.json"
            or meter_ref["sha256"] != receipt["artifacts"]["run/comparison-metering.json"]):
        raise ValueError("energy estimate source is not receipt-bound")
    phases = diagnostic["phases"]
    for name in ("service", "tail"):
        phase, canonical = phases[name], meter[name]["power"]
        if (phase.get("all_gpu_boundaries_bracketed") is not True
                or phase.get("canonical_energy_j") is not None
                or canonical.get("energy_j") is not None):
            raise ValueError("energy estimate requires bracketed boundaries and missing canonical energy")
        if any(phase[k] != meter[name][k] for k in ("start_s", "end_s")):
            raise ValueError("estimated phase differs from canonical interval")
        raw_strict, raw_estimate = integrate_projection(projection, phase["start_s"], phase["end_s"])
        if (not math.isclose(raw_strict, canonical["covered_energy_j"], rel_tol=0, abs_tol=1e-6)
                or not math.isclose(raw_estimate, phase["trapezoidal_estimate_j"], rel_tol=0, abs_tol=1e-6)):
            raise ValueError("energy estimate differs from bound power projection")
        expected_phase = {"strict_covered_energy_j": canonical["covered_energy_j"],
                          "strict_coverage_fraction": canonical["coverage_fraction"],
                          "max_gap_s": canonical["max_gap_s"]}
        for key, expected in expected_phase.items():
            if not finite(phase.get(key)) or not math.isclose(phase[key], expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError("energy estimate differs from canonical coverage")
        per_gpu = phase["per_gpu"]
        if any(type(p["gpu"]) is not int for p in per_gpu) or sorted(p["gpu"] for p in per_gpu) != list(range(8)):
            raise ValueError("energy estimate must retain all eight GPUs")
        gap_total = 0.0
        for device in per_gpu:
            area = 0.0
            for gap in device["gaps"]:
                a, z, pa, pz = [gap[k] for k in ("read_before_s", "read_after_s", "left_w", "right_w")]
                if not all(finite(v) for v in (a, z, pa, pz)) or z - a <= 1 or min(pa, pz) < 0:
                    raise ValueError("invalid sampling-gap endpoints")
                left, right = max(a, phase["start_s"]), min(z, phase["end_s"])
                if right <= left:
                    raise ValueError("sampling gap lies outside annotated phase")
                pl = pa + (pz - pa) * (left - a) / (z - a)
                pr = pa + (pz - pa) * (right - a) / (z - a)
                area += (pl + pr) * 0.5 * (right - left)
            if not math.isclose(area, device["gap_interpolated_j"], rel_tol=0, abs_tol=1e-6):
                raise ValueError("gap contribution cannot be reproduced")
            gap_total += area
        estimate = canonical["covered_energy_j"] + gap_total
        if (not math.isclose(estimate, phase["trapezoidal_estimate_j"], rel_tol=0, abs_tol=1e-6)
                or not math.isclose(gap_total, phase["estimated_gap_contribution_j"], rel_tol=0, abs_tol=1e-6)):
            raise ValueError("estimated energy cannot be reproduced")
    service, tail = phases["service"], phases["tail"]
    expected = dict(zip(ENERGY_COLUMNS, (
        "trapezoidal_estimate_over_sampling_gaps", service["trapezoidal_estimate_j"],
        tail["trapezoidal_estimate_j"], service["trapezoidal_estimate_j"] + tail["trapezoidal_estimate_j"],
        service["estimated_gap_contribution_j"], tail["estimated_gap_contribution_j"],
        service["strict_coverage_fraction"], tail["strict_coverage_fraction"],
        service["max_gap_s"], tail["max_gap_s"], True,
        "per_gpu_acquisition_time_linear_trapezoid_including_gaps_over_1s", False,
        columns[ENERGY_COLUMNS[-2]], columns[ENERGY_COLUMNS[-1]],
    )))
    if (columns != expected or columns[ENERGY_COLUMNS[10]] is not True
            or columns[ENERGY_COLUMNS[12]] is not False
            or not all(finite(columns[k]) for k in ENERGY_COLUMNS[1:10])):
        raise ValueError("energy annotation differs from verified estimate")


def validate_io(annotation, receipt):
    binding, columns = annotation["binding"], annotation["columns"]
    diagnostic = checked_json(columns[IO_COLUMNS[-2]], columns[IO_COLUMNS[-1]])
    if diagnostic.get("schema") != "independent-review-io-incident/v1":
        raise ValueError("unsupported I/O incident schema")
    window, operation = diagnostic["mixed_window"], diagnostic["operation"]
    if window["receipt"] != {"path": binding["receipt_path"], "sha256": binding["receipt_sha256"]}:
        raise ValueError("I/O incident receipt binding differs")
    metrics = receipt["result"]["metrics"]
    if any(window[k] != metrics[k] for k in ("service_start_s", "service_end_s")):
        raise ValueError("I/O incident service interval differs")
    if (operation.get("wall_start_s") is not None or operation.get("wall_end_s") is not None
            or window.get("overlap") != "unknown_possible" or window.get("causal_effect") != "not_determined"):
        raise ValueError("this I/O schema requires unknown wall-time overlap")
    sizes = [f["size_bytes"] for f in diagnostic["files"]]
    elapsed = operation["reported_tool_elapsed_s"]
    if (not all(type(v) is int and v >= 0 for v in sizes) or sum(sizes) != diagnostic["total_size_bytes"]
            or not finite(elapsed) or elapsed < 0):
        raise ValueError("invalid I/O incident size or elapsed duration")
    expected = dict(zip(IO_COLUMNS, (
        "possible_overlap_timing_unknown", sum(sizes), elapsed, "", "unknown",
        "reported_command_elapsed_not_measured_io_busy_time", columns[IO_COLUMNS[-2]], columns[IO_COLUMNS[-1]],
    )))
    if columns != expected or type(columns[IO_COLUMNS[1]]) is not int or not finite(columns[IO_COLUMNS[2]]):
        raise ValueError("I/O annotation differs from incident or invents overlap")


def recovered_device_columns(power):
    """Map physical GPU identity in its bound order, preserving missing statistics."""
    uuids = power["gpu_uuids"]
    devices = power["service"]["utilization"].get("per_gpu", {})
    if len(uuids) != 8 or len(set(uuids)) != 8 or not set(devices).issubset(uuids):
        raise ValueError("recovered utilization UUID identity differs")
    fields = {}
    for index, uuid in enumerate(uuids):
        device = devices.get(uuid, {})
        prefix = f"recovered_gpu{index}_"
        fields[prefix + "uuid"] = uuid
        mapping = {"util_mean_pct": "mean_pct", "util_peak_pct": "peak_pct",
                   "util_coverage_fraction": "coverage_fraction", "util_max_gap_s": "max_gap_s"}
        for column, key in mapping.items():
            value = device.get(key)
            if value is not None and (not finite(value) or value < 0 or
                    (key in ("mean_pct", "peak_pct") and value > 100) or
                    (key == "coverage_fraction" and value > 1)):
                raise ValueError("invalid recovered per-GPU utilization")
            fields[prefix + column] = "" if value is None else value
        missing = [key for key in mapping.values() if device.get(key) is None]
        status = device.get("status")
        if device.get("mean_pct") is not None and device.get("peak_pct") is not None:
            if device["mean_pct"] > device["peak_pct"]:
                raise ValueError("recovered utilization mean exceeds peak")
        if status == "complete" and not missing and (
                device["coverage_fraction"] != 1 or device["max_gap_s"] > 1):
            raise ValueError("recovered utilization completeness differs from coverage")
        fields[prefix + "util_status"] = ("unknown_missing_fields:" + ",".join(missing)
            if missing else status or "unknown_missing_status")
    return fields


def recovered_token_columns(native):
    """Use the already bound scalar cache; never reopen native requests or journals."""
    ref = native.get("scalar_requests")
    fields = dict(recovered_native_goodput_token_s_lower_bound="",
        recovered_native_window_good_output_tokens_lower_bound="",
        recovered_native_window_goodput_status="unknown_missing_scalar_request_evidence",
        recovered_scalar_requests_path="", recovered_scalar_requests_sha256="")
    if not ref:
        return fields
    rows = checked_json(ref["path"], ref["sha256"])
    point_ref = native["bindings"]["point_artifact"]
    point = checked_json(point_ref["path"], point_ref["sha256"])
    point_digest = sha(json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    counts = native["counts"]
    if (point_digest != native["bindings"]["point_sha256"] or not isinstance(rows, list)
            or len(rows) != counts["expected_requests"]
            or any(type(r.get("idx")) is not int for r in rows)
            or sorted(r["idx"] for r in rows) != list(range(len(rows)))):
        raise ValueError("recovered scalar request identity or count differs")
    success = [r for r in rows if r.get("ok") is True]
    if any(not finite(r.get(k)) or r[k] < 0 for r in success for k in ("ttft_s", "tpot_s", "finished_s")):
        raise ValueError("recovered scalar timing missing or invalid")
    if any(type(r.get("completion_tokens")) is not int or r["completion_tokens"] < 0 for r in success):
        raise ValueError("recovered scalar output token count invalid")
    good = [r for r in success if r["ttft_s"] <= point["slo"]["ttft_s"]
            and r["tpot_s"] <= point["slo"]["tpot_s"]]
    window = [r for r in good if r["finished_s"] <= native["service_ended_s"]]
    duration = native["service_ended_s"] - native["service_started_s"]
    if (duration <= 0 or len(success) != counts["native_successful_requests"]
            or len(good) != counts["native_joint_slo_requests"]
            or sum(r["completion_tokens"] for r in good) != counts["native_joint_slo_completion_tokens"]
            or len(window) != counts["native_good_requests_finished_by_window_end"]
            or not math.isclose(len(window)/duration,
                native["native_goodput_finished_window_request_s_lower_bound"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("recovered scalar goodput differs from frozen native review")
    tokens = sum(r["completion_tokens"] for r in window)
    fields.update(recovered_native_goodput_token_s_lower_bound=tokens/duration,
        recovered_native_window_good_output_tokens_lower_bound=tokens,
        recovered_native_window_goodput_status="conservative_lower_bound_native_joint_SLO_success_finished_s_by_service_end_not_last_token_timestamp",
        recovered_scalar_requests_path=ref["path"], recovered_scalar_requests_sha256=ref["sha256"])
    return fields


def recovered_columns(native, power, native_ref, power_ref):
    """Fields are separate from every canonical metric and qualification."""
    counts, service = native["counts"], power["service"]
    source, raw, manifest, method = (power["bindings"]["source_manifest"], power["session_power_raw"],
                                    power["session_power_manifest"], power["original_method"])
    fields = dict(
        recovered_status="native_requests_and_complete_service_recovered_outer_drain_failed",
        recovered_offered_requests=counts["expected_requests"],
        recovered_native_successful_requests=counts["native_successful_requests"],
        recovered_native_failed_requests=counts["native_failed_or_missing_requests"],
        recovered_joint_slo_requests=counts["native_joint_slo_requests"],
        recovered_joint_slo_rate=counts["native_joint_slo_rate"],
        recovered_native_slo_pass=native["native_request_slo_pass"],
        recovered_latency_sample_scope="successful_native_requests_only; TTFT_from_planned_arrival; TPOT_from_client_first_last_token; journal_not_replayed",
        recovered_service_start_s=power["service_start_s"], recovered_service_end_s=power["service_end_s"],
        recovered_service_energy_j=power["recovered_energy_service_j"],
        recovered_service_mean_power_w=power["recovered_service_mean_power_w"],
        recovered_service_power_coverage_fraction=service["power"]["coverage_fraction"],
        recovered_service_minimum_gpu_coverage_fraction=service["power"]["minimum_gpu_coverage_fraction"],
        recovered_service_max_gap_s=service["power"]["max_gap_s"],
        recovered_service_gpu_util_mean_pct=power["recovered_gpu_util_mean_pct"],
        recovered_service_util_coverage_fraction=service["utilization"]["coverage_fraction"],
        recovered_tail_status="unknown_no_accepted_drain_end", recovered_tail_energy_j="",
        recovered_used_for_ranking=False,
        recovered_native_goodput_request_s_lower_bound=native["native_goodput_finished_window_request_s_lower_bound"],
        recovered_native_cohort_goodput_request_s=native["native_cohort_goodput_request_s"],
        recovered_native_cohort_goodput_token_s=native["native_cohort_goodput_token_s"],
        recovered_goodput_scope="native_slo_success; window_uses_finished_s_conservative_lower_bound; cohort_uses_actual_request_outcome_horizon_not_verified_GPU_drain",
        recovered_client_pre_dispatch_delay_p99_s=native["client_pre_dispatch_delay"]["p99_s"],
        recovered_client_peak_pre_dispatch_outstanding=native["client_peak_pre_dispatch_outstanding"],
        recovered_client_send_queue_status=native["client_send_queue_status"],
        recovered_source_manifest_path=source["path"], recovered_source_manifest_sha256=source["sha256"],
        recovered_native_completion_path=native["native_completion"]["path"],
        recovered_native_completion_sha256=native["native_completion"]["sha256"],
        recovered_power_manifest_path=manifest["path"], recovered_power_manifest_review_sha256=manifest["sha256"],
        recovered_power_raw_path=raw["path"], recovered_power_raw_original_manifest_sha256=raw["sha256"],
        recovered_power_binding_scope=RECOVERED_BINDING_SCOPE,
        recovered_power_method_path=method["path"], recovered_power_method_sha256=method["sha256"],
        recovered_native_review_path=native_ref["path"], recovered_native_review_sha256=native_ref["sha256"],
        recovered_power_review_path=power_ref["path"], recovered_power_review_sha256=power_ref["sha256"],
    )
    for name in ("ttft", "tpot"):
        fields.update({"recovered_" + name + "_" + key: native["native_" + name][key] for key in RECOVERED_STATS})
    fields.update(recovered_device_columns(power))
    fields.update(recovered_token_columns(native))
    return {key: "" if value is None else value for key, value in fields.items()}


def validate_recovered(annotation, receipt):
    binding, columns = annotation["binding"], annotation["columns"]
    native_ref = dict(path=columns["recovered_native_review_path"], sha256=columns["recovered_native_review_sha256"])
    power_ref = dict(path=columns["recovered_power_review_path"], sha256=columns["recovered_power_review_sha256"])
    native = checked_json(native_ref["path"], native_ref["sha256"])
    power = checked_json(power_ref["path"], power_ref["sha256"])
    if (native.get("schema") != "distserve-native-failed-window-scalar-review/v1"
            or power.get("schema") != "failed-native-recovered-service-power/v1"
            or "result" in receipt or not receipt.get("error")
            or receipt.get("cleanup_passed") is not False):
        raise ValueError("recovery requires an original failed window without canonical result")
    directory = Path(binding["receipt_path"]).parent
    point = checked_json(directory / "point.json", receipt["artifacts"]["point.json"])
    point_digest = sha(json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (point_digest != binding["point_sha256"] or point.get("name") != binding["point_id"]
            or any(point.get(key) != binding[key] for key in ("system", "run_id", "revision"))):
        raise ValueError("recovered point identity differs")
    source = checked_json(point["source_manifest"]["path"], point["source_manifest"]["sha256"])
    if source.get("source_sha256") != binding["revision"]:
        raise ValueError("recovered source revision differs")
    reference = dict(path=binding["receipt_path"], sha256=binding["receipt_sha256"])
    for review in (native, power):
        if (review.get("canonical_modified") is not False or review.get("used_for_ranking") is not False
                or review.get("canonical_energy_service_j") is not None or review.get("canonical_energy_tail_j") is not None
                or review["bindings"]["receipt"] != reference
                or review["bindings"]["point_sha256"] != binding["point_sha256"]
                or review["bindings"]["source_manifest"] != point["source_manifest"]
                or review["bindings"]["trace"] != point["trace"]):
            raise ValueError("recovery diagnostic binding or analysis-only scope differs")
    if (native["native_completion"] != dict(path=str(directory/"run/completion.json"), sha256=receipt["artifacts"]["run/completion.json"])
            or power["native_scalar_review"] != native_ref
            or power.get("recovered_tail_available") is not False or power.get("outer_drain_complete") is not False
            or power.get("recovered_energy_tail_j") is not None):
        raise ValueError("recovery cannot invent drain completion or replace native source")
    for name, filename in (("session_power_manifest", "session-power.json"), ("session_power_raw", "session-power.samples.jsonl.gz")):
        ref = power[name]
        if (Path(ref["path"]) != directory.parent.parent/filename or
                not isinstance(ref.get("sha256"), str) or len(ref["sha256"]) != 64):
            raise ValueError("recovered power source does not belong to the failed session")
    if power.get("raw_binding_scope") != "raw SHA stored in original session-power manifest; manifest SHA first captured by this independent review, not included in the failed window receipt":
        raise ValueError("recovery must distinguish original raw SHA and review-time manifest SHA")
    method = checked_json(directory/"run/metering-method-failure.json", receipt["artifacts"]["run/metering-method-failure.json"])
    if method["factory"]["sha256"] != power["original_method"]["sha256"]:
        raise ValueError("recovered integration method differs from executed sampler")
    counts = native["counts"]
    count_keys = ("expected_requests", "recorded_outcomes", "native_successful_requests",
                  "native_failed_or_missing_requests", "native_joint_slo_requests")
    if (any(type(counts[k]) is not int or counts[k] < 0 for k in count_keys)
            or counts["expected_requests"] <= 0 or counts["recorded_outcomes"] != counts["expected_requests"]
            or counts["native_successful_requests"] + counts["native_failed_or_missing_requests"] != counts["expected_requests"]
            or counts["native_joint_slo_requests"] > counts["native_successful_requests"]
            or counts.get("all_outcomes_accounted") is not True
            or native.get("missing_outcome_indices") or native.get("invalid_success_timing_indices")
            or not math.isclose(counts["native_joint_slo_rate"], counts["native_joint_slo_requests"]/counts["expected_requests"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("recovered native counts inconsistent")
    for name in ("ttft", "tpot"):
        st = native["native_"+name]
        values = [st[k] for k in ("p50_s", "p90_s", "p95_s", "p99_s", "max_s")]
        if (st["samples"] != counts["native_successful_requests"] or
                any(not finite(v) or v < 0 for v in values+[st["mean_s"]]) or
                values != sorted(values) or st["mean_s"] > st["max_s"]):
            raise ValueError("recovered percentile sample scope differs")
    expected_slo = (counts["native_successful_requests"] == counts["expected_requests"] and
                    counts["native_joint_slo_rate"] >= .9 and
                    native["native_ttft"]["p99_s"] <= point["slo"]["ttft_s"] and
                    native["native_tpot"]["p99_s"] <= point["slo"]["tpot_s"])
    if (type(native["native_request_slo_pass"]) is not bool or native["native_request_slo_pass"] != expected_slo
            or native["client_send_queue_status"] != "unknown_actual_send_and_connector_queue"):
        raise ValueError("recovered SLO verdict or actual-send claim is inconsistent")
    service = power["service"]; pw = service["power"]; per_gpu = pw["per_gpu"]
    uuids = point["engine_identity"]["fleet_gpu_uuids"]
    if (power.get("recovered_service_available") is not True or power.get("power_source_verified") is not True
            or power.get("power_error_affects_service") is not False or len(uuids) != 8 or set(per_gpu) != set(uuids)
            or power["gpu_uuids"] != uuids or method["gpu_uuids"] != uuids
            or pw["status"] != "complete" or pw["coverage_fraction"] != 1.
            or pw["minimum_gpu_coverage_fraction"] != 1. or not finite(pw["max_gap_s"]) or pw["max_gap_s"] > 1.
            or any(r["status"] != "complete" or not finite(r["integral"]) or r["integral"] < 0 for r in per_gpu.values())
            or any(power[k] != native[nk] for k,nk in (("service_start_s","service_started_s"),("service_end_s","service_ended_s")))
            or service["start_s"] != power["service_start_s"] or service["end_s"] != power["service_end_s"]
            or not math.isclose(service["duration_s"], point["duration_s"], rel_tol=0, abs_tol=1e-8)
            or not math.isclose(sum(r["integral"] for r in per_gpu.values()), power["recovered_energy_service_j"], rel_tol=0, abs_tol=1e-6)
            or not math.isclose(pw["energy_j"], power["recovered_energy_service_j"], rel_tol=0, abs_tol=1e-6)):
        raise ValueError("recovered service energy lacks complete bound eight-GPU coverage")
    expected = recovered_columns(native, power, native_ref, power_ref)
    if columns != expected or columns["recovered_used_for_ranking"] is not False:
        raise ValueError("recovered annotation differs from verified diagnostic")


def auxiliary_phase(phase, kind, uuids, max_gap):
    """Validate the small frozen integrator result, without reopening its raw file."""
    data = phase[kind]
    devices = data["per_gpu"]
    if set(devices) != set(uuids):
        raise ValueError("auxiliary phase GPU identity differs")
    for metric in [data, *devices.values()]:
        for key in ("coverage_fraction", "max_gap_s"):
            value = metric.get(key)
            if not finite(value) or value < 0 or (key == "coverage_fraction" and value > 1):
                raise ValueError("invalid auxiliary sampling coverage")
    if (not finite(data.get("minimum_gpu_coverage_fraction")) or
            not math.isclose(data["minimum_gpu_coverage_fraction"],
                             min(d["coverage_fraction"] for d in devices.values()), abs_tol=1e-12)):
        raise ValueError("auxiliary minimum GPU coverage differs")
    complete = (data.get("status") == "complete" and data["coverage_fraction"] == 1
        and data["minimum_gpu_coverage_fraction"] == 1 and data["max_gap_s"] <= max_gap
        and all(d.get("status") == "complete" and d["coverage_fraction"] == 1
                and d["max_gap_s"] <= max_gap for d in devices.values()))
    value_key = "energy_j" if kind == "power" else "mean_pct"
    value = data.get(value_key)
    if not complete:
        if value is not None:
            raise ValueError("incomplete auxiliary phase cannot supply a complete integral")
        return data, False, ""
    if any(not finite(d.get("integral")) or d["integral"] < 0 for d in devices.values()):
        raise ValueError("invalid auxiliary device integral")
    total = sum(d["integral"] for d in devices.values())
    duration = phase["duration_s"]
    expected = total if kind == "power" else total / (duration * len(uuids))
    if (not finite(value) or value < 0 or (kind == "utilization" and value > 100)
            or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("auxiliary aggregate differs from all eight integrals")
    return data, True, value


def auxiliary_columns(window, config, method_ref, artifact_ref):
    fields = {k: "" for k in AUXILIARY_COLUMNS}
    fields.update(auxiliary_status="unavailable", auxiliary_gpu_count=8,
        auxiliary_gpu_uuids_json=json.dumps(config["gpu_uuids"], separators=(",", ":")),
        auxiliary_source_sha256=config["observer_source"]["source_sha256"],
        auxiliary_method_sha256=method_ref["sha256"], auxiliary_artifact_path=artifact_ref["path"],
        auxiliary_artifact_sha256=artifact_ref["sha256"], auxiliary_used_for_ranking=False)
    if window.get("available") is False:
        reason = window.get("unavailable_reason")
        if not isinstance(reason, str) or not reason or "summary" in window:
            raise ValueError("unavailable auxiliary observation must explain missing evidence")
        fields["auxiliary_unavailable_reason"] = reason
        return fields
    if window.get("available") is not True:
        raise ValueError("auxiliary availability must be explicit")
    summary = window["summary"]
    uuids = config["gpu_uuids"]
    if (summary.get("gpu_uuids") != uuids or summary.get("gpu_count") != 8
            or summary.get("gpu_uuid_binding_verified") is not True
            or summary.get("maximum_interpolation_gap_s") != config["max_gap_s"]
            or summary.get("polling_interval_s") != config["interval_s"]
            or summary.get("utilization_timestamp_semantics") != "per_device_acquisition_time"):
        raise ValueError("auxiliary summary identity or sampling method differs")
    for name, start, end in (("service", window["service_start_s"], window["service_end_s"]),
                             ("tail", window["service_end_s"], window["tail_end_s"])):
        phase = summary[name]
        if (not all(finite(t) for t in (start, end)) or end <= start
                or phase["start_s"] != start or phase["end_s"] != end
                or not math.isclose(phase["duration_s"], end-start, abs_tol=1e-8)):
            raise ValueError("auxiliary phase boundaries differ from canonical interval")
    service, service_ok, service_j = auxiliary_phase(summary["service"], "power", uuids, config["max_gap_s"])
    tail, tail_ok, tail_j = auxiliary_phase(summary["tail"], "power", uuids, config["max_gap_s"])
    util, util_ok, mean_util = auxiliary_phase(summary["service"], "utilization", uuids, config["max_gap_s"])
    source_ok = summary.get("power_source_verified") is True and summary.get("power_error_affects_window") is False
    if not source_ok:
        service_ok = tail_ok = False
        service_j = tail_j = ""
    reasons = []
    if not service_ok: reasons.append("service_power_incomplete_or_source_error")
    if not tail_ok: reasons.append("tail_power_incomplete_or_source_error")
    if not util_ok: reasons.append("service_utilization_incomplete")
    fields.update(auxiliary_status="partial" if reasons else "complete",
        auxiliary_energy_service_j=service_j, auxiliary_energy_tail_j=tail_j,
        auxiliary_energy_service_tail_j=service_j+tail_j if service_ok and tail_ok else "",
        auxiliary_service_mean_power_w=service_j/window["duration_s"] if service_ok else "",
        auxiliary_gpu_util_mean_pct=mean_util,
        auxiliary_service_power_coverage_fraction=service["coverage_fraction"],
        auxiliary_service_power_min_gpu_coverage_fraction=service["minimum_gpu_coverage_fraction"],
        auxiliary_service_power_max_gap_s=service["max_gap_s"],
        auxiliary_tail_power_coverage_fraction=tail["coverage_fraction"],
        auxiliary_tail_power_min_gpu_coverage_fraction=tail["minimum_gpu_coverage_fraction"],
        auxiliary_tail_power_max_gap_s=tail["max_gap_s"],
        auxiliary_service_util_coverage_fraction=util["coverage_fraction"],
        auxiliary_service_util_max_gap_s=util["max_gap_s"],
        auxiliary_unavailable_reason=";".join(reasons))
    return fields


def validate_auxiliary(annotation, receipt):
    binding, columns = annotation["binding"], annotation["columns"]
    artifact_ref = dict(path=columns["auxiliary_artifact_path"], sha256=columns["auxiliary_artifact_sha256"])
    artifact = checked_json(artifact_ref["path"], artifact_ref["sha256"])
    if artifact.get("schema") != "auxiliary-window-bound-summary/v1" or artifact.get("binding") != binding:
        raise ValueError("auxiliary artifact row binding differs")
    final_ref, scope_ref = artifact["final"], artifact["scope"]
    final = checked_json(final_ref["path"], final_ref["sha256"])
    scope = checked_json(scope_ref["path"], scope_ref["sha256"])
    if (final.get("schema") != "auxiliary-resident-meter/v1" or final.get("status") != "completed"
            or final.get("auxiliary_only") is not True or final.get("canonical_modified") is not False
            or final.get("restart_count") != 0 or final.get("fragment_joining") is not False
            or scope.get("schema") != "auxiliary-meter-prospective-scope/v1"
            or scope.get("used_for_ranking") is not False
            or scope.get("selection_independent_of_primary_result") is not True
            or scope.get("all_future_pd_windows") is not True
            or scope.get("canonical_columns_modified") is not False):
        raise ValueError("auxiliary scope cannot replace primary or select favorable windows")
    config = checked_json(final["config"]["path"], final["config"]["sha256"])
    method = checked_json(final["method"]["path"], final["method"]["sha256"])
    stop = checked_json(final["stop_request"]["path"], final["stop_request"]["sha256"])
    if (config.get("auxiliary_only") is not True or config.get("production_factory") is not True
            or config["script"] != scope["script"] or config["observer_source"]["manifest"] != scope["observer_source"]
            or config["interval_s"] != scope["sampling_interval_s"] or config["max_gap_s"] != scope["max_gap_s"]
            or scope["created_s"] > config["created_s"]
            or method["gpu_uuids"] != config["gpu_uuids"] or len(set(config["gpu_uuids"])) != 8
            or method.get("maximum_interpolation_gap_s") != config["max_gap_s"]
            or method.get("polling_interval_s") != config["interval_s"]
            or method.get("formal_eligible") is not False
            or stop.get("config") != final["config"] or stop.get("no_next_service_until_finalized") is not True
            or artifact.get("session_id") != config["session_id"]
            or artifact.get("session_id") != receipt.get("session_id")
            or artifact.get("window_index") != receipt.get("window_index")):
        raise ValueError("auxiliary plan, method, or session identity differs")
    reference = dict(path=binding["receipt_path"], sha256=binding["receipt_sha256"])
    matches = [w for w in final["windows"] if w.get("receipt") == reference]
    stopped = [w for w in stop["windows"] if w.get("receipt") == reference]
    if len(matches) != 1 or len(stopped) != 1:
        raise ValueError("auxiliary window missing or duplicated in terminal projection")
    window = matches[0]
    if (any(window.get(k) != v for k, v in stopped[0].items())
            or window.get("auxiliary_only") is not True or window.get("canonical_modified") is not False
            or window.get("used_for_ranking") is not False or window.get("point") != binding["point_id"]):
        raise ValueError("auxiliary terminal-window evidence differs")
    directory = Path(binding["receipt_path"]).parent
    point_ref = dict(path=str(directory/"point.json"), sha256=receipt["artifacts"]["point.json"])
    point = checked_json(point_ref["path"], point_ref["sha256"])
    point_digest = sha(json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (point_digest != binding["point_sha256"] or point.get("name") != binding["point_id"]
            or any(point.get(k) != binding[k] for k in ("system", "run_id", "revision"))
            or point["system"] != "pdblend" or point["run_id"] != scope["run_id"]
            or point["source_manifest"] != scope["observer_source"]
            or point["revision"] not in config["execution_source_sha256"]
            or point["engine_identity"]["fleet_gpu_uuids"] != config["gpu_uuids"]
            or directory.parent.parent != Path(config["session_dir"])):
        raise ValueError("auxiliary primary point or source identity differs")
    groups = [g for g in scope["groups"] if g["model_id"] == point["model_id"]]
    if len(groups) != 1:
        raise ValueError("auxiliary point is outside the prospective model scope")
    group = checked_json(groups[0]["group"]["path"], groups[0]["group"]["sha256"])
    if config["engine_signature"] != group["engine_signature"]:
        raise ValueError("auxiliary continuation changed the frozen engine configuration")
    source = checked_json(scope["observer_source"]["path"], scope["observer_source"]["sha256"])
    if source["source_sha256"] != point["revision"] or config["observer_source"]["source_sha256"] != point["revision"]:
        raise ValueError("auxiliary frozen source differs")
    for key in ("factory", "backend", "power_sampler", "wrapper"):
        ref = method[key]
        relative = str(Path(ref["path"]).relative_to(Path(scope["observer_source"]["path"]).parent))
        if source["files"].get(relative) != ref["sha256"]:
            raise ValueError("auxiliary method is not bound to the frozen execution source")
    raw_ref = final["raw_snapshot"]
    if (Path(raw_ref["path"]).parent != Path(final_ref["path"]).parent
            or not isinstance(raw_ref.get("sha256"), str) or len(raw_ref["sha256"]) != 64):
        raise ValueError("auxiliary raw-evidence reference differs")
    # The raw snapshot is deliberately not read: final.json binds the integrator's
    # projection and its raw digest, just as the primary receipt binds its summary.
    if window["available"]:
        if window.get("point_artifact") != point_ref:
            raise ValueError("auxiliary projected point artifact differs")
        for key, filename in (("result", "result.json"), ("drain", "drain.json")):
            ref = window[key]
            if ref != dict(path=str(directory/filename), sha256=receipt["artifacts"][filename]):
                raise ValueError("auxiliary phase evidence is not primary-receipt-bound")
        result = checked_json(window["result"]["path"], window["result"]["sha256"])
        drain = checked_json(window["drain"]["path"], window["drain"]["sha256"])
        if (drain.get("passed") is not True or receipt.get("cleanup_passed") is not True
                or window["tail_end_s"] != drain["tail_end_s"]
                or any(window[k] != result["metrics"][k] for k in ("service_start_s", "service_end_s", "tail_end_s"))
                or not math.isclose(window["duration_s"], window["service_end_s"]-window["service_start_s"], abs_tol=1e-8)):
            raise ValueError("auxiliary service or tail interval is not primary-bound")
        identity = window["execution_identity"]
        if (identity["source_sha256"] != point["revision"] or identity["gpu_uuids"] != config["gpu_uuids"]
                or any(identity[k] != point["engine_identity"][k] for k in
                       ("image_digest", "model_hash", "tokenizer_hash", "runtime_source_sha256", "measurement_source_sha256"))):
            raise ValueError("auxiliary execution identity differs from measured point")
    elif window.get("unavailable_reason") == "no_recorded_service_result" and "result" in receipt:
        raise ValueError("auxiliary missing-result reason contradicts primary receipt")
    expected = auxiliary_columns(window, config, final["method"], artifact_ref)
    if columns != expected or columns["auxiliary_used_for_ranking"] is not False:
        raise ValueError("auxiliary columns differ from independently projected summary")


def validate_annotation(annotation, schema=SCHEMA):
    binding, columns = annotation["binding"], annotation["columns"]
    if set(binding) != set(BINDING) or not all(type(v) is str and v for v in binding.values()):
        raise ValueError("incomplete or invalid row binding")
    if set(columns) != set(REGISTRY_COLUMNS[schema]):
        raise ValueError("annotation must contain only the eight approved columns" if schema == SCHEMA
                         else "annotation contains unapproved columns")
    receipt = checked_json(binding["receipt_path"], binding["receipt_sha256"])
    if receipt.get("point_sha256") != binding["point_sha256"]:
        raise ValueError("receipt point digest does not match row binding")
    if schema == ENERGY_SCHEMA:
        return validate_energy(annotation, receipt)
    if schema == IO_SCHEMA:
        return validate_io(annotation, receipt)
    if schema == RECOVERED_SCHEMA:
        return validate_recovered(annotation, receipt)
    if schema == AUXILIARY_SCHEMA:
        return validate_auxiliary(annotation, receipt)
    diagnostic = checked_json(columns["writer_startup_diagnostic_path"],
                              columns["writer_startup_diagnostic_sha256"])
    if diagnostic.get("schema") != "pdblend-cold-writer-service-overlap-review/v1":
        raise ValueError("unsupported diagnostic schema")
    if diagnostic.get("receipt") != {"path": binding["receipt_path"],
                                     "sha256": binding["receipt_sha256"]}:
        raise ValueError("diagnostic receipt does not match row binding")
    times = [diagnostic[k] for k in ("writer_launch_recorded_s", "writer_first_publish_s",
                                    "service_start_s", "service_end_s")]
    if not all(finite(v) for v in times):
        raise ValueError("non-finite diagnostic timestamp")
    start, end, service_start, service_end = times
    if end < start or service_end <= service_start:
        raise ValueError("invalid diagnostic interval")
    overlap = max(0.0, min(end, service_end) - max(start, service_start))
    if overlap != diagnostic["launch_to_first_publish_overlap_with_service_s"]:
        raise ValueError("diagnostic overlap cannot be reproduced")
    expected = dict(zip(COLUMNS, (
        "observed_launch_to_first_publish_overlap", overlap, start, end,
        SEMANTICS, "unknown", columns["writer_startup_diagnostic_path"],
        columns["writer_startup_diagnostic_sha256"],
    )))
    if columns != expected or any(isinstance(columns[k], bool) for k in COLUMNS[1:4]):
        raise ValueError("annotation differs from verified diagnostic")
    if diagnostic.get("exact_cpu_export_start_instrumented") is not False:
        raise ValueError("diagnostic does not establish the expected timing scope")


def prepare_csv(data, registry):
    """Validate small bound artifacts and return CSV bytes without changing files."""
    schema = registry.get("schema")
    if schema not in REGISTRY_COLUMNS:
        raise ValueError("unsupported annotation registry")
    annotation_columns = REGISTRY_COLUMNS[schema]
    defaults = annotation_defaults(schema)
    if registry.get("default_columns") != defaults:
        raise ValueError("unannotated rows must not imply zero interference")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields) or not set(BINDING).issubset(fields):
        raise ValueError("invalid CSV header")
    rows = list(reader)
    if any(set(r) != set(fields) or any(v is None for v in r.values()) for r in rows):
        raise ValueError("malformed CSV row")
    wanted = [dict(defaults) for _ in rows]
    selected = set()
    annotations = registry.get("annotations", [])
    if not annotations:
        raise ValueError("empty annotation registry")
    if schema == AUXILIARY_SCHEMA:
        # Selection is all actual windows in the supplied immutable session
        # exports, including unavailable windows, never a choice by energy/SLO.
        finals = registry.get("finals", [])
        if not finals or not registry.get("scope"):
            raise ValueError("auxiliary registry must declare its complete session exports")
        expected_receipts = []
        for ref in finals:
            final = checked_json(ref["path"], ref["sha256"])
            expected_receipts.extend((w["receipt"]["path"], w["receipt"]["sha256"]) for w in final["windows"])
        actual_receipts = [(a["binding"]["receipt_path"], a["binding"]["receipt_sha256"]) for a in annotations]
        if (len(set(expected_receipts)) != len(expected_receipts)
                or sorted(actual_receipts) != sorted(expected_receipts)):
            raise ValueError("auxiliary registry must retain every available and unavailable window")
        for a in annotations:
            artifact = checked_json(a["columns"]["auxiliary_artifact_path"], a["columns"]["auxiliary_artifact_sha256"])
            if artifact["scope"] != registry["scope"] or artifact["final"] not in finals:
                raise ValueError("auxiliary artifact is outside the registry's declared exports")
    for annotation in annotations:
        validate_annotation(annotation, schema)
        binding = annotation["binding"]
        matches = [i for i, row in enumerate(rows) if all(row[k] == binding[k] for k in BINDING)]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError("annotation requires one unique complete row binding")
        if schema == AUXILIARY_SCHEMA:
            artifact = checked_json(annotation["columns"]["auxiliary_artifact_path"], annotation["columns"]["auxiliary_artifact_sha256"])
            if any(rows[matches[0]].get(k) != str(artifact[k]) for k in ("session_id", "window_index")):
                raise ValueError("CSV auxiliary session/window identity differs from bound primary receipt")
        selected.add(matches[0])
        wanted[matches[0]] = {k: str(v) for k, v in annotation["columns"].items()}
    updated = []
    for row, values in zip(rows, wanted):
        for k, value in values.items():
            if row.get(k, "") not in ("", value):
                raise ValueError(f"conflicting existing annotation: {k}")
        updated.append(dict(row, **values))
    if all(all(row.get(k) == values[k] for k in annotation_columns) for row, values in zip(rows, wanted)):
        return data, {"rows": len(rows), "annotated_rows": len(selected), "changed": False,
                      "existing_nonannotation_cells_preserved": len(rows) * len([k for k in fields if k not in annotation_columns])}
    output_fields = fields + [k for k in annotation_columns if k not in fields]
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(updated)
    result = out.getvalue().encode("utf-8")
    reread = list(csv.DictReader(io.StringIO(result.decode("utf-8"), newline="")))
    original_fields = [k for k in fields if k not in annotation_columns]
    if len(rows) != len(reread) or any(a[k] != b[k] for a, b in zip(rows, reread) for k in original_fields):
        raise ValueError("annotation would change an existing nonannotation cell")
    return result, {"rows": len(rows), "annotated_rows": len(selected), "changed": True,
                    "existing_nonannotation_cells_preserved": len(rows) * len(original_fields)}


@contextlib.contextmanager
def exclusive_existing(path):
    # Never create a misspelled lock path or acquire a new, ineffective lock.
    with Path(path).open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def annotate(csv_path, registry_path, registry_sha256, writer_lock, queue_path, *, apply=False):
    csv_path, queue_path = Path(csv_path), Path(queue_path)
    registry = checked_json(registry_path, registry_sha256)
    with exclusive_existing(writer_lock), exclusive_existing(str(queue_path) + ".lock"):
        queue = json.loads(queue_path.read_bytes())
        if not isinstance(queue.get("leases"), dict):
            raise ValueError("queue leases are not readable")
        if any(lease.get("status") == "active" for lease in queue["leases"].values()):
            raise RuntimeError("active GPU lease: annotations require an idle queue")
        original = csv_path.read_bytes()
        result, report = prepare_csv(original, registry)
        report.update(input_csv_sha256=sha(original), output_csv_sha256=sha(result),
                      registry_sha256=registry_sha256, applied=bool(apply),
                      gpu_queue_modified=False, measurement_artifacts_modified=False)
        if apply and report["changed"]:
            fd, temp = tempfile.mkstemp(prefix=csv_path.name + ".annotation-", dir=csv_path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    os.fchmod(handle.fileno(), stat.S_IMODE(csv_path.stat().st_mode))
                    handle.write(result)
                    handle.flush()
                    os.fsync(handle.fileno())
                if csv_path.read_bytes() != original:
                    raise RuntimeError("CSV changed while annotations were prepared")
                os.replace(temp, csv_path)
                directory = os.open(csv_path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                Path(temp).unlink(missing_ok=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("csv", "registry", "registry-sha256", "writer-lock", "queue"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--apply", action="store_true", help="Atomically replace the CSV; default validates only.")
    args = parser.parse_args()
    report = annotate(args.csv, args.registry, args.registry_sha256, args.writer_lock, args.queue, apply=args.apply)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
