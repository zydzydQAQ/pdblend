"""Recompute GPU scalability results from raw request and eight-GPU evidence.

Missing evidence is invalid, never a successful result. Expected arrivals are
the denominator, including explicit admission refusals and request timeouts.
Unknown engineering failures invalidate a run instead of becoming a capacity
failure. No summary supplied by a runner is used by this auditor.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ecopadg.serving.measurement import power_evidence
from .statistics import backlog_stability, quantiles


def _json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _true(value):
    return value is True or value == 1 or isinstance(value, str) and value.lower() in ("1", "true")


def _integer(value):
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _integrate(power, start, end, gpus):
    times = np.asarray([p[0] for p in power])
    values = np.asarray([p[1] for p in power])[:, gpus].sum(axis=1)
    inside = (times > start) & (times < end)
    x = np.concatenate(([start], times[inside], [end]))
    y = np.concatenate(([np.interp(start, times, values)], values[inside], [np.interp(end, times, values)]))
    return float(np.trapezoid(y, x))


def _request(row, raw, expected, slo):
    """Reclassify a request; never trust CSV slo_ok or token_ids_verified."""
    if row is None or raw is None:
        return dict(classification="unknown_failure", error="missing request row or raw output", good=False)
    output = _integer(expected.get("output_len"))
    prompt = _integer(expected.get("prompt_len"))
    count = _integer(raw.get("generated_tokens"))
    ids = raw.get("token_ids")
    ids_valid = isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids)
    ttft = _number(raw.get("ttft", raw.get("ttft_s")))
    latency = _number(raw.get("latency", raw.get("latency_s")))
    raw_start, raw_end = _number(raw.get("arrival_s")), _number(raw.get("finish_s"))
    tpot = ((latency - ttft) / (count - 1) if count and count > 1 and ttft is not None and latency is not None
            else 0. if count == 1 and ttft is not None else None)
    identity = (_integer(row.get("prompt_len")) == prompt and _integer(row.get("output_len")) == output
                and _integer(row.get("generated_tokens")) == count)
    csv_match = all(a is not None and b is not None and math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-8)
                    for a, b in ((_number(row.get("ttft_s")), ttft), (_number(row.get("tpot_s")), tpot)))
    complete = (identity and csv_match and _true(raw.get("success")) and _true(row.get("success"))
                and not raw.get("error") and not row.get("error") and output is not None and output > 0
                and count == output and ids_valid and len(ids) == output
                and raw.get("token_count_source") == "server_usage"
                and row.get("token_count_source") == "server_usage"
                and _integer(raw.get("input_tokens")) == prompt
                and _integer(row.get("input_tokens")) == prompt
                and row.get("output_token_sha256") == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                and ttft is not None and ttft >= 0 and latency is not None and latency >= ttft
                and raw_start is not None and raw_end is not None and latency <= raw_end - raw_start + 1e-6)
    if complete:
        good = ttft < slo[0] and tpot < slo[1]
        exact = (raw.get("token_events_exact") is True and isinstance(raw.get("token_itl"), list)
                 and len(raw["token_itl"]) == count - 1
                 and all(_number(v) is not None and v >= 0 for v in raw["token_itl"]))
        if exact and not math.isclose(sum(raw["token_itl"]), latency - ttft, abs_tol=1e-6, rel_tol=1e-6):
            return dict(classification="unknown_failure", good=False, error="raw token intervals and token timing differ")
        return dict(classification="complete", good=good, ttft_s=ttft, tpot_s=tpot,
                    latency_s=latency, token_itl_s=raw["token_itl"] if exact else None,
                    generated_tokens=count)
    status = _integer(raw.get("http_status", row.get("http_status")))
    rejection = raw.get("admission_rejection") or row.get("admission_rejection")
    error = str(raw.get("error") or row.get("error") or "")
    if (identity and not _true(raw.get("success")) and not _true(row.get("success"))
            and status == 429 and rejection in ("admission_queue_full", "admission_deadline")
            and _integer(row.get("http_status")) == status and row.get("admission_rejection") == rejection
            and count == 0 and ids == [] and ttft is None
            and raw.get("token_count_source") == "missing"):
        return dict(classification="capacity_rejection", good=False, reason=rejection)
    timeout = (raw.get("failure_kind") == "request_timeout"
               or error.startswith(("TimeoutError:", "ServerTimeoutError:", "SocketTimeoutError:")))
    if identity and not _true(raw.get("success")) and not _true(row.get("success")) and timeout:
        return dict(classification="request_timeout", good=False, error=error)
    return dict(classification="unknown_failure", good=False,
                error=error or "incomplete, inconsistent, or unverifiable generated work")


def audit_run(path):
    """Audit a raw directory (or its manifest.json) and return a JSON-safe dict."""
    path = Path(path).resolve()
    directory = path.parent if path.is_file() else path
    errors = []
    try:
        manifest = _json(directory / "manifest.json", {})
        trace = _json(directory / "trace.json", {})
        measurement = _json(directory / "measurement.json", {})
        rows = _read_csv(directory / "bench.csv")
        raw = _json(directory / "outputs.json", [])
        if isinstance(raw, dict):
            raw = raw.get("outputs", [])
        if not isinstance(raw, list):
            raw = []
        requests = trace.get("requests", [])
        source = _json(directory / "power_source.json", {})
        metadata = _jsonl(directory / "power_metadata.jsonl")
        controls = _jsonl(directory / "control.jsonl")
        planning = _jsonl(directory / "planning.jsonl")
        backlog = _jsonl(directory / "backlog.jsonl")
        native_kv = _jsonl(directory / "native-kv.jsonl")
        native_proof = _json(directory / "native-kv-provenance.json", {})
    except (ValueError, TypeError, OSError) as exc:
        return dict(scope="gpu_serving", measurement_valid=False, capacity_pass=False,
                    manifest_path=str(directory / "manifest.json"), audit_errors=[str(exc)])
    if manifest.get("scope", "gpu_serving") != "gpu_serving":
        errors.append("GPU auditor cannot accept another evidence scope")
    native_path = directory / 'native-kv.jsonl'
    if manifest.get('native_kv_capture_required'):
        if (not native_path.is_file() or not native_proof.get('files')
                or native_proof.get('output_sha256') != hashlib.sha256(native_path.read_bytes()).hexdigest()):
            errors.append('native KV event capture missing or changed')
    gpus = manifest.get("allocated_gpu_ids", [])
    if (not isinstance(gpus, list) or not gpus or any(type(g) is not int or not 0 <= g < 8 for g in gpus)
            or len(set(gpus)) != len(gpus) or len(gpus) != manifest.get("n_gpus")):
        errors.append("allocated GPU IDs must be unique measured members of GPU 0..7")
        gpus = []
    start, arrival_end, end = (_number(measurement.get(k)) for k in ("start_s", "arrival_end_s", "end_s"))
    window = _number(manifest.get("arrival_window_s"))
    timing_valid = (None not in (start, arrival_end, end, window) and window > 0
                    and start < arrival_end <= end and math.isclose(arrival_end - start, window, abs_tol=.05))
    if not timing_valid:
        errors.append("missing or inconsistent full arrival and actual drain measurement boundary")
    for flag in ("initial_quiescent", "terminal_quiescent", "hardware_qualification_verified", "source_freeze_verified"):
        if measurement.get(flag) is not True:
            errors.append("missing verified " + flag)
    for field in ("errors", "cleanup_errors"):
        if measurement.get(field):
            errors.append("measurement " + field + ": " + str(measurement[field]))
    trace_digest = _sha(directory / "trace.json") if (directory / "trace.json").is_file() else None
    if manifest.get("trace_sha256") != trace_digest:
        errors.append("trace digest missing or differs from manifest")
    hashes = manifest.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        errors.append("source freeze hashes absent")
    else:
        for name, digest in hashes.items():
            frozen = Path(name)
            if not frozen.is_absolute():
                frozen = directory / frozen
            if not frozen.is_file() or _sha(frozen) != digest:
                errors.append("source freeze mismatch: " + str(name))
    slo = (_number(manifest.get("slo_ttft_s")), _number(manifest.get("slo_tpot_s")))
    if any(v is None or v <= 0 for v in slo):
        errors.append("finite positive TTFT and mean-TPOT SLO required")
        slo = (0., 0.)
    expected_ids = [str(q.get("request_id", index)) for index, q in enumerate(requests)]
    observed_ids = [str(r.get("request_id")) for r in rows]
    raw_ids = [str(r.get("request_id")) for r in raw]
    if (not requests or len(set(expected_ids)) != len(requests) or sorted(observed_ids) != sorted(expected_ids)
            or sorted(raw_ids) != sorted(expected_ids)):
        errors.append("exactly one bench row and raw output per offered trace request required")
    by_id, raw_by_id = {str(r.get("request_id")): r for r in rows}, {str(r.get("request_id")): r for r in raw}
    audited = []
    for rid, request in zip(expected_ids, requests):
        row, output = by_id.get(rid), raw_by_id.get(rid)
        detail = _request(row, output, request, slo)
        detail["request_id"] = rid
        audited.append(detail)
        output_length = _integer(request.get("output_len"))
        if output_length is None or output_length < 2:
            errors.append("at least two output tokens required for mean TPOT: " + rid)
        expected_timeout = max(120., slo[0] + (output_length - 1) * slo[1] + 30.) if output_length else None
        declared_timeout = _number(request.get("timeout_s"))
        raw_timeout = _number(output.get("declared_timeout_s")) if output else None
        if (expected_timeout is None or declared_timeout != expected_timeout or raw_timeout != expected_timeout):
            errors.append("per-request timeout declaration missing or differs from protocol: " + rid)
        if timing_valid:
            relative = _number(request.get("arrival_s"))
            observed = _number(row.get("arrival_s")) if row else None
            finished = _number(row.get("finish_s")) if row else None
            if (relative is None or not 0 <= relative < window or observed is None or finished is None
                    or not start <= observed < arrival_end or not observed <= finished <= end + .05
                    or not math.isclose(observed - start, relative, abs_tol=1e-6, rel_tol=0.)):
                errors.append("request outside declared measurement boundary: " + rid)
            raw_arrival = _number(output.get("arrival_s")) if output else None
            raw_finish = _number(output.get("finish_s")) if output else None
            if (observed is None or finished is None or raw_arrival is None or raw_finish is None
                    or not math.isclose(raw_arrival, observed, abs_tol=1e-6, rel_tol=0.)
                    or not math.isclose(raw_finish, finished, abs_tol=1e-6, rel_tol=0.)):
                errors.append("raw output and bench arrival/finish timestamps differ: " + rid)
            # A declared request deadline may fire later under event-loop load,
            # but an early network timeout cannot stand in for this protocol limit.
            if (detail["classification"] == "request_timeout" and raw_finish is not None and raw_arrival is not None
                    and expected_timeout is not None and raw_finish - raw_arrival < expected_timeout - .1):
                errors.append("request timed out before its declared deadline: " + rid)
    counts = Counter(r["classification"] for r in audited)
    if counts["unknown_failure"]:
        errors.append("unknown engineering failures or incomplete output evidence")
    good = sum(r["good"] for r in audited)
    attainment = good / len(requests) if requests else None
    power = []
    utilization = []
    try:
        for r in _read_csv(directory / "power.csv"):
            power.append((float(r["t_s"]), [float(r[f"gpu{g}_w"]) for g in range(8)]))
            utilization.append([_number(r.get(f"gpu{g}_util_pct")) for g in range(8)])
        if (len(power) < 2 or any(not math.isfinite(t) or any(not math.isfinite(w) or w < 0 for w in watts)
                                 for t, watts in power)
                or any(b[0] <= a[0] for a, b in zip(power, power[1:]))):
            raise ValueError("invalid eight-GPU power samples")
    except (KeyError, ValueError, TypeError):
        errors.append("missing or invalid raw eight-GPU power samples")
        power = []
    provenance = power_evidence(power, source, metadata)
    if not provenance["power_source_verified"]:
        errors.extend(provenance["power_source_errors"])
    power_coverage = bool(timing_valid and power and power[0][0] <= start and power[-1][0] >= end)
    if not power_coverage:
        errors.append("power samples must bracket the full arrival plus actual drain window")
    if measurement.get("sampling_error"):
        errors.append("power sampling error: " + str(measurement["sampling_error"]))
    energy_alloc = _integrate(power, start, end, gpus) if power_coverage and gpus else None
    energy_node = _integrate(power, start, end, list(range(8))) if power_coverage else None
    stability = backlog_stability(backlog, arrival_end) if timing_valid else dict(valid=False, upper_rps=None)
    if not stability["valid"]:
        errors.append("missing or invalid backlog stability evidence")
    rate = _number(manifest.get("rate_rps"))
    if rate is None or rate <= 0:
        errors.append("finite positive arrival rate required")
    duration = end - start if timing_valid else None
    route_counts = Counter()
    transfer_bytes = []
    for event in controls:
        event_time = _number(event.get("at_s", event.get("t_s")))
        if not timing_valid or event_time is None or not start <= event_time <= end:
            continue
        if event.get("kind") == "admission":
            plan = event.get("plan", {})
            routes = plan.get("routes", []) if isinstance(plan, dict) else []
            for route in routes:
                route_counts["mixed" if route.get("prefill_id") == route.get("decode_id") else "pd"] += 1
        value = _number(event.get("kv_transfer_bytes"))
        if value is not None:
            transfer_bytes.append(value)
    token_itl = [v for r in audited for v in (r.get("token_itl_s") or [])]
    valid = not errors
    formal_eligible = (measurement.get("declared_formal") is True
                       and manifest.get("formal_eligible") is True
                       and manifest.get("stage") in ("capacity", "weak"))
    capacity_eligible = formal_eligible or (manifest.get("stage") == "pilot"
        and measurement.get("hardware_qualification_verified") is True
        and measurement.get("source_freeze_verified") is True)
    planning = [event for event in planning if timing_valid
                and _number(event.get("at_s", event.get("t_s"))) is not None
                and start <= _number(event.get("at_s", event.get("t_s"))) <= end]
    planner_latencies = [_number(r.get("elapsed_ms")) / 1000 for r in planning
        if r.get("kind") == "admission_planning" and _number(r.get("elapsed_ms")) is not None]
    lock_latencies = [_number(r.get("elapsed_ms")) / 1000 for r in planning
        if r.get("kind") == "action_lock_wait" and _number(r.get("elapsed_ms")) is not None]
    plan_count = len(planner_latencies)
    result = {key: manifest.get(key) for key in ("system", "dataset", "n_gpus", "seed", "rate_rps", "stage",
              "model", "host_id", "profile_sha256", "source_config_sha256", "source_hashes", "q",
              "slo_ttft_s", "slo_tpot_s")}
    result.update(scope="gpu_serving", audit_schema=1, manifest_path=str(directory / "manifest.json"),
        trace_sha256=trace_digest,
        measurement_valid=valid, audit_errors=sorted(set(errors)), formal_eligible=formal_eligible,
        capacity_valid=bool(valid and capacity_eligible),
        capacity_pass=bool(valid and capacity_eligible and attainment >= .9 and stability["upper_rps"] <= .01 * rate),
        capacity_stability_upper_rps=stability.get("upper_rps"), capacity_stability=stability,
        offered_requests=len(requests), good_requests=good, classification_counts=dict(counts),
        slo_attainment=attainment, goodput_rps=good / duration if duration else None,
        throughput_rps=counts["complete"] / duration if duration else None,
        measurement_start_s=start, arrival_end_s=arrival_end, measurement_end_s=end,
        measurement_duration_s=duration, actual_drain_s=end - arrival_end if timing_valid else None,
        allocated_gpu_ids=gpus, energy_allocated_j=energy_alloc, energy_node8_j=energy_node,
        joules_per_good_request=energy_alloc / good if energy_alloc is not None and good else None,
        latency_s=quantiles([r.get("latency_s") for r in audited]),
        ttft_s=quantiles([r.get("ttft_s") for r in audited]),
        tpot_s=quantiles([r.get("tpot_s") for r in audited]), token_itl_s=quantiles(token_itl),
        token_itl_count=len(token_itl), route_counts=dict(route_counts) if route_counts else None,
        planning_latency_s=quantiles(planner_latencies), action_lock_wait_s=quantiles(lock_latencies),
        planner_budget_fallback_ratio=(sum(r.get("kind") == "admission_planning" and r.get("fallback") is True
                                          for r in planning) / plan_count if plan_count else None),
        planner_stale_retry_ratio=(sum(r.get("kind") == "discard" and r.get("reason") == "stale"
                                      for r in planning) / plan_count if plan_count else None),
        kv_transfer_bytes=sum(transfer_bytes) if transfer_bytes else None,
        request_audit=audited, power_evidence=provenance,
        claims="raw GPU serving evidence within the recorded host and allocated GPU subset")
    if timing_valid:
        from .native_logs import summarize_kv
        try:
            result.update(summarize_kv(native_kv, controls, start, end))
        except (KeyError, ValueError, TypeError) as exc:
            result['audit_errors'].append('invalid native KV timing: '+str(exc))
            result.update(measurement_valid=False,capacity_valid=False,capacity_pass=False)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="raw run directory or its manifest.json")
    parser.add_argument("--out", type=Path, help="optional independently audited JSON result")
    args = parser.parse_args()
    result = audit_run(args.run)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
