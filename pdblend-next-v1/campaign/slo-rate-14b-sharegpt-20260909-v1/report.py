"""CPU-only reports of audited 14B ShareGPT SLO-rate observations.

Input: A|C/observations.json[l], A|C/measurements/*/audited.json, and
B/reference.json = {"source": {"path": ..., "sha256": ...}, "cell_ids": [...]}
(cell_ids is optional). An observation may be flat or {row, metrics}. JSON
containers may use observations/points or observation. Published new cells also
receive independent local core CSV checks. No GPU work, historical edits,
qualification replay, confidence intervals, or cross-host ratios.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

HOST_SCALES = {"A": 0.5, "B": 1.0, "C": 2.0}
SYSTEMS = ("pdblend", "mixed", "distserve", "ecoserve", "dynamollm")
FINAL_MEASUREMENT_COUNTS = {"A": 21, "C": 41}
NUMBERS = ("rate_rps", "slo_scale", "slo_attainment", "slo_ttft_s", "slo_tpot_s",
           "ttft_avg_s", "tpot_avg_s", "goodput_measurement_rps", "energy_j",
           "energy_per_good_request_j", "gpu_util", "measurement_duration_s",
           "completion_fraction", "n_expected", "completed_work_requests", "good_requests",
           "full_operation_energy_j", "measurement_start_s", "measurement_end_s")
COLUMNS = ("cell_id", "engineering_attempt", "measurement_host", "slo_scale", "system", "rate_rps", "repeat",
           "reference_only", "status", "report_eligible", "measurement_valid", "work_complete",
           "slo_pass", "slo_attainment", "slo_ttft_s", "slo_tpot_s", "ttft_avg_s",
           "tpot_avg_s", "goodput_measurement_rps", "energy_j", "energy_per_good_request_j",
           "gpu_util", "energy_measured_gpu_count", "n_expected", "completed_work_requests",
           "good_requests", "completion_fraction", "measurement_duration_s",
           "measurement_start_s", "measurement_end_s", "full_operation_energy_j",
           "seed", "arrival_window_s", "trace_sha256", "source_integrity", "metric_source_integrity",
           "qualification_mirror_status", "qualification_inventory_status", "full_checkpoint_inventory_status", "local_context_complete", "audit_accepted",
           "missing_metrics", "report_issues", "report_source", "raw_requests", "raw_power",
           "checkpoint", "receipt", "summary", "binding")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def records(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise ValueError("observation input must be an object or list")
    for key in ("observations", "points"):
        if key in value:
            return records(value[key])
    if "observation" in value:
        return records(value["observation"])
    if "cell_id" in value or "row" in value:
        return [value]
    return []


def flatten(value):
    result = dict(value.get("row", {}))
    result.update(value.get("metrics", {}))
    result.update({k: v for k, v in value.items() if k not in ("row", "metrics")})
    return clean(result)


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield {"path": value["path"], "sha256": value["sha256"]}
        for key, child in value.items():
            if key == "artifacts" and isinstance(child, dict):
                for path, digest in child.items():
                    if isinstance(digest, str):
                        yield {"path": path, "sha256": digest}
            elif key != "report_source":
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def number(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize(value, host, source, *, reference_only=False):
    result = flatten(value)
    if not result.get("cell_id"):
        raise ValueError("observation lacks cell_id: " + source["path"])
    issues = []
    claimed_host = result.get("measurement_host", result.get("node"))
    if claimed_host and claimed_host not in (host, "Anew20260909" if host == "A" else host):
        issues.append("physical_host_mismatch")
    result.update(source_measurement_host=claimed_host, measurement_host=host,
                  reference_only=reference_only, report_source=source)
    for key in NUMBERS:
        result[key] = number(result.get(key))
    if result["slo_scale"] != HOST_SCALES[host]:
        issues.append("host_scale_mismatch")
    if result.get("model") != "14b" or result.get("dataset") != "sharegpt":
        issues.append("model_dataset_mismatch")
    if result.get("system") not in SYSTEMS:
        issues.append("unknown_system")
    if result.get("seed", result.get("arrival_seed")) != 701:
        issues.append("seed_mismatch_or_missing")
    if result.get("arrival_window_s", result.get("trace_duration_s")) != 100:
        issues.append("arrival_window_mismatch_or_missing")
    if result.get("energy_measured_gpu_count") != 8:
        issues.append("eight_gpu_scope_not_verified")
    if not reference_only:
        for key in ("strict_slo_recomputed", "service_terminal_valid", "actual_slo_config_verified"):
            if result.get(key) is not True:
                issues.append(key + "_not_verified")
        if result.get("work_complete") is False and result.get("capacity_failures_independently_audited") is not True:
            issues.append("incomplete_work_not_independently_classified")
    raw = result.get("verification", {}).get("raw", {})
    result["audit_accepted"] = (result.get("independently_recomputed") is True or
        (raw.get("raw_requests_recomputed") is True and raw.get("all_eight_gpu_energy_reintegrated") is True))
    result["slo_pass"] = result["slo_attainment"] >= 0.9 if result["slo_attainment"] is not None else None
    # Derive arithmetic only from present denominators; undefined remains null.
    energy, good = result["energy_j"], result["good_requests"]
    if result["energy_per_good_request_j"] is None and energy is not None and good is not None and good > 0:
        result["energy_per_good_request_j"] = energy / good
    result["missing_metrics"] = [key for key in ("slo_attainment", "ttft_avg_s", "tpot_avg_s",
        "goodput_measurement_rps", "energy_j", "energy_per_good_request_j", "gpu_util") if result[key] is None]
    result["report_issues"] = issues
    return result


def collect(root):
    root = Path(root)
    loaded, inputs = [], []
    for host in HOST_SCALES:
        directory = root / host
        if host == "B":
            path = directory / "reference.json"
            if not path.exists():
                continue
            config = read(path)
            inputs.append(ref(path))
            source = config["source"]
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("reference source hash changed: " + source["path"])
            inputs.append(source)
            selected = set(config.get("cell_ids", []))
            for value in records(read(source["path"])):
                row = flatten(value)
                if (row.get("model"), row.get("dataset"), row.get("measurement_host", row.get("node")),
                    number(row.get("slo_scale"))) != ("14b", "sharegpt", "B", 1.0):
                    continue
                if selected and row["cell_id"] not in selected:
                    continue
                loaded.append(normalize(value, host, source, reference_only=True))
            continue
        paths = [p for p in (directory / "observations.jsonl", directory / "observations.json") if p.exists()]
        paths.extend(sorted((directory / "measurements").glob("*/audited.json")))
        for path in paths:
            source = ref(path)
            inputs.append(source)
            values = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                      if path.suffix == ".jsonl" else [read(path)])
            for container in values:
                for value in records(container):
                    loaded.append(normalize(value, host, source))
    distinct = {}
    for row in loaded:
        key = (row["cell_id"], row.get("engineering_attempt", 1))
        if key in distinct:
            previous = {k: v for k, v in distinct[key].items() if k != "report_source"}
            current = {k: v for k, v in row.items() if k != "report_source"}
            # A supervisor may add the immutable audited-file reference and
            # attempt metadata to the otherwise identical per-cell observation.
            if any(previous[k] != current[k] for k in previous.keys() & current.keys()):
                raise ValueError("conflicting observations for cell_id/attempt: " + str(key))
            distinct[key].update(current)
        else:
            distinct[key] = row
    return sorted(distinct.values(), key=lambda x: (x["measurement_host"], x.get("rate_rps") or -1,
        x.get("system", ""), x.get("repeat", 1), x["cell_id"])), inputs


def audit_sources(rows, inputs):
    index = {}
    qualification_cache = {}

    def verify(reference):
        key = (reference["path"], reference["sha256"])
        if key not in index:
            path = Path(reference["path"])
            actual = sha(path) if path.is_file() else None
            index[key] = dict(reference, actual_sha256=actual,
                status="verified" if actual == reference["sha256"] else "missing" if actual is None else "mismatch")
        return index[key]["status"]

    def qualification_inventory(reference):
        """Inspect declared local inventory only; never download or execute it."""
        key = (reference["path"], reference["sha256"])
        if key in qualification_cache:
            return qualification_cache[key]
        statuses = {verify(reference)}
        qualification_cache[key] = statuses
        if statuses != {"verified"}:
            return statuses
        try:
            document = read(reference["path"])
            declared = list(references(document))
            for inventory in (document.get("files", {}), document.get("source_files", {})):
                if isinstance(inventory, dict):
                    declared.extend(dict(path=path, sha256=digest) for path, digest in inventory.items()
                        if isinstance(path, str) and path.startswith("/") and isinstance(digest, str) and len(digest) == 64)
            statuses.update(verify(item) for item in declared)
            previous = document.get("previous_qualification")
            if isinstance(previous, dict) and "path" in previous and "sha256" in previous:
                statuses.update(qualification_inventory(previous))
        except (OSError, ValueError, AttributeError):
            statuses.add("unreadable")
        return statuses

    for reference in inputs:
        verify(reference)
    for row in rows:
        refs = list(references(row))
        core_refs = [row["report_source"]]
        for key in ("raw_requests", "raw_power", "checkpoint", "receipt", "summary", "binding",
                    "trace_reference", "audit_reference"):
            if isinstance(row.get(key), dict):
                core_refs.extend(references(row[key]))
        if row.get("trace") and row.get("trace_sha256"):
            core_refs.append(dict(path=row["trace"], sha256=row["trace_sha256"]))
        # Expand the checkpoint inventory to include raw artifacts, without
        # loading large raw traces or recomputing measurements in this reporter.
        checkpoint = row.get("checkpoint")
        inventory_statuses = set()
        if isinstance(checkpoint, dict) and verify(checkpoint) == "verified":
            try:
                cp = read(checkpoint["path"])
                refs.extend(references(cp))
                inventory_statuses = {verify(item) for item in references({"artifacts": cp.get("artifacts", {})})}
            except (ValueError, OSError):
                row["report_issues"].append("checkpoint_unreadable")
        qualification = row.get("qualification")
        qualification_statuses = qualification_inventory(qualification) if isinstance(qualification, dict) else set()
        statuses = {verify(reference) for reference in refs} | qualification_statuses
        core_statuses = {verify(reference) for reference in core_refs}
        for key in ("raw_requests", "raw_power", "checkpoint"):
            if not isinstance(row.get(key), dict):
                row["report_issues"].append(key + "_reference_missing")
        row["source_integrity"] = "mismatch" if "mismatch" in statuses else "missing" if statuses & {"missing", "unreadable"} else "verified"
        row["metric_source_integrity"] = "mismatch" if "mismatch" in core_statuses else "missing" if "missing" in core_statuses else "verified"
        row["qualification_mirror_status"] = verify(qualification) if isinstance(qualification, dict) else "not_referenced"
        row["qualification_inventory_status"] = ("mismatch" if "mismatch" in qualification_statuses else
            "missing" if "missing" in qualification_statuses else "unreadable" if "unreadable" in qualification_statuses else
            "verified" if qualification_statuses else "not_referenced")
        row["full_checkpoint_inventory_status"] = ("mismatch" if "mismatch" in inventory_statuses else
            "missing" if "missing" in inventory_statuses else "verified" if inventory_statuses else "not_referenced")
        row["local_context_complete"] = row["source_integrity"] == "verified"
        row["report_eligible"] = (row.get("measurement_valid") is True and row["audit_accepted"] and
            row["metric_source_integrity"] == "verified" and not row["report_issues"])
        row["status"] = (("completed" if row["local_context_complete"] else "audited_metrics_context_pending") if row["report_eligible"] else
            "engineering_invalid" if row.get("measurement_valid") is False else "unverified")
    return sorted(index.values(), key=lambda x: (x["path"], x["sha256"]))


def boundaries(rows):
    groups = []
    for host, scale in HOST_SCALES.items():
        local = [row for row in rows if row["measurement_host"] == host]
        pdb = [row for row in local if row["system"] == "pdblend" and row["report_eligible"]]
        loss = [row for row in pdb if row["slo_pass"] is False]
        complete_loss = [row for row in loss if row.get("work_complete") is True]
        first = min((row["rate_rps"] for row in loss if row["rate_rps"] is not None), default=None)
        complete_first = min((row["rate_rps"] for row in complete_loss if row["rate_rps"] is not None), default=None)
        groups.append(dict(measurement_host=host, slo_scale=scale, reference_only=host == "B",
            reporting_status=("reference_available" if local else "reference_missing") if host == "B" else
                ("observations_available_not_completion_certified" if local else "awaiting_observations"),
            observations=len(local), eligible_observations=sum(row["report_eligible"] for row in local),
            first_slo_loss_rps=first, first_complete_slo_loss_rps=complete_first,
            first_loss_cell_ids=[row["cell_id"] for row in loss if row["rate_rps"] == first],
            boundary_repeat_observed=any(row.get("repeat", 1) > 1 for row in pdb if row["rate_rps"] == first),
            measured_rates=sorted({row["rate_rps"] for row in local if row["rate_rps"] is not None}),
            low_slo_observations=sum(row["slo_pass"] is False and row["report_eligible"] for row in local),
            incomplete_work_observations=sum(row.get("work_complete") is False for row in local),
            missing_metrics=sum(bool(row["missing_metrics"]) for row in local),
            system_counts={system: sum(row["system"] == system and row["report_eligible"] for row in local) for system in SYSTEMS},
            claim="Descriptive service boundary on this host; no cross-host causal comparison or independent-seed CI."))
    return groups


def energy_ledger(root):
    """Keep setup/failed operation measurements separate; never sum overlapping windows."""
    result = []
    for host in HOST_SCALES:
        for name in ("setup-energy-ledger.json", "failure-ledger.json", "energy-ledger.json"):
            path = Path(root) / host / name
            if not path.exists():
                continue
            value = read(path)
            entries = value if isinstance(value, list) else value.get("entries", value.get("rows", []))
            for entry in entries:
                result.append(dict(measurement_host=host, category=name, source=ref(path),
                    ledger_entry=entry, primary_plus_outer_total_j=None,
                    accounting_note="Primary and outer windows may overlap; no sum or campaign total computed."))
    return result


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(row.get(key)), ensure_ascii=False, allow_nan=False)
                if isinstance(row.get(key), (dict, list)) else clean(row.get(key)) for key in fields})


def grid_series(rows, metric, multiplier=1, *, upper_rate=None):
    """Keep absent and ineligible rates as actual line breaks on the .25 grid."""
    by_rate = {}
    for row in rows:
        rate = row.get("rate_rps")
        if rate is not None and math.isfinite(rate):
            previous = by_rate.get(rate)
            if previous is None or row.get("engineering_attempt", 1) > previous.get("engineering_attempt", 1):
                by_rate[rate] = row
    upper = upper_rate if upper_rate is not None else max(by_rate, default=0)
    rates = [i / 4 for i in range(1, int(round(upper * 4)) + 1)]
    values = []
    for rate in rates:
        row = by_rate.get(rate)
        value = row.get(metric) if row and row.get("report_eligible") else None
        values.append(value * multiplier if value is not None and math.isfinite(value) else math.nan)
    return rates, values


def load_local_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_supervisor(root, node):
    """Read only direct node supervisors; qualification statuses cannot win."""
    candidates = []
    for path in sorted((Path(root) / node).glob("*/status.json")):
        try:
            before = path.read_bytes()
            value = json.loads(before)
            if (value.get("schema") != "slo-rate-node-status-v1" or value.get("node") != node or
                type(value.get("started_s")) not in (int, float) or
                not math.isfinite(value["started_s"]) or value["started_s"] <= 0 or
                type(value.get("pid")) is not int or value["pid"] <= 0 or
                not isinstance(value.get("startticks"), str) or not value["startticks"].isdigit()):
                continue
            if path.read_bytes() != before:
                continue
            reference = dict(path=str(path), sha256=hashlib.sha256(before).hexdigest())
            candidates.append((value["started_s"], str(path), value, reference))
        except (OSError, ValueError, AttributeError):
            continue
    if not candidates:
        return None, None
    _, _, status, reference = max(candidates, key=lambda item: item[:2])
    return status, reference


def completion_status(root, rows, checks):
    """Certify completed measurement grids independently of deferred context."""
    root = Path(root)
    contract = load_local_module(Path(__file__).with_name("contract.py"), "slo_report_completion_contract")
    checker_ref = checks.get("checker", {})
    checker_current = (checker_ref.get("path") == str(Path(__file__).with_name("reports") / "crosscheck.py") and
        checker_ref.get("sha256") == sha(Path(__file__).with_name("reports") / "crosscheck.py"))
    check_entries = {}
    for entry in checks.get("entries", []):
        check_entries.setdefault((entry.get("cell_id"), entry.get("engineering_attempt", 1)), []).append(entry)
    assessed, sources = {}, []
    workloads = {}

    def materializer(rate):
        rate = contract.number(rate)
        if rate not in workloads:
            path = root / "workloads" / ("r" + rate) / "workload.json"
            workload = read(path)
            # Verify the frozen workload against its pinned manifest and executed
            # trace. The 300-second source is deferred context, not a new replay.
            manifest_ref, trace_ref = workload["materialization_manifest"], workload["trace_reference"]
            observed = [row for row in rows if not row.get("reference_only") and
                        contract.number(row["rate_rps"]) == rate]
            if any(row.get("materialization_manifest") != manifest_ref or
                   row.get("trace_sha256") != trace_ref["sha256"] for row in observed):
                raise ValueError("workload manifest or trace differs from the immutable observations")
            if sha(manifest_ref["path"]) != manifest_ref["sha256"] or sha(trace_ref["path"]) != trace_ref["sha256"]:
                raise ValueError("frozen workload manifest or trace changed")
            manifest, trace = read(manifest_ref["path"]), read(trace_ref["path"])
            payload = dict(workload)
            payload.pop("materialization_manifest")
            if (hashlib.sha256(contract.encode(payload)).hexdigest() != manifest["workload_payload_sha256"] or
                workload["campaign_id"] != contract.CAMPAIGN or contract.number(workload["rate_rps"]) != rate or
                Path(trace_ref["path"]).resolve() != (path.parent / "trace.json").resolve() or
                trace_ref["sha256"] != workload["trace_sha256"] or
                trace["content_pairing_sha256"] != workload["content_pairing_sha256"] or
                trace["n_requests"] != workload["n_expected"]):
                raise ValueError("frozen workload payload or identity changed")
            workloads[rate] = workload
            sources.append(ref(path))
            sources.extend((manifest_ref, trace_ref))
        return workloads[rate]

    for node, expected_count in FINAL_MEASUREMENT_COUNTS.items():
        local = [row for row in rows if row["measurement_host"] == node and not row.get("reference_only")]
        status, status_ref = latest_supervisor(root, node)
        reasons, decision = [], None
        assessment = dict(complete=False, expected_measurements=expected_count, observed_measurements=len(local),
            supervisor=status_ref, supervisor_phase=status.get("phase") if status else None,
            full_context_mirror_status="verified" if local and all(row.get("local_context_complete") for row in local) else "pending",
            pdb_first_loss_rps=None, pdb_boundary_confirmed=False, expected_cell_ids=[], reasons=reasons)
        if status_ref:
            sources.append(status_ref)
        if not status:
            reasons.append("supervisor_missing")
        elif (status.get("complete") is not True or status.get("five_system_complete") is not True or
            status.get("phase") != "complete" or status.get("node_lease_held") is not False or
            type(status.get("finished_s")) not in (int, float) or not math.isfinite(status["finished_s"]) or
            status["finished_s"] < status["started_s"] or
            status.get("error") or status.get("unsettled_child")):
            reasons.append("supervisor_not_clean_terminal_complete")
        elif status.get("child") and (status["child"].get("exitcode") != 0 or
            not status["child"].get("finished_s") or status["child"].get("physical_lease_release_not_certified")):
            reasons.append("supervisor_child_not_terminal")
        if len(local) != expected_count:
            reasons.append("expected_measurement_count_not_met")
        canonical = []
        for row in local:
            cid = row["cell_id"]
            raw_ref = row.get("audit_reference")
            try:
                if (not isinstance(raw_ref, dict) or sha(raw_ref["path"]) != raw_ref["sha256"]):
                    raise ValueError("audit hash missing or changed")
                audit_rows = records(read(raw_ref["path"]))
                if len(audit_rows) != 1:
                    raise ValueError("expected one immutable audited observation")
                raw = flatten(audit_rows[0])
                restored = dict(row, measurement_host=row.get("source_measurement_host", row["measurement_host"]))
                if any(restored.get(key) != value for key, value in raw.items()):
                    raise ValueError("aggregate observation differs from its audited source")
                if (raw.get("measurement_valid") is not True or row.get("report_eligible") is not True or
                    row.get("metric_source_integrity") != "verified" or row.get("engineering_attempt", 1) != 1):
                    raise ValueError("measurement or core source is not verified")
                raw.update(audit_reference=raw_ref, engineering_attempt=row.get("engineering_attempt", 1))
                canonical.append(raw)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reasons.append("invalid_audit:" + cid + ":" + str(exc))
            entries = check_entries.get((cid, row.get("engineering_attempt", 1)), [])
            if (not checker_current or len(entries) != 1 or entries[0].get("status") != "passed" or
                entries[0].get("audit_reference") != raw_ref):
                reasons.append("core_crosscheck_not_passed:" + cid)
        if checks.get("issues"):
            reasons.append("core_crosscheck_or_trace_pairing_issue")
        if canonical:
            try:
                decision = contract.evaluate_group(node, canonical, materializer=materializer)
                assessment["pdb_first_loss_rps"] = decision.get("rate_rps") if decision.get("cap_observed") else None
                assessment["pdb_boundary_confirmed"] = decision.get("status") == "cap_confirmed"
                if decision.get("status") == "cap_confirmed":
                    grid = decision["eligible_rates"]
                    expected = {contract.cell_id(node, rate, system) for rate in grid for system in contract.SYSTEMS}
                    expected.add(contract.cell_id(node, decision["rate_rps_decimal"], "pdblend", 2))
                    assessment["expected_cell_ids"] = sorted(expected)
                    if len(expected) != expected_count or {row["cell_id"] for row in canonical} != expected:
                        reasons.append("contract_grid_missing_or_extra_cells")
                if decision.get("complete") is not True:
                    reasons.append("contract_grid_incomplete")
                if status and status.get("complete") is True and status.get("decision") != decision:
                    reasons.append("supervisor_decision_differs_from_contract")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reasons.append("contract_validation_failed:" + str(exc))
        else:
            reasons.append("no_verified_observations")
        assessment["complete"] = not reasons
        assessed[node] = assessment
    return dict(complete=all(value["complete"] for value in assessed.values()), nodes=assessed,
        full_context_mirror_status="verified" if all(value["full_context_mirror_status"] == "verified" for value in assessed.values()) else "pending"), sources


def figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = (("slo_attainment", "SLO attainment (%)", 100), ("ttft_avg_s", "Mean TTFT (s)", 1),
        ("tpot_avg_s", "Mean TPOT (ms)", 1000), ("goodput_measurement_rps", "Goodput / full measurement (req/s)", 1),
        ("energy_j", "Eight-GPU primary energy (kJ)", 0.001),
        ("energy_per_good_request_j", "Energy / SLO-qualified request (J)", 1))
    colors = dict(zip(SYSTEMS, ("#c63e35", "#ba8517", "#178369", "#667f99", "#8657a6")))
    for host, scale in HOST_SCALES.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        local = [row for row in rows if row["measurement_host"] == host]
        upper_rate = max((row["rate_rps"] for row in local if row["rate_rps"] is not None), default=0)
        for ax, (metric, label, multiplier) in zip(axes.flat, metrics):
            for system in SYSTEMS:
                values = [row for row in local if row["system"] == system]
                for repeat in sorted({row.get("repeat", 1) for row in values}):
                    series = sorted([row for row in values if row.get("repeat", 1) == repeat], key=lambda row: row["rate_rps"])
                    rates, plotted = grid_series(series, metric, multiplier, upper_rate=upper_rate)
                    ax.plot(rates, plotted,
                        color=colors[system], marker="o" if repeat == 1 else "s", ls="-" if repeat == 1 else "--",
                        label=system + (" / repeat " + str(repeat) if repeat > 1 else ""))
                    incomplete = [row for row in series if row["report_eligible"] and row.get("work_complete") is False and row[metric] is not None]
                    ax.scatter([row["rate_rps"] for row in incomplete], [row[metric] * multiplier for row in incomplete],
                        color="black", marker="x", s=70, zorder=5)
            if metric == "slo_attainment":
                ax.axhline(90, color="#777", linestyle=":", linewidth=1)
                ax.set_ylim(0, 105)
            ax.set(xlabel="Offered rate (req/s)", ylabel=label)
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            if not any(row["report_eligible"] for row in local):
                ax.text(0.5, 0.5, "No audited observations", transform=ax.transAxes, ha="center")
        fig.suptitle(f"14B ShareGPT · host {host} · SLO {scale:g}x" + (" · historical reference only" if host == "B" else "") +
            "\nSeparate host series; one seed; no confidence intervals; x = incomplete work")
        for extension in ("png", "svg", "pdf"):
            fig.savefig(Path(out) / f"host-{host}-slo-{scale:g}-six-panel.{extension}", dpi=150)
        plt.close(fig)


def build(root, out=None, *, plots=True):
    root = Path(root).resolve()
    out = Path(out).resolve() if out else root / "reports" / "current"
    out.mkdir(parents=True, exist_ok=True)
    rows, inputs = collect(root)
    inputs.append(ref(__file__))
    for name in ("contract.py", "generate.py", "reports/crosscheck.py"):
        inputs.append(ref(Path(__file__).parent / name))
    ledger = energy_ledger(root)
    inputs.extend(reference for entry in ledger for reference in references(entry))
    index = audit_sources(rows, inputs)
    crosscheck = load_local_module(Path(__file__).parent / "reports/crosscheck.py", "slo_report_raw_crosscheck")
    checks = crosscheck.build(root, rows)
    completion, completion_sources = completion_status(root, rows, checks)
    inputs.extend(completion_sources)
    indexed = {(item["path"], item["sha256"]): item for item in index}
    for reference in completion_sources:
        path = Path(reference["path"])
        actual = sha(path) if path.is_file() else None
        indexed[(reference["path"], reference["sha256"])] = dict(reference, actual_sha256=actual,
            status="verified" if actual == reference["sha256"] else "missing" if actual is None else "mismatch")
    index = sorted(indexed.values(), key=lambda item: (item["path"], item["sha256"]))
    summary = boundaries(rows)
    for group in summary:
        assessment = completion["nodes"].get(group["measurement_host"])
        if assessment:
            group.update(measurement_grid_complete=assessment["complete"],
                full_context_mirror_status=assessment["full_context_mirror_status"],
                supervisor_phase=assessment["supervisor_phase"], expected_measurements=assessment["expected_measurements"])
            if assessment["complete"]:
                group["reporting_status"] = ("measurements_complete_context_pending" if
                    assessment["full_context_mirror_status"] == "pending" else "measurements_complete")
        else:
            group.update(measurement_grid_complete=None, full_context_mirror_status="reference_only",
                supervisor_phase=None, expected_measurements=None)
    check_summary = {key: checks[key] for key in ("passed_count", "failed_count", "pending_count", "issues")}
    check_summary["ledger"] = ref(root / "reports/crosscheck-ledger.json")
    result = dict(schema="slo-rate-14b-sharegpt-report-v1", created_s=time.time(), observations=rows,
        new_measurement_status="measurements_complete" if completion["complete"] else
            "measurements_incomplete" if any(not row["reference_only"] for row in rows) else "awaiting_observations",
        complete=completion["complete"], completion_acceptance=completion,
        full_context_mirror_status=completion["full_context_mirror_status"], local_cpu_crosscheck=check_summary,
        groups=summary, energy_ledger=ledger, source_inputs=inputs, reporting_code=ref(__file__),
        semantics=dict(missing="JSON null / CSV empty / plotted gaps; never zero-filled", source_audit="Producer-source hash verification plus local core CSV crosschecks; full qualification and service-failure classification are not replayed",
            pairing="Within physical host and SLO only; no cross-host causal energy or boundary ratios",
            confidence_intervals=False, seed=701, arrival_window_s=100,
            energy="All eight GPU boards; primary and outer windows never added",
            completion="Latest clean terminal supervisor, exact contract grid A21/C41, immutable valid audits, verified core hashes and passing local CSV crosscheck; full context mirroring is separate",
            reference="B 1x is immutable historical reference; A 0.5x and C 2x are new runs"))
    save(out / "results.json", result)
    save(out / "summary.json", summary)
    save(out / "completion-acceptance.json", completion)
    save(out / "raw-hash-index.json", index)
    save(out / "energy-ledger.json", ledger)
    write_csv(out / "observations.csv", rows, COLUMNS)
    write_csv(out / "boundaries.csv", summary, tuple(summary[0]))
    write_csv(out / "raw-hash-index.csv", index, ("path", "sha256", "actual_sha256", "status"))
    if plots:
        figures(out, rows)
    (out / "README.md").write_text("# 14B ShareGPT SLO-rate observations\n\n"
        "A = 0.5×，C = 2×，B = 1×历史参考。每台机器独立绘制六项指标，不能从跨机器差异推断 SLO 倍率的因果效果。\n\n"
        "observations.csv 保留逐次观测、低 SLO、未完整工作及证据问题；缺项为空，缺失或无效倍率使曲线断开。summary.json / boundaries.csv 独立描述首次 SLO 下降和完整工作下的首次下降。completion-acceptance.json 同时检查最新调度器终态、合同网格 A21/C41、逐条有效审计与核心哈希及原始 CSV 复算，才确认正式测量完成；资格与完整 checkpoint 镜像状态单独报告。"
        "同 seed 重复不构成独立到达重复，不输出置信区间。\n\n"
        "raw-hash-index.json 核验生产者审计与原始证据的文件哈希；reports/crosscheck-ledger.json 独立复算新观测的核心 CSV 指标，不重演完整服务故障分类或资格验证。正式测量期间只镜像最小指标证据；qualification_mirror_status 仅表示资格文档本体，qualification_inventory_status 表示文档声明的本地文件清单，full_checkpoint_inventory_status 表示完整 checkpoint 文件清单。上下文待同步不等于完整验证已完成。energy-ledger.json 保留 setup/failure 账目，primary 与 outer 能耗不相加。\n")
    contextual_missing = [entry for entry in index if entry["status"] != "verified"]
    if contextual_missing:
        with (out / "README.md").open("a") as stream:
            stream.write("\n来源完整性提醒：以下引用未通过本地哈希核验。metric_source_integrity 单独核验测量的 trace、bench、power、checkpoint、receipt、summary、binding 与已发布审计；资格验证、其完整原始文件及历史背景引用缺失仍完整列出。\n\n")
            for entry in contextual_missing:
                stream.write(f"- {entry['status']}: `{entry['path']}`，预期 SHA256 `{entry['sha256']}`。\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = build(args.root, args.out, plots=not args.no_plots)
    print(json.dumps({"observations": len(result["observations"]), "groups": result["groups"]}, ensure_ascii=False))
