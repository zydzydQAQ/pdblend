"""CPU-only declarations and finite-attempt state machine for the 14B SLO/rate run.

Auditors supply verified request/service terminality separately from output
completion. This module never infers capacity failure from a timeout or 429.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = "slo-rate-14b-sharegpt-20260909-v1"
PROTOCOL = "14b-sharegpt-joint-slo-rate-v1"
SYSTEMS = ("pdblend", "mixed", "distserve", "dynamollm", "ecoserve")
NODE_SCALES = {"A": Decimal("0.5"), "C": Decimal("2")}
MEASUREMENT_HOSTS = {"A": "Anew20260909", "C": "C"}
STEP = Decimal("0.25")
SEED = 701
SAMPLING_SEED = 20260907
WINDOW_S = 100.0
TIMEOUT_S = 120.0
TARGET = 0.90
MAX_ENGINEERING_ATTEMPTS = 2


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    need(reference_shape(reference), "immutable reference required")
    need(sha(reference["path"]) == reference["sha256"], "frozen input changed: " + reference["path"])
    return read(reference["path"])


def reference_shape(reference):
    return (isinstance(reference, dict) and isinstance(reference.get("path"), str)
            and isinstance(reference.get("sha256"), str) and len(reference["sha256"]) == 64
            and all(ch in "0123456789abcdef" for ch in reference["sha256"]))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def number(value):
    need(not isinstance(value, bool), "boolean is not a rate")
    try:
        value = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid rate") from exc
    need(value.is_finite() and value > 0 and value % STEP == 0,
         "rate must be positive and on the exact 0.25 rps grid")
    return format(value.normalize(), "f")


def node_name(node):
    node = "A" if node == "Anew20260909" else node
    need(node in NODE_SCALES, "new measurements are only A/scale0.5 and C/scale2")
    return node


def grid(limit):
    limit = Decimal(number(limit))
    return [number(STEP * i) for i in range(1, int(limit / STEP) + 1)]


def cell_id(node, rate, system, repeat=1):
    node = node_name(node)
    need(system in SYSTEMS, "unknown system")
    need(type(repeat) is int and repeat in ((1, 2) if system == "pdblend" else (1,)),
         "only PDB boundary has a second normal repeat")
    scale = format(NODE_SCALES[node].normalize(), "f")
    return f"{CAMPAIGN}-{node}-14b-sharegpt-r{number(rate)}-s701-w100-{system}-slo{scale}-repeat{repeat}"


def make_row(node, rate, system, repeat=1, *, workload=None):
    """Return a row, materializing its frozen workload once if not supplied."""
    node = node_name(node)
    cid = cell_id(node, rate, system, repeat)
    if workload is None:
        generator = load_module(HERE / "generate.py", "_slo_rate_materializer")
        workload = generator.materialize_rate(rate)
    need((workload.get("model"), workload.get("dataset"), number(workload.get("rate_rps")))
         == ("14b", "sharegpt", number(rate)), "wrong workload")
    need((workload.get("seed"), workload.get("sampling_seed"), workload.get("arrival_window_s"))
         == (SEED, SAMPLING_SEED, WINDOW_S), "workload sampling/window changed")
    scale = NODE_SCALES[node]
    ttft, tpot = float(Decimal("5") * scale), float(Decimal("0.15") * scale)
    row = copy.deepcopy(workload)
    row.update(campaign_id=CAMPAIGN, protocol_id=PROTOCOL, cell_id=cid,
        node=node, measurement_host=MEASUREMENT_HOSTS[node], model="14b", dataset="sharegpt",
        system=system, comparison_system=system, repeat=repeat, rate_rps=float(number(rate)),
        rate_rps_decimal=number(rate), phase="slo_rate", part="slo_rate",
        slo_scale=float(scale), allowed_slo_scales=[float(scale)],
        slo_ttft_s=ttft, slo_tpot_s=tpot, slo_attainment_target=TARGET,
        slo={"ttft_s": ttft, "tpot_s": tpot, "attainment_target": TARGET},
        strategy=None, controller_config=None, policy_binding_required=True,
        execution_binding_required=True, execution_status="not_run", formal_eligible=False,
        request_hard_timeout_s=TIMEOUT_S, drain_after_arrival_window_s=TIMEOUT_S,
        arrival_window_s=WINDOW_S, trace_duration_s=WINDOW_S, seed=SEED,
        arrival_seed=SEED, sampling_seed=SAMPLING_SEED,
        measurement_purpose="normal", conditional=system == "pdblend" and repeat == 2,
        reuse_main_cell_id=None, original_actual_source_relabelled=False,
        measurement_window="100s arrivals plus actual drain/control tail; all eight GPUs")
    return row


PAIR_FIELDS = ("campaign_id", "cell_id", "node", "measurement_host", "model", "dataset",
               "system", "repeat", "seed", "sampling_seed", "arrival_window_s",
               "trace_sha256", "content_pairing_sha256", "slo_scale", "slo_ttft_s", "slo_tpot_s")


def identity(observation):
    row = dict(observation.get("row", {}))
    row.update({k: v for k, v in observation.items() if k != "row"})
    return tuple(row.get(k) for k in PAIR_FIELDS) + (number(row.get("rate_rps")),)


def row_of(observation):
    row = dict(observation.get("row", {}))
    row.update({k: v for k, v in observation.items() if k != "row"})
    return row


def valid_service_observation(observation):
    """Trust the independently executed auditor; never fabricate its findings.

    The caller verifies audit_reference bytes before calling this pure state
    machine. service_terminal_valid covers all offered request identities and
    terminal outcomes; capacity/timeouts require the auditor's positive finding.
    """
    q = observation.get("slo_attainment")
    return bool(observation.get("measurement_valid") is True
        and observation.get("service_terminal_valid") is True
        and observation.get("strict_slo_recomputed") is True
        and observation.get("independently_recomputed") is True
        and observation.get("unknown_error_count") == 0
        and reference_shape(observation.get("audit_reference"))
        and type(q) in (int, float) and math.isfinite(q) and 0 <= q <= 1
        and (observation.get("work_complete") is True
             or (observation.get("work_complete") is False
                 and observation.get("capacity_failures_independently_audited") is True)))


def select_attempts(observations):
    """One normal cell may have one diagnosed engineering replacement only."""
    buckets = {}
    for observation in observations:
        row = row_of(observation)
        cid = row.get("cell_id")
        need(isinstance(cid, str), "cell identity missing")
        attempt = observation.get("engineering_attempt", 1)
        need(type(attempt) is int and 1 <= attempt <= MAX_ENGINEERING_ATTEMPTS,
             "at most one engineering replacement per normal cell")
        bucket = buckets.setdefault(cid, {})
        need(attempt not in bucket, "duplicate cell attempt")
        bucket[attempt] = observation
    selected = {}
    for cid, attempts in buckets.items():
        need(set(attempts) == set(range(1, max(attempts) + 1)), "missing earlier engineering attempt")
        if 2 in attempts:
            need(not valid_service_observation(attempts[1]), "a valid outcome cannot be replaced")
            need(reference_shape(attempts[2].get("repair_reference")), "diagnosed repair reference required")
        selected[cid] = attempts[max(attempts)]
    return selected


def evaluate_rate(node, rate, observations, *, workload=None):
    """Evaluate one PDB rate; baseline observations do not influence its cap."""
    node, rate = node_name(node), number(rate)
    rows = {repeat: make_row(node, rate, "pdblend", repeat, workload=workload) for repeat in (1, 2)}
    relevant = []
    for observation in observations:
        actual = row_of(observation)
        need(actual.get("system") in SYSTEMS, "unknown observed system")
        expected = make_row(node, rate, actual["system"], actual.get("repeat"), workload=workload)
        need(identity(observation) == identity(expected), "observation identity differs from declaration")
        if actual["system"] == "pdblend":
            relevant.append(observation)
    selected = select_attempts(relevant)
    normal = selected.get(rows[1]["cell_id"])
    confirmation = selected.get(rows[2]["cell_id"])
    need(confirmation is None or (normal is not None and valid_service_observation(normal)
         and normal["slo_attainment"] < TARGET), "confirmation requires a valid first loss")
    losses = [o for o in (normal, confirmation) if o is not None
              and valid_service_observation(o) and o["slo_attainment"] < TARGET]
    cap = bool(losses)
    result = dict(node=node, rate_rps=float(rate), rate_rps_decimal=rate,
        cap_observed=cap, cap_trigger_cell_id=row_of(losses[0])["cell_id"] if cap else None,
        cap_rate_rps=float(rate) if cap else None, next_tasks=[], increase_rate_allowed=False,
        threshold_straddles=bool(cap and confirmation is not None
            and valid_service_observation(confirmation) and confirmation["slo_attainment"] >= TARGET))
    wanted = rows[2] if cap else rows[1]
    active = confirmation if cap else normal
    if active is None:
        result.update(status="confirm_boundary" if cap else "measure_pdblend", next_tasks=[wanted])
    elif not valid_service_observation(active):
        attempts = active.get("engineering_attempt", 1)
        result.update(status="blocked_engineering" if attempts >= MAX_ENGINEERING_ATTEMPTS
                      else "engineering_diagnosis", engineering_fault_cell_id=wanted["cell_id"],
                      engineering_attempts_used=attempts,
                      engineering_replacement_remaining=MAX_ENGINEERING_ATTEMPTS - attempts)
    elif cap:
        result.update(status="cap_confirmed", confirmation_complete=True)
    else:
        result.update(status="advance", increase_rate_allowed=True)
    return result


def evaluate_group(node, observations, *, materializer=None):
    """Choose the next exact rate, or all pending baselines through a fixed cap.

    No predeclared upper rate exists. Invalid observations never cause another
    automatic attempt; diagnosis must explicitly return an audited replacement.
    """
    node = node_name(node)
    materializer = materializer or load_module(HERE / "generate.py", "_slo_rate_group_materializer").materialize_rate
    grouped = {}
    for observation in observations:
        row = row_of(observation)
        need(node_name(row.get("node")) == node, "cross-node observation")
        grouped.setdefault(number(row.get("rate_rps")), []).append(observation)
    all_selected = select_attempts(observations)
    rate = STEP
    while True:
        rate_s = number(rate)
        workload = materializer(rate_s)
        decision = evaluate_rate(node, rate_s, grouped.get(rate_s, []), workload=workload)
        if decision["status"] == "advance":
            rate += STEP
            continue
        need(not any(Decimal(r) > rate for r in grouped), "higher rate observed before lower rate authorized")
        decision.update(complete=False, baseline_tasks=[], baseline_engineering_faults=[],
                        eligible_rates=grid(rate_s) if decision["cap_observed"] else [])
        if decision["cap_observed"]:
            for candidate in grid(rate_s):
                w = materializer(candidate)
                for system in SYSTEMS[1:]:
                    expected = make_row(node, candidate, system, workload=w)
                    actual = all_selected.get(expected["cell_id"])
                    if actual is None:
                        decision["baseline_tasks"].append(expected)
                    else:
                        need(identity(actual) == identity(expected), "baseline identity differs")
                        if not valid_service_observation(actual):
                            decision["baseline_engineering_faults"].append(dict(
                                cell_id=expected["cell_id"],
                                status="blocked_engineering" if actual.get("engineering_attempt", 1) >= 2
                                    else "engineering_diagnosis"))
            decision["complete"] = (decision["status"] == "cap_confirmed"
                and not decision["baseline_tasks"] and not decision["baseline_engineering_faults"])
        return decision


def protocol():
    return dict(schema=PROTOCOL, campaign_id=CAMPAIGN,
        model="14b", dataset="sharegpt", systems=list(SYSTEMS),
        node_scales={k: float(v) for k, v in NODE_SCALES.items()}, measurement_hosts=MEASUREMENT_HOSTS,
        baseline_reference="B existing scale1 is preserved separately; cross-host results are not causal SLO-only pairs",
        arrival_seed=SEED, sampling_seed=SAMPLING_SEED, arrival_window_s=WINDOW_S,
        request_hard_timeout_s=TIMEOUT_S, drain_after_arrival_window_s=TIMEOUT_S,
        rate_start_rps=float(STEP), rate_step_rps=float(STEP), rate_upper_bound=None,
        base_slo_ttft_s=5.0, base_slo_tpot_s=0.15, slo_attainment_target=TARGET,
        slo_definition="complete request AND TTFT < scaled threshold AND TPOT < scaled threshold; all offered denominator",
        stop_rule="first independently audited service-terminal-valid PDB observation below 90%; one same-trace confirmation cannot undo cap",
        normal_repeats=1, pdb_boundary_confirmation_repeats=1,
        baselines="each system once at every grid rate from 0.25 through and including PDB first-loss rate",
        incomplete_work_policy="audited capacity rejection/deadline is a scored service failure, not an engineering invalidation",
        unknown_failure_policy="pause; one explicitly diagnosed engineering replacement maximum; then blocked, never silently reschedule",
        max_engineering_attempts_per_normal_cell=MAX_ENGINEERING_ATTEMPTS,
        valid_outcomes_replaceable=False, original_attempts_preserved=True,
        actual_source_freeze="node binding, policy config, profile, image, topology, frequency domain; only joint SLO thresholds and rate vary",
        auditor_required_fields=["measurement_valid", "service_terminal_valid", "strict_slo_recomputed",
            "independently_recomputed", "unknown_error_count", "audit_reference", "slo_attainment", "work_complete"],
        incomplete_work_additional_auditor_field="capacity_failures_independently_audited=true",
        audit_reference_verification="caller verifies immutable audit file before state-machine evaluation",
        independent_seeds=False, uncertainty="same seed/trace repetitions are not independent-seed confidence intervals",
        meter_scope="all eight GPUs, full arrival plus actual drain/control tail; setup recorded separately")
