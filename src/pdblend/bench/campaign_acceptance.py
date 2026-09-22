"""Fail-closed acceptance for paired multi-system campaigns.

The helper intentionally separates two notions of identity.  A profile is
identified by its own system/TP/PP/profile hash.  A *pair* is identified by
the workload, model, runtime and measurement environment that must be shared
between systems.  TP/PP and profile hashes are therefore never compared
across systems.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..profile.identity import canonical_json, profile_identity, sha256_value

# Active matrix comparisons intentionally use one reproducible workload seed.
# Historical three-seed artifacts remain readable when callers pass an
# explicit seed list, but the default campaign gate is seed 701 only.
SINGLE_SEED = 701
SEEDS = (SINGLE_SEED,)
SEED_POLICY = "single_seed_701"
DEFAULT_SYSTEMS = ("mixed", "distserve", "dynamollm", "ecoserve", "pdblend")
COMMON_IDENTITY_FIELDS = (
    "model", "dataset", "rate", "duration", "trace_sha256", "corpus_sha256",
    "image", "source_sha256", "hardware", "clock_protocol", "energy_protocol",
    "engine_revision", "vllm", "torch", "cuda",
)
PROFILE_FIELDS = ("system", "model", "tp", "pp", "role", "profile_sha256")
METRIC_FIELDS = ("joint_slo_rate", "success_rate", "j_per_token")


class IdentityMismatch(ValueError):
    """Raised by the strict pairing API when records cannot be compared."""


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _identity(record: Mapping) -> Mapping:
    value = record.get("identity")
    value = value if isinstance(value, Mapping) else record
    if "model" not in value and value.get("model_id") is not None:
        return {**dict(value), "model": value["model_id"]}
    return value


def _raw_identity(record: Mapping) -> Mapping:
    value = record.get("identity")
    return value if isinstance(value, Mapping) else record


def _summary(record: Mapping) -> Mapping:
    value = record.get("summary")
    return value if isinstance(value, Mapping) else record


def _metric(record: Mapping, name: str):
    summary = _summary(record)
    if name in summary:
        return summary[name]
    slo = summary.get("slo")
    if isinstance(slo, Mapping) and name in slo:
        return slo[name]
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics.get(name)
    return None


def _profile_fields(record: Mapping, ident: Mapping) -> dict:
    nested = record.get("profile_identity")
    if not isinstance(nested, Mapping):
        nested = ident.get("profile_identity")
    if isinstance(nested, Mapping):
        try:
            extracted = profile_identity({"profile_key": nested, "environment": nested.get("environment", {})})
        except (TypeError, ValueError):
            extracted = dict(nested)
        result = {**dict(ident), **{k: extracted.get(k) for k in PROFILE_FIELDS if k in extracted}}
        result["profile_identity"] = extracted
        return result
    return dict(ident)


def _meter_scope(record: Mapping, ident: Mapping) -> str | None:
    for source in (record, ident, _summary(record)):
        for key in ("meter_scope", "energy_scope", "power_scope"):
            if source.get(key) is not None:
                return str(source[key]).lower()
    protocol = str(ident.get("energy_protocol", "")).lower()
    if "nvml" in protocol:
        return "gpu_board"
    return None


def validate_evidence(record: Mapping | None, *, require_profile: bool = True) -> dict:
    """Validate one result without interpreting its SLO.

    A completed run whose SLO is below target returns ``measured_failure``.  A
    missing, stale or unverifiable run returns ``inconclusive`` and can never
    be treated as a failed system measurement.
    """
    if not isinstance(record, Mapping):
        return {"status": "inconclusive", "reasons": ["missing_evidence"]}
    reasons: list[str] = []
    ident = _identity(record)
    if not isinstance(ident, Mapping):
        return {"status": "inconclusive", "reasons": ["missing_identity"]}
    if record.get("status") != "complete":
        reasons.append("execution_incomplete")
    if record.get("returncode") != 0:
        reasons.append("execution_failed")
    if record.get("inputs_unchanged") is not True:
        reasons.append("inputs_not_attested")
    if not record.get("identity_sha256"):
        reasons.append("missing_identity_digest")
    elif record["identity_sha256"] != sha256_value(dict(_raw_identity(record))):
        reasons.append("identity_digest_mismatch")
    required = [field for field in COMMON_IDENTITY_FIELDS if field != "trace_sha256"] + list(PROFILE_FIELDS) + ["seed"]
    for field in required:
        if ident.get(field) in (None, "", "unknown"):
            reasons.append("missing_identity:" + field)
    # A workload digest may be called workload_sha256 by newer runners; the
    # trace digest is the equivalent binding in the matrix runner.
    if ident.get("trace_sha256") in (None, "", "unknown") and ident.get("workload_sha256") in (None, "", "unknown"):
        reasons.append("missing_identity:workload")
    if ident.get("seed") != SINGLE_SEED:
        reasons.append("unsupported_seed:" + str(ident.get("seed")))
    slo_target = ident.get("slo") or ident.get("slo_target") or _summary(record).get("slo_target")
    if slo_target in (None, "", "unknown"):
        if not any(ident.get(k) not in (None, "", "unknown") for k in ("ttft_slo_s", "tpot_slo_s")):
            reasons.append("missing_identity:slo_target")
    if not _finite(_metric(record, "joint_slo_rate")):
        reasons.append("missing_metric:joint_slo_rate")
    if not _finite(_metric(record, "j_per_token")):
        reasons.append("missing_metric:j_per_token")
    if require_profile:
        nested = record.get("profile_identity") or ident.get("profile_identity")
        if nested is not None and not isinstance(nested, Mapping):
            reasons.append("invalid_profile_identity")
        elif isinstance(nested, Mapping) and nested.get("system") not in (None, ident.get("system")):
            reasons.append("cross_system_profile")
        # A profile hash without its own system/layout binding is ambiguous.
        if nested is None and any(ident.get(k) in (None, "", "unknown") for k in ("system", "tp", "pp", "profile_sha256")):
            reasons.append("unbound_profile")
    scope = _meter_scope(record, ident)
    if scope in ("whole_host", "host") and "nvml" in str(ident.get("energy_protocol", "")).lower():
        reasons.append("nvml_is_gpu_board_not_whole_host")
    if reasons:
        return {"status": "inconclusive", "reasons": sorted(set(reasons)), "identity": dict(ident)}
    slo = float(_metric(record, "joint_slo_rate"))
    status = "measured_failure" if slo < 0.9 else "measured_pass"
    return {"status": status, "reasons": (["slo_below_0.9"] if status == "measured_failure" else []),
            "identity": dict(ident), "metrics": {name: _metric(record, name) for name in METRIC_FIELDS}}


def common_identity(record: Mapping) -> dict:
    """Return fields that must match when pairing different systems."""
    ident = _identity(record)
    return {field: _common_value(ident, field) for field in COMMON_IDENTITY_FIELDS} | {"slo_target": _slo_target(ident, record)}


def _common_value(ident: Mapping, field: str):
    if field == "trace_sha256" and ident.get(field) in (None, "", "unknown"):
        return ident.get("workload_sha256")
    return ident.get(field)


def _slo_target(ident: Mapping, record: Mapping | None = None):
    value = ident.get("slo") or ident.get("slo_target")
    if value in (None, "", "unknown") and isinstance(record, Mapping):
        value = _summary(record).get("slo_target")
    if value in (None, "", "unknown"):
        value = {key: ident.get(key) for key in ("ttft_slo_s", "tpot_slo_s") if ident.get(key) not in (None, "", "unknown")}
    return value


def pair_key(record: Mapping) -> tuple:
    ident = _identity(record)
    return tuple(_common_value(ident, field) for field in COMMON_IDENTITY_FIELDS) + (_slo_target(ident, record), ident.get("seed"))


def pair_records(candidate: Mapping | None, baseline: Mapping | None) -> dict:
    """Pair two measured systems without requiring equal TP or profile hash."""
    left = validate_evidence(candidate)
    right = validate_evidence(baseline)
    if left["status"] == "inconclusive" or right["status"] == "inconclusive":
        return {"status": "inconclusive", "reasons": ["missing_or_invalid_evidence"],
                "candidate": left, "baseline": right, "single_seed": True,
                "seed_policy": SEED_POLICY}
    a, b = left["identity"], right["identity"]
    mismatches = [field for field in COMMON_IDENTITY_FIELDS + ("seed",)
                  if _common_value(a, field) != _common_value(b, field)]
    if _slo_target(a, candidate) != _slo_target(b, baseline):
        mismatches.append("slo_target")
    if mismatches:
        return {"status": "inconclusive", "reasons": ["identity_mismatch:" + ",".join(mismatches)],
                "candidate": left, "baseline": right, "single_seed": True,
                "seed_policy": SEED_POLICY}
    candidate_slo = left["metrics"]["joint_slo_rate"] >= .9
    baseline_slo = right["metrics"]["joint_slo_rate"] >= .9
    energy_win = left["metrics"]["j_per_token"] < right["metrics"]["j_per_token"]
    reasons = []
    if not candidate_slo:
        reasons.append("candidate_slo")
    if not energy_win:
        reasons.append("energy_loss")
    return {"status": "failed" if reasons else "passed", "reasons": reasons,
            "candidate_slo": candidate_slo, "baseline_slo": baseline_slo,
            "energy_win": energy_win, "candidate": left, "baseline": right,
            "single_seed": True, "seed_policy": SEED_POLICY}


def require_pair(candidate: Mapping, baseline: Mapping) -> dict:
    result = pair_records(candidate, baseline)
    if result["status"] == "inconclusive":
        raise IdentityMismatch(";".join(result["reasons"]))
    return result


def _flatten_records(records: Mapping | Iterable[Mapping]) -> list[Mapping]:
    rows: list[Mapping] = []
    if isinstance(records, Mapping):
        for system, values in records.items():
            values = values if isinstance(values, (list, tuple)) else [values]
            for value in values:
                if isinstance(value, Mapping):
                    if "identity" not in value:
                        value = {**value, "identity": {**dict(value.get("identity", {})), "system": system}}
                    rows.append(value)
    else:
        rows = [r for r in records if isinstance(r, Mapping)]
    return rows


def _campaign_group_key(record: Mapping) -> tuple:
    ident = _identity(record)
    # TP/PP and profile hash are deliberately excluded: systems may use
    # different legal layouts and independently calibrated profiles.
    return tuple(canonical_json(_common_value(ident, field)) for field in COMMON_IDENTITY_FIELDS) + (canonical_json(_slo_target(ident, record)),)


def accept_campaign(records: Mapping | Iterable[Mapping], *, systems: Sequence[str] = DEFAULT_SYSTEMS,
                    seeds: Sequence[int] | None = None) -> dict:
    """Audit a campaign and require every system to have the active seed.

    Missing evidence is ``inconclusive``.  Complete evidence with a bad SLO is
    a measured failure, allowing a real failed baseline to remain visible.

    ``seeds`` is retained as an explicit escape hatch for auditing historical
    artifacts.  The active campaign gate rejects such schedules so an old
    three-seed requirement cannot accidentally return a formal result.
    """
    systems = tuple(systems)
    seeds = SEEDS if seeds is None else tuple(int(s) for s in seeds)
    single_seed = seeds == SEEDS
    seed_policy = SEED_POLICY if single_seed else "custom"
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in _flatten_records(records):
        groups[_campaign_group_key(row)].append(row)
    rows, reasons = [], []
    if not groups:
        return {"status": "inconclusive", "formal_eligible": False,
                "reasons": ["missing_evidence"], "rows": [],
                "seeds": list(seeds), "single_seed": single_seed,
                "seed_policy": seed_policy}
    for group, group_rows in groups.items():
        by_system_seed: dict[tuple[str, int], list[Mapping]] = defaultdict(list)
        for row in group_rows:
            ident = _identity(row)
            if ident.get("system") is not None and ident.get("seed") is not None:
                by_system_seed[(str(ident["system"]), int(ident["seed"]))].append(row)
        for seed in seeds:
            for system in systems:
                matches = by_system_seed.get((system, seed), [])
                if len(matches) != 1:
                    reason = f"{system}:seed{seed}:" + ("missing_evidence" if not matches else "duplicate_evidence")
                    rows.append({"group": group, "system": system, "seed": seed,
                                 "status": "inconclusive", "reasons": [reason]})
                    reasons.append(reason)
                    continue
                checked = validate_evidence(matches[0])
                checked.update(group=group, system=system, seed=seed)
                rows.append(checked)
                if checked["status"] == "inconclusive":
                    reasons.extend(f"{system}:seed{seed}:{r}" for r in checked["reasons"])
    if not single_seed:
        reasons.append(f"unsupported_seed_policy:{SEED_POLICY}")
    if reasons:
        return {"status": "inconclusive", "formal_eligible": False, "reasons": sorted(set(reasons)), "rows": rows,
                "seeds": list(seeds), "single_seed": single_seed, "seed_policy": seed_policy}
    failed = [row for row in rows if row["status"] == "measured_failure"]
    return {"status": "failed" if failed else "passed", "formal_eligible": not failed,
            "reasons": [f"{r['system']}:seed{r['seed']}:slo_below_0.9" for r in failed], "rows": rows,
            "complete": True, "seeds": list(seeds), "systems": list(systems),
            "single_seed": single_seed, "seed_policy": seed_policy}


def audit_campaign(records: Mapping | Iterable[Mapping], **kwargs) -> dict:
    """Compatibility alias for callers that use ``audit_*`` naming."""
    return accept_campaign(records, **kwargs)


def campaign_digest(records: Iterable[Mapping]) -> str:
    """Stable digest of identities and measured outcomes, excluding ordering."""
    rows = []
    for record in records:
        ident = _identity(record)
        rows.append({"identity": dict(ident), "metrics": {name: _metric(record, name) for name in METRIC_FIELDS}})
    return hashlib.sha256(canonical_json(sorted(rows, key=canonical_json)).encode()).hexdigest()
