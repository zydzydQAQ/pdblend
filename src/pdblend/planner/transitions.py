"""Exact, evidence-bound transition cost catalog; gross energy is never a switch penalty."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pdblend.planner.pool import Plan

KIND = "pdblend_transition_catalog_v1"
# A group is satisfied by one equivalent native phase. All groups are required.
REQUIRED_PHASES = {
    "role": (("validate", "verify"), ("publish", "route_publish")),
    "frequency": (("clock_set", "clock_reset"), ("validate", "verify"), ("publish", "route_publish")),
    "park": (("proxy_drain",), ("native_drain",), ("park", "stop"), ("publish", "route_publish")),
    "wake": (("unpark", "start"), ("ready", "validate", "verify"), ("publish", "route_publish")),
    "cold_start": (("start", "load"), ("ready",), ("warm", "warmup"),
                   ("validate", "verify"), ("publish", "route_publish")),
    "restart": (("proxy_drain",), ("native_drain",), ("stop",), ("start", "load"),
                ("ready",), ("warm", "warmup"), ("validate", "verify"), ("publish", "route_publish")),
    "tp_rebuild": (("proxy_drain",), ("native_drain",), ("release",), ("load",),
                   ("warm", "warmup"), ("validate", "verify"), ("publish", "route_publish")),
    "recovery": (("recovery", "rollback"), ("validate", "verify"), ("publish", "route_publish")),
}


def identity(model):
    return dict(system=model.system, model_id=model.profile_key.get("model_id", model.model),
                tp=model.tp, pp=model.pp, profile_key=model.profile_key)


def signature(plan: Plan):
    if any(not isinstance(n, int) or n < 0 for n in plan.counts.values()):
        raise ValueError("transition counts must be nonnegative integers")
    return dict(counts={role: n for role, n in sorted(plan.counts.items()) if n},
                clocks={role: getattr(plan, f"f_{role}") for role in ("P", "D", "M")
                        if plan.counts.get(role, 0)}, tau=plan.tau, tp=plan.tp, pp=plan.pp,
                pool_id=plan.pool_id, generation=plan.generation)


def actions_for(source: Plan, target: Plan):
    actions = set()
    if (source.tp, source.pp) != (target.tp, target.pp):
        actions.add("tp_rebuild")
    if any(source.counts.get(r, 0) != target.counts.get(r, 0) for r in ("P", "D", "M")) or source.tau != target.tau:
        actions.add("role")
    if any(source.counts.get(r, 0) and target.counts.get(r, 0)
           and getattr(source, f"f_{r}") != getattr(target, f"f_{r}") for r in ("P", "D", "M")):
        actions.add("frequency")
    if target.active() < source.active() or any(target.counts.get(r, 0) > source.counts.get(r, 0) for r in ("L1", "off")):
        actions.add("park")
    if target.active() > source.active() or any(target.counts.get(r, 0) < source.counts.get(r, 0) for r in ("L1", "off")):
        actions.add("wake")
    if target.counts.get("off", 0) < source.counts.get("off", 0):
        actions.add("cold_start")
    return sorted(actions)


def _read(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return json.loads(data), dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


def _finite(value, *, nonnegative=True):
    return isinstance(value, (int, float)) and math.isfinite(value) and (not nonnegative or value >= 0)


def _geometry(raw):
    phases = raw.get("phases", [])
    if not phases:
        return []
    start = min(row["started_s"] for row in phases)
    return sorted((tuple(sorted(row["gpus"])), round(row["started_s"] - start, 6),
                   round(row["finished_s"] - start, 6)) for row in phases)


def build_catalog_entry(receipt_path, source: Plan, target: Plan, model, *, actions=None,
                        counterfactual_path=None):
    """Produce an auditable entry from measure_transitions output and optional matched control.

    Incomplete receipts are retained with ``qualified=False`` for timing review.
    A paired control must itself be a common-sampler receipt, with matching
    pairing metadata and interval geometry. Gross energy alone stays unqualified.
    """
    raw, evidence = _read(receipt_path)
    declared = sorted(set(actions_for(source, target) if actions is None else actions))
    if set(declared) - set(REQUIRED_PHASES) or set(actions_for(source, target)) - set(declared):
        raise ValueError("transition actions omit a changed state or contain an unknown action")
    source_key, target_key, model_key = signature(source), signature(target), identity(model)
    expected_binding = dict(identity=model_key, source=source_key, target=target_key)
    reasons = []
    if raw.get("schema") != 1:
        reasons.append("unsupported transition receipt schema")
    if raw.get("binding") != expected_binding or raw.get("target_verified") is not True:
        reasons.append("missing exact profile/source/target native verification binding")
    phases = raw.get("phases", [])
    valid_times = bool(phases) and all(_finite(r.get("started_s")) and _finite(r.get("finished_s"))
                                     and r["finished_s"] >= r["started_s"] for r in phases)
    operations = {r.get("operation") for r in phases if r.get("status") == "passed"}
    if not valid_times or any(r.get("status") != "passed" for r in phases):
        reasons.append("incomplete or failed transition phases")
    missing = ["/".join(group) for action in declared for group in REQUIRED_PHASES[action]
               if not set(group) & operations]
    if missing:
        reasons.append("missing required phases: " + ", ".join(sorted(set(missing))))
    critical = (max(r["finished_s"] for r in phases) - min(r["started_s"] for r in phases)) if valid_times else None
    extra = uncertainty = planning = None
    refs = [evidence]
    if counterfactual_path is None:
        reasons.append("paired counterfactual energy unavailable")
    else:
        reference, ref_evidence = _read(counterfactual_path)
        refs.append(ref_evidence)
        pairing, ref_pairing = raw.get("pairing", {}), reference.get("pairing", {})
        paired = (pairing.get("pair_id") and pairing.get("workload_sha256")
                  and all(pairing.get(k) == ref_pairing.get(k) for k in ("pair_id", "workload_sha256"))
                  and reference.get("binding") == expected_binding
                  and pairing.get("role") == "transition" and ref_pairing.get("role") == "counterfactual"
                  and raw.get("energy_complete") is True and reference.get("energy_complete") is True
                  and raw.get("power_source") and raw.get("power_source") == reference.get("power_source")
                  and not raw.get("sampler_error") and not reference.get("sampler_error")
                  and _finite(raw.get("measured_union_energy_j"))
                  and _finite(reference.get("measured_union_energy_j"))
                  and _finite(pairing.get("uncertainty_j")) and _finite(ref_pairing.get("uncertainty_j")))
        try:
            paired = bool(paired and valid_times and _geometry(raw) == _geometry(reference))
        except (KeyError, TypeError, ValueError):
            paired = False
        if paired:
            extra = raw["measured_union_energy_j"] - reference["measured_union_energy_j"]
            uncertainty = pairing["uncertainty_j"] + ref_pairing["uncertainty_j"]
            planning = max(0.0, extra + uncertainty)
        else:
            reasons.append("counterfactual lacks matched workload/intervals/uncertainty evidence")
    return dict(identity=model_key, source=source_key, target=target_key, actions=declared,
                evidence=refs, critical_path_s=critical,
                measured_union_energy_j=raw.get("measured_union_energy_j"),
                incremental_energy_j=extra, uncertainty_j=uncertainty, planning_energy_j=planning,
                qualified=not reasons, reasons=reasons, formal_eligible=False)


class TransitionCatalog:
    """Callable adapter accepted by PlannerConfig.transition_estimator."""
    def __init__(self, entries, model, *, qualified_only=False):
        self.model_identity = identity(model)
        self.qualified_only = qualified_only
        self.entries = {}
        for entry in entries:
            if entry["identity"] != self.model_identity:
                raise ValueError("transition catalog profile/model/TP/PP identity mismatch")
            key = self._key(entry["source"], entry["target"])
            if key in self.entries:
                raise ValueError("duplicate transition source/target in catalog")
            self.entries[key] = entry

    @staticmethod
    def _key(source, target):
        return json.dumps([source, target], sort_keys=True, separators=(",", ":"))

    @classmethod
    def load(cls, path, *, model, qualified_only=False):
        path = Path(path).resolve()
        document = json.loads(path.read_text())
        if document.get("kind") != KIND or document.get("identity") != identity(model):
            raise ValueError("transition catalog profile/model/TP/PP identity mismatch")
        entries = []
        for entry in document.get("entries", []):
            if entry.get("identity") != identity(model):
                raise ValueError("transition catalog entry identity mismatch")
            refs = entry["evidence"]
            if len(refs) not in (1, 2):
                raise ValueError("transition catalog requires measured and optional control receipt")
            resolved = []
            for ref in refs:
                evidence_path = (path.parent / ref["path"]).resolve()
                _, observed = _read(evidence_path)
                if observed["sha256"] != ref["sha256"]:
                    raise ValueError("transition evidence checksum mismatch")
                resolved.append(evidence_path)
            def plan(shape):
                return Plan(shape["counts"], *(shape["clocks"].get(r, max(model.freqs)) for r in ("P", "D", "M")),
                            shape["tau"], 0, 0, 0, tp=shape["tp"], pp=shape["pp"],
                            pool_id=shape["pool_id"], generation=shape["generation"])
            audited = build_catalog_entry(resolved[0], plan(entry["source"]), plan(entry["target"]), model,
                                           actions=entry["actions"],
                                           counterfactual_path=resolved[1] if len(resolved) == 2 else None)
            expected = dict(entry, evidence=audited["evidence"])
            if audited != expected:
                raise ValueError("transition catalog cannot reproduce its qualification audit")
            entries.append(audited)
        return cls(entries, model, qualified_only=qualified_only)

    def __call__(self, current: Plan, new: Plan):
        source, target = signature(current), signature(new)
        entry = self.entries.get(self._key(source, target))
        for plan in (current, new):
            if plan.profile_key:
                try:
                    if json.loads(plan.profile_key) != self.model_identity["profile_key"]:
                        entry = None
                except (TypeError, ValueError):
                    entry = None
        if entry is None:
            return dict(qualified=False, incremental_energy_j=None, qualified_only=self.qualified_only,
                        provenance="uncovered_transition", source=source, target=target)
        return dict(qualified=entry["qualified"], incremental_energy_j=entry["planning_energy_j"]
                    if entry["qualified"] else None, qualified_only=self.qualified_only,
                    provenance="paired_transition_measurement", source=source, target=target,
                    measured_incremental_energy_j=entry["incremental_energy_j"],
                    uncertainty_j=entry["uncertainty_j"], critical_path_s=entry["critical_path_s"],
                    evidence=entry["evidence"], reasons=entry["reasons"])


def main():
    import argparse
    from pdblend.profile.query.versions import load_profile
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "source-plan", "target-plan", "profile", "model-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--counterfactual")
    args = parser.parse_args()
    model = load_profile(args.profile, system="pdblend", model_id=args.model_id, tp=args.tp, pp=args.pp).model
    source, target = (Plan(**json.loads(Path(path).read_text())) for path in (args.source_plan, args.target_plan))
    entry = build_catalog_entry(args.receipt, source, target, model, counterfactual_path=args.counterfactual)
    Path(args.output).write_text(json.dumps(dict(kind=KIND, identity=identity(model), entries=[entry]), indent=2) + "\n")


if __name__ == "__main__":
    main()
