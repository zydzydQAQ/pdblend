"""Aggregate benchmark points into per-dataset and cross-dataset comparisons against a reference policy."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional


def load_points(root: Path) -> list[dict]:
    points = []
    for path in sorted(Path(root).rglob("summary.json")):
        try:
            s = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "slo" not in s:
            continue
        s["_dir"] = str(path.parent)
        points.append(s)
    return points


def point_key(s: dict) -> tuple:
    meta = s.get("trace_meta", {})
    return (s.get("model"), meta.get("dataset"), meta.get("source", "poisson"), round(s["trace"].get("mean_rps", 0), 2),
            meta.get("scale"), meta.get("seed"))


def flatten(s: dict) -> dict:
    att = s["slo"]
    meta = s.get("trace_meta", {})
    return dict(model=s.get("model"), dataset=meta.get("dataset"), scale=meta.get("scale"), seed=meta.get("seed"),
                mean_rps=round(s["trace"].get("mean_rps", 0), 3), policy=s["policy"]["name"],
                offered=att["offered"], success_rate=att["success_rate"], joint_slo_rate=att["joint_slo_rate"],
                ttft_p50=att["ttft_p50"], ttft_p90=att["ttft_p90"], tpot_p50=att["tpot_p50"], tpot_p90=att["tpot_p90"],
                energy_j=s["energy_j"], window_energy_j=s["window_energy_j"], mean_power_w=s["mean_power_w"],
                j_per_request=s["j_per_request"], j_per_token=s["j_per_token"], tail_s=s.get("tail_s"),
                plan_events=s.get("controller", {}).get("events", {}).get("plan", 0),
                wake_events=s.get("controller", {}).get("events", {}).get("wake", 0),
                park_events=s.get("controller", {}).get("events", {}).get("park", 0),
                shield_events=len(s.get("controller", {}).get("shield_events", [])), dir=s["_dir"])


def compare(points: list[dict], reference: str = "static_best", slo_floor: float = 0.9,
            degrade_pp: float = 0.01) -> dict:
    """Per point: saving vs reference; per dataset and overall: equal-weight means over valid points.

    A point is valid when the candidate meets the SLO floor and does not degrade success/SLO by more
    than `degrade_pp` versus the reference. Reference-infeasible points are reported as inconclusive.
    """
    rows = [flatten(p) for p in points]
    groups: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        k = (r["model"], r["dataset"], r["scale"], r["mean_rps"], r["seed"])
        groups.setdefault(k, {})[r["policy"]] = r
    comparisons = []
    for k, by_policy in sorted(groups.items(), key=str):
        ref = by_policy.get(reference)
        for name, r in by_policy.items():
            if name == reference:
                continue
            row = dict(model=k[0], dataset=k[1], scale=k[2], mean_rps=k[3], seed=k[4], policy=name,
                       joint_slo_rate=r["joint_slo_rate"], energy_j=r["energy_j"])
            if ref is None:
                row.update(status="no_reference", saving=None)
            elif ref["joint_slo_rate"] < slo_floor:
                row.update(status="inconclusive_reference_infeasible", saving=None, ref_slo=ref["joint_slo_rate"])
            else:
                saving = 1.0 - r["energy_j"] / ref["energy_j"] if ref["energy_j"] else None
                ok = (r["joint_slo_rate"] >= slo_floor and r["joint_slo_rate"] >= ref["joint_slo_rate"] - degrade_pp
                      and r["success_rate"] >= ref["success_rate"] - degrade_pp)
                row.update(status="valid" if ok else "service_degraded", saving=saving,
                           ref_slo=ref["joint_slo_rate"], ref_energy_j=ref["energy_j"])
            comparisons.append(row)
    summary: dict = {}
    for name in {c["policy"] for c in comparisons}:
        mine = [c for c in comparisons if c["policy"] == name]
        per_dataset = {}
        for ds in sorted({c["dataset"] for c in mine}, key=str):
            valid = [c["saving"] for c in mine if c["dataset"] == ds and c["status"] == "valid"]
            statuses = {}
            for c in mine:
                if c["dataset"] == ds:
                    statuses[c["status"]] = statuses.get(c["status"], 0) + 1
            per_dataset[str(ds)] = dict(points=len([c for c in mine if c["dataset"] == ds]), valid=len(valid),
                                        mean_saving=sum(valid) / len(valid) if valid else None, statuses=statuses)
        means = [d["mean_saving"] for d in per_dataset.values() if d["mean_saving"] is not None]
        summary[name] = dict(per_dataset=per_dataset, overall_saving=sum(means) / len(means) if means else None,
                             all_points_valid=all(c["status"] == "valid" for c in mine))
    return dict(reference=reference, comparisons=comparisons, summary=summary, rows=rows)


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fields: list = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def report(root: Path, reference: str = "static_best", out: Optional[Path] = None) -> dict:
    points = load_points(root)
    result = compare(points, reference)
    out = Path(out or root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(result["rows"], out / "points.csv")
    write_csv(result["comparisons"], out / "comparisons.csv")
    (out / "report.json").write_text(json.dumps(result, indent=1, default=str))
    return result
