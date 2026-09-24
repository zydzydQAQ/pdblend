#!/usr/bin/env python3
"""Freeze a read-only inventory of comparison records; never select by outcome."""
import csv
import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/plot-inventory-20260925-v1"
STANDARD = {0.25, 0.5, 0.75, 1.0}
SYSTEMS = ["mixed", "dynamollm", "distserve", "ecoserve", "pdblend"]


def write_csv(name, rows, fields=None):
    if not rows and fields is None:
        return
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    OUT = parser.parse_args().out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/compare.csv"
    raw = source.read_bytes()
    (OUT / "compare-snapshot.csv").write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    observed = [r for r in rows if r["status"] == "measured"]
    write_csv("observations.csv", observed)
    write_csv("failed-attempts.csv", [r for r in rows if r["status"] == "failed"])
    grouped = defaultdict(list)
    for row in rows:
        key = (row["model_id"], row["dataset"], row["system"], float(row["rate_scale"]))
        grouped[key].append(row)
    coverage = []
    for (model, dataset, system, scale), records in sorted(grouped.items()):
        measured = [r for r in records if r["status"] == "measured"]
        coverage.append({
            "model_id": model,
            "dataset": dataset,
            "system": system,
            "rate_scale": scale,
            "offered_rps": ";".join(sorted({r["offered_rps"] for r in records}, key=float)),
            "standard_point": scale in STANDARD,
            "measured_records": len(measured),
            "failed_records": sum(r["status"] == "failed" for r in records),
            "prepared_records": sum(r["status"] == "prepared" for r in records),
            "service_energy_records": sum(bool(r["energy_service_j"]) for r in measured),
            "service_and_tail_records": sum(bool(r["energy_service_j"] and r["energy_tail_j"]) for r in measured),
            "measured_revisions": ";".join(sorted({r["revision"] for r in measured})),
            "measured_receipts": ";".join(r["receipt_path"] for r in measured),
            "note": "Coverage across revisions; no revision or attempt selected for plotting.",
        })
    write_csv("coverage.csv", coverage)
    write_csv("extended-coverage.csv", [r for r in coverage if not r["standard_point"]])
    rates = sorted({(r["model_id"], r["dataset"], float(r["rate_scale"]), float(r["offered_rps"])) for r in observed})
    write_csv("rate-map.csv", [dict(zip(["model_id", "dataset", "rate_scale", "offered_rps"], r)) for r in rates])
    matrix = []
    for model in ["7B", "14B", "32B"]:
        for dataset in ["alpaca", "sharegpt", "longbench"]:
            item = {"model": model, "dataset": dataset}
            for system in SYSTEMS:
                matching = [r for r in coverage if r["model_id"] == f"Qwen2.5-{model}-Instruct" and r["dataset"] == dataset and r["system"] == system]
                item[system + "_measured_scales"] = ";".join(format(r["rate_scale"], ".12g") for r in matching if r["measured_records"])
                item[system + "_unmeasured_scales"] = ";".join(format(r["rate_scale"], ".12g") for r in matching if not r["measured_records"])
            matrix.append(item)
    write_csv("matrix.csv", matrix)
    recent_path = ROOT / "results/2026-09-25/sharegpt-lowload-retest-v1/results.csv"
    recent_raw = recent_path.read_bytes()
    (OUT / "retest-snapshot.csv").write_bytes(recent_raw)
    summary = {
        "snapshot_time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_path": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_mtime": datetime.fromtimestamp(source.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat(),
        "record_status_counts": dict(Counter(r["status"] for r in rows)),
        "measured_records": len(observed),
        "unique_measured_coordinates": sum(bool(r["measured_records"]) for r in coverage),
        "unique_standard_measured_coordinates": sum(bool(r["measured_records"]) and r["standard_point"] for r in coverage),
        "unique_extended_measured_coordinates": sum(bool(r["measured_records"]) and not r["standard_point"] for r in coverage),
        "coordinates_with_any_service_energy": sum(bool(r["service_energy_records"]) for r in coverage),
        "measured_service_energy_records": sum(bool(r["energy_service_j"]) for r in observed),
        "measured_strict_rank_eligible_records": sum(r["strict_rank_eligible"].lower() == "true" for r in observed),
        "retest_source_path": str(recent_path),
        "retest_source_sha256": hashlib.sha256(recent_raw).hexdigest(),
        "definitions": {
            "coordinate": "model_id, dataset, system, rate_scale; revision is retained separately",
            "measured": "compare.csv status=measured; does not mean SLO pass, full energy, or formal qualification",
            "prepared": "Not evidence of an executed measurement; may duplicate another record already measured",
            "failed": "Attempted but lacks a complete comparison observation; partial native metrics excluded",
            "retests": "Separate snapshot; not merged with comparison coordinates or counted as independent seeds",
            "energy": "Presence of service-window energy only; no inferred values and no outcome-based selection",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "definitions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
