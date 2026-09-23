#!/usr/bin/env python3
"""Read-only catalog of frozen baseline artifacts, without inventing provenance.

A retrospective summary proves neither the execution image nor the GPU UUIDs
nor that inputs were unchanged while it ran. Keep those fields unknown until
original run manifests establish them. Never copy identities from a candidate.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

BASELINES = ("mixed", "distserve_static", "dynamollm", "ecoserve")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def trace_shape_sha(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        rows = sorted((json.loads(line) for line in f if line.strip()), key=lambda r: r["idx"])
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["idx"], r["arrival_s"], r["input_tokens"], r["max_tokens"]],
                            separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate_spec", type=Path)
    ap.add_argument("baseline_root", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(a.candidate_spec.read_text())
    rows = []
    for point in spec["points"]:
        for policy in BASELINES:
            name = f"{point['dataset']}-x{point['scale']:g}-{policy}"
            folder = a.baseline_root / name
            if not (folder / "summary.json").exists():
                rows.append(dict(name=name, status="missing"))
                continue
            summary = json.loads((folder / "summary.json").read_text())
            artifacts = {str(p.resolve()): sha(p) for p in folder.iterdir()
                         if p.is_file() and (p.name.endswith(".json") or ".jsonl" in p.name)}
            outcome = next((folder / n for n in ("outcomes.jsonl", "outcomes.jsonl.gz")
                            if (folder / n).exists()), None)
            rows.append(dict(
                name=name, status="historical_unverified", summary=str((folder / "summary.json").resolve()),
                artifacts=artifacts, trace_shape_sha256=trace_shape_sha(outcome) if outcome else None,
                observed_metadata=summary.get("trace_meta", {}),
                pending=["original_source_profile_image_hardware_provenance",
                         "before_after_input_verification", "sampled_clock_evidence_if_missing"]))
    report = dict(status="inconclusive", rows=rows,
                  note="Artifact inventory only. No execution success or hardware identity inferred.")
    (a.out / "pairing-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(points=len(rows), status=report["status"])))

if __name__ == "__main__":
    main()
