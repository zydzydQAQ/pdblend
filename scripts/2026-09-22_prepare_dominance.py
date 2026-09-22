#!/usr/bin/env python3
"""Create an isolated PDblend dominance screening matrix.

Only PDblend points from an existing matrix spec are copied. Baseline points,
profiles and historical summaries are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source_spec", type=Path)
p.add_argument("out_root", type=Path)
p.add_argument("profile", type=Path)
args = p.parse_args()
source = json.loads(args.source_spec.read_text())
points = []
for point in source["points"]:
    if point.get("policy") != "pdblend":
        continue
    row = dict(point)
    row["policy"] = "pdblend_dominance"
    row["profile"] = str(args.profile.resolve())
    points.append(row)
out = dict(source, root=str(args.out_root.resolve()), points=points)
args.out_root.mkdir(parents=True, exist_ok=True)
(args.out_root / "spec.json").write_text(json.dumps(out, indent=1))
(args.out_root / "provenance.json").write_text(json.dumps({
    "source_spec": str(args.source_spec.resolve()),
    "source_spec_sha256": hashlib.sha256(args.source_spec.read_bytes()).hexdigest(),
    "profile": str(args.profile.resolve()),
    "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
    "policy": "pdblend_dominance",
    "baseline_points_excluded": True,
    "points": len(points),
}, indent=1))
print(json.dumps({"root": str(args.out_root), "points": len(points)}, indent=1))
