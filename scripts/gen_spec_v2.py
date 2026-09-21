#!/usr/bin/env python3
"""Generate the eval-7b-v2 spec: 9 rate scales, baselines at every scale, pdblend last.

Run order: sharegpt-x0.5 quick-five -> all baseline points -> pdblend points,
so baselines are frozen before any pdblend measurement starts.
"""
import json
from pathlib import Path

from pdblend2.bench.matrix import eval_spec

SCALES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QUICK_FIVE = [f"sharegpt-x0.5-{p}" for p in ("mixed", "distserve_static", "dynamollm", "ecoserve", "pdblend")]

spec = eval_spec(
    Path("results/v2/profile-7b/profile.json"), "0,1,2,3,4,5,6,7", Path("results/v2/eval-7b-v2"),
    corpus=Path("datasets/prepared/2026-09-21-7b-v2-half"),
    scales=SCALES, core=("mixed", "pdblend"),
    ported=("distserve_static", "dynamollm", "ecoserve"), ported_scales=SCALES, ported_datasets=None,
    reduced_core=("mixed_dvfs", "mixed_dvfs_park", "static_best"),
    ablations=("pdblend_no_park", "pdblend_no_pd", "pdblend_no_shield", "pdblend_fixed_pools"),
    reduced_datasets=("alpaca", "sharegpt", "longbench"), reduced_scale=0.5,
    stages="", azure=(), seed=701)

by_name = {p["name"]: p for p in spec["points"]}
first = [by_name[n] for n in QUICK_FIVE]
rest = [p for p in spec["points"] if p["name"] not in QUICK_FIVE]
baselines = [p for p in rest if p["policy"] != "pdblend"]
pdblend_pts = [p for p in rest if p["policy"] == "pdblend"]
spec["points"] = first + baselines + pdblend_pts
root = Path(spec["root"])
(root / "spec.json").write_text(json.dumps(spec, indent=1))
print(json.dumps(dict(capacity_rps=spec["capacity_rps"], groups=spec["groups"], points=len(spec["points"]),
                      first5=[p["name"] for p in spec["points"][:5]], root=str(root)), indent=1))
