#!/usr/bin/env python3
"""Offline sanity: what does pdblend's warm-start plan look like per controlled point?

Builds the same trace the runner builds (same seed/rate/duration), takes the offline
forecast, and prints the planner's choice with pdblend's config vs static_best's.
"""
import json
import sys
from pathlib import Path

from pdblend2.bench import client as bc
from pdblend2.bench.matrix import DEFAULTS, build_trace
from pdblend2.bench.run import offline_forecast
from pdblend2.control.planner import PlannerConfig, PoolPlanner, SLO
from pdblend2.control.policies import get_policy
from pdblend2.profile.model import PerfModel

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2/eval-7b-v2")
spec = json.loads((root / "spec.json").read_text())
model = PerfModel.load(Path(spec["defaults"]["profile"]))
corpus = Path(spec["defaults"]["corpus"])

for p in spec["points"]:
    if p["policy"] != "pdblend" or "-staged-" in p["name"] or "azure" in p:
        continue
    ds, sc = p["dataset"], p["scale"]
    records = bc.load_split(corpus, ds, "evaluation")
    a = dict(DEFAULTS, **spec["defaults"], **p)
    trace, _ = build_trace(a, records)
    fc = offline_forecast(trace)
    slo = SLO(*bc.SLOS[ds])
    row = {}
    for pol in ("pdblend", "static_best"):
        cfg = get_policy(pol).planner_config(PlannerConfig(slots=8, slo=slo, freqs=model.freqs))
        plan = PoolPlanner(model, cfg).plan(fc)
        row[pol] = (plan.counts, plan.f_P, plan.f_D, plan.f_M, round(plan.power_w)) if plan else "fallback"
    print(f"{p['name']:<28} rate={p['rate']:>7.2f} pdblend={row['pdblend']} static={row['static_best']}")
