#!/usr/bin/env python3
"""Compare an isolated profile against the frozen profile and audit M4 admission."""
import json
import sys
from pathlib import Path

from pdblend.bench import client as bc
from pdblend.bench.run import offline_forecast
from pdblend.control.planner import PlannerConfig, PoolPlanner, SLO
from pdblend.control.policies import get_policy
from pdblend.profile.model import PerfModel


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_calibration.py <candidate-profile.json> <out.json>")
    candidate, out = Path(sys.argv[1]), Path(sys.argv[2])
    frozen = Path("results/v2/profile-7b/profile.json")
    corpus = Path("datasets/prepared/2026-09-21-7b-v2-half")
    records = bc.load_split(corpus, "sharegpt", "evaluation")
    trace = bc.poisson_trace(records, 8.109, 300.0, 701, "audit")
    fc = offline_forecast(trace)
    rows = []
    for label, path in (("frozen", frozen), ("candidate", candidate)):
        model = PerfModel.load(path)
        cfg = get_policy("pdblend").planner_config(PlannerConfig(8, SLO(5.0, .15), freqs=model.freqs))
        planner = PoolPlanner(model, cfg)
        row = dict(profile=label, residuals=model.residuals,
                   planner=planner.plan(fc).counts if planner.plan(fc) else None)
        for clocks in (2100, 2520):
            p = planner.evaluate({"M": 4, "L1": 4}, clocks, clocks, clocks, 0, fc)
            row[f"m4_{clocks}"] = None if p is None else dict(power_w=p.power_w, ttft_s=p.ttft_s, tpot_s=p.tpot_s)
        rows.append(row)
    out.write_text(json.dumps(dict(trace=fc.__dict__, rows=rows), indent=2, default=str))
    print(json.dumps(dict(candidate=str(candidate), output=str(out), rows=rows), indent=1, default=str))


if __name__ == "__main__":
    main()
