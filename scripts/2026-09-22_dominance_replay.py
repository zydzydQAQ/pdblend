#!/usr/bin/env python3
"""Replay all dominance points through the pressure-aware planner on CPU."""
import json
import sys
from pathlib import Path

from pdblend.bench import client as bc
from pdblend.bench.matrix import DEFAULTS, build_trace
from pdblend.bench.run import offline_forecast
from pdblend.control.planner import PlannerConfig, PoolPlanner, SLO
from pdblend.control.policies import get_policy
from pdblend.profile.model import PerfModel

spec_root = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else spec_root / 'cpu-replay.json'
spec = json.loads((spec_root / 'spec.json').read_text())
model = PerfModel.load(Path(spec['defaults']['profile']))
rows = []
for point in spec['points']:
    args = dict(DEFAULTS, **spec['defaults'])
    args.update(point)
    records = bc.load_split(Path(args['corpus']), args['dataset'], 'evaluation')
    trace, _ = build_trace(args, records)
    fc = offline_forecast(trace)
    planner = PoolPlanner(model, get_policy('pdblend_dominance').planner_config(
        PlannerConfig(8, SLO(*bc.SLOS[args['dataset']]), freqs=model.freqs)))
    planner.cfg.pressure_controls = True
    initial = planner.plan(fc)
    predicted = planner.mixed_pressure(fc, initial.counts.get('M', 0), initial.f_M)
    planner.cfg.pd_pressure_active = (fc.input_p95 >= planner.cfg.pd_min_input_tokens
                                      and predicted['pressure'] >= .75)
    pressured = planner.plan(fc, initial)
    planner.cfg.min_m_instances = 2
    low = planner.plan(fc, pressured)
    rows.append(dict(name=point['name'], dataset=args['dataset'], scale=args['scale'], rate=args['rate'],
                     input_p95=fc.input_p95, pressure=predicted['pressure'],
                     initial=initial.counts, pressured=pressured.counts, pressured_tau=pressured.tau,
                     low_load=low.counts))
out.write_text(json.dumps(dict(profile=spec['defaults']['profile'], rows=rows), indent=1))
print(json.dumps(dict(points=len(rows), output=str(out)), indent=1))
