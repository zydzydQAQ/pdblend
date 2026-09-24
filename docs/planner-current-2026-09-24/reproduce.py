#!/usr/bin/env python3
"""Run the current planner on a synthetic, CPU-only explanatory workload.

This does not launch engines or claim measured GPU performance. All outputs
live next to this script; source hashes identify the implementation explained.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests/pdblend')]

from synthetic import synthetic_model
from pdblend.planner.forecast import Forecast, InFlightWork
from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO, assign_roles
from pdblend.online.policies import get_policy
from pdblend.online.router import Router


def demand(output):
    inputs = (256, 512, 2048, 4096) * 16
    return Forecast(18, 0, 1728, 4096, output, 0, inputs, (output,) * len(inputs),
                    length_pairs=tuple((i, output) for i in inputs))


def main():
    policy = get_policy('pdblend')
    planner = PoolPlanner(synthetic_model(), policy.planner_config(PlannerConfig(8, SLO(1, .020))))
    fc = demand(64)
    old = planner.evaluate({'M': 8}, 2520, 2520, 2520, 0, fc)
    steady = planner.candidates(fc)
    chosen = planner.plan(fc, old)
    assert old is not None and chosen.tau == 1024
    assert chosen.counts == {'P': 2, 'D': 1, 'M': 4, 'off': 1}
    assert steady[0].f_M == 2100 and chosen.f_M == 2520

    specs = {
        'A_current': ({'M': 8}, 2520, 2520, 2520, 0),
        'B_mixed6': ({'M': 6, 'off': 2}, 2520, 2520, 2520, 0),
        'C_lowest_steady': ({'P': 2, 'D': 1, 'M': 4, 'off': 1}, 2520, 2520, 2100, 1024),
        'D_selected': ({'P': 2, 'D': 1, 'M': 4, 'off': 1}, 2520, 2520, 2520, 1024),
        'E_tau4096': ({'P': 2, 'D': 1, 'M': 4, 'off': 1}, 2100, 2520, 2520, 4096),
        'F_mixed5': ({'M': 5, 'off': 3}, 2520, 2520, 2520, 0),
    }
    rows = {}
    for label, spec in specs.items():
        relaxed = planner.evaluate(*spec, fc, strict=False)
        strict = planner.evaluate(*spec, fc)
        switch = planner.switch_energy_j(old, relaxed) if relaxed else None
        rows[label] = dict(feasible=strict is not None, plan=asdict(relaxed) if relaxed else None,
                           switch_j=switch, total_energy_j=60 * relaxed.power_w + switch if relaxed else None)
    assert rows['C_lowest_steady']['total_energy_j'] > rows['D_selected']['total_energy_j']

    stages = []
    current = old
    roles = {f'G{i}': 'M' for i in range(8)}
    for output in (64, 128, 256):
        forecast = demand(output)
        candidates = planner.candidates(forecast)
        old_relaxed = planner.evaluate(current.counts, current.f_P, current.f_D, current.f_M,
                                       current.tau, forecast, strict=False)
        old_strict = planner.evaluate(current.counts, current.f_P, current.f_D, current.f_M,
                                      current.tau, forecast)
        new = planner.plan(forecast, current)
        switch = planner.switch_energy_j(current, new)
        roles_before = dict(roles)
        roles = assign_roles(roles, new.counts)
        pd, mixed = forecast.split_forecasts(new.tau)
        has_pd = bool(new.counts.get('P'))
        stages.append(dict(output_tokens=output, forecast=asdict(forecast),
                           current_rechecked=asdict(old_relaxed) if old_relaxed else None,
                           current_feasible=old_strict is not None, feasible_candidates=len(candidates),
                           selected=asdict(new), roles_before=roles_before, roles_after=roles,
                           switch_j=switch, horizon_cost_j=60 * new.power_w + switch,
                           pd_rps=pd.rate_rps if has_pd else 0,
                           mixed_rps=mixed.rate_rps if has_pd else forecast.rate_rps,
                           branch_pd=asdict(pd) if has_pd else None,
                           branch_m=asdict(mixed) if has_pd else asdict(forecast)))
        current = new
    assert stages[1]['selected']['tau'] == 4096 and not stages[1]['current_feasible']
    assert stages[2]['feasible_candidates'] == 0 and stages[2]['selected']['detail']['fallback']
    assert stages[2]['selected']['tpot_s'] > .017

    # Normal homogeneous Router, matching the concrete G0...G7 mapping.
    router = Router([f'G{i}' for i in range(8)], pd_threshold_tokens=1024)
    router.set_roles({iid: ('parked' if role == 'off' else role)
                      for iid, role in stages[0]['roles_after'].items()})
    for iid, value in {'G0': 2, 'G1': 0, 'G2': 1, 'G3': 3, 'G6': 8}.items():
        router.loads[iid].inflight_seqs = value
    router.loads['G4'].inflight_prefill_tokens = 4096
    router.loads['G5'].inflight_prefill_tokens = 1024
    short = router.dispatch('short-256', 256, 64)
    long = router.dispatch('long-2048', 2048, 64)
    assert (short.path, short.decode_instance) == ('M', 'G1')
    assert (long.path, long.prefill_instance, long.decode_instance) == ('PD', 'G5', 'G6')
    reserved = {i: asdict(v) for i, v in router.loads.items()}
    router.first_token(short)
    router.first_token(long)
    first = {i: asdict(v) for i, v in router.loads.items()}
    router.finish(short, 64)
    router.finish(long, 64)
    finished = {i: asdict(v) for i, v in router.loads.items()}
    assert finished['G1']['inflight_seqs'] == 0
    assert reserved['G5']['inflight_prefill_tokens'] == 3072
    assert first['G5']['inflight_prefill_tokens'] == 1024
    assert reserved['G6']['inflight_seqs'] == 9 and finished['G6']['inflight_seqs'] == 8

    model = planner.model
    sources = ['src/pdblend/planner/pool.py', 'src/pdblend/planner/forecast.py',
               'src/pdblend/planner/topology.py', 'src/pdblend/planner/native_layout.py',
               'src/pdblend/online/policies.py', 'src/pdblend/online/controller.py',
               'src/pdblend/online/router.py', 'src/pdblend/online/shield.py',
               'src/pdblend/online/server.py', 'src/pdblend/online/observations.py',
               'src/pdblend/profile/query/model.py', 'src/pdblend/bench/run.py',
               'tests/pdblend/synthetic.py']
    data = dict(label='Current source; synthetic CPU example, not measured GPU results',
                policy=asdict(policy), config=asdict(planner.cfg), rows=rows, stages=stages,
                route_example=dict(short=asdict(short), long=asdict(long), reserved=reserved,
                                   after_first_token=first, after_finish=finished,
                                   kv_bytes=2048 * model.kv_bytes_per_token,
                                   transfer_ms=model.transfer_seconds(2048) * 1000),
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                assertions_passed=True)
    (HERE / 'example.json').write_text(json.dumps(data, indent=2, ensure_ascii=False, default=sorted) + '\n')
    for label, row in rows.items():
        p = row['plan']
        print(label, 'feasible=', row['feasible'],
              'W=', round(p['power_w'], 6) if p else None,
              'J=', round(row['total_energy_j'], 6) if p else None,
              'TTFT_ms=', round(p['ttft_s'] * 1000, 6) if p else None,
              'TPOT_ms=', round(p['tpot_s'] * 1000, 6) if p else None)
    print('Feasible candidates:', [s['feasible_candidates'] for s in stages])
    print('Assertions passed; example.json written.')


if __name__ == '__main__':
    main()
