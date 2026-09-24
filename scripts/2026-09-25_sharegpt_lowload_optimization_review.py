#!/usr/bin/env python3
"""CPU-only review of the frozen controller; no GPU discovery or hardware actions."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/home/pdblend4')
OUT = ROOT/'results/2026-09-25/sharegpt-lowload-retest-v1'
SOURCE = ROOT/'results/2026-09-24/pdblend-matrix-integration-candidate-v1/sources/d4fe2c6ac5f5b753e710d56f8c072a276eb6a9032e298a12e567ad4feac222fb'
sys.path.insert(0, str(SOURCE))

from pdblend.bench.client import Request
from pdblend.bench.run import _make_controller
from pdblend.bench.pdblend_runtime_options import CONTROL_OPTIONS
from pdblend.online.policies import get_policy
from pdblend.online.router import Router, RequestRecord
from pdblend.online.shield import Shield, Pressure
from pdblend.planner.pool import Plan, SLO
from pdblend.profile.query.versions import load_profile


def bound(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


async def review():
    template = json.loads((OUT/'templates/pd1500.json').read_text())
    choice = json.loads((OUT/'candidate-choice.json').read_text())
    trace = json.loads(Path(template['trace']['path']).read_text())
    loaded = load_profile(template['inputs']['profiles'][0]['path'], system='pdblend',
        model_id=template['model_id'], tp=1, pp=1, usage='development')
    instances = {f'pd{i}':SimpleNamespace(spec=SimpleNamespace(gpus=(i,), generation=0,
        max_num_seqs=32, max_model_len=8192, tp=1, pp=1, pool_id='')) for i in range(8)}
    class FakeFleet:
        def __init__(self): self.instances = instances
        def __getitem__(self, iid): return instances[iid]
    calls = []
    fake_gpus = SimpleNamespace(set_clock=lambda gpu,mhz:calls.append([gpu,mhz]),
        reset_clock=lambda gpu:calls.append([gpu,None]))
    ctl = _make_controller(FakeFleet(), Router(instances), fake_gpus, loaded.model,
        get_policy('pdblend'), SLO(**template['slo']), [Request(**r) for r in trace['requests']],
        OUT, 10., initial_plan=Plan(**choice['plan']),
        pdblend_runtime={k:v for k,v in choice['runtime_options'].items() if k in CONTROL_OPTIONS})
    ctl.log_path = None
    async def fake_hardware(iid, operation, action): return action()
    ctl._hardware = fake_hardware
    cfg = ctl.planner.cfg
    startup = ctl._startup_plan(ctl.initial_plan)
    assert cfg.freqs == (900,1200,1500) and ctl.max_freq == 1500
    assert startup.f_M == startup.f_P == startup.f_D == 1500
    assert cfg.min_m_instances == 4 and not cfg.capacity_floors
    assert ctl._fail_open_plan().f_M == 1500
    await ctl._set_clock('pd0', 2520)
    assert calls[-1] == [0,1500]
    ctl.plan_now = startup
    async def fake_execute(plan): ctl.plan_now = plan
    ctl.execute = fake_execute
    ctl.freqs = {iid:900 for iid in instances}
    ctl.router.deadline_risk_snapshot = lambda **kw:dict(risk=True, unavailable=False)
    await ctl._deadline_clock_action(dict(risk=True))
    assert ctl._deadline_floor_mhz == 1500 and ctl.plan_now.f_M == 1500

    # Build live request records that have a correlated short token gap while
    # accumulated TPOT and the whole output budget remain comfortably safe.
    now = 100.
    records = [RequestRecord(str(i), 'M', 'pd0', 'pd0', 500, 512,
        submitted_s=96.8, first_token_s=97., tokens_so_far=100,
        last_token_s=now-(.573 if i<6 else .02)) for i in range(12)]
    shield = Shield(SLO(**template['slo']), mode='budget_aware')
    pressure = shield.observe(records, now)
    assert pressure.decode and shield._collective_only(pressure)
    assert not pressure.decode_budget_risks and not pressure.decode_sustained_stalls
    assert not pressure.decode_short_output_risks and not pressure.prefill
    assert shield.update(pressure, now) == 1
    low = replace(ctl.initial_plan, f_M=900)
    capped = shield.apply(low, pressure, ctl.max_freq)
    original = shield.apply(low, pressure, 2100)
    assert capped.f_M == 1500 and original.f_M == 2100 and capped.counts == low.counts
    # Repeated short collective evidence does not prolong the 30-second probe.
    assert shield.update(pressure, now+1.) == 1
    assert shield.update(pressure, now+31.) == 0

    diagnosis = json.loads((OUT/'diagnosis.json').read_text())
    window = Path(diagnosis['original_receipts']['pd2100']['path']).parent
    native = json.loads((window/'run/native-result.json').read_text())
    rows = [json.loads(line) for line in (window/'run/controller.jsonl').read_text().splitlines()]
    plans = [r for r in rows if r.get('kind') == 'plan']
    events = []
    fractions = (.25,.5,.500001,.75)
    for row in rows:
        p = row.get('pressure') or {}
        if row.get('kind') != 'forecast' or not p.get('decode'):
            continue
        pressure = Pressure(**p)
        thresholds = {}
        for fraction in fractions:
            s = Shield(SLO(**template['slo']), mode='budget_aware', stalled_fraction=fraction)
            # Replay classification from the logged counters. Strong risk is
            # independent of the fraction; recompute decode, not just criterion.
            q = replace(pressure, decode=bool(pressure.decode_sustained_stalls
                or pressure.decode_budget_risks or pressure.decode_short_output_risks
                or pressure.tpot_p90 > .8*template['slo']['tpot_s']
                or (pressure.decode_stalled >= s.stalled_min_requests
                    and pressure.decode_stalled_fraction >= fraction)))
            thresholds[str(fraction)] = dict(decode=q.decode, collective_only=s._collective_only(q))
        previous = next((p for p in reversed(plans) if p['t'] < row['t']), None)
        events.append(dict(offset_s=row['t']-native['service_started_s'], pressure=p,
            preceding_plan_f_M=previous['f_M'] if previous else None,
            seconds_since_preceding_plan=row['t']-previous['t'] if previous else None,
            fraction_counterfactuals=thresholds))
    return dict(schema='lowload-optimization-review-cpu/v1', hardware_executed=False,
        frozen_source=str(SOURCE), source_files={name:bound(SOURCE/name) for name in (
            'pdblend/online/controller.py','pdblend/online/shield.py','pdblend/bench/run.py',
            'pdblend/planner/pool.py','pdblend/bench/low_m_tuning.py','pdblend/bench/capacity_floor_v2.py')},
        cpu_check=dict(passed=True, source_import=str(sys.modules['pdblend.online.controller'].__file__),
            planner_frequency_grid=list(cfg.freqs), controller_max_frequency=ctl.max_freq,
            startup_frequency=startup.f_M, fail_open_frequency=ctl._fail_open_plan().f_M,
            direct_clock_2520_request_was_capped_to=calls[0][1], deadline_floor=ctl._deadline_floor_mhz,
            mixed_floor=cfg.min_m_instances, qualified_floor_count=len(cfg.capacity_floors),
            synthetic_collective_pressure=asdict(shield.observe(records,now)),
            synthetic_original_shield_frequency=original.f_M, synthetic_candidate_shield_frequency=capped.f_M,
            collective_probe_added_capacity=False, collective_probe_quiet_decay_s=30),
        historical_controller=bound(window/'run/controller.jsonl'), historical_collective_events=events,
        limitations=['Synthetic records verify the rule, not physical frequency response or latency.',
            'Logged controller pressure is sampled; suppressing a rescue changes subsequent observations.',
            'Neither replay nor historical success establishes the counterfactual safety of suppressing rescue.'])


if __name__ == '__main__':
    print(json.dumps(asyncio.run(review()), indent=2))
