#!/usr/bin/env python3
"""Audit profile evidence, model residuals and independent fixed-layout observations."""
import argparse
import json
from pathlib import Path

from pdblend.bench import client as bc
from pdblend.bench.run import offline_forecast
from pdblend.control.planner import PlannerConfig, PoolPlanner, SLO
from pdblend.control.policies import get_policy
from pdblend.profile.model import PerfModel
from pdblend.profile.merge import sha256
from pdblend.profile.acceptance import quality_audit, relative_error

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / 'results/2026-09-21/pdblend-optimization'
CORPUS = ROOT / 'datasets/prepared/2026-09-21-7b-v2-half'
LAYOUTS = [('m2-1800', 2, 1800, None), ('m4-2100', 4, 2100, 'fixed-m4-2100'),
           ('m5-2100', 5, 2100, 'fixed-m5'), ('m6-2100', 6, 2100, None),
           ('m4-2520', 4, 2520, 'fixed-m4-2520')]


def forecast_for(corpus, dataset, rate, seed, duration):
    return offline_forecast(bc.poisson_trace(bc.load_split(corpus, dataset, 'evaluation'), rate, duration, seed, 'audit'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('profile', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    candidate = args.profile.resolve()
    raw = json.loads((candidate.parent / 'raw.json').read_text())
    model = PerfModel.load(candidate)
    report = quality_audit(raw, model, candidate.parent)
    report.update(profile=str(candidate), profile_sha256=sha256(candidate),
                  raw_sha256=sha256(candidate.parent / 'raw.json'), environment=raw['environment'],
                  status='experimental', ground_truths=[], pending=[], points_replay=[], comparisons=[])
    fc = forecast_for(CORPUS, 'sharegpt', 8.109, 701, 300)
    # Evaluate physical feasibility separately from the adaptive empirical safety floor.
    models = [('frozen_baseline', ROOT / 'results/v2/profile-7b/profile.json'),
              ('previous_pdblend', ROOT / 'results/v2/profile-7b-pdblend/profile.json'),
              ('profile_v2', candidate)]
    for name, path in models:
        m = PerfModel.load(path)
        planner = PoolPlanner(m, PlannerConfig(8, SLO(5, .15), freqs=m.freqs))
        comparison = dict(name=name, path=str(path), sha256=sha256(path),
                          training_residuals=m.residuals, layouts=[])
        for label, n, f, history in LAYOUTS:
            # Fixed-layout model feasibility is evaluated without PDblend's
            # temporary M>=4 adaptive safety floor.
            p = planner.evaluate({'M': n, 'L1': 8 - n}, f, f, f, 0, fc)
            measurement = OPT / history / 'summary.json' if history else None
            measured = json.loads(measurement.read_text()) if measurement and measurement.exists() else None
            row = dict(label=label, counts={'M': n, 'L1': 8 - n}, freq_mhz=f, feasible=p is not None,
                       predicted_w=p.power_w if p else None,
                       ttft_s=p.ttft_s if p else None, tpot_s=p.tpot_s if p else None)
            if measured:
                row.update(measurement=str(measurement), measurement_sha256=sha256(measurement),
                           measured_w=measured['mean_power_w'], measured_window_w=measured['window_mean_power_w'],
                           measured_joint_slo=measured['slo']['joint_slo_rate'],
                           power_error=relative_error(p.power_w, measured['mean_power_w']) if p else None)
                row['verdict'] = 'PASS' if p and row['power_error'] <= .05 and row['measured_joint_slo'] >= .9 else 'FAIL'
            else:
                row['verdict'] = 'PENDING'
            comparison['layouts'].append(row)
            if name == 'profile_v2':
                report['ground_truths'].append(row)
                if row['verdict'] == 'FAIL':
                    report['failures'].append(dict(metric='fixed_layout', point=row))
                if not measured:
                    report['pending'].append(dict(label=label, reason='requires independent 300 s fixed-layout measurements'))
        report['comparisons'].append(comparison)
    # Offline replay does not grant publication eligibility.
    spec = json.loads((ROOT / 'results/v2/eval-7b-v2/spec.json').read_text())
    for pt in spec['points']:
        if pt['policy'] != 'pdblend' or 'azure' in pt or '-staged-' in pt['name']:
            continue
        ds = pt['dataset']
        forecast = forecast_for(CORPUS, ds, float(pt['rate']), 701, 300)
        planner = PoolPlanner(model, get_policy('pdblend').planner_config(PlannerConfig(8, SLO(*bc.SLOS[ds]), freqs=model.freqs)))
        plan = planner.plan(forecast)
        report['points_replay'].append(dict(name=pt['name'], counts=plan.counts if plan else None,
            f_M=plan.f_M if plan else None, power_w=plan.power_w if plan else None,
            fallback=plan.detail.get('fallback', False) if plan else True))
    report['basic_passed'] = report['passed'] = not report['failures']
    report['formal_eligible'] = False  # pending fixed/adaptive evidence cannot be bypassed by a marker
    report['status'] = 'basic_passed_pending_gpu_validation' if report['basic_passed'] else 'experimental_failed_audit'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, allow_nan=False))
    marker = candidate.parent / 'AUDIT-PASSED'
    if report['basic_passed']:
        marker.write_text(json.dumps(dict(profile_sha256=report['profile_sha256'], raw_sha256=report['raw_sha256'], audit=str(args.out.resolve()))))
    elif marker.exists():
        marker.unlink()
    print(json.dumps({k: report[k] for k in ('status', 'failures', 'pending')}, indent=1))
    raise SystemExit(0 if report['basic_passed'] else 2)


if __name__ == '__main__':
    main()
