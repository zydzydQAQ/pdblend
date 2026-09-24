#!/usr/bin/env python3
"""Freeze a small, hash-bound inventory of recorded baseline energy gaps.

This never starts hardware, changes a receipt, or selects results by SLO/energy.
The first recorded complete-energy window at each logical point is the frozen
comparison reference. All earlier incomplete windows remain historical data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def inventory(csv_path):
    csv_path = Path(csv_path).resolve()
    raw_csv = csv_path.read_bytes()
    rows = list(csv.DictReader(raw_csv.decode().splitlines()))
    grouped = {}
    for row in rows:
        if row['system'] == 'pdblend' or not row.get('receipt_path'):
            continue
        receipt_ref = binding(row['receipt_path'])
        if receipt_ref['sha256'] != row['receipt_sha256']:
            raise ValueError('CSV receipt checksum changed: ' + row['point_id'])
        receipt = json.loads(Path(receipt_ref['path']).read_text())
        window = Path(receipt_ref['path']).parent
        point_ref, result_ref = binding(window/'point.json'), binding(window/'result.json')
        point, result = (json.loads(Path(ref['path']).read_text()) for ref in (point_ref, result_ref))
        if (receipt.get('point') != row['point_id'] or point.get('name') != row['point_id']
                or receipt.get('point_sha256') != digest(point)
                or receipt.get('artifacts', {}).get('point.json') != point_ref['sha256']
                or receipt.get('artifacts', {}).get('result.json') != result_ref['sha256']
                or receipt.get('result') != result):
            raise ValueError('recorded point/result binding changed: ' + row['point_id'])
        metrics = result.get('metrics', {})
        start, end = metrics.get('service_start_s'), metrics.get('service_end_s')
        recorded = (metrics.get('duration_s') == 150 and finite(start) and finite(end)
                    and math.isclose(end-start, 150, rel_tol=0, abs_tol=1e-5)
                    and type(metrics.get('offered_requests')) is int and metrics['offered_requests'] > 0)
        if not recorded:
            continue
        source = point.get('source_manifest', point.get('inputs', {}).get('source_manifest'))
        if source and binding(source['path'])['sha256'] != source['sha256']:
            raise ValueError('source manifest changed: ' + row['point_id'])
        entry = dict(point_id=point['name'], model_id=point['model_id'], system=point['system'],
            dataset=point['dataset'], rate_scale=point['scale'], offered_rps=point['rate_rps'],
            seed=point['seed'], duration_s=point['duration_s'], revision=point['revision'],
            point=point_ref, point_sha256=digest(point), receipt=receipt_ref, result=result_ref,
            source_manifest=source, engine_identity=point.get('engine_identity'),
            engine_signature=receipt.get('engine_signature'), trace=point.get('trace'),
            inputs=point.get('inputs'), slo=point['slo'], service_start_s=start,
            energy_service_j=metrics.get('energy_service_j'), energy_tail_j=metrics.get('energy_tail_j'),
            offered_requests=metrics['offered_requests'], successful_requests=metrics.get('successful_requests'),
            recorded_window_complete=receipt.get('recorded_window_complete'),
            cleanup_passed=receipt.get('cleanup_passed'),
            measurement_protocol_version=point.get('measurement_protocol_version'),
            metering_execution=point.get('metering_execution', 'in_process'))
        grouped.setdefault(point['name'], []).append(entry)
    complete, gaps, historical = [], [], []
    for name, attempts in sorted(grouped.items()):
        attempts.sort(key=lambda r: (r['service_start_s'], r['receipt']['sha256']))
        valid = [r for r in attempts if finite(r['energy_service_j']) and finite(r['energy_tail_j'])]
        selected = (valid or attempts)[0]
        selected = dict(selected, selection_rule='first_recorded_complete_service_and_tail_energy_else_first_recorded',
                        historical_receipts=[a['receipt'] for a in attempts])
        if valid:
            complete.append(selected)
        else:
            gaps.append(dict(selected, reason='recorded_requests_missing_full_cycle_energy',
                             missing_components=[name for name in ('energy_service_j', 'energy_tail_j')
                                                 if not finite(selected[name])],
                             rerun_policy='append_one_attempt_with_isolated_metering_preserve_original'))
        historical.extend(a['receipt'] for a in attempts)
    return dict(schema='pdblend-baseline-energy-gap-inventory/v2',
        source_csv=dict(path=str(csv_path), sha256=hashlib.sha256(raw_csv).hexdigest()),
        selection_uses_slo_or_energy_magnitude=False, historical_evidence_modified=False,
        recorded_baseline_points=len(grouped), complete_energy_points=len(complete),
        missing_energy_points=len(gaps), frozen_baselines=complete, gaps=gaps,
        historical_receipts=historical)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('results/compare.csv'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(manifest=binding(args.out), recorded=result['recorded_baseline_points'],
                         complete=result['complete_energy_points'], missing=result['missing_energy_points'])))


if __name__ == '__main__':
    main()
