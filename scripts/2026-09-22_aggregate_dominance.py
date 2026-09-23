#!/usr/bin/env python3
"""Aggregate strict dominance rows for the active single-seed campaign."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pdblend.seed_config import SINGLE_SEED, SEED_POLICY, has_active_seeds
from pdblend.bench.dominance import load_evidence

METRICS = ('j_per_token', 'mean_power_w', 'window_mean_power_w', 'joint_slo_rate',
           'ttft_p99', 'tpot_p99', 'plans', 'wakes', 'parks')


def load_rows(root):
    rows = []
    for p in sorted(Path(root).glob('*-pdblend_dominance/summary.json')):
        d = json.loads(p.read_text())
        evidence = load_evidence(p.parent)
        events = d.get('controller', {}).get('events', {})
        rows.append(dict(name=p.parent.name.removesuffix('-pdblend_dominance'),
                         evidence_complete=evidence is not None,
                         seed=d.get('trace_meta', {}).get('seed'),
                         j_per_token=d.get('j_per_token'), mean_power_w=d.get('mean_power_w'),
                         window_mean_power_w=d.get('window_mean_power_w'),
                         joint_slo_rate=d.get('slo', {}).get('joint_slo_rate'),
                         ttft_p99=d.get('slo', {}).get('ttft_p99'), tpot_p99=d.get('slo', {}).get('tpot_p99'),
                         plans=events.get('plan', 0), wakes=events.get('wake', 0), parks=events.get('park', 0)))
    return rows


def aggregate(root, out):
    groups = {}
    for row in load_rows(root):
        groups.setdefault(row['name'], []).append(row)
    result = []
    for name, rows in groups.items():
        seeds = [r['seed'] for r in rows]
        row = dict(name=name, seeds=json.dumps(seeds),
                   complete=has_active_seeds(seeds) and all(r['evidence_complete'] for r in rows),
                   single_seed=True, seed_policy=SEED_POLICY)
        for metric in METRICS:
            values = [r[metric] for r in rows if isinstance(r[metric], (int, float)) and math.isfinite(r[metric])]
            row[metric + '_mean'] = statistics.mean(values) if values else None
            row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
            row[metric + '_worst'] = min(values) if metric == 'joint_slo_rate' and values else (max(values) if values else None)
        row['status'] = 'single_seed_ready' if row['complete'] else 'screening_only'
        result.append(row)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in result for k in r})
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(result)
    out.with_suffix('.json').write_text(json.dumps({'rows': result, 'single_seed': True,
                                                    'seed_policy': SEED_POLICY}, indent=1))
    print(json.dumps({'points': len(result), 'single_seed': sum(r['complete'] for r in result),
                      'seed_policy': SEED_POLICY, 'output': str(out)}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('out'); a = p.parse_args(); aggregate(a.root, a.out)
