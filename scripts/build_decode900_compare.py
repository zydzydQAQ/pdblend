#!/usr/bin/env python3
"""Write an independent round6 comparison table with multi-seed statistics."""
import csv, json, statistics
from pathlib import Path

root=Path('/home/pdblend4')
out=Path(__file__).resolve().parents[1]/'results/2026-09-22/decode900/round6/compare-pdblend-profile-v2.csv'
rows=[]
baseline=root/'results/v2/eval-7b-v2/compare.csv'
if baseline.exists():
    for r in csv.DictReader(baseline.open()):
        if r.get('name') in ('sharegpt-x0.5-ecoserve','sharegpt-x0.5-pdblend'):
            rows.append(dict(name=r['name'],kind='frozen_baseline',profile=r.get('policy',''),seed_count='',
              j_per_token=r.get('j_per_token',''),mean_power_w=r.get('mean_power_w',''),joint_slo_rate=r.get('joint_slo_rate',''),
              ttft_p99=r.get('ttft_p99',''),tpot_p99=r.get('tpot_p99',''),plans=r.get('plans',''),wakes=r.get('wakes',''),parks=r.get('parks','')))
fixed=json.loads((out.parent/'acceptance/fixed-summary.json').read_text())
for name,group in [('m2-1800',fixed['m2_rows']),*fixed['fixed'].items()]:
    for metric in ('j_per_token','mean_power_w','joint_slo_rate','ttft_p99','tpot_p99'):
        vals=[r[metric] for r in group if r.get('complete')]
        if not vals: continue
        rows.append(dict(name=name,kind='fixed_profile-v2_round6',profile=str(out.parent/'profile.json'),seed_count=len(vals),metric=metric,
                         mean=statistics.mean(vals),std=statistics.stdev(vals) if len(vals)>1 else 0,worst=max(vals)))
adaptive=json.loads((out.parent/'adaptive-summary.json').read_text())
for metric,s in adaptive['stats'].items():
    rows.append(dict(name='adaptive-sharegpt-x0.5',kind='adaptive_profile-v2_round6',profile=adaptive['profile'],seed_count=3,metric=metric,mean=s['mean'],std=s['std'],worst=s['worst']))
out.parent.mkdir(parents=True,exist_ok=True)
fields=sorted({k for r in rows for k in r})
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(f'wrote {out} rows={len(rows)}')
