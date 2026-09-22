#!/usr/bin/env python3
"""Build an isolated PDblend-vs-frozen-baseline matrix comparison."""
import csv, json
from pathlib import Path

root=Path('/home/pdblend4')
matrix=root/'results/2026-09-22/matrix-profile-round6'
out=matrix/'compare-pdblend-profile-round6.csv'
rows=[]
baseline=root/'results/v2/eval-7b-v2/compare.csv'
if baseline.exists():
    for r in csv.DictReader(baseline.open()):
        rows.append(dict(r, comparison='frozen_baseline', profile='results/v2/profile-7b/profile.json'))
for summary in sorted(matrix.glob('*/summary.json')):
    d=json.loads(summary.read_text()); s=d['slo']; c=d.get('controller',{}).get('events',{})
    rows.append(dict(name=summary.parent.parent.name, comparison='pdblend_profile_round6', profile=d.get('profile',''),
                     policy=d.get('policy',{}).get('name',''), dataset=d.get('trace_meta',{}).get('dataset',''),
                     scale=d.get('trace_meta',{}).get('scale',''), seed=d.get('trace_meta',{}).get('seed',''),
                     joint_slo_rate=s.get('joint_slo_rate',''), ttft_p90=s.get('ttft_p90',''),ttft_p99=s.get('ttft_p99',''),
                     tpot_p99=s.get('tpot_p99',''),mean_power_w=d.get('mean_power_w',''),window_mean_power_w=d.get('window_mean_power_w',''),
                     j_per_token=d.get('j_per_token',''),plans=c.get('plan',0),wakes=c.get('wake',0),parks=c.get('park',0),
                     final_roles=json.dumps(d.get('final_roles',{}),sort_keys=True)))
out.parent.mkdir(parents=True,exist_ok=True)
fields=sorted({k for r in rows for k in r})
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(f'wrote {out} rows={len(rows)}')
