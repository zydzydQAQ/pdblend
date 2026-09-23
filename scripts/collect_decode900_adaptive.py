#!/usr/bin/env python3
"""Validate the active adaptive workload seed and emit labelled statistics."""
import argparse, json, statistics
from pathlib import Path
from pdblend.profile.model import PerfModel
from pdblend.profile.validation import benchmark_evidence, verify_audit_binding
from pdblend.seed_config import SEEDS, has_active_seeds, seed_metadata

p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('profile',type=Path); p.add_argument('out',type=Path); a=p.parse_args()
verify_audit_binding(a.profile)
rows=[]
for seed in SEEDS:
    folder=a.root/f'adaptive-profile-v2-round6-seed-{seed}'
    d,fc,prov=benchmark_evidence(folder,a.profile,seed,'pdblend','', 'P=2520,D=2520,M=2100')
    s=d['slo']; e=d['controller']['events']; rows.append(dict(seed=seed,complete=True,joint_slo_rate=s['joint_slo_rate'],ttft_p90=s['ttft_p90'],ttft_p99=s['ttft_p99'],tpot_p99=s['tpot_p99'],mean_power_w=d['mean_power_w'],window_mean_power_w=d['window_mean_power_w'],j_per_token=d['j_per_token'],plans=e.get('plan',0),wakes=e.get('wake',0),parks=e.get('park',0),paths=s.get('paths',{}),provenance=prov))
def stats(key):
    vals=[r[key] for r in rows]; return dict(n=len(vals),mean=statistics.mean(vals),std=statistics.stdev(vals) if len(vals)>1 else None,worst=min(vals) if key=='joint_slo_rate' else max(vals))
result=dict(profile=str(a.profile.resolve()),profile_sha256=verify_audit_binding(a.profile),rows=rows,stats={k:stats(k) for k in ('joint_slo_rate','ttft_p90','ttft_p99','tpot_p99','mean_power_w','window_mean_power_w','j_per_token','plans','wakes','parks')},gate=dict(all_slo=all(r['joint_slo_rate']>=.9 and r['ttft_p99']<=5 and r['tpot_p99']<=.15 for r in rows),seed_coverage=has_active_seeds(r['seed'] for r in rows)), **seed_metadata())
result['gate']['passed']=all(result['gate'].values())
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1));raise SystemExit(0 if result['gate']['passed'] else 2)
