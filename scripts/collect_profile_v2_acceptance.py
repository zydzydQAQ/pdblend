#!/usr/bin/env python3
"""Compare each completed fixed run against its own seed's trace prediction."""
import argparse
import json
from pathlib import Path
from pdblend.control.planner import PoolPlanner, PlannerConfig, SLO
from pdblend.profile.model import PerfModel
from pdblend.profile.acceptance import SEEDS, m2_gate, relative_error
from pdblend.profile.validation import benchmark_evidence, verify_audit_binding

p=argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('out', type=Path)
p.add_argument('profile', type=Path)
p.add_argument('--prefix', default='profile-v2-round6')
a=p.parse_args()
model=PerfModel.load(a.profile)
verify_audit_binding(a.profile)
planner=PoolPlanner(model,PlannerConfig(8,SLO(5,.15),freqs=model.freqs))
result=dict(profile=str(a.profile.resolve()),m2_rows=[],fixed={},failures=[])
for name,n,freq in [('m2-1800',2,1800),('m4-2100',4,2100),('m5-2100',5,2100),('m6-2100',6,2100),('m4-2520',4,2520)]:
 rows=[]
 for seed in SEEDS:
  folder=a.root/f'{a.prefix}-{name}-seed-{seed}'
  if not (folder/'measurement/summary.json').is_file(): continue
  try:
   d,fc,provenance=benchmark_evidence(folder,a.profile,seed,'manual',f'M={n},L1={8-n}',f'P=2520,D=2520,M={freq}')
   # No PDblend M>=4 policy floor here: physical model feasibility is independent.
   plan=planner.evaluate({'M':n,'L1':8-n},2520,2520,freq,0,fc)
   s=d['slo']; observed=d['mean_power_w']; predicted=plan.power_w if plan else None
   power_error=relative_error(predicted,observed) if predicted is not None else None
   measured_feasible=s['joint_slo_rate']>=.9 and s['ttft_p99']<=5 and s['tpot_p99']<=.15
   row=dict(seed=seed,complete=True,joint_slo_rate=s['joint_slo_rate'],ttft_p90=s['ttft_p90'],ttft_p99=s['ttft_p99'],tpot_p99=s['tpot_p99'],mean_power_w=observed,predicted_power_w=predicted,power_error=power_error,
            j_per_token=d['j_per_token'],model_feasible=plan is not None,measured_feasible=measured_feasible,
            feasibility_agrees=(plan is not None)==measured_feasible,summary=str(folder/'measurement/summary.json'),provenance=provenance)
   row['passed']=measured_feasible and row['feasibility_agrees'] and power_error is not None and power_error<=.05
  except (ValueError,KeyError,OSError) as exc:
   row=dict(seed=seed,complete=False,passed=False,error=str(exc))
  rows.append(row)
  if not row['passed']: result['failures'].append(dict(layout=name,**row))
 if name=='m2-1800': result['m2_rows']=rows
 else: result['fixed'][name]=rows
result['m2_gate']=m2_gate(result['m2_rows'])
result['adaptive_eligible_layouts']=[name for name,rows in result['fixed'].items() if rows and all(r['passed'] for r in rows)]
result['status']='calibration_experiment'
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1))
