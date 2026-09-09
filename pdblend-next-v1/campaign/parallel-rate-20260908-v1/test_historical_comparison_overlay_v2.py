"""Actual720 input,19 diagnosed cells and counterexamples without editing raw files."""
import copy,json,hashlib,os,socket,time
from pathlib import Path
import historical_comparison_overlay_v2 as v
R=Path(__file__).resolve().parent

def run():
 points=json.loads((R/'reports/historical-scale-before-suffix-001/results.json').read_text())['points'];before=copy.deepcopy(points);sources={};ref=dict(path=str(R/'C/all-model-original-Dynamo90-arrival-audit-v2.json'),sha256=v.legacy.AUDIT_SHA)
 overlay=v.verify(points,ref,sources);view=v.comparison_view(points,overlay)
 assert points==before and len(points)==720 and len(view)==720
 assert sum(p['phase']=='main' and p['metrics_verified'] for p in points)==450 and sum(p['phase']=='scale' and p['metrics_verified'] for p in points)==259
 assert {p['cell_id'] for p in view if p['raw_arithmetic_verified'] and not p['scientific_comparison_eligible']}==v.QUARANTINED
 for a,b in zip(points,view):
  assert a['energy_j']==b['energy_j'] and a.get('failed_requests')==b.get('failed_requests')
  if a['cell_id'] not in v.QUARANTINED:assert a['metrics_verified']==b['metrics_verified']
 assert next(p for p in view if p['cell_id']=='32b-longbench-r1-s701-w100-ecoserve-slo1')['scientific_comparison_eligible'] is True
 checks=['actual720_unchanged','raw450_scale259_gap11_unchanged','exact19_exclusions','raw_energy_failed_counts_preserved','nonwindow_capacity_negative_retained']
 for name,mut in [('missing_exclusion',lambda x:x['quarantined'].pop()),('duplicate_exclusion',lambda x:x['quarantined'].append(x['quarantined'][0])),('changed_energy',lambda x:x['quarantined'][0].update(energy_j=-1)),('false_valid',lambda x:x['quarantined'][0].update(scientific_comparison_eligible=True))]:
  x=copy.deepcopy(overlay);mut(x)
  try:v.comparison_view(points,x)
  except ValueError:checks.append(name+'_rejected')
  else:raise AssertionError(name)
 altered=copy.deepcopy(points);next(p for p in altered if p['cell_id'] in v.ECO_QUARANTINED)['energy_j']=-1
 try:v.verify_eco(altered,[dict(path=p,sha256=h) for p,h in v.ECO_AUDITS.items()],{})
 except ValueError:checks.append('raw_metric_mismatch_rejected')
 else:raise AssertionError('raw metric mismatch')
 return dict(passed=True,count=len(checks),checks=checks,physical_host=socket.gethostname(),pid=os.getpid(),created_s=time.time(),source=dict(path=str(Path(__file__).resolve()),sha256=v.sha(__file__)),overlay_source=dict(path=str(Path(v.__file__)),sha256=v.sha(v.__file__)),raw_inputs_reverified=len(sources),quarantined=19,original_records_unchanged=True)
if __name__=='__main__':print(json.dumps(run()))
