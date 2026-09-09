"""CPU-only immutable whole-group Dynamo replacement; no hardware action."""
import copy,json,hashlib,time,sys,math
from pathlib import Path
B=Path(__file__).resolve().parent;REPO=B.parents[2]
sys.path.insert(0,str(B));import baseline_reconciliation_v3 as ledger
p=ledger.p
OUT=B/'cooperative-dynamo8-001'

def main():
 p.need(not OUT.exists(),'new immutable declaration required')
 parent_path=B/'completion-rates-p4-001/declaration.json';parent=p.read(parent_path)
 selected=[c for c in parent['cells'] if c['system']=='dynamollm' and (c['dataset'],c['rate_rps_decimal']) in {('sharegpt','1.25'),('sharegpt','1.5'),('alpaca','5'),('alpaca','4')}]
 selected.sort(key=lambda c:(c['repeat'],{('alpaca','4'):0,('sharegpt','1.25'):1,('sharegpt','1.5'):2,('alpaca','5'):3}[(c['dataset'],c['rate_rps_decimal'])]))
 p.need(len(selected)==8,'exact four-rates two-repeat scope required')
 timing_path=B.parent/'C/B-original-Dynamo30-arrival-audit-v1.json';timing=p.read(timing_path)
 old=next(c for c in timing['points'] if c['cell_id']=='32b-alpaca-r4-s701-w100-dynamollm-slo1')
 cp=p.checked(old['checkpoint']);receipt=p.checked(old['receipt']);s=receipt['summary'];cell=Path(old['receipt']['path']).parents[2]/'cells'/old['cell_id']
 p.need(all(p.sha(f)==h for f,h in old['raw_files'].items()),'old timing raw changed')
 energy=ledger.integral(cell/'power.csv',s['measurement_start_s'],s['measurement_end_s']);p.need(math.isclose(energy,s['energy_j'],rel_tol=1e-9),'old eight-GPU energy differs')
 quarantine=dict(schema='B-original-Alp4-Dynamo-engineering-quarantine-v1',created_s=time.time(),cell_id=old['cell_id'],checkpoint=old['checkpoint'],receipt=old['receipt'],source=old['source'],timing_audit=p.ref(timing_path),raw_timing_evidence=old['raw_files'],raw_energy_measurement_valid=s['measurement_valid'],raw_energy_recomputed_j=energy,scientific_comparison_eligible=False,reason='Synchronous ready-queue event-loop starvation distorts original100s dispatch schedule',actual_dispatch_lateness_max_s=old['actual_dispatch_lateness_max_s'],controller_max_event_gap_s=old['controller_max_event_gap_s'],original_checkpoint_unchanged=True,replacement_scope='same original Alp4 trace/seed701, two predeclared cooperative repeats paired one-to-one with PDB4 repeats; no SG2 rerun above PDB boundary')
 qp=B/'original-alp4-dynamo-engineering-quarantine-001.json';p.write(qp,quarantine,exclusive=True)
 rows=[]
 for i,c in enumerate(selected):
  row=copy.deepcopy(c);row.update(cell_id=c['cell_id'].replace('parallel-rate-p4-completion-','parallel-rate-cooperative-b-v1-'),original_cell_id=c['cell_id'],source_row=c,sequence=i,engineering_replacement=True,execution_status='not_run')
  pdb=next(x for x in parent['cells'] if x['system']=='pdblend' and x['workload_id']==c['workload_id'] and x['repeat']==c['repeat'])
  cp_path=B/'fixed-screen-p4-001/results/checkpoints'/(pdb['cell_id']+'.json');actual=p.read(cp_path)
  p.need(actual['work_complete'] and actual['row']['trace_sha256']==c['trace_sha256'] and actual['row']['slo']==c['slo'],'paired PDB repeat differs')
  row['paired_pdb_checkpoint']=p.ref(cp_path)
  if c['rate_rps_decimal']=='4':row['replaces_original_historical_id']=old['cell_id']
  rows.append(row)
 host=REPO/'releases/five-system100-B32B-baseline-cooperative-v1-runtime'
 decl=dict(schema='B-whole-group-cooperative-Dynamo-eight-v1',created_s=time.time(),ready_for_GPU=False,readiness_scope='CPU declaration only; foreign owner must terminate/restore, exact TP2 physical identity and fresh actual qualification required',cells=rows,parent_declaration=p.ref(parent_path),host_release=str(host),host_manifest=p.ref(host/'manifest.json'),cpu_validation=p.ref(B/'cooperative-b-cpu-validation-001.json'),original_alp4_quarantine=p.ref(qp),new_rate_count=6,original_alp4_repair_count=2,original_newrate_group_rows=[c for c in selected if c['rate_rps_decimal']!='4'],retired_unexecuted_original_newrate_repeat2=[c['cell_id'] for c in selected if c['rate_rps_decimal']!='4' and c['repeat']==2],original_r1_observations_retained=True,selection_rule='replace complete original Dynamo new-rate group irrespective of outcomes; add original Alp4 engineering repair both repeats; no higher-than-PDB-boundary points',deadline_s=None,campaign_lifecycle='until_declared_complete_v1',arrival_window_s=100,request_hard_timeout_s=120,drain_after_arrival_window_s=120,cleanup_local_budget_s=90,energy_gpu_indices=list(range(8)),failure_rule='any request/clock/native/measurement failure preserves checkpoint and cleanup, then halts; diagnosed continuation requires a new immutable declaration; no automatic replay',source_files={str(Path(__file__).resolve()):p.sha(__file__)})
 p.write(OUT/'declaration.json',decl,exclusive=True);print(json.dumps(dict(declaration=p.ref(OUT/'declaration.json'),original_alp4_quarantine=p.ref(qp),cells=len(rows),gpu_executed=False)))
if __name__=='__main__':main()
