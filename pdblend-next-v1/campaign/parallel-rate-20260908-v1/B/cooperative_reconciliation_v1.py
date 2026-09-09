"""Reconcile original cooperative group capacity negatives without replay or timing waiver."""
from pathlib import Path
import sys
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import cooperative_dynamo_runner_v2 as original
import baseline_reconciliation_v3 as legacy
p=original.p

def audit(path):
 d=p.read(path);rows,_=original.contract();expected=rows['cells']
 p.need(d['schema']=='B-cooperative-eight-capacity-reconciliation-v1' and d['declaration']==p.ref(original.DECL) and d['execution_rules']==p.ref(original.RULES),'same original cooperative group and arrival timing qualification required')
 p.need(d['deadline_s'] is None and d['automatic_retries'] is False,'original no-total-deadline and no retries required')
 for source,digest in d['source_files'].items():p.need(p.sha(source)==digest,'frozen reconciliation source changed')
 allrows={r['cell_id']:r for r in expected};observed={};actual_bad=set();diagnoses={x['cell_id']:x for x in d['capacity_negatives']}
 for entry in d['prior_stages']:
  status=p.checked(entry['status']);root=Path(entry['status']['path']).parent
  p.need(status['finished_s'] and not status.get('node_lease_held') and not legacy.alive(status['pid']),'previous cooperative owner not terminal/clean')
  p.need(status['attempted']==status['completed'],'every actual attempt must have a preserved CP')
  bad={x['cell_id'] for x in status['failed']};actual_bad.update(bad)
  for cid in status['attempted']:
   p.need(cid in allrows and cid not in observed,'cooperative duplicate/unpredeclared observation')
   cpref=p.ref(root/'results/checkpoints'/(cid+'.json'));cp=p.checked(cpref)
   p.need(cp['row']==allrows[cid] and cp['declaration']==p.ref(original.DECL) and cp['execution_rules']==p.ref(original.RULES),'cooperative row or original timing rule differs')
   p.need(all(p.sha(f)==h for f,h in cp['artifacts'].items()),'cooperative raw observation changed');binding=p.checked(cp['binding']);receipt=p.checked(cp['receipt'])
   base=p.checked(p.ref(original.qualification.OUT/'dynamollm/binding.json'))
   p.need(binding['host_release']==base['host_release'] and binding['instances']==base['instances'] and binding['configs']==base['configs'],'cooperative actual source/identity/policy differs')
   p.need(receipt['cell_id']==cid and receipt['trace_sha256']==allrows[cid]['trace_sha256'],'cooperative receipt belongs to another trace or point')
   timing=original.timing_gate(receipt,root/'results');p.need(timing==cp['arrival_fidelity_gate'] and timing['passed'],'arrival engineering failure is never capacity-negative evidence')
   if cid in bad:
    p.need(cid in diagnoses,'failed cooperative observation lacks independent diagnosis')
    proof=p.checked(diagnoses[cid]['diagnosis']);raw=legacy.capacity_negative(cp,receipt)
    p.need(proof['checkpoint']==cpref and proof['raw_audit']==raw and proof['arrival_fidelity_gate']==timing,'cooperative capacity diagnosis differs from raw')
    fresh=p.checked(proof['fresh_node_audit']);p.need(fresh['passed'] and fresh['cooperative_binding']==p.ref(original.qualification.OUT/'dynamollm/binding.json'),'fresh same cooperative identity/qualification required')
   else:p.need(original.c.fixed.engineering_gate(receipt)['passed'],'unreconciled cooperative measurement failure')
   observed[cid]=cpref
 p.need(actual_bad==set(diagnoses),'cooperative capacity-negative set differs')
 p.need(d['remaining_cells']==[row for row in expected if row['cell_id'] not in observed],'remaining eight-row group omitted, changed or replayed')
 p.need(d['already_observed_count']==len(observed) and d['original_group_count']==8,'cooperative exact group count differs')
 return d
