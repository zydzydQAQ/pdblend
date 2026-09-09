"""Independent actual overlay mapping and repeat policy validation on B CPU."""
import copy,hashlib,json,os,socket,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent
sys.path[:0]=[str(R),str(R.parent/'main-slo-improvement-v7')]
import protocol as p
import final_baseline_overlay_v2 as overlay
from baseline_row_metadata_v1 import metadata
inputs=p.read(B/'root-eco-overlay-review-inputs-v1.json')
for path,digest in inputs['files'].items():p.need(p.sha(path)==digest,'review source/input differs '+path)
originals=p.original_points();before=copy.deepcopy(originals)
result=overlay.validate(p,dict(baseline_overlay=inputs['overlay']),originals)
p.need(originals==before,'overlay altered original raw points')
p.need((len(result['quarantine']),len(result['retired']),len(result['fresh']))==(21,108,111),'whole group counts changed')
fresh=result['fresh'];a=[v for v in fresh.values() if v['row']['model']=='14b'];p.need(len(a)==31 and all(v['execution_scope_pending_final_pdb'] for v in a),'A logical pending scope changed')
for model,dataset,rate in [('14b','alpaca',12.),('32b','longbench',.25),('7b','alpaca',12.)]:
 rows=[v for v in fresh.values() if v['row']['model']==model and v['row']['system']=='ecoserve' and v['row']['dataset']==dataset and v['row']['rate_rps']==rate]
 p.need(len(rows)==2 and {v['row']['repeat'] for v in rows}=={1,2} and all(v['reuse_original_rate_first_repeat'] is False for v in rows),'dedicated R2 must never fall back to R1')
for v in fresh.values():
 row=v['row']
 if row['system']=='ecoserve' and v['existing_original_rate'] and row['repeat']==1 and (row['model'],row['dataset'],row['rate_rps']) not in {('14b','alpaca',12.),('32b','longbench',.25),('7b','alpaca',12.)}:
  p.need(v['reuse_original_rate_first_repeat'] is True,'allowed original R1 pairing scope changed')
p.need(not any(v['row']['model']=='32b' and v['row']['system']=='ecoserve' and (v['row']['dataset'],v['row']['rate_rps']) in {('sharegpt',2.),('longbench',1.)} for v in fresh.values()),'B existing above-boundary exclusion omitted')
for cid in result['quarantine']:
 point=dict(cell_id=cid,measurement_valid=True,status='completed',energy_j=1.)
 overlay.annotate(point,result);p.need(point['raw_arithmetic_verified'] and not point['scientific_comparison_eligible'] and not point['selected_for_final_comparison'] and point['energy_j']==1.,'quarantine erases raw or enters win/loss')
ordinary=dict(cell_id='other-valid-capacity-negative',measurement_valid=True,work_complete=False,slo_attainment=.2)
overlay.annotate(ordinary,result);p.need(overlay.eligible(ordinary),'normal capacity-negative observation spuriously excluded')
q=dict(passed=True,created_s=time.time(),physical_host=socket.gethostname(),pid=os.getpid(),source=p.ref(Path(__file__)),inputs=p.ref(B/'root-eco-overlay-review-inputs-v1.json'),quarantine_count=21,retired_count=108,fresh_count=111,source_count=len(result['sources']),a31_pending=True,explicit_R2_has_no_R1_fallback=True,normal_original_R1_reuse_retained=True,b_only_preexisting_above_boundary_exclusions=True,original450_unchanged=True,raw_energy_preserved=True,normal_capacity_negative_still_eligible=True,necessary_fix='Root already added A/B helper hashes to final collector code_sources during this review; no additional block found in four reviewed mapping/overlay files',limitations=['Mapping and eligibility source review only, not independent audit of every new GPU request; actual observation auditor remains separately required.'])
p.write(B/'independent-root-eco-overlay-v2-review.json',q,exclusive=True);print(json.dumps(q))
