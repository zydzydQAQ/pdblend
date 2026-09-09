"""CPU evidence reconciliation for preserved baseline capacity negatives only."""
import csv,json,math,sys
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'execution-completion-p4-002'))
import runner as fixed
p=fixed.p

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def integral(path,start,end):
 rows=[]
 for r in csv.DictReader(Path(path).open()):
  t=float(r['t_s']);watts=[float(r[f'gpu{g}_w']) for g in range(8)]
  p.need(math.isfinite(t) and all(math.isfinite(w) and w>=0 for w in watts),'invalid actual eight-GPU power')
  rows.append((t,sum(watts)))
 p.need(rows and rows[0][0]<=start<end<=rows[-1][0] and all(a[0]<b[0] for a,b in zip(rows,rows[1:])),'whole energy window is not continuously bracketed')
 total=0.
 for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
  a,b=max(start,t0),min(end,t1)
  if b>a:
   pa=p0+(p1-p0)*(a-t0)/(t1-t0);pb=p0+(p1-p0)*(b-t0)/(t1-t0);total+=(pa+pb)*(b-a)/2
 return total

def capacity_negative(cp,receipt):
 s=receipt['summary'];gate=fixed.engineering_gate(receipt)
 p.need(set(gate['errors'])=={'request failure','request timeout','incomplete prescribed work'},'failure includes an engineering or non-timeout fault')
 p.need(s['failed_requests']==s['request_timeouts']>0 and s['work_complete'] is False and s['measurement_valid'] is True and s['gpu_count']==8 and s['fixed_window_valid'] is True and s['drain_complete'] is True and s['post_measurement_cleanup']['cleanup_complete'] is True,'invalid capacity-negative measurement/cleanup')
 p.need(all(x['complete'] is True and not x['errors'] for x in receipt['restoration'].values()),'native restoration incomplete')
 paths=[k for k in cp['artifacts'] if Path(k).name=='bench.csv' and Path(k).parent.name==receipt['cell_id']];p.need(len(paths)==1,'actual request work required');rows=list(csv.DictReader(Path(paths[0]).open()));bad=[r for r in rows if r['success']!='1']
 p.need(len(rows)==s['n_expected'] and len(bad)==s['failed_requests'] and len(rows)-len(bad)==s['completed_work_requests'],'capacity-negative denominator differs')
 p.need(all(r['request_timeout']=='True' and r['error'] in ('request_hard_timeout','request_hard_timeout_before_dispatch') and r['slo_ok']=='0' and r['token_ids_verified']=='0' and math.isclose(float(r['request_deadline_s'])-float(r['planned_arrival_s']),120,abs_tol=1e-7) for r in bad),'failure is not the original hard request deadline')
 cell=Path(paths[0]).parent;events=[json.loads(l) for l in (cell/'control.jsonl').open()];timing={r['client_request_id']:r for r in events if r['kind']=='request_timing'}
 p.need(all(r['request_id'] in timing and timing[r['request_id']]['completed'] is False for r in bad),'failed requests were not preserved as native incomplete')
 energy=integral(cell/'power.csv',s['measurement_start_s'],s['measurement_end_s']);p.need(math.isclose(energy,s['energy_j'],rel_tol=1e-9,abs_tol=1e-7),'eight-GPU energy differs from retained raw')
 return dict(passed=True,classification='frozen-baseline-capacity-negative-at-original-120s',complete_requests=s['completed_work_requests'],n_expected=s['n_expected'],failed_requests=s['failed_requests'],request_timeouts=s['request_timeouts'],energy_j=energy,slo_attainment=s['slo_attainment'],power_includes_failed_partial_work=True,failed_work_is_not_valid_full_work_comparison=True,failed_rows=[{k:r[k] for k in ('request_id','error','output_len','generated_tokens','token_count_source','token_ids_verified','n_text_chunks','ttft_s','request_deadline_s')} for r in bad])

def audit(path):
 d=p.read(path);p.need(d['schema']=='B-baseline-original-repeats-capacity-reconciliation-v1' and d['deadline_s'] is None and d['campaign_lifecycle']=='until_declared_complete_v1','wrong reconciliation lifecycle')
 for f,h in d['source_files'].items():p.need(p.sha(f)==h,'reconciliation source changed')
 parent=p.checked(d['parent_declaration']);allrows={c['cell_id']:c for c in parent['cells']}
 observed={};negatives={r['cell_id']:r for r in d['capacity_negatives']};quarantines={r['cell_id']:r for r in d.get('engineering_quarantines',[])};actualnegative=set()
 for entry in d['prior_stages']:
  state=p.checked(entry['status']);root=Path(entry['status']['path']).parent
  p.need(state.get('finished_s') and not state.get('node_lease_held') and not alive(state['pid']),'previous queue still live or unclean')
  p.need(state['attempted']==state['completed'],'observed outcome missing its checkpoint')
  statebad={x['cell_id'] for x in state['failed']};actualnegative.update(statebad)
  for cid in state['attempted']:
   p.need(cid not in observed and cid in allrows,'duplicate/unpredeclared original observation')
   cp=p.read(root/'results/checkpoints'/(cid+'.json'));p.need(cp['row']==allrows[cid],'original trace/SLO/source row changed')
   p.need(all(p.sha(f)==h for f,h in cp['artifacts'].items()),'previous raw observation changed');p.checked(cp['binding']);receipt=p.checked(cp['receipt'])
   p.need(receipt['cell_id']==cid and receipt['trace_sha256']==allrows[cid]['trace_sha256'],'receipt identity differs')
   if cid in statebad and cid in quarantines:
    item=quarantines[cid];proof=p.checked(item['quarantine']);node=p.checked(item['fresh_node_audit']);p.need(proof['checkpoint']==p.ref(root/'results/checkpoints'/(cid+'.json')) and proof['receipt']==cp['receipt'] and proof['binding']==cp['binding'] and proof['cell_id']==cid,'quarantine refers to another observation')
    p.need(proof['scientific_comparison_eligible'] is False and proof['raw_energy_measurement_valid'] is True and proof['work_complete'] is False and proof['actual_dispatch_lateness_max_s']==receipt['summary']['dispatch_delay_max_s'] and proof['actual_dispatch_lateness_max_s']>100,'specific engineering quarantine evidence differs')
    p.need(all(p.sha(f)==h for f,h in proof['raw_timing_evidence'].items()) and all(p.sha(f)==h for f,h in proof['actual_source'].items()),'quarantined source/raw changed')
    p.need(node['passed'] is True and receipt['child_stopped'] and receipt['clock_restore_complete'] and not receipt['outer_cleanup_errors'] and all(v['complete'] and not v['errors'] for v in receipt['restoration'].values()),'quarantine final native/clock/identity cleanup incomplete')
    cell=Path(cp['receipt']['path']).parents[2]/'cells'/cid;s=receipt['summary'];p.need(math.isclose(integral(cell/'power.csv',s['measurement_start_s'],s['measurement_end_s']),proof['raw_energy_recomputed_j'],rel_tol=1e-9),'quarantine raw energy differs')
   elif cid in statebad:
    p.need(cid in negatives,'unreconciled request failure');raw=capacity_negative(cp,receipt);proof=p.checked(negatives[cid]['diagnosis']);p.need(raw==proof['raw_audit'],'capacity diagnosis differs from actual raw');node=p.checked(proof['fresh_node_audit']);p.need(node['passed'] is True,'fresh native/identity proof missing')
   else:p.need(fixed.engineering_gate(receipt)['passed'],'other baseline cell has engineering/request failure')
   observed[cid]=cp
 p.need(actualnegative==set(negatives)|set(quarantines),'capacity-negative set differs')
 selected={w['workload_id'] for w in parent['workloads'] if not w.get('existing_original_rate') and all(c['cell_id'] in p.read(B/'fixed-screen-p4-001/status.json')['completed'] for c in parent['cells'] if c['workload_id']==w['workload_id'] and c['system']=='pdblend')}
 expected=[c for c in parent['cells'] if c['workload_id'] in selected and c['system']!='pdblend'];expected.sort(key=lambda c:(c['repeat'],c['sequence']))
 retired=d.get('retired_unexecuted_cells',[]);retired_ids={c['cell_id'] for c in retired}
 if retired:
  p.need(retired==[c for c in expected if c['system']=='dynamollm' and c['repeat']==2] and not retired_ids.intersection(observed),'only the entire originally unexecuted Dynamo repeat2 group may retire')
  p.need(d['replacement_dynamo_original_rows']==[c for c in expected if c['system']=='dynamollm'] and d['superseded_dynamo_observation_ids']==[c['cell_id'] for c in expected if c['system']=='dynamollm' and c['repeat']==1],'all three-rate Dynamo repeats must be replaced as one group')
 p.need(d['remaining_cells']==[c for c in expected if c['cell_id'] not in observed and c['cell_id'] not in retired_ids],'remaining original repeats changed, omitted or replayed')
 p.need(len(expected)==d['original_baseline_count']==24 and len(observed)==d['already_observed_count'],'wrong original scope count')
 return d
