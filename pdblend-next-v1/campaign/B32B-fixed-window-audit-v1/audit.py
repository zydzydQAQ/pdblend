"""Independent raw-work, fixed-window and eight-GPU integration audit."""
import argparse,csv,hashlib,importlib.util,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'B32B-deadline-24h-v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def yes(x):return x in (True,1,'1','true','True')
def require(v,m):
 if not v:raise ValueError(m)
spec=importlib.util.spec_from_file_location('integration',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py')
integ=importlib.util.module_from_spec(spec);spec.loader.exec_module(integ)
def audit(cid):
 row=next(x for x in read(BASE/'runspec.json')['cells'] if x['cell_id']==cid)
 out=BASE/'cells'/cid;op=BASE/'operations'/cid;s=read(out/'summary.json');receipt=read(BASE/'receipts'/(cid+'.json'))
 require(receipt['screen_valid'] and receipt['outer_cleanup']['complete'],'wrapper not fully valid')
 require(s['measurement_valid'] and s['fixed_window_valid'] and s['measurement_schema']==3
  and s['measurement_window_protocol']=='per-dataset-slo-fixed-window-v2','fixed-window schema invalid')
 epoch=read(op/'actual_epoch_gate.json');w=s['fixed_window'];limits=receipt['limits']
 require(epoch['accepted'] and epoch['before_any_request_worker'] and epoch['actual_epoch_s']==w['arrival_epoch_s']
  and epoch['actual_epoch_s']<=limits['latest_arrival_epoch_s'],'actual epoch differs')
 require(w['arrival_window_s']==300 and w['arrival_window_end_s']==epoch['actual_epoch_s']+300
  and w['window_observed_complete'] and w['clock_consistent'],'idle suffix not observed')
 require(abs(s['measurement_start_s']-epoch['actual_epoch_s'])<1e-5 and s['measurement_end_s']>=w['arrival_window_end_s']
  and s['measurement_end_s']<=limits['cell_execution_deadline_s'] and receipt['finished_s']<=limits['restore_deadline_s'],'actual limits/window differ')
 cfg=read(out/'runtime_config.json');expected=read(BASE/'inputs/controller.fixed.json')
 expected.update(journal=str(out/'control.jsonl'),slo_protocol='per-dataset-slo-v1',slo_attainment_target=.9,
  **{k:row[k] for k in ('slo_scale','slo_ttft_s','slo_tpot_s')})
 require(cfg==expected,'actual policy or SLO changed')
 trace=read(row['trace']);require(sha(row['trace'])==row['trace_sha256']==s['trace_sha256'],'trace changed')
 records=list(csv.DictReader((out/'bench.csv').open()));require(len(records)==row['n_requests']==s['n_expected'],'denominator differs')
 require({int(x['idx']) for x in records}==set(range(row['n_requests'])),'idx set differs')
 good=complete=generated=inputs=0;failures=[];dispatch=[];misses=[]
 for r in records:
  idx=int(r['idx']);req=trace['requests'][idx]
  require(int(r['prompt_len'])==req['prompt_len'] and int(r['output_len'])==req['output_len'],'offered work changed')
  require(abs(float(r['planned_arrival_s'])-epoch['actual_epoch_s']-req['arrival_s'])<1e-5 and yes(r['open_loop_independent']),'arrival protocol changed')
  require(abs(float(r['actual_dispatch_s'])-float(r['planned_arrival_s'])-float(r['dispatch_delay_s']))<1e-5,'actual dispatch provenance differs')
  full=yes(r['success']) and int(r['generated_tokens'])==req['output_len']
  ok=full and math.isfinite(float(r['ttft_s'])) and math.isfinite(float(r['tpot_s'])) and float(r['ttft_s'])<=row['slo_ttft_s'] and float(r['tpot_s'])<=row['slo_tpot_s']
  require(yes(r['slo_ok'])==ok,'joint SLO score differs')
  if full:require(int(r['input_tokens'])==req['prompt_len'] and yes(r['token_ids_verified']),'completed work/token identity differs')
  else:failures.append(dict(idx=idx,error=r['error'],generated=int(r['generated_tokens']),expected=req['output_len']))
  if not ok:misses.append(dict(idx=idx,ttft_s=float(r['ttft_s']),tpot_s=float(r['tpot_s']),complete=full))
  complete+=full;good+=ok;generated+=int(r['generated_tokens']);inputs+=int(r['input_tokens']);dispatch.append(float(r['actual_dispatch_s']))
 require(complete==s['completed_work_requests'] and good==s['good_requests'] and generated==s['generated_tokens'],'summary counts differ')
 powers=integ.power_rows(out/'power.csv');gpu=integ.integrate(powers,s['measurement_start_s'],s['measurement_end_s'])
 require(abs(sum(gpu)-s['energy_j'])<1e-5,'main all8 energy differs')
 outer=integ.integrate(integ.power_rows(op/'power/power.csv'),receipt['operation_start_s'],receipt['operation_end_s'])
 require(abs(sum(outer)-receipt['full_operation_energy_j'])<1e-5,'outer all8 energy differs')
 events=[];native={}
 for iid,n in receipt['outer_cleanup']['native'].items():
  proof=n['barrier'];after=n['after'];require(n['complete'] and proof['send_counters_verified'] and len(proof['transfers'])==2,'native rank proof missing')
  for rank in proof['transfers']:
   require(rank['send_counters_observed'] and rank['send_healthy'] and rank['send_started']==rank['send_completed'] and rank['send_failed']==0
    and not any(rank[k] for k in ('buffered_tensors','inflight_sends','inflight_receives','allocations','buffered_gpu_bytes')),'rank residue')
  require(after['accepting'] and after['role']=='mixed' and after['mode']=='continuous' and after['scheduler_budget_effective']==dict(max_num_batched_tokens=8192,max_num_seqs=32)
   and after['generation']==after['acknowledged_generation'] and after['scheduler_budget_pending'] is None,'owner not restored')
  require(len(after['scheduler_io'])==1 and after['scheduler_io'][0]['controls']['runtime']['generation']==after['generation'],'cache ACK differs')
  es=[json.loads(line) for line in (op/(iid+'.events.jsonl')).read_text().splitlines() if line]
  events+=es;native[iid]=dict(rank_proofs=2,owner_cache_proofs=1,restored_generation=after['generation'])
 require(all(e['mode']=='continuous' and e['role']=='mixed' and e['tokens']<=8192 for e in events),'forbidden owner execution')
 require(receipt['outer_cleanup']['clock_release_complete'],'clocks not released')
 suffix_start=max(max(dispatch),max(float(r['planned_arrival_s']) for r in records))
 suffix_gpu=integ.integrate(powers,suffix_start,w['arrival_window_end_s']) if suffix_start<w['arrival_window_end_s'] else None
 return dict(schema=1,cell_id=cid,measurement_valid=True,n_expected=len(records),completed_work_requests=complete,
  good_requests=good,slo_attainment=good/len(records),work_complete=complete==len(records),failures=failures,slo_misses=misses,
  energy_j=sum(gpu),energy_difference_j=sum(gpu)-s['energy_j'],per_gpu_energy_j=gpu,energy_per_good_request_j=sum(gpu)/good if good else None,
  whole_operation_energy_j=sum(outer),actual_arrival_epoch_s=epoch['actual_epoch_s'],arrival_window_s=300,
  measurement_duration_s=s['measurement_end_s']-s['measurement_start_s'],actual_dispatch_span_s=max(dispatch)-min(dispatch),
  last_arrival_to_window_end_j=sum(suffix_gpu) if suffix_gpu else 0,
  generated_tokens=generated,input_tokens=inputs,ttft_avg_s=s['ttft_avg_s'],ttft_p99_s=s['ttft_p99_s'],tpot_avg_s=s['tpot_avg_s'],tpot_p99_s=s['tpot_p99_s'],
  goodput_rps=s['goodput_rps'],goodput_fixed_window_rps=good/300,max_power_gap_s=max(y[0]-x[0] for x,y in zip(powers,powers[1:])),
  native=native,clock_release_complete=True,max_actual_decode_batch=max((e['decode'] for e in events),default=0),
  original_temporal_gate_passed=False,profile_candidate='explicit decode-phase composition',formal_eligible=False,
  source_sha256={str(x):sha(x) for x in [BASE/'package-manifest.json',BASE/'freeze.json',Path(row['trace']),out/'summary.json',out/'bench.csv',out/'power.csv',op/'power/power.csv',BASE/'receipts'/(cid+'.json')]})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',default='32b-alpaca-r1-s701-w300-scale1');a=p.parse_args();r=audit(a.cell)
 out=ROOT/(a.cell+'.json');require(not out.exists(),'audit already exists');out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','native','per_gpu_energy_j','failures','slo_misses')}))
