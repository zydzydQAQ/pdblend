"""CPU raw audit of completed r2 cells; r1 carried Alpaca remains immutable."""
import csv,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_integral',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 status=read(ROOT/'status.pilots.json');assert status['complete'] and status['phase']=='finished' and status['baseline_preservation_verified']
 specs={r['cell_id']:r for r in read(ROOT/'runspec.json')['cells']};results=[]
 for r in status['cells']:
  cid=r['cell_id'];row=specs[cid];out=ROOT/'cells'/cid;op=ROOT/'operations'/cid;s=read(out/'summary.json');trace=read(Path(row['trace']));bench=list(csv.DictReader((out/'bench.csv').open()))
  assert r['screen_valid'] and r['child_exitcode']==0 and s['measurement_valid'] and s['work_complete'] and s['drain_complete'] and not s['incomplete_drain']
  assert s['post_measurement_cleanup']['cleanup_complete'] and r['controller_cleanup']['cleanup_complete'] and r['outer_cleanup']['complete'] and r['outer_cleanup']['clock_release_complete']
  cfg=read(out/'runtime_config.json');expected=read(ROOT/'inputs/controller.fixed.json');expected.update(journal=str(out/'control.jsonl'),slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s']);assert cfg==expected
  assert len(bench)==len(trace['requests'])==64
  byidx={int(t['idx']):t for t in trace['requests']};good=inputs=outputs=0
  for b in bench:
   t=byidx[int(b['idx'])];assert b['success']=='1' and int(b['input_tokens'])==t['prompt_len'] and int(b['generated_tokens'])==t['output_len']
   ok=float(b['ttft_s'])<=row['slo_ttft_s'] and float(b['tpot_s'])<=row['slo_tpot_s'];assert (b['slo_ok']=='1')==ok;good+=ok;inputs+=int(b['input_tokens']);outputs+=int(b['generated_tokens'])
   assert b['open_loop_independent'] in ('1','True','true') and b['request_timeout'] in ('0','False','false')
  assert good==s['good_requests'] and s['completed']==64 and inputs==s['input_tokens'] and outputs==s['output_tokens']
  gpu=m.integrate(m.power_rows(out/'power.csv'),s['measurement_start_s'],s['measurement_end_s']);assert len(gpu)==8 and abs(sum(gpu)-s['energy_j'])<1e-5
  ogpu=m.integrate(m.power_rows(op/'power/power.csv'),r['operation_start_s'],r['operation_end_s']);assert abs(sum(ogpu)-r['full_operation_energy_j'])<1e-5
  native={}
  for iid,x in r['outer_cleanup']['native'].items():
   before=x['before'];proof=x['barrier'];after=x['after'];assert x['complete'] and before['accepting'] is False
   assert proof['drained'] and proof['send_counters_verified'] and proof['generation']==before['generation']+1 and len(proof['transfers'])==2
   for rank in proof['transfers']:
    assert rank['send_counters_observed'] and rank['send_healthy'] and rank['send_started']==rank['send_completed'] and rank['send_failed']==0 and rank['listener_alive']
    assert not any(rank[k] for k in ['buffered_tensors','inflight_sends','inflight_receives','allocations','buffered_gpu_bytes'])
   assert after['accepting'] and after['admit_prefill'] and after['role']=='mixed' and after['mode']=='continuous' and after['scheduler_budget_effective']==dict(max_num_batched_tokens=8192,max_num_seqs=32)
   assert after['generation']==after['acknowledged_generation'] and all(x['controls']['runtime']['generation']==after['generation'] for x in after['scheduler_io'])
   events=[json.loads(e) for e in (op/(iid+'.events.jsonl')).read_text().splitlines()];assert all(e['mode']=='continuous' and e['role']=='mixed' and e['tokens']<=8192 for e in events)
   native[iid]=dict(barrier_generation=proof['generation'],restored_generation=after['generation'],rank_count=2,steps=len(events))
  dispatch=[float(b['actual_dispatch_s']) for b in bench];planned=[float(b['planned_arrival_s']) for b in bench]
  results.append(dict(cell_id=cid,valid=True,completed=64,good_requests=good,slo_attainment=good/64,input_tokens=inputs,output_tokens=outputs,energy_j=s['energy_j'],reintegrated_energy_j=sum(gpu),energy_difference_j=sum(gpu)-s['energy_j'],per_gpu_energy_j=gpu,energy_per_good_request_j=sum(gpu)/good,full_operation_energy_j=sum(ogpu),actual_dispatch_span_s=max(dispatch)-min(dispatch),planned_arrival_span_s=max(planned)-min(planned),native=native,clock_release_complete=True,source_sha256={str(p):sha(p) for p in [out/'summary.json',out/'bench.csv',out/'power.csv',out/'runtime_config.json',op/'outer-cleanup.json']}))
 result=dict(schema_version=1,all_valid=True,baseline_preservation_verified=True,original_temporal_gate_passed=False,carried_alpaca_audit_sha256=sha(ROOT/'inputs/carried-alpaca-audit.json'),status_sha256=sha(ROOT/'status.pilots.json'),cells=results)
 (ROOT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'}));print(json.dumps([{k:v for k,v in r.items() if k not in ('source_sha256','native','per_gpu_energy_j')} for r in results]))
if __name__=='__main__':main()
