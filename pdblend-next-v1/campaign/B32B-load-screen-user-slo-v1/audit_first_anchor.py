"""CPU audit preserves failed wrapper status while validating the completed raw Alpaca cell."""
import csv,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_integral',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 status=json.loads((ROOT/'status.pilots.json').read_text());r=status['cells'][0];row=json.loads((ROOT/'runspec.json').read_text())['cells'][0]
 out=ROOT/'cells'/r['cell_id'];op=ROOT/'operations'/r['cell_id'];summary=json.loads((out/'summary.json').read_text());trace=json.loads(Path(row['trace']).read_text())
 assert status['phase']=='failed' and r['screen_valid'] is False and r['error']=="RuntimeError('owner did not become idle/ACKed')"
 assert r['child_exitcode']==0 and summary['measurement_valid'] and summary['work_complete'] and summary['drain_complete'] and not summary['incomplete_drain']
 assert summary['post_measurement_cleanup']['cleanup_complete'] and r['controller_cleanup']['cleanup_complete'] and r['outer_cleanup']['complete']
 cfg=json.loads((out/'runtime_config.json').read_text());expected=json.loads((ROOT/'inputs/controller.fixed.json').read_text());expected.update(journal=str(out/'control.jsonl'),slo_ttft_s=1.,slo_tpot_s=.1);assert cfg==expected
 bench=list(csv.DictReader((out/'bench.csv').open()));assert len(bench)==64
 good=0;inputs=outputs=0
 for b,t in zip(bench,trace['requests']):
  assert int(b['idx'])==t['idx'] and b['success']=='1' and int(b['input_tokens'])==t['prompt_len'] and int(b['generated_tokens'])==t['output_len']
  ok=float(b['ttft_s'])<=1 and float(b['tpot_s'])<=.1;assert (b['slo_ok']=='1')==ok;good+=ok;inputs+=int(b['input_tokens']);outputs+=int(b['generated_tokens'])
 assert good==summary['good_requests']==51 and summary['completed']==64
 power=m.power_rows(out/'power.csv');gpu=m.integrate(power,summary['measurement_start_s'],summary['measurement_end_s']);assert abs(sum(gpu)-summary['energy_j'])<1e-5
 outer=m.power_rows(op/'power/power.csv');outergpu=m.integrate(outer,r['operation_start_s'],r['operation_end_s']);assert abs(sum(outergpu)-r['full_operation_energy_j'])<1e-5
 native={}
 for iid,x in r['outer_cleanup']['native'].items():
  before=x['before'];proof=x['barrier'];after=x['after']
  assert x['complete'] and before['accepting'] is False and before['admit_prefill'] is False and all(not before[k] for k in ['active','running','waiting','kv_allocations'])
  assert proof['drained'] and proof['send_counters_verified'] and proof['drain_proof_type']=='synchronous_put_owner_barrier' and proof['generation']==before['generation']+1 and len(proof['transfers'])==2
  for rank in proof['transfers']:
   assert rank['send_counters_observed'] and rank['send_healthy'] and rank['send_started']==rank['send_completed'] and rank['send_failed']==0 and rank['listener_alive']
   assert not any(rank[k] for k in ['buffered_tensors','inflight_sends','inflight_receives','allocations','buffered_gpu_bytes'])
  assert after['accepting'] and after['admit_prefill'] and after['role']=='mixed' and after['mode']=='continuous' and after['scheduler_budget_effective']==dict(max_num_batched_tokens=8192,max_num_seqs=32)
  assert after['generation']==after['acknowledged_generation'] and all(x['controls']['runtime']['generation']==after['generation'] for x in after['scheduler_io'])
  native[iid]=dict(paused_generation=before['generation'],barrier_generation=proof['generation'],restored_generation=after['generation'],rank_count=2)
  events=[json.loads(s) for s in (op/(iid+'.events.jsonl')).read_text().splitlines()];assert all(x['mode']=='continuous' and x['role']=='mixed' and x['tokens']<=8192 for x in events)
 paths=[ROOT/'status.pilots.json',ROOT/'freeze.json',ROOT/'package-manifest.json',out/'summary.json',out/'bench.csv',out/'power.csv',out/'runtime_config.json',op/'outer-cleanup.json']
 result=dict(schema_version=1,original_wrapper_failed=True,original_status_modified=False,observed_cell_valid=True,work_complete=True,completed=64,n_expected=64,good_requests=good,slo_attainment=good/64,input_tokens=inputs,output_tokens=outputs,
  energy_j=sum(gpu),per_gpu_energy_j=gpu,energy_per_good_request_j=sum(gpu)/good,whole_operation_energy_j=sum(outergpu),native=native,
  original_error='post-quiesce check wrongly required accepting=True',real_paused_after_quiesce_valid=True,clock_release_complete=r['outer_cleanup']['clock_release_complete'],
  source_sha256={str(p):sha(p) for p in paths},baseline_preservation_verified=status['baseline_preservation_verified'],
  interpretation='Valid original measured Alpaca anchor can be referenced by continuation; SLO79.6875% retained. No replay, threshold change or old status rewrite.')
 (ROOT/'first-anchor-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','per_gpu_energy_j','native']}))
if __name__=='__main__':main()
