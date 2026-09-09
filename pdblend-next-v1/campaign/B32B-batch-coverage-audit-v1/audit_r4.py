"""Read-only raw audit and bounded descriptive batch evidence; no ProfilePoint export."""
import csv
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'B32B-batch-coverage-observation-v2-r4'
spec=importlib.util.spec_from_file_location('independent_integral',ROOT.parent/'B32B-budget-paired-longbench-v1/audit.py')
integ=importlib.util.module_from_spec(spec);spec.loader.exec_module(integ)
def read(p):return json.loads(p.read_text())
def digest(tokens):return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def diff(a,b):
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return dict(output_position_1based=i+1,reference_token=x,observed_token=y)
 return None if len(a)==len(b) else dict(output_position_1based=min(len(a),len(b))+1,lengths=[len(a),len(b)])
def main():
 status=read(BASE/'status.json');require(status.get('complete'),'outer operation not terminal')
 rows=[];reference=None;token_rows=[];all_ids=set()
 report=read(BASE/'results/campaign.json')
 for point in report['points']:
  path=Path(point['path']);raw=read(path/'raw.json');batch=raw['spec']['batch_size'];freq=raw['spec']['clock_command_mhz']
  if raw.get('error'):
   rows.append(dict(point=path.name,batch=batch,clock_command_mhz=freq,valid=False,error=raw['error']));continue
  require(len(raw['requests'])==batch,'natural batch incomplete')
  for request in raw['requests']:
   require(request['success'] and len(request['prompt_token_ids'])==512 and len(request['output_token_ids'])==256,'actual whole batch work differs')
   rid=request['request_id'];require(rid not in all_ids,'request ID reused');all_ids.add(rid)
   tokens=request['output_token_ids'];reference=tokens if reference is None else reference
   token_rows.append(dict(point=path.name,batch=batch,clock_command_mhz=freq,request_id=rid,token_sha256=digest(tokens),first_difference_from_first_observed=diff(reference,tokens)))
  warmup=raw['warmup'];require(warmup['success'] and len(warmup['output_token_ids'])==64,'warmup incomplete');all_ids.add(warmup['request_id'])
  powers=integ.power_rows(path/'power.csv');gpu=integ.integrate(powers,raw['measurement_start_s'],raw['measurement_end_s'])
  profile=read(path/'profile.json');expected=profile['energy_all_eight_gpus_j'];require(abs(sum(gpu)-expected)<1e-5,'unfiltered all8 nested energy differs')
  events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
  ids={r['request_id'] for r in raw['requests']}
  require(all(e['mode']=='continuous' and e['role']=='mixed' and e['tokens']<=8192 for e in events),'forbidden scheduler execution')
  decode=[e for e in events if e.get('prefill')==0 and e.get('decode')==batch and set(e.get('request_ids',[]))==ids]
  require(len(decode)>=64,'full natural decode batch not sustained')
  durations=[e['finished_s']-e['started_s'] for e in decode]
  clock_samples=read(path/'clocks.json');sm=[[],[]]
  for t,frequencies in clock_samples:
   if raw['measurement_start_s']<=t<=raw['measurement_end_s']:
    for g in (0,1):sm[g].append(frequencies[g])
  rows.append(dict(point=path.name,valid=True,batch=batch,clock_command_mhz=freq,actual_gpu01_sm_median_mhz=[statistics.median(x) for x in sm],requests=batch,generated_tokens=batch*256,whole_batch_energy_j=sum(gpu),per_gpu_energy_j=gpu,energy_difference_j=sum(gpu)-expected,energy_per_output_token_j=sum(gpu)/(batch*256),batch_window_s=raw['measurement_end_s']-raw['measurement_start_s'],full_batch_decode_steps=len(decode),owner_decode_step_median_s=statistics.median(durations),owner_decode_step_mean_s=statistics.mean(durations),owner_decode_step_max_s=max(durations)))
 require(all_ids=={rid for port,rid in status.get('owned_request_ids',[])} if report.get('complete') else True,'durable own-request journal differs from completed raw')
 outer=integ.integrate(integ.power_rows(BASE/'power/power.csv'),status['measurement_start_s'],status['measurement_end_s'])
 require(abs(sum(outer)-status['total_node_energy_j'])<1e-5,'whole operation all8 integral differs')
 groups=[]
 for batch in (4,8):
  for freq in (2520,1500):
   selected=[r for r in rows if r.get('valid') and r['batch']==batch and r['clock_command_mhz']==freq]
   groups.append(dict(batch=batch,clock_command_mhz=freq,n=len(selected),mean_whole_batch_energy_j=statistics.mean(r['whole_batch_energy_j'] for r in selected) if selected else None,mean_owner_decode_step_s=statistics.mean(r['owner_decode_step_mean_s'] for r in selected) if selected else None))
 result=dict(schema_version=1,outer_passed=status['passed'],outer_measurement_valid=status['measurement_valid'],points=rows,groups=groups,complete_points=sum(r.get('valid',False) for r in rows),owned_request_count=len(all_ids),unique_output_token_sequences=len({r['token_sha256'] for r in token_rows}),outputs=token_rows,operation_energy_j=sum(outer),operation_per_gpu_energy_j=outer,operation_energy_difference_j=sum(outer)-status['total_node_energy_j'],cleanup_complete=status.get('cleanup_complete'),clock_release_complete=status.get('clock_release_complete'),baseline_preservation_verified=status.get('baseline_preservation_verified'),profile_exported=False,kv_correctness_certified=False,original_temporal_gate_passed=False,interpretation='Natural batch and full-work observations only. Owner CPU wall steps are not isolated kernel time; nested batch energy overlaps whole-operation energy. No baseline is executed and no source/profile is changed.',source_sha256={str(p):sha(p) for p in [BASE/'manifest.json',BASE/'status.json',BASE/'results/campaign.json',BASE/'power/power.csv']})
 output=ROOT/'audit-r4.json';require(not output.exists(),'prior audit must not be overwritten');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('points','outputs','source_sha256','operation_per_gpu_energy_j')}))
if __name__=='__main__':main()
