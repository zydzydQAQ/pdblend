"""P8 actual autonomous gates; P6 numerical provenance remains explicitly P6."""
from pathlib import Path
import json,math,sys,importlib.util
A=Path(__file__).resolve().parent;R=A.parent;CODE=A/'load-p8-code-001'
sys.path[:0]=[str(CODE),str(A),str(R),'/root/workspace/pdblend/.runtime-deps']
import continue_p6_qualification_v3 as prior
_s=importlib.util.spec_from_file_location('p8_actual_gate_source',CODE/'capacity_load_calibrate.py')
_gate=importlib.util.module_from_spec(_s);_s.loader.exec_module(_gate)
autonomous_gate_evidence,validate_trace=_gate.autonomous_gate_evidence,_gate.validate_trace
_cs=importlib.util.spec_from_file_location('p8_explicit_actual_compatibility',CODE/'calibration_compatibility.py')
_cm=importlib.util.module_from_spec(_cs);_cs.loader.exec_module(_cm)
validate_compatibility=_cm.validate_compatibility
require,fixed,ref,sha,close=prior.require,prior.fixed,prior.ref,prior.sha,prior.close

def saved_identity(rows, instances, started_s, finished_s, *, restored=False):
 require(isinstance(rows,list) and len(rows)==len(instances),'all original identities must be observed')
 by_id={r['runtime']['id']:r for r in rows}
 require(len(by_id)==len(instances) and set(by_id)=={i['id'] for i in instances},'saved identity owner differs')
 policy_path=R/'hosts/14b-capacity-p8/src/ecopadg/serving/completion_policy.py'
 module=_gate.load(policy_path,'p8_saved_native_completion_policy')
 for instance in instances:
  row=by_id[instance['id']];actual=row['container'];container=instance['container'];native=row['runtime']
  require(actual['Id']==container['id'] and actual['Image']==container['image'] and actual['Name'].lstrip('/')==container['name'] and actual['State']['StartedAt']==container['StartedAt'] and actual['State']['Pid']==instance['host_pid'] and actual['State']['Running'] is True,'saved Docker identity differs')
  require(all(row['provenance'].get(k)==v for k,v in instance['provenance'].items()),'saved engine source/provenance differs')
  stamp=native.get('timestamp')
  require(type(stamp) in (int,float) and math.isfinite(stamp) and started_s<=stamp<=finished_s,'saved native snapshot outside its actual invocation')
  require(not module.engine_residual(native,stamp),'saved native requests/KV/transfer/ACK not empty and healthy')
  counts=[native.get(k) for k in ('transfer_send_started','transfer_send_completed','transfer_send_failed')]
  require(native.get('transfer_send_counters_observed') is True and native.get('transfer_inflight_sends_observed') is True and native.get('transfer_send_healthy') is True and native.get('transfer_inflight_sends')==0 and all(type(v) is int and v>=0 for v in counts) and counts[0]==counts[1] and counts[2]==0,'saved native sends unsettled')
  require(native.get('scheduler_budget_pending') is None,'saved native budget pending')
  caches=[r.get('controls',{}).get('runtime') for r in native.get('scheduler_io',[])]
  require(len(caches)==instance.get('scheduler_cache_count',1) and all(c and c.get('generation')==native['generation'] and c.get('error') is None for c in caches),'saved native scheduler ACK missing')
  if restored:
   budget=native.get('scheduler_budget_effective',{})
   require(native.get('accepting') is True and budget.get('max_num_batched_tokens')==instance['restore_budget_tokens'] and budget.get('max_num_seqs')==32,'saved original native budget not restored')
 return True

def audit_terminal(out,expected,spec_reference):
 # Historical eligibility is bound to snapshots of this invocation. Live
 # admission separately checks the current node under its exclusive lease.
 s=fixed(ref(out/'status.json'))
 require(s.get('complete') and s.get('cleanup_complete') and not s.get('error') and not s.get('cleanup_errors') and s.get('finished_s'),'work or cleanup incomplete '+str(out))
 require(len(s['completed'])==expected and len({r['path'] for r in s['completed']})==expected,'declared distinct observation count differs')
 require(not prior.alive(s['pid']),'predecessor still owns stage')
 prior.raw_measurement(s['full_operation_measurement'])
 require(fixed(ref(out/'spec-reference.json'))==spec_reference,'executed predecessor specification differs')
 sp=fixed(spec_reference);instances=fixed(sp['original_binding'])['instances']
 i=fixed(ref(out/'inventory.json'))
 require(i['complete'] and not i['transition_inflight'] and i['active_instances']==instances and i['identity']==fixed(sp['capacity_binding'])['identity'],'actual exact initial-two cleanup required')
 require(not any(e['kind'] in ('transition_failed','rollback_failed') for e in i['events']),'physical failure in predecessor')
 require(all(v.get('state') in ('stopped','stopped_after_failure') for k,v in i['known_instances'].items() if k not in i['initial_ids']),'saved owned extra process has not stopped')
 for reference in s['completed']:prior.audit_observation(reference,sp,i)
 for name in ('identity.before.json','identity.after.json'):
  saved_identity(fixed(ref(out/name)),instances,s['started_s'],s['finished_s'])
 require(s.get('retained_restoration_complete') is True and s['retained_restoration']==ref(out/'retained-restoration.json'),'actual independent retained restoration missing')
 restoration=fixed(s['retained_restoration'])
 require(restoration['passed'] and restoration['capacity_cleanup_complete'] and restoration['original_controller_failure_not_waived'] and restoration['experiment_requests_sent']==0,'saved retained restoration not complete')
 saved_identity(restoration['before'],instances,s['started_s'],s['finished_s'])
 saved_identity(restoration['after'],instances,s['started_s'],s['finished_s'],restored=True)
 require(len(restoration['replies'])==len(instances),'all restored native controls required')
 for reply,instance in zip(restoration['replies'],instances):
  before,after,payload=reply['before'],reply['after'],reply['control']
  require(before['id']==after['id']==instance['id'] and payload['generation']==before['generation']+1==after['generation'] and after['accepting'] is True,'actual original native generation control not acknowledged')
  require(payload['scheduler_budget']['max_num_batched_tokens']==instance['restore_budget_tokens'] and payload['scheduler_budget']['max_num_seqs']==32,'original native requested budget differs')
 return s


def audit_rows(r,trace,config,raw,rows,duration):
 n=trace['n_requests'];require(n>0 and len(rows)==len(trace['requests'])==len(trace['prompts'])==n==r['n_expected']==r['n_rows'],'actual900 denominator differs')
 epoch=r['actual_arrival_epoch_s'];require(close(r['measured_arrival_duration_s'],duration),'actual900 window differs')
 require(len({x['request_id'] for x in rows})==n,'distinct actual900 request identities required')
 good=0
 for idx,(x,t,prompt) in enumerate(zip(rows,trace['requests'],trace['prompts'])):
  require(x['idx']==idx and str(x['request_id'])==str(idx),'actual900 row identity/order differs')
  require(x['success']==1 and x['request_timeout'] is False and x['token_ids_verified']==1 and x['generated_tokens']==x['output_len']==t['output_len'] and x['input_tokens']==x['prompt_len']==t['prompt_len']==len(prompt),'independent900 full-work/token audit failed')
  require(math.isclose(x['planned_arrival_s'],epoch+t['arrival_s'],rel_tol=0,abs_tol=1e-6) and close(x['request_deadline_s']-x['planned_arrival_s'],120),'actual900 arrival/deadline differs')
  require(all(type(x[k]) in (int,float) and math.isfinite(x[k]) and x[k]>=0 for k in ('ttft_s','tpot_s')),'actual900 timing invalid')
  ok=x['ttft_s']<config['slo_ttft_s'] and x['tpot_s']<config['slo_tpot_s'];require(x['slo_ok']==int(ok),'actual900 request SLO differs');good+=ok
 require(good==r['n_good'] and close(good/n,r['slo_attainment']),'actual900 SLO summary differs')
 require(r.get('failed_requests')==0 and r.get('request_timeouts')==0 and r.get('work_complete') is True,'actual900 failure/incomplete work')
 require(close(r['energy_j'],raw['energy_j']) and close(r['offered_rate_rps'],n/duration),'actual900 reported energy/rate differs from raw')
 require(raw['measurement_start_s']<=epoch and raw['measurement_end_s']>=epoch+duration,'actual900 raw power must cover the full arrival window')
 # Qualification proves lifecycle behavior. A complete low-SLO observation is
 # retained; only the subsequent original 100s experiment assesses formal SLO.
 return dict(n_expected=n,n_good=good,slo_attainment=good/n,energy_j=raw['energy_j'])

def audit_stage(out,sr,stage):
 out=Path(out);sp=fixed(sr);cap=fixed(sp['capacity_binding']);config=fixed(sp['config'])
 validate_compatibility(sp,cap)
 require(sp['files'] and all(sha(p)==h for p,h in sp['files'].items()),'actual P8 stage source/raw files changed')
 require(config['capacity_binding_path']==sp['capacity_binding']['path'] and config['capacity_binding_sha256']==sp['capacity_binding']['sha256'] and config['profiles']==sp['profiles']['path'],'executed configuration source binding differs')
 status=audit_terminal(out,1,sr)
 require(sp['mode']==('automatic_underload_gate' if stage=='gate' else 'qualification900') and sp['arm']=='dynamic','actual P8 autonomous stage required')
 r=fixed(status['completed'][0]);duration=60 if stage=='gate' else 900
 require(r['trace']==sp['trace'] and r['demand_domain_sha256']==sp['demand_domain_sha256'],'actual source trace/domain changed')
 trace=fixed(r['trace']);validate_trace(trace,sp['demand_domain_sha256'],duration=duration)
 if stage=='gate':require(trace['n_requests']==752,'exact declared752 autonomous first gate')
 else:require([p['name'] for p in trace['phases']]==['low','high','low'] and all(p['duration_s']==300 for p in trace['phases']),'original900 low/high/low required')
 actual=fixed(ref(out/'runtime-config.json'));expected=dict(config,journal=str(out/'control.jsonl'),capacity_inventory_path=str(out/'inventory.json'))
 require(actual==expected,'executed runtime configuration differs')
 paths=[p for p in r['artifacts'] if Path(p).name=='requests.json'];require(paths==[str(out/('automatic_underload_gate' if stage=='gate' else 'qualification900')/'requests.json')],'actual request artifact path differs')
 raw=prior.raw_measurement(r['raw_measurement']);fullraw=prior.raw_measurement(status['full_operation_measurement'])
 metrics=audit_rows(r,trace,config,raw,json.loads(Path(paths[0]).read_text()),duration)
 inv=json.loads((out/'inventory.json').read_text());commits=[e for e in inv['events'] if e['kind']=='physical_commit']
 require({e['operation'] for e in commits}=={'restore','remove'} and all(e['execution_verified'] and raw['measurement_start_s']<=e['started_s']<=e['finished_s']<=raw['measurement_end_s'] for e in commits),'physical growth/return must be inside full eight-GPU service energy')
 control=[json.loads(l) for l in (out/'control.jsonl').open()];dispatch=[json.loads(l) for l in (out/'engine-dispatch.jsonl').open()]
 if stage=='gate':
  evidence=autonomous_gate_evidence(inv,dispatch,control,r['n_expected'])
  require(evidence==fixed(ref(out/'autonomous-gate-evidence.json')),'actual autonomous gate route evidence differs')
 else:
  evidence=autonomous_gate_evidence(inv,dispatch,control,r['n_expected'])
 misses=[e for e in inv['events'] if e['kind']=='empirical_transition_estimate_exceeded']
 files={str(p):sha(p) for p in out.rglob('*') if p.is_file()};files.update(sp['files']);files[str(Path(__file__))]=sha(__file__);files[sr['path']]=sr['sha256']
 return dict(schema='P8-actual-autonomous-stage-audit-v1',passed=True,stage=stage,source=ref(Path(sp['host_release'])/'manifest.json'),profile=sp['profiles'],controller_calibration_compatibility=sp['controller_calibration_compatibility'],result=status['completed'][0],status=ref(out/'status.json'),spec=sr,output=str(out),full_work=True,failed_requests=0,request_timeouts=0,raw_energy_recomputed=True,full_operation_energy_j=fullraw['energy_j'],physical_commits=len(commits),autonomous_evidence=evidence,empirical_cost_prediction_misses=misses,empirical_bounds_not_hard_guarantees=True,files=files,**metrics)

def audit_gate():
 declaration=fixed(ref(A/'p8-autonomous-gate-inputs-001/declaration.json'))
 return audit_stage(A/'p8-autonomous-gate-001',declaration['specs']['dynamic'],'gate')

def audit900():
 declaration=fixed(ref(A/'p8-qualification900-inputs-001/declaration.json'))
 return audit_stage(A/'p8-qualification900-dynamic-001',declaration['specs']['dynamic'],'900')
