"""P7 actual autonomous gates; P6 numerical provenance remains explicitly P6."""
from pathlib import Path
import json,math,sys,importlib.util
A=Path(__file__).resolve().parent;R=A.parent;CODE=A/'load-p7-code-001'
sys.path[:0]=[str(CODE),str(A),str(R)]
import continue_p6_qualification_v3 as prior
_s=importlib.util.spec_from_file_location('p7_actual_gate_source',CODE/'capacity_load_calibrate.py')
_gate=importlib.util.module_from_spec(_s);_s.loader.exec_module(_gate)
autonomous_gate_evidence,validate_trace=_gate.autonomous_gate_evidence,_gate.validate_trace
from calibration_compatibility import validate_compatibility
require,fixed,ref,sha,close=prior.require,prior.fixed,prior.ref,prior.sha,prior.close

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
 status=prior.audit(out,1,sr)
 require(sp['mode']==('automatic_underload_gate' if stage=='gate' else 'qualification900') and sp['arm']=='dynamic','actual P7 autonomous stage required')
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
  extra=set(inv['known_instances'])-set(inv['initial_ids'])
  requests={e['request_id'] for e in control if e['kind']=='admission' and any(z['decode_id'] in extra for z in e['plan']['routes'])}
  served=[e for e in dispatch if e['instance_id'] in extra and e['request_id'] in requests]
  require(served,'actual newly added instance must serve declared workload')
  evidence=dict(passed=True,main_native_requests_on_added=len(served),added_instances=sorted(extra))
 files={str(p):sha(p) for p in out.rglob('*') if p.is_file()};files.update(sp['files']);files[str(Path(__file__))]=sha(__file__);files[sr['path']]=sr['sha256']
 return dict(schema='P7-actual-autonomous-stage-audit-v1',passed=True,stage=stage,source=ref(Path(sp['host_release'])/'manifest.json'),profile=sp['profiles'],controller_calibration_compatibility=sp['controller_calibration_compatibility'],result=status['completed'][0],status=ref(out/'status.json'),spec=sr,output=str(out),full_work=True,failed_requests=0,request_timeouts=0,raw_energy_recomputed=True,full_operation_energy_j=fullraw['energy_j'],physical_commits=len(commits),autonomous_evidence=evidence,files=files,**metrics)

def audit_gate():
 declaration=fixed(ref(A/'p7-autonomous-gate-inputs-001/declaration.json'))
 return audit_stage(A/'p7-autonomous-gate-001',declaration['specs']['dynamic'],'gate')

def audit900():
 declaration=fixed(ref(A/'p7-qualification900-inputs-001/declaration.json'))
 return audit_stage(A/'p7-qualification900-dynamic-001',declaration['specs']['dynamic'],'900')
