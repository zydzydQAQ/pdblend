"""No-lease continuation: complete full27 -> raw certificate -> fixed/dynamic900 pair."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,math
A=Path(__file__).resolve().parent;CODE=A/'load-p6-code-001';sys.path.insert(0,str(CODE))
from capacity_executor import fixed,require,durable,sha
from capacity_certificate import raw_measurement,window_energy,ref,idle_result,close,source_identity
from capacity_load_calibrate import validate_trace
OUT=A/'p6-qualification-continuation-003'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def audit_observation(reference,sp,inventory):
 r=fixed(reference)
 require(r.get('schema')=='capacity-load-measurement-v1','unexpected completed result schema')
 expected_source=dict(original_binding=sp['original_binding'],capacity_binding=sp['capacity_binding'],config=sp['config'],host_manifest=ref(Path(sp['host_release'])/'manifest.json'))
 require(r.get('source')==expected_source,'measured source differs from executed specification')
 if r.get('phase_kind')=='idle':
  # Original idle records deliberately contain no request counters. Their
  # eligibility comes from exact empty work plus continuous native/power raw.
  require(inventory is not None,'idle physical inventory required')
  idle_result(reference,fixed(sp['capacity_binding'])['identity'],inventory)
 else:
  require(r.get('complete') is True and r.get('work_complete') is True and r.get('native_idle') is True and r.get('failed_requests')==0 and r.get('request_timeouts')==0,'request failure prevents any successor')
  require(r.get('artifacts') and all(sha(p)==h for p,h in r['artifacts'].items()),'completed request artifacts changed')
  raw_measurement(r['raw_measurement'])
 return r

def audit(out,expected,spec_reference):
 s=json.loads((out/'status.json').read_text());require(s.get('complete') and s.get('cleanup_complete') and not s.get('error') and not s.get('cleanup_errors') and s.get('finished_s'),'work or cleanup incomplete '+str(out))
 require(len(s['completed'])==expected and len({r['path'] for r in s['completed']})==expected,'declared distinct observation count differs')
 require(not alive(s['pid']),'predecessor still owns stage')
 raw_measurement(s['full_operation_measurement'])
 sr=json.loads((out/'spec-reference.json').read_text());require(sr==spec_reference,'executed predecessor specification differs');sp=fixed(sr)
 inv=out/'inventory.json';i=json.loads(inv.read_text()) if inv.exists() else None
 if i is not None:
  require(i['complete'] and not i['transition_inflight'] and {x['id'] for x in i['active_instances']}=={'nextv3a6','nextv3a7'},'actual initial-two cleanup required')
  require(not any(e['kind'] in ('transition_failed','rollback_failed') for e in i['events']),'physical failure in predecessor')
 for reference in s['completed']:audit_observation(reference,sp,i)
 require(set(subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split())=={'pdb-v2-nextv3a6','pdb-v2-nextv3a7'},'unexpected live process on A')
 return s

def audit_rows(r,trace,config,raw,rows):
 n=trace['n_requests'];require(n>0 and len(rows)==len(trace['requests'])==len(trace['prompts'])==n==r['n_expected']==r['n_rows'],'actual900 denominator differs')
 epoch=r['actual_arrival_epoch_s'];require(close(r['measured_arrival_duration_s'],900),'actual900 window differs')
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
 require(close(r['energy_j'],raw['energy_j']) and close(r['offered_rate_rps'],n/900),'actual900 reported energy/rate differs from raw')
 require(raw['measurement_start_s']<=epoch and raw['measurement_end_s']>=epoch+900,'actual900 raw power must cover the full arrival window')
 # Qualification proves lifecycle behavior. A complete low-SLO observation is
 # retained; only the subsequent original 100s experiment assesses formal SLO.
 return dict(n_expected=n,n_good=good,slo_attainment=good/n,energy_j=raw['energy_j'])

def audit900(out,status,declaration,arm):
 sr=declaration['specs'][arm];sp=fixed(sr)
 require(json.loads((out/'spec-reference.json').read_text())==sr,'900 executed specification differs')
 require(sp['mode']=='qualification900' and sp['arm']==arm and sp['deadline_s'] is None and sp['campaign_lifecycle']=='until_declared_complete_v1','900 arm/lifecycle differs')
 require(sp['trace']==declaration['trace'] and sp['profiles']==declaration['profile'] and sp['capacity_binding']==declaration['capacity_binding'] and sp['actual_certificate']==declaration['certificate'] and ref(Path(sp['host_release'])/'manifest.json')==declaration['source'],'paired900 source/profile/trace/certificate differs')
 require(sp['files'] and all(sha(p)==h for p,h in sp['files'].items()),'paired900 frozen files changed')
 cap=fixed(sp['capacity_binding']);source_identity(sp['capacity_binding'],cap['identity'])
 config=fixed(sp['config']);fixed(sp['profiles']);fixed(sp['actual_certificate'])
 require(config['profiles']==sp['profiles']['path'] and (config.get('capacity_integration_v1') is True)==(arm=='dynamic') and config['capacity_binding_path']==sp['capacity_binding']['path'] and config['capacity_binding_sha256']==sp['capacity_binding']['sha256'],'paired900 runtime configuration differs')
 r=fixed(status['completed'][0]);require(r['trace']==sp['trace'] and r['demand_domain_sha256']==sp['demand_domain_sha256'],'actual900 trace/domain differs')
 trace=fixed(r['trace']);validate_trace(trace,sp['demand_domain_sha256'],duration=900)
 require([p['name'] for p in trace['phases']]==['low','high','low'] and all(p['duration_s']==300 for p in trace['phases']),'paired900 phases differ')
 paths=[p for p in r['artifacts'] if Path(p).name=='requests.json'];require(paths==[str(out/'qualification900/requests.json')],'actual900 request artifact identity differs')
 require(all(sha(p)==h for p,h in r['artifacts'].items()),'actual900 artifacts changed')
 rows=json.loads(Path(paths[0]).read_text());raw=raw_measurement(r['raw_measurement']);derived=audit_rows(r,trace,config,raw,rows)
 if arm=='dynamic':
  inv=json.loads((out/'inventory.json').read_text());commits=[e for e in inv['events'] if e['kind']=='physical_commit'];require({e['operation'] for e in commits}=={'restore','remove'},'actual automatic growth and measured return required')
 return dict(result=status['completed'][0],status=ref(out/'status.json'),full_work=True,failed_requests=0,request_timeouts=0,**derived,raw_energy_recomputed=True)

def main():
 require('PDBLEND_NODE_LOCK_FD' not in os.environ,'waiter may not hold a node lease');require(not OUT.exists(),'fresh continuation output');OUT.mkdir()
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_idle_pair_without_lease',node_lease_held=False,complete=False,steps=[])
 sources=[Path(__file__).resolve(),Path(idle_result.__code__.co_filename).resolve(),A/'register_alpaca_capacity_p6_idle.py',A/'prepare_p6_qualification900_v2.py',CODE/'capacity_load_calibrate.py',A/'load-p6-full-inputs-002/spec.json',A/'load-p6-idle-inputs-001/spec.json'];pins={str(p):sha(p) for p in sources};durable(OUT/'source-manifest.json',pins)
 def update(**v):state.update(v,updated_s=time.time());durable(OUT/'status.json',state)
 def check():
  require(not (A/'STOP_P6_QUALIFICATION900').exists(),'STOP prevents next stage');require(all(sha(p)==h for p,h in pins.items()),'prepared continuation source changed')
 def run(name,args):
  check();update(phase=name)
  with (OUT/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);update(child_pid=child.pid);code=child.wait()
  state['steps'].append(dict(phase=name,exitcode=code));state.pop('child_pid',None);update();require(code==0,name+' failed; no automatic retry')
 update()
 try:
  while True:
   check();p=A/'load-p6-idle-001/status.json'
   if not p.exists():time.sleep(3);continue
   s=json.loads(p.read_text())
   if not alive(s['pid']):break
   time.sleep(3)
  audit(A/'load-p6-idle-001',2,ref(A/'load-p6-idle-inputs-001/spec.json'));run('register',[sys.executable,str(A/'register_alpaca_capacity_p6_idle.py')]);run('prepare900',[sys.executable,str(A/'prepare_p6_qualification900_v2.py')])
  declaration=json.loads((A/'p6-qualification900-inputs-002/declaration.json').read_text());results={}
  for arm in ['fixed2','dynamic']:
   sr=declaration['specs'][arm];sp=fixed(sr);out=A/f'p6-qualification900-{arm}-002'
   run(arm,[sys.executable,'-u',str(CODE/'capacity_load_calibrate.py'),'--spec',sr['path'],'--spec-sha256',sr['sha256'],'--out',str(out),'--run'])
   status=audit(out,1,sr);results[arm]=audit900(out,status,declaration,arm)
  require(results['fixed2']['result']['path']!=results['dynamic']['result']['path'],'independent paired executions required')
  durable(OUT/'qualification.json',dict(passed=True,created_s=time.time(),source=declaration['source'],profile=declaration['profile'],certificate=declaration['certificate'],capacity_binding=declaration['capacity_binding'],trace=declaration['trace'],arms=results,initial_instances=2,actual_growth_and_return=True,formal_100s_performance_not_inferred=True,development_only=True))
  update(phase='complete',complete=True)
 except BaseException as exc:update(phase='stopped_failure',error=repr(exc));raise
 finally:update(finished_s=time.time())
if __name__=='__main__':main()
