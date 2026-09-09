"""Pure saved six-transition audit; no live processes, native RPC, or GPU writes."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
R=Path(__file__).resolve().parent.parent.parent

def need(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(r):need(sha(r['path'])==r['sha256'],'saved bytes changed '+r['path']);return read(r['path'])
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(reference):
 e=checked(reference);need(e['schema']=='postpark-idle-transition-diagnostic-evidence-v1','unknown diagnostic evidence')
 files=e['files'];need(files and all(sha(p)==h for p,h in files.items()),'diagnostic raw changed')
 out=Path(reference['path']).parent;s=checked(e['spec']);need(files.get(e['spec']['path'])==e['spec']['sha256'],'spec not frozen')
 source=R/'common/postpark-idle-transition-probe-v1/manifest.json';need(sha(source)=='824daac3891ff74d90eac4065f41b1c0948df43d4f95130b56a7ea8757ffc6aa','wrong actual diagnostic source')
 actual=load(source.parent/'run.py','saved_idle_probe_source');b=actual.spec_check(s)
 q=load(R/'common/distributed14b-qualification-v1/verify.py','saved_q');nv=load(R/'common/distributed14b-profile-validation-v1/validate.py','saved_native')
 status=read(out/'status.json');need(status['spec']==e['spec'] and status['binding']==s['binding'] and status['model_requests']==0,'actual declaration mismatch')
 need(status['complete'] is True and not status['errors'] and status['clock_restore_complete'] is True and len(status['points'])==6,'incomplete diagnostic/cleanup')
 allfiles={**s['files'],**files};allfiles.update({reference['path']:reference['sha256'],str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)})
 for side in ('identity.before.json','identity.after.json'):
  rows=read(out/side);ids=q.identities(rows,b['instances'],allfiles)
  for i in b['instances']:nv.native_saved(ids[i['id']]['runtime'],i['id'],8192)
 for i,clean in zip(b['instances'],status['native_cleanup']):nv.cleanup_saved(clean,i['id']) if clean['before'].get('scheduler_budget_effective',{}).get('max_num_batched_tokens')==2048 else native_restore(clean,i,actual.deploy)
 power,clocks=q.power_operation(out,status,s['host_manifest'],files)
 observer=load(R/'isolated_measurement_audit_v1.py','saved_observer');proof=observer.audit_samplers(status['isolated_samplers'],s['host_manifest'],artifacts=files);observer_match=observer.match_power_directory(out/'power',proof,artifacts=files)
 with (out/'power/power.csv').open() as f:power_rows=[(float(x['t_s']),[float(x[f'gpu{g}_w']) for g in range(8)]) for x in csv.DictReader(f)]
 mathmod=q.original_power_functions(s['host_manifest']);results=[]
 for declared,item in zip(s['points'],status['points']):
  need(item['point']==declared and item['complete'] is True and files.get(item['raw']['path'])==item['raw']['sha256'],'predeclared point mismatch')
  raw=checked(item['raw']);need(raw['point']==declared and raw['complete'] is True and not raw['errors'],'raw point incomplete')
  iid=declared['instance_id'];need(len(raw['all_native_before'])==2,'both native engines must be empty')
  for i,n in zip(b['instances'],raw['all_native_before']):nv.native_saved(n,i['id'],8192)
  for key in ('native_before','native_after_reset','native_after'):
   nv.native_saved(raw[key],iid,8192);need(raw['started_s']-1<=raw[key]['timestamp']<=raw['finished_s'],'native proof belongs to another diagnostic interval')
  for key,target,startkey in (('reset_observation',2520,'reset_finished_s'),('target_observation',2100,'write_finished_s')):
   o=raw[key];start=raw[startkey];deadline=start+s['observation_limit_s'];need(o['target_mhz']==target and o['diagnostic_deadline_s']==deadline,'observation limit changed')
   rows=o['rows'];need(rows and all(x['gpu']==declared['gpu'] and start<=x['started_s']<=x['finished_s']<=raw['finished_s'] for x in rows),'actual observation membership/window differs')
   need(all(a['finished_s']<=b['started_s'] for a,b in zip(rows,rows[1:])),'clock observations overlap/reordered')
   rebuilt=actual.observation_result(rows,target,start,deadline,15,.05);need(rebuilt['passed'] is True and all(o.get(k)==v for k,v in rebuilt.items()),'saved observation qualification differs')
  for startkey,endkey,valuekey in (('reset_started_s','reset_observation','reset_to_default_observation_energy_j'),('write_started_s','target_observation','write_to_target_observation_energy_j')):
   energy=mathmod['trapezoid_energy'](mathmod['clip_power_window'](power_rows,raw[startkey],raw[endkey]['stable_confirmed']['finished_s'],pad_s=0))
   need(math.isclose(energy,raw[valuekey],rel_tol=1e-9,abs_tol=1e-6),'transition all8 energy differs')
  results.append(dict(point=declared,raw=item['raw'],first_observed_target_s=raw['target_observation']['observed_first_latency_s'],stable_observed_target_s=raw['target_observation']['observed_confirmation_latency_s'],write_to_target_energy_j=raw['write_to_target_observation_energy_j']))
 events=read(out/'clock-commands.json');commands=[x for x in events if x.get('stage')=='command_completed'];need(len(commands)==12 and all(x['completed'] is True for x in commands),'six completed reset/set physical transactions required')
 for j,item in enumerate(results):
  raw=checked(item['raw']);pair=commands[2*j:2*j+2]
  need(raw['reset_started_s']<=pair[0]['started_s']<=pair[0]['finished_s']<=raw['reset_finished_s'] and raw['write_started_s']<=pair[1]['started_s']<=pair[1]['finished_s']<=raw['write_finished_s'],'native command timestamps differ from measured intervals')
  need([x['kind'] for x in pair]==['physical_clock_park','physical_clock_write'] and [x['gpu'] for x in pair]==[item['point']['gpu']]*2 and pair[1]['target_mhz']==2100,'physical command sequence differs')
 need(not any(x.get('stage')=='command_uncertain' for x in events),'uncertain diagnostic writes')
 for p in (R/'common/distributed14b-qualification-v1/verify.py',R/'common/distributed14b-profile-validation-v1/validate.py',R/'isolated_measurement_audit_v1.py'):allfiles[str(p)]=sha(p)
 return dict(passed=True,independently_recomputed=True,node=s['node'],hostname=s['hostname'],binding=s['binding'],host_manifest=s['host_manifest'],measurement=reference,model_requests=0,points=results,maximum_first_observation_s=max(x['first_observed_target_s'] for x in results),maximum_stable_observation_s=max(x['stable_observed_target_s'] for x in results),whole_operation_power=power,observer_match=observer_match,native_cleanup_complete=True,clock_restore_complete=True,files=allfiles,empirical_only=True,service_timeout_not_registered=True)

def native_restore(clean,i,deploy):
 common=load(R/'common/execution-until-complete-v1/run.py','original_native_restore');nv=load(R/'common/distributed14b-profile-validation-v1/validate.py','saved_nv_restore')
 need(clean['complete'] is True and not clean['errors'],'native restoration incomplete');common.barrier(clean['before'],clean['proof'],i)
 r=clean['resumed'];nv.native_saved(r['after'],i['id'],8192)
 need(r['control']==dict(generation=r['before']['generation']+1,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True,scheduler_budget=dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)) and r['after']['generation']==r['control']['generation'],'native restored control changed')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--measurement',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();x=verify(ref(a.measurement))
 with a.out.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,maximum_first_observation_s=x['maximum_first_observation_s'],maximum_stable_observation_s=x['maximum_stable_observation_s'],receipt=ref(a.out))))
