"""Owned reset -> covered2100 idle observation. Diagnostic only, no model requests."""
import argparse,asyncio,copy,hashlib,importlib.util,json,math,os,socket,sys,time,signal
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent
sys.path.insert(0,str(R/'common/distributed14b-deployment-v1'));import deploy
read,sha,ref,need=deploy.read,deploy.sha,deploy.reference,deploy.require

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def checked(r):need(sha(r['path'])==r['sha256'],'changed reference '+r['path']);return read(r['path'])
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def spec_check(s):
 need(s['schema']=='postpark-idle-transition-diagnostic-input-v1' and s['authorized'] is True,'explicit diagnostic declaration required')
 need(s['node'] in ('B','C') and s['target_mhz']==2100 and s['observation_limit_s']==2.0 and s['tolerance_mhz']==15 and s['stable_interval_s']==.05 and s['sample_interval_s']==.01,'fixed diagnostic scope changed')
 need(s['model_requests']==0 and s['service_timeouts_changed'] is False and s['automatic_retries'] is False,'diagnostic must not issue model work or change serving contract')
 for p,h in s['files'].items():need(sha(p)==h,'frozen source differs '+p)
 b=checked(s['binding']);need(b['hostname']==s['hostname'] and b['model']=='14b' and b['system']=='pdblend','wrong actual binding')
 need(len(b['instances'])==2 and {tuple(i['gpus']) for i in b['instances']}=={(6,),(7,)} and all(i['tp']==1 and i['native_kind']=='v3' for i in b['instances']),'exact retained fixed2 required')
 need(ref(Path(b['host_release'])/'manifest.json')==s['host_manifest'],'wrong source manifest')
 need(s['points']==[dict(point_id=f'gpu{g}-repeat{j}',gpu=g,repeat=j,instance_id=next(i['id'] for i in b['instances'] if i['gpus']==[g])) for g in (6,7) for j in (1,2,3)],'six predeclared resets required')
 return b

def observation_result(rows,target,started,deadline,tolerance,stable):
 run=[];first=None;last_wrong=None
 for r in rows:
  value=r['observed_mhz'];need(type(value) in (int,float) and math.isfinite(value) and value>0,'invalid actual clock')
  if abs(value-target)<=tolerance:
   if first is None:first=r
   run.append(r)
   if len(run)>=2 and run[-1]['finished_s']-run[0]['finished_s']>=stable and run[-1]['finished_s']<=deadline:
    return dict(passed=True,first_in_band=first,stable_first=run[0],stable_confirmed=run[-1],last_prior_out_of_band=last_wrong,observed_confirmation_latency_s=run[-1]['finished_s']-started,observed_first_latency_s=first['finished_s']-started,physical_settling_time_independently_known=False)
  else:run=[];first=None;last_wrong=r
 return dict(passed=False,reason='no complete stable actual observation before diagnostic limit',physical_settling_time_independently_known=False)

async def observe(hw,cl,gpu,target,command_finished,s):
 rows=[];deadline=command_finished+s['observation_limit_s'];loop=asyncio.get_running_loop()
 def sample():
  start=time.time();value=hw.current_freq(gpu)
  try:idle=hw.clock_idle(gpu)
  except Exception as exc:idle=repr(exc)
  return dict(gpu=gpu,observed_mhz=value,started_s=start,finished_s=time.time(),original_clock_idle_flag=idle)
 while time.time()<=deadline:
  rows.append(await loop.run_in_executor(cl.pool,sample))
  result=observation_result(rows,target,command_finished,deadline,s['tolerance_mhz'],s['stable_interval_s'])
  if result['passed']:return dict(target_mhz=target,diagnostic_deadline_s=deadline,rows=rows,**result)
  await asyncio.sleep(min(s['sample_interval_s'],max(0,deadline-time.time())))
 return dict(target_mhz=target,diagnostic_deadline_s=deadline,rows=rows,**observation_result(rows,target,command_finished,deadline,s['tolerance_mhz'],s['stable_interval_s']))

async def execute(s,out,lease):
 b=spec_check(s);deploy.require_lease(lease);need(socket.gethostname()==s['hostname'] and not out.exists(),'new output on assigned node required');out.mkdir(parents=True)
 common=deploy.adapter.load_runtime(b['host_release'],Path(b['executor']).parent)
 import aiohttp
 from ecopadg.measure.backends import PynvmlBackend
 from ecopadg.serving.backend import ClockOwner
 hooks=load(Path(s['measurement_hooks']['path']),'idle_probe_hooks');hooks.install(out/'isolated-samplers',s['host_manifest'],s['measurement_adapter'])
 hw=await asyncio.to_thread(PynvmlBackend,power_mode='instant');status=dict(schema='postpark-idle-transition-diagnostic-result-v1',pid=os.getpid(),started_s=time.time(),complete=False,passed=False,points=[],errors=[],model_requests=0,binding=s['binding'],spec=ref(out/'spec.json') if (out/'spec.json').exists() else None)
 write(out/'spec.json',s);status['spec']=ref(out/'spec.json');meter=deploy.Measurement(out,hw,status);cl=None;events=[];stop=False
 def request_stop():
  nonlocal stop;stop=True
 for sig in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(sig,request_stop)
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
  try:
   common.validate_binding(b);write(out/'identity.before.json',await common.identity(session,b));await meter.start()
   cl=await asyncio.to_thread(ClockOwner,hw,(6,7),max_service_frequency_mhz=2100);cl.write_guard=lambda *args:dict(allowed=True,diagnostic_only=True,node_lease_fd=lease)
   cl.clock_event=lambda e:events.append(copy.deepcopy(e))
   for p in s['points']:
    need(not stop and not (out/'STOP').exists(),'diagnostic stopped at a completed transition boundary')
    status['current_point']=p;write(out/'status.json',status);i=next(i for i in b['instances'] if i['id']==p['instance_id']);d=out/p['point_id'];d.mkdir();raw=dict(point=p,started_s=time.time(),complete=False,errors=[])
    try:
     raw['native_before']=await common.wait_idle(session,i);raw['all_native_before']=[await common.wait_idle(session,x) for x in b['instances']]
     raw['reset_started_s']=time.time()
     async with cl.lock:
      await cl.physical_park(p['gpu']);cl.applied.pop(p['gpu'],None)
     raw['reset_finished_s']=time.time();raw['reset_observation']=await observe(hw,cl,p['gpu'],s['observed_default_mhz'],raw['reset_finished_s'],s)
     need(raw['reset_observation']['passed'],'reset did not reproduce the observed postpark default within diagnostic bound')
     raw['native_after_reset']=await common.wait_idle(session,i)
     raw['write_started_s']=time.time()
     async with cl.lock:await cl.physical_write(p['gpu'],2100,asyncio.get_running_loop())
     raw['write_finished_s']=time.time();raw['target_observation']=await observe(hw,cl,p['gpu'],2100,raw['write_finished_s'],s)
     need(raw['target_observation']['passed'],'target did not reach stable observed domain within diagnostic bound')
     cl.applied[p['gpu']]=2100;raw['native_after']=await common.wait_idle(session,i);raw['complete']=True
    except BaseException as exc:raw['errors'].append(repr(exc));raise
    finally:raw['finished_s']=time.time();write(d/'raw.json',raw);status['points'].append(dict(point=p,raw=ref(d/'raw.json'),complete=raw['complete']));write(out/'status.json',status)
   status['complete']=len(status['points'])==6
  except BaseException as exc:status['errors'].append(repr(exc))
  finally:
   status['native_cleanup']=[]
   for i in b['instances']:
    try:status['native_cleanup'].append(await asyncio.wait_for(common.restore(session,i),90))
    except BaseException as exc:status['native_cleanup'].append(dict(complete=False,errors=[repr(exc)]))
   if cl is not None:
    try:await asyncio.wait_for(cl.close(),90);status['clock_restore_complete']=True
    except BaseException as exc:status['errors'].append('owned clock close '+repr(exc))
   if not all(x.get('complete') and not x.get('errors') for x in status['native_cleanup']):status['errors'].append('native cleanup incomplete')
   try:write(out/'identity.after.json',await common.identity(session,b))
   except BaseException as exc:status['errors'].append('final identity '+repr(exc))
   await meter.finish()
   try:
    from ecopadg.measure.power import trapezoid_energy
    from ecopadg.metrics import clip_power_window
    for item in status['points']:
     path=Path(item['raw']['path']);raw=read(path)
     if raw['complete']:
      raw['reset_to_default_observation_energy_j']=trapezoid_energy(clip_power_window(meter.sampler.samples,raw['reset_started_s'],raw['reset_observation']['stable_confirmed']['finished_s'],pad_s=0))
      raw['write_to_target_observation_energy_j']=trapezoid_energy(clip_power_window(meter.sampler.samples,raw['write_started_s'],raw['target_observation']['stable_confirmed']['finished_s'],pad_s=0))
      raw['energy_scope']='all eight instantaneous power samples; observation interval, not isolated physical-command energy'
      write(path,raw);item['raw']=ref(path)
   except BaseException as exc:status['errors'].append('transition energy replay '+repr(exc))
   write(out/'clock-commands.json',events)
   try:
    roots=hooks.directories(out/'isolated-samplers');hooks.completed_artifacts(roots,s['host_manifest'],s['measurement_adapter']);status['isolated_samplers']=hooks.sampler_references(roots)
   except BaseException as exc:status['errors'].append('sampler close '+repr(exc))
   status.update(finished_s=time.time(),passed=status['complete'] and status.get('measurement_valid') is True and not status['errors']);write(out/'status.json',status)
   files={str(p):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'evidence-manifest.json',dict(schema='postpark-idle-transition-diagnostic-evidence-v1',spec=ref(out/'spec.json'),passed=status['passed'],files=files))
 return status

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true');a=p.parse_args();s=read(a.spec);b=spec_check(s)
 deploy.adapter.load_runtime(b['host_release'],Path(b['executor']).parent)
 if not a.run:print(json.dumps(dict(CPU_only=True,points=6,model_requests=0)));return
 from ecopadg.serving.campaign import node_lease
 need('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh exclusive node lease required')
 with node_lease() as lease:status=asyncio.run(execute(s,a.out.resolve(),lease))
 print(json.dumps({k:status[k] for k in ('passed','complete','errors')}));raise SystemExit(0 if status['passed'] else 1)
if __name__=='__main__':main()
