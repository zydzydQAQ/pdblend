"""Actual controller/backend CPU regressions; all profile2400 fixtures are synthetic."""
import argparse,ast,asyncio,copy,importlib,json,sys,tempfile,time,faulthandler
faulthandler.dump_traceback_later(15, repeat=False)
from pathlib import Path
R=Path(__file__).resolve().parent.parent

class Hardware:
 def __init__(self,low=False):self.writes=[];self.resets=[];self.freq={0:900,1:900};self.low=low
 def set_clock(self,g,f):self.writes.append((g,f));self.freq[g]=f
 def current_freq(self,g):return 900 if self.low else self.freq[g]
 def reset_clock(self,g):self.resets.append(g)
 def clock_idle(self,g):return False

async def backend_cases(b):
 out=[]
 for tp in (1,2):
  print('backend tp',tp,flush=True)
  with tempfile.TemporaryDirectory() as d:
   h=Hardware();c=b.ClockOwner(h,range(tp),lock_dir=d,max_service_frequency_mhz=2400)
   try:await c.set(list(range(tp)),2520)
   except ValueError:pass
   else:raise AssertionError('above-domain accepted')
   assert not h.writes
   for f in (900,1500,2100,2400):
    print('set',f,flush=True);await asyncio.wait_for(c.set(list(range(tp)),f),3)
   assert all(f<=2400 for g,f in h.writes);await asyncio.wait_for(c.close(),3);assert h.resets==list(range(tp))
   out.append(dict(case='actual-set-bound-and-reset',tp=tp,passed=True))
  for deferred in (False,True):
   print('fallback',tp,deferred,flush=True)
   with tempfile.TemporaryDirectory() as d:
    h=Hardware(low=True);c=b.ClockOwner(h,range(tp),lock_dir=d,max_service_frequency_mhz=2400,settle_timeout_s=.01);c.failure_fresh_confirmation=True
    fallback=[]
    def guard(gs,f,reason,bootstrap):
     if 'failure fallback' in reason:fallback.append(f);return dict(allowed=False,error='original pending route blocks fallback')
     return dict(allowed=True)
    c.write_guard=guard;c.applied.update({g:2100 if deferred else 900 for g in range(tp)})
    if deferred:c.deferred.update({g:dict(target=2100,gpus=tuple(range(tp)),active_since=time.monotonic()-.1) for g in range(tp)})
    try:await(c.verify_deferred() if deferred else c.set(list(range(tp)),2100))
    except b.ClockWriteUncertain:pass
    else:raise AssertionError('unknown state/blocked fallback admitted')
    assert fallback and set(fallback)=={2400};assert not c.lock.locked();await asyncio.wait_for(c.close(),3);assert h.resets==list(range(tp))
    out.append(dict(case='guarded-fallback-domain-and-unknown-reject',tp=tp,deferred=deferred,passed=True))
 for x in (0,2401,True,None,2400.,'2400'):
  with tempfile.TemporaryDirectory() as d:
   try:b.ClockOwner(Hardware(),[0],lock_dir=d,max_service_frequency_mhz=x)
   except ValueError:pass
   else:raise AssertionError('unsupported domain accepted')
  out.append(dict(case='invalid-domain',value=x,passed=True))
 return out

def controller_cases(module):
 original=json.load(open(R/'A/final-p9-release-001/configs/longbench.json'));base=json.load(open(original['profiles']))
 with tempfile.TemporaryDirectory() as d:
  # CPU fixtures only; never publish these mapped point values as observations.
  profile=copy.deepcopy(base)
  for p in profile['points']:
   if p['frequency_mhz']==2520:p['frequency_mhz']=2400
  path=Path(d)/'SYNTHETIC-CPU-profile.json';path.write_text(json.dumps(profile))
  cases=[]
  for strategy in ('pdblend-joint','mixed','ecoserve','distserve','dynamollm-resident','dynamollm'):
   cfg=dict(original,profiles=str(path),max_service_frequency_mhz=2400,strategy=strategy,journal=str(Path(d)/(strategy+'.jsonl')),capacity_integration_v1=False,interconnect=None)
   cfg.pop('capacity_binding_path',None);cfg.pop('capacity_binding_sha256',None)
   if not strategy.startswith('pdblend'):cfg['measured_frequency_write_guard_v1']=False
   if strategy.startswith('dynamollm'):
    cfg.update(dynamo_assignments={i['id']:'LL' for i in cfg['instances']},topology=dict(runtime_dir='/tmp/CPU-only-unused',image='CPU-only',engine_template='/tmp/CPU-only-unused.json'),topology_costs=[dict(source_tps=[1],target_tps=[2],duration_upper_s=1,energy_upper_j=1,source_sha256='synthetic-CPU',cached_weights=True)])
   if strategy=='distserve':cfg['instances']=copy.deepcopy(cfg['instances']);cfg['instances'][0]['role']='prefill';cfg['instances'][1]['role']='decode'
   c=module.Controller(cfg);assert c.max_service_frequency_mhz==c.planner.max_frequency==2400
   for scheduler in (c.eco_scheduler,c.distserve_scheduler,c.dynamo_scheduler,c.frequency_planner,getattr(c,'dynamo_topology',None)):
    if scheduler is not None:assert scheduler.max_service_frequency_mhz==2400
   if c.distserve_scheduler:assert c.distserve_scheduler.estimator.max_frequency==2400
   if c.dynamo_scheduler:assert c.dynamo_scheduler.estimator.max_frequency==2400
   cases.append(dict(case='actual-controller-full-constructor-chain',strategy=strategy,passed=True))
  bad=dict(original,max_service_frequency_mhz=2400,journal=str(Path(d)/'bad.jsonl'),capacity_integration_v1=False)
  try:module.Controller(bad)
  except ValueError as exc:assert 'profile frequencies exceed' in str(exc)
  else:raise AssertionError('unfiltered high profile accepted')
  cases.append(dict(case='old2520-profile-cannot-masquerade2400',passed=True))
 return cases

def default_proof(host):
 import importlib.util
 sp=importlib.util.spec_from_file_location('p10_builder_audit',R/'build_hardware_max_p10.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 parent=R/'hosts/14b-capacity-p9';old=json.load(open(parent/'manifest.json'));new=json.load(open(host/'manifest.json'));changed=[]
 for name,digest in old['files'].items():
  p=parent/name;q=host/name
  if p.read_bytes()!=q.read_bytes():
   assert m.normalize_default(q.read_text(),q.name)==ast.dump(ast.parse(p.read_text()),include_attributes=False),name
   changed.append(name)
 # Initialization uses asyncio.to_thread; cap must be forwarded to actual constructor.
 tree=ast.parse((host/'src/ecopadg/serving/runtime.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to_thread' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='ClockOwner']
 assert len(calls)==1 and any(k.arg=='max_service_frequency_mhz' for k in calls[0].keywords)
 return dict(case='all-changed-default-specialization-and-real-thread-forward',changed_files=changed,passed=True)

async def main(a):
 sys.path[:0]=[str(a.host/'src'),str(a.host),'/root/workspace/pdblend/.runtime-deps']
 b=importlib.import_module('ecopadg.serving.backend');runtime=importlib.import_module('ecopadg.serving.runtime')
 assert Path(b.__file__).resolve()==(a.host/'src/ecopadg/serving/backend.py').resolve()
 cases=await asyncio.wait_for(backend_cases(b),30);print('controllers',flush=True);cases+=controller_cases(runtime);cases.append(default_proof(a.host));faulthandler.cancel_dump_traceback_later()
 out=dict(schema='P10-hardware-max-independent-CPU-v1',passed=True,hardware_actions=False,synthetic_profiles_never_published=True,cases=cases,tests=len(cases),source=str(a.host))
 assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=True,tests=len(cases))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',type=Path,required=True);p.add_argument('--out',type=Path,required=True);asyncio.run(main(p.parse_args()))
