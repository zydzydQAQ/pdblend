"""Independent CPU TP2 transaction counterexamples, real ClockOwner; no GPU calls."""
import argparse,asyncio,copy,hashlib,json,tempfile,time,sys,threading
from pathlib import Path
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent.parent

def ref(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def find_native(x):
 if isinstance(x,dict):
  if 'acknowledged_generation' in x and 'scheduler_io' in x:return x
  for y in x.values():
   a=find_native(y)
   if a:return a
 if isinstance(x,list):
  for y in x:
   a=find_native(y)
   if a:return a
class Hardware:
 def __init__(self,spec):self.spec=spec;self.values={6:2520,7:2520};self.writes=[];self.resets=[];self.reads=[];self.ready=threading.Event();self.pending={}
 def set_clock(self,g,f):
  self.writes.append([g,f]);self.ready.set()
  if self.spec.get('cancel'):time.sleep(.10)
  if self.spec.get('partial') and g==7:raise RuntimeError('CPU second member write error')
  delay=self.spec.get('settle',0)
  if delay:self.pending[g]=(f,time.monotonic()+delay)
  else:self.values[g]=f
 def current_freq(self,g):
  if g in self.pending and time.monotonic()>=self.pending[g][1]:self.values[g]=self.pending.pop(g)[0]
  v=1500 if self.spec.get('wrong_second') and g==7 else self.values[g]
  self.reads.append([g,v,time.time()]);return v
 def reset_clock(self,g):self.resets.append(g)

async def main(a):
 sys.path[:0]=[str(a.host/'src'),str(a.host),'/root/workspace/pdblend/.runtime-deps']
 from ecopadg.serving.backend import ClockOwner,ClockWriteUncertain
 from ecopadg.serving.idle_domain_reacquire import reacquire
 from ecopadg.serving.state import StateManager,ExpiredPlan
 from ecopadg.serving.types import InstanceState,RuntimeSnapshot
 source=R/'B/distributed-14b-v1/pdb-ordinary-001/identity.before.json'
 raw0=find_native(json.loads(source.read_text()));assert raw0
 specs=[dict(name='TP2-forced-full-write-cache-'+str(v),retained=v) for v in (None,2100)]
 specs += [dict(name='TP2-partial-write',partial=True,error=True),dict(name='TP2-cancel-physical-write',cancel=True,error=True),dict(name='prewrite-plan-expiry',expired=True,prewrite=True),dict(name='confirmed-original-plan-expired',journal=.35,settle=.105,expiry=.05),dict(name='slowjournal350ms-physical105ms',journal=.35,settle=.105),dict(name='slowjournal-wrong-TP-member',journal=.35,settle=.105,wrong_second=True,error=True),dict(name='slowjournal-fresh-flag-off',journal=.35,settle=.105,fresh=False,error=True),dict(name='TP-membership-mismatch',membership=True,prewrite=True)]
 for k,v in [('active',None),('running',False),('waiting',1),('kv_allocations',None),('transfer_allocations',None),('transfer_send_failed',False),('acknowledged_generation',True)]:specs.append(dict(name='strict-native-'+k,patch={k:v},prewrite=True))
 results=[]
 with tempfile.TemporaryDirectory() as td:
  for case in specs:
   now=time.time();hw=Hardware(case);cl=ClockOwner(hw,[6,7],lock_dir=Path(td)/case['name'],max_service_frequency_mhz=2100)
   inst=InstanceState('CPU-tp2','mixed',2,(6,7),now,raw0['generation'],2100,49856,0,0,free_transfer_bytes=4294967296)
   state=StateManager(RuntimeSnapshot(1,now,(inst,)))
   backend=NS(clocks=cl,instances={'CPU-tp2':dict(tp=2,gpus=[6,7])},max_service_frequency_mhz=2100,last={},frequency={},parked={'CPU-tp2'},idle_since={})
   async def native(iid,path):
    raw=copy.deepcopy(raw0);raw.update(id=iid,timestamp=time.time(),transfer_observed_s=time.time());raw.update(case.get('patch',{}));return raw
   backend.json=native
   ctl=NS(backend=backend,state=state,action_lock=asyncio.Lock(),planner=NS(telemetry_ttl_s=2.),config=dict(max_service_frequency_mhz=2100,clock_failure_fresh_confirmation_v1=case.get('fresh',True)),max_service_frequency_mhz=2100)
   cl.state_lock=state.lock;cl.failure_fresh_confirmation=case.get('fresh',True);guards=[];events=[]
   def guard(g,f,r,b):guards.append([list(g),f,r]);return dict(allowed=True,CPU_fixture_only=True)
   cl.write_guard=guard
   def journal(e):
    events.append(copy.deepcopy(e))
    if case.get('journal') and e.get('kind')=='physical_clock_write' and e.get('stage')=='command_completed':time.sleep(case['journal'])
   cl.clock_event=journal
   if case.get('retained') is not None:cl.applied={6:case['retained'],7:case['retained']}
   if case.get('membership'):backend.instances['CPU-tp2']['gpus']=[6]
   plan=NS(expires_s=now+(-1 if case.get('expired') else case.get('expiry',10)))
   req=NS(request_id=case['name'],hard_deadline_s=now+120)
   obs=[dict(gpu=g,observed_mhz=2520,started_s=now,finished_s=now) for g in (6,7)]
   async def operation():
    async with ctl.action_lock,cl.lock,state.lock:return await reacquire(ctl,inst,plan,req,obs,[900,1500,2100])
   err=None;result=None
   task=asyncio.create_task(operation())
   if case.get('cancel'):
    while not hw.ready.is_set():await asyncio.sleep(.001)
    task.cancel()
   try:result=await task
   except (ClockWriteUncertain,ExpiredPlan) as e:err=e
   if case.get('prewrite'):
    assert err is not None and not hw.writes and not cl.physical_command_uncertainty,(case,repr(err))
   elif case.get('error'):
    assert isinstance(err,ClockWriteUncertain) and hw.writes and cl.physical_command_uncertainty,(case,repr(err))
    assert not any(e.get('kind')=='idle_domain_reacquisition' and e.get('confirmed') for e in events)
   else:
    assert err is None and result[2]==2100 and hw.writes==[[6,2100],[7,2100]],(case,repr(err))
    assert cl.applied=={6:2100,7:2100} and not cl.physical_command_uncertainty
    assert all(g[0]==[6,7] and g[1]==2100 for g in guards) and len(guards)==2
    event=next(e for e in events if e.get('kind')=='idle_domain_reacquisition')
    if case.get('journal'):
     assert event['confirmed_current_only'] is True and event['confirmed_before_original_settle_deadline'] is False
     final=event['final_confirmation'];assert final['settled_within_original_bound_proven'] is False and final['additional_physical_writes']==0 and final['deadline_extended'] is False
     assert [r['gpu'] for r in final['observations']]==[6,7]
    if case.get('expiry'):assert event['original_plan_expired'] is True and plan.expires_s<time.time()
   assert req.hard_deadline_s==now+120 and not ctl.action_lock.locked() and not state.lock.locked() and not cl.lock.locked()
   sticky=copy.deepcopy(cl.physical_command_uncertainty)
   await cl.close()
   assert hw.resets==[6,7] and not cl.pending_physical_commands and not cl.physical_command_uncertainty and all(f.closed for f in cl.files)
   results.append(dict(name=case['name'],passed=True,error=repr(err),physical_writes=hw.writes,fullTP_guards=guards,sticky_before_owned_close=sticky,owned_close_complete=True,events=events))
 receipt=dict(schema='P11-independent-TP2-CPU-v1',passed=True,tests=len(results),CPU_only=True,real_GPU_commands=False,actual_host=__import__('socket').gethostname(),source_manifest=ref(a.host/'manifest.json'),test_source=ref(__file__),native_shape_reference=ref(source),limitations=['Hardware and native GET are controlled CPU fixtures; TP2 performance qualification is not claimed.','Original physical eligibility planner is separately covered by C actual Controller tests.'],cases=results)
 with a.out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(dict(passed=True,tests=len(results),receipt=ref(a.out))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',type=Path,required=True);p.add_argument('--out',type=Path,required=True);asyncio.run(main(p.parse_args()))
