"""CPU-only real v7 clock/guard and real profile/SLO planner regression."""
import asyncio,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import pytest
ROOT=Path(__file__).resolve().parent
HOST=ROOT/'hosts'/(os.environ.get('PDB_PARALLEL_MODEL','14b')+'-fixed-p1')
sys.path.insert(0,'/root/workspace/pdblend/.runtime-deps')
sys.path.insert(0,str(HOST/'src'))
from ecopadg.serving.backend import ClockOwner,ClockWriteUncertain
from ecopadg.serving.state import StateManager
from ecopadg.serving.types import InstanceState,RuntimeSnapshot,RequestBudget
from ecopadg.serving.profiles import ProfileStore
from ecopadg.serving.physical_frequency import evaluate
from ecopadg.serving.tails import admission_budget

def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
candidate=load('ecopadg.serving.startup_frequency',HOST/'src/ecopadg/serving/startup_frequency.py')
planner_module=load('ecopadg.serving.candidate_startup_planner',HOST/'src/ecopadg/serving/planner.py')
CONFIG=Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v7/C/fixed-release-001/configs/alpaca.json')
config=json.loads(CONFIG.read_text());profiles=ProfileStore.load(config['profiles'])

class Hardware:
 def __init__(self,observed=2520):self.observed=observed;self.writes=[]
 def set_clock(self,g,f):self.writes.append((g,f))
 def current_freq(self,g):return self.observed.get(g,2520) if isinstance(self.observed,dict) else self.observed
 def clock_idle(self,g):return False
 def reset_clock(self,g):pass

async def setup(tmp_path,*,observed=2520,frequency=2520,tp=1):
 now=time.time();gpus=(6,) if tp==1 else (6,7)
 instance=InstanceState('i','mixed',tp,gpus,now,1,frequency,100000,0,0)
 state=StateManager(RuntimeSnapshot(1,now,(instance,)))
 hardware=Hardware(observed);clocks=ClockOwner(hardware,gpus,lock_dir=str(tmp_path),settle_timeout_s=.001)
 clocks.applied={g:frequency for g in gpus};clocks.state_lock=state.lock
 planner=planner_module.JointPlanner(profiles,allow_pd=False,clock_settle_s=.3,frequency_costs=config['frequency_costs'],reserved_batch_guard=True,protect_pending_decode=True)
 raw=dict(timestamp=now,active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={})
 backend=SimpleNamespace(clocks=clocks,instances={'i':dict(id='i',gpus=list(gpus),tp=tp)},last={'i':raw})
 controller=SimpleNamespace(config=dict(observed_first_admission_frequency_v1=True,measured_frequency_write_guard_v1=True),strategy='pdblend-joint',state=state,planner=planner,backend=backend,action_lock=asyncio.Lock(),frequency_pending_budgets=lambda:())
 clocks.write_guard=lambda g,f,r,b=None:evaluate(controller,g,f,r,b)
 return controller,clocks,hardware

def request(now):return RequestBudget('r',now,18,129,1.,.1,129,hard_deadline_s=now+120)
def plans(planner,snapshot,req):return planner.candidates(snapshot,req,time.time())

@pytest.mark.parametrize('frequency',[2100,2520])
def test_fresh_observed_frequency_is_used_without_low_startup_write(tmp_path,frequency):
 async def run():
  c,clocks,hw=await setup(tmp_path,observed=frequency,frequency=frequency)
  try:
   p=await candidate.admission_planner(c,c.state.snapshot);req=request(time.time());choices=plans(p,c.state.snapshot,req)
   assert choices and all(a.frequency_mhz==frequency for plan in choices for a in plan.frequencies)
   plan=min(choices,key=lambda x:x.routes[0].incremental_j);await c.state.reserve(plan,time.time(),admission_budget(plan,req,now=time.time()))
   await clocks.set((6,),frequency);assert not hw.writes
   assert evaluate(c,(6,),frequency,'requested clock action')['allowed']
  finally:await clocks.close()
 asyncio.run(run())

def test_original_real_v7_failure_reproduced_and_no_safe_retry(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path);req=request(time.time());choices=plans(c.planner,c.state.snapshot,req)
  low=next(p for p in choices if p.frequencies[0].frequency_mhz==2100)
  try:
   await c.state.reserve(low,time.time(),admission_budget(low,req,now=time.time()))
   assert c.state.snapshot.instances[0].waiting==1 and c.backend.last['i']['waiting']==0
   with pytest.raises(ClockWriteUncertain,match='outstanding first-token'):await clocks.set((6,),2100)
   assert hw.writes==[(6,2100)] and clocks.coverage_limits[-1]['safely_replannable'] is False
   candidate.failed(c,low,ClockWriteUncertain('unknown'))
   await c.state.release('r',unissued=True);clocks.transaction_writes=[];clocks.applied[6]=2520
   p=await candidate.admission_planner(c,c.state.snapshot);assert not plans(p,c.state.snapshot,req)
  finally:await clocks.close()
 asyncio.run(run())

@pytest.mark.parametrize('bad',['observed','partial_tp','deferred','unfinished_write','stale','native','missing_command'])
def test_unconfirmed_startup_blocks_without_any_write(tmp_path,bad):
 async def run():
  c,clocks,hw=await setup(tmp_path,tp=2 if bad=='partial_tp' else 1)
  try:
   if bad=='observed':hw.observed=2100
   if bad=='partial_tp':hw.observed={6:2520,7:1500}
   if bad=='deferred':clocks.deferred[6]=dict(target=2520)
   if bad=='unfinished_write':clocks.transaction_writes=[dict(completed=False)]
   if bad=='stale':c.backend.last['i']['timestamp']-=2
   if bad=='native':c.backend.last['i']['active']=1
   if bad=='missing_command':clocks.applied.pop(6)
   p=await candidate.admission_planner(c,c.state.snapshot)
   assert p.startup_frequency_restrictions=={'i':None} and not hw.writes
   assert not plans(p,c.state.snapshot,request(time.time()))
  finally:await clocks.close()
 asyncio.run(run())

def test_completed_startup_and_nonempty_work_use_original_planner(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path)
  try:
   req=request(time.time());choices=plans(c.planner,c.state.snapshot,req);plan=choices[0]
   candidate.committed(c,plan);assert await candidate.admission_planner(c,c.state.snapshot) is c.planner
   c.startup_admission_used=set();i=c.state.snapshot.instances[0];warm=replace(i,requests=(replace(req,emitted=1,first_token_s=time.time()),),running=1)
   snap=replace(c.state.snapshot,instances=(warm,));c.state.snapshot=snap
   assert await candidate.admission_planner(c,snap) is c.planner
  finally:await clocks.close()
 asyncio.run(run())

def test_actual_profile_and_slo_guards_remain(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path)
  try:
   p=await candidate.admission_planner(c,c.state.snapshot);req=request(time.time())
   assert not plans(p,c.state.snapshot,replace(req,ttft_s=.000001))
   p.profiles=ProfileStore(())
   assert not plans(p,c.state.snapshot,req)
  finally:await clocks.close()
 asyncio.run(run())

def test_real_idle_pstate_keeps_existing_guarded_wakeup_not_300mhz(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path,observed=300)
  hw.clock_idle=lambda gpu:True
  clocks.deferred[6]=dict(target=2520)
  try:
   p=await candidate.admission_planner(c,c.state.snapshot)
   assert p.startup_frequency_restrictions=={} and not hw.writes
   before=plans(c.planner,c.state.snapshot,request(time.time()));after=plans(p,c.state.snapshot,request(time.time()))
   assert {a.frequency_mhz for v in before for a in v.frequencies}=={a.frequency_mhz for v in after for a in v.frequencies}
   assert all(a.frequency_mhz!=300 for v in after for a in v.frequencies)
  finally:await clocks.close()
 asyncio.run(run())

def test_actual_post_first_token_dvfs_can_still_lower_frequency(tmp_path):
 from ecopadg.serving.frequency import FrequencyPlanner,FrequencyCost
 from ecopadg.serving.pending_frequency import frequency_plan
 async def run():
  c,clocks,hw=await setup(tmp_path)
  try:
   now=time.time();req=replace(request(now),emitted=32,first_token_s=now-.2,last_token_s=now-.01)
   instance=replace(c.state.snapshot.instances[0],requests=(req,),running=1,timestamp_s=now)
   snapshot=replace(c.state.snapshot,instances=(instance,),timestamp_s=now)
   engine=FrequencyPlanner(c.planner,[FrequencyCost(**cost) for cost in config['frequency_costs']])
   plan=frequency_plan(engine,snapshot,now,(),(),True)
   assert plan.frequencies and any(a.frequency_mhz<2520 for a in plan.frequencies)
  finally:await clocks.close()
 asyncio.run(run())

def test_unknown_startup_cannot_get_a_blind_nonfeasible_fallback_write(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path,observed=2100)
  try:
   p=await candidate.admission_planner(c,c.state.snapshot);req=request(time.time())
   plan=p.plan(c.state.snapshot,(req,),now=time.time())
   assert not plan.feasible and plan.frequencies==() and not hw.writes
  finally:await clocks.close()
 asyncio.run(run())

def test_default_off_does_not_observe_or_change_original_policy(tmp_path):
 async def run():
  c,clocks,hw=await setup(tmp_path)
  try:
   c.config['observed_first_admission_frequency_v1']=False
   hw.current_freq=lambda gpu:(_ for _ in ()).throw(AssertionError('unexpected observation'))
   assert await candidate.admission_planner(c,c.state.snapshot) is c.planner
  finally:await clocks.close()
 asyncio.run(run())

@pytest.mark.parametrize('kind',['pending','uncertain','both'])
def test_composed_startup_never_reads_after_unresolved_physical_write(tmp_path,kind):
 async def run():
  c,clocks,hw=await setup(tmp_path,tp=2)
  future=asyncio.get_running_loop().create_future()
  reads=[]
  def read(gpu):
   reads.append(gpu)
   raise AssertionError('physical read must not queue behind an unresolved write')
  hw.current_freq=read
  if kind in ('pending','both'):clocks.pending_physical_commands={1:future}
  if kind in ('uncertain','both'):clocks.physical_command_uncertainty=[dict(physical_state_unknown=True)]
  # Per-call accounting can be empty while the persistent barrier remains.
  clocks.transaction_writes=[]
  try:
   with pytest.raises(ClockWriteUncertain,match='startup observation blocked'):
    await asyncio.wait_for(candidate.admission_planner(c,c.state.snapshot),.5)
   assert reads==[] and hw.writes==[]
   assert clocks.pending_physical_commands or clocks.physical_command_uncertainty
  finally:
   future.set_result(None)
   await clocks.close()
 asyncio.run(run())

def test_each_model_controller_accepts_explicit_composed_feature():
 from ecopadg.serving.runtime import Controller
 model=os.environ.get('PDB_PARALLEL_MODEL','14b')
 letter={'7b':'C','14b':'A','32b':'B'}[model]
 path=ROOT.parent/'main-slo-improvement-v7'/letter/'fixed-release-001/configs/alpaca.json'
 cfg=json.loads(path.read_text())
 cfg['observed_first_admission_frequency_v1']=True
 cfg['measured_frequency_write_guard_v1']=True
 c=Controller(cfg)
 try:
  assert c.config['observed_first_admission_frequency_v1'] is True
  assert c.planner.allow_pd is False
 finally:
  c.planning_executor._executor.shutdown(wait=True)
