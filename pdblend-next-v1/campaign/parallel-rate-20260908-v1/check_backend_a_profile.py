"""Real v6/v7 writer and StateManager, mocked two-GPU hardware; no GPU operations."""
import argparse,asyncio,hashlib,json,sys,tempfile,threading,time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
P=argparse.ArgumentParser();P.add_argument('--host',required=True,type=Path);P.add_argument('--out',required=True,type=Path);A=P.parse_args()
HOST=A.host.resolve();sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.backend import ClockOwner,HttpEngineBackend,ClockWriteUncertain
from ecopadg.serving.runtime import Controller
from ecopadg.serving.state import StateManager
from ecopadg.serving.types import RequestBudget,InstanceState,RuntimeSnapshot,ControlPlan
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
CFG=REPO/'campaign/main-slo-improvement-v7/A/fixed-release-001/configs/alpaca.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FILES={str(HOST/n):h for n,h in json.loads((HOST/'manifest.json').read_text())['files'].items()};FILES[str(CFG)]=sha(CFG);FILES[str(Path(__file__).resolve())]=sha(__file__)
assert all(sha(p)==h for p,h in FILES.items())
V7=True

async def case(kind):
 cfg=json.loads(CFG.read_text());c=Controller(cfg);now=time.time()
 requests=tuple(RequestBudget(f'r{i}',now-.1,128,256,1.,.1,emitted=16,first_token_s=now-.1,hard_deadline_s=now+120) for i in range(14))
 instance=InstanceState('nextv3b0','mixed',2,(0,1),now,0,2520,47104,14,0,requests=requests)
 c.state=StateManager(RuntimeSnapshot(1,now,(instance,)))
 c.state.reservations['r0']=NS(decode_id='nextv3b0',prefill_id='nextv3b0',reserve_tokens=1,transfer_reserve_bytes=0)
 c.backend=NS(instances={'nextv3b0':dict(id='nextv3b0',gpus=[0,1],tp=2)},last={'nextv3b0':dict(active=14)})
 loop=asyncio.get_running_loop();entered=asyncio.Event();proceed=threading.Event();completed=threading.Event();physical={0:2520,1:2520};writes=[]
 def hardware(g,f):
  writes.append((g,f))
  if (kind=='release_race' and g==0) or (kind=='cancel_second' and g==1):
   loop.call_soon_threadsafe(entered.set)
   assert proceed.wait(2),'CPU test hardware release timed out'
  if kind=='second_hardware_error' and g==1:raise RuntimeError('simulated second physical board error')
  physical[g]=f
  if kind=='cancel_second' and g==1:completed.set()
 with tempfile.TemporaryDirectory(prefix='b-clock-cpu-') as tmp:
  reset_calls=[]
  x=ClockOwner(NS(set_clock=hardware,current_freq=lambda g:physical[g],clock_idle=lambda g:False,reset_clock=lambda g:reset_calls.append(g)),(0,1),lock_dir=tmp)
  x.applied={0:2520,1:2520};x.write_guard=c.clock_write_allowed
  if V7:x.state_lock=c.state.lock;x.write_guard_journal=str(Path(tmp)/'journal.jsonl')
  task=asyncio.create_task(x.set((0,1),1500));release=None;record=dict(kind=kind)
  try:
   if kind=='release_race':
    await asyncio.wait_for(entered.wait(),1)
    release=asyncio.create_task(c.state.release('r0'));await asyncio.sleep(.01)
    record['release_blocked_by_full_tp_transaction']=not release.done();proceed.set()
   if kind=='cancel_second':
    await asyncio.wait_for(entered.wait(),1);task.cancel()
   try:await task;record['outcome']='completed'
   except BaseException as e:record.update(outcome=type(e).__name__,error=str(e))
   if release:await release
   record.update(writes=writes.copy(),applied_after_task=dict(x.applied),physical_after_task=dict(physical),
                 state_native=c.state.snapshot.instances[0].running,ledger_after_task=len(c.state.snapshot.instances[0].requests))
   if kind=='cancel_second' and V7:
    record['physical_worker_still_running_after_task_exit']=not completed.is_set()
    i=c.state.snapshot.instances[0];c.state.snapshot=replace(c.state.snapshot,instances=(replace(i,timestamp_s=time.time()-5),))
    try:await x.set((0,1),1500);record['following_stale_attempt']='completed'
    except BaseException as e:record['following_stale_attempt']=type(e).__name__
    record['persistent_uncertainty_count']=len(x.physical_command_uncertainty)
    record['pending_future_count_before_close']=len(x.pending_physical_commands)
    b=HttpEngineBackend([dict(id='nextv3b0',gpus=[0,1],tp=2)],None,x)
    plan=ControlPlan(1,time.time(),time.time()+10)
    try:await b.execute(plan);record['route_only_next_plan']='accepted'
    except BaseException as e:record['route_only_next_plan']=type(e).__name__
    record['route_only_confirmation']=await b.confirm(plan)
    closing=asyncio.create_task(x.close());await asyncio.sleep(.01)
    record['close_waited_for_actual_worker']=not closing.done()
    record['latch_kept_until_worker_done']=bool(x.physical_command_uncertainty)
    proceed.set();await closing
    record['pending_after_successful_close']=len(x.pending_physical_commands)
    record['uncertainty_after_successful_close']=len(x.physical_command_uncertainty)
    record['applied_after_successful_close']=dict(x.applied)
    record['reset_calls']=reset_calls.copy()
   if V7:record['journal']=[json.loads(line) for line in Path(x.write_guard_journal).read_text().splitlines()]
   print(json.dumps(dict(case_record=record)),flush=True)
   if kind=='normal':assert record['outcome']=='completed' and physical==x.applied=={0:1500,1:1500}
   if kind=='release_race':
    assert record['state_native']==14 and record['ledger_after_task']==13
    if V7:assert record['outcome']=='completed' and record['release_blocked_by_full_tp_transaction'] and physical=={0:1500,1:1500}
    else:assert record['outcome']=='RuntimeError' and not record['release_blocked_by_full_tp_transaction'] and physical=={0:1500,1:2520}
   if kind=='second_hardware_error' and V7:assert record['outcome']=='ClockWriteUncertain' and record['journal'][-1]['physical_state_unknown'] and physical=={0:1500,1:2520}
   if kind=='cancel_second' and V7:
    assert record['outcome']=='ClockWriteUncertain' and record['physical_worker_still_running_after_task_exit']
    assert record['following_stale_attempt']=='ClockWriteUncertain'
    assert record['persistent_uncertainty_count'] and record['pending_future_count_before_close']==1
    assert record['route_only_next_plan']=='ClockWriteUncertain' and record['route_only_confirmation'] is False
    assert record['close_waited_for_actual_worker'] and record['latch_kept_until_worker_done']
    assert not record['pending_after_successful_close'] and not record['uncertainty_after_successful_close'] and not record['applied_after_successful_close']
    assert record['reset_calls']==[0,1]
   return record
  finally:
   proceed.set()
   await asyncio.to_thread(x.pool.shutdown,wait=True)
   for f in x.files:f.close()
   c.planning_executor._executor.shutdown(wait=True)

async def main():
 kinds=['normal','release_race']+(['second_hardware_error','cancel_second'] if V7 else [])
 records=[await case(k) for k in kinds]
 assert all(sha(p)==h for p,h in FILES.items())
 result=dict(schema=1,host=str(HOST),source_sha256=FILES,cpu_cases=records,hardware_mocked=True,gpu_executed=False,
  real_StateManager_and_ClockOwner=True,actual_A_profile_loaded=True,performance_claim=False,
  v7_fixes_observed_release_race=V7,persistent_cancellation_uncertainty_fixed=True)
 with A.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps([dict(kind=r['kind'],outcome=r['outcome'],writes=r['writes'],following_stale_attempt=r.get('following_stale_attempt'),release_blocked=r.get('release_blocked_by_full_tp_transaction')) for r in records]))
asyncio.run(main())
