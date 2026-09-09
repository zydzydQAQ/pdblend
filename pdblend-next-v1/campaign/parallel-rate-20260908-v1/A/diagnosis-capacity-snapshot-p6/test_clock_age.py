"""CPU reproduction of actual P6 tick's pre-snapshot clock rejecting fresh spare.

Uses the measured certificate with a fake device snapshot/physical boundary. It
performs no GPU operation and does not create calibration or serving evidence.
"""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
A=Path(__file__).resolve().parent.parent
HOST=A.parent/'hosts/14b-capacity-p6'
sys.path[:0]=[str(HOST),str(HOST/'src'),'/root/workspace/pdblend/.runtime-deps']
import capacity_runtime as runtime
CAP=json.loads((A/'p6-qualification900-inputs-002/capacity-binding.json').read_text())
M=runtime.load_planner(CAP['planner_source'])
PLANNER=runtime.calibration_model(M,CAP)
class AtPhysicalBoundary(Exception):pass

def service(stale=False,future=False,busy=False):
 s=runtime.CapacityService.__new__(runtime.CapacityService)
 s.stopping=False;s.tick_lock=asyncio.Lock();s.planner=PLANNER;s.module=M;s.identity=PLANNER.identity
 s.binding=CAP;s.state=M.State();s.current_task=None;s.test_events=[];s.test_clock={}
 s.inventory=SimpleNamespace(event=lambda kind,**v:s.test_events.append(dict(kind=kind,**v)))
 async def snapshot():
  await asyncio.sleep(.01)
  at=time.time()-(2 if stale else 0)+(2 if future else 0);s.test_clock['spare_observed_at_s']=at
  rs=tuple(M.Resident('nextv3a'+str(g),(g,),at,1,at-400,active_requests=20,removable=False) for g in (6,7))
  return M.Snapshot(s.identity,1,rs,(M.Spare((5,),at,at-60,45_000_000_000,gpu_processes=int(busy)),))
 s.snapshot=snapshot
 s.demand=lambda now:M.Demand(now,400,10,12,CAP['demand_domain']['sha256'],100,70)
 def startup(snapshot,demand,now):
  s.test_clock['decision_at_s']=now
  return dict(cpu_fixture=True)
 s.startup_risk=startup
 async def execute(proposal):
  s.test_proposal=proposal
  raise AtPhysicalBoundary('CPU fake physical boundary; no device call')
 s.executor=SimpleNamespace(cleanup_reserve_s=90,execute=execute)
 return s

def test_actual_supplied_clock_rejects_newly_observed_free_gpu():
 async def run():
  s=service();before=time.time();r=await s.tick(before)
  assert r.reason=='no_verified_free_gpu_restore' and r.proposal is None
  assert s.test_clock['decision_at_s']<s.test_clock['spare_observed_at_s']
 asyncio.run(run())

def test_original_optional_fresh_clock_path_reaches_qualified_restore():
 async def run():
  s=service()
  with pytest.raises(AtPhysicalBoundary):await s.tick()
  assert s.test_proposal.action=='restore' and s.test_proposal.gpus==(5,)
  assert s.test_clock['decision_at_s']>=s.test_clock['spare_observed_at_s']
 asyncio.run(run())

def test_genuinely_stale_spare_is_still_rejected():
 async def run():
  s=service(stale=True);r=await s.tick()
  assert r.reason=='no_verified_free_gpu_restore' and r.proposal is None
 asyncio.run(run())

@pytest.mark.parametrize('variant',[dict(future=True),dict(busy=True)])
def test_actual_future_timestamp_and_busy_gpu_still_rejected(variant):
 async def run():
  s=service(**variant);r=await s.tick()
  assert r.reason=='no_verified_free_gpu_restore' and r.proposal is None
 asyncio.run(run())

def test_unknown_memory_never_becomes_usable_spare():
 now=time.time()
 with pytest.raises(ValueError):M.Spare((5,),now,now-60,None)
