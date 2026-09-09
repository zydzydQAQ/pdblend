"""Independent p2 empty-admission checks with the actual A profile."""
import asyncio
import json
import importlib.util
import os
import sys
import time
import pytest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO=Path('/root/workspace/pdblend-next-v1')
HOST=Path(os.environ['PDB_A_P2_HOST'])
sys.path[:0]=[str(HOST/'src'),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.backend import ClockOwner,ClockWriteUncertain
from ecopadg.serving.state import StateManager
from ecopadg.serving.types import InstanceState,RuntimeSnapshot,RequestBudget
from ecopadg.serving.profiles import ProfileStore
from ecopadg.serving.planner import JointPlanner
if os.environ.get('PDB_A_P2_IDLE_HELPER'):
    _spec=importlib.util.spec_from_file_location('ecopadg.serving.idle_admission',os.environ['PDB_A_P2_IDLE_HELPER'])
    _module=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_module;_spec.loader.exec_module(_module)
from ecopadg.serving.idle_admission import current_clock_first_plan
from ecopadg.serving.physical_frequency import evaluate
CONFIG=json.loads((REPO/'campaign/main-slo-improvement-v7/A/fixed-release-001/configs/alpaca.json').read_text())
PROFILES=ProfileStore.load(CONFIG['profiles'])

class Hardware:
    def __init__(self):self.writes=[];self.observed=2520;self.idle=False
    def set_clock(self,g,f):self.writes.append((g,f))
    def current_freq(self,g):return self.observed
    def clock_idle(self,g):return self.idle
    def reset_clock(self,g):pass

def setup(tmp_path):
    now=time.time();instance=InstanceState('i','mixed',1,(6,),now,1,2520,100000,0,0)
    state=StateManager(RuntimeSnapshot(1,now,(instance,)))
    hw=Hardware();clocks=ClockOwner(hw,(6,),lock_dir=str(tmp_path),settle_timeout_s=.001)
    clocks.applied={6:2520};clocks.state_lock=state.lock
    planner=JointPlanner(PROFILES,allow_pd=False,clock_settle_s=.3,frequency_costs=CONFIG['frequency_costs'],reserved_batch_guard=True,protect_pending_decode=True)
    raw=dict(timestamp=now,active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={})
    c=SimpleNamespace(config=dict(observed_first_admission_frequency_v1=False,observed_idle_admission_frequency_v2=True,measured_frequency_write_guard_v1=True),strategy='pdblend-joint',state=state,planner=planner,backend=SimpleNamespace(clocks=clocks,instances={'i':dict(id='i',gpus=[6],tp=1)},last={'i':raw}),action_lock=asyncio.Lock(),frequency_pending_budgets=lambda:())
    clocks.write_guard=lambda g,f,r,b=None:evaluate(c,g,f,r,b)
    return c,clocks,hw

def req():
    now=time.time();return RequestBudget('r',now,18,129,1.,.1,129,hard_deadline_s=now+120)

def original_low(c, request):
    choices=c.planner.candidates(c.state.snapshot,request,time.time())
    return min(choices,key=lambda p:p.frequencies[0].frequency_mhz)

def test_used_empty_replans_at_actual_covered_clock(tmp_path):
    async def run():
        c,clock,hw=setup(tmp_path)
        try:
            c.startup_admission_used={'i'};r=req();original=original_low(c,r)
            assert original.frequencies[0].frequency_mhz<2520
            async with c.action_lock:chosen=await current_clock_first_plan(c,original,r)
            assert chosen.frequencies[0].frequency_mhz==2520 and hw.writes==[]
        finally:await clock.close()
    asyncio.run(run())

def test_parked_and_uncommanded_empty_replans_from_actual_2520(tmp_path):
    async def run():
        c,clock,hw=setup(tmp_path)
        try:
            c.startup_admission_used={'i'}
            c.state.snapshot=replace(c.state.snapshot,instances=(replace(c.state.snapshot.instances[0],parked=True),))
            clock.applied={};r=req();original=original_low(c,r)
            async with c.action_lock:chosen=await current_clock_first_plan(c,original,r)
            assert chosen.feasible and chosen.frequencies[0].frequency_mhz==2520 and hw.writes==[]
        finally:await clock.close()
    asyncio.run(run())

def test_nonempty_existing_decode_preserves_original_plan(tmp_path):
    async def run():
        c,clock,hw=setup(tmp_path)
        try:
            r=req();original=original_low(c,r)
            existing=replace(req(),request_id='prior',emitted=32,first_token_s=time.time()-.2,last_token_s=time.time()-.01)
            c.state.snapshot=replace(c.state.snapshot,instances=(replace(c.state.snapshot.instances[0],requests=(existing,),running=1),))
            hw.current_freq=lambda gpu:(_ for _ in ()).throw(AssertionError('nonempty path must not read/replace frequency'))
            async with c.action_lock:chosen=await current_clock_first_plan(c,original,r)
            assert chosen is original and hw.writes==[]
        finally:await clock.close()
    asyncio.run(run())

def test_genuine_parked_idle_300_preserves_original_guarded_wakeup_target(tmp_path):
    async def run():
        c,clock,hw=setup(tmp_path)
        try:
            c.state.snapshot=replace(c.state.snapshot,instances=(replace(c.state.snapshot.instances[0],parked=True),))
            hw.observed=300;hw.idle=True;clock.applied={};r=req();original=original_low(c,r)
            async with c.action_lock:chosen=await current_clock_first_plan(c,original,r)
            assert chosen.feasible and chosen.frequencies[0].frequency_mhz==original.frequencies[0].frequency_mhz
            assert chosen.frequencies[0].frequency_mhz!=300 and 'deferred wakeup' in chosen.reason and hw.writes==[]
        finally:await clock.close()
    asyncio.run(run())

@pytest.mark.parametrize('kind',['pending','uncertain','both'])
def test_persistent_unknown_physical_state_blocks_before_read(tmp_path,kind):
    async def run():
        c,clock,hw=setup(tmp_path)
        r=req();original=original_low(c,r);future=asyncio.get_running_loop().create_future()
        hw.current_freq=lambda gpu:(_ for _ in ()).throw(AssertionError('unsafe queued frequency read'))
        if kind in ('pending','both'):clock.pending_physical_commands={1:future}
        if kind in ('uncertain','both'):clock.physical_command_uncertainty=[dict(physical_state_unknown=True)]
        try:
            async with c.action_lock:
                with pytest.raises(ClockWriteUncertain,match='owned close'):
                    await current_clock_first_plan(c,original,r)
            assert hw.writes==[]
        finally:
            future.set_result(None)
            await clock.close()
    asyncio.run(run())

def test_v2_default_off_preserves_original_plan_without_observation(tmp_path):
    async def run():
        c,clock,hw=setup(tmp_path)
        r=req();original=original_low(c,r)
        c.config.pop('observed_idle_admission_frequency_v2')
        hw.current_freq=lambda gpu:(_ for _ in ()).throw(AssertionError('default off must not observe'))
        try:
            async with c.action_lock:chosen=await current_clock_first_plan(c,original,r)
            assert chosen is original and hw.writes==[]
        finally:await clock.close()
    asyncio.run(run())
