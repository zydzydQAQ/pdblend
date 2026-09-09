"""CPU-only race between a real capacity lifecycle method and backend admission."""
import asyncio,importlib.util,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
HOST=Path(os.environ.get('A_CONTROLLER_UNDER_TEST','/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/hosts/14b-capacity-p5'))
sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.backend import HttpEngineBackend,ClockWriteUncertain
SOURCE=Path(os.environ['A_CAPACITY_BACKEND_UNDER_TEST'])
module_spec=importlib.util.spec_from_file_location('capacity_backend_under_test',SOURCE)
m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m)

class EndBeforeDocker(Exception):pass

@pytest.mark.parametrize('operation',['start','park_stopped'])
def test_normal_pending_capacity_write_serializes_before_admission(tmp_path,operation):
 async def check():
  started=asyncio.Event();finish=asyncio.Event();pending={};calls=[]
  async def write(*args,**kwargs):
   pending[139]='owned CPU future';started.set();await finish.wait();pending.clear()
  clocks=NS(set=write,park=write,epochs={5:1},pending_physical_commands=pending,physical_command_uncertainty=[])
  backend=HttpEngineBackend.__new__(HttpEngineBackend);backend.clocks=clocks;backend.inflight_actions=0;backend._execute=AsyncMock(side_effect=lambda _:calls.append('admitted'))
  controller=NS(backend=backend,config={'measured_frequency_write_guard_v1':True},action_lock=asyncio.Lock())
  cap=m.PinnedDockerBackend.__new__(m.PinnedDockerBackend);cap.controller=controller;cap.inputs={};cap.binding={'environment':{'PYTHONPATH':'cpu'},'engine_pythonpath':'cpu','image':'CPU-only','security_options':[]};cap.owner='cpu';cap.inventory=NS(event=lambda *a,**k:None)
  cap.engine_launch=lambda i,p,e:(['CPU-only'],e);cap.clock_ownership_proof=AsyncMock(return_value={'cpu':True});cap.command=AsyncMock(side_effect=EndBeforeDocker())
  i=dict(id='cap-cpu-1',gpus=[5],container_name='pdb-v2-cap-cpu-1',engine_config=str(tmp_path/'engine.json'),planned_config={'cpu':True})
  task=asyncio.create_task(cap.start(i,None,None) if operation=='start' else cap.park_stopped(i))
  await started.wait()
  async def admit():
   async with controller.action_lock:await backend.execute(NS(frequencies=[],roles=[],windows=[]))
  admission=asyncio.create_task(admit());await asyncio.sleep(0);await asyncio.sleep(0)
  serialized=not admission.done()
  finish.set();r=await asyncio.gather(task,admission,return_exceptions=True)
  assert serialized, 'normal capacity clock command escaped action_lock and rejected admission'
  assert r[1] is None and calls==['admitted']
  assert not controller.action_lock.locked() and not pending
 asyncio.run(check())

def test_sticky_uncertainty_still_rejects_admission():
 async def check():
  backend=HttpEngineBackend.__new__(HttpEngineBackend);backend.clocks=NS(pending_physical_commands={},physical_command_uncertainty=[{'cancelled_physical_write':True}]);backend._execute=AsyncMock()
  with pytest.raises(ClockWriteUncertain):await backend.execute(NS(frequencies=[],roles=[],windows=[]))
  backend._execute.assert_not_awaited()
 asyncio.run(check())
