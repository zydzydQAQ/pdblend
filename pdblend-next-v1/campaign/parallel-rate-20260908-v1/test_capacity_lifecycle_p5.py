"""Lifecycle checks execute the new controller methods with bounded fake hardware."""
import asyncio,sys,time,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import os
HOST=Path(os.environ['PDB_TEST_HOST'])
sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
import ecopadg.serving.runtime as module
Controller=module.Controller

class Lifecycle(unittest.IsolatedAsyncioTestCase):
    def setup_controller(self):
        c=Controller.__new__(Controller);c.control_stop=asyncio.Event();c.slow_task=None;c.capacity_task=None;c.failure=None;c.journal=NS(emit=AsyncMock());c.role_task=None;c.slow_pending=[];c.action_lock=asyncio.Lock();c.backend=NS(last_action_finished_s=0)
        c.control_tick=AsyncMock(side_effect=[True,False]);return c
    async def test_current_transition_completes_before_quiescence_returns(self):
        c=self.setup_controller();started=asyncio.Event();finish=asyncio.Event();done=asyncio.Event()
        async def tick(now):started.set();await finish.wait();done.set()
        async def quiesce():await asyncio.shield(c.slow_task)
        c.capacity_service=NS(tick=tick,quiesce=quiesce);c.capacity_task=asyncio.create_task(c.capacity_control());await started.wait()
        q=asyncio.create_task(c.quiesce_controls());await asyncio.sleep(0);self.assertFalse(q.done());self.assertFalse(c.slow_task.cancelled())
        finish.set();await q;self.assertTrue(done.is_set());self.assertEqual(c.control_tick.await_count,2)
    async def test_timer_cancel_does_not_cancel_physical_transition(self):
        c=self.setup_controller();started=asyncio.Event();finish=asyncio.Event()
        async def tick(now):started.set();await finish.wait()
        c.capacity_service=NS(tick=tick);timer=asyncio.create_task(c.capacity_control());await started.wait();timer.cancel()
        with self.assertRaises(asyncio.CancelledError):await timer
        self.assertFalse(c.slow_task.done());finish.set();await c.slow_task
    async def test_measurement_cleanup_restores_initial_before_final_drains(self):
        c=self.setup_controller();c.request_tasks=set();c.active={};c.config={'telemetry_ttl_s':1};c.backend.instances={'old1':{},'old2':{},'new':{}};events=[]
        async def quiesce():events.append('quiesce');return None
        async def cleanup():events.append('cleanup');c.backend.instances.pop('new')
        async def rpc(i,path,payload=None):
            events.append(path+':'+i)
            return {'generation':0} if path=='/runtime' else {'drained':True,'accepting':False,'generation':1,'drain_proof_type':'synchronous_put_owner_barrier'}
        c.quiesce_controls=quiesce;c.capacity_service=NS(finish_to_initial=cleanup);c.backend.json=rpc
        old=module.engine_residual;module.engine_residual=lambda *a,**k:{}
        try:result=await c.finish_measurement(time.time()+2)
        finally:module.engine_residual=old
        self.assertTrue(result['drain_complete']);self.assertEqual(events[:2],['quiesce','cleanup']);self.assertEqual(set(result['drain_barriers']),{'old1','old2'});self.assertNotIn('/drain:new',events)
    async def test_cleanup_failure_is_preserved_and_cannot_claim_drain_complete(self):
        c=self.setup_controller();c.request_tasks=set();c.active={};c.config={};c.quiesce_controls=AsyncMock(return_value=None)
        c.capacity_service=NS(finish_to_initial=AsyncMock(side_effect=RuntimeError('owned removal incomplete')))
        result=await c.finish_measurement(time.time()+2);self.assertFalse(result['drain_complete']);self.assertIn('owned removal incomplete',result['error'])
    async def test_capacity_error_stops_new_ticks_and_records_failure(self):
        c=self.setup_controller();c.capacity_service=NS(tick=AsyncMock(side_effect=RuntimeError('unverified rollback')))
        await c.capacity_control();self.assertTrue(c.control_stop.is_set());self.assertIn('unverified rollback',c.failure);self.assertEqual(c.journal.emit.await_count,1)

if __name__=='__main__':unittest.main()
