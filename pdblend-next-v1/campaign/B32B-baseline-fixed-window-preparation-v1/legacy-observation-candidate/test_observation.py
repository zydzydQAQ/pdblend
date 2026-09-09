"""Actual legacy EngineService methods with CPU owner/rank stand-ins only."""
import ast
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('legacy_observation_engine',ROOT/'engine.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)


class Observation(unittest.TestCase):
    def service(self,tp=2):
        obj=engine.EngineService.__new__(engine.EngineService)
        obj.config=dict(id='cpu-only',tp=tp,peers={'peer':{}},verify_transport=False)
        obj.state=dict(generation=3,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        obj.streams={};obj.accepting=True;obj.error=None;obj.control_lock=asyncio.Lock()
        obj.transfer_snapshot={};obj.transfer_snapshot_s=0;obj.transfer_bytes_per_token=16
        blocks=SimpleNamespace(get_num_free_gpu_blocks=lambda:10)
        scheduler=SimpleNamespace(block_manager=blocks,running=[],waiting=[],_pdblend_runtime=dict(generation=3))
        obj.engine=SimpleNamespace(scheduler=[scheduler],cache_config=SimpleNamespace(block_size=16,num_gpu_blocks=20))
        obj.rank_states=[dict(buffer_capacity_bytes=4096,allocations={},buffered_tensors=0,inflight_receives=0,listener_alive=True) for _ in range(tp)]
        obj.control_rpc=lambda method:copy.deepcopy(obj.rank_states)
        async def call(method,*args):return method(*args)
        obj.call=call
        obj.commit_state=lambda value:setattr(obj,'state',value)
        request=SimpleNamespace(json=self.payload)
        return obj,request

    async def payload(self):return dict(expected_generation=3)

    def module_patch(self):
        return patch.dict(sys.modules,{'vllm':SimpleNamespace(),'vllm.pdblend_runtime':SimpleNamespace(transfer_state=object())})

    def test_snapshot_retains_unknown_sends_and_actual_timestamp(self):
        obj,_=self.service()
        with self.module_patch():obj.update_snapshot()
        self.assertEqual(obj.snapshot['transfer_buffered_tensors'],0)
        self.assertEqual(obj.snapshot['transfer_inflight_receives'],0)
        self.assertIsNone(obj.snapshot['transfer_inflight_sends'])
        self.assertFalse(obj.snapshot['transfer_inflight_sends_observed'])
        self.assertLess(abs(time.time()-obj.snapshot['transfer_observed_s']),1)

    def test_snapshot_records_actual_nonzero_rank_residual(self):
        obj,_=self.service();obj.rank_states[1].update(buffered_tensors=2,inflight_receives=3)
        with self.module_patch():obj.update_snapshot()
        self.assertEqual(obj.snapshot['transfer_buffered_tensors'],2)
        self.assertEqual(obj.snapshot['transfer_inflight_receives'],3)

    def test_snapshot_missing_tp_rank_rejected(self):
        obj,_=self.service();obj.rank_states.pop()
        with self.module_patch(),self.assertRaisesRegex(RuntimeError,'TP-rank transfer snapshot'):obj.update_snapshot()

    def test_actual_tp2_drain_returns_two_rank_states(self):
        obj,request=self.service()
        with self.module_patch():result=json.loads(asyncio.run(obj.drain(request)).body)
        self.assertEqual(len(result['transfers']),2)
        self.assertEqual(result['generation'],4)
        self.assertEqual(result['drain_proof_type'],'synchronous_put_owner_barrier')
        self.assertTrue(result['drained']);self.assertFalse(result['accepting'])
        self.assertFalse(obj.state['admit_prefill'])

    def test_actual_tp1_drain_returns_one_rank(self):
        obj,request=self.service(1)
        with self.module_patch():result=json.loads(asyncio.run(obj.drain(request)).body)
        self.assertEqual(len(result['transfers']),1)

    def test_drain_missing_rank_rejected_and_resume_accepting(self):
        obj,request=self.service();obj.rank_states.pop()
        with self.module_patch(),self.assertRaisesRegex(RuntimeError,'TP-rank drain barrier'):asyncio.run(obj.drain(request))
        self.assertTrue(obj.accepting);self.assertEqual(obj.state['generation'],3)

    def test_drain_real_pending_transfer_rejected(self):
        obj,request=self.service();obj.rank_states[1]['inflight_receives']=1
        with self.module_patch(),self.assertRaisesRegex(RuntimeError,'transport did not drain'):asyncio.run(obj.drain(request))
        self.assertTrue(obj.accepting)

    def test_only_observation_methods_differ(self):
        old=ast.parse(Path('/root/workspace/pdblend/src/ecopadg/serving/engine.py').read_text())
        new=ast.parse((ROOT/'engine.py').read_text())
        def methods(tree):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineService')
            return {n.name:ast.dump(n) for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        a,b=methods(old),methods(new);self.assertEqual(set(a),set(b))
        self.assertEqual({k for k in a if a[k]!=b[k]},{'update_snapshot','drain'})


if __name__=='__main__':unittest.main()
