"""CPU safety boundaries; no network, CUDA, NVML or subprocess execution."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

SPEC=importlib.util.spec_from_file_location('budget_outer_gate',Path(__file__).with_name('run.py'))
gate=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(gate)


def fixture():
    old=json.loads((gate.OLD/'validation-status.json').read_text())
    row=old['final_cleanup']['33501']
    before=dict(row['restored'],generation=row['drain']['generation']-1)
    return before,json.loads(json.dumps(row['drain']))


def bare():
    obj=object.__new__(gate.Gate)
    obj.state={};obj.deadline=None;obj.child=None;obj.clocks=None;obj.verified=True
    return obj


class TestCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_valid_native_proof_and_resume(self):
        obj=bare();before,proof=fixture();calls=[]
        async def settled(*a,**kw):return before
        async def http(*a,**kw):return proof
        async def resume(port):calls.append(port);return {'accepting':True}
        obj.settled_idle=settled;obj.http=http;obj.resume=resume
        await obj.restore_one(33501)
        self.assertEqual(calls,[33501]);self.assertNotIn('proof_error',obj.state['cleanup']['33501'])

    async def test_failed_proof_still_resumes_but_raises(self):
        obj=bare();before,proof=fixture();proof['send_counters_verified']=False;calls=[]
        async def settled(*a,**kw):return before
        async def http(*a,**kw):return proof
        async def resume(port):calls.append(port);return {'accepting':True}
        obj.settled_idle=settled;obj.http=http;obj.resume=resume
        with self.assertRaises(RuntimeError):await obj.restore_one(33501)
        self.assertEqual(calls,[33501]);self.assertIn('proof_error',obj.state['cleanup']['33501'])

    async def test_pending_timeout_still_attempts_resume(self):
        obj=bare();calls=[]
        async def settled(*a,**kw):raise TimeoutError('pending budget')
        async def resume(port):calls.append(port);return {}
        obj.settled_idle=settled;obj.resume=resume
        with self.assertRaises(TimeoutError):await obj.restore_one(33501)
        self.assertEqual(calls,[33501])

    async def test_unverified_identity_never_controls(self):
        obj=bare();obj.verified=False;calls=[]
        async def stop():calls.append('stop-self-only')
        async def forbidden(*a,**kw):raise AssertionError('control before verified identity')
        obj.stop_child=stop;obj.restore_one=forbidden;obj.http=forbidden
        await obj.cleanup()
        self.assertEqual(calls,['stop-self-only'])

    async def test_inner_failure_and_one_outer_failure_cannot_skip_other_owner(self):
        obj=bare();calls=[];obj.state['validator_passed']=False
        async def stop():calls.append('stop')
        async def restore(port):
            calls.append(port)
            if port==33500:raise RuntimeError('proof failure')
        obj.stop_child=stop;obj.owned_ids=lambda:set();obj.restore_one=restore
        await obj.cleanup()
        self.assertEqual(set(calls),{'stop',33500,33501});self.assertFalse(obj.state['cleanup_complete'])

    async def test_clocks_released_even_when_restore_fails(self):
        obj=bare();calls=[]
        class Clock:
            async def close(self):calls.append('clock-close')
        async def stop():pass
        async def restore(port):raise RuntimeError('proof failure')
        obj.stop_child=stop;obj.owned_ids=lambda:set();obj.restore_one=restore;obj.clocks=Clock()
        await obj.cleanup()
        self.assertEqual(calls,['clock-close']);self.assertLess(obj.state['cleanup_elapsed_s'],90)

    def test_deadline_caps_http_time_and_refuses_after_expiry(self):
        obj=bare();obj.deadline=time.monotonic()+.2
        self.assertLessEqual(obj.remaining(10),.2)
        obj.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):obj.remaining(10)

    def test_partial_dispatch_tail_does_not_hide_complete_owned_ids(self):
        obj=bare();old_root=gate.ROOT
        with tempfile.TemporaryDirectory() as temp:
            gate.ROOT=Path(temp)
            try:
                row=dict(port=33500,route='/v1/completions',request_id='budget-123')
                (gate.ROOT/'dispatch.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(row)[:20])
                self.assertEqual(obj.owned_ids(),{(33500,'budget-123')})
                self.assertIn('dispatch_incomplete_tail',obj.state)
            finally:gate.ROOT=old_root

    def test_validator_bytes_unchanged(self):
        self.assertEqual((gate.ROOT/'budget_validation.frozen.py').read_bytes(),
            (gate.OLD/'budget_validation.frozen.py').read_bytes())

    def test_one_event_file_error_does_not_skip_other_evidence(self):
        obj=bare();obj.event_offsets={'missing':0,'good':0}
        root,old=gate.ROOT,gate.OLD
        with tempfile.TemporaryDirectory() as temp:
            gate.ROOT=Path(temp);gate.OLD=Path(temp)/'old';(gate.OLD/'runtime').mkdir(parents=True)
            try:
                (gate.OLD/'runtime/good.control.events.jsonl').write_text('{}\n')
                obj.capture_events()
                self.assertEqual((gate.ROOT/'good.events.jsonl').read_text(),'{}\n')
                self.assertEqual(len(obj.state['event_capture_errors']),1)
            finally:gate.ROOT=root;gate.OLD=old


if __name__=='__main__':unittest.main(verbosity=2)
