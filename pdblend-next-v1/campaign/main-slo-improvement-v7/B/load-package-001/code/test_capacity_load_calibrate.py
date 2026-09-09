import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch, AsyncMock

import capacity_load_calibrate as c


class TraceAndEvidenceTests(unittest.TestCase):
    def test_spec_rejects_overlap_incomplete_duplicate_or_outside_extra_gpu_group(self):
        original={'instances':[{'id':'a','gpus':[0,1]},{'id':'b','gpus':[2,3]}]}
        config={'instances':original['instances']}
        binding={'identity':{'tp':2}}
        spec=dict(schema='capacity-load-calibration-spec-v1',authorized=True,automatic_retries=False,
            files={'CPU-only':'h'},original_binding={},capacity_binding={},config={})
        for group in ([1,2],[4],[],[4,4],[4,8],[4,True],['4',5]):
            with self.subTest(group=group),patch.object(c,'sha',return_value='h'),patch.object(c,'fixed',side_effect=[original,binding,config]):
                with self.assertRaisesRegex(ValueError,'GPU group'):c.validate_spec(dict(spec,gpus=group))
    def trace(self,seed=91):
        return c.generate_trace([dict(prompt=[1,2,3],output_len=64)],
            [dict(name='low',duration_s=300,rate_rps=.1),dict(name='high',duration_s=300,rate_rps=.5),
             dict(name='low',duration_s=300,rate_rps=.1)],seed,'a'*64)

    def test_low_high_low_reproducible_independent(self):
        first=self.trace();c.validate_trace(first,'a'*64,duration=900)
        self.assertEqual(first,self.trace())
        self.assertNotEqual(first['requests'],self.trace(92)['requests'])
        self.assertEqual([p['duration_s'] for p in first['phases']],[300,300,300])
        self.assertFalse(first['formal_eligible'])

    def test_domain_and_duration_cannot_be_widened(self):
        with self.assertRaises(ValueError):c.validate_trace(self.trace(),'b'*64,duration=900)
        with self.assertRaises(ValueError):c.validate_trace(self.trace(),'a'*64,duration=100)

    def test_full_token_completion_and_measured_tail_are_required(self):
        rows=[dict(success=1,token_ids_verified=1,generated_tokens=63,output_len=64,slo_ok=1)]
        trace=dict(requests=[{}],duration_s=100)
        meter=dict(measurement_start_s=1,measurement_end_s=121,energy_j=1200)
        result=c.phase_metrics(rows,trace,meter,True)
        self.assertFalse(result['work_complete'])
        self.assertFalse(result['empirical_sustainable_sample'])
        self.assertEqual(result['goodput_measurement_rps'],1/120)
        rows[0]['generated_tokens']=64
        self.assertTrue(c.phase_metrics(rows,trace,meter,True)['work_complete'])
        self.assertFalse(c.phase_metrics(rows,trace,meter,False)['work_complete'])

    def test_low_slo_is_preserved_as_empirical_failure(self):
        rows=[dict(success=1,token_ids_verified=1,generated_tokens=64,output_len=64,slo_ok=0)]
        result=c.phase_metrics(rows,dict(requests=[{}],duration_s=100),
             dict(measurement_start_s=1,measurement_end_s=101,energy_j=1000),True)
        self.assertTrue(result['work_complete'])
        self.assertFalse(result['empirical_sustainable_sample'])
        self.assertEqual(result['slo_attainment'],0)

    def test_trace_rejects_missing_and_unordered_requests(self):
        trace=self.trace();trace['prompts'].pop()
        with self.assertRaises(ValueError):c.validate_trace(trace,'a'*64)
        trace=self.trace();trace['requests'][1]['arrival_s']=-1
        with self.assertRaises(ValueError):c.validate_trace(trace,'a'*64)


class PhaseLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.host=self.root/'host';self.host.mkdir()
        c.durable(self.host/'manifest.json',{'cpu_fake':True})
        self.trace=dict(requests=[{'arrival_s':0,'prompt_len':3,'output_len':64}],
                        prompts=[[1,2,3]],duration_s=.001)
        c.durable(self.root/'trace.json',self.trace)
        dummy=dict(path='CPU-only',sha256='a'*64)
        self.spec=dict(mode='qualification900',arm='fixed2',api_base='cpu://fake',served_model='fake',
            deadline_s=time.time()+600,demand_domain_sha256='a'*64,original_binding=dummy,
            capacity_binding=dummy,config=dummy,host_release=str(self.host))
        self.events=[]
        self.controller=SimpleNamespace(active={},request_tasks=set(),config=dict(slo_ttft_s=1,slo_tpot_s=.1),
            backend=SimpleNamespace(json=AsyncMock(return_value={'cpu_only':True}),instances={'a':{'id':'a','gpus':[6]},'b':{'id':'b','gpus':[7]}}))
        self.controller.finish_measurement=AsyncMock(side_effect=self.drain)
        self.rows=[dict(success=1,token_ids_verified=1,generated_tokens=64,output_len=64,slo_ok=1)]
        host=Path(__file__).resolve().parents[1]/'hosts/14b-fixed-v4'
        sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
        from benchmarks.scripts import bench_vllm
        self.bench=bench_vllm
        events=self.events
        class Meter:
            def __init__(self,out):self.out=out
            async def start(self):events.append('meter_start');return self
            async def finish(self):
                events.append('meter_finish');self.out.mkdir(parents=True)
                return dict(measurement_valid=True,measurement_start_s=1,measurement_end_s=2,energy_j=10,
                            receipt=dict(path='cpu-meter',sha256='f'*64))
        self.patchers=[patch('capacity_backend.TransitionMeter',Meter),patch.object(c,'native_idle',AsyncMock(return_value=True)),
                       patch.object(self.bench,'bench_rows',return_value=self.rows),
                       patch('ecopadg.serving.completion_policy.engine_residual',return_value={})]
        for p in self.patchers:p.start();self.addCleanup(p.stop)

    async def drain(self,*args):
        self.events.append('actual_dynamic_or_fixed_drain')
        return {'drain_complete':True}

    async def test_final_dynamic_removal_is_inside_measured_window(self):
        async def work(*args,**kwargs):return [dict(planned_arrival_s=time.time()-.002)],.002
        with patch.object(self.bench,'run_trace',work):
            result=await c.measure_phase(self.controller,None,c.ref(self.root/'trace.json'),self.root/'phase',self.spec)
        self.assertTrue(result['work_complete'])
        self.assertLess(self.events.index('actual_dynamic_or_fixed_drain'),self.events.index('meter_finish'))
        self.assertEqual(result['resident_groups'],[[6],[7]])

    async def test_failed_request_retains_partial_sink_and_power(self):
        async def send(*args,**kwargs):raise RuntimeError('real failure shape')
        async def work(*args,**kwargs):
            sink={'0':dict(generated_tokens=7,success=False,error='partial')}
            await self.bench.send_request(_result_sink=sink)
        with patch.object(self.bench,'send_request',send),patch.object(self.bench,'run_trace',work):
            with self.assertRaisesRegex(RuntimeError,'real failure shape'):
                await c.measure_phase(self.controller,None,c.ref(self.root/'trace.json'),self.root/'failed',self.spec)
        value=json.loads((self.root/'failed/partial-client-results.json').read_text())
        self.assertEqual(value['0']['generated_tokens'],7)
        result=json.loads((self.root/'failed/result.json').read_text())
        self.assertFalse(result['complete']);self.assertIn('raw_measurement',result)

    async def test_incomplete_final_drain_cannot_certify_phase(self):
        async def work(*args,**kwargs):return [dict(planned_arrival_s=time.time()-.002)],.002
        self.controller.finish_measurement=AsyncMock(return_value={'drain_complete':False})
        with patch.object(self.bench,'run_trace',work):
            with self.assertRaisesRegex(ValueError,'cleanup incomplete'):
                await c.measure_phase(self.controller,None,c.ref(self.root/'trace.json'),self.root/'failed',self.spec)
        result=json.loads((self.root/'failed/result.json').read_text())
        self.assertFalse(result['complete']);self.assertNotIn('empirical_sustainable_sample',result)

    async def test_idle_has_no_requests_or_slo_and_retains_raw_power(self):
        now=[1000.]
        async def advance(seconds):now[0]+=seconds
        self.spec['deadline_s']=10000.
        with patch.object(c.time,'time',side_effect=lambda:now[0]),patch.object(c.asyncio,'sleep',side_effect=advance), \
                patch.object(self.bench,'run_trace',side_effect=AssertionError('idle must not issue a trace')):
            result=await c.measure_idle(self.controller,self.root/'idle',self.spec,60.)
        self.assertEqual(result['n_expected'],0);self.assertIsNone(result['slo_attainment'])
        self.assertFalse(result['empirical_sustainable_sample'])
        self.assertEqual(result['measured_idle_duration_s'],60.)
        self.assertEqual(json.loads((self.root/'idle/requests.json').read_text()),[])
        self.assertIn('raw_measurement',result)

    async def test_request_entering_idle_invalidates_the_observation(self):
        now=[1000.]
        async def advance(seconds):
            now[0]+=seconds
            if now[0]>=1030:self.controller.active['foreign-observed']=object()
        self.spec['deadline_s']=10000.
        with patch.object(c.time,'time',side_effect=lambda:now[0]),patch.object(c.asyncio,'sleep',side_effect=advance):
            with self.assertRaisesRegex(ValueError,'request entered idle'):
                await c.measure_idle(self.controller,self.root/'idle-failed',self.spec,60.)
        result=json.loads((self.root/'idle-failed/result.json').read_text())
        self.assertFalse(result['complete']);self.assertIn('raw_measurement',result)


if __name__=='__main__':unittest.main()
