import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import capacity_load_calibrate as c


class TraceAndEvidenceTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
