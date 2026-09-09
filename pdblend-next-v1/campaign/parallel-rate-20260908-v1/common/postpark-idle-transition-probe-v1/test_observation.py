import ast,copy,unittest
import run as m
class Observation(unittest.TestCase):
 def rows(self,delayed=False):return [dict(gpu=6,observed_mhz=2520 if delayed and i<6 else 2100,started_s=i/10,finished_s=i/10+.001) for i in range(10)]
 def result(self,r,deadline=2):return m.observation_result(r,2100,0,deadline,15,.05)
 def test_immediate(self):self.assertTrue(self.result(self.rows())['passed'])
 def test_late_point_fifty(self):
  r=self.result(self.rows(True));self.assertTrue(r['passed']);self.assertGreater(r['observed_first_latency_s'],.5);self.assertFalse(r['physical_settling_time_independently_known'])
 def test_original_point_three_would_fail(self):self.assertFalse(self.result(self.rows(True),.3)['passed'])
 def test_wrong(self):self.assertFalse(self.result([dict(x,observed_mhz=2520) for x in self.rows()])['passed'])
 def test_one_read_not_stable(self):self.assertFalse(self.result(self.rows()[:1])['passed'])
 def test_interrupted_stability(self):self.assertFalse(self.result([dict(gpu=6,observed_mhz=v,started_s=i*.02,finished_s=i*.02) for i,v in enumerate([2100,2520,2100])])['passed'])
 def test_nan(self):
  with self.assertRaises(ValueError):self.result([dict(gpu=6,observed_mhz=float('nan'),started_s=0,finished_s=.1)])
 def test_no_request_calls(self):
  source=(m.HERE/'run.py').read_text();self.assertNotIn('/v1/completions',source);self.assertNotIn('profiler.run',source)
 def test_original_owned_physical_calls(self):
  source=(m.HERE/'run.py').read_text();self.assertIn('await cl.physical_write(',source);self.assertIn('await cl.physical_park(',source);self.assertIn('await asyncio.wait_for(cl.close(),90)',source)
if __name__=='__main__':unittest.main(verbosity=2)
