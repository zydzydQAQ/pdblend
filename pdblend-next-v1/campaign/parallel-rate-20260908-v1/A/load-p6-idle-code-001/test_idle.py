"""CPU-only boundary counterexamples against the unchanged original verifier."""
import ast,copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import capacity_certificate as c
import capacity_load_calibrate as m
class Tests(unittest.TestCase):
 def setUp(self):
  self.before=[dict(id='i6',gpus=[6]),dict(id='i7',gpus=[7])]
  self.result=dict(resident_before=self.before,resident_after=self.before,resident_groups=[[6],[7]],actual_idle_end_s=160.002)
  self.inventory=dict(complete=True,transition_inflight=False,initial_ids=['i6','i7'],known_instances={v['id']:dict(verified=True,gpus=v['gpus']) for v in self.before})
  self.rows=[]
  for at in [100.+i for i in range(60)]+[160.001]:
   raw=[]
   for item in self.before:
    r=dict(id=item['id'],active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},transfer_buffered_tensors=0,transfer_inflight_receives=0,transfer_inflight_sends=0,timestamp=at-.001,transfer_observed_s=at-.001,transport_healthy=True,error=None,runtime_error=None,generation=1,acknowledged_generation=1,scheduler_budget_pending=None);raw.append(r)
   self.rows.append(dict(at_s=at,raw=raw))
  # Cadence <=1 even for the final post-window sample.
  self.rows[-2]['at_s']=159.001
  for r in self.rows[-2]['raw']:r['timestamp']=r['transfer_observed_s']=159.
  self.rows[-3]['at_s']=158.001
  for r in self.rows[-3]['raw']:r['timestamp']=r['transfer_observed_s']=158.
  # Uniform .2-second cadence provides a simple valid sixty-second baseline.
  self.rows=[]
  for i in range(302):
   at=99.999+i*.2;raw=[dict(id=item['id'],active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={},transfer_buffered_tensors=0,transfer_inflight_receives=0,transfer_inflight_sends=0,timestamp=at-.001,transfer_observed_s=at-.001,transport_healthy=True,error=None,runtime_error=None,generation=1,acknowledged_generation=1,scheduler_budget_pending=None) for item in self.before];self.rows.append(dict(at_s=at,raw=raw))
  self.rows[-1]['at_s']=160.001
  for r in self.rows[-1]['raw']:r['timestamp']=r['transfer_observed_s']=160.
 def check(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'native.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in self.rows));c.verify_idle_native(path,self.result,self.inventory,100.,160.)
 def test_post_declared_before_actual_end_is_valid(self):self.check()
 def test_missing_declared_end_rejected(self):
  self.rows[-1]['at_s']=159.999
  for r in self.rows[-1]['raw']:r['timestamp']=r['transfer_observed_s']=159.998
  with self.assertRaises(ValueError):self.check()
 def test_original_1039ms_gap_rejected(self):
  self.rows[15:20]=[]
  self.rows[15]['at_s']=self.rows[14]['at_s']+1.039372
  with self.assertRaises(ValueError):self.check()
 def test_busy_rejected(self):
  self.rows[5]['raw'][0]['active']=1
  with self.assertRaises(ValueError):self.check()
 def test_stale_rejected(self):
  self.rows[5]['raw'][0]['timestamp']-=2
  with self.assertRaises(ValueError):self.check()
 def test_unqualified_layout_rejected(self):
  self.inventory['known_instances']['i7']['verified']=False
  with self.assertRaises(ValueError):self.check()
 def test_declared_mode(self):
  sp=json.loads((D.parent/'load-p6-full-inputs-002/spec.json').read_text());sp['mode']='idle_calibration';sp['cycles']=[{k:sp['cycles'][1][k] for k in ('restore','remove')}];m.validate_spec(sp)
  for mutate in [lambda x:x['cycles'].append(copy.deepcopy(x['cycles'][0])),lambda x:x['cycles'][0].update(low={}),lambda x:x.update(matched_idle_duration_s=59)]:
   bad=copy.deepcopy(sp);mutate(bad)
   with self.assertRaises(ValueError):m.validate_spec(bad)
 def test_five_modules_and_cleanup_unchanged(self):
  old=D.parent/'load-p6-code-001'
  for name in ['capacity_backend.py','capacity_certificate.py','capacity_executor.py','capacity_planner.py','capacity_runtime.py']:self.assertEqual((D/name).read_bytes(),(old/name).read_bytes())
  def final(source):
   tree=ast.parse(source);f=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='execute');return ast.dump(next(n for n in f.body if isinstance(n,ast.Try)).finalbody[0],include_attributes=False),[ast.dump(n,include_attributes=False) for n in next(n for n in f.body if isinstance(n,ast.Try)).finalbody]
  self.assertEqual(final((D/'capacity_load_calibrate.py').read_text()),final((old/'capacity_load_calibrate.py').read_text()))
if __name__=='__main__':unittest.main()
