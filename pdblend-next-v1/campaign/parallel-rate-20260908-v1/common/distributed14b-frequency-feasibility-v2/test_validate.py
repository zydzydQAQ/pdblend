import copy,unittest
import validate as v

def fixture():
 p=dict(point_id='fixture',instance_id='i6',gpu=6,frequency_mhz=2100,input_tokens=128,output_tokens=64,batch=1)
 state=dict(id='i6',generation=3,acknowledged_generation=3,scheduler_budget_effective=dict(max_num_batched_tokens=2048),runtime_error=None,error=None)
 state.update({k:0 for k in ('active','running','waiting','kv_allocations','transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')})
 r=dict(request_id='pdb-a-batch-fixture',success=True,done_marker=True,http_status=200,prompt_token_ids=([9707,1879,13]*43)[:128],requested_output_tokens=64,output_token_ids=list(range(64)),token_received_s=list(range(2,66)),usage=dict(prompt_tokens=128,completion_tokens=64),dispatch_s=1)
 state.update(acknowledged_generations=[3],observed_control_generation=3,scheduler_budget_pending=None,transport_healthy=True,transfer_send_counters_observed=True,transfer_inflight_sends_observed=True,transfer_send_healthy=True,transfer_send_started=0,transfer_send_completed=0,transfer_send_failed=0,scheduler_io=[dict(controls=dict(runtime=dict(generation=3,error=None)))],role='mixed',mode='continuous',accepting=True)
 state['scheduler_budget_effective']['max_num_seqs']=32
 after=copy.deepcopy(state);after.update(generation=5,acknowledged_generation=5,acknowledged_generations=[5],observed_control_generation=5);after['scheduler_budget_effective']['max_num_batched_tokens']=8192;after['scheduler_io'][0]['controls']['runtime']['generation']=5
 cleanup=dict(complete=True,errors=[],before=copy.deepcopy(state),proof=dict(generation=4,drained=True,accepting=False,drain_proof_type='synchronous_put_owner_barrier',send_counters_verified=True,transfers=[dict(send_counters_observed=True,send_healthy=True,listener_alive=True,buffered_tensors=0,inflight_sends=0,send_failed=0,inflight_receives=0,buffered_gpu_bytes=0,allocations={},send_started=0,send_completed=0)]),resumed=dict(after=after,control=dict(generation=5)))
 raw=dict(point=p,requests=[r],error=None,cleanup=cleanup,started_s=1,finished_s=70,runtime_before=state,runtime_after_requests=copy.deepcopy(state))
 events=[dict(started_s=1.1+j,finished_s=1.9+j,request_ids=[r['request_id']],role='mixed',mode='continuous',generation=3,prefill=int(j==0),decode=int(j!=0),tokens=128 if j==0 else 1) for j in range(64)]
 clocks=[(1.5+j,[2100]*8) for j in range(64)]
 return raw,events,clocks
class Tests(unittest.TestCase):
 def test_good(self):self.assertTrue(v.validate_point(*fixture())['passed'])
 def reject(self,fn):
  args=fixture();fn(*args)
  with self.assertRaises((ValueError,KeyError)):v.validate_point(*args)
 def test_truncated(self):self.reject(lambda r,e,c:r['requests'][0]['output_token_ids'].pop())
 def test_usage(self):self.reject(lambda r,e,c:r['requests'][0]['usage'].update(completion_tokens=63))
 def test_owner(self):self.reject(lambda r,e,c:e[2].update(request_ids=['foreign']))
 def test_gen(self):self.reject(lambda r,e,c:e[2].update(generation=4))
 def test_prefill(self):self.reject(lambda r,e,c:e[0].update(tokens=127))
 def test_tail(self):self.reject(lambda r,e,c:e.pop())
 def test_clock_low(self):self.reject(lambda r,e,c:c.__setitem__(3,(4.5,[2084]*8)))
 def test_clock_nan(self):self.reject(lambda r,e,c:c.__setitem__(3,(4.5,[float('nan')]*8)))
 def test_cleanup(self):self.reject(lambda r,e,c:r['cleanup'].update(complete=False))
 def test_empty_clock(self):self.reject(lambda r,e,c:c.clear())
 def test_error(self):self.reject(lambda r,e,c:r.update(error='timeout'))
 def test_native_residue(self):self.reject(lambda r,e,c:r['runtime_after_requests'].update(waiting=1))
 def test_negative_prediction_retained(self):
  raw,e,c=fixture();result=v.validate_point(raw,e,c);p=raw['point'];profile=dict(points=[dict(role='mixed',tp=1,frequency_mhz=2100,input_tokens=128,batch=1,context_tokens=256,iteration_s=.001,prefill_s=.001)])
  p['frequency_mhz']=2400
  value=v.prediction_error(result,p,profile);self.assertEqual(value['iteration_relative_errors'],[]);self.assertFalse(value['same_frequency_reference_available']);self.assertTrue(value['not_new_target_profile'])
class WriteTests(unittest.TestCase):
 def function(self,path,name):
  import ast,json,os
  from pathlib import Path
  ns=dict(Path=Path,json=json,os=os);tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);return ns[name]
 def test_mutable_progress(self):
  import tempfile,json
  from pathlib import Path
  fn=self.function(Path(__file__).parent/'run.py','mutable_write')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';fn(p,dict(n=0));fn(p,dict(n=1));self.assertEqual(json.loads(p.read_text()),dict(n=1))
 def test_frozen_exclusive(self):
  import tempfile
  from pathlib import Path
  fn=self.function(Path(__file__).parent.parent/'distributed14b-deployment-v1/runtime_adapter.py','write')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'spec.json';fn(p,dict(n=0))
   with self.assertRaises(FileExistsError):fn(p,dict(n=1))
if __name__=='__main__':unittest.main()
