"""CPU-only scope, timing, and failure-stop counterexamples; no GPU imports."""
import ast,copy,csv,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import contract as c
import run as runner

class Scope(unittest.TestCase):
 def setUp(self):
  self.logical=c.checked(dict(path=str(c.LOGICAL),sha256=c.LOGICAL_SHA))
  self.rows=copy.deepcopy(self.logical['cells']);self.observed=[]
  for row in self.rows:
   r=copy.deepcopy(row);r['system']='pdblend';self.observed.append(dict(row=r,slo_attainment=1.))
  self.scope=dict(required_cells=copy.deepcopy(self.rows),excluded_cells=[],final_first_miss_by_dataset=dict(alpaca=None,sharegpt=None,longbench=None))
 def good(self):return c.select_rows(self.logical,self.scope,self.observed,{})
 def test_actual_original31_all_selected(self):self.assertEqual(len(self.good()[0]),31)
 def test_missing_original(self):
  self.scope['required_cells'].pop()
  with self.assertRaises(ValueError):self.good()
 def test_duplicate_original(self):
  self.scope['required_cells'].append(self.rows[0])
  with self.assertRaises(ValueError):self.good()
 def test_changed_original_output(self):
  self.scope['required_cells'][0]['n_requests']+=1
  with self.assertRaises(ValueError):self.good()
 def test_repeat2_cannot_fallback_to_repeat1(self):
  self.observed=[v for v in self.observed if v['row']['repeat']==1]
  with self.assertRaises(ValueError):self.good()
 def test_original_repeat1_can_reuse_repeat2(self):
  for v in self.observed:v['row']['repeat']=2
  self.assertEqual(len(self.good()[0]),31)
 def test_wrong_claimed_first_loss(self):
  self.scope['final_first_miss_by_dataset']['alpaca']=12.
  with self.assertRaises(ValueError):self.good()
 def test_actual_boundary_precise_partition(self):
  dataset='longbench';rates=sorted({r['rate_rps'] for r in self.rows if r['dataset']==dataset});edge=rates[-2]
  for v in self.observed:
   if v['row']['dataset']==dataset and v['row']['rate_rps']==edge:v['slo_attainment']=.899
  self.scope['final_first_miss_by_dataset'][dataset]=edge
  excluded=[r for r in self.rows if r['dataset']==dataset and r['rate_rps']>edge]
  self.scope['required_cells']=[r for r in self.rows if r not in excluded]
  self.scope['excluded_cells']=[dict(cell_id=r['cell_id'],dataset=dataset,rate_rps=r['rate_rps'],reason='above_first_complete_PDB_SLO_loss',first_complete_slo_miss_rate=edge) for r in excluded]
  self.assertEqual(len(self.good()[0])+len(excluded),31)
 def test_cannot_exclude_boundary(self):
  r=self.rows[0];self.scope['required_cells'].pop(0);self.scope['final_first_miss_by_dataset'][r['dataset']]=r['rate_rps']
  self.observed[0]['slo_attainment']=.8
  self.scope['excluded_cells']=[dict(cell_id=r['cell_id'],dataset=r['dataset'],rate_rps=r['rate_rps'],reason='above_first_complete_PDB_SLO_loss',first_complete_slo_miss_rate=r['rate_rps'])]
  with self.assertRaises(ValueError):self.good()
 def test_nan_slo(self):
  self.observed[0]['slo_attainment']=float('nan')
  with self.assertRaises(ValueError):self.good()
 def test_wrong_source_model(self):
  self.observed[0]['row']['model']='7b'
  with self.assertRaises(ValueError):self.good()
 def append(self):
  rows=[]
  for system in ('pdblend','mixed','distserve','dynamollm','ecoserve'):
   for repeat in (1,2):
    r=copy.deepcopy(self.rows[0]);r.update(cell_id='append-'+system+'-'+str(repeat),rate_rps=99.,repeat=repeat,system=system,trace_sha256='new-trace',content_pairing_sha256='new-content');rows.append(r)
  self.scope['required_cells'] += [copy.deepcopy(r) for r in rows if r['system']=='ecoserve']
  self.observed += [dict(row=copy.deepcopy(r),slo_attainment=1.) for r in rows if r['system']=='pdblend']
  return {'append':dict(cells=rows)}
 def test_append_five_systems_two_repeats(self):
  app=self.append();self.assertEqual(len(c.select_rows(self.logical,self.scope,self.observed,app)[0]),33)
 def test_append_missing_baseline_repeat(self):
  app=self.append();app['append']['cells'].pop(2)
  with self.assertRaises(ValueError):c.select_rows(self.logical,self.scope,self.observed,app)
 def test_append_mismatched_trace(self):
  app=self.append();app['append']['cells'][2]['trace_sha256']='changed'
  with self.assertRaises(ValueError):c.select_rows(self.logical,self.scope,self.observed,app)
 def test_append_missing_eco_repeat2(self):
  app=self.append();self.scope['required_cells'].pop()
  with self.assertRaises(ValueError):c.select_rows(self.logical,self.scope,self.observed,app)
 def test_append_pdb_repeat2_missing(self):
  app=self.append();self.observed.pop()
  with self.assertRaises(ValueError):c.select_rows(self.logical,self.scope,self.observed,app)

class Engineering(unittest.TestCase):
 def setUp(self):
  self.receipt=dict(measurement_valid=True,child_stopped=True,child_exitcode=0,clock_restore_complete=True,outer_cleanup_errors=[],sampling_error=None,summary=dict(measurement_valid=True,work_complete=True,failed_requests=0,request_timeouts=0,gpu_count=8,power_source_verified=True,fixed_window_valid=True,runtime_error=None,slo_attainment=.01,post_measurement_cleanup=dict(cleanup_complete=True)))
  self.bench=[];self.events=[]
  for n in range(100):
   start=1000.+n;actual=start+.01;rid='request-'+str(n)
   self.bench.append(dict(request_id=rid,planned_arrival_s=str(start),actual_dispatch_s=str(actual),request_deadline_s=str(start+120),open_loop_independent='True'))
   self.events.append(dict(kind='request_timing',client_request_id=rid,planned_arrival_s=start,actual_dispatch_s=actual,hard_deadline_s=start+120,handler_arrival_s=actual+.001))
 def test_complete_low_slo_is_valid(self):self.assertTrue(runner.gate(self.receipt)['passed'])
 def test_every_original_failure_gate_stops(self):
  for key,value in [('failed_requests',1),('request_timeouts',1),('work_complete',False),('measurement_valid',False),('gpu_count',7),('power_source_verified',False),('fixed_window_valid',False),('runtime_error','error')]:
   with self.subTest(key=key):
    r=copy.deepcopy(self.receipt);r['summary'][key]=value;self.assertFalse(runner.gate(r)['passed'])
 def test_native_cleanup_and_sampler_failure_stop(self):
  for key,value in [('child_stopped',False),('child_exitcode',1),('clock_restore_complete',False),('outer_cleanup_errors',['error']),('sampling_error','error')]:
   r=copy.deepcopy(self.receipt);r[key]=value;self.assertFalse(runner.gate(r)['passed'])
  self.receipt['summary']['post_measurement_cleanup']['cleanup_complete']=False;self.assertFalse(runner.gate(self.receipt)['passed'])
 def test_timing_original_protocol(self):self.assertTrue(runner.timing(self.bench,self.events)['passed'])
 def test_duplicate_timing_row(self):
  self.events.append(self.events[0])
  with self.assertRaises(ValueError):runner.timing(self.bench,self.events)
 def test_dispatch_late(self):
  self.bench[0]['actual_dispatch_s']='1001.1'
  with self.assertRaises(ValueError):runner.timing(self.bench,self.events)
 def test_p99_lateness(self):
  for b,e in zip(self.bench,self.events):b['actual_dispatch_s']=str(float(b['planned_arrival_s'])+.2);e['actual_dispatch_s']=float(b['actual_dispatch_s']);e['handler_arrival_s']=e['actual_dispatch_s']+.001
  with self.assertRaises(ValueError):runner.timing(self.bench,self.events)
 def test_deadline_shortening(self):
  self.bench[0]['request_deadline_s']='1119'
  with self.assertRaises(ValueError):runner.timing(self.bench,self.events)
 def test_handler_starvation(self):
  self.events[0]['handler_arrival_s']+=1.1
  with self.assertRaises(ValueError):runner.timing(self.bench,self.events)
 def test_source_control_retains_checkpoint_before_gate(self):
  source=(HERE/'run.py').read_text();ast.parse(source)
  self.assertLess(source.index('receipt,arrival=checkpoint()'),source.index("engineering=gate(receipt)"))
  self.assertIn("if rp.exists() and not cp.exists():checkpoint()",source)
  self.assertIn("with node_lease():",source);self.assertNotIn('time.time()+',source)
 def test_original_common_byte_identical(self):self.assertEqual(c.sha(c.COMMON),c.COMMON_SHA)

if __name__=='__main__':unittest.main()
