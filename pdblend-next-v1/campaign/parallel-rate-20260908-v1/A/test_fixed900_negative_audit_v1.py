"""Mutation tests over the real retained raw reference; no GPU or source edits."""
from pathlib import Path
import copy,json,unittest
import fixed900_negative_audit_v1 as m
A=Path(m.__file__).parent;OUT=A/'p6-qualification900-fixed2-002'
class NegativeAuditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.decl=json.loads((A/'p6-qualification900-inputs-002/declaration.json').read_text());cls.sp=m.fixed(cls.decl['specs']['fixed2']);cls.result=m.fixed(m.ref(OUT/'qualification900/result.json'));cls.trace=m.fixed(cls.result['trace']);cls.cfg=m.fixed(cls.sp['config']);cls.raw=m.prior.raw_measurement(cls.result['raw_measurement']);cls.rows=json.loads((OUT/'qualification900/requests.json').read_text());cls.timings=[x for x in map(json.loads,(OUT/'control.jsonl').read_text().splitlines()) if x['kind']=='request_timing'];cls.clock=list(map(json.loads,(OUT/'control.clock-guard.jsonl').read_text().splitlines()));cls.proof=m.fixed(m.ref(A/'p6-fixed900-negative-diagnosis-001/fresh-terminal-snapshot.json'));cls.status=m.fixed(m.ref(OUT/'status.json'));cls.binding=m.fixed(cls.sp['original_binding'])
 def rowcheck(self,rows=None,result=None,timing=None):return m.audit_rows(result or self.result,self.trace,self.cfg,self.raw,rows or self.rows,timing or self.timings)
 def test_real_all_rows(self):
  r=self.rowcheck();self.assertEqual((r['n_expected'],r['n_completed'],r['failed_requests'],r['request_timeouts'],r['admission_rejections']),(3767,3541,226,22,204));self.assertEqual(r['observed_partial_timeout_chunks'],39);self.assertFalse(r['generated_token_count_complete'])
 def test_denominator_cannot_drop_failures(self):
  r=copy.deepcopy(self.result);r['n_expected']-=226
  with self.assertRaises(ValueError):self.rowcheck(result=r)
 def test_successful_truncation_rejected(self):
  x=copy.deepcopy(self.rows);x[0]['generated_tokens']-=1
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_premature_timeout_rejected(self):
  x=copy.deepcopy(self.rows);x[116]['finish_s']=x[116]['request_deadline_s']-1
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_modified_timeout_budget_rejected(self):
  x=copy.deepcopy(self.rows);x[116]['request_deadline_s']+=1
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_503_rejected(self):
  x=copy.deepcopy(self.rows);i=next(i for i,v in enumerate(x) if v['admission_rejection']);x[i]['http_status']=503
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_named_code_cannot_hide_other_error(self):
  x=copy.deepcopy(self.rows);i=next(i for i,v in enumerate(x) if v['admission_rejection']);x[i]['error']=x[i]['error'].replace('admission_queue_full','physical_clock_failure')
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_dispatch_starvation_rejected(self):
  x=copy.deepcopy(self.rows);x[116]['actual_dispatch_s']+=20;x[116]['dispatch_delay_s']+=20
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_native_timeout_terminal_required(self):
  t=copy.deepcopy(self.timings);t=[v for v in t if v['client_request_id']!='116']
  with self.assertRaises(ValueError):self.rowcheck(timing=t)
 def test_partial_chunks_not_tokens(self):
  x=copy.deepcopy(self.rows);i=next(i for i,v in enumerate(x) if v['request_timeout'] and v['n_text_chunks']);x[i]['generated_tokens']=x[i]['n_text_chunks']
  with self.assertRaises(ValueError):self.rowcheck(rows=x)
 def test_unterminated_clock_rejected(self):
  c=copy.deepcopy(self.clock);c.pop(next(i for i,v in enumerate(c) if v.get('stage')=='command_completed'))
  with self.assertRaises(ValueError):m.clock_audit(c)
 def test_foreign_process_rejected(self):
  p=copy.deepcopy(self.proof);p['actual_identity'][0]['container']['State']['Pid']+=1
  with self.assertRaises(ValueError):m.validate_terminal_proof(p,OUT,self.status,self.binding)
 def test_terminal_residual_rejected(self):
  p=copy.deepcopy(self.proof);p['actual_identity'][0]['runtime']['active']=1
  with self.assertRaises(ValueError):m.validate_terminal_proof(p,OUT,self.status,self.binding)
 def test_pre_cleanup_snapshot_rejected(self):
  p=copy.deepcopy(self.proof);p['started_s']=self.status['finished_s']-1
  with self.assertRaises(ValueError):m.validate_terminal_proof(p,OUT,self.status,self.binding)
 def test_full_real_audit(self):
  r=m.audit(OUT,self.decl['specs']['fixed2'],self.decl);self.assertTrue(r['passed']);self.assertFalse(r['full_work']);self.assertFalse(r['strict_qualification_passed']);self.assertFalse(r['equal_work_energy_comparison_eligible'])
if __name__=='__main__':unittest.main()
