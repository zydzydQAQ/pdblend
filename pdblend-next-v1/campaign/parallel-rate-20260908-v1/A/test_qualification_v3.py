"""CPU-only independent audit counterexamples; no GPU or process mutation."""
import copy,importlib.util,json,math,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'A/continue_p6_qualification_v3.py'
spec=importlib.util.spec_from_file_location('a_qualification_v3_review',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def setUp(self):
  epoch=1788870000.
  self.t=dict(n_requests=2,requests=[dict(arrival_s=0.,prompt_len=2,output_len=3),dict(arrival_s=500.,prompt_len=1,output_len=4)],prompts=[[1,2],[3]])
  self.c=dict(slo_ttft_s=1.,slo_tpot_s=.1)
  self.rows=[dict(idx=i,request_id=str(i),success=1,request_timeout=False,token_ids_verified=1,generated_tokens=t['output_len'],output_len=t['output_len'],input_tokens=t['prompt_len'],prompt_len=t['prompt_len'],planned_arrival_s=epoch+t['arrival_s'],request_deadline_s=epoch+t['arrival_s']+120,ttft_s=.1 if i==0 else 2.,tpot_s=.05,slo_ok=1 if i==0 else 0) for i,t in enumerate(self.t['requests'])]
  self.r=dict(n_expected=2,n_rows=2,n_good=1,slo_attainment=.5,actual_arrival_epoch_s=epoch,measured_arrival_duration_s=900,energy_j=1234.,offered_rate_rps=2/900,failed_requests=0,request_timeouts=0,work_complete=True)
  self.raw=dict(energy_j=1234.,measurement_start_s=epoch-1,measurement_end_s=epoch+901)
 def run_audit(self):return m.audit_rows(self.r,self.t,self.c,self.raw,self.rows)
 def test_complete_low_slo_is_retained(self):self.assertEqual(self.run_audit()['slo_attainment'],.5)
 def test_bad_fields_rejected(self):
  mutants=[('energy',lambda:self.r.update(energy_j=1)),('slo_summary',lambda:self.r.update(slo_attainment=.9)),('good_count',lambda:self.r.update(n_good=2)),('rate',lambda:self.r.update(offered_rate_rps=1)),('incomplete',lambda:self.r.update(work_complete=False)),('summary_failed',lambda:self.r.update(failed_requests=1)),('summary_timeout',lambda:self.r.update(request_timeouts=1)),('duration100',lambda:self.r.update(measured_arrival_duration_s=100)),('nrows',lambda:self.r.update(n_rows=1)),('failed',lambda:self.rows[0].update(success=0)),('timeout',lambda:self.rows[0].update(request_timeout=True)),('unverified',lambda:self.rows[0].update(token_ids_verified=0)),('partial_output',lambda:self.rows[0].update(generated_tokens=2)),('wrong_input',lambda:self.rows[0].update(input_tokens=1)),('duplicate_id',lambda:self.rows[1].update(request_id='0')),('row_order',lambda:self.rows[0].update(idx=1)),('row_slo',lambda:self.rows[0].update(slo_ok=0)),('nan_timing',lambda:self.rows[0].update(ttft_s=float('nan'))),('negative_timing',lambda:self.rows[0].update(tpot_s=-1)),('deadline',lambda:self.rows[0].update(request_deadline_s=self.rows[0]['request_deadline_s']+1)),('arrival',lambda:self.rows[0].update(planned_arrival_s=self.rows[0]['planned_arrival_s']+.01)),('window_start',lambda:self.raw.update(measurement_start_s=self.r['actual_arrival_epoch_s']+1)),('window_end',lambda:self.raw.update(measurement_end_s=self.r['actual_arrival_epoch_s']+899))]
  for name,change in mutants:
   with self.subTest(name=name):
    self.setUp();change()
    with self.assertRaises((RuntimeError,ValueError)):self.run_audit()
 def test_idle_uses_original_verifier(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);(d/'manifest.json').write_text('{}');sp={k:dict(path=k,sha256='x') for k in ['original_binding','capacity_binding','config']};sp['host_release']=str(d)
   r=dict(schema='capacity-load-measurement-v1',phase_kind='idle',source=dict(original_binding=sp['original_binding'],capacity_binding=sp['capacity_binding'],config=sp['config'],host_manifest=m.ref(d/'manifest.json')))
   def fixed(ref):return {'identity':{'source':'p6'}} if ref==sp['capacity_binding'] else r
   with patch.object(m,'fixed',fixed),patch.object(m,'idle_result') as old:
    self.assertEqual(m.audit_observation('idle-ref',sp,{'complete':True}),r);old.assert_called_once_with('idle-ref',{'source':'p6'},{'complete':True})
   with patch.object(m,'fixed',fixed),patch.object(m,'idle_result',side_effect=RuntimeError('native busy')):
    with self.assertRaises((RuntimeError,ValueError)):m.audit_observation('idle-ref',sp,{'complete':True})
   r['phase_kind']='load'
   with patch.object(m,'fixed',fixed):
    with self.assertRaises((RuntimeError,ValueError)):m.audit_observation('load-ref',sp,None)
if __name__=='__main__':unittest.main()
