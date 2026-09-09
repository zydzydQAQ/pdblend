import copy,csv,unittest
from pathlib import Path
import audit_original_timeout_negative_v1 as a
P=a.C/'boundary-baseline-continuation-p4v2-005/dynamollm/parallel-rate-p4-baseline-v2-7b-sharegpt-r3.75-s701-w100-dynamollm-slo1-repeat1/results/cells/parallel-rate-p4-baseline-v2-7b-sharegpt-r3.75-s701-w100-dynamollm-slo1-repeat1/bench.csv'
class TestOriginalBudget(unittest.TestCase):
 def setUp(self):self.rows=list(csv.DictReader(P.open()));self.bad=next(r for r in self.rows if r['success']!='1')
 def test_actual(self):self.assertEqual(a.validate_timeout_rows(self.rows)['unverified_partial_output_requests'],18)
 def test_changed_budget(self):self.bad['request_deadline_s']=str(float(self.bad['request_deadline_s'])+1);self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_early_timeout(self):self.bad['finish_s']=str(float(self.bad['request_deadline_s'])-.01);self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_late_timeout(self):self.bad['finish_s']=str(float(self.bad['request_deadline_s'])+1.01);self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_wrong_error(self):self.bad['error']='HTTP 503';self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_wrong_status(self):self.bad['http_status']='503';self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_starved_arrival(self):self.bad['actual_dispatch_s']=str(float(self.bad['planned_arrival_s'])+1.01);self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
 def test_false_token_claim(self):self.bad['token_count_source']='server_usage';self.assertRaises(ValueError,a.validate_timeout_rows,self.rows)
if __name__=='__main__':unittest.main()
