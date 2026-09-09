import copy, importlib.util, json, pathlib, unittest
HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('subject',HERE/'distributed14b_static_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
R=HERE.parent
OUT=R/'A/p4-minimal/fixed-screen-001/results'
ID='parallel-rate-p4-fixed2-14b-sharegpt-r1.5-s701-w100-pdblend-slo1-repeat1'
class Checks(unittest.TestCase):
    def setUp(self):
        self.receipt=json.loads((OUT/'operations'/ID/'receipt.json').read_text());self.timing=m.arrivals(self.receipt,OUT)
    def test_real_complete(self):self.assertTrue(m.engineering(self.receipt,self.timing)['passed'])
    def test_real_arrivals(self):self.assertTrue(self.timing['passed'])
    def test_slo_low_valid(self):
        self.receipt['summary']['slo_attainment']=.5
        self.assertTrue(m.engineering(self.receipt,self.timing)['passed']);self.assertEqual(m.boundary_rate(self.receipt),.5)
    def test_failure_stops(self):
        self.receipt['summary']['failed_requests']=1;self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_timeout_stops(self):
        self.receipt['summary']['request_timeouts']=1;self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_missing_counts_stops(self):
        del self.receipt['summary']['failed_requests'];self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_incomplete_stops(self):
        self.receipt['summary']['work_complete']=False;self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_native_invalid_stops(self):
        self.receipt['measurement_valid']=False;self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_late_stops(self):
        self.timing=dict(passed=False,errors=['late']);self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_partial_count_stops(self):
        self.receipt['summary']['completed_work_requests']-=1;self.assertFalse(m.engineering(self.receipt,self.timing)['passed'])
    def test_nonfinite_slo_stops(self):
        self.receipt['summary']['slo_attainment']=float('nan')
        with self.assertRaises(ValueError):m.boundary_rate(self.receipt)
    def test_exact_mapping(self):
        jobs=json.loads((R/'distributed-14b-v1/B-jobs.json').read_text())
        rows=[m.flat(c) for c in jobs['pdb_cells']];self.assertEqual(len(rows),12)
        for c,r in zip(jobs['pdb_cells'],rows):
            self.assertEqual({k:v for k,v in r.items() if k not in ('cell_id','repeat')},{k:v for k,v in c['source_row'].items() if k not in ('cell_id','repeat')})
            self.assertEqual(r['cell_id'],c['cell_id']);self.assertEqual(r['repeat'],c['repeat'])
if __name__=='__main__':unittest.main()
