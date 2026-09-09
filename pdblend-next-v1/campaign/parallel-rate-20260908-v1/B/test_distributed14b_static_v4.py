import ast,copy,importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('subject',HERE/'distributed14b_static_v4.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
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
class SourceChecks(unittest.TestCase):
    def test_original_raw_math_unchanged(self):
        def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        old,new=nodes(HERE/'distributed14b_static_v1.py'),nodes(HERE/'distributed14b_static_v4.py')
        for name in ('flat','arrivals','engineering','boundary_rate'):self.assertEqual(old[name],new[name])
    def test_original_executor_invocation(self):
        tree=ast.parse((HERE/'distributed14b_static_v4.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='common.run_one']
        self.assertEqual(len(calls),1);self.assertEqual([ast.unparse(a) for a in calls[0].args],['session','binding','row','output','hardware']);self.assertFalse(calls[0].keywords)
    def test_baseline_cannot_enter_pdb_boundary_runner(self):
        with tempfile.TemporaryDirectory() as out:
            p=pathlib.Path(out)/'release.json';p.write_text(json.dumps(dict(schema='distributed14b-static-release-v2',files={},jobs=m.ref(R/'distributed-14b-v1/B-jobs.json'),node='B',model='14b',system='mixed',binding=m.ref(R/'B/distributed-14b-v1/pdb-deployment-001/binding-base.json'))))
            with self.assertRaisesRegex(ValueError,'PDB only'):m.load_release(p)
    def test_hardware_capture_is_readonly(self):
        tree=ast.parse((HERE/'distributed14b_static_v4.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='hardware_identity')
        strings=[n.value for n in ast.walk(f) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertIn('--query-gpu=index,name,uuid',strings);self.assertFalse(any(x in strings for x in ('--lock-gpu-clocks','--reset-gpu-clocks','docker')))
if __name__=='__main__':unittest.main(verbosity=2)

