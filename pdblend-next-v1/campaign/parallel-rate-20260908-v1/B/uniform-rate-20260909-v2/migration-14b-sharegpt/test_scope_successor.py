import copy
import tempfile
import unittest
from pathlib import Path
import audit_sharegpt_scope as a


class ScopeTests(unittest.TestCase):
    def configs(self):
        parent=dict(strategy='pdblend-joint',capacity_integration_v1=False,slo_tpot_s=.1,
                    instances=[dict(id='g'+str(g),tp=1,gpus=[g],role='mixed') for g in (6,7)])
        b=copy.deepcopy(parent);b['slo_tpot_s']=.15
        old=copy.deepcopy(b);old.pop('capacity_integration_v1')
        return old,b,parent

    def test_absent_capacity_matches_original_false(self):
        self.assertEqual(a.compare_configs(*self.configs())['parent_changed_fields'],['slo_tpot_s'])

    def test_changed_capacity_strategy_and_geometry_rejected(self):
        for key,value in [('capacity_integration_v1',True),('strategy','mixed'),('idle_domain_reacquire_v1',True)]:
            x,b,p=self.configs();b[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):a.compare_configs(x,b,p)
        x,b,p=self.configs();b['instances'][0]['gpus']=[5]
        with self.assertRaises(ValueError):a.compare_configs(x,b,p)

    def test_unexplained_parent_change_rejected(self):
        x,b,p=self.configs();b['unknown_policy_change']=True
        with self.assertRaises(ValueError):a.compare_configs(x,b,p)

    def test_control_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots=[Path(temp)/k for k in ('a','b')]
            for root in roots:
                (root/'benchmarks/scripts').mkdir(parents=True)
                (root/'runtime.py').write_text('original')
                (root/'benchmarks/scripts/bench_vllm.py').write_text(root.name)
            self.assertEqual(a.compare_sources(*roots)['identical_control_and_support_files'],1)
            (roots[1]/'runtime.py').write_text('changed')
            with self.assertRaises(ValueError):a.compare_sources(*roots)


if __name__=='__main__': unittest.main()
