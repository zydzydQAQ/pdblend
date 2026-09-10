import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import baseline_scope_successor_v2 as s

class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.plan=s.p.read(s.HERE/'plan-009.json')

    def test_real_cpu_only_predecessor_accepted(self):
        prior,decision=s.validate_handoff(self.plan)
        self.assertEqual(len(prior['observations']),7)
        self.assertTrue(decision['pdb_boundary_complete'])

    def test_another_failure_and_changed_observations_rejected(self):
        original=s.p.checked
        for field,value in [('error','unclassified GPU failure'),('observations',[])]:
            def checked(ref):
                result=original(ref)
                if ref==self.plan['reviewed_preflight_failure']:
                    result=copy.deepcopy(result);result[field]=value
                return result
            with self.subTest(field=field),patch.object(s.p,'checked',side_effect=checked),self.assertRaises(ValueError):
                s.validate_handoff(self.plan)

    def test_hardware_action_evidence_rejected(self):
        original=Path.exists
        target=Path(self.plan['failed_preflight_operation']['path']).parent/'deployment/creation-intents'
        with patch.object(Path,'exists',lambda path:True if path==target else original(path)),self.assertRaisesRegex(ValueError,'hardware action'):
            s.validate_handoff(self.plan)

    def test_new_ports_nonoverlapping_outside_ephemeral_and_source_exact(self):
        low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
        ranges=[[30100+i,*range(31000+32*i,31000+32*(i+1))] for i in range(8)]
        flat=sum(ranges,[])
        self.assertEqual(len(flat),len(set(flat)))
        self.assertTrue(all(not low<=port<=high for port in flat))
        new=(s.HERE/'baseline_producer_v3.py').read_text()
        reversed_source=new.replace('(30200, 31400) if hetero else (30100, 31000)',
                                    '(39200, 60000) if hetero else (39000, 59000)')
        self.assertEqual(reversed_source,(s.HERE/'baseline_producer_v2.py').read_text())

if __name__=='__main__':unittest.main()
