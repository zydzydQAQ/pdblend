"""CPU proof replay and negative coverage for fresh baseline/source qualifications."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import baseline_verify as v
import meter_binding as meter
p=v.p


class NativeAndPlatform(unittest.TestCase):
    def test_both_native_source_transformations_exact(self):
        ref=p.ref(v.ROOT/'A/uniform-rate-20260909-v2/baseline-preparation/qualification/native-source-equivalence.json')
        self.assertEqual(len(v.native_source(ref,False)),3)
        self.assertEqual(len(v.native_source(ref,True)),3)
    def test_platform_all_runtime_bytes_replay(self):
        for name in ('runtime-baseline-002','runtime-baseline-eco-drain-002'):
            files={};v.platform(v.ROOT/'A/uniform-rate-20260909-v2/baseline-preparation/platform-domain'/name,files)
            self.assertGreater(len(files),100)
    def test_terminal_owner_refuses_active_or_failed(self):
        owner=dict(complete=True,finished_s=4,node_lease_held=False,pid=123)
        with patch.object(p,'active_owner',return_value=False):v.terminal_owner(owner)
        with patch.object(p,'active_owner',return_value=True):
            with self.assertRaises(ValueError):v.terminal_owner(owner)
        with patch.object(p,'active_owner',return_value=False):
            for key,value in [('complete',False),('error','failure'),('node_lease_held',True)]:
                bad=dict(owner,**{key:value})
                with self.assertRaises(ValueError):v.terminal_owner(bad)


class ExistingCollector(unittest.TestCase):
    def fixture(self,root):
        collector=p.checked(p.ref(v.ROOT/'common/token-evidence-v2/manifest.json'))
        host=root/'host';client=host/meter.CLIENT;client.parent.mkdir(parents=True)
        client.write_bytes(Path(collector['collector']['path']).read_bytes())
        p.save(host/'manifest.json',dict(files={meter.CLIENT:p.sha(client)}))
        binding=root/'native-binding.json';p.save(binding,dict(host_release=str(host),files={},output='old',model='14b',system='mixed'))
        validator=root/'native-validator.py';validator.write_text("def verify(ref):\n return dict(passed=True,independently_recomputed=True,binding=ref,files={})\n")
        return p.ref(binding),p.ref(validator),client
    def test_existing_new_collector_wraps_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,validator,client=self.fixture(root);before=p.sha(client)
            ref=meter.prepare(binding,root/'wrapped',validator);result=meter.verify(ref)
            self.assertTrue(result['passed']);self.assertEqual(p.sha(client),before)
            wrapped=p.checked(result['binding']);self.assertEqual(wrapped['host_release'],str(root/'host'))
    def test_collector_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binding,validator,client=self.fixture(root)
            ref=meter.prepare(binding,root/'wrapped',validator);client.write_text('changed')
            with self.assertRaises(ValueError):meter.verify(ref)

if __name__=='__main__':unittest.main()
