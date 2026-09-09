import ast,copy,importlib.util,json,pathlib,tempfile,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v4',P/'distributed14b_static_v4.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Successor(unittest.TestCase):
 def setUp(self):
  self.parent={'path':'CPU-only-parent','sha256':'5dcae91099369a188f4aecb92f7a8c06b138e6ae0ae181b786f7b7011d034b84'}
  self.q=dict(max_service_frequency_mhz=2100,profile={'path':'CPU-profile'},profile_registration={'registration':{'path':'CPU-registration'}},host_manifest={'path':'CPU-P11'},controller_successor={'path':'CPU-lineage'})
  self.s=dict(max_service_frequency_mhz=2100,profile=self.q['profile'],profile_registration=self.q['profile_registration']['registration'],host_manifest=self.q['host_manifest'],cpu_validation={'path':'CPU-proof'},idle_domain_reacquisition=dict(enabled=True,flag='idle_domain_reacquire_v1',parent_controller_manifest=self.parent))
  self.lineage=dict(schema='distributed14b-P10-to-P11-idle-domain-successor-v1',parent_manifest=self.parent,host_manifest=self.q['host_manifest'],cpu_validation=self.s['cpu_validation'],profile=self.q['profile'],profile_registration=self.s['profile_registration'])
 def run_check(self):
  with patch.object(m,'checked',return_value=self.lineage):m.verify_successor_qualification(self.s,self.q)
 def test_positive(self):self.run_check()
 def test_disabled(self):
  for v in (False,None,1):
   self.s['idle_domain_reacquisition']['enabled']=v
   with self.assertRaises(ValueError):self.run_check()
 def test_flag(self):
  self.s['idle_domain_reacquisition']['flag']='wrong'
  with self.assertRaises(ValueError):self.run_check()
 def test_source(self):
  self.s['host_manifest']={'path':'CPU-other'}
  with self.assertRaises(ValueError):self.run_check()
 def test_parent(self):
  self.s['idle_domain_reacquisition']['parent_controller_manifest']={'path':'CPU-wrong'}
  with self.assertRaises(ValueError):self.run_check()
 def test_cpu(self):
  self.lineage['cpu_validation']={'path':'CPU-wrong'}
  with self.assertRaises(ValueError):self.run_check()
 def test_profile(self):
  self.lineage['profile']={'path':'CPU-wrong'}
  with self.assertRaises(ValueError):self.run_check()
 def test_registration(self):
  self.lineage['profile_registration']={'path':'CPU-wrong'}
  with self.assertRaises(ValueError):self.run_check()
 def test_original_physical_and_full_protocol_AST(self):
  def nodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  old,new=nodes(P.parent/'C/distributed14b_static_v3.py'),nodes(P/'distributed14b_static_v4.py')
  for name in old:
   if name not in ('verify_successor_qualification','load_release'):self.assertEqual(old[name],new[name],name)
if __name__=='__main__':unittest.main(verbosity=2)
