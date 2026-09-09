"""Synthetic reference-contract counterexamples; never actual A55 evidence."""
import copy,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import audit_eco_observations_v2 as a

class References(unittest.TestCase):
 def setUp(self):
  self.docs={};self.hashes={}
  def ref(name,doc):
   path='/synthetic/'+name;self.docs[path]=doc;self.hashes[path]=name;return dict(path=path,sha256=name)
  self.scope=ref('scope.json',{})
  self.rules=ref('rules.json',dict(schema='A-Eco-fixed100-arrival-engineering-v1',scope=self.scope,arrival_window_s=100,request_timeout_s=120,cleanup_local_budget_s=90,all8gpu_power=True,max_dispatch_lateness_s=1,p99_dispatch_lateness_s=.1,max_handler_lateness_s=1))
  config=ref('config.json',dict(strategy='ecoserve',journal='old'))
  source=ref('source.py',{})
  files={r['path']:r['sha256'] for r in (config,source)}
  self.base=dict(instances=[dict(id='x')],host_release='/synthetic/host',correctness_evidence='/synthetic/gate',mechanism_proof=dict(complete=True),deployment_receipt='/synthetic/deployment',configs=dict(alpaca=config['path']),files=files)
  base_ref=ref('base.json',self.base)
  frozen=copy.deepcopy(self.base);frozen.update(output='/synthetic/unused',files=dict(files));frozen['files'].update({r['path']:r['sha256'] for r in (self.rules,self.scope,base_ref)})
  self.frozen_ref=ref('frozen.json',frozen)
  self.release=dict(schema='A-Eco-scoped-qualified-execution-release-v1',declaration=self.scope,model='14b',ready_for_gpu=True,execution_rules=self.rules,host_release=self.base['host_release'],source_package=dict(path=str(a.a_scope.PACKAGE),sha256=a.a_scope.PACKAGE_SHA),files=copy.deepcopy(frozen['files']),binding=self.frozen_ref,qualification_binding=base_ref)
  self.release_ref=ref('release.json',self.release)
  self.actual=copy.deepcopy(frozen);self.actual['output']='/synthetic/results';self.actual['files'].update({self.release_ref['path']:self.release_ref['sha256'],self.scope['path']:self.scope['sha256']})
  self.cp=dict(declaration=self.scope,execution_rules=self.rules)
  def need(ok,msg):
   if not ok:raise ValueError(msg)
  def checked(ref):need(self.hashes[ref['path']]==ref['sha256'],'reference changed');return self.docs[ref['path']]
  self.p=SimpleNamespace(need=need,checked=checked,read=lambda path:self.docs[str(path)],sha=lambda path:self.hashes[str(path)])
  self.proof=dict(request_count=55,legacy_gate_passed=True,raw_gate_recomputed=True)
  self.module=SimpleNamespace(qualification=lambda reference:(self.base,copy.deepcopy(self.proof)))
  self.mock=patch.object(a.a_scope,'contract',return_value=(self.module,{}));self.mock.start();self.addCleanup(self.mock.stop);a._cache.clear()
 def execute(self):return a.a_execution(self.p,self.cp,self.actual)
 def test_synthetic_reference_shape_is_compatible(self):self.assertTrue(self.execute()['independently_recomputed'])
 def test_release_model_changed(self):
  self.release['model']='7b'
  with self.assertRaises(ValueError):self.execute()
 def test_unqualified_release(self):
  self.release['ready_for_gpu']=False
  with self.assertRaises(ValueError):self.execute()
 def test_release_source_file_unpinned(self):
  del self.actual['files']['/synthetic/source.py']
  with self.assertRaises(ValueError):self.execute()
 def test_release_source_hash_changed(self):
  self.hashes['/synthetic/source.py']='changed'
  with self.assertRaises(ValueError):self.execute()
 def test_unapproved_binding_mutation(self):
  self.actual['new_setting']=True
  with self.assertRaises(ValueError):self.execute()
 def test_frozen_binding_hash_changed(self):
  self.hashes[self.frozen_ref['path']]='changed'
  with self.assertRaises(ValueError):self.execute()
 def test_source_package_changed(self):
  self.release['source_package']['sha256']='changed'
  with self.assertRaises(ValueError):self.execute()
 def test_original_request_deadline_changed(self):
  self.docs[self.rules['path']]['request_timeout_s']=121
  with self.assertRaises(ValueError):self.execute()
 def test_54_requests_rejected(self):
  self.proof['request_count']=54
  with self.assertRaises(ValueError):self.execute()
 def test_legacy_false_rejected(self):
  self.proof['legacy_gate_passed']=False
  with self.assertRaises(ValueError):self.execute()
 def test_raw_gate_not_recomputed_rejected(self):
  self.proof['raw_gate_recomputed']=False
  with self.assertRaises(ValueError):self.execute()

if __name__=='__main__':unittest.main()
