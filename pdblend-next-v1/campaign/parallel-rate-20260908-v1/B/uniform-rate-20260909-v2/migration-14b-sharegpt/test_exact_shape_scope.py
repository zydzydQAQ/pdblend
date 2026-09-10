"""Run the real frozen verifier's completeness and identity rejection branches on CPU fixtures."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import baseline_verify as native
p=native.p
SOURCE=native.ROOT/'A/uniform-rate-20260909-v2/baseline-preparation/qualification/verify_frequency.py'
v=p.load(SOURCE,'frozen_exact_candidate_scope_test')


class ReachedRawJournals(Exception):pass


class ExactCandidateScope(unittest.TestCase):
    def exercise(self,change):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);profile_ref=p.ref(native.ROOT/'A/uniform-rate-20260909-v2/baseline-preparation/platform-domain/profiles-baseline-domain2100.json')
            profile=p.checked(profile_ref)
            instances=[dict(id='native'+str(g),tp=1,gpus=[g]) for g in range(8)]
            p.save(out/'binding.json',dict(instances=instances,host_release='fixture'));binding_ref=p.ref(out/'binding.json')
            cases=[dict(instance_id=i['id'],frequency=f,input_length=n,batch=b) for i in instances for f,n,b in v.q.shapes(profile,1)]
            self.assertEqual(len(cases),336)
            change(cases)
            state=dict(passed=True,complete=True,finished_s=2,node_lease_held=False,cleanup_errors=[],binding=binding_ref,
                profile=profile_ref,cost_values_recalibrated=False,cases=cases)
            p.save(out/'status.json',state);(out/'power').mkdir();(out/'power/clocks.csv').write_text('t_s,'+','.join('gpu'+str(i)+'_sm_mhz' for i in range(8))+'\n')
            def journals(_):raise ReachedRawJournals()
            audit=SimpleNamespace(identities=lambda *_:None,power=lambda *_:{},lines=journals)
            measurement=ModuleType('ecopadg.serving.measurement');measurement.power_evidence=lambda *_:None
            with patch.object(v.q,'paths',return_value=None),patch.object(v.p,'load',return_value=audit),patch.dict(sys.modules,{'ecopadg.serving.measurement':measurement}):
                return v.verify(out,binding_ref,profile_ref)
    def test_exact_set_reaches_raw_journal_verification(self):
        with self.assertRaises(ReachedRawJournals):self.exercise(lambda rows:None)
    def test_missing_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'candidate set incomplete or duplicated'):self.exercise(lambda rows:rows.pop())
    def test_duplicate_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'candidate set incomplete or duplicated'):self.exercise(lambda rows:rows.append(dict(rows[0])))
    def test_unknown_model_rejected(self):
        binding=dict(files={},model='32b',hostname=native.HOSTNAMES['B'],fresh_legacy_qualification=dict(model='32b',node='B'))
        with patch.object(native.p,'checked',return_value=binding):
            with self.assertRaisesRegex(ValueError,'model/node'):native.verify(dict(path='binding',sha256='fixture'))
    def test_new_binding_cannot_move_qualified_GPU(self):
        boot=dict(hostname=native.HOSTNAMES['B'],model='14b',instances=[dict(id='n',tp=1,gpus=[0])],fresh_node_native_identity=True,old_node_qualification_inherited=False)
        binding=dict(files={},model='14b',hostname=native.HOSTNAMES['B'],instances=[dict(id='n',tp=1,gpus=[1])],fresh_legacy_qualification=dict(model='14b',node='B',old_node_qualification_inherited=False,bootstrap={'path':'boot'},profile={'path':'profile'}))
        def read(ref):return {'binding':binding,'boot':boot,'profile':{}}[ref['path']]
        with patch.object(native.p,'checked',side_effect=read):
            with self.assertRaisesRegex(ValueError,'bootstrap identity differs'):native.verify(dict(path='binding',sha256='fixture'))

if __name__=='__main__':unittest.main()
