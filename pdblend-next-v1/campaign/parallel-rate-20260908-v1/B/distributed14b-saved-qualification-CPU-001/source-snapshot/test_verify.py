"""Real B saved fixtures plus in-memory rejection cases; no physical actions."""
import copy, importlib.util, json, unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
R=HERE.parent.parent
sp=importlib.util.spec_from_file_location('qualification_test_subject',HERE/'verify.py')
q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)

class SavedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=R/'B/distributed-14b-v1'
        cls.b=q.read(cls.root/'pdb-deployment-001/binding-base.json')
        cls.host=q.ref(R/'hosts/14b-capacity-p9/manifest.json')
        cls.files=dict(cls.b['files'])
        cls.vref=q.ref(R/'common/distributed14b-profile-validation-v1/validate.py')
        cls.files[cls.vref['path']]=cls.vref['sha256']
        for directory in ('pdb-ordinary-001','profile-validation-001','frequency2400-feasibility-001'):
            for path in (cls.root/directory).rglob('*'):
                if path.is_file():cls.files[str(path)]=q.sha(path)
            status=q.read(cls.root/directory/'status.json')
            if 'spec' in status:cls.files.update(q.checked(status['spec'])['files'])
        cls.v=q.module(cls.vref,cls.files)
        cls.oref=q.ref(cls.root/'pdb-ordinary-001/status.json')
        cls.status=q.checked(cls.oref)
        cls.raw_path=cls.root/'frequency2400-feasibility-001/gpu6-mid16-2400/raw.json'
        cls.raw=q.read(cls.raw_path)
        cls.events=[json.loads(s) for s in Path(cls.raw['events']['path']).read_text().splitlines()]
        cls.power,cls.clocks=q.power_operation(cls.root/'frequency2400-feasibility-001',q.read(cls.root/'frequency2400-feasibility-001/status.json'),cls.host,cls.files)

    def test_real_native_ordinary_raw(self):
        x=q.ordinary(self.oref,self.b,self.host,self.v,self.files)
        self.assertEqual(x['requests'],4)
        self.assertGreater(x['raw_power']['energy_j'],0)

    def test_real2400_saved_wrapper(self):
        status=q.ref(self.root/'frequency2400-feasibility-001/status.json')
        case=dict(status=status,binding=q.checked(status)['binding'],raw=q.ref(self.raw_path),point_id=self.raw['point']['point_id'],validator=self.vref)
        x=q.frequency_case(case,self.b,self.host,self.files)
        self.assertEqual(x['point']['frequency_mhz'],2400)
        self.assertTrue(x['independently_recomputed'])

    def test_real2520_unstable_still_rejected(self):
        status=q.ref(self.root/'profile-validation-001/status.json')
        path=self.root/'profile-validation-001/gpu6-mid16-2520/raw.json'
        case=dict(status=status,binding=q.checked(status)['binding'],raw=q.ref(path),point_id='gpu6-mid16-2520',validator=self.vref)
        with self.assertRaisesRegex(ValueError,'loaded SM outside'):q.frequency_case(case,self.b,self.host,self.files)

    def test_reference_requires_frozen_closure(self):
        with self.assertRaisesRegex(ValueError,'outside frozen'):q.bound(self.oref,{})

    def test_reference_modified_digest(self):
        bad=dict(self.oref,sha256='0'*64)
        with self.assertRaises(ValueError):q.bound(bad,{bad['path']:bad['sha256']})

    def ordinary_bad(self,modify):
        value=copy.deepcopy(self.status);modify(value)
        original=q.read
        def altered(path):return value if str(path)==self.oref['path'] else original(path)
        with patch.object(q,'read',side_effect=altered),self.assertRaises(ValueError):q.ordinary(self.oref,self.b,self.host,self.v,self.files)

    def test_missing_reply(self):self.ordinary_bad(lambda x:x['ordinary']['replies'].pop())
    def test_missing_output(self):self.ordinary_bad(lambda x:x['ordinary']['replies'][0]['response']['token_ids'].pop())
    def test_wrong_output(self):self.ordinary_bad(lambda x:x['ordinary']['replies'][0]['response']['token_ids'].__setitem__(0,-1))
    def test_native_cleanup_false(self):self.ordinary_bad(lambda x:x.__setitem__('clock_restore_complete',False))

    def raw_bad(self,modify):
        raw=copy.deepcopy(self.raw);modify(raw)
        with self.assertRaises(ValueError):self.v.validate_point(raw,self.events,self.clocks)

    def test_output_budget_not_relaxed(self):self.raw_bad(lambda x:x['requests'][0]['output_token_ids'].pop())
    def test_native_generation_not_relaxed(self):self.raw_bad(lambda x:x['runtime_after_requests'].__setitem__('acknowledged_generation',-1))
    def test_unconfirmed_send_not_relaxed(self):self.raw_bad(lambda x:x['cleanup']['proof']['transfers'][0].__setitem__('inflight_sends',1))
    def test_shape_not_relabelled(self):self.raw_bad(lambda x:x['point'].__setitem__('input_tokens',2304))
    def test_clock_band_not_relaxed(self):
        bad=[(t,[2520]*8) for t,_ in self.clocks]
        with self.assertRaisesRegex(ValueError,'loaded SM outside'):self.v.validate_point(self.raw,self.events,bad)

if __name__=='__main__':unittest.main(verbosity=2)
