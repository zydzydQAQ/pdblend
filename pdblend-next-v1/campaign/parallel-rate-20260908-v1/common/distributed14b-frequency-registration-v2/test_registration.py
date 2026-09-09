"""Synthetic CPU aggregation tests, never submitted as measured profiles."""
import copy,math,sys,unittest
from pathlib import Path
import registration as r
sys.path[:0]=[str(r.ROOT/'hosts/14b-capacity-p9/src'),'/root/workspace/pdblend/.runtime-deps']
v=r.load(Path(next(iter(r.DRIVERS))).parent/'validate.py','profile_geometry_fixture')
REFERENCE=r.ROOT.parent/'main-slo-improvement-v1/A/long-batch6-profile-001/profiles.development.json'


def fixture():
    profile=r.read(REFERENCE);rows=[]
    binding=dict(instances=[dict(id='synthetic6',tp=1,gpus=[6]),dict(id='synthetic7',tp=1,gpus=[7])])
    for p in v.declared_points(profile,binding):
        rows.append(dict(point=p,passed=True,same_shape_interference_only=True,output_sha256='synthetic-output',
            late_context=dict(per_request={str(i):dict(attention_after_min=p['context_tokens']) for i in range(p['batch'])}),
            full_decode_runs=[dict(finish_spacing_values_s=[.01,.02],target_gpu_mean_power_w=180.)],
            prefill_phases=[dict(duration_s=.4,target_gpu_power_w=170.)],idle_residency=dict(target_gpu_power_w=80.),
            measurement_references=dict(raw=dict(path='synthetic-never-published-'+p['point_id'],sha256='0'*64))))
    transitions=[dict(gpu=g,source_mhz=a,target_mhz=b,actual_duration_s=.08,all8_energy_j=30.)
                 for g in (6,7) for a in (900,1500,2100) for b in (900,1500,2100) if a!=b]
    return profile,rows,transitions

def derive(values):
    return r.aggregate(*values,source_reference=dict(path='synthetic',sha256='0'*64),hostname='synthetic',node='CPU',measurement_reference=dict(path='synthetic',sha256='0'*64))

class Tests(unittest.TestCase):
    def test_complete_empirical_two_cards(self):
        original,rows,transitions=fixture();p,domains,costs=derive((original,rows,transitions))
        self.assertEqual(len(domains),14);self.assertEqual(len(costs),6)
        self.assertEqual(sum(x['frequency_mhz']==2100 for x in p['points']),28)
        self.assertTrue(all(x['frequency_mhz']<=2100 for x in p['points']+p['interference_points']))
        self.assertEqual([x for x in p['points'] if x['frequency_mhz']<=1500],[x for x in original['points'] if x['frequency_mhz']<=1500])
        self.assertEqual({x['role'] for x in p['points'] if x['frequency_mhz']==2100},{'mixed','decode'})
        self.assertTrue(all(x['samples']==2 for x in p['points'] if x['frequency_mhz']==2100))
        self.assertTrue(p['frequency_registration']['empirical_estimates_not_hard_guarantees'])
    def reject(self,mutation):
        values=fixture();mutation(*values)
        with self.assertRaises((ValueError,KeyError,TypeError)):derive(values)
    def test_missing_point(self):self.reject(lambda p,r,t:r.pop())
    def test_same_gpu_twice(self):self.reject(lambda p,r,t:r[1]['point'].update(gpu=6))
    def test_repeat_claim_three(self):self.reject(lambda p,r,t:r[1]['point'].update(repeat=3))
    def test_changed_geometry(self):self.reject(lambda p,r,t:[x['point'].update(context_tokens=9000) for x in r[:2]])
    def test_failed_sample(self):self.reject(lambda p,r,t:r[0].update(passed=False))
    def test_fake_cross_shape_interference(self):self.reject(lambda p,r,t:r[0].update(same_shape_interference_only=False))
    def test_wrong_outputs(self):self.reject(lambda p,r,t:r[0].update(output_sha256='different'))
    def test_command_only_frequency(self):self.reject(lambda p,r,t:r[0]['point'].update(frequency_mhz=2520))
    def test_shortened_micro_output(self):self.reject(lambda p,r,t:r[0]['point'].update(output_tokens=64))
    def test_insufficient_shared_context(self):self.reject(lambda p,r,t:next(iter(r[0]['late_context']['per_request'].values())).update(attention_after_min=2559))
    def test_missing_decode(self):self.reject(lambda p,r,t:r[0]['full_decode_runs'][0].update(finish_spacing_values_s=[]))
    def test_negative_decode(self):self.reject(lambda p,r,t:r[0]['full_decode_runs'][0].update(finish_spacing_values_s=[-.1]))
    def test_nonfinite_prefill(self):self.reject(lambda p,r,t:r[0]['prefill_phases'][0].update(duration_s=float('inf')))
    def test_nonfinite_power(self):self.reject(lambda p,r,t:r[0]['prefill_phases'][0].update(target_gpu_power_w=float('inf')))
    def test_missing_switch(self):self.reject(lambda p,r,t:t.pop())
    def test_wrong_switch_direction(self):self.reject(lambda p,r,t:t[0].update(target_mhz=2520))
    def test_switch_same_gpu(self):self.reject(lambda p,r,t:t[6].update(gpu=6))
    def test_nonfinite_switch_cost(self):self.reject(lambda p,r,t:t[0].update(actual_duration_s=float('nan')))
    def test_zero_switch_energy(self):self.reject(lambda p,r,t:t[0].update(all8_energy_j=0.))
    def test_original_geometries_are_not_relabelled(self):
        old=r.load(r.HERE.parent/'distributed14b-frequency-profile-v1/validate.py','original2400_geometry')
        self.assertEqual(v.geometry(r.read(REFERENCE)),old.geometry(r.read(REFERENCE)))
        self.assertEqual((v.HERE/'context_export.py').read_bytes(),(old.HERE/'context_export.py').read_bytes())
        self.assertEqual((v.HERE/'source_order.py').read_bytes(),(old.HERE/'source_order.py').read_bytes())
    def test_changed_ref(self):
        with self.assertRaises(ValueError):r.fixed(dict(path=str(REFERENCE),sha256='0'*64))
    def test_published_profile_requires_exact_object(self):
        source=Path(r.__file__).read_text();self.assertIn("fixed(profile_reference)==profile",source)
    def test_public_verify_does_not_import_parent_modules(self):
        before={k:id(v) for k,v in sys.modules.items() if k.startswith('ecopadg')}
        with self.assertRaisesRegex(ValueError,'independent saved-evidence verifier rejected'):
            r.verify(dict(path=str(REFERENCE),sha256='0'*64))
        self.assertEqual(before,{k:id(v) for k,v in sys.modules.items() if k.startswith('ecopadg')})
    def test_fresh_process_contract(self):
        source=Path(r.__file__).read_text();self.assertIn("sys.executable,'-I'",source)
        self.assertIn("relative=str(source.relative_to(host.resolve()))",source)
    def test_partial_measurement_cannot_register(self):
        source=Path(r.__file__).read_text();self.assertIn("status['complete'] and status['measurement_valid']",source)
        self.assertIn("not status['errors']",source)

if __name__=='__main__':unittest.main()
