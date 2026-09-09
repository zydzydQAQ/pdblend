"""Synthetic CPU counterexamples; these values are never registered profiles."""
import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
import validate as v
sys.path[:0]=[str(v.ROOT/'hosts/14b-capacity-p9/src'),'/root/workspace/pdblend/.runtime-deps']

def fixture():
    old=v.load(v.PARENT.parent/'test_validate.py','profile_cpu_native_fixture')
    raw,_,_=old.fixture();p=raw['point'];p.update(frequency_mhz=2100,output_tokens=256,context_tokens=256,arrival_offsets_s=[0.])
    row=raw['requests'][0];row.update(requested_output_tokens=256,output_token_ids=list(range(256)),
        token_received_s=[1.14+.05*j for j in range(256)],arrival_offset_s=0.,planned_arrival_s=1.,stream_end_s=14.)
    row['usage']['completion_tokens']=256;raw.update(started_s=1.,finished_s=14.5,
        idle_residency=dict(start_s=-2.5,end_s=-.4,native=copy.deepcopy(raw['runtime_before'])))
    events=[dict(started_s=1.10+.05*j,finished_s=1.14+.05*j,request_ids=[row['request_id']],role='mixed',mode='continuous',generation=3,
                 prefill=int(j==0),decode=int(j!=0),tokens=128 if j==0 else 1) for j in range(256)]
    clocks=[(-3+j*.02,[2100]*8) for j in range(950)];power=[(t,[100.]*8) for t,_ in clocks]
    return raw,events,power,clocks

class Tests(unittest.TestCase):
    def test_real_reference_fourteen_geometries(self):
        profile=v.read(v.ROOT.parent/'main-slo-improvement-v1/A/long-batch6-profile-001/profiles.development.json')
        shapes=v.geometry(profile);self.assertEqual(len(shapes),14);self.assertIn(dict(input_tokens=128,batch=2,context_tokens=256),shapes)
        self.assertIn(dict(input_tokens=7168,batch=6,context_tokens=7680),shapes)
        binding=dict(instances=[dict(id='a6',tp=1,gpus=[6]),dict(id='a7',tp=1,gpus=[7])])
        points=v.declared_points(profile,binding);self.assertEqual(len(points),28)
        self.assertEqual({p['gpu'] for p in points},{6,7});self.assertTrue(all(p['output_tokens']==1024 for p in points))
    def test_profile_positive(self):self.assertTrue(v.validate_profile_point(*fixture(),source_order_verified=True)['passed'])
    def reject(self,fn):
        values=fixture();fn(*values)
        with self.assertRaises((ValueError,KeyError)):v.validate_profile_point(*values,source_order_verified=True)
    def test_unknown_source(self):
        with self.assertRaises(ValueError):v.validate_profile_point(*fixture())
    def test_full_context_not_inferred(self):self.reject(lambda r,e,p,c:r['point'].update(context_tokens=1024))
    def test_native_tail(self):self.reject(lambda r,e,p,c:e.pop())
    def test_failed_work(self):self.reject(lambda r,e,p,c:r['requests'][0].update(success=False))
    def test_bad_ack(self):self.reject(lambda r,e,p,c:r['runtime_after_requests'].update(acknowledged_generation=2))
    def test_native_cleanup(self):self.reject(lambda r,e,p,c:r['cleanup'].update(complete=False))
    def test_prefill_phase_actual_low(self):
        self.reject(lambda r,e,p,c:c.__setitem__(206,(c[206][0],[2360]*8)))
    def test_prefill_energy_missing_gpu(self):self.reject(lambda r,e,p,c:p.__setitem__(206,(p[206][0],[100.]*7)))
    def test_phase_no_bracket(self):
        with self.assertRaises(ValueError):v.phase_clock([(1,[2100]*8),(1.1,[2100]*8)],6,2100,.9,1.05)
    def test_phase_gap(self):
        with self.assertRaises(ValueError):v.phase_clock([(0,[2100]*8),(.3,[2100]*8)],6,2100,.01,.29)
    def test_phase_tolerance(self):
        self.assertEqual(v.phase_clock([(0,[2085]*8),(.1,[2115]*8)],6,2100,.01,.09)['min_mhz'],2085)
    def test_request_io_original_byte_identity(self):
        self.assertEqual((v.HERE/'stream.py').read_bytes(),(v.ROOT/'A/frequency2400-short16-code-001/stream.py').read_bytes())
    def test_source_order_original_byte_identity(self):
        for name in ('source_order.py','source-order-contract.json','context_export.py'):
            self.assertEqual((v.HERE/name).read_bytes(),(v.ROOT/'A/frequency2400-short16-code-001'/name).read_bytes())
    def test_retained_native_and_meter_original_delegate(self):
        tree=ast.parse((v.HERE/'run.py').read_text());text=(v.HERE/'run.py').read_text()
        self.assertIn('deploy.Measurement(out,hardware,status)',text)
        self.assertIn('common.restore(session,instance)',text)
        self.assertIn("hooks.completed_artifacts(roots,spec['host_manifest'],spec['measurement_adapter'])",text)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('set_frequency','set_clock') for n in ast.walk(tree)))

class SwitchTests(unittest.TestCase):
    def fixture(self,directory):
        import frequency_switch as f
        raw,_,power,clocks=fixture();row=raw['requests'][0]
        row.update(output_token_ids=list(range(1024)),token_received_s=[1.14+.01*j for j in range(1024)],requested_output_tokens=1024,stream_end_s=12.)
        row['usage']['completion_tokens']=1024
        raw.update(complete=True,gpu=6,instance_id='i6',reference=dict(output_token_ids=list(range(32))))
        events=[dict(started_s=1.10+.01*j,finished_s=1.108+.01*j,request_ids=[row['request_id']],role='mixed',mode='continuous',generation=3,
                     prefill=int(j==0),decode=int(j!=0),tokens=128 if j==0 else 1) for j in range(1024)]
        path=Path(directory)/'events.jsonl';path.write_text(''.join(json.dumps(e)+'\n' for e in events));raw['events']=v.ref(path)
        raw['switches']=[]
        for j,(source,target) in enumerate(f.pairs()):
            a=2.+j*.2;b=a+.1;state=copy.deepcopy(raw['runtime_before']);state.update(active=1,running=1)
            raw['switches'].append(dict(source_mhz=source,target_mhz=target,started_s=a,finished_s=b,
                source_observed=[dict(at_s=a-.08+k*.02,actual_sm_mhz=[source]) for k in range(3)],
                target_observed=[dict(at_s=a+.02+k*.02,actual_sm_mhz=[target]) for k in range(3)],
                actual_runtime_before=state,applied_after={'6':target}))
        return raw,power,clocks
    def test_switch_positive(self):
        import frequency_switch as f
        with tempfile.TemporaryDirectory() as d:self.assertEqual(len(f.derive(*self.fixture(d))['switches']),6)
    def reject(self,change):
        import frequency_switch as f
        with tempfile.TemporaryDirectory() as d:
            values=self.fixture(d);change(values[0])
            with self.assertRaises((ValueError,KeyError)):f.derive(*values)
    def test_missing_direction(self):self.reject(lambda r:r['switches'].pop())
    def test_wrong_observed_clock(self):self.reject(lambda r:r['switches'][0]['target_observed'][2].update(actual_sm_mhz=[2384]))
    def test_not_actual_active(self):self.reject(lambda r:r['switches'][0]['actual_runtime_before'].update(active=0))
    def test_incomplete_output(self):self.reject(lambda r:r['requests'][0]['output_token_ids'].pop())
    def test_missing_native_cleanup(self):self.reject(lambda r:r['cleanup'].update(complete=False))
    def test_relabel_target(self):self.reject(lambda r:r['switches'][0]['applied_after'].update({'6':2520}))
    def test_transition_after_output(self):self.reject(lambda r:r['switches'][0].update(finished_s=15.))

if __name__=='__main__':unittest.main()
