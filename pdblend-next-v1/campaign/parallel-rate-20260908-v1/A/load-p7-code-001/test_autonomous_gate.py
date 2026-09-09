"""CPU tests of fresh autonomous qualification; no GPU or synthetic calibration."""
import ast,copy,importlib.util,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent;A=HERE.parent;R=A.parent
sys.path[:0]=[str(HERE),str(R)]
s=importlib.util.spec_from_file_location('p7_autonomous_gate_under_test',HERE/'capacity_load_calibrate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.fixture
def spec(tmp_path):
 old=json.loads((A/'p6-qualification900-inputs-002/dynamic/spec.json').read_text())
 cp=R/'common/P6-calibration-P7-controller-compatibility-v1.json';proof=m.fixed(m.ref(cp))
 cap=m.fixed(proof['measured_capacity_binding']);cap['controller_calibration_compatibility']=m.ref(cp)
 p=tmp_path/'cap.json';p.write_text(json.dumps(cap))
 cfg=m.fixed(old['config']);cfg.update(capacity_binding_path=str(p),capacity_binding_sha256=m.sha(p))
 c=tmp_path/'config.json';c.write_text(json.dumps(cfg))
 old.update(mode='automatic_underload_gate',host_release=str(R/'hosts/14b-capacity-p7'),controller_calibration_compatibility=m.ref(cp),capacity_binding=m.ref(p),config=m.ref(c),paired_trace_result=m.ref(A/'load-p6-gate-002/underload_gate/result.json'))
 old['trace']=m.fixed(old['paired_trace_result'])['trace'];old['files'].update(proof['files']);old['files'][str(cp)]=m.sha(cp)
 return old

def test_exact_declared_autonomous_gate_and_measured_reuse(spec):
 _,cap,cfg=m.validate_spec(spec)
 assert cfg['capacity_integration_v1'] and cap['calibrated_source_semantics']['candidate_manifest']['path'].endswith('14b-capacity-p6/manifest.json')
 assert spec['host_release'].endswith('14b-capacity-p7')

@pytest.mark.parametrize('change',[dict(cycles=[{}]),dict(manual_restore_at_s=10),dict(arm='fixed2')])
def test_manual_restore_or_reference_arm_cannot_be_autonomous_gate(spec,change):
 spec.update(change)
 with pytest.raises(ValueError):m.validate_spec(spec)

def fixture_evidence():
 inv=dict(initial_ids=['a6','a7'],known_instances=dict(a6=dict(gpus=[6]),a7=dict(gpus=[7]),extra=dict(gpus=[5])),events=[dict(kind='physical_commit',operation='restore'),dict(kind='physical_commit',operation='remove'),dict(kind='capacity_decision',proposal=dict(action='restore'),demand=dict(queued_requests=2))])
 dispatch=[dict(instance_id='extra',request_id='main')]
 control=[dict(kind='admission',request_id='main',client_request_id='1',plan=dict(routes=[dict(decode_id='extra')]))]
 return inv,dispatch,control

def test_added_instance_must_serve_main_workload():
 inv,dispatch,control=fixture_evidence();assert m.autonomous_gate_evidence(inv,dispatch,control,752)['main_native_requests_on_added']==1
 for change in [[],[dict(instance_id='extra',request_id='capacity-correctness')]]:
  with pytest.raises(ValueError):m.autonomous_gate_evidence(inv,change,control,752)

@pytest.mark.parametrize('change',['no_remove','no_queued','wrong_gpu','invalid_main_index'])
def test_lifecycle_workload_or_measured_domain_failure_rejected(change):
 inv,dispatch,control=fixture_evidence()
 if change=='no_remove':inv['events'].pop(1)
 if change=='no_queued':inv['events'][-1]['demand']['queued_requests']=0
 if change=='wrong_gpu':inv['known_instances']['extra']['gpus']=[4]
 if change=='invalid_main_index':control[0]['client_request_id']='752'
 with pytest.raises(ValueError):m.autonomous_gate_evidence(inv,dispatch,control,752)

def test_unmodified_measuring_and_finally_source():
 old=ast.parse((A/'load-p6-code-001/capacity_load_calibrate.py').read_text());new=ast.parse((HERE/'capacity_load_calibrate.py').read_text())
 find=lambda tree,name:next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
 for name in ['generate_trace','validate_trace','phase_metrics','local_deadline','native_idle','measure_idle','main']:
  assert ast.dump(find(old,name),include_attributes=False)==ast.dump(find(new,name),include_attributes=False),name
 oldexecute=find(old,'execute');newexecute=find(new,'execute')
 ot=next(x for x in oldexecute.body if isinstance(x,ast.Try));nt=next(x for x in newexecute.body if isinstance(x,ast.Try))
 assert ast.dump(ast.Module(body=ot.finalbody,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=nt.finalbody,type_ignores=[]),include_attributes=False)
 for name in ['capacity_executor.py','capacity_backend.py','capacity_runtime.py','capacity_planner.py','capacity_certificate.py']:
  assert (HERE/name).read_bytes()==(R/'hosts/14b-capacity-p7'/name).read_bytes()==(A/'load-p6-code-001'/name).read_bytes()
