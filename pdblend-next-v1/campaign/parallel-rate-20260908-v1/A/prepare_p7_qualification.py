"""Freeze only the P7 autonomous gate, then the original900 candidate after its gate."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent;CODE=A/'load-p7-code-001';HOST=R/'hosts/14b-capacity-p7'
sys.path[:0]=[str(CODE),str(R)]
from capacity_executor import fixed,require,sha,durable
from calibration_compatibility import validate_compatibility

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def prepare(stage):
 require(stage in ('gate','900'),'declared P7 stages only')
 old_status=fixed(ref(A/'p6-qualification900-dynamic-002/status.json'))
 require(not old_status['complete'] and old_status['cleanup_complete'] and not old_status['cleanup_errors'] and not alive(old_status['pid']),'old P6 failed observation must be terminal and cleaned')
 for out in ('p6-qualification-continuation-004','final-p6-continuation-003'):
  s=fixed(ref(A/out/'status.json'));require(not s['complete'] and s['phase']=='stopped_failure' and not alive(s['pid']),'old successor must stop without GPU')
 require(not (A/'final-p6-003').exists(),'failed P6 qualification cannot have formal results')
 cpu=fixed(ref(CODE/'cpu-validation.json'));require(cpu['passed'] and not cpu['gpu_qualified'] and all(sha(p)==h for p,h in cpu['files'].items()),'actual new driver CPU source differs')
 manifest=fixed(ref(CODE/'manifest.json'));require(all(sha(CODE/p)==h for p,h in manifest['files'].items()),'P7 driver package changed')
 compatibility=ref(R/'common/P6-calibration-P7-controller-compatibility-v1.json');proof=fixed(compatibility)
 require(compatibility['sha256']=='da1df6aa2d00cab52caef332048141a04122e32eb9f1335a5fc5d81870c37555','frozen explicit P6 numerical/P7 controller proof changed')
 p7cpu=ref(R/'cpu-validation-p7.json');require(p7cpu['sha256']=='78cbc9a8212d85051d958f2457d0f6927215114dae350b9a173c66c5ae63f8be' and fixed(p7cpu)['passed'],'common P7 CPU differs')
 oldspec=fixed(ref(A/'p6-qualification900-inputs-002/dynamic/spec.json'))
 out=A/('p7-autonomous-gate-001' if stage=='gate' else 'p7-qualification900-dynamic-001')
 inputs=A/('p7-autonomous-gate-inputs-001' if stage=='gate' else 'p7-qualification900-inputs-001')
 require(not inputs.exists() and not out.exists(),'fresh immutable P7 invocation required')
 gate_evidence=None
 if stage=='900':
  from p7_qualification_audit import audit_gate
  gate_evidence=audit_gate()
  require(gate_evidence['passed'],'fresh autonomous gate required')
 cap=fixed(proof['measured_capacity_binding'])
 files=dict(cap['files']);files.update(oldspec['files']);files.update(proof['files']);files.update(cpu['files'])
 hostmanifest=fixed(ref(HOST/'manifest.json'))
 files.update({str(HOST/p):h for p,h in hostmanifest['files'].items()})
 references=[compatibility,p7cpu,ref(HOST/'manifest.json'),ref(CODE/'manifest.json'),ref(CODE/'cpu-validation.json'),ref(Path(__file__)),ref(A/'p6-qualification900-dynamic-002/status.json'),ref(A/'p6-qualification900-dynamic-002/qualification900/result.json')]
 references += [ref(p) for p in CODE.glob('*.py')]
 if gate_evidence is not None:
  references.append(ref(A/'p7_qualification_audit.py'))
  write(inputs/'autonomous-gate-audit.json',gate_evidence)
  references.append(ref(inputs/'autonomous-gate-audit.json'));files.update(gate_evidence['files'])
 for rr in references:files[rr['path']]=rr['sha256']
 cap.update(owner_id='ap7gate' if stage=='gate' else 'ap7q900',http_port_base=31500 if stage=='gate' else 31600,
    runtime_dir=str(out/'runtime'),controller_calibration_compatibility=compatibility,files=files)
 write(inputs/'capacity-binding.json',cap);capref=ref(inputs/'capacity-binding.json')
 config=fixed(oldspec['config']);config.update(port=31550 if stage=='gate' else 31650,capacity_binding_path=capref['path'],capacity_binding_sha256=capref['sha256'])
 write(inputs/'config.json',config);configref=ref(inputs/'config.json')
 trace=oldspec['trace']
 if stage=='gate':trace=fixed(ref(A/'load-p6-gate-002/underload_gate/result.json'))['trace']
 work=dict(schema='P7-autonomous-qualification-work-v1',authorized=True,stage=stage,created_s=time.time(),trace=trace,source=ref(HOST/'manifest.json'),profile=oldspec['profiles'],certificate=cap['calibration'],capacity_binding=capref,controller_calibration_compatibility=compatibility,initial_instances=2,automatic_retries=False,manual_restore_used=False,arrival_duration_s=60 if stage=='gate' else 900,request_timeout_s=120,all8gpu_energy=True,formal_performance_not_inferred=True,reused_P6_measurements_not_relabelled=True)
 write(inputs/'work-declaration.json',work)
 spec=copy.deepcopy(oldspec)
 for name in ('cold_start_after_s','paired_failed_result','requires_full_zero_failure_short_gate','input_declaration'):spec.pop(name,None)
 spec.update(mode='automatic_underload_gate' if stage=='gate' else 'qualification900',arm='dynamic',trace=trace,config=configref,capacity_binding=capref,host_release=str(HOST),api_base='http://127.0.0.1:'+str(config['port']),controller_calibration_compatibility=compatibility,input_declaration=ref(inputs/'work-declaration.json'),cpu_evidence=p7cpu,stop_path=str(A/'STOP_P7_QUALIFICATION'),files=dict(files))
 if stage=='gate':spec['paired_trace_result']=ref(A/'load-p6-gate-002/underload_gate/result.json')
 else:spec['required_autonomous_gate_audit']=ref(inputs/'autonomous-gate-audit.json')
 for rr in [configref,capref,trace,ref(inputs/'work-declaration.json'),*([spec['paired_trace_result']] if stage=='gate' else [spec['required_autonomous_gate_audit']])]:spec['files'][rr['path']]=rr['sha256']
 validate_compatibility(spec,cap)
 write(inputs/'spec.json',spec);sr=ref(inputs/'spec.json')
 declaration=dict(work,specs=dict(dynamic=sr),output=str(out),parent_original900_declaration=ref(A/'p6-qualification900-inputs-002/declaration.json'))
 write(inputs/'declaration.json',declaration)
 print(json.dumps(dict(stage=stage,spec=sr,out=str(out),declaration=ref(inputs/'declaration.json'))))
 return sr,out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['gate','900']);prepare(p.parse_args().stage)
