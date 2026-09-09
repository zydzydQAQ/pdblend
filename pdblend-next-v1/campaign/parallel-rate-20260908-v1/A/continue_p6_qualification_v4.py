"""Resume the unexecuted dynamic900 only after an audited fixed2 capacity negative.

The earlier all-full paired qualification remains failed. This new declaration
qualifies only the dynamic candidate's complete lifecycle; the reference arm is
not an equal-work energy comparison. No serving source, trace or policy changes.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
A=Path(__file__).resolve().parent
CODE=A/'load-p6-code-001'
sys.path.insert(0,str(A))
import continue_p6_qualification_v3 as prior
OUT=A/'p6-qualification-continuation-004'

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 prior.require(not OUT.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ,'new CPU-only continuation required')
 old=A/'p6-qualification-continuation-003/status.json';oldstatus=prior.fixed(prior.ref(old))
 prior.require(not prior.alive(oldstatus['pid']) and oldstatus['phase']=='stopped_failure' and not oldstatus['complete'],'old strict paired900 failure must be terminal and retained')
 oldformal=prior.fixed(prior.ref(A/'final-p6-continuation-002/status.json'))
 prior.require(not prior.alive(oldformal['pid']) and oldformal['phase']=='stopped_failure' and not oldformal['complete'],'old formal successor must have stopped without GPU')
 prior.require(not (A/'final-p6-002').exists(),'no formal measurement under failed all-full gate')
 declaration=prior.fixed(prior.ref(A/'p6-qualification900-inputs-002/declaration.json'))
 source=A/'fixed900_negative_audit_v1.py';negative=load(source,'fixed900_negative')
 fixed=negative.audit(A/'p6-qualification900-fixed2-002',declaration['specs']['fixed2'],declaration)
 cpu_ref=prior.ref(A.parent/'C/A-fixed900-negative-cpu-validation-001.json');cpu=prior.fixed(cpu_ref)
 prior.require(cpu_ref['sha256']=='a470d500f58cee99b69ff15192ac8ba98da6e6c84f31343f41b824e607b53d1e' and cpu['passed'] and cpu['source']==prior.ref(source) and fixed==prior.fixed(cpu['real_audit']),'independent frozen negative audit/CPU proof differs')
 prior.require(fixed['passed'] and fixed['measurement_valid'] and not fixed['full_work'] and not fixed['equal_work_energy_comparison_eligible'],'fixed2 must be a proven reference-only capacity negative')
 sr=declaration['specs']['dynamic'];sp=prior.fixed(sr);out=A/'p6-qualification900-dynamic-002'
 prior.require(not out.exists(),'declared dynamic900 must be unexecuted, never retried')
 if '--dry-run' in sys.argv:
  print(json.dumps(dict(cpu_only=True,reference_negative_audited=True,fixed_requests=fixed['n_expected'],fixed_failed=fixed['failed_requests'],dynamic_spec=sr,hardware_actions=False,output_created=False)))
  return
 OUT.mkdir();state=dict(pid=os.getpid(),started_s=time.time(),phase='fixed_reference_audited',complete=False,node_lease_held=False,steps=[],candidate_only_qualification=True,reference_all_full_gate_failed=True)
 def update(**v):state.update(v,updated_s=time.time());prior.durable(OUT/'status.json',state)
 fixedref=OUT/'fixed-reference-audit.json';prior.durable(fixedref,fixed)
 files={str(p):prior.sha(p) for p in (Path(__file__).resolve(),Path(prior.__file__),source,CODE/'capacity_load_calibrate.py',A/'p6-qualification900-inputs-002/declaration.json',old,A/'final-p6-continuation-002/status.json',fixedref)}
 declaration2=dict(schema='p6-candidate900-after-fixed-reference-negative-v1',authorized=True,created_s=time.time(),files=files,previous_all_full_gate_status=prior.ref(old),previous_formal_stopped_status=prior.ref(A/'final-p6-continuation-002/status.json'),original_pair_declaration=prior.ref(A/'p6-qualification900-inputs-002/declaration.json'),fixed_reference_audit=prior.ref(fixedref),dynamic_spec=sr,dynamic_output=str(out),same_original_trace_source_profile_capacity_policy=True,candidate_requires_complete_zero_failure=True,reference_equal_work_energy_comparison_eligible=False,formal100_fresh_required=True,automatic_retries=False)
 files[cpu_ref['path']]=cpu_ref['sha256'];files[cpu['real_audit']['path']]=cpu['real_audit']['sha256']
 declaration2['independent_reference_cpu']=cpu_ref
 prior.durable(OUT/'declaration.json',declaration2);prior.durable(OUT/'source-manifest.json',files);update()
 try:
  prior.require(not (A/'STOP_P6_QUALIFICATION900').exists(),'STOP prevents dynamic900')
  prior.require(all(prior.sha(p)==h for p,h in files.items()),'new continuation inputs changed')
  update(phase='dynamic')
  with (OUT/'dynamic.log').open('xb') as log:
   child=subprocess.Popen([sys.executable,'-u',str(CODE/'capacity_load_calibrate.py'),'--spec',sr['path'],'--spec-sha256',sr['sha256'],'--out',str(out),'--run'],stdout=log,stderr=subprocess.STDOUT)
   update(child_pid=child.pid);code=child.wait()
  state['steps'].append(dict(phase='dynamic',exitcode=code));state.pop('child_pid',None);update()
  prior.require(code==0,'dynamic900 failed; no automatic retry or formal successor')
  status=prior.audit(out,1,sr);dynamic=prior.audit900(out,status,declaration,'dynamic')
  prior.require(dynamic['full_work'] and dynamic['failed_requests']==0 and dynamic['request_timeouts']==0,'candidate900 requires all requested work')
  negative.validate_evidence(prior.ref(fixedref))
  prior.durable(OUT/'qualification.json',dict(schema='p6-candidate900-qualified-reference-negative-v1',passed=True,created_s=time.time(),source=declaration['source'],profile=declaration['profile'],certificate=declaration['certificate'],capacity_binding=declaration['capacity_binding'],trace=declaration['trace'],arms=dict(fixed2=fixed,dynamic=dynamic),fixed_reference_audit=prior.ref(fixedref),prior_all_full_gate=prior.ref(old),continuation_declaration=prior.ref(OUT/'declaration.json'),candidate_requires_complete_zero_failure=True,reference_observation_only=True,equal_work_energy_comparison_eligible=False,initial_instances=2,actual_growth_and_return=True,formal_100s_performance_not_inferred=True,development_only=True))
  update(phase='complete',complete=True)
 except BaseException as exc:update(phase='stopped_failure',error=repr(exc));raise
 finally:update(finished_s=time.time())

if __name__=='__main__':main()
