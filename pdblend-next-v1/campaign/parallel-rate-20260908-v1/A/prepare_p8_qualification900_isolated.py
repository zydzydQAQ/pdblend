"""A new full900 observation after an explicitly preserved primary-power failure."""
import copy,hashlib,json,time
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent;CODE=A/'load-p8-isolated-code-002';ADAPTER=A/'isolated-power-v1';OUT=A/'p8-qualification900-inputs-002';RUN=A/'p8-qualification900-dynamic-002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def checked(r):
 assert sha(r['path'])==r['sha256'];return json.loads(Path(r['path']).read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 assert not OUT.exists() and not RUN.exists()
 oldref=ref(A/'p8-qualification900-inputs-001/spec.json');old=checked(oldref)
 statusref=ref(A/'p8-qualification900-dynamic-001/status.json');status=checked(statusref)
 assert status['complete'] is False and status['cleanup_complete'] and not status['cleanup_errors'] and status['error']=="ValueError('whole-node phase power invalid')"
 from p8_qualification_audit_v2 import audit_gate,prior
 assert not prior.alive(status['pid']);gate=audit_gate();assert gate['passed']
 cpu=checked(ref(CODE/'cpu-validation.json'));assert cpu['passed'] and all(sha(p)==h for p,h in cpu['files'].items())
 adapterref=ref(ADAPTER/'manifest.json');adapter=checked(adapterref)
 files={**old['files'],**cpu['files'],**adapter['files']}
 for r in [oldref,statusref,ref(A/'diagnosis-p8-primary-power-001/diagnosis.json'),ref(CODE/'manifest.json'),ref(CODE/'cpu-validation.json'),adapterref,ref(__file__)]:files[r['path']]=r['sha256']
 for p in (A/'p8-qualification900-dynamic-001').rglob('*'):
  if p.is_file():files[str(p)]=sha(p)
 cap=checked(old['capacity_binding']);cap.update(owner_id='ap8q900b',http_port_base=32500,kv_port_base=62000,runtime_dir=str(RUN/'runtime'),files=files)
 write(OUT/'capacity-binding.json',cap);capref=ref(OUT/'capacity-binding.json')
 cfg=checked(old['config']);cfg.update(port=32550,capacity_binding_path=capref['path'],capacity_binding_sha256=capref['sha256'])
 write(OUT/'config.json',cfg);cfgref=ref(OUT/'config.json')
 write(OUT/'autonomous-gate-audit.json',gate)
 work=dict(schema='P8-original900-isolated-sampler-reobservation-v1',authorized=True,created_s=time.time(),original_spec=oldref,original_failed_status=statusref,diagnosis=ref(A/'diagnosis-p8-primary-power-001/diagnosis.json'),arrival_duration_s=900,original_request_deadline_s=120,source=ref(R/'hosts/14b-capacity-p8/manifest.json'),profile=old['profiles'],trace=old['trace'],certificate=cap['calibration'],measurement_adapter=adapterref,control_policy_and_certificate_unchanged=True,all8_50hz_field186=True,original250ms_and_age_thresholds_unchanged=True,old_primary_power_invalid_not_relabelled=True,automatic_retries=False)
 write(OUT/'work-declaration.json',work)
 sp=copy.deepcopy(old);sp.update(config=cfgref,capacity_binding=capref,api_base='http://127.0.0.1:32550',input_declaration=ref(OUT/'work-declaration.json'),required_autonomous_gate_audit=ref(OUT/'autonomous-gate-audit.json'),measurement_adapter=adapterref,measurement_cpu_evidence=ref(CODE/'cpu-validation.json'),measurement_driver=ref(CODE/'capacity_load_calibrate.py'),prior_failed_power_observation=statusref,files=dict(files))
 for r in [capref,cfgref,sp['input_declaration'],sp['required_autonomous_gate_audit']]:sp['files'][r['path']]=r['sha256']
 write(OUT/'spec.json',sp);sr=ref(OUT/'spec.json');write(OUT/'declaration.json',dict(work,specs=dict(dynamic=sr),output=str(RUN)))
 print(json.dumps(dict(spec=sr,out=str(RUN),declaration=ref(OUT/'declaration.json'))))
if __name__=='__main__':main()
