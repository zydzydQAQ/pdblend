"""Freeze the single-domain fixed2/dynamic900 pair from an actual complete P6 certificate."""
from pathlib import Path
import json,hashlib,copy,sys,time
A=Path(__file__).resolve().parent;ROOT=A.parent;HOST=ROOT/'hosts/14b-capacity-p6';CODE=A/'load-p6-code-001'
sys.path[:0]=[str(CODE),str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
from capacity_executor import fixed,durable,require,sha
from capacity_certificate import ref,validate
from capacity_runtime import calibration_model,load_planner

def main():
 out=A/'p6-qualification900-inputs-001';require(not out.exists(),'new900 declaration required')
 certref=ref(A/'alpaca-capacity-certificate-p6-001/certificate.json');cert=fixed(certref)
 original_spec=fixed(ref(A/'load-p6-full-inputs-002/spec.json'));base=fixed(original_spec['original_binding']);cap=fixed(original_spec['capacity_binding'])
 validate(cert,cap['identity']);require(cert['identity']==cap['identity'],'same candidate source required')
 cpu_path=ROOT/'cpu-validation-p6.json';cpu=fixed(ref(cpu_path));require(cpu.get('passed') is True,'P6 CPU proof required')
 manifest=fixed(ref(HOST/'manifest.json'));require(manifest['model']=='14b' and manifest['default_enabled'] is False,'P6 parent-preserving candidate required')
 require(sha(HOST/'manifest.json')=='6d4e854b166f79ae6a8a4ffe495fc723ca3e99dfea363d8d9a7ede28cf120ab5','frozen P6 manifest differs')
 require(sha(cpu_path)=='a77b1a1ff97cf1021e497eaa96ab5c1f6b331cfeb049ab516b1903892bf9ac1c','frozen P6 CPU proof differs')
 out.mkdir()
 cap.update(calibration=certref,calibration_only=False,production_ready=False,owner_id='ap6q900',http_port_base=31200,kv_port_base=62000,max_creations=8,runtime_dir=str(A/'p6-qualification900-dynamic-001/runtime'),rate_observation_window_s=60.,arrival_count_margin=2.,policy={})
 cap['files'][str(HOST/'manifest.json')]=sha(HOST/'manifest.json');cap['files'][str(cpu_path)]=sha(cpu_path);cap['files'][str(A/'prepare_p6_qualification900.py')]=sha(A/'prepare_p6_qualification900.py')
 durable(out/'capacity-binding.json',cap);capref=ref(out/'capacity-binding.json');calibration_model(load_planner(cap['planner_source']),cap)
 config=fixed(original_spec['config']);require(config['allow_pd'] is False and config['prepare_peers'] is False and config['transfers']==[],'independent mixed exact parent configuration required')
 trace=ref(A/'load-alpaca-inputs-001/qualification900-trace.json');t=fixed(trace);require(t['duration_s']==900 and t['demand_domain_sha256']==cap['demand_domain']['sha256'],'exact single-domain900 required')
 base['files'].update(cap['files']);base['files'][certref['path']]=certref['sha256'];base['files'][trace['path']]=trace['sha256'];base['files'][str(out/'capacity-binding.json')]=capref['sha256'];base['files'][original_spec['profiles']['path']]=original_spec['profiles']['sha256']
 for r in cert['raw_measurements']:
  value=fixed(r);base['files'][r['path']]=r['sha256'];base['files'].update(value['artifacts'])
 durable(out/'execution-binding.json',base)
 specs={}
 for arm in ('fixed2','dynamic'):
  d=out/arm;d.mkdir();cfg=copy.deepcopy(config);cfg.update(capacity_integration_v1=arm=='dynamic',capacity_binding_path=capref['path'],capacity_binding_sha256=capref['sha256'])
  durable(d/'config.json',cfg)
  sp=copy.deepcopy(original_spec);sp.update(mode='qualification900',arm=arm,original_binding=ref(out/'execution-binding.json'),capacity_binding=capref,config=ref(d/'config.json'),trace=trace,host_release=str(HOST),stop_path=str(A/'STOP_P6_QUALIFICATION900'),cpu_evidence=ref(cpu_path),actual_certificate=certref)
  sp.pop('cycles');sp['files'].update(base['files']);sp['files'][str(d/'config.json')]=sha(d/'config.json');sp['files'][str(CODE/'capacity_load_calibrate.py')]=sha(CODE/'capacity_load_calibrate.py');durable(d/'spec.json',sp);specs[arm]=ref(d/'spec.json')
 durable(out/'declaration.json',dict(schema='P6-A12-qualification900-pair-v1',created_s=time.time(),specs=specs,source=ref(HOST/'manifest.json'),profile=original_spec['profiles'],capacity_binding=capref,certificate=certref,trace=trace,initial_instances=2,only_measured_extra_gpus=[5],window_s=900,phase_seconds=[300,300,300],request_timeout_s=120,no_global_deadline=True,formal_100s_eligible=False,qualification_cannot_imply_100s_slo90=True))
 print(json.dumps(specs))
if __name__=='__main__':main()
