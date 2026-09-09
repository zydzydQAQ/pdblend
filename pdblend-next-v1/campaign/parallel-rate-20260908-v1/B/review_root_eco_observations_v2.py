"""Independent read-only actual terminal Eco audit, executed on physical B CPU."""
import hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent;sys.path.insert(0,str(R))
import final_selected_collect_v4 as collect
import final_selected_baseline_v4 as actual
import final_baseline_overlay_v2 as overlay_module
import source_identity_v4 as identity
inputs=json.loads((B/'root-eco-observation-review-inputs-v1.json').read_text())
audit=collect.load_audit();p=audit.p
out=B/'root-Eco-observations-independent-review-002';out.mkdir(exist_ok=False)
record=dict(schema='independent-root-Eco-observations-review-v1',physical_host=socket.gethostname(),pid=os.getpid(),started_s=time.time(),inputs=p.ref(B/'root-eco-observation-review-inputs-v1.json'),review_source=p.ref(Path(__file__)),cpu_tests=dict(path=str(R/'test_audit_eco_observations_v1.py'),sha256=p.sha(R/'test_audit_eco_observations_v1.py'),passed=16),points=[],complete=False)
p.write(out/'status.json',record)
print(json.dumps(dict(phase='verifying_read_only_inputs',host=record['physical_host'],pid=record['pid'],files=len(inputs['files']),checkpoints=len(inputs['checkpoints']))),flush=True)
try:
 for path,digest in inputs['files'].items():p.need(p.sha(path)==digest,'review input differs '+path)
 originals=p.original_points();cache={}
 for old in originals:
  if old['system']!='ecoserve' or old['model'] not in ('7b','32b'):continue
  key=(old['executed_source']['binding_path'],old['dataset'])
  if key not in cache:cache[key]=identity.original_identity(old)
  old.update(cache[key])
 overlay=overlay_module.validate(p,dict(baseline_overlay=p.read(B/'root-eco-overlay-review-inputs-v1.json')['overlay']),originals)
 declarations={model:p.ref(R/f'{node}/eco-drain37-v1/declaration.json') for model,node in [('7b','C'),('32b','B')]}
 rows={model:{row['cell_id']:row for row in p.checked(ref)['cells']} for model,ref in declarations.items()}
 record['previous_run']=p.ref(B/'root-Eco-observations-independent-review-001/status.json')
 record['previous_C_errors_classification']='auditor_environment_missing_dependency; no GPU defect or remeasurement implied'
 record['environment']=p.ref(B/'cpu-review-venv-v1/environment.json')
 for cp_ref in inputs['checkpoints']:
  if p.checked(cp_ref)['row']['model']!='7b':continue
  started=time.time();cp=p.checked(cp_ref);row=cp['row'];entry=dict(checkpoint=cp_ref,cell_id=row['cell_id'],model=row['model'],started_s=started)
  try:
   exact=rows[row['model']][row['cell_id']]
   point=actual.inspect(audit,exact,Path(cp_ref['path']),declarations[row['model']])
   overlay_module.annotate(point,overlay);overlay_module.verify_fresh_identity(p,point,overlay,originals)
   entry.update(point=point,passed=point['measurement_valid'] is True,error=point.get('error'))
  except BaseException as exc:entry.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
  entry['finished_s']=time.time();record['points'].append(entry);p.write(out/'status.json',record)
  print(json.dumps(dict(cell=entry['cell_id'],passed=entry['passed'],error=entry.get('error'),seconds=entry['finished_s']-started,n=len(record['points']))),flush=True)
 record.update(complete=True,passed=all(x['passed'] for x in record['points']),finished_s=time.time(),count=len(record['points']))
except BaseException as exc:
 record.update(complete=False,passed=False,error=repr(exc),traceback=traceback.format_exc(),finished_s=time.time());print(traceback.format_exc(),flush=True)
p.write(out/'status.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ['points','traceback']}),flush=True)
