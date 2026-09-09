"""Read-only incremental B raw audit; never owns or changes the GPU queue."""
import hashlib,json,os,socket,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent;sys.path.insert(0,str(R))
import final_selected_collect_v4 as collect
import final_selected_baseline_v4 as actual
import final_baseline_overlay_v2 as overlays
import source_identity_v4 as identity
audit=collect.load_audit();p=audit.p
out=B/'eco-drain37-v1/independent-tail-audit';out.mkdir(exist_ok=False)
record=dict(schema='B-Eco35-independent-incremental-raw-audit-v1',physical_host=socket.gethostname(),pid=os.getpid(),started_s=time.time(),source=p.ref(Path(__file__)),points=[],complete=False,read_only=True)
p.write(out/'status.json',record)
inputs=p.read(B/'root-eco-observation-review-inputs-v1.json')
record['root_auditor_sources']={path:sha for path,sha in inputs['files'].items() if Path(path).parent==R and path.endswith('.py')}
for path,sha in record['root_auditor_sources'].items():p.need(p.sha(path)==sha,'root audited source changed '+path)
originals=p.original_points();cache={}
for old in originals:
 if old['model']=='32b' and old['system']=='ecoserve':
  key=(old['executed_source']['binding_path'],old['dataset'])
  if key not in cache:cache[key]=identity.original_identity(old)
  old.update(cache[key])
overlay=overlays.validate(p,dict(baseline_overlay=p.read(B/'root-eco-overlay-review-inputs-v1.json')['overlay']),originals)
ref=p.ref(B/'eco-drain37-v1/declaration.json');decl=p.checked(ref);record['declaration']=ref
previous_ref=p.ref(B/'root-Eco-observations-independent-review-001/status.json');prior=p.checked(previous_ref)
prior_rows=[entry for entry in prior['points'] if entry['model']=='32b'];p.need(len(prior_rows)==15 and all(e['passed'] for e in prior_rows),'first15 not independently verified')
rows={row['cell_id']:row for row in decl['cells']};seen=set()
for entry in prior_rows:
 p.checked(entry['checkpoint']);p.need(entry['cell_id'] in rows,'prior observation outside exact declared35')
 record['points'].append(dict(cell_id=entry['cell_id'],checkpoint=entry['checkpoint'],passed=True,point=entry['point'],reused_independent_observation=previous_ref));seen.add(entry['cell_id'])
record['prior_audit']=previous_ref;record['verified_count']=len(seen);p.write(out/'status.json',record)
print(json.dumps(dict(started=True,pid=os.getpid(),host=socket.gethostname(),prior_verified=len(seen))),flush=True)
while True:
 for cid,row in rows.items():
  cp_path=B/'eco-drain37-v1/performance/results/checkpoints'/(cid+'.json')
  if cid in seen or not cp_path.exists():continue
  entry=dict(cell_id=cid,checkpoint=p.ref(cp_path),started_s=time.time())
  try:
   point=actual.inspect(audit,row,cp_path,ref);overlays.annotate(point,overlay);overlays.verify_fresh_identity(p,point,overlay,originals)
   entry.update(point=point,passed=point['measurement_valid'] is True,error=point.get('error'))
  except BaseException as exc:entry.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
  entry['finished_s']=time.time();record['points'].append(entry);seen.add(cid);record['verified_count']=sum(e['passed'] for e in record['points']);p.write(out/'status.json',record)
  print(json.dumps(dict(cell_id=cid,passed=entry['passed'],error=entry.get('error'),verified_count=record['verified_count'])),flush=True)
  if not entry['passed']:
   record.update(complete=False,passed=False,requires_review=True,finished_s=time.time());p.write(out/'status.json',record);raise SystemExit(2)
 status=p.read(B/'eco-drain37-v1/performance/status.json')
 if len(seen)==len(rows) and status.get('complete') is True and status.get('node_lease_held') is False:
  record.update(complete=True,passed=True,finished_s=time.time(),performance_terminal=p.ref(B/'eco-drain37-v1/performance/status.json'));p.write(out/'status.json',record);print(json.dumps(dict(complete=True,verified_count=record['verified_count'])),flush=True);break
 if status.get('failed'):
  record.update(complete=False,passed=False,requires_review=True,performance_failure=status['failed'],finished_s=time.time());p.write(out/'status.json',record);raise SystemExit(2)
 time.sleep(15)
