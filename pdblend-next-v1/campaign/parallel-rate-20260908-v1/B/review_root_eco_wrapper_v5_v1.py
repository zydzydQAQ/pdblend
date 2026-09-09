"""Three actual old observations plus CPU-only A declaration counterexamples."""
import hashlib,json,os,socket,sys,time,traceback,copy
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parent;sys.path.insert(0,str(R))
import final_selected_collect_v4 as collect
import final_selected_baseline_v5 as current
import final_selected_baseline_v4 as previous
import audit_eco_observations_v2 as eco
inputs=json.loads((B/'root-eco-wrapper-v5-inputs-v1.json').read_text());audit=collect.load_audit();p=audit.p
out=B/'independent-root-Eco-wrapper-v5-review-v1';out.mkdir(exist_ok=False)
record=dict(schema='B-independent-root-Eco-wrapper-v5-review-v1',physical_host=socket.gethostname(),pid=os.getpid(),started_s=time.time(),inputs=p.ref(B/'root-eco-wrapper-v5-inputs-v1.json'),source=p.ref(__file__),points=[],counterexamples=[],A_actual_fresh55_status='pending_no_actual_A_Eco_observation_or_qualification',complete=False)
p.write(out/'status.json',record)
try:
 for path,h in inputs['files'].items():p.need(p.sha(path)==h,'review source changed '+path)
 for item in inputs['points']:
  cp=p.checked(item['checkpoint']);row=cp['row'];actual=current.inspect(audit,row,Path(item['checkpoint']['path']),item['declaration']);old=p.checked(item['prior_audit']);oldpoint=next(e['point'] for e in old['points'] if e['cell_id']==row['cell_id'])
  keys=['measurement_valid','work_complete','slo_attainment','energy_j','failure_classification','classified_native_queue_refusals','controller_source_sha256','profile_sha256','policy_sha256','completed_work_requests','generated_tokens']
  p.need(all(actual.get(k)==oldpoint.get(k) for k in keys),'three-point regression differs '+row['cell_id'])
  p.need(actual['measurement_valid'] is True,'actual regression failed '+str(actual.get('error')))
  record['points'].append(dict(checkpoint=item['checkpoint'],prior_audit=item['prior_audit'],passed=True,point=actual));p.write(out/'status.json',record);print(json.dumps(dict(cell=row['cell_id'],passed=True,work_complete=actual['work_complete'],classification=actual['failure_classification'])),flush=True)
 # Synthetic metadata guards only; never treated as native evidence.
 scope_ref=dict(path='/synthetic/scope.json',sha256='scope');logical=dict(path='/synthetic/logical.json',sha256='logical');append=dict(path='/synthetic/append.json',sha256='append');row=dict(cell_id='eco-test',repeat=2)
 scope=dict(schema='A-Eco-final-PDB-execution-scope-v1',logical_declaration=logical,append_declarations=[append],required_cells=[row])
 class Fake:
  need=staticmethod(p.need)
  checked=staticmethod(lambda ref: scope if ref==scope_ref else p.need(False,'wrong synthetic ref'))
 cp=dict(row=copy.deepcopy(row),declaration=scope_ref,logical_declaration=logical)
 for source in (logical,append,scope_ref):
  got=current.declaration_chain(Fake,cp,row,source);p.need(got==scope_ref,'correct scope chain rejected');record['counterexamples'].append(dict(case='exact-scoped-row-'+source['sha256'],passed=True,synthetic=True))
 for name,mutator in [('wrong_logical',lambda d:d.update(logical_declaration=append)),('changed_row',lambda d:d['row'].update(repeat=1))]:
  for source in (logical,append,scope_ref):
   changed=copy.deepcopy(cp);mutator(changed)
   try:current.declaration_chain(Fake,changed,row,source)
   except (ValueError,KeyError):record['counterexamples'].append(dict(case=name+'-'+source['sha256'],passed=True,synthetic=True))
   else:raise ValueError('invalid A chain accepted '+name+'-'+source['sha256'])
 record.update(complete=True,passed=True,finished_s=time.time())
except BaseException as exc:record.update(complete=False,passed=False,error=repr(exc),traceback=traceback.format_exc(),finished_s=time.time());print(traceback.format_exc(),flush=True)
p.write(out/'status.json',record);print(json.dumps(dict(complete=record['complete'],passed=record['passed'],points=len(record['points']),counterexamples=len(record['counterexamples']),report=p.ref(out/'status.json'))),flush=True)
