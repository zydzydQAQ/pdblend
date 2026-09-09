"""CPU rejection probes for the append-only historical continuation."""
import importlib.util,copy,json,time,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('historical',B/'historical_scale11_v2.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=m.contract();cases=[]
def check(name,fn,reject=False):
 try:fn()
 except (RuntimeError,ValueError):
  if not reject:raise
  cases.append(dict(name=name,passed=True,rejected=True))
 else:
  assert not reject,name+' should reject';cases.append(dict(name=name,passed=True))
check('exact_11_original_rows_and_sources',m.contract)
original_read=m.p.read
for name,mut in [('changed_SLO',lambda x:x['cells'][0]['slo'].update(ttft_s=999)),('changed_trace',lambda x:x['cells'][0].update(trace_sha256='0'*64)),('rerun_completed_row',lambda x:x['cells'].__setitem__(0,x['cells'][1])),('added_repeat',lambda x:x['cells'].append(x['cells'][0])),('absolute_deadline',lambda x:x.update(deadline_s=99999999999))]:
 changed=copy.deepcopy(d);mut(changed);m.p.read=lambda p,x=changed:x if Path(p)==m.DECL else original_read(p)
 check(name,m.contract,True)
m.p.read=original_read
chain=m.fixed.load(m.CHAIN,'historical_cpu_chain');c=chain.contract();old=m.p.checked(d['original_binding']);fresh=copy.deepcopy(old);fresh.update(deadline_s=None,campaign_lifecycle='until_declared_complete_v1')
check('exact_historical_policy_lifecycle_only',lambda:m.compatible(c,old,fresh))
bad=copy.deepcopy(fresh);bad['instances'][0]['tp']=1
check('reject_changed_TP_layout',lambda:m.compatible(c,old,bad),True)
bad=copy.deepcopy(fresh);bad['model']='7b'
check('reject_wrong_model',lambda:m.compatible(c,old,bad),True)
check('reject_expired_old_binding',lambda:m.compatible(c,old,old),True)
for name,state in [('failed_predecessor',dict(pid=123,complete=False,node_lease_held=False)),('live_predecessor',dict(pid=123,complete=True,node_lease_held=False))]:
 m.p.read=lambda _:state;m.alive=lambda _:name=='live_predecessor';check(name,m.prerequisite,True)
m.p.read=original_read
out=dict(schema=1,passed=True,cpu_only=True,gpu_executed=False,created_s=time.time(),cases=cases,source={'path':str(B/'historical_scale11_v2.py'),'sha256':hashlib.sha256((B/'historical_scale11_v2.py').read_bytes()).hexdigest()},declaration={'path':str(m.DECL),'sha256':hashlib.sha256(m.DECL.read_bytes()).hexdigest()},inherited_request_failure_gate='execution-completion-p4-002/runner.py engineering_gate and baseline-completion-cpu-validation.json; no mechanism change')
(B/'historical-scale11-cpu-validation-v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(cases)}))
