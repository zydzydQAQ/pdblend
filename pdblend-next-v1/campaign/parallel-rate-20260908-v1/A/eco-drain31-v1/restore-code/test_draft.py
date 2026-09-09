"""Pure CPU review; synthetic evidence cannot authorize a deployment."""
import ast,copy,importlib.util,json,tempfile,time
from pathlib import Path
D=Path(__file__).resolve().parent
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load(D/'restore_adapter.py','a_draft_test');parent=load(D/'restore_parent.py','a_parent_test')
old=Path('/root/workspace/pdblend-next-v1/campaign/A14B-resident-restore-v3/restore.py')
def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
x,y=funcs(old),funcs(D/'restore_parent.py');names=['Limit','command','http','idle','native','ready','static_container','verify_stopped','verify_restarted','archive_runtime','verify_prefixes','energy_evidence']
assert all(x[n]==y[n] for n in names)
checks=[dict(name='original_native_identity_hardware_cleanup_primitives_AST_unchanged',passed=True)]
assert 719<parent.Limit(time.time()+720).left()<=720
try:parent.Limit(None)
except RuntimeError:checks.append(dict(name='finite_local_budget_required',passed=True))
else:raise AssertionError('unbounded local limit accepted')
orig_a=ast.parse(Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/A/restore_baselines_p4_v2.py').read_text())
new_a=ast.parse((D/'restore_adapter.py').read_text())
def nested(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='native_dispatch')
assert ast.dump(nested(orig_a),include_attributes=False)==ast.dump(nested(new_a),include_attributes=False)
checks.append(dict(name='original_v3_native_dispatch_AST_unchanged',passed=True))
with tempfile.TemporaryDirectory(prefix='A-restore-cpu-') as td:
 root=Path(td);cpdir=root/'checkpoints';cpdir.mkdir();cid='synthetic-cell'
 def save(name,value):
  p=root/name;p.write_text(json.dumps(value));return a.ref(p)
 marker=save('marker.json',{'synthetic':True});gate=save('gate.json',dict(passed=True,errors=[]));decl=save('declaration.json',dict(cells=[dict(cell_id=cid)]))
 binding=save('binding.json',dict(model='14b',system='pdblend',files={marker['path']:marker['sha256']}))
 def fixture(summary_changes=None,receipt_changes=None,status_changes=None,capacity_changes=None):
  summary=dict(work_complete=True,failed_requests=0,request_timeouts=0,admission_rejections=0,post_measurement_cleanup=dict(cleanup_complete=True));summary.update(summary_changes or {})
  receipt=dict(measurement_valid=True,child_stopped=True,clock_restore_complete=True,child_exitcode=0,outer_cleanup_errors=[],summary=summary);receipt.update(receipt_changes or {});rr=save('receipt.json',receipt)
  cp=dict(measurement_valid=True,work_complete=True,receipt=rr,binding=binding,artifacts={rr['path']:rr['sha256'],marker['path']:marker['sha256']});(cpdir/(cid+'.json')).write_text(json.dumps(cp))
  status=dict(complete=True,failed=[],node_lease_held=False,pid=123,completed=[cid]);status.update(status_changes or {});sr=save('status.json',status)
  capacity=dict(cleanup_complete=True,errors=[]);capacity.update(capacity_changes or {});cr=save('capacity.json',capacity)
  audit=dict(schema='A-final-pdb-terminal-for-baseline-restore-v1',passed=True,model='14b',release={'path':'/synthetic/release','sha256':'synthetic'},previous_binding={'path':'/synthetic/previous','sha256':'synthetic'},files={cr['path']:cr['sha256'],str(cpdir/(cid+'.json')):a.sha(cpdir/(cid+'.json'))},capacity_cleanup_verified=True,capacity_cleanup_evidence=[cr],stages=[dict(status=sr,declaration=decl,expected_cell_ids=[cid],checkpoint_root=str(cpdir),engineering_gates={cid:gate})])
  a.REFS=dict(release=audit['release'],previous=audit['previous_binding'],terminal={'path':'/synthetic/audit','sha256':'synthetic'});return audit
 a.alive=lambda _:False;assert a.terminal_contract(fixture())['passed'];checks.append(dict(name='synthetic_full_terminal_contract',passed=True))
 cases=[('request_failure',dict(summary_changes={'failed_requests':1})),('timeout',dict(summary_changes={'request_timeouts':1})),('incomplete_work',dict(summary_changes={'work_complete':False})),('unclean_native',dict(summary_changes={'post_measurement_cleanup':{'cleanup_complete':False}})),('clock_failure',dict(receipt_changes={'clock_restore_complete':False})),('held_lease',dict(status_changes={'node_lease_held':True})),('incomplete_capacity_cleanup',dict(capacity_changes={'cleanup_complete':False})),('capacity_cleanup_error',dict(capacity_changes={'errors':['unknown owner']}))]
 for name,kwargs in cases:
  try:a.terminal_contract(fixture(**kwargs))
  except RuntimeError:checks.append(dict(name=name,passed=True,rejected=True))
  else:raise AssertionError(name)
 a.alive=lambda _:True
 try:a.terminal_contract(fixture())
 except RuntimeError:checks.append(dict(name='live_predecessor',passed=True,rejected=True))
 else:raise AssertionError('live predecessor')
result=dict(schema=1,passed=True,cpu_only=True,synthetic_only=True,hardware_actions=False,created_s=time.time(),checks=checks,files={str(p):a.sha(p) for p in [D/'restore_adapter.py',D/'restore_parent.py',Path(__file__).resolve()]},qualification_not_completed=True)
(D/'cpu-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(checks),gpu_executed=False)))
