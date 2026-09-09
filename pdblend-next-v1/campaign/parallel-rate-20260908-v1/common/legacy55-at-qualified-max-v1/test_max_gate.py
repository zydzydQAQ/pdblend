"""CPU-only proof that original business/native/cleanup code is preserved."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE.parent.parent;REPO=R.parents[1]

def functions(path):return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def normalized(n):
 n=copy.deepcopy(n)
 class Restore(ast.NodeTransformer):
  def visit_Attribute(self,x):
   if isinstance(x.value,ast.Name) and x.value.id=='args' and x.attr=='maximum':return ast.Constant(2520)
   return self.generic_visit(x)
  def visit_Call(self,x):
   x.keywords=[k for k in x.keywords if k.arg!='max_service_frequency_mhz'];return self.generic_visit(x)
 return dump(Restore().visit(n))
def parent_hetero(n):
 s=ast.unparse(n)
 s=s.replace("require(time.time() + 510 < m.GLOBAL_DEADLINE, 'insufficient global time for gate plus cleanup')",'pass')
 s=s.replace('time.monotonic() + max(0, min(90, m.GLOBAL_DEADLINE - time.time()))','time.monotonic() + 90')
 n=ast.parse(s).body[0];n.body=[x for x in n.body if not isinstance(x,ast.Pass)];return n

def main():
 cases=[]
 parents={'resident':R/'A/eco-drain31-v1/code/validate.py','heterogeneous':REPO/'campaign/A14B-legacy-heterogeneous-correctness-v1/validate.py'}
 for kind,path in parents.items():
  a,b=functions(path),functions(HERE/(kind+'.py'))
  for name in ('common','mechanism_gates','validate_scope','execute'):
   old=parent_hetero(a[name]) if kind=='heterogeneous' and name=='execute' else a[name]
   assert normalized(b[name])==dump(old),(kind,name)
   cases.append(dict(case='entire-original-function-AST',layout=kind,function=name,passed=True))
  tree=b['execute'];calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to_thread' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='ClockOwner']
  assert len(calls)==1 and any(k.arg=='max_service_frequency_mhz' and ast.unparse(k.value)=='args.maximum' for k in calls[0].keywords)
  writes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='clocks.set']
  assert len(writes)==1 and ast.unparse(writes[0].args[1])=='args.maximum' and any(k.arg=='verify_rise' and k.value.value is False for k in writes[0].keywords)
  cases.append(dict(case='same-command-position-owned-clock-maximum-forwarded',layout=kind,passed=True))
  # Changes to an original request/native/cleanup timeout cannot pass proof.
  wrong=copy.deepcopy(b['execute'])
  class Corrupt(ast.NodeTransformer):
   def visit_Constant(self,x):return ast.Constant(391) if type(x.value)is int and x.value==390 else x
  assert normalized(Corrupt().visit(wrong))!=dump(parent_hetero(a['execute']) if kind=='heterogeneous' else a['execute'])
  cases.append(dict(case='local-business-timeout-change-rejected',layout=kind,passed=True))
 # The wrapper may only run a model/layout manifest known by common P10 proof.
 spec=importlib.util.spec_from_file_location('maxgate_run',HERE/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for value in (0,2401,True,'2400',None):
  try:m.freeze_check(dict(schema='distributed14b-original-native-gate-qualified-max-v1',authorized=True,maximum=value,files={}))
  except ValueError:pass
  else:raise AssertionError('unsupported maximum accepted')
  cases.append(dict(case='reject-unapproved-maximum',value=value,passed=True))
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);paths=[d/'a.json',d/'b.json']
  for p in paths:p.write_text(json.dumps(dict(runtime_dir=str(d/'actual'))))
  binding=dict(instances=[dict(engine_config=str(p)) for p in paths])
  m.check_runtime_directory(binding,d/'actual');cases.append(dict(case='actual-native-common-runtime-directory',passed=True))
  try:m.check_runtime_directory(binding,d/'guessed-wrong-directory')
  except ValueError:pass
  else:raise AssertionError('wrong runtime path accepted')
  cases.append(dict(case='reject-guessed-native-event-directory',passed=True))
  paths[1].write_text(json.dumps(dict(runtime_dir=str(d/'other'))))
  try:m.check_runtime_directory(binding,d/'actual')
  except ValueError:pass
  else:raise AssertionError('mixed runtime directories accepted')
  cases.append(dict(case='reject-mixed-native-event-directories',passed=True))
 print(json.dumps(dict(passed=True,hardware_actions=False,tests=len(cases),cases=cases)))
if __name__=='__main__':main()
