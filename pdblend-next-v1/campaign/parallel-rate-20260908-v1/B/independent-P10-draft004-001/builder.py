"""Explicit hardware maximum; all original defaults remain2520. CPU builder only."""
import ast,copy,hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
FIELD='max_service_frequency_mhz'
CLASSES={'runtime.py':{'Controller'},'backend.py':{'ClockOwner','HttpEngineBackend'},'frequency.py':{'FrequencyPlanner'},'ecoserve.py':{'EcoServeScheduler'},'distserve.py':{'DistServeScheduler'},'dynamo.py':{'DynamoScheduler'},'dynamo_topology.py':{'DynamoTopologyPlanner'},'baselines.py':{'DistServeSearch'}}
FUNCTIONS={'physical_frequency.py':{'bootstrap_allowed':'controller'}}
NESTED={'topology.py':{'TopologyManager':'self.backend'},'pd_topology.py':{'PDBlendTopologyPlanner':'self.planner'},'capacity_backend.py':{'PinnedDockerBackend':'self.controller'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expr(s):return ast.parse(s,mode='eval').body
def statements(s):return ast.parse(s).body
def access(owner='self'):return expr(f"getattr({owner},'{FIELD}',2520)")
def target(node,name):return isinstance(node,ast.Attribute) and node.attr==name

def transform(source,name):
 tree=ast.parse(source)
 class ReplaceConstants(ast.NodeTransformer):
  def __init__(self,owner):self.owner=owner
  def visit_Constant(self,node):return ast.copy_location(access(self.owner),node) if type(node.value)is int and node.value==2520 else node
 class Rewire(ast.NodeTransformer):
  def __init__(self,kind):self.kind=kind
  def visit_Call(self,node):
   node=self.generic_visit(node);called=node.func.id if isinstance(node.func,ast.Name) else None
   if self.kind=='Controller' and (called in ('ClockOwner','HttpEngineBackend','EcoServeScheduler','DistServeScheduler','DynamoScheduler','DynamoTopologyPlanner','FrequencyPlanner') or (isinstance(node.func,ast.Attribute) and node.func.attr=='to_thread' and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id=='ClockOwner')):
    node.keywords.append(ast.keyword(arg=FIELD,value=access()))
   if called=='JointPlanner' and self.kind in ('Controller','DistServeScheduler','DynamoScheduler'):
    node.keywords.append(ast.keyword(arg='max_frequency',value=access()))
   if called=='recovery_actions' and self.kind=='FrequencyPlanner':node.keywords.append(ast.keyword(arg='maximum',value=access()))
   if called=='online_profile_points' and self.kind=='DistServeSearch':node.keywords.append(ast.keyword(arg='frequency_mhz',value=access()))
   return node
 for cls in tree.body:
  if isinstance(cls,(ast.FunctionDef,ast.AsyncFunctionDef)) and cls.name in FUNCTIONS.get(name,{}):
   cls.body=[ReplaceConstants(FUNCTIONS[name][cls.name]).visit(x) for x in cls.body]
  if not isinstance(cls,ast.ClassDef):continue
  selected=cls.name in CLASSES.get(name,set());nested=NESTED.get(name,{}).get(cls.name)
  if not selected and not nested:continue
  owner=nested or 'self'
  for fn in cls.body:
   if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
    fn.body=[ReplaceConstants(owner).visit(x) for x in fn.body]
    if selected:fn.body=[Rewire(cls.name).visit(x) for x in fn.body]
  if selected:
   init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
   if cls.name=='Controller':value="config.get('max_service_frequency_mhz',2520)"
   else:
    init.args.kwonlyargs.append(ast.arg(arg=FIELD));init.args.kw_defaults.append(ast.Constant(2520));value=FIELD
   init.body[:0]=statements(f"self.{FIELD}={value}\nif type(self.{FIELD}) is not int or self.{FIELD} not in (900,1500,2100,2400,2520):\n raise ValueError('unsupported hardware maximum frequency')\n")
   # The actual physical writer rejects commands outside a nondefault domain,
   # including any forgotten/future planner path. Never clamp an unknown request.
   if cls.name=='ClockOwner':
    for fn in cls.body:
     if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)) and fn.name in ('set','physical_write'):
      fn.body[:0]=statements(f"if getattr(self,'{FIELD}',2520)!=2520 and (type(frequency) is not int or not 0<frequency<=self.{FIELD}):\n raise ValueError('clock command outside qualified hardware maximum')\n")
   if cls.name=='Controller':
    # Reject, rather than silently relabel or extrapolate, profiles above cap.
    for index,node in enumerate(init.body):
     if isinstance(node,ast.Assign) and any(target(t,'profiles') for t in node.targets):
      init.body[index+1:index+1]=statements(f"if self.{FIELD}!=2520 and any(p.frequency_mhz>self.{FIELD} for p in self.profiles.points):\n raise ValueError('profile frequencies exceed qualified hardware maximum')\n")
      break
   if cls.name in ('FrequencyPlanner',):
    # Propagate to callers which hold only the estimator (e.g. virtual layouts).
    pass
 # The PD topology virtual state obtains the already explicit estimator maximum.
 if name=='pd_topology.py':
  class PD(ast.NodeTransformer):
   def visit_Call(self,n):
    if isinstance(n.func,ast.Name) and n.func.id=='getattr' and len(n.args)==3 and ast.unparse(n.args[0])=='self.planner' and isinstance(n.args[1],ast.Constant) and n.args[1].value==FIELD:n.args[1]=ast.Constant('max_frequency')
    return self.generic_visit(n)
  tree=PD().visit(tree)
 return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def normalize_default(source,name):
 tree=ast.parse(source)
 class Default(ast.NodeTransformer):
  def visit_Assign(self,n):
   if any(target(t,FIELD) for t in n.targets):return None
   return self.generic_visit(n)
  def visit_If(self,n):
   # Only guards introduced by this builder have these precise messages.
   messages=[x.value for x in ast.walk(n) if isinstance(x,ast.Constant) and isinstance(x.value,str)]
   if any(x in messages for x in ('unsupported hardware maximum frequency','clock command outside qualified hardware maximum','profile frequencies exceed qualified hardware maximum')):return None
   return self.generic_visit(n)
  def visit_FunctionDef(self,n):
   pairs=[(a,d) for a,d in zip(n.args.kwonlyargs,n.args.kw_defaults) if a.arg!=FIELD];n.args.kwonlyargs=[a for a,d in pairs];n.args.kw_defaults=[d for a,d in pairs]
   return self.generic_visit(n)
  visit_AsyncFunctionDef=visit_FunctionDef
  def visit_Call(self,n):
   # Additional keywords at the old default equal the original callee defaults.
   n.keywords=[k for k in n.keywords if not(k.arg==FIELD or (k.arg in ('max_frequency','maximum','frequency_mhz') and isinstance(k.value,ast.Call) and isinstance(k.value.func,ast.Name) and k.value.func.id=='getattr' and len(k.value.args)==3 and isinstance(k.value.args[1],ast.Constant) and k.value.args[1].value==FIELD))]
   if isinstance(n.func,ast.Name) and n.func.id=='getattr' and len(n.args)==3 and isinstance(n.args[1],ast.Constant) and (n.args[1].value==FIELD or name=='pd_topology.py' and n.args[1].value=='max_frequency') and isinstance(n.args[2],ast.Constant) and n.args[2].value==2520:return ast.Constant(2520)
   return self.generic_visit(n)
 tree=Default().visit(tree)
 return ast.dump(ast.fix_missing_locations(tree),include_attributes=False)

def build(parent,out):
 parent,out=Path(parent),Path(out)
 if out.exists():raise ValueError('new output required')
 manifest=json.loads((parent/'manifest.json').read_text())
 for p,h in manifest['files'].items():
  pp=Path(p) if Path(p).is_absolute() else parent/p
  if sha(pp)!=h:raise ValueError('parent changed '+str(pp))
 shutil.copytree(parent,out,ignore=shutil.ignore_patterns('__pycache__'))
 changed=[];proof=[]
 for rel in manifest['files']:
  path=Path(rel)
  if path.is_absolute():continue
  name=path.name
  if name not in CLASSES and name not in NESTED and name not in FUNCTIONS:continue
  p=out/path;old=(parent/path).read_text();new=transform(old,name)
  if ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False):continue
  p.write_text(new);valid=normalize_default(new,name)==ast.dump(ast.parse(old),include_attributes=False)
  proof.append(dict(file=rel,default_specialized_AST_equal=valid,parent_sha256=sha(parent/path),actual_sha256=sha(p)))
  if not valid:raise ValueError('default AST changed: '+rel)
  changed.append(rel)
 # A manifest is written only once, after source and the default proof pass.
 (out/'manifest.json').unlink()
 fs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
 record=dict(schema=1,files=fs,parent=dict(path=str(parent/'manifest.json'),sha256=sha(parent/'manifest.json')),hardware_max_frequency_entry=FIELD,default_mhz=2520,changed_files=changed,default_equivalence=proof,builder=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),CPU_only=True)
 (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');return record
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();m=build(a.parent,a.out);print(json.dumps(dict(changed=len(m['changed_files']),all_default_equal=all(x['default_specialized_AST_equal'] for x in m['default_equivalence']))))
