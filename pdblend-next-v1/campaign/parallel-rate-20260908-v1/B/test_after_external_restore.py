"""CPU counterexamples for exact retained B restoration after an external lease."""
import ast,copy,importlib.util,json,sys
from pathlib import Path
import pytest
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import baseline_control_after_external_v1 as control

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
new=module(B/'baseline-return-after-external-source-v1/execution.py','new_restore')
old_path=B/'baseline-return-completion-source/execution.py'

def fixtures():
 parent={'instances':[{'container':{'id':'B-original-id','name':'B-original','image':'image-sha'}}]}
 inv=[{'Id':'B-original-id','Name':'/B-original','Image':'image-sha','State':{'Running':True}}]
 return parent,inv

def test_running_targets_refused_without_same_deployment_proof():
 p,i=fixtures()
 with pytest.raises(RuntimeError):new.target_identity(p,i)

def test_same_exact_retained_running_targets_allowed():
 p,i=fixtures();assert new.target_identity(p,i,allow_running_same_targets=True)['B-original-id']==i[0]

def test_wrong_id_name_image_rejected_even_same_deployment_flag():
 for key,value in [('Id','foreign'),('Name','/foreign'),('Image','foreign')]:
  p,i=fixtures();i[0][key]=value
  with pytest.raises(RuntimeError):new.target_identity(p,i,allow_running_same_targets=True)

def test_extra_container_rejected():
 p,i=fixtures();i.append(dict(i[0],Id='foreign'))
 with pytest.raises(RuntimeError):new.target_identity(p,i,allow_running_same_targets=True)

def test_core_patch_is_only_explicit_same_target_allowance():
 before=ast.parse(old_path.read_text());after=ast.parse((B/'baseline-return-after-external-source-v1/execution.py').read_text())
 fn=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='target_identity')
 oldfn=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='target_identity');after.body[after.body.index(fn)]=oldfn
 core=next(n for n in after.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='restore_core')
 calls=[n for n in ast.walk(core) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='target_identity']
 assert len(calls)==1 and len(calls[0].keywords)==1
 assert ast.unparse(calls[0].keywords[0].value)=="{i['container']['id'] for i in previous['instances']} == set(ids)"
 calls[0].keywords=[]
 assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)

def test_external_contract_absent_does_not_claim_ready():
 assert not (B/'external-release-for-B-001.json').exists()
 with pytest.raises(FileNotFoundError):control.external_predecessor()

def test_native_freshness_and_ordinary_gate_preserved():
 source=(B/'baseline_control_after_external_v1.py').read_text()
 assert "a['container']['StartedAt']!=b['container']['StartedAt']" in source
 assert "raw.get('restored') is True" in source
 assert "q.same_policy(last['instances'],actual['instances'])" in source
 assert 'with node_lease():' in source
 old=module(old_path,'old_restore')
 assert ast.dump(ast.parse(__import__('inspect').getsource(new.ordinary_gate)),include_attributes=False)==ast.dump(ast.parse(__import__('inspect').getsource(old.ordinary_gate)),include_attributes=False)
