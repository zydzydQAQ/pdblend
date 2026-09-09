"""CPU proof of exact historical per-system source and unchanged policy configuration."""
import copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
root=HERE/'baseline-source-v2-cpu';root.mkdir()
sp=importlib.util.spec_from_file_location('c_source_only_binder',HERE/'baseline-until-complete-v2/bind.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
base=read(HERE/'baseline-boundary-restore-p4v2/deployment.json');decl=read(HERE/'boundary-p4v2/declaration.json');cases=[]
for system,host in decl['baseline_controller_hosts'].items():
 spec=read(HERE/'baseline-strategy-specs-p4v2'/(system+'.json'))
 m.scope(spec);strategy='dynamollm-resident' if system=='dynamollm' else system
 assert m.selected(spec,strategy,['alpaca','sharegpt','longbench'])==['alpaca','sharegpt','longbench']
 differences={k for k in set(base)|set(spec) if base.get(k)!=spec.get(k)}
 assert differences<= {'host_release','baseline_source_strategy','files'}
 assert spec['host_release']==host and spec['baseline_source_strategy']==system
 assert all(sha(p)==h for p,h in spec['files'].items())
 instances=[dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],port=i['port'],kv_port=i['kv_port'],url=i['url'],container={'name':i['container_name']}) for i in spec['instances']]
 baseline_spec=copy.deepcopy(spec);baseline_spec['host_release']=base['host_release']
 for label,value in [('actual',spec),('v1-policy-reference',baseline_spec)]:
  out=root/system/label;out.mkdir(parents=True);frozen={}
  configs,notes=m.configurations(value,instances,strategy,['alpaca','sharegpt','longbench'],out,out/'results',frozen)
  for ds in configs:assert all(sha(p)==h for p,h in frozen.items())
 for ds in ['alpaca','sharegpt','longbench']:
  a=read(root/system/'actual/configs'/(ds+'.json'));b=read(root/system/'v1-policy-reference/configs'/(ds+'.json'))
  # Controller source metadata is the only changed value. C resident topology does not allocate new instances.
  delta={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
  assert delta==({'controller_source_release'} if system=='dynamollm' else set()),(system,ds,delta)
  cases.append(dict(system=system,dataset=ds,passed=True,source_host=host,changed_policy_values=sorted(delta)))
old=(REPO/'campaign/AC-legacy-resident-correctness-v1/checks.py')
assert sha(old)=='e0965f922ae42245b275d9c17342c31689290c61bcac5dfaf33a2a352fd29931'
value=dict(passed=True,cpu_only=True,created_s=time.time(),cases=cases,fixture_configs_not_runtime_bindings=True,
 baseline_policy_deltas_only_controller_source_metadata=True,old_native_checks_sha256=sha(old),
 source_package=dict(path=str(HERE/'boundary-baseline-package-p4v2.json'),sha256=sha(HERE/'boundary-baseline-package-p4v2.json')))
(root/'validation.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(cases),path=str(root/'validation.json'))))
