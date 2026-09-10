from pathlib import Path
import json,shutil,hashlib,re,ast
R=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
N=R.parent/'slo-rate-14b-sharegpt-20260909-v1'; E=N/'env'
B=R/'B/uniform-rate-20260909-v2/migration-14b-sharegpt'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def ref(p):return {'path':str(p),'sha256':sha(p)}
def copytree(src,dst):
 if not dst.exists():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
copytree(B/'pipeline-007/meter-pdblend/runtime',E/'runtime')
copytree(R/'A/uniform-rate-20260909-v1/meter-runtime',E/'meter')
copytree(R/'A/uniform-rate-20260909-v1/isolated-power',E/'isolated-power')
for folder in ['meter','isolated-power']:
 p=E/folder/'manifest.json';d=json.loads(p.read_text());src=(R/'A/uniform-rate-20260909-v1'/('meter-runtime' if folder=='meter' else folder));d['source_manifest']=ref(src/'manifest.json');d['files']={str(E/folder/Path(x).name):sha(E/folder/Path(x).name) for x in d['files']};save(p,d)
shutil.copy2(R/'common/execution-until-complete-v1/run.py',E/'run.py');shutil.copy2(R/'common/execution-until-complete-v1/child.py',E/'child.py')
for name in ['model-manifest.json','native-template.json','numerical-reference.json']:shutil.copy2(R/'A/uniform-rate-20260909-v1'/name,E/name)
profile=R/'B/distributed-14b-v1/frequency2100-registered-001/profiles.development.json';shutil.copy2(profile,E/'profiles.json')
cfg=json.loads((B/'pipeline-007/idle-qualification-001/configs/sharegpt.json').read_text())
# Preserve all algorithm values. Materialize external read-only evidence under this campaign.
original_cfg=ref(B/'pipeline-007/idle-qualification-001/configs/sharegpt.json'); mapping={}
def mapped(v):
 if isinstance(v,dict):return {k:mapped(x) for k,x in v.items()}
 if isinstance(v,list):return [mapped(x) for x in v]
 if isinstance(v,str) and v.startswith('/root/workspace'):
  p=Path(v)
  if p.is_file():
   target=E/'evidence'/ (sha(p)+'-'+p.name);target.parent.mkdir(exist_ok=True);shutil.copy2(p,target);mapping[v]=ref(target);return str(target)
 return v
cfg=mapped(cfg)
cfg['profiles']=str(E/'profiles.json')
for k in ['host_source_release','controller_source_release','engine_source_release']:cfg[k]=str(E/'runtime')
cfg['instances']=[]
save(E/'sharegpt-template.json',cfg)
save(E/'source-mapping.json',dict(source_config=original_cfg,paths=mapping,controller_source_bytes_unchanged=True,profile=ref(profile)))
for node,hostname,scale in [('A','iZwz9274emxme9019d2sjgZ',.5),('C','iZwz9gfq11hx1sbob59yrgZ',2.)]:
 T=N/node/'tooling';T.mkdir(parents=True,exist_ok=True)
 save(T/'node.json',dict(node=node,hostname=hostname,scale=scale))
 (T/'bootstrap.py').write_text('''from pathlib import Path\nimport json,hashlib,importlib.util\ndef checked(r):\n p=Path(r["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"],str(p);return json.loads(p.read_text())\ndef save(p,d):\n p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(d,indent=2,allow_nan=False)+"\\n");t.replace(p)\ndef load(p,name):\n if isinstance(p,dict):checked_source=p;p=p["path"]\n spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m\n''')
 ptxt=(B/'power_selftest.py').read_text();ptxt=ptxt.replace("HOST = ROOT / 'hosts/14b-capacity-p12'",f"HOST = Path('{E/'runtime'}')").replace("METER = ROOT / 'A/uniform-rate-20260909-v1/meter-runtime'",f"METER = Path('{E/'meter'}')").replace("ADAPTER = ROOT / 'A/uniform-rate-20260909-v1/isolated-power/manifest.json'",f"ADAPTER = Path('{E/'isolated-power/manifest.json'}')").replace("identity['node'] == 'B'",f"identity['node'] == '{node}'").replace("'iZwz9i5bte3xkpmcoes3t2Z'",repr(hostname));ptxt+='\nNODE='+repr(node)+'\nEXPECTED_HOSTNAME='+repr(hostname)+'\n'
 (T/'power_selftest.py').write_text(ptxt)
 for name in ['qualify_fixed.py','verify_fixed.py','qualify_idle.py','verify_idle.py','stream.py']:
  txt=(B/name).read_text().replace("'iZwz9i5bte3xkpmcoes3t2Z'",'p.EXPECTED_HOSTNAME').replace("node='B'","node=p.NODE").replace("prior['node']=='B'","prior['node']==p.NODE")
  (T/name).write_text(txt)
 # Actual cold-start path from new-A bootstrap; scoped validation supplied separately.
 src=(R/'A/uniform-rate-20260909-v1/bootstrap.py').read_text()
 mod=ast.parse(src); start=next(x for x in mod.body if isinstance(x,ast.FunctionDef) and x.name=='validate');lines=src.splitlines(keepends=True)
 lines[start.lineno-1:start.end_lineno]=['def validate(spec):\n','    from inputs import validate_bootstrap\n','    return validate_bootstrap(spec)\n']
 src=''.join(lines).replace("node='Anew20260909'","node=p.NODE").replace('pdblend.uniform.owner=Anew20260909-bootstrap','pdblend.slo14.owner='+node)
 (T/'cold_bootstrap.py').write_text(src)
 print(node,T)
save(E/'build-receipt.json',dict(pdb_parent=ref(B/'pipeline-007/meter-pdblend/binding.json'),runtime_parent=ref(B/'pipeline-007/meter-pdblend/runtime/manifest.json'),runtime=ref(E/'runtime/manifest.json'),built=True))
