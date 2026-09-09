"""Copy missing immutable C baseline preparation inputs; no GPU actions."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
from operate import remote,write_new

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=HERE/'baseline-restore-p3/deployment.json'
value=json.loads(spec.read_text());files=dict(value['files'])
for p in [spec,HERE/'run_baseline_new_rates_p3.py',HERE/'new-rates-p1/declaration.json']:
    files[str(p)]=sha(p)
declaration=json.loads((HERE/'new-rates-p1/declaration.json').read_text())
for row in declaration['cells']:files[row['trace_path']]=row['trace_sha256']
for root in ['AC-baseline-binding-v2','AC-legacy-resident-correctness-v1']:
    directory=HERE.parents[1]/root
    for p in directory.glob('*'):
        if p.is_file() and p.suffix in ('.py','.json'):files[str(p)]=sha(p)
assert all(sha(p)==h for p,h in files.items())
check='''import sys,json,pathlib,hashlib
v=json.load(sys.stdin);r=dict(missing=[],different=[],same=0)
for p,h in v.items():
 f=pathlib.Path(p)
 if not f.exists():r['missing'].append(p)
 elif hashlib.sha256(f.read_bytes()).hexdigest()!=h:r['different'].append(p)
 else:r['same']+=1
print(json.dumps(r))'''
audit=json.loads(remote(check,json.dumps(files).encode()));assert not audit['different'],audit
missing={p:files[p] for p in audit['missing']}
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w') as tar:
    for p in missing:tar.add(p,arcname=p.lstrip('/'),recursive=False)
extract='''import pathlib,sys,tarfile,hashlib
expected='''+repr(missing)+'''
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  p=pathlib.Path('/')/member.name;assert member.isfile() and str(p) in expected
  data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected[str(p)]
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
print('staged')'''
remote(extract,archive.getvalue())
verify=json.loads(remote(check,json.dumps(files).encode()));assert not verify['different'] and not verify['missing'],verify
result=dict(captured_s=time.time(),staged=len(missing),files_verified=len(files),inspection=audit,files=files)
p=HERE/('baseline-stage-p3-'+str(time.time_ns())+'.json');write_new(p,json.dumps(result,indent=2).encode())
print(json.dumps(dict(staged=len(missing),files_verified=len(files),receipt=str(p))))
