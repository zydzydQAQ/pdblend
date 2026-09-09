"""Append-only remote staging of the frozen baseline package, no GPU action."""
import io,json,tarfile,time
from pathlib import Path
from operate import remote,sha,write_new
HERE=Path(__file__).resolve().parent
package=HERE/'boundary-baseline-package-p4.json';files=json.loads(package.read_text())['files'];files[str(package)]=sha(package)
assert all(sha(p)==h for p,h in files.items())
inspect='''import sys,json,pathlib,hashlib
files=json.load(sys.stdin);out=dict(missing=[],different=[],same=0)
for p,h in files.items():
 f=pathlib.Path(p)
 if not f.exists():out['missing'].append(p)
 elif hashlib.sha256(f.read_bytes()).hexdigest()!=h:out['different'].append(p)
 else:out['same']+=1
print(json.dumps(out))'''
audit=json.loads(remote(inspect,json.dumps(files).encode()));assert not audit['different'],audit
missing={p:files[p] for p in audit['missing']};archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w') as tar:
 for p in missing:
  assert p.startswith('/root/workspace/pdblend-next-v1/');tar.add(p,arcname=p.lstrip('/'),recursive=False)
extract='''import pathlib,sys,tarfile,hashlib
expected='''+repr(missing)+'''
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  dest=pathlib.Path('/')/member.name
  assert member.isfile() and str(dest) in expected
  data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected[str(dest)]
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(data)
print('staged')'''
remote(extract,archive.getvalue());verify=json.loads(remote(inspect,json.dumps(files).encode()));assert not verify['different'] and not verify['missing']
p=HERE/('boundary-stage-'+str(time.time_ns())+'.json');write_new(p,json.dumps(dict(captured_s=time.time(),package=str(package),files=files,inspection=audit,verified=verify),indent=2).encode())
print(json.dumps(dict(staged=len(missing),verified=len(files),audit=str(p))))
