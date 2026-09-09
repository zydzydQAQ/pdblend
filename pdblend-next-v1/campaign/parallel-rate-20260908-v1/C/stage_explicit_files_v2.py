"""Append-only staging with the manifest on stdin, avoiding OS argv size limits."""
import io,json,tarfile
from pathlib import Path
import operate

def stage(files):
 assert all(operate.sha(p)==h for p,h in files.items())
 inspect='import sys,json,pathlib,hashlib;f=json.load(sys.stdin);print(json.dumps({p:("missing" if not pathlib.Path(p).exists() else "same" if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h else "different") for p,h in f.items()}))'
 audit=json.loads(operate.remote(inspect,json.dumps(files).encode()));assert not [p for p,s in audit.items() if s=='different']
 missing={p:files[p] for p,s in audit.items() if s=='missing'};buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as tar:
  for p in missing:tar.add(p,arcname=p.lstrip('/'),recursive=False)
 code='''import sys,tarfile,pathlib,hashlib,json
expected=json.loads(sys.stdin.buffer.readline())
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  p=pathlib.Path('/')/m.name;assert str(p) in expected and m.isfile();b=tar.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[str(p)];p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
print('staged')'''
 operate.remote(code,json.dumps(missing).encode()+b'\n'+buf.getvalue())
 verified=json.loads(operate.remote(inspect,json.dumps(files).encode()));assert set(verified.values())=={'same'}
 return dict(verified=len(files),staged=len(missing))
