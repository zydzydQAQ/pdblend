"""Append-only transport of explicit immutable files to the C node."""
import io,json,tarfile
from pathlib import Path
import operate
C=Path(__file__).resolve().parent

def stage(files):
 assert all(operate.sha(p)==h for p,h in files.items())
 code='import json,pathlib,hashlib;f='+repr(files)+';print(json.dumps({p:("missing" if not pathlib.Path(p).exists() else "same" if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h else "different") for p,h in f.items()}))'
 audit=json.loads(operate.remote(code));assert not [p for p,s in audit.items() if s=='different'];missing={p:files[p] for p,s in audit.items() if s=='missing'};buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as tar:
  for p in missing:tar.add(p,arcname=p.lstrip('/'),recursive=False)
 code='import sys,tarfile,pathlib,hashlib\nexpected='+repr(missing)+'''\nwith tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  p=pathlib.Path('/')/m.name;assert str(p) in expected and m.isfile();b=tar.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[str(p)];p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
print('staged')'''
 operate.remote(code,buf.getvalue());return dict(verified=len(files),staged=len(missing))
if __name__=='__main__':
 out=C/'baseline-after-gamma-restore-002';files=json.load(open(out/'deployment.json'))['files'];files.update({str(p):operate.sha(p) for p in out.rglob('*') if p.is_file()});print(stage(files))
