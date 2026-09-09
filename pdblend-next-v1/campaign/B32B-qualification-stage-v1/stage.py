import hashlib,json,os,subprocess,tarfile
from pathlib import Path
ROOT=Path('/root/workspace/pdblend-next-v1');C=ROOT/'campaign';S=C/'B32B-qualification-stage-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archives={'B32B-temporal-qualification-v2.tar.gz':'3ee5ed5b5dfd0e59b40ba1c1e06e67abcefa6dcc1f2543ef1867f134f0c79fd7','B32B-ecoserve-qualified-binding-v1.tar.gz':'726ec1cacfb44dcdf65195dba7e130590dfb2573e001b67ba6221aa5370e8023'}
new=[];same=[]
for archive,h in archives.items():
 p=S/archive;assert sha(p)==h
 with tarfile.open(p) as t:
  ms=t.getmembers();assert len({m.name for m in ms})==len(ms)
  for m in ms:
   p=ROOT/m.name;assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   raw=t.extractfile(m).read()
   if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
  for m in ms:
   p=ROOT/m.name
   if p.exists():same.append(str(p))
   else:p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(t.extractfile(m).read());new.append(str(p))
p=S/'registered-oracle.json';assert sha(p)=='b60cfda246aca334b6412dd03bcc1f836f1ff477df19df66ace8d9d865b0dc04'
t=C/'B32B-temporal-qualification-native002-v1/registered-oracle.json';t.parent.mkdir(exist_ok=True)
if t.exists():assert sha(t)==sha(p)
else:t.open('xb').write(p.read_bytes())
host=ROOT/'releases/five-system100-B32B-v1-runtime';env=dict(os.environ,PYTHONPATH=f'{host}/src:{host}:/root/workspace/pdblend/.runtime-deps')
checks=[]
for argv in [['python3',str(C/'B32B-temporal-qualification-v2/qualification.py'),'--check'],['python3',str(C/'B32B-ecoserve-qualified-binding-v1/bind.py'),'check']]:
 r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60);checks.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
(S/'receipt.json').write_text(json.dumps(dict(new=new,same=same,checks=checks),indent=2)+'\n');print(json.dumps(dict(new=len(new),same=len(same),checks=checks)))
assert all(r['exit_code']==0 for r in checks)
