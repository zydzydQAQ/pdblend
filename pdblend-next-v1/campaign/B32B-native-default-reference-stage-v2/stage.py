import hashlib,json,os,socket,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('/root/workspace/pdblend-next-v1');OUT=ROOT/'campaign/B32B-native-default-reference-stage-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((OUT/'stage-manifest.json').read_text());assert socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z'
assert sha(OUT/'package.tar.gz')==m['archive_sha256']
before={p:sha(p) for p in m['guards']};assert before==m['guards'];(OUT/'guards-before.json').write_text(json.dumps(before,indent=2)+'\n')
new=[];same=[]
with tarfile.open(OUT/'package.tar.gz') as t:
 members=t.getmembers();assert len(members)==m['members'] and len({x.name for x in members})==len(members)
 for x in members:
  p=Path('/'+x.name);assert x.isfile() and ROOT in p.parents and str(p) in m['files'];raw=t.extractfile(x).read();assert hashlib.sha256(raw).hexdigest()==m['files'][str(p)]
  if p.exists():assert p.is_file() and not p.is_symlink() and sha(p)==m['files'][str(p)];same.append(str(p))
 for x in members:
  p=Path('/'+x.name)
  if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(t.extractfile(x).read());new.append(str(p))
 for p,h in m['files'].items():assert sha(p)==h
host=ROOT/'releases/five-system100-B32B-v1-runtime';env=dict(os.environ,PYTHONPATH=f'{host}/src:{host}:/root/workspace/pdblend/.runtime-deps')
argv=['python3',str(ROOT/'campaign/B32B-native-default-reference-execution-v2/run.py')]
p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
after={p:sha(p) for p in m['guards']};assert before==after
result=dict(schema=1,hostname=socket.gethostname(),new=new,same=same,check=dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),guards_unchanged=True,guards_count=len(before),finished_s=time.time())
(OUT/'stage-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(new=len(new),same=len(same),guards=len(before),check_exit=p.returncode,stdout=p.stdout)))
assert p.returncode==0
