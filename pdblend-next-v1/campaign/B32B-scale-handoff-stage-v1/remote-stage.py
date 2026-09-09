import hashlib,json,os,socket,subprocess,tarfile,time
from pathlib import Path
R=Path('/root/workspace/pdblend-next-v1');C=R/'campaign';S=C/'B32B-scale-handoff-stage-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z' and not os.environ.get('PDBLEND_NODE_LOCK_FD')
def save(n,v):
 with (S/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
bp=C/'B32B-ecoserve-qualified-main-v1/binding.json';b=json.loads(bp.read_text())
guards=[bp,C/'current-experiment.json',C/'CURRENT_EXPERIMENT.md',C/'five-system-fixed-window-v1/sources/B32B/manifest.json',C/'five-system-execution-v3/run.py',C/'five-system-execution-v3/child.py',Path(b['host_release'])/'manifest.json',*map(Path,b['configs'].values())]
before={str(p):sha(p) for p in guards};save('guards-before.json',before)
new=[];same=[]
for original,h in json.loads((S/'archives.json').read_text()).items():
 archive=S/Path(original).name;assert sha(archive)==h
 with tarfile.open(archive) as t:
  ms=t.getmembers();assert len({m.name for m in ms})==len(ms)
  for m in ms:
   p=R/m.name;assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   if p.exists():assert not p.is_symlink() and p.is_file() and p.read_bytes()==t.extractfile(m).read()
  for m in ms:
   p=R/m.name
   if p.exists():same.append(str(p))
   else:p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(t.extractfile(m).read());new.append(str(p))
H=Path(b['host_release']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{H}/src:{H}:/root/workspace/pdblend/.runtime-deps')
checks=[]
argvs=[['python3',str(C/'B32B-qualified-main-release-v1/release.py'),'check'],['python3','-c',"import sys;sys.path.insert(0,'/root/workspace/pdblend-next-v1/campaign/scale-only-continuation-B32B-v1');import supervise;supervise.package_check();print('CPU package/hash check: no real release/spec yet; no scale run')"],['python3',str(C/'B32B-scale-binding-preparation-v1/prepare.py')],['python3','-c',"import ast,hashlib,json,pathlib;p=pathlib.Path('/root/workspace/pdblend-next-v1/campaign/B32B-scale-current-index-v1');m=json.loads((p/'manifest.json').read_text());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m['files'].items());ast.parse((p/'watch.py').read_text());print('observer CPU hash/syntax check; not started')"]]
for argv in argvs:
 r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=90);checks.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if r.returncode:break
if all(x['exit_code']==0 for x in checks):
 argv=['python3',str(C/'B32B-qualified-main-release-v1/release.py'),'prepare-spec','--eco-binding',str(bp),'--eco-binding-sha256',sha(bp),'--out',str(S/'main-spec.json')]
 r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=90);checks.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
after={str(p):sha(p) for p in guards};save('guards-after.json',after)
proc=Path('/proc/497017/cmdline');live=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None
receipt=dict(hostname=socket.gethostname(),new=new,same=same,checks=checks,guards_unchanged=before==after,live_main_argv=live,finished_s=time.time(),proof_created=False,release_created=False,scale_started=False,observer_started=False)
save('receipt.json',receipt);print(json.dumps(dict(new=len(new),same=len(same),checks=[x['exit_code'] for x in checks],guards_unchanged=before==after,live_main=live)))
assert before==after and all(x['exit_code']==0 for x in checks)
