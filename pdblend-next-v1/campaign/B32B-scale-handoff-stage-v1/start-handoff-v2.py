import hashlib,json,os,socket,subprocess,tarfile,time
from pathlib import Path
R=Path('/root/workspace/pdblend-next-v1');C=R/'campaign';S=C/'B32B-scale-handoff-stage-v1';P=C/'B32B-main-to-scale-handoff-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert socket.gethostname()=='iZwz9i5bte3xkpmcoes3t2Z' and 'PDBLEND_NODE_LOCK_FD' not in os.environ
archive=S/'B32B-main-to-scale-handoff-v2.tar.gz';assert sha(archive)=='3badc80a04442d62d267e624624912647c94ff36ab870724c948b4e6d8afe814'
guards=json.loads((S/'guards-after.json').read_text());assert all(sha(p)==h for p,h in guards.items())
new=[];same=[]
with tarfile.open(archive) as t:
 ms=t.getmembers();assert len(ms)==5 and len({m.name for m in ms})==5
 for m in ms:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  p=R/m.name;assert P in p.parents
  if p.exists():assert not p.is_symlink() and p.read_bytes()==t.extractfile(m).read()
 for m in ms:
  p=R/m.name
  if p.exists():same.append(str(p))
  else:p.parent.mkdir(parents=True,exist_ok=True);p.open('xb').write(t.extractfile(m).read());new.append(str(p))
assert sha(P/'manifest.json')=='a2b798d9f26a42c9736e6aaf954c704db2a8bfb825c39de7f20bd1446f9fc2ee'
host=R/'releases/five-system100-B32B-v1-runtime';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{host}/src:{host}:/root/workspace/pdblend/.runtime-deps')
argv=['python3','-u',str(P/'handoff.py')]
r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=60)
check=dict(exitcode=r.returncode,stdout=r.stdout,stderr=r.stderr)
assert r.returncode==0 and not (P/'attempt-001').exists() and not (P/'launch.json').exists()
assert all(sha(p)==h for p,h in guards.items())
argv+=['--run','--out',str(P/'attempt-001')]
with (P/'handoff.log').open('xb') as log:
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,close_fds=True,start_new_session=True)
launch=dict(pid=child.pid,argv=argv,started_s=time.time(),manifest_sha256=sha(P/'manifest.json'))
(P/'launch.json').open('x').write(json.dumps(launch,indent=2)+'\n')
time.sleep(1)
actual=[s.decode() for s in Path('/proc',str(child.pid),'cmdline').read_bytes().split(b'\0') if s]
assert actual==argv
status=json.loads((P/'attempt-001/status.json').read_text());assert status['phase']=='waiting_actual_eco_main' and status['complete'] is False
mainactual=[s.decode() for s in Path('/proc/497017/cmdline').read_bytes().split(b'\0') if s]
count=len(list((C/'B32B-baseline-main-first-sequence-v1/attempt-001/bindings/ecoserve/results/checkpoints').glob('*.json')))
receipt=dict(hostname=socket.gethostname(),new=new,same=same,default_check=check,launch=launch,actual_argv=actual,status=status,main_pid=497017,main_actual_argv=mainactual,eco_checkpoint_count=count,guards_unchanged=all(sha(p)==h for p,h in guards.items()),finished_s=time.time())
(S/'handoff-v2-start-receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(pid=child.pid,phase=status['phase'],main_pid=497017,eco_checkpoint_count=count,default_exit=check['exitcode'])))
