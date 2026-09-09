"""Owned local handoff: terminal PDB -> bounded retained restore -> gated baselines."""
import json,time
from pathlib import Path
from operate import remote,sha,write_new
HERE=Path(__file__).resolve().parent;OUT=HERE/'boundary-local-handoff-p4v2'
PACKAGE=HERE/'boundary-baseline-package-p4v2.json'
EXPECTED='a6c24ce13bab049f76d239a974b281eec8922dcdcc144978988b2506d86d75e1'

def read(p):return json.loads(Path(p).read_text())
def save():
 state['updated_s']=time.time();t=OUT/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(OUT/'status.json')
def check():
 assert not (OUT/'STOP').exists(),'owned local handoff STOP; no successor'
 assert sha(PACKAGE)==EXPECTED,'frozen package changed'
 for p,h in read(PACKAGE)['files'].items():assert sha(p)==h,'frozen file changed '+p

def snapshot(path,pid=None):
 code="""import pathlib,json,time
p=pathlib.Path(PATH);v=json.loads(p.read_text()) if p.exists() else None
print(json.dumps(dict(captured_s=time.time(),status=v,alive=pathlib.Path('/proc/'+str(PID)).exists() if PID else None)))"""
 return json.loads(remote('PATH='+repr(str(path))+'\nPID='+repr(pid)+'\n'+code))
def launch(entry,tag):
 check()
 code="""import pathlib,subprocess,json,time
root=pathlib.Path(ROOT);entry=pathlib.Path(ENTRY);tag=TAG
cmd=['python3','-B',str(entry)];check=subprocess.run(cmd,capture_output=True,text=True);assert check.returncode==0,check.stderr
with (root/(tag+'.log')).open('xb') as log:
 p=subprocess.Popen(cmd+['--run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/root/workspace/pdblend-next-v1')
v=dict(pid=p.pid,argv=cmd+['--run'],started_s=time.time(),defaultcheck=check.stdout)
with (root/(tag+'.launch.json')).open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(v))"""
 v=json.loads(remote('ROOT='+repr(str(HERE))+'\nENTRY='+repr(str(entry))+'\nTAG='+repr(tag)+'\n'+code))
 write_new(OUT/(tag+'.launch.json'),json.dumps(v,indent=2).encode());return v

def mirror_terminal_tree(path):
 import base64
 known={str(p):sha(p) for p in path.rglob('*') if p.is_file()}
 code="""import sys,pathlib,json,hashlib,base64
root=pathlib.Path(ROOT);known=json.load(sys.stdin);rows=[]
for p in root.rglob('*'):
 if not p.is_file():continue
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 assert str(p) not in known or known[str(p)]==h,'terminal artifact changed '+str(p)
 if known.get(str(p))!=h:rows.append(dict(path=str(p),sha256=h,data=base64.b64encode(b).decode()))
print(json.dumps(rows))"""
 rows=json.loads(remote('ROOT='+repr(str(path))+'\n'+code,json.dumps(known).encode()))
 for r in rows:
  b=base64.b64decode(r['data']);import hashlib;assert hashlib.sha256(b).hexdigest()==r['sha256'];write_new(Path(r['path']),b)
 return len(rows)

assert not OUT.exists();OUT.mkdir()
state=dict(started_s=time.time(),phase='wait-p4-completion',complete=False,deadline_s=None,campaign_lifecycle='until_declared_complete_v1',gpu_control_scope='C only; one native node lease per active stage',failed=False)
save()
try:
 while True:
  check();v=snapshot(HERE/'p4-completion/status.json',891376);s=v['status'];state['predecessor']=v;save()
  assert s and not s.get('failed') and not s.get('engineering_gate_failed') and s.get('phase')!='failed','PDB failed; no baseline expansion'
  if not v['alive']:
   assert s.get('complete') is True and s['phase']=='complete' and len(s['completed'])==33 and s['node_lease_held'] is False,'PDB incomplete terminal; no successor'
   break
  time.sleep(15)
 state['phase']='restore';state['restore']=launch(HERE/'restore_boundary_baselines_p4v2.py','boundary-retained-restore-p4v2');save()
 while True:
  v=snapshot(HERE/'baseline-boundary-restore-p4v2/deployment-receipt.json',state['restore']['pid']);state['restore_observation']=v;save()
  if not v['alive']:break
  time.sleep(15)
 r=v['status'];mirror_terminal_tree(HERE/'baseline-boundary-restore-p4v2')
 assert r and r.get('complete') is True and r.get('measurement_valid') is True and not r.get('errors'),'restoration failed; no baseline successor'
 check();state['phase']='baseline-queue';state['queue']=launch(HERE/'boundary_baseline_queue_p4v2.py','boundary-baseline-queue-p4v2');save()
 while True:
  v=snapshot(HERE/'boundary-baseline-queue-p4v2/status.json',state['queue']['pid']);state['queue_observation']=v;save()
  if not v['alive']:break
  time.sleep(15)
 q=v['status'];mirror_terminal_tree(HERE/'boundary-baseline-queue-p4v2')
 for name in ['boundary-baseline-bootstrap-p4v2','boundary-baseline-gate-p4v2','boundary-baseline-bindings-p4v2']:
  mirror_terminal_tree(HERE/name)
 assert q and q.get('complete') is True and q['phase']=='complete' and len(q['completed_groups'])==4,'baseline queue stopped or failed; retain evidence and diagnose'
 state.update(phase='complete',complete=True)
except BaseException as exc:
 state.update(phase='failed',failed=True,error=repr(exc));raise
finally:
 state['finished_s']=time.time();save()
