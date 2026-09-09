"""One bounded handoff to the frozen diagnostic; default is a CPU-only check."""
import argparse, hashlib, importlib.util, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
PACKAGE=CAMPAIGN/'B32B-temporal-observation-execution-v3'
PACKAGE_SHA='8b25efcfc61952691af405ab93693856ba709f9b873c9098c58180fd960a696d'
HOST=CAMPAIGN.parent/'releases/five-system100-B32B-v1-runtime'
SEQUENCE=CAMPAIGN/'B32B-baseline-main-first-sequence-v1/attempt-001'
ATTEMPT=CAMPAIGN/'B32B-temporal-observation-attempt-002'
NODE='iZwz9i5bte3xkpmcoes3t2Z'
DEADLINE=1788872770.0400891
OBSERVER=CAMPAIGN/'B32B-current-index-observer-v3/attempt-001/status.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def require(ok,message):
 if not ok:raise RuntimeError(message)
def proc(pid):
 try:
  p=Path('/proc',str(pid));stat=(p/'stat').read_text();rest=stat[stat.rfind(')')+2:].split();args=(p/'cmdline').read_bytes().split(b'\0');args=[x.decode() for x in args if x]
  return dict(pid=pid,state=rest[0],start_ticks=int(rest[19]),argv=args) if args and rest[0]!='Z' else None
 except FileNotFoundError:return None

def readiness(status,live):
 if not status.get('complete'):
  require(live(status['pid']) is not None,'main supervisor disappeared before terminal record')
  return False
 require(status.get('phase') in ('main_complete','main_incomplete_correctness'),'main stopped/failed; diagnostic forbidden')
 if live(status['pid']) is not None:return False
 require(status.get('steps'),'missing main stage journal')
 for step in status['steps']:
  require(step.get('complete') is True and type(step.get('exitcode')) is int and step['exitcode']==0,'main stage not cleanly terminal')
  if live(step['pid']) is not None:return False
 return True

def frozen():
 m=read(ROOT/'manifest.json')
 for name,digest in m['files'].items():require(sha(ROOT/name)==digest,'launcher source changed: '+name)
 require(sha(PACKAGE/'manifest.json')==PACKAGE_SHA,'frozen diagnostic package changed')
 sys.path.insert(0,str(PACKAGE));import common
 common.package_check();return common

def bind_producers(before,after,spec):
 require(before==after and after.get('actual_main_producers')==90,'main producer evidence changed during prepare')
 for path,digest in after['files'].items():require(spec['files'].get(path)==digest,'actual spec does not freeze producer input: '+path)

def index(spec,pid,handle,phase,terminal,run_status):
 """Metadata only. Live publication requires the exact actual diagnostic argv."""
 current=CAMPAIGN/'current-experiment.json';md=CAMPAIGN/'CURRENT_EXPERIMENT.md';old=read(current)
 actual=proc(pid)
 if not terminal:
  require(actual is not None and actual['start_ticks']==handle['start_ticks'] and actual['argv']==handle['argv'],'diagnostic process changed; do not publish')
  require(str(PACKAGE/'run.py') in actual['argv'] and str(ATTEMPT/'spec.json') in actual['argv'],'foreign diagnostic argv')
 else:require(actual is None,'terminal metadata requires actual process exit')
 history=CAMPAIGN/'current-experiment-history'/(str(time.time_ns())+'-temporal-observation-'+phase)
 history.mkdir(parents=True)
 for p in (current,md):
  if p.is_file():(history/p.name).write_bytes(p.read_bytes())
 value=dict(old);value.update(schema=3,written_s=time.time(),hostname=NODE,model='32b',active_system='diagnostic',implementation_variant='bounded temporal metadata/logits observation; not a performance result',active_phase=phase,queue_pid=pid,queue_running=not terminal,queue_exitcode=run_status.get('exitcode') if terminal else None,supervisor_status=str(ROOT/'attempt-001/status.json'),binding=None,actual_controller_configs={},diagnostic_spec=dict(path=str(ATTEMPT/'spec.json'),sha256=sha(ATTEMPT/'spec.json')),diagnostic_status=str(ATTEMPT/'results/status.json'),diagnostic_engine_config=spec['diagnostic_instance']['config'],output_correctness_verified=False,original_temporal_gate_passed=False,fresh_correctness_and_binding_required_after_restore=True,performance_evidence=False,historical_index=str(history))
 value.pop('bridge',None)
 if terminal:
  restored=ATTEMPT/'results/restored-bootstrap.binding.json'
  value['restored_bootstrap_binding']={'path':str(restored),'sha256':sha(restored)} if restored.is_file() else None
 write(current,value)
 md.write_text('# Current actual-host experiment\n\n32B temporal metadata/logits diagnosis. This is not a baseline performance measurement.\n\nPhase: '+phase+'. Process running: '+str(not terminal)+'.\n\nSpec: '+str(ATTEMPT/'spec.json')+'\n\nStatus: '+str(ATTEMPT/'results/status.json')+'\n\nThe original full 64-token comparison remains unchanged. Restored baseline containers require a fresh correctness gate and binding before subsequent service experiments.\n')
 return dict(path=str(current),sha256=sha(current),phase=phase,queue_running=not terminal)

def no_other_launcher():
 for directory in CAMPAIGN.glob('B32B-temporal-observation-launch-v*'):
  if directory==ROOT:continue
  path=directory/'attempt-001/status.json'
  if path.is_file():
   status=read(path);require(type(status.get('pid')) is int and proc(status['pid']) is None,'another diagnostic launcher is live or lacks real PID: '+str(path))

def launch():
 c=frozen();require(socket.gethostname()==NODE,'wrong actual host');require(not os.environ.get('PDBLEND_NODE_LOCK_FD'),'parent must not inherit or own a lease')
 no_other_launcher()
 out=ROOT/'attempt-001';require(not out.exists(),'launcher is one attempt; existing output is retained');out.mkdir()
 stop=[False]
 for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:stop.__setitem__(0,True))
 state=dict(schema=1,pid=os.getpid(),started_s=time.time(),phase='waiting_three_main',complete=False,automatic_retries=False,parent_owns_lease=False,package_sha256=PACKAGE_SHA,index_publications=[])
 def save():write(out/'status.json',state)
 save();child=None
 try:
  while True:
   require(not stop[0] and not (ROOT/'STOP').exists(),'launcher STOP before diagnostic')
   require(time.time()+1200<DEADLINE,'insufficient original deadline; no diagnostic start')
   s=read(SEQUENCE/'status.json')
   if readiness(s,proc):
    observer=read(OBSERVER)
    if proc(observer['pid']) is None:break
   time.sleep(5)
  no_other_launcher()
  state['phase']='verifying_main90';save();gate=c.main_gate(SEQUENCE/'main-proof.json');state['main_gate']=gate['groups']
  import readiness_producer
  producers=readiness_producer.verify_main_producers(SEQUENCE/'main-proof.json',live=lambda pid:proc(pid) is not None)
  write(out/'producer-proof.before-prepare.json',producers);state['producer_proof_before']=dict(path=str(out/'producer-proof.before-prepare.json'),sha256=sha(out/'producer-proof.before-prepare.json'),actual_main_producers=producers['actual_main_producers']);save()
  require(not ATTEMPT.exists(),'prepared attempt already exists; no retry');require(not stop[0] and not (ROOT/'STOP').exists(),'STOP before prepare')
  env=dict(os.environ);env['PYTHONPATH']=str(HOST/'src')+':'+str(HOST)+':/root/workspace/pdblend/.runtime-deps'
  command=[sys.executable,str(PACKAGE/'prepare.py'),'--prepare','--out',str(ATTEMPT)]
  state['phase']='preparing';state['prepare_argv']=command;save()
  with (out/'prepare.log').open('xb') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,start_new_session=True);state['prepare_pid']=child.pid;save();started=time.monotonic();sent=False
   while child.poll() is None:
    if not sent and (stop[0] or (ROOT/'STOP').exists() or time.monotonic()-started>300):child.send_signal(signal.SIGINT);sent=True
    time.sleep(.25)
   state['prepare_exitcode']=child.returncode;save();require(child.returncode==0 and not sent,'prepare failed/interrupted; retained without retry')
  spec=read(ATTEMPT/'spec.json');spec_sha=sha(ATTEMPT/'spec.json');state['spec']=str(ATTEMPT/'spec.json');state['spec_sha256']=spec_sha
  after_producers=readiness_producer.verify_main_producers(SEQUENCE/'main-proof.json',live=lambda pid:proc(pid) is not None)
  bind_producers(producers,after_producers,spec)
  write(out/'producer-proof.before-run.json',after_producers);state['producer_proof_before_run']=dict(path=str(out/'producer-proof.before-run.json'),sha256=sha(out/'producer-proof.before-run.json'),actual_main_producers=90,spec_sha256=spec_sha);save()
  require(not stop[0] and not (ROOT/'STOP').exists(),'STOP before actual diagnostic');require(time.time()+1020<DEADLINE,'insufficient remaining deadline')
  command=[sys.executable,str(PACKAGE/'run.py'),'--spec',str(ATTEMPT/'spec.json'),'--spec-sha256',spec_sha,'--run']
  state['phase']='running_diagnostic';state['run_argv']=command;save()
  with (out/'diagnostic.log').open('xb') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,close_fds=True,start_new_session=True);state['run_pid']=child.pid;handle=proc(child.pid);require(handle is not None,'diagnostic child exited before live handle');state['run_handle']=handle;save();sent=False;previous=None
   while child.poll() is None:
    if not sent and (stop[0] or (ROOT/'STOP').exists()):child.send_signal(signal.SIGINT);sent=True;state['interrupt_forwarded_s']=time.time();save()
    path=ATTEMPT/'results/status.json';observation=read(path) if path.is_file() else {};phase=observation.get('phase','diagnostic_preflight')
    if phase!=previous:
     try:state['index_publications'].append(index(spec,child.pid,handle,phase,False,{}));previous=phase
     except Exception as e:state.setdefault('index_errors',[]).append(repr(e));previous=phase
     save()
    time.sleep(.5)
   state['run_exitcode']=child.returncode;state['diagnostic_status']=str(ATTEMPT/'results/status.json')
   try:state['index_publications'].append(index(spec,child.pid,handle,'diagnostic_terminal',True,dict(exitcode=child.returncode)))
   except Exception as e:state.setdefault('index_errors',[]).append(repr(e))
   save();require(child.returncode==0,'diagnostic failed; preserve complete failure evidence and do not retry')
  state['phase']='complete'
 except BaseException as e:
  state['phase']='failed';state['error']=repr(e)
  if child is not None and child.poll() is None:
   child.send_signal(signal.SIGINT);state['interrupt_forwarded_s']=time.time();save();child.wait();state['child_exitcode']=child.returncode
 finally:
  state['complete']=True;state['finished_s']=time.time();save()
 return state

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args();frozen()
 if not a.run:print(json.dumps(dict(cpu_only=True,hardware_actions=False,one_attempt=True)));return
 s=launch();print(json.dumps({k:s.get(k) for k in ('phase','spec_sha256','run_pid','run_exitcode','error')}));require(s['phase']=='complete','launcher stopped without a completed observation')
if __name__=='__main__':main()
