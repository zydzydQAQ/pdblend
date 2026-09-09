"""One B Eco-main terminal handoff to the unchanged scale executor; default CPU only."""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT.parent
NODE='iZwz9i5bte3xkpmcoes3t2Z';DEADLINE=1788872770.0400891
MAIN=C/'B32B-ecoserve-qualified-main-launch-v1';BIND=C/'B32B-ecoserve-qualified-main-v1/binding.json'
SOURCE=C/'five-system-fixed-window-v1/sources/B32B/manifest.json'
BIND_SHA='87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119';SOURCE_SHA='4b9494c6b0a38cb9d44dbc490530d88e9a0eec76b44854f6e40db0328bbfe5ed'
RELEASE=C/'B32B-qualified-main-release-v1';SCALE=C/'scale-only-continuation-B32B-v1';HELPER=C/'B32B-scale-binding-preparation-v1';OBSERVER=C/'B32B-scale-current-index-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def ref(p):return dict(path=str(p),sha256=sha(p))
def proc(pid):
 try:
  p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(') ',1)[1].split();argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];a=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return dict(pid=pid,start_ticks=int(s[19]),argv=argv) if argv and s[0]!='Z' and a[0]!='Z' and a[19]==s[19] else None
 except (FileNotFoundError,ProcessLookupError):return None

def frozen():
 m=read(ROOT/'manifest.json')
 for name,h in m['files'].items():require(sha(ROOT/name)==h,'handoff source changed')
 for path,h in m['dependencies'].items():require(sha(path)==h,'handoff frozen dependency changed '+path)
 require('PDBLEND_NODE_LOCK_FD' not in os.environ,'handoff cannot inherit node lease')
 return m

def ready(launch,terminal,invocations,live):
 require(launch['binding_sha256']==BIND_SHA and launch['source_sha256']==SOURCE_SHA,'actual main binding/source changed')
 expected=['python3','-u',str(C/'five-system-execution-v3/run.py'),'--manifest',str(SOURCE),'--binding',str(BIND),'--system','ecoserve','--phase','main','--max-cells','30','--run']
 require(launch['argv']==expected and type(launch['pid']) is int and type(launch['monitor_pid']) is int,'actual main launch declaration differs')
 active=live(launch['pid']);monitor=live(launch['monitor_pid'])
 if active:
  require(active['argv']==expected,'wrong actual main PID/argv')
  return False
 if terminal is None:
  require(monitor is not None,'main producer and monitor gone without terminal evidence')
  return False
 for key in ('pid','monitor_pid','argv','started_s','binding_sha256','source_sha256'):
  require(terminal[key]==launch[key],'terminal launch attribution changed')
 require(type(terminal.get('exit_code')) is int and terminal['exit_code']==0 and terminal['finished_s']>=launch['started_s'],'actual main failed or unconfirmed; no continuation')
 if monitor is not None:return False # lets original terminal index publication finish
 matches=[i for i in invocations if i.get('pid')==launch['pid']]
 require(len(matches)==1,'exactly one actual Eco main producer required')
 inv=matches[0]
 require(inv.get('complete') is True and not inv.get('error') and inv.get('binding_sha256')==BIND_SHA and inv.get('manifest_sha256')==SOURCE_SHA and inv.get('phase')=='main' and inv.get('system')=='ecoserve','Eco invocation is not a clean actual terminal')
 require(launch['started_s']<=inv['started_s']<=inv['finished_s']<=terminal['finished_s']<DEADLINE,'actual main terminal timestamps differ')
 require(sha(SOURCE)==SOURCE_SHA,'actual main source changed')
 expected_ids={r['cell_id'] for r in read(SOURCE)['cells'] if r['system']=='ecoserve' and r['phase']=='main'}
 require(len(expected_ids)==30 and inv['declared_selected_cells']==30 and len(inv['completed'])==30
         and set(inv['completed'])==expected_ids and not inv.get('skipped'),'exact declared30 must be actual producer completions')
 return True

class Handoff:
 def __init__(self,out):self.out=out;self.stop=False;self.state=None;self.child=None;self.observer=None
 def persist(self):save(self.out/'status.json',self.state)
 def stopping(self):return self.stop or (ROOT/'STOP').exists() or (self.out/'STOP').exists()
 def boundary(self):
  require(not self.stopping(),'handoff STOP; no successor')
  require(time.time()+600<DEADLINE,'original deadline lacks scale start and cleanup reserve')
  frozen()
 def command(self,label,argv,env):
  self.boundary();step=dict(label=label,argv=argv,started_s=time.time(),complete=False);self.state['steps'].append(step);self.state['phase']=label;self.persist()
  with (self.out/(label+'.log')).open('xb') as log:
   self.child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,close_fds=True,start_new_session=True)
   step['pid']=self.child.pid;self.persist();sent=False
   while self.child.poll() is None:
    if not sent and (self.stopping() or time.time()-step['started_s']>600):
     self.child.send_signal(signal.SIGINT);sent=True;step['interrupt_s']=time.time();self.persist()
    if sent and time.time()-step['interrupt_s']>120:
     step['exit_unconfirmed']=True;self.persist();raise RuntimeError('CPU phase exit unconfirmed; no successor')
    time.sleep(.25)
   step.update(complete=True,exitcode=self.child.returncode,finished_s=time.time());self.persist()
   require(not sent and self.child.returncode==0,'CPU phase failed/interrupted; preserve partial outputs and stop')
 def observe_exit(self):
  if self.observer is not None and self.observer.poll() is not None and 'observer_exitcode' not in self.state:
   self.state.update(observer_exitcode=self.observer.returncode,observer_error='metadata observer exited; performance remains governed by the frozen executor',observer_exit_observed_s=time.time());self.persist()
 def run(self):
  frozen();require(socket.gethostname()==NODE,'actual B host only');require(not self.out.exists(),'new single handoff attempt required')
  self.out.mkdir();self.state=dict(schema=1,model='32b',pid=os.getpid(),started_s=time.time(),phase='waiting_actual_eco_main',complete=False,steps=[],parent_owns_node_lease=False,automatic_retries=False,deadline_s=DEADLINE)
  for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(self,'stop',True))
  self.persist()
  try:
   with (ROOT/'handoff.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    b=read(BIND);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f"{b['host_release']}/src:{b['host_release']}:/root/workspace/pdblend/.runtime-deps")
    while True:
     self.boundary();launch=read(MAIN/'launch.json');tp=MAIN/'terminal.json'
     invs=[read(p) for p in (Path(b['output'])/'invocations').glob('*.json')]
     if ready(launch,read(tp) if tp.exists() else None,invs,proc):break
     time.sleep(5)
    save(self.out/'main-terminal.json',dict(launch=ref(MAIN/'launch.json'),terminal=ref(tp),observed_s=time.time(),producer_live=False,monitor_live=False))
    spec=self.out/'main-spec.json';proof=self.out/'main-proof.json';release=self.out/'B32B-model-release.json'
    self.command('declare_actual_main',['python3',str(RELEASE/'release.py'),'prepare-spec','--eco-binding',str(BIND),'--eco-binding-sha256',BIND_SHA,'--out',str(spec)],env)
    self.command('prove_actual_main150',['python3',str(RELEASE/'release.py'),'prove-main','--spec',str(spec),'--out',str(proof)],env)
    self.command('assemble_verified_release',['python3',str(RELEASE/'release.py'),'assemble-release','--proof',str(proof),'--proof-sha256',sha(proof),'--out',str(release)],env)
    bindings=self.out/'scale-bindings'
    self.command('derive_fresh_scale_bindings',['python3',str(HELPER/'prepare.py'),'--prepare','--release',str(release),'--release-sha256',sha(release),'--out',str(bindings)],env)
    scale_spec=bindings/'spec.json';base=['python3','-u',str(SCALE/'supervise.py'),'--spec',str(scale_spec),'--spec-sha256',sha(scale_spec),'--release',str(release),'--release-sha256',sha(release)]
    self.command('scale_default_check',base,env)
    self.boundary()
    for g in read(scale_spec)['groups']:
     if g['identity_mode']!='reuse_only':require(not (Path(read(g['scale_binding']['path'])['output'])/'STOP').exists(),'original output STOP preserved; no scale start')
    scale_out=self.out/'scale';argv=base+['--run','--out',str(scale_out)]
    with (self.out/'scale.log').open('xb') as log:
     self.child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,close_fds=True,start_new_session=True)
     self.state.update(phase='running_scale',scale_pid=self.child.pid,scale_argv=argv,scale_spec=ref(scale_spec),release=ref(release),scale_status=str(scale_out/'status.json'));self.persist();sent=False
     while self.child.poll() is None:
      if not sent and (self.stopping() or time.time()+520>=DEADLINE):
       # Frozen scale supervisor turns SIGTERM into STOP at its cell boundary.
       self.child.send_signal(signal.SIGTERM);sent=True;self.state['scale_boundary_stop_sent_s']=time.time();self.persist()
      if self.observer is None and (scale_out/'status.json').exists():
       op=[sys.executable,str(OBSERVER/'watch.py'),'--status',str(scale_out/'status.json')]
       with (self.out/'observer.log').open('xb') as observer_log:
        self.observer=subprocess.Popen(op,stdout=observer_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,close_fds=True,start_new_session=True)
       self.state.update(observer_pid=self.observer.pid,observer_argv=op);self.persist()
      self.observe_exit()
      if time.time()>=DEADLINE+5:
       self.state['scale_exit_unconfirmed']=True;self.persist();raise RuntimeError('original scale cleanup exit unconfirmed; no automatic recovery')
      time.sleep(1)
     self.state['scale_exitcode']=self.child.returncode;self.persist()
     require(not sent and self.child.returncode==0,'scale stopped/failed; original raw and energy retained; no retry')
     final=read(scale_out/'status.json');require(final.get('complete') is True and final.get('phase')=='selected_scale_groups_finished','scale producer did not publish actual terminal')
    self.state.update(complete=True,phase='complete')
  except BaseException as e:
   self.state.update(phase='stopped' if self.stopping() else 'failed',error=repr(e));raise
  finally:
   self.state['finished_s']=time.time();self.persist()
   # The scale observer reads terminal status and exits itself; no serving controls here.
  return self.state

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args();frozen()
 if not a.run:print(json.dumps(dict(cpu_only=True,main150_asserted=False,scale_started=False,original_deadline=DEADLINE)));return
 require(a.out is not None,'new handoff output required');Handoff(a.out.resolve()).run()
if __name__=='__main__':main()
