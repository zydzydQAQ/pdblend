"""Wait without a lease, then acquire only through each frozen complete stage."""
import json,os,subprocess,time,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
OUT=B/'continuation-completion-001'
STATUS=OUT/'status.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def save(state):
 state['updated_s']=time.time();t=STATUS.with_suffix('.json.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(STATUS)

def main():
 assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
 OUT.mkdir(exist_ok=False)
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_pdb_without_lease',node_lease_held=False,complete=False,automatic_retries=False,campaign_lifecycle='until_declared_complete_v1',steps=[]);save(state)
 try:
  manifest=read(B/'baseline-completion-source-manifest.json')
  for p,h in manifest['files'].items():assert sha(p)==h
  assert read(B/'baseline-completion-cpu-validation.json')['passed']
  while True:
   pdb=read(B/'fixed-screen-p4-001/status.json')
   if not alive(pdb['pid']):
    assert pdb.get('complete') is True and not pdb['failed'] and not pdb.get('node_lease_held'),'PDB did not finish qualified selection; do not restore/expand'
    break
   time.sleep(5)
  state['pdb_terminal']=dict(path=str(B/'fixed-screen-p4-001/status.json'),sha256=sha(B/'fixed-screen-p4-001/status.json'))
  commands=[('restore',[str(B/'baseline_control_completion.py'),'restore']),('gate',[str(B/'baseline_control_completion.py'),'gate']),('qualify',[str(B/'baseline_control_completion.py'),'qualify'])]
  for repeat in (1,2):
   commands.append(('baseline_repeat'+str(repeat),[str(B/'baseline_runner_completion.py'),'--system-group','baselines','--repeat',str(repeat),'--bindings',str(B/'baseline-bindings-completion.json'),'--out',str(B/('baselines-completion-r'+str(repeat)+'-001')),'--run']))
  for label,argv in commands:
   for p,h in manifest['files'].items():assert sha(p)==h
   state['phase']=label;save(state)
   log=OUT/(label+'.log');started=time.time()
   with log.open('xb') as stream:
    child=subprocess.Popen(['python3','-u',*argv],cwd=B,stdout=stream,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;save(state);code=child.wait()
   state['steps'].append(dict(phase=label,argv=argv,started_s=started,finished_s=time.time(),exitcode=code,log=str(log)));save(state)
   assert code==0,label+' failed; retain all observed work and stop'
  state.update(phase='complete',complete=True)
 except BaseException as exc:
  state.update(phase='stopped_failure_or_incomplete',error=repr(exc));raise
 finally:
  state.pop('child_pid',None);state['finished_s']=time.time();save(state)

if __name__=='__main__':main()
