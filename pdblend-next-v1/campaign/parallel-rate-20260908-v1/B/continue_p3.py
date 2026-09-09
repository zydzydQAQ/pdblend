"""Observe terminal leases, then launch each authorized stage exactly once."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEADLINE=1788872770.0400891

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def need(ok,why):
 if not ok:raise ValueError(why)
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
 q=argparse.ArgumentParser();q.add_argument('--manifest',type=Path,required=True);q.add_argument('--manifest-sha256',required=True);a=q.parse_args()
 need('PDBLEND_NODE_LOCK_FD' not in os.environ,'supervisor must never hold GPU lease')
 out=ROOT/'continuation-p3-001';need(not out.exists(),'fresh supervisor required');out.mkdir()
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_fixed16',complete=False,steps=[],failed=[],automatic_retries=False)
 def save():state['updated_s']=time.time();write(out/'status.json',state)
 def check():
  need(sha(a.manifest)==a.manifest_sha256,'continuation manifest changed')
  m=read(a.manifest)
  for p,h in m['files'].items():need(sha(p)==h,'frozen continuation input changed '+p)
 def stage(name,cmd,reserve):
  check();need(time.time()+reserve<DEADLINE,'insufficient full stage and cleanup reserve')
  state['phase']=name;record=dict(name=name,argv=cmd,started_s=time.time());state['steps'].append(record);save()
  with (out/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   record['pid']=child.pid;save()
   while child.poll() is None:
    # The child itself uses the original cutoff and cleanup. Never interrupt it.
    time.sleep(5)
   record.update(exitcode=child.returncode,finished_s=time.time());save()
  need(child.returncode==0,'stage failed; retained actual evidence '+name)
 try:
  check();save()
  while True:
   fixed=read(ROOT/'fixed-screen-p3-001/status.json')
   need(not fixed.get('failed') and fixed.get('phase')!='failed','fixed screening engineering failure')
   if not alive(fixed['pid']):
    need(fixed.get('complete') and len(fixed['completed'])==16 and not fixed.get('node_lease_held'),'fixed16 incomplete; no continuation')
    break
   need(time.time()<DEADLINE,'deadline reached while observing predecessor');time.sleep(5)
  stage('added-pdb',[sys.executable,str(ROOT/'added_runner_p3.py'),'--system-group','pdblend','--out',str(ROOT/'added-pdb-p3-001'),'--run'],1100)
  new=read(ROOT/'added-pdb-p3-001/status.json');need(new.get('complete') and len(new['completed'])==6 and not new['failed'],'new PDB repeat group incomplete')
  stage('baseline-restore',[sys.executable,str(ROOT/'baseline_control_p3.py'),'restore'],1500)
  stage('baseline-original27',[sys.executable,str(ROOT/'baseline_control_p3.py'),'gate'],900)
  stage('baseline-qualify',[sys.executable,str(ROOT/'baseline_control_p3.py'),'qualify'],400)
  stage('added-baselines',[sys.executable,str(ROOT/'added_runner_p3.py'),'--system-group','baselines','--bindings',str(ROOT/'baseline-bindings-p3.json'),'--out',str(ROOT/'added-baselines-p3-001'),'--run'],400)
  end=read(ROOT/'added-baselines-p3-001/status.json');state.update(complete=bool(end.get('complete')),phase='complete' if end.get('complete') else 'stopped_at_boundary',remaining=end.get('remaining'))
 except BaseException as exc:state.update(phase='failed_or_incomplete',error=repr(exc));state['failed'].append(repr(exc));raise
 finally:state.update(finished_s=time.time());save()

if __name__=='__main__':main()
