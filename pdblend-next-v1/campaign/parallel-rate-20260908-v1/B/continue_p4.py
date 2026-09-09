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
 out=ROOT/'continuation-p4-001';need(not out.exists(),'fresh supervisor required');out.mkdir()
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
   fixed=read(ROOT/'fixed-screen-p4-001/status.json')
   need(not fixed.get('failed') and fixed.get('phase')!='failed','fixed screening engineering failure')
   if not alive(fixed['pid']):
    need(fixed.get('complete') and len(fixed['completed'])==16 and not fixed.get('node_lease_held'),'fixed16 incomplete; no continuation')
    break
   need(time.time()<DEADLINE,'deadline reached while observing predecessor');time.sleep(5)
  stage('added-pdb-first-repeat',[sys.executable,str(ROOT/'added_runner_p4.py'),'--system-group','pdblend','--repeat','1','--out',str(ROOT/'added-pdb-p4-001'),'--run'],800)
  new=read(ROOT/'added-pdb-p4-001/status.json');need(new.get('complete') and len(new['completed'])==3 and not new['failed'],'new PDB first-repeat group incomplete')
  stage('baseline-restore',[sys.executable,str(ROOT/'baseline_control_p4.py'),'restore'],1500)
  stage('baseline-original27',[sys.executable,str(ROOT/'baseline_control_p4.py'),'gate'],900)
  stage('baseline-qualify',[sys.executable,str(ROOT/'baseline_control_p4.py'),'qualify'],400)
  stage('added-baselines-first-repeat',[sys.executable,str(ROOT/'added_runner_p4.py'),'--system-group','baselines','--repeat','1','--bindings',str(ROOT/'baseline-bindings-p4.json'),'--out',str(ROOT/'added-baselines-p4-001'),'--run'],400)
  first=read(ROOT/'added-baselines-p4-001/status.json')
  declaration=read(ROOT/'added-rates-p4-001/declaration.json')
  remaining_pdb=[c['cell_id'] for c in declaration['cells'] if c['system']=='pdblend' and c['repeat']==2]
  remaining_baseline2=[c['cell_id'] for c in declaration['cells'] if c['system']!='pdblend' and c['repeat']==2]
  if first.get('complete') and time.time()+400<DEADLINE:
   stage('added-baselines-second-repeat',[sys.executable,str(ROOT/'added_runner_p4.py'),'--system-group','baselines','--repeat','2','--bindings',str(ROOT/'baseline-bindings-p4.json'),'--out',str(ROOT/'added-baselines-p4-repeat2-001'),'--run'],400)
   second=read(ROOT/'added-baselines-p4-repeat2-001/status.json');remaining_baseline2=second.get('remaining',remaining_baseline2)
  state.update(complete=False,first_repeat_five_system_complete=bool(first.get('complete')),phase='priority_first_repeat_finished' if first.get('complete') else 'stopped_at_boundary',
   remaining=first.get('remaining',[])+remaining_baseline2+remaining_pdb,
   deferred_pdb_second_repeat_reason='explicit first-round five-system priority; avoid two additional measured node restorations before fixed deadline')
 except BaseException as exc:state.update(phase='failed_or_incomplete',error=repr(exc));state['failed'].append(repr(exc));raise
 finally:state.update(finished_s=time.time());save()

if __name__=='__main__':main()
