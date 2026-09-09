"""One declared C suffix after actual stopped-node restore and fresh full27 gate."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
C=Path(__file__).resolve().parent
ROOT=C/'after-gamma-handoff-002'
RESTORE=C/'baseline-after-gamma-restore-003'
CODE=C/'boundary-continuation-p4v2-004'
GATE=C/'boundary-baseline-gate-after-gamma-002'
BOOT=C/'boundary-baseline-bootstrap-after-gamma-002'
HOST=Path('/root/workspace/pdblend-next-v1/releases/five-system100-C7B-baseline-v1-runtime')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def validate():
 m=read(C/'after-gamma-code-manifest-002.json');assert all(sha(p)==h for p,h in m['files'].items()),'handoff source changed'
 s=read(CODE/'continuation.json');old=read(C/'boundary-p4v2/declaration.json');assert s['logical_declaration']==ref(C/'boundary-p4v2/declaration.json')
 assert len(s['remaining_cell_ids'])==11 and len(s['executed_checkpoints'])==13
 assert set(s['remaining_cell_ids'])=={r['cell_id'] for r in old['cells']}-{x['cell_id'] for x in s['executed_checkpoints']}
 for x in s['executed_checkpoints']:assert sha(x['path'])==x['sha256']
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();validate();assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
 if not a.run:print(json.dumps(dict(passed=True,cpu_only=True,restore_fresh27_then_original_11=True)));return
 assert not ROOT.exists();ROOT.mkdir();s=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_actual_restore',complete=False,node_lease_held=False,steps=[])
 def save():
  t=ROOT/'status.tmp';t.write_text(json.dumps(s,indent=2)+'\n');t.replace(ROOT/'status.json')
 def check():validate();assert not (ROOT/'STOP').exists(),'boundary STOP'
 def run(tag,args):
  check();s.update(phase=tag);save();env=dict(os.environ,PYTHONPATH=f'{HOST}/src:{HOST}:/root/workspace/pdblend/.runtime-deps',PYTHONDONTWRITEBYTECODE='1')
  with (ROOT/(tag+'.log')).open('xb') as f:
   child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env);s['child_pid']=child.pid;save();rc=child.wait()
  s.pop('child_pid',None);s['steps'].append(dict(phase=tag,pid=child.pid,exitcode=rc,finished_s=time.time()));save();assert rc==0 and not alive(child.pid),'child failed: no successor'
 save()
 try:
  launch=read(RESTORE/'launch.json')
  while alive(launch['pid']):check();time.sleep(3)
  receipt=read(RESTORE/'deployment-receipt.json');assert receipt['complete'] and receipt['measurement_valid'] and not receipt['errors'] and len(receipt['restarted'])==8
  assert all(sha(p)==h for p,h in receipt['artifacts'].items()),'restoration evidence changed'
  reservation=read(RESTORE/'startup-port-reservation.json');assert reservation['complete'] and reservation['restored'] and reservation['before']==reservation['after'],'startup port setting not restored before gate'
  binder=C/'baseline-until-complete-v2/bind.py';validator=C/'baseline-until-complete-v2/validate.py'
  run('bootstrap',[sys.executable,'-B',str(binder),'--spec',str(RESTORE/'deployment.json'),'--receipt',str(RESTORE/'deployment-receipt.json'),'--out',str(BOOT)])
  run('fresh27',[sys.executable,'-B',str(validator),'--binding',str(BOOT/'binding.json'),'--runtime-dir',str(C.parents[1]/'AC-baseline-deployment-prepared-v1/C-resident/runtime'),'--out',str(GATE),'--run'])
  gate=read(GATE/'status.json');assert gate['complete'] and gate['passed'] and gate['measurement_valid'] and gate['native_cleanup_complete'] and gate['clock_restore_complete'] and not gate['cleanup_errors']
  files=dict(read(C/'boundary-continuation-p4v2-001/package.json')['files']);files.update(validate()['files']);files.update(read(RESTORE/'deployment.json')['files'])
  for root in (CODE,RESTORE,GATE,BOOT,C/'baseline-strategy-specs-after-gamma-002'):
   files.update({str(f):sha(f) for f in root.rglob('*') if f.is_file()})
  for path in (C/'after-gamma-code-manifest-002.json',C/'run_boundary_baseline_after_gamma_v2.py'):files[str(path)]=sha(path)
  assert all(sha(p)==h for p,h in files.items());write(CODE/'package.json',dict(schema='C-same-logical-suffix-after-fresh-restore-v1',files=files,restore=ref(RESTORE/'deployment-receipt.json'),fresh_gate=ref(GATE/'status.json'),logical_declaration=read(CODE/'continuation.json')['logical_declaration'],actual_source_policy_unchanged=True))
  run('queue-default',[sys.executable,'-B',str(CODE/'queue.py')]);run('remaining11',[sys.executable,'-B',str(CODE/'queue.py'),'--run'])
  final=read(CODE/'execution/status.json');assert final['complete'] and len(final['completed'])==11 and final['node_lease_held'] is False
  s.update(phase='complete',complete=True)
 except BaseException as exc:s.update(phase='failed',error=repr(exc));raise
 finally:s['finished_s']=time.time();save()
if __name__=='__main__':main()
