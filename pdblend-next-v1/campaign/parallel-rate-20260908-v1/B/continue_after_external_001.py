"""No-lease owner: natural restore -> full fresh qualification -> ten exact cells."""
import os,sys,subprocess,time,json
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import baseline_control_after_external_v2 as c
p=c.p;OUT=B/'after-external-continuation-001'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def main():
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ and not OUT.exists(),'fresh unleased continuation required');OUT.mkdir()
 paths=[Path(__file__).resolve(),B/'baseline_control_after_external_v2.py',B/'baseline-after-external-source-manifest-v2.json',B/'baseline_runner_reconciled_006.py',B/'baseline-reconciliation-006/declaration.json',B/'baseline-reconciliation-006/launch-manifest.json']
 pins={str(f):p.sha(f) for f in paths};p.write(OUT/'source-manifest.json',pins,exclusive=True)
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_natural_restore',node_lease_held=False,complete=False,steps=[],automatic_retries=False)
 def save(**v):state.update(v,updated_s=time.time());p.write(OUT/'status.json',state)
 def check():
  p.need(not (B/'STOP-after-external').exists(),'STOP prevents next child')
  p.need(all(p.sha(f)==h for f,h in pins.items()),'continuation source changed')
 def run(name,args):
  check();save(phase=name)
  with (OUT/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);save(child_pid=child.pid);code=child.wait()
  state['steps'].append(dict(phase=name,exitcode=code,finished_s=time.time()));state.pop('child_pid',None);save();p.need(code==0,name+' failed: no automatic successor')
 save()
 try:
  launch=p.read(B/'baseline-after-external-restore-launch-001.json');p.need(launch['pid']==877612,'wrong restore owner')
  while alive(launch['pid']):check();time.sleep(3)
  restore=p.read(B/'baseline-return-after-external-001/status.json');p.need(restore['complete'] and restore['clock_restore_complete'] and not restore.get('error') and not restore.get('errors') and not restore.get('sampling_error'),'natural restoration failed')
  c.external_predecessor();run('fresh_original27',[sys.executable,'-u',str(B/'baseline_control_after_external_v2.py'),'gate'])
  run('fresh_four_bindings',[sys.executable,'-u',str(B/'baseline_control_after_external_v2.py'),'qualify'])
  refs=p.read(B/'baseline-bindings-after-external-001.json');p.need(set(refs)==set(p.BASELINES),'four new bindings required')
  for system,r in refs.items():p.need(c.audit_performance(system,p.checked(r))['passed'],'fresh actual mechanism qualification failed')
  run('non_dynamo10',[sys.executable,'-u',str(B/'baseline_runner_reconciled_006.py'),'--system-group','baselines','--bindings',str(B/'baseline-bindings-after-external-001.json'),'--out',str(B/'baselines-reconciled-006'),'--run'])
  terminal=p.read(B/'baselines-reconciled-006/status.json');p.need(terminal['complete'] and not terminal['failed'] and not terminal['node_lease_held'] and not alive(terminal['pid']),'all ten must finish naturally')
  save(phase='complete',complete=True)
 except BaseException as exc:save(phase='stopped_failure',error=repr(exc));raise
 finally:save(finished_s=time.time())
if __name__=='__main__':main()
