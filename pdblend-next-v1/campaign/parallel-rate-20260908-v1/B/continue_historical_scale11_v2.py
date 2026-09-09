"""Wait without a lease for all new-rate baselines, then execute original Eco suffix."""
import hashlib,json,os,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
OUT=B/'historical-scale11-continuation-002'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False
def save(s):
 s['updated_s']=time.time();p=OUT/'status.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(p)
def main():
 assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
 OUT.mkdir(exist_ok=False)
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_new_rates_without_lease',node_lease_held=False,complete=False,automatic_retries=False,campaign_lifecycle='until_declared_complete_v1');save(state)
 try:
  declaration=B/'historical-ecoserve-scale11-declaration-v2.json';digest=sha(declaration);d=read(declaration)
  for p,h in d['source_files'].items():assert sha(p)==h
  assert read(B/'historical-scale11-cpu-validation-v2.json')['passed']
  while True:
   if (B/'STOP-historical-scale11').exists():raise RuntimeError('historical STOP before GPU stage')
   prior=read(B/'continuation-completion-002/status.json')
   if not alive(prior['pid']):
    assert prior.get('complete') is True and not prior.get('node_lease_held'),'new-rate predecessor failed/incomplete; historical suffix must wait for explicit reconciliation'
    break
   time.sleep(5)
  assert sha(declaration)==digest
  for p,h in d['source_files'].items():assert sha(p)==h
  state.update(phase='historical_scale11',declaration=dict(path=str(declaration),sha256=digest));save(state)
  with (OUT/'runner.log').open('xb') as log:
   child=subprocess.Popen(['python3','-u',str(B/'historical_scale11_v2.py'),'--run'],cwd=B,stdout=log,stderr=subprocess.STDOUT)
   state['child_pid']=child.pid;save(state);code=child.wait()
  state['exitcode']=code;assert code==0,'historical suffix failed; retain observations and stop'
  result=read(B/'historical-ecoserve-scale11-002/status.json')
  assert result['complete'] and not result['failed'] and len(result['completed'])==11 and not result['node_lease_held']
  state.update(phase='complete',complete=True)
 except BaseException as exc:state.update(phase='stopped_failure_or_incomplete',error=repr(exc));raise
 finally:state.pop('child_pid',None);state['finished_s']=time.time();save(state)
if __name__=='__main__':main()
