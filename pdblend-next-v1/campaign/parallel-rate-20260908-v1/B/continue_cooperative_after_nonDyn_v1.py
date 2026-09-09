"""No-lease fixed-SLO handoff: terminal non-Dynamo -> fresh native27 -> Dynamo8."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,time
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import cooperative_dynamo_runner_v2 as r
p=r.p

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--previous',type=int,required=True);args=parser.parse_args()
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'handoff must not own GPU lease')
 out=B/'cooperative-dynamo-handoff-001';p.need(not out.exists(),'new immutable handoff output required');out.mkdir()
 _,rule=r.contract();files=dict(rule['source_files']);files.update({str(Path(__file__).resolve()):p.sha(__file__),str(B/'prepare_completed_non_dynamo_fixedslo_v1.py'):p.sha(B/'prepare_completed_non_dynamo_fixedslo_v1.py'),str(r.RULES):p.sha(r.RULES),str(B/f'baseline-reconciliation-{args.previous:03d}/declaration.json'):p.sha(B/f'baseline-reconciliation-{args.previous:03d}/declaration.json')})
 p.write(out/'source-manifest.json',files,exclusive=True)
 state=dict(pid=os.getpid(),started_s=time.time(),complete=False,phase='waiting_original_non_dynamo_without_lease',node_lease_held=False,steps=[],automatic_retries=False,historical_scale11_excluded=True)
 stopping=False
 def stop(*_):
  nonlocal stopping
  stopping=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 def save(**values):state.update(values,updated_s=time.time());p.write(out/'status.json',state)
 def check():
  p.need(not stopping and not (B/'STOP-cooperative-dynamo8').exists(),'STOP at safe stage boundary')
  p.need(all(p.sha(path)==digest for path,digest in files.items()),'frozen handoff source changed')
 def step(name,argv):
  check();save(phase=name)
  with (out/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);save(child_pid=child.pid);code=child.wait()
  state['steps'].append(dict(name=name,argv=argv,exitcode=code));state.pop('child_pid',None);save();p.need(code==0,name+' failed; preserve child and stop successors')
 save()
 try:
  while True:
   check();sp=B/f'baselines-reconciled-{args.previous:03d}/status.json'
   if sp.exists():
    status=p.read(sp)
    if not alive(status['pid']):
     p.need(status['complete'] and not status['failed'] and not status.get('node_lease_held'),'original non-Dynamo predecessor failed/incomplete; manual diagnosis required')
     break
   time.sleep(3)
  step('final_original_ledger',['/usr/bin/python3',str(B/'prepare_completed_non_dynamo_fixedslo_v1.py'),'--previous',str(args.previous)])
  step('fresh_native_gate',['/usr/bin/python3',str(B/'cooperative_dynamo_qualify_v2.py'),'gate'])
  step('derive_ordinary_qualification',['/usr/bin/python3',str(B/'cooperative_dynamo_qualify_v2.py'),'qualify'])
  step('cooperative_eight',['/usr/bin/python3',str(B/'cooperative_dynamo_runner_v2.py'),'--out',str(B/'cooperative-dynamo8-execution-001'),'--run'])
  terminal=p.read(B/'cooperative-dynamo8-execution-001/status.json');p.need(terminal['complete'] and not terminal['failed'] and len(terminal['completed'])==8 and not terminal.get('node_lease_held'),'eight-group terminal incomplete')
  save(phase='complete',complete=True)
 except BaseException as exc:save(phase='stopped_failure',error=repr(exc));raise
 finally:save(finished_s=time.time())
if __name__=='__main__':main()
