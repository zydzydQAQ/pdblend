"""Wait without a lease, audit the completed exact-trace gate, then run full27 once."""
import pathlib,json,hashlib,subprocess,sys,time,asyncio,os
A=pathlib.Path(__file__).resolve().parent
OUT=A/'p6-load-continuation-001'
SPEC=A/'load-p6-full-inputs-001/spec.json'
SPEC_SHA='b9035d5872641c7b869a3cd49163e8c31f06e285f8496d201ac8087d428b987b'
CODE=A/'load-p6-code-001';sys.path.insert(0,str(CODE))
from capacity_executor import require,durable,fixed,sha
from capacity_certificate import raw_measurement

def alive(pid):
 try:return pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def main():
 require('PDBLEND_NODE_LOCK_FD' not in os.environ,'waiter must not inherit lease')
 require(not OUT.exists(),'fresh continuation required');OUT.mkdir()
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_gate_without_lease',complete=False,node_lease_held=False)
 def update(**v):state.update(v,updated_s=time.time());durable(OUT/'status.json',state)
 update()
 try:
  require(sha(SPEC)==SPEC_SHA,'full declaration changed');sp=json.loads(SPEC.read_text())
  require(all(sha(p)==h for p,h in sp['files'].items()),'full physical source changed')
  gate=A/'load-p6-gate-001'
  while True:
   require(not (A/'STOP_LOAD_P6_FULL').exists(),'STOP prevents successor')
   prior=json.loads((gate/'status.json').read_text())
   if not alive(prior['pid']):break
   time.sleep(2)
  require(prior.get('complete') and prior.get('cleanup_complete') and not prior.get('error') and not prior.get('cleanup_errors'),'failed gate prevents full qualification')
  r=json.loads((gate/'underload_gate/result.json').read_text());require(r.get('work_complete') is True and r.get('failed_requests')==0 and r.get('request_timeouts')==0 and r['n_expected']==752,'request failure prevents full qualification')
  raw_measurement(r['raw_measurement']);raw_measurement(prior['full_operation_measurement'])
  i=json.loads((gate/'inventory.json').read_text());require(i['complete'] and not i['transition_inflight'] and {x['id'] for x in i['active_instances']}=={'nextv3a6','nextv3a7'},'gate did not actually return to initial two')
  require(not any(e['kind'] in ('transition_failed','rollback_failed') for e in i['events']),'physical failure prevents successor')
  require(json.loads((gate/'remove.json').read_text())['execution_verified'],'actual remove receipt required')
  require(set(subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split())=={'pdb-v2-nextv3a6','pdb-v2-nextv3a7'},'unexpected live container before full qualification')
  require(sha(SPEC)==SPEC_SHA and all(sha(p)==h for p,h in sp['files'].items()),'successor source changed')
  durable(OUT/'gate-audit.json',dict(passed=True,source_status_sha256=sha(gate/'status.json'),source_result_sha256=sha(gate/'underload_gate/result.json'),inventory_sha256=sha(gate/'inventory.json'),all8_energy_recomputed=True,actual_two_restored=True))
  args=[sys.executable,'-u',str(CODE/'capacity_load_calibrate.py'),'--spec',str(SPEC),'--spec-sha256',SPEC_SHA,'--out',str(A/'load-p6-full-001'),'--run']
  with (OUT/'runner.log').open('xb') as log:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);update(phase='full27',child_pid=child.pid);code=child.wait()
  update(exitcode=code);require(code==0,'full27 failed; retain all observations')
  terminal=json.loads((A/'load-p6-full-001/status.json').read_text());require(terminal.get('complete') and terminal.get('cleanup_complete') and not terminal.get('error') and len(terminal['completed'])==27,'full27 terminal incomplete')
  update(phase='complete',complete=True)
 except BaseException as exc:update(phase='stopped_failure',error=repr(exc));raise
 finally:update(finished_s=time.time())
if __name__=='__main__':main()
