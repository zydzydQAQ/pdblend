"""No-lease continuation: complete full27 -> raw certificate -> fixed/dynamic900 pair."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
A=Path(__file__).resolve().parent;CODE=A/'load-p6-code-001';sys.path.insert(0,str(CODE))
from capacity_executor import fixed,require,durable,sha
from capacity_certificate import raw_measurement,window_energy,ref
OUT=A/'p6-qualification-continuation-001'

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def audit(out,expected):
 s=json.loads((out/'status.json').read_text());require(s.get('complete') and s.get('cleanup_complete') and not s.get('error') and not s.get('cleanup_errors') and s.get('finished_s'),'work or cleanup incomplete '+str(out))
 require(len(s['completed'])==expected,'declared observation count differs')
 require(not alive(s['pid']),'predecessor still owns stage')
 raw_measurement(s['full_operation_measurement'])
 for reference in s['completed']:
  r=fixed(reference)
  if r.get('schema')=='capacity-load-measurement-v1':require(r.get('work_complete') is True and r.get('failed_requests')==0 and r.get('request_timeouts')==0,'request failure prevents any successor')
 inv=out/'inventory.json'
 if inv.exists():
  i=json.loads(inv.read_text());require(i['complete'] and not i['transition_inflight'] and {x['id'] for x in i['active_instances']}=={'nextv3a6','nextv3a7'},'actual initial-two cleanup required')
  require(not any(e['kind'] in ('transition_failed','rollback_failed') for e in i['events']),'physical failure in predecessor')
 require(set(subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).split())=={'pdb-v2-nextv3a6','pdb-v2-nextv3a7'},'unexpected live process on A')
 return s

def main():
 require('PDBLEND_NODE_LOCK_FD' not in os.environ,'waiter may not hold a node lease');require(not OUT.exists(),'fresh continuation output');OUT.mkdir()
 state=dict(pid=os.getpid(),started_s=time.time(),phase='waiting_full27_without_lease',node_lease_held=False,complete=False,steps=[])
 sources=[A/'register_alpaca_capacity_p6.py',A/'prepare_p6_qualification900.py',CODE/'capacity_load_calibrate.py',A/'load-p6-full-inputs-002/spec.json'];pins={str(p):sha(p) for p in sources};durable(OUT/'source-manifest.json',pins)
 def update(**v):state.update(v,updated_s=time.time());durable(OUT/'status.json',state)
 def check():
  require(not (A/'STOP_P6_QUALIFICATION900').exists(),'STOP prevents next stage');require(all(sha(p)==h for p,h in pins.items()),'prepared continuation source changed')
 def run(name,args):
  check();update(phase=name)
  with (OUT/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);update(child_pid=child.pid);code=child.wait()
  state['steps'].append(dict(phase=name,exitcode=code));state.pop('child_pid',None);update();require(code==0,name+' failed; no automatic retry')
 update()
 try:
  while True:
   check();p=A/'load-p6-full-002/status.json';require(p.exists(),'declared full27 owner status missing');s=json.loads(p.read_text())
   if not alive(s['pid']):break
   time.sleep(3)
  audit(A/'load-p6-full-002',27);run('register',[sys.executable,str(A/'register_alpaca_capacity_p6.py')]);run('prepare900',[sys.executable,str(A/'prepare_p6_qualification900.py')])
  declaration=json.loads((A/'p6-qualification900-inputs-001/declaration.json').read_text());results={}
  for arm in ['fixed2','dynamic']:
   sr=declaration['specs'][arm];sp=fixed(sr);out=A/f'p6-qualification900-{arm}-001'
   run(arm,[sys.executable,'-u',str(CODE/'capacity_load_calibrate.py'),'--spec',sr['path'],'--spec-sha256',sr['sha256'],'--out',str(out),'--run'])
   status=audit(out,1);r=fixed(status['completed'][0]);trace=fixed(r['trace']);rows=json.loads((out/'qualification900/requests.json').read_text())
   require(len(rows)==trace['n_requests']==r['n_expected'] and all(x['success']==1 and not x['request_timeout'] and x['token_ids_verified']==1 and x['generated_tokens']==t['output_len'] for x,t in zip(rows,trace['requests'])),'independent900 full-work audit failed')
   raw=raw_measurement(r['raw_measurement']);recomputed=sum(x['ttft_s']<1 and x['tpot_s']<.1 for x in rows);require(recomputed==r['n_good'],'actual900 SLO differs')
   if arm=='dynamic':
    inv=json.loads((out/'inventory.json').read_text());commits=[e for e in inv['events'] if e['kind']=='physical_commit'];require({e['operation'] for e in commits}=={'restore','remove'},'actual automatic growth and measured return required')
   results[arm]=dict(result=status['completed'][0],status=ref(out/'status.json'),full_work=True,failed_requests=0,request_timeouts=0,slo_attainment=r['slo_attainment'],energy_j=r['energy_j'],raw_energy_recomputed=True)
  require(results['fixed2']['result']['path']!=results['dynamic']['result']['path'],'independent paired executions required')
  durable(OUT/'qualification.json',dict(passed=True,created_s=time.time(),source=declaration['source'],profile=declaration['profile'],certificate=declaration['certificate'],capacity_binding=declaration['capacity_binding'],trace=declaration['trace'],arms=results,initial_instances=2,actual_growth_and_return=True,formal_100s_performance_not_inferred=True,development_only=True))
  update(phase='complete',complete=True)
 except BaseException as exc:update(phase='stopped_failure',error=repr(exc));raise
 finally:update(finished_s=time.time())
if __name__=='__main__':main()
