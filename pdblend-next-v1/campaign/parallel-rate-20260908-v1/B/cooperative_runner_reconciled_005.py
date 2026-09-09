"""Predeclared cooperative Dynamo8; preserve every outcome and stop on any failure."""
import argparse,asyncio,copy,csv,json,math,os,signal,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import cooperative_dynamo_qualify_v2 as qualification
c=qualification.c;p=c.p;DECL=qualification.DECL;RULES=B/'cooperative-dynamo8-001/execution-rules-v2.json'
RECON=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/B/cooperative-reconciliation-005/declaration.json')
RECON_SHA='14876ceec590ec84539dc8a3a65bc064b99eee5ca3cee889562a25a64663c5ce'
DECL_SHA='ac1ccee28d1a8f00a5c2e566dc358badf36c19cccf8df7e968c8d1568d62f431'

def timing_values(rows,summary,maximum=1.,p99_limit=.1):
 p.need(len(rows)==summary['n_expected']>0,'all actual request rows required for arrival timing')
 delays=[]
 for row in rows:
  planned=float(row['planned_arrival_s']);actual=float(row['actual_dispatch_s']);reported=float(row['dispatch_delay_s'])
  p.need(all(math.isfinite(x) for x in (planned,actual,reported)) and actual>=planned-1e-6,'actual dispatch missing/nonfinite/early')
  delay=max(0.,actual-planned);p.need(math.isclose(delay,reported,rel_tol=0,abs_tol=1e-6),'raw dispatch delay differs');delays.append(delay)
 delays.sort();position=.99*(len(delays)-1);lo=int(position);hi=min(lo+1,len(delays)-1);percentile=delays[lo]+(delays[hi]-delays[lo])*(position-lo);maximum_observed=delays[-1]
 p.need(math.isclose(summary['dispatch_delay_max_s'],maximum_observed,rel_tol=0,abs_tol=1e-6) and math.isclose(summary['dispatch_delay_p99_s'],percentile,rel_tol=0,abs_tol=1e-6),'reported dispatch distribution differs from actual raw')
 errors=[]
 if maximum_observed>maximum:errors.append('actual dispatch max exceeds predeclared1s')
 if percentile>p99_limit:errors.append('actual dispatch linear p99 exceeds predeclared0.1s')
 return dict(passed=not errors,errors=errors,n_requests=len(rows),actual_dispatch_max_s=maximum_observed,actual_dispatch_p99_s=percentile,max_limit_s=maximum,p99_limit_s=p99_limit,percentile_method='linear at(n-1)*0.99',engineering_qualification_only=True,violations_are_not_capacity_negative_evidence=True)

def timing_gate(receipt,output):
 path=output/'cells'/receipt['cell_id']/'bench.csv'
 try:result=timing_values(list(csv.DictReader(path.open())),receipt['summary'])
 except BaseException as exc:result=dict(passed=False,errors=[repr(exc)],engineering_qualification_only=True,violations_are_not_capacity_negative_evidence=True)
 if path.exists():result['raw_requests']=p.ref(path)
 result['execution_rules']=p.ref(RULES);return result

def contract():
 p.need(p.sha(DECL)==DECL_SHA,'original eight-row declaration changed');d,files=qualification.host_contract();rule=p.read(RULES)
 p.need(rule['schema']=='B-cooperative-Dynamo8-arrival-engineering-v1' and rule['declaration']==p.ref(DECL) and rule['host_manifest']==d['host_manifest'],'execution rules bind a different group/source')
 p.need((rule['actual_dispatch_max_limit_s'],rule['actual_dispatch_p99_limit_s'])==(1.,.1) and rule['arrival_window_s']==100 and rule['request_hard_timeout_s']==120 and rule['cleanup_local_budget_s']==90,'arrival or measurement limits changed')
 for f,h in rule['source_files'].items():p.need(p.sha(f)==h,'frozen cooperative execution source changed')
 p.need(len(d['cells'])==8 and len({r['cell_id'] for r in d['cells']})==8,'exact eight distinct points required')
 p.need(p.sha(RECON)==RECON_SHA,'reconciliation changed');import cooperative_reconciliation_v1 as reconcile
 checked=reconcile.audit(RECON);d['cells']=copy.deepcopy(checked['remaining_cells'])
 return d,rule

async def execute(a,state):
 d,rule=contract();previous=qualification.prerequisite();reference=p.ref(qualification.OUT/'dynamollm/binding.json');base=p.checked(reference);qualification.audit(reference)
 common=c.fixed.runtime(base['host_release'])
 from ecopadg.serving.campaign import node_lease
 from ecopadg.measure.backends import PynvmlBackend
 import aiohttp
 p.need('PDBLEND_NODE_LOCK_FD' not in os.environ and not a.out.exists(),'fresh cooperative queue/lease required');a.out.mkdir();output=a.out/'results';output.mkdir()
 p.write(a.out/'declaration-order.json',d['cells'],exclusive=True);p.write(a.out/'execution-rules-reference.json',p.ref(RULES),exclusive=True);p.write(a.out/'predecessor.json',previous,exclusive=True)
 state.update(declared=len(d['cells']),original_group_count=8,reconciliation=p.ref(RECON),remaining=[r['cell_id'] for r in d['cells']],declaration=p.ref(DECL),execution_rules=p.ref(RULES))
 def save():state.update(updated_s=time.time());p.write(a.out/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;save();hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  async with aiohttp.ClientSession(trust_env=False) as session:
   await common.identity(session,base)
   for row in d['cells']:
    if a.stop_requested or (B/'STOP-cooperative-dynamo8').exists():state.update(phase='stopped_at_boundary');break
    contract();binding=copy.deepcopy(base);binding.update(output=str(output),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',formal_eligible=False)
    binding['files'].update({str(RECON):RECON_SHA,str(B/'cooperative_reconciliation_v1.py'):p.sha(B/'cooperative_reconciliation_v1.py'),str(DECL):p.sha(DECL),str(RULES):p.sha(RULES),str(Path(__file__).resolve()):p.sha(__file__),row['trace_path']:row['trace_sha256']})
    bp=a.out/'bindings'/(row['cell_id']+'.json');p.write(bp,binding,exclusive=True);common.validate_binding(binding)
    state.update(phase='running',current_cell=row['cell_id']);state['attempted'].append(row['cell_id']);save();rp=output/'operations'/row['cell_id']/'receipt.json';cp=output/'checkpoints'/(row['cell_id']+'.json')
    try:
     receipt=await common.run_one(session,binding,row,output,hardware)
     arrival=timing_gate(receipt,output);artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
     p.write(cp,dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt['measurement_valid'],work_complete=receipt['summary'].get('work_complete'),arrival_fidelity_gate=arrival,execution_rules=p.ref(RULES),reconciliation=p.ref(RECON),execution_source=p.ref(Path(__file__).resolve()),completed_s=time.time()),exclusive=True)
     state['completed'].append(row['cell_id']);gate=c.fixed.engineering_gate(receipt);gate['arrival_fidelity_gate']=arrival;gate['passed']=gate['passed'] and arrival['passed'];gate['errors'].extend(arrival['errors']);p.write(a.out/'engineering-gates'/(row['cell_id']+'.json'),gate,exclusive=True)
     p.need(gate['passed'],'request/measurement/clock/native/arrival failure stops expansion '+str(gate['errors']))
    except BaseException as exc:
     if rp.exists() and not cp.exists():
      receipt=p.read(rp);artifacts={str(f):p.sha(f) for directory in (rp.parent,output/'cells'/row['cell_id']) for f in directory.rglob('*') if f.is_file()}
      p.write(cp,dict(row=row,declaration=p.ref(DECL),binding=p.ref(bp),receipt=p.ref(rp),artifacts=artifacts,measurement_valid=receipt.get('measurement_valid') is True,work_complete=receipt.get('summary',{}).get('work_complete'),arrival_fidelity_gate=timing_gate(receipt,output),execution_rules=p.ref(RULES),reconciliation=p.ref(RECON),execution_source=p.ref(Path(__file__).resolve()),execution_error=repr(exc),failed_observation_preserved=True,completed_s=time.time()),exclusive=True)
     state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc)));raise
    finally:state['remaining']=[r['cell_id'] for r in d['cells'] if r['cell_id'] not in state['attempted']];save()
  state['complete']=len(state['completed'])==len(d['cells'])
  if state['complete']:state['phase']='complete'
 state['node_lease_held']=False;save()

def main():
 q=argparse.ArgumentParser();q.add_argument('--out',type=Path,required=True);q.add_argument('--run',action='store_true');a=q.parse_args();a.stop_requested=False
 if not a.run:d,rule=contract();print(dict(cpu_only=True,declared=len(d['cells']),ready_for_GPU=False));return
 p.need(not a.out.exists(),'new attempt required')
 def stop(*_):a.stop_requested=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 state=dict(pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[],automatic_retries=False)
 try:asyncio.run(execute(a,state))
 except BaseException as exc:state.update(phase='failed',error=repr(exc));raise
 finally:
  if a.out.exists():state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);p.write(a.out/'status.json',state)
if __name__=='__main__':main()
