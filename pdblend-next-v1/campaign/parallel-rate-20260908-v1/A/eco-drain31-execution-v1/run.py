"""One unchanged fixed100 Eco GPU queue; preserve CP before every failure stop."""
import argparse,asyncio,copy,csv,json,math,os,signal,socket,sys,time
from pathlib import Path
from contract import *

def timing(bench,events):
 records=[e for e in events if e.get('kind')=='request_timing'];byid={r['client_request_id']:r for r in records}
 need(bench and len(byid)==len(records)==len(bench) and set(byid)=={r['request_id'] for r in bench},'complete unique raw timing IDs required');delays=[];handlers=[]
 for row in bench:
  start,actual,deadline=(float(row[k]) for k in ('planned_arrival_s','actual_dispatch_s','request_deadline_s'));event=byid[row['request_id']]
  need(all(math.isfinite(v) for v in (start,actual,deadline)) and -1e-5<=actual-start<=1 and abs(deadline-start-120)<1e-5 and row['open_loop_independent']=='True','original arrival/request budget changed')
  need((event['planned_arrival_s'],event['actual_dispatch_s'],event['hard_deadline_s'])==(start,actual,deadline),'client/controller timing differs');handler=event['handler_arrival_s']-actual;need(math.isfinite(handler) and 0<=handler<=1,'handler timing starved');delays.append(actual-start);handlers.append(handler)
 delays.sort();q=(len(delays)-1)*.99;k=int(q);p99=delays[k]+(delays[min(k+1,len(delays)-1)]-delays[k])*(q-k);need(p99<=.1,'actual dispatch p99 exceeds .1s')
 return dict(passed=True,actual_dispatch_lateness_max_s=max(delays),actual_dispatch_lateness_p99_s=p99,handler_lateness_max_s=max(handlers))

def gate(receipt):
 s=receipt.get('summary',{});errors=[]
 for good,why in [(receipt.get('measurement_valid') is True and s.get('measurement_valid') is True,'measurement invalid'),(s.get('work_complete') is True and s.get('failed_requests')==0 and s.get('request_timeouts')==0,'request failed or incomplete'),(receipt.get('child_stopped') is True and receipt.get('child_exitcode')==0 and receipt.get('clock_restore_complete') is True and not receipt.get('outer_cleanup_errors') and not receipt.get('sampling_error') and s.get('post_measurement_cleanup',{}).get('cleanup_complete') is True,'child/clock/cleanup failed'),(s.get('gpu_count')==8 and s.get('power_source_verified') is True and s.get('fixed_window_valid') is True,'all8 original window evidence invalid'),(s.get('runtime_error') is None,'runtime error')]:
  if not good:errors.append(why)
 return dict(passed=not errors,errors=errors,low_SLO_not_a_failure=True)

def validate(path):
 release=read(path);need(release['schema']=='A-Eco-scoped-qualified-execution-release-v1' and release['ready_for_gpu'] is True,'actual scoped/fresh release required')
 for file,h in release['files'].items():need(sha(file)==h,'frozen execution input changed '+file)
 need(release['common_executor']==dict(path=str(COMMON),sha256=COMMON_SHA) and release['host_manifest']==dict(path=str(HOST/'manifest.json'),sha256=HOST_SHA),'source/protocol changed')
 scope=checked(release['declaration']);base=checked(release['binding']);need(release['required_count']==len(scope['required_cells']),'declared count differs')
 need(base['hostname']==release['hostname']==socket.gethostname(),'A original physical node required')
 return release,scope,base

async def execute(args,state):
 release,scope,base=validate(args.release);validate_scope(release['declaration'],release['qualification_binding'])
 sys.path[:0]=[str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps'];common=load(COMMON,'a_ecodrain_fixed100_common')
 from ecopadg.serving.campaign import node_lease
 from ecopadg.measure.backends import PynvmlBackend
 import aiohttp
 need(not args.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ,'new queue and exclusive original node lease required');args.out.mkdir();output=args.out/'results';output.mkdir()
 rows=scope['required_cells'];write(args.out/'declaration-order.json',rows);state.update(model='14b',stage='baseline_fixed_slo_eco',declaration=release['declaration'],release=ref(args.release),declared=len(rows),remaining=[r['cell_id'] for r in rows])
 def save():state['updated_s']=time.time();common.write(args.out/'status.json',state)
 save()
 with node_lease():
  state['node_lease_held']=True;save();hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
  async with aiohttp.ClientSession(trust_env=False) as session:
   await common.identity(session,base)
   for row in rows:
    if args.stop_requested:state['phase']='stopped_at_boundary';break
    validate(args.release);binding=copy.deepcopy(base);binding['output']=str(output);binding['files'].update({str(args.release):sha(args.release),release['declaration']['path']:release['declaration']['sha256']});bp=args.out/'bindings'/(row['cell_id']+'.json');write(bp,binding);common.validate_binding(binding)
    state['attempted'].append(row['cell_id']);state.update(phase='running',current_cell=row['cell_id']);save();rp=output/'operations'/row['cell_id']/'receipt.json';cp=output/'checkpoints'/(row['cell_id']+'.json')
    def checkpoint():
     receipt=read(rp);cell=output/'cells'/row['cell_id'];artifacts={str(p):sha(p) for directory in (rp.parent,cell) for p in directory.rglob('*') if p.is_file()}
     try:arrival=timing(list(csv.DictReader((cell/'bench.csv').open())),[json.loads(s) for s in (cell/'control.jsonl').read_text().splitlines()])
     except BaseException as exc:arrival=dict(passed=False,error=repr(exc))
     arrival['rule']=release['execution_rules'];record=dict(row=row,declaration=release['declaration'],logical_declaration=release['logical_declaration'],binding=ref(bp),receipt=ref(rp),artifacts=artifacts,measurement_valid=receipt.get('measurement_valid') is True,work_complete=receipt.get('summary',{}).get('work_complete'),arrival_timing_qualification=arrival,execution_rules=release['execution_rules'],completed_s=time.time());write(cp,record);return receipt,arrival
    try:
     receipt=await common.run_one(session,binding,row,output,hardware);receipt,arrival=checkpoint();state['completed'].append(row['cell_id']);engineering=gate(receipt);engineering['arrival']=arrival;write(args.out/'engineering-gates'/(row['cell_id']+'.json'),engineering);need(engineering['passed'] and arrival['passed'],'any request/measurement/arrival failure stops after CP and cleanup')
    except BaseException as exc:
     if rp.exists() and not cp.exists():checkpoint()
     state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc)));raise
    finally:state['remaining']=[r['cell_id'] for r in rows if r['cell_id'] not in state['attempted']];save()
   common.write(args.out/'identity.after.json',await common.identity(session,base))
  state['complete']=len(state['completed'])==len(rows);state['phase']='complete' if state['complete'] else state['phase']
 state['node_lease_held']=False;save()

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args();args.stop_requested=False
 if not args.run:release,scope,base=validate(args.release);print(json.dumps(dict(cpu_only=True,points=len(scope['required_cells']),no_gpu=True)));raise SystemExit(0)
 def stop(*_):args.stop_requested=True
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 state=dict(pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[],node_lease_held=False,automatic_retries=False)
 try:asyncio.run(execute(args,state))
 except BaseException as exc:state.update(phase='needs_attention',error=repr(exc),complete=False);raise
 finally:
  if args.out.exists():state.update(finished_s=time.time(),node_lease_held=False);state.pop('current_cell',None);path=args.out/'status.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
