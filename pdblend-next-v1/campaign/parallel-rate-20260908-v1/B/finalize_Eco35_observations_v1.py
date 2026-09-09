"""Seal exact Eco35 observations after natural owner exit and read-only native identity."""
import argparse,asyncio,json,os,socket,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import eco_drained_runner_v2 as runner
p=runner.p

def alive(pid):
 try:return Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
 except OSError:return False

def ready():
 state=p.read(B/'eco-drain37-v1/performance/status.json');review=p.read(B/'eco-drain37-v1/independent-tail-audit/status.json')
 p.need(not state.get('failed') and not review.get('requires_review'),'failure remains for explicit diagnosis; finalizer performs no retry')
 return state,review,bool(state.get('complete') and state.get('node_lease_held') is False and not alive(state['pid']) and review.get('complete') and review.get('passed') and not alive(review['pid']))

def finalize():
 state,review,ok=ready();p.need(ok,'GPU queue and independent reviewer must finish and exit')
 declaration,rules=runner.contract();rows=declaration['cells'];ids=[r['cell_id'] for r in rows]
 p.need(state['attempted']==state['completed']==ids and not state['remaining'] and len(ids)==len(set(ids))==35,'all exact declared35 must have one observation')
 by_id={e['cell_id']:e for e in review['points']};p.need(set(by_id)==set(ids) and len(review['points'])==35 and review['verified_count']==35,'independent raw audit omitted or repeated observations')
 all37=declaration['declared_cells'];excluded=declaration['excluded_cells'];p.need(len(all37)==37 and len(excluded)==2,'original37/2conditional exclusion scope changed')
 for item in excluded:p.need(not (B/'eco-drain37-v1/performance/results/checkpoints'/(item['row']['cell_id']+'.json')).exists(),'excluded above-boundary point unexpectedly measured')
 observed=[]
 for row in rows:
  entry=by_id[row['cell_id']];cp=p.checked(entry['checkpoint']);receipt=p.checked(cp['receipt']);summary=receipt['summary'];point=entry['point']
  p.need(entry['passed'] and point['measurement_valid'] and cp['row']==row and cp['declaration']==p.ref(runner.DECL),'raw audited row or declaration differs')
  for path,h in cp['artifacts'].items():p.need(p.sha(path)==h,'final raw changed '+path)
  p.need(summary['work_complete'] and summary['failed_requests']==summary['request_timeouts']==0 and point['energy_j']==summary['energy_j'] and point['slo_attainment']==summary['slo_attainment'],'final point metrics or full work changed')
  observed.append(dict(cell_id=row['cell_id'],checkpoint=entry['checkpoint'],raw_audit_source=p.ref(B/'eco-drain37-v1/independent-tail-audit/status.json'),n_expected=summary['n_expected'],complete_requests=summary['completed_work_requests'],work_complete=True,failed_requests=0,request_timeouts=0,slo_attainment=summary['slo_attainment'],energy_j=summary['energy_j'],arrival_fidelity_gate=cp['arrival_fidelity_gate'],source_sha256=point['controller_source_sha256'],profile_sha256=point['profile_sha256'],policy_sha256=point['policy_sha256']))
 reference=p.ref(runner.qualification.OUT/'ecoserve/binding.json');proof=runner.qualification.audit(reference);base=p.checked(reference);common=runner.c.fixed.runtime(base['host_release'])
 async def actual_identity():
  import aiohttp
  async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,base)
 actual=asyncio.run(actual_identity());hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,clocks.current.sm','--format=csv,noheader,nounits'],text=True)
 p.need(len(hardware.strip().splitlines())==8,'final actual all8 hardware inventory missing')
 result=dict(schema='B-fixed-SLO-Eco37-declared35-executed-final-observations-v1',created_s=time.time(),physical_host=socket.gethostname(),pid=os.getpid(),source=p.ref(Path(__file__)),declaration=p.ref(runner.DECL),execution_rules=p.ref(runner.RULES),performance_terminal=p.ref(B/'eco-drain37-v1/performance/status.json'),independent_raw_audit=p.ref(B/'eco-drain37-v1/independent-tail-audit/status.json'),declared_count=37,required_execution_count=35,observed_count=35,all_required_observations_complete=True,all_request_work_complete=True,remaining_required_count=0,excluded_above_first_PDB_full_SLO_loss=excluded,old_original_records_unchanged=True,legacy_temporal_exact_mismatch_retained=True,qualified_numerical_scope='registered native-default trajectory at the same execution shape',final_fresh_qualification=proof,final_actual_identity_and_native_idle=actual,hardware=hardware,final_GPU_and_raw_audit_owners_exited=True,node_lease_held=False,rows=observed,historical_scale11_outside_current_scope=True,low_SLO_negative_observations_retained=True)
 out=B/'eco-drain37-v1/final-observations';p.need(not out.exists(),'new immutable final archive required');p.write(out/'completion-audit.json',result,exclusive=True);print(json.dumps(dict(completion=p.ref(out/'completion-audit.json'),observed=35,excluded_above_boundary=2,remaining_required=0)),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
 if args.wait:
  print(json.dumps(dict(waiter_pid=os.getpid(),physical_host=socket.gethostname(),read_only=True,node_lease_held=False)),flush=True)
  while not ready()[2]:time.sleep(15)
 finalize()
