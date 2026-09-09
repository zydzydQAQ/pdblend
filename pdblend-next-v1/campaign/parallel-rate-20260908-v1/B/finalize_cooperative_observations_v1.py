"""Final raw eight-observation ledger, preserving capacity failures separately from completion."""
import argparse,asyncio,copy,json,subprocess,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import cooperative_reconciliation_v1 as q
p=q.p

def finalize(stage=None,completed_ledger=None):
 if completed_ledger is not None:
  d=copy.deepcopy(q.audit(completed_ledger));p.need(d['remaining_cells']==[] and d['already_observed_count']==8,'final failed point needs a completed independently diagnosed ledger')
 else:
  root=B/stage;state=p.read(root/'status.json')
  p.need(Path(stage).name==stage and stage.startswith('cooperative-dynamo8-'),'exact cooperative stage required')
  p.need(state['complete'] and not state['failed'] and not state['remaining'] and not state['node_lease_held'] and state['finished_s'] and not q.legacy.alive(state['pid']),'last queue must naturally complete its remaining declared work')
  d=copy.deepcopy(p.checked(state['reconciliation']));q.audit(state['reconciliation']['path'])
  p.need(state['attempted']==state['completed']==[r['cell_id'] for r in d['remaining_cells']],'last remaining rows differ or were omitted')
  d.update(created_s=time.time(),already_observed_count=d['already_observed_count']+len(state['completed']),remaining_cells=[]);d['prior_stages'].append(dict(status=p.ref(root/'status.json')))
 out=B/'cooperative-final-observations-001';p.need(not out.exists(),'new immutable final observation archive required')
 p.write(out/'declaration.json',d,exclusive=True);q.audit(out/'declaration.json');p.need(d['already_observed_count']==8,'all exact eight observations required')
 rows=[]
 for entry in d['prior_stages']:
  status=p.checked(entry['status']);parent=Path(entry['status']['path']).parent
  for cid in status['completed']:
   cpref=p.ref(parent/'results/checkpoints'/(cid+'.json'));cp=p.checked(cpref);receipt=p.checked(cp['receipt']);summary=receipt['summary'];cell=parent/'results/cells'/cid
   energy=q.legacy.integral(cell/'power.csv',summary['measurement_start_s'],summary['measurement_end_s'])
   p.need(abs(energy-summary['energy_j'])<=max(1e-7,abs(energy)*1e-9),'actual final eight-GPU raw integral differs')
   rows.append(dict(cell_id=cid,checkpoint=cpref,work_complete=summary['work_complete'],failed_requests=summary['failed_requests'],request_timeouts=summary['request_timeouts'],complete_requests=summary['completed_work_requests'],n_expected=summary['n_expected'],slo_attainment=summary['slo_attainment'],energy_j=energy,arrival_fidelity_gate=cp['arrival_fidelity_gate']))
 qualification=q.original.qualification;bp=p.ref(qualification.OUT/'dynamollm/binding.json');proof=qualification.audit(bp);base=p.checked(bp);common=qualification.c.fixed.runtime(base['host_release'])
 async def identity():
  import aiohttp
  async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,base)
 actual=asyncio.run(identity());hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,clocks.current.sm','--format=csv,noheader,nounits'],text=True)
 result=dict(schema='B-fixed-SLO-cooperative-eight-final-observations-v1',created_s=time.time(),all_eight_observed=True,observed_count=8,all_request_work_complete=all(row['work_complete'] for row in rows),complete_work_observations=sum(row['work_complete'] for row in rows),capacity_negative_observations=sum(not row['work_complete'] for row in rows),remaining_count=0,original_declaration=p.ref(q.original.DECL),reconciliation=p.ref(out/'declaration.json'),current_scope=p.ref(qualification.SCOPE),execution_rules=p.ref(q.original.RULES),final_fresh_qualification=proof,final_actual_identity_and_native_idle=actual,hardware=hardware,final_owners_exited=True,node_lease_held=False,rows=rows,historical_scale11_excluded=True,failed_partial_work_and_all8_energy_preserved=True,no_capacity_negative_relabelled_as_fullwork=True,source=p.ref(Path(__file__).resolve()))
 p.write(out/'completion-audit.json',result,exclusive=True);print(json.dumps(dict(completion=p.ref(out/'completion-audit.json'),observed=8,complete_work=result['complete_work_observations'],capacity_negative=result['capacity_negative_observations'],remaining=0)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--stage');group.add_argument('--completed-ledger',type=Path);args=parser.parse_args();finalize(args.stage,args.completed_ledger)
