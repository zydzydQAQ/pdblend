"""Freeze complete original non-Dynamo ledger and next fixed-SLO native gate prerequisite."""
import argparse,asyncio,copy,json,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import baseline_reconciliation_v3 as ledger
import cooperative_dynamo_qualify_v2 as qualification
p=ledger.p

def prepare(previous):
 stage=B/f'baselines-reconciled-{previous:03d}'
 state=p.read(stage/'status.json');d=ledger.audit(B/f'baseline-reconciliation-{previous:03d}/declaration.json')
 p.need(state['complete'] and not state['failed'] and not state['remaining'] and not state['node_lease_held'] and state['finished_s'] and not ledger.alive(state['pid']),'last original queue must complete and its owner exit')
 p.need(state['attempted']==state['completed']==[r['cell_id'] for r in d['remaining_cells']],'last queue must observe every exact remaining original row')
 for cid in state['completed']:
  cp=p.read(stage/'results/checkpoints'/(cid+'.json'));p.need(ledger.fixed.engineering_gate(p.checked(cp['receipt']))['passed'],'final queue cannot conceal any failed observation')
 d=copy.deepcopy(d);d.update(created_s=time.time(),remaining_cells=[],already_observed_count=d['already_observed_count']+len(state['completed']),already_full_work_count=d['already_full_work_count']+len(state['completed']),historical_suffix_pending=0,historical_suffix_excluded_from_this_task=11,current_scope=p.ref(qualification.SCOPE))
 d['prior_stages'].append(dict(status=p.ref(stage/'status.json')))
 path=B/'baseline-reconciliation-final-fixedslo-v1/declaration.json';p.write(path,d,exclusive=True);ledger.audit(path)
 p.need(d['already_observed_count']==21 and len(d['retired_unexecuted_cells'])==3,'original18 nonDyn plus3 archived Dyn and3 retired rows exact scope required')
 c=qualification.c;c.modules();refs=p.read(B/'baseline-bindings-after-external-001.json');bindings={k:p.checked(v) for k,v in refs.items()};common=c.fixed.runtime(bindings['dynamollm']['host_release'])
 audits={k:c.audit_performance(k,v) for k,v in bindings.items()}
 async def identity():
  import aiohttp
  async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,bindings['dynamollm'])
 actual=asyncio.run(identity())
 fresh=B/'baseline-reconciliation-final-fixedslo-v1/fresh-node-audit.json';p.write(fresh,dict(passed=True,created_s=time.time(),actual_identity_and_native_idle=actual,bindings=refs,qualification_reconstructed=audits,last_owner_pid=state['pid'],last_owner_exited=True,gpu_work_executed=False),exclusive=True)
 prereq=dict(schema='B-cooperative-after-original-nonDyn-fixedSLO-v2',created_s=time.time(),completed_non_dynamo_ledger=p.ref(path),current_scope=p.ref(qualification.SCOPE),fresh_node_audit=p.ref(fresh),source=p.ref(Path(__file__).resolve()),original_eight_declaration=p.ref(qualification.DECL),historical_scale11_excluded=True,cooperative_native_gate_still_required=True)
 pr=B/'cooperative-dynamo8-001/qualification-prerequisite.json';p.write(pr,prereq,exclusive=True);qualification.prerequisite();print(json.dumps(dict(prerequisite=p.ref(pr),ledger=p.ref(path),fresh_node=p.ref(fresh),ready_for_native_qualification=True)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--previous',type=int,required=True);args=parser.parse_args();prepare(args.previous)
