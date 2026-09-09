"""Manual diagnosed continuation; raw arrival engineering must pass before capacity classification."""
from pathlib import Path
import argparse,copy,json,sys,time,subprocess
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
import cooperative_reconciliation_v1 as q,remote
p=q.p
parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);parser.add_argument('--index',type=int,required=True);a=parser.parse_args()
p.need(a.stage.startswith('cooperative-dynamo8-') and Path(a.stage).name==a.stage,'exact B cooperative stage required')
root=B/a.stage;s=p.read(root/'status.json');p.need(not s['complete'] and len(s['failed'])==1 and not s['node_lease_held'] and s['finished_s'],'one naturally stopped cooperative point required')
cid=s['failed'][0]['cell_id'];cpref=p.ref(root/'results/checkpoints'/(cid+'.json'));cp=p.checked(cpref);receipt=p.checked(cp['receipt']);arrival=q.original.timing_gate(receipt,root/'results');p.need(arrival['passed'] and arrival==cp['arrival_fidelity_gate'],'arrival engineering failure cannot be called capacity negative');raw=q.legacy.capacity_negative(cp,receipt)
num=f'{a.index:03d}';out=B/f'cooperative-reconciliation-{num}';p.need(not out.exists(),'fresh immutable reconciliation required')
payload=dict(path=str(out/'fresh-node-audit.json'),owners=[s['pid'],receipt['child_pid']])
code='''import sys,json,time,asyncio
from pathlib import Path
B=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/B');sys.path.insert(0,str(B));import cooperative_dynamo_qualify_v2 as q
x=json.loads(PAYLOAD);reference=q.p.ref(q.OUT/'dynamollm/binding.json');qualified=q.audit(reference);b=q.p.checked(reference);common=q.c.fixed.runtime(b['host_release'])
async def identity():
 import aiohttp
 async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,b)
actual=asyncio.run(identity());owners={}
for pid in x['owners']:
 try:owners[str(pid)]=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]
 except OSError:owners[str(pid)]=None
q.p.need(all(v in (None,'Z') for v in owners.values()),'previous cooperative owner not exited')
v=dict(passed=True,created_s=time.time(),cooperative_binding=reference,fresh_qualification_reconstructed=qualified,actual_identity_and_native_idle=actual,previous_owners=owners,gpu_work_executed=False);q.p.write(x['path'],v,exclusive=True);print(json.dumps(q.p.ref(x['path'])))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
print(remote.run(code,timeout=90),flush=True)
subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=39.108.209.97','172.16.50.102:'+str(out),str(B)+'/'],check=True)
proof=dict(schema='B-cooperative-original120s-capacity-negative-v1',created_s=time.time(),checkpoint=cpref,raw_audit=raw,arrival_fidelity_gate=arrival,fresh_node_audit=p.ref(out/'fresh-node-audit.json'),repeat_semantics='Only remaining originally declared independent rows; never repeat an observed failed ID. Original negative raw work, time and energy preserved.')
p.write(out/'capacity-diagnosis.json',proof,exclusive=True)
if 'reconciliation' in s:
 d=p.checked(s['reconciliation']);d=copy.deepcopy(d)
else:
 d=dict(schema='B-cooperative-eight-capacity-reconciliation-v1',declaration=p.ref(q.original.DECL),execution_rules=p.ref(q.original.RULES),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',automatic_retries=False,original_group_count=8,already_observed_count=0,prior_stages=[],capacity_negatives=[],remaining_cells=p.read(q.original.DECL)['cells'],source_files={str(B/'cooperative_reconciliation_v1.py'):p.sha(B/'cooperative_reconciliation_v1.py')})
d.update(created_s=time.time(),already_observed_count=d['already_observed_count']+len(s['attempted']))
d['prior_stages'].append(dict(status=p.ref(root/'status.json')));d['capacity_negatives'].append(dict(cell_id=cid,diagnosis=p.ref(out/'capacity-diagnosis.json')));d['remaining_cells']=[row for row in d['remaining_cells'] if row['cell_id'] not in s['attempted']]
p.write(out/'declaration.json',d,exclusive=True);q.audit(out/'declaration.json')
if not d['remaining_cells']:
 paths=[B/'cooperative_reconciliation_v1.py',Path(__file__).resolve(),out/'fresh-node-audit.json',out/'capacity-diagnosis.json',out/'declaration.json'];manifest=dict(created_s=time.time(),source_files={str(x):p.sha(x) for x in paths},original_eight_declaration=p.ref(q.original.DECL),original_arrival_engineering_rules=p.ref(q.original.RULES),all_eight_observations_preserved=True,all_request_work_complete=False,complete_group_is_not_all_work_success=True,already_observed=d['already_observed_count'],remaining=0,gpu_executed=False,no_empty_successor_queue=True)
 p.write(out/'manifest.json',manifest,exclusive=True);print(remote.stage_files(paths+[out/'manifest.json']),flush=True)
 print(json.dumps(dict(declaration=p.ref(out/'declaration.json'),already_observed=d['already_observed_count'],remaining=0,all_eight_observed=True,no_successor_GPU_required=True,raw={k:v for k,v in raw.items() if k!='failed_rows'})));sys.exit(0)
source=(B/'cooperative_dynamo_runner_v2.py').read_text()
source=source.replace("DECL_SHA=", "RECON=Path("+repr(str(out/'declaration.json'))+")\nRECON_SHA="+repr(p.sha(out/'declaration.json'))+"\nDECL_SHA=",1)
source=source.replace(' return d,rule\n'," p.need(p.sha(RECON)==RECON_SHA,'reconciliation changed');import cooperative_reconciliation_v1 as reconcile\n checked=reconcile.audit(RECON);d['cells']=copy.deepcopy(checked['remaining_cells'])\n return d,rule\n",1)
source=source.replace('state.update(declared=8,remaining=',"state.update(declared=len(d['cells']),original_group_count=8,reconciliation=p.ref(RECON),remaining=")
source=source.replace("binding['files'].update({str(DECL)","binding['files'].update({str(RECON):RECON_SHA,str(B/'cooperative_reconciliation_v1.py'):p.sha(B/'cooperative_reconciliation_v1.py'),str(DECL)")
source=source.replace("execution_rules=p.ref(RULES),completed_s=", "execution_rules=p.ref(RULES),reconciliation=p.ref(RECON),execution_source=p.ref(Path(__file__).resolve()),completed_s=")
source=source.replace("execution_rules=p.ref(RULES),execution_error=", "execution_rules=p.ref(RULES),reconciliation=p.ref(RECON),execution_source=p.ref(Path(__file__).resolve()),execution_error=")
source=source.replace("state['complete']=len(state['completed'])==8", "state['complete']=len(state['completed'])==len(d['cells'])")
runner=B/f'cooperative_runner_reconciled_{num}.py';runner.open('x').write(source);compile(source,str(runner),'exec')
paths=[runner,B/'cooperative_reconciliation_v1.py',Path(__file__).resolve(),out/'fresh-node-audit.json',out/'capacity-diagnosis.json',out/'declaration.json'];manifest=dict(created_s=time.time(),source_files={str(x):p.sha(x) for x in paths},derived_from=p.ref(B/'cooperative_dynamo_runner_v2.py'),original_eight_declaration=p.ref(q.original.DECL),original_arrival_engineering_rules=p.ref(q.original.RULES),measurement_child_and_failure_gate_unchanged=True,already_observed=d['already_observed_count'],remaining=len(d['remaining_cells']),gpu_executed=False)
p.write(out/'manifest.json',manifest,exclusive=True);paths.append(out/'manifest.json');print(remote.stage_files(paths),flush=True)
print(remote.run('import sys;sys.path.insert(0,'+repr(str(B))+');import cooperative_runner_reconciled_'+num+' as r;d,_=r.contract();print(dict(actual_CPU_passed=True,remaining=len(d["cells"])))',timeout=90),flush=True)
print(json.dumps(dict(declaration=p.ref(out/'declaration.json'),remaining=len(d['remaining_cells']),raw={k:v for k,v in raw.items() if k!='failed_rows'})))
