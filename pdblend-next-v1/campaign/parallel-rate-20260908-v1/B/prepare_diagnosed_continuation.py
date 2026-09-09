"""Manual CPU/read-only reconciliation after a stopped queue; never launches GPU."""
from pathlib import Path
import argparse,sys,json,time,copy,re,subprocess
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import baseline_reconciliation_v2 as q,remote
parser=argparse.ArgumentParser();parser.add_argument('--previous',type=int,required=True);parser.add_argument('--index',type=int,required=True);a=parser.parse_args();prev=f'{a.previous:03d}';num=f'{a.index:03d}';stage=B/f'baselines-reconciled-{prev}';state=q.p.read(stage/'status.json');assert len(state['failed'])==1 and not state['complete'] and not state['node_lease_held'];cid=state['failed'][0]['cell_id'];cpref=q.p.ref(stage/'results/checkpoints'/(cid+'.json'));cp=q.p.checked(cpref);receipt=q.p.checked(cp['receipt']);raw=q.capacity_negative(cp,receipt)
fresh=B/f'reconciliation-{num}-fresh-node-audit.json'
payload=dict(out=str(fresh),owners=[state['pid'],receipt['child_pid']],system=cp['row']['system'])
code=r'''import sys,json,asyncio,time,subprocess
from pathlib import Path
B=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/B');sys.path.insert(0,str(B));import baseline_control_completion as q
x=json.loads(PAYLOAD);p=q.p;refs=p.read(B/'baseline-bindings-completion.json');bindings={k:p.checked(r) for k,r in refs.items()};common=q.fixed.runtime(bindings[x['system']]['host_release']);proofs={k:q.audit_performance(k,b) for k,b in bindings.items()}
async def check():
 import aiohttp
 async with aiohttp.ClientSession(trust_env=False) as session:return await common.identity(session,bindings[x['system']])
id=asyncio.run(check());owners={}
for pid in x['owners']:
 try:owners[str(pid)]=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()[0]
 except OSError:owners[str(pid)]=None
assert all(s in [None,'Z'] for s in owners.values())
evidence=dict(schema='B-capacity-negative-fresh-node-audit-v1',created_s=time.time(),passed=True,bindings=refs,qualification_reconstructed=proofs,identity_and_native_idle=id,previous_owners=owners,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,clocks.current.sm','--format=csv,noheader,nounits'],text=True),gpu_work_executed=False);p.write(x['out'],evidence,exclusive=True);print(json.dumps({'passed':True,'path':x['out'],'sha256':p.sha(x['out'])}))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
print(remote.run(code,timeout=90),flush=True)
subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=39.108.209.97','172.16.50.102:'+str(fresh),str(fresh)],check=True)
proof=dict(schema='B-frozen-baseline-capacity-negative-diagnosis-v1',created_s=time.time(),checkpoint=cpref,raw_audit=raw,fresh_node_audit=q.p.ref(fresh),authorization='After manual stopped-queue diagnosis, Root explicitly authorized the remaining originally predeclared independent measurements including repeat2; never retry a failed already observed cell. Any further failure stops again.')
diag=B/f'reconciliation-{num}-capacity-diagnosis.json';q.p.write(diag,proof,exclusive=True)
d=q.p.read(B/f'baseline-reconciliation-{prev}/declaration.json');d['created_s']=time.time();d['prior_stages'].append({'status':q.p.ref(stage/'status.json')});d['capacity_negatives'].append(dict(cell_id=cid,diagnosis=q.p.ref(diag)));d['remaining_cells']=[r for r in d['remaining_cells'] if r['cell_id'] not in state['attempted']];d['already_observed_count']+=len(state['attempted']);d['already_full_work_count']+=len(state['attempted'])-1
D=B/f'baseline-reconciliation-{num}';D.mkdir();q.p.write(D/'declaration.json',d,exclusive=True);q.audit(D/'declaration.json')
s=(B/f'baseline_runner_reconciled_{prev}.py').read_text().replace(f"baseline-reconciliation-{prev}/declaration.json",f"baseline-reconciliation-{num}/declaration.json");s=re.sub("RECON_SHA='[0-9a-f]+'","RECON_SHA="+repr(q.p.sha(D/'declaration.json')),s);runner=B/f'baseline_runner_reconciled_{num}.py';assert not runner.exists();runner.write_text(s);compile(s,str(runner),'exec')
for field,value in [('runtime_error','engineering'),('request_timeouts',raw['request_timeouts']-1),('energy_j',1),('drain_complete',False)]:
 bad=copy.deepcopy(receipt);bad['summary'][field]=value
 try:q.capacity_negative(cp,bad)
 except ValueError:pass
 else:raise AssertionError('negative audit accepted '+field)
paths=[runner,D/'declaration.json',diag,fresh,Path(__file__).resolve()];m=dict(created_s=time.time(),passed=True,cpu_cases=6,prior_13_cases_retained=True,source_files={str(p):q.p.sha(p) for p in paths},measurement_child_unchanged=True,failure_gate_unchanged=True,baseline_policy_unchanged=True,automatic_retry=False);q.p.write(D/'launch-manifest.json',m,exclusive=True);paths.append(D/'launch-manifest.json');print(remote.stage_files(paths),flush=True);print(remote.run('import sys;sys.path.insert(0,'+repr(str(B))+');import baseline_runner_reconciled_'+num+' as r;d,_,_=r.contract();print({"actual_host_cpu_contract":True,"remaining":len(d["cells"])})',timeout=60),flush=True);print(json.dumps({'new_declaration':q.p.ref(D/'declaration.json'),'remaining':len(d['remaining_cells']),'already_observed':d['already_observed_count'],'capacity_negative':{k:v for k,v in raw.items() if k!='failed_rows'}}))
