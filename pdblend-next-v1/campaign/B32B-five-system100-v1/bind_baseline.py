"""Read-only live binding for original B baselines; performance requires terminal mechanism proof."""
import argparse,asyncio,copy,hashlib,importlib.util,json,pathlib,socket,sys
R=pathlib.Path('/root/workspace/pdblend-next-v1');P=R/'campaign/B32B-five-system100-v1'
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
SYSTEMS={'mixed':'mixed','distserve':'distserve','ecoserve':'ecoserve','dynamollm-resident':'dynamollm'}
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def write(p,v):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def mechanism_ready(gate,checks,system):
    require(gate.get('complete') is True and gate.get('measurement_valid') is True and gate.get('native_cleanup_complete') is True and gate.get('clock_restore_complete') is True and not gate.get('cleanup_errors') and not gate.get('sampling_error'),'terminal clean measured gate required')
    require(checks.get('complete') is True and checks.get('cleanup',{}).get('complete') is True,'raw native cleanup required')
    c=checks['checks'];temporal=gate.get('temporal_owner_proof',{})
    proof=bool(temporal) and all(p.get('prefill_steps') and p.get('decode_steps') and p.get('mixed_steps')==0 for p in temporal.values())
    actual=dict(ordinary=bool(c.get('ordinary_cross_replica_exact')),pd=bool(c.get('pd_exact_all_declared_pairs') and c.get('cancel_all_tp_ranks')),temporal=bool(c.get('temporal_exact') and proof))
    require(gate.get('mechanism_gate')==actual,'terminal mechanism headers differ from raw completed checks')
    needed={'mixed':('ordinary',),'dynamollm-resident':('ordinary',),'distserve':('ordinary','pd'),'ecoserve':('ordinary','temporal')}[system]
    require(all(actual[k] for k in needed),'required original baseline mechanism has not passed: '+str(needed))
    return dict(required=list(needed),verified=actual,overall_runtime_gate_passed=gate.get('passed'),original_failure_preserved=True)

async def build(a):
    spec=read(a.spec);require(spec['model']=='32b' and socket.gethostname()==spec['hostname'],'B physical scope differs')
    executor=load('b_baseline_executor',pathlib.Path(spec['executor_release'])/'run.py');executor.validate_binding(read(spec['pdb_binding']))
    for p,h in spec['files'].items():require(sha(p)==h,'frozen deployment input changed: '+p)
    receipt=read(a.receipt);require(receipt.get('complete') is True and receipt.get('measurement_valid') is True,'successful measured deployment required')
    require(not a.out.exists(),'new binding directory required')
    created={r['name']:r['container_id'] for r in receipt['created']};names=(await executor.command('docker','ps','--format','{{.Names}}')).split();require(set(names)==set(created),'unexpected running engines')
    inventory=json.loads(await executor.command('docker','inspect',*names));by_name={c['Name'].lstrip('/'):c for c in inventory};instances=[]
    source=read(spec['pdb_binding']);files=dict(source['files']);files.update(spec['files']);files[str(a.spec.resolve())]=sha(a.spec);files[str(a.receipt.resolve())]=sha(a.receipt);files[str(pathlib.Path(__file__).resolve())]=sha(__file__)
    import aiohttp
    async with aiohttp.ClientSession(trust_env=False) as session:
        for i in spec['instances']:
            c=by_name[i['container_name']];cfg=read(i['config'])
            require(c['Id']==created[i['container_name']] and c['Image']==IMAGE and c['State']['Running'],'new engine process/image changed')
            require(c['Config']['Cmd']==['python3',i['engine_entry'],'--config',i['config']],'actual engine entry/config differs')
            require(c['Config']['Env']==i['environment'] or set(c['Config']['Env'])==set(i['environment']),'actual original environment differs')
            p=await executor.http(session,i,'/provenance');r=await executor.wait_idle(session,i)
            require(p.get('instance_id')==i['id'] and p.get('tp')==2 and p.get('model')==cfg['model']=='/models/Qwen2.5-32B-Instruct' and p.get('dtype')=='bfloat16' and p.get('max_model_len')==8192 and p.get('cuda_visible_devices')==','.join(map(str,i['gpus'])),'real model/TP/static limit/GPU differs')
            imported={str(f):sha(f) for f in pathlib.Path(i['engine_entry']).parent.glob('*.py')};require(p.get('source_files_at_import')==imported and all(files.get(f)==h for f,h in imported.items()),'actual imported source not frozen')
            require(r.get('accepting') is True and r.get('generation')==r.get('acknowledged_generation'),'actual idle owner ACK missing')
            record={k:i[k] for k in ('id','tp','gpus','role','port','kv_port','url','native_kind','scheduler_cache_observed')}
            record.update(engine_config=i['config'],container=dict(name=i['container_name'],id=c['Id'],image=c['Image'],StartedAt=c['State']['StartedAt']),provenance={k:p[k] for k in ('instance_id','pid','tp','model','dtype','max_model_len','cuda_visible_devices','source_files_at_import')});instances.append(record)
    a.out.mkdir(parents=True);write(a.out/'identity.json',inventory);files[str(a.out.resolve()/'identity.json')]=sha(a.out/'identity.json')
    configs={};mechanism=None;canonical='mixed'
    if a.strategy:
        require(a.gate is not None,'performance bindings require a terminal gate')
        gate=read(a.gate/'status.json');checks=read(a.gate/'checks/checks.json');mechanism=mechanism_ready(gate,checks,a.strategy);canonical=SYSTEMS[a.strategy]
        # Bind the actual same resident processes that completed the gate.
        for name in ('identity.before.json','identity.after.json'):
            seen=read(a.gate/name);require(len(seen)==4,'gate resident count differs')
            by_id={r['provenance']['instance_id']:r for r in seen}
            for i in instances:
                r=by_id[i['id']];require(r['container']['Id']==i['container']['id'] and r['container']['Image']==i['container']['image'] and r['container']['State']['StartedAt']==i['container']['StartedAt'] and all(r['provenance'].get(k)==v for k,v in i['provenance'].items()),'gate ran on different source/model/process')
        for f in a.gate.rglob('*'):
            if f.is_file():files[str(f.resolve())]=sha(f)
        roles=None
        for dataset in ('alpaca','sharegpt','longbench'):
            original=P/f'configs/{dataset}.{a.strategy}.json';cfg=read(original);require(files[str(original)]==sha(original),'prepared historical config changed')
            mapping={i['id']:i for i in instances};require(set(mapping)=={i['id'] for i in cfg['instances']},'historical routes changed')
            for ci in cfg['instances']:
                i=mapping[ci['id']];require(all(ci[k]==i[k] for k in ('tp','gpus','url','port','kv_port')),'prepared endpoint/GPU route differs')
            this_roles={i['id']:i['role'] for i in cfg['instances']};require(roles is None or roles==this_roles,'dataset-specific physical roles require separate binding');roles=this_roles
            cfg.update(controller_source_release=spec['host_release'],comparison_system=canonical)
            q=a.out/'configs'/f'{dataset}.json';write(q,cfg);files[str(q.resolve())]=sha(q);configs[dataset]=str(q.resolve())
        for i in instances:i['role']=roles[i['id']]
    # Empty configurations make the bootstrap object unusable as a serving workload.
    binding=dict(schema=1,protocol_id=spec['protocol_id'],model='32b',system=canonical,implementation_variant=a.strategy or 'correctness-only',hostname=spec['hostname'],deadline_s=spec['deadline_s'],host_release=spec['host_release'],output=str((a.out/'results').resolve()),configs=configs,instances=instances,files=files,large_inputs=source.get('large_inputs',{}),window_s=100,seeds=[701],deployment_receipt=str(a.receipt.resolve()),correctness_gate_required_before_performance=not bool(a.strategy),output_correctness_verified=bool(a.strategy),mechanism_proof=mechanism,formal_eligible=False)
    if a.gate:binding['correctness_evidence']=str(a.gate.resolve())
    write(a.out/'binding.json',binding);executor.validate_binding(binding);print(json.dumps(dict(binding=str(a.out/'binding.json'),sha256=sha(a.out/'binding.json'),system=canonical,mode='performance' if a.strategy else 'correctness-only',hardware_actions=False)))
    return binding

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('spec','receipt','out'):p.add_argument('--'+n,type=pathlib.Path,required=True)
    p.add_argument('--strategy',choices=SYSTEMS);p.add_argument('--gate',type=pathlib.Path);a=p.parse_args();s=read(a.spec);h=pathlib.Path(s['host_release']);sys.path[:0]=[str(h/'src'),str(h),'/root/workspace/pdblend/.runtime-deps'];asyncio.run(build(a))
if __name__=='__main__':main()
