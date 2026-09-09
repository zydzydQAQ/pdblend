"""Identity-only legacy bootstrap, then mechanism-gated historical A/C policies."""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from gate_evidence import audit,files,read,require,sha

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
DEPLOY=WORKSPACE/'campaign/AC-baseline-deployment-v1'
POLICY=WORKSPACE/'campaign/AC-baseline-100s-preparation-v1'
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
SYSTEMS=('mixed','distserve','ecoserve','dynamollm','dynamollm-resident')
DATASETS=('alpaca','sharegpt','longbench')

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(values):
    for p,h in values.items():require(sha(p)==h,'frozen input changed: '+p)
def package_files(root):
    manifest=root/'manifest.json';m=read(manifest);result={str(root/p):h for p,h in m['files'].items()}
    result[str(manifest)]=sha(manifest);verify(result);return result

def selected(spec,strategy,datasets):
    require(spec['model'] in ('14b','7b') and spec['layout'] in ('resident','distserve-longbench'),'unknown A/C physical scope')
    if strategy is None:
        require(not datasets,'correctness-only bootstrap does not select performance datasets');return []
    require(strategy in SYSTEMS,'unknown historical strategy')
    require(strategy!='dynamollm' or spec['model']=='14b','C resident Dynamo cannot be labelled full Dynamo')
    require(strategy!='dynamollm-resident' or spec['model']=='7b','A historical Dynamo implementation is full')
    datasets=list(datasets or DATASETS);require(len(set(datasets))==len(datasets) and set(datasets)<=set(DATASETS),'invalid dataset selection')
    if spec['layout']=='distserve-longbench':
        require(spec['model']=='14b' and strategy=='distserve' and datasets==['longbench'],'heterogeneous gate only permits A DistServe LongBench')
    elif spec['model']=='14b' and strategy=='distserve':
        require('longbench' not in datasets,'resident A DistServe cannot run historical TP2 LongBench route')
    return datasets

def scope(spec):
    actual=[(i['tp'],i['gpus']) for i in spec['instances']]
    expected=[(1,[j]) for j in range(8)] if spec['layout']=='resident' else [(1,[j]) for j in range(5)]+[(2,[6,7])]
    require(actual==expected,'historical resident/heterogeneous layout differs')
    require(spec['layout']=='resident' or spec['model']=='14b','C does not have the A heterogeneous route')
    for i in spec['instances']:
        require(i['native_kind']=='legacy_sync_put' and i.get('scheduler_cache_observed') is False and i['image']==IMAGE,
            'legacy capability/image changed')

async def live(spec,receipt,executor,frozen):
    import aiohttp
    created={x['name']:x['container_id'] for x in receipt['created']}
    require(set(created)=={i['container_name'] for i in spec['instances']},'deployment receipt layout differs')
    names=(await executor.command('docker','ps','--format','{{.Names}}')).split()
    require(set(names)==set(created),'unexpected resident process on energy-measured node')
    inventory=json.loads(await executor.command('docker','inspect',*names));by_name={x['Name'].lstrip('/'):x for x in inventory};instances=[]
    async with aiohttp.ClientSession(trust_env=False) as session:
        for i in spec['instances']:
            c=by_name[i['container_name']];cfg=read(i['config'])
            require(c['Id']==created[i['container_name']] and c['Image']==IMAGE and c['State']['Running'] is True,'actual deployment process differs')
            require(c['Config']['Cmd']==['python3',i['engine_entry'],'--config',i['config']],'actual standalone entry/config differs')
            require(set(c['Config']['Env'])==set(i['environment']),'actual historical environment differs')
            require((cfg['tp'],cfg['max_model_len'],cfg['max_num_batched_tokens'],cfg['max_num_seqs'])==(i['tp'],8192,8192,32),'actual static work/budget differs')
            p=await executor.http(session,i,'/provenance');r=await executor.wait_idle(session,i)
            imported={str(f):sha(f) for f in Path(i['engine_entry']).parent.glob('*.py')}
            require(p.get('instance_id')==i['id'] and p.get('tp')==i['tp'] and p.get('model')==cfg['model']=='/models/Qwen2.5-'+spec['model'].upper()+'-Instruct'
                and p.get('dtype')=='bfloat16' and p.get('max_model_len')==8192 and type(p.get('pid')) is int
                and p.get('cuda_visible_devices')==','.join(map(str,i['gpus'])) and p.get('source_files_at_import')==imported,
                'actual model/TP/source/PID differs')
            require(imported and all(frozen.get(f)==h for f,h in imported.items()),'actual imported engine files not frozen')
            require(r.get('accepting') is True and r.get('generation')==r.get('acknowledged_generation'),'actual accepting idle owner ACK required')
            record={k:i[k] for k in ('id','tp','gpus','role','port','kv_port','url','native_kind','scheduler_cache_observed')}
            # Outer safety starts and restores mixed; the actual Controller
            # applies each frozen dataset's own historical PD roles on startup.
            record['role']='mixed'
            record.update(engine_config=i['config'],container=dict(name=i['container_name'],id=c['Id'],image=c['Image'],StartedAt=c['State']['StartedAt']),
                provenance={k:p[k] for k in ('instance_id','pid','tp','model','dtype','max_model_len','cuda_visible_devices','source_files_at_import')})
            instances.append(record)
    return instances,inventory

def configurations(spec,instances,strategy,datasets,out,output,frozen):
    original=load('ac_binding_historical_policy',POLICY/'bind.py');configs={};notes={}
    if strategy=='dynamollm':
        topology=Path(spec['host_release'])/'src/ecopadg/serving/topology.py'
        require('observation_engine_entry' in topology.read_text() and 'observation_engine_sha256' in topology.read_text(),
            'full Dynamo lifecycle lacks explicit frozen engine entry')
    for dataset in datasets:
        old,_=original.configuration(spec['model'],dataset,strategy)
        roles={(i['tp'],tuple(i['gpus'])):i['role'] for i in old['instances']}
        layout=[dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],port=i['port'],kv_port=i['kv_port'],url=i['url'],
            container_name=i['container']['name'],role=roles[i['tp'],tuple(i['gpus'])]) for i in instances]
        dep=dict(model=spec['model'],historical_engine_observation_only=True,layouts={strategy+':'+dataset:layout},topology_source_path_verified=strategy=='dynamollm')
        cfg,note=original.configuration(spec['model'],dataset,strategy,deployment=dep,out=output/'unused')
        cfg.update(comparison_system='dynamollm' if strategy.startswith('dynamollm') else strategy,controller_source_release=spec['host_release'])
        if strategy=='dynamollm':
            template=read(spec['instances'][0]['config']);template.update(observation_engine_entry=spec['source_entry'],observation_engine_sha256=sha(spec['source_entry']))
            path=out/'engine-template.json'
            if not path.exists():write(path,template)
            cfg['topology'].update(runtime_dir=str(out/'dynamic-runtime'),image=IMAGE,engine_template=str(path))
            frozen[str(path.resolve())]=sha(path)
        path=out/'configs'/(dataset+'.json');write(path,cfg);configs[dataset]=str(path.resolve());frozen[str(path.resolve())]=sha(path);notes[dataset]=note
        def external(value,key=''):
            if isinstance(value,dict):
                for k,v in value.items():external(v,k)
            elif isinstance(value,list):
                for v in value:external(v,key)
            elif isinstance(value,str) and value.startswith('/') and key!='journal' and Path(value).is_file():frozen[value]=sha(value)
        external(cfg)
    return configs,notes

async def build(a):
    spec=read(a.spec);scope(spec);datasets=selected(spec,a.strategy,a.dataset)
    require(socket.gethostname()==spec['hostname'],'wrong physical node')
    require(not a.out.exists(),'new binding directory required; no overwrite')
    require(bool(a.strategy)==bool(a.gate),'performance requires a gate; bootstrap does not consume one')
    frozen=dict(spec['files']);verify(frozen);frozen.update(package_files(ROOT));frozen.update(package_files(POLICY))
    contract=read(ROOT/'source-contract.json');verify(contract['files']);frozen.update(contract['files'])
    executor=load('ac_binding_execution',Path(spec['executor_release'])/'run.py')
    source=read(spec['pdb_binding']);executor.validate_binding(source);frozen.update(source['files'])
    receipt=read(a.receipt);require(receipt.get('complete') is True and receipt.get('measurement_valid') is True,'successful measured deployment required')
    for p in (a.spec,a.receipt):frozen[str(p.resolve())]=sha(p)
    instances,inventory=await live(spec,receipt,executor,frozen)
    mechanism=None
    if a.strategy:
        from ecopadg.serving.measurement import power_evidence
        mechanism,raw_files=audit(a.gate,instances,a.strategy,power_evidence,hetero=spec['layout']!='resident')
        frozen.update(raw_files)
    # Only now may a runnable configuration directory be created.
    a.out.mkdir(parents=True);write(a.out/'identity.json',inventory);frozen[str((a.out/'identity.json').resolve())]=sha(a.out/'identity.json')
    configs={};notes={};output=(a.output or a.out/'results').resolve()
    if a.strategy:configs,notes=configurations(spec,instances,a.strategy,datasets,a.out,output,frozen)
    canonical='mixed' if not a.strategy else 'dynamollm' if a.strategy.startswith('dynamollm') else a.strategy
    binding=dict(schema=1,protocol_id=spec['protocol_id'],model=spec['model'],system=canonical,implementation_variant=a.strategy or 'correctness-only',
        hostname=spec['hostname'],deadline_s=spec['deadline_s'],host_release=spec['host_release'],output=str(output),configs=configs,
        instances=instances,files=frozen,large_inputs=source.get('large_inputs',{}),window_s=100,seeds=[701],historical_policy=notes,
        deployment_receipt=str(a.receipt.resolve()),correctness_gate_required_before_performance=not bool(a.strategy),
        output_correctness_verified=bool(a.strategy),mechanism_proof=mechanism,formal_eligible=False,
        runtime_role_restore='mixed continuous; frozen per-dataset historical roles applied by Controller.initialize')
    if a.gate:binding['correctness_evidence']=str(a.gate.resolve())
    # Recheck all read-only evidence after reconstruction and policy binding.
    executor.validate_binding(binding)
    import aiohttp
    async with aiohttp.ClientSession(trust_env=False) as session:await executor.identity(session,binding)
    write(a.out/'binding.json',binding)
    print(json.dumps(dict(binding=str(a.out/'binding.json'),sha256=sha(a.out/'binding.json'),mode='performance' if a.strategy else 'correctness-only',datasets=datasets,hardware_actions=False)))
    return binding

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','receipt','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--strategy',choices=SYSTEMS);p.add_argument('--gate',type=Path)
    p.add_argument('--dataset',action='append',choices=DATASETS);a=p.parse_args()
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'fresh lease required; inherited active lease FD refused')
    spec=read(a.spec);host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.campaign import node_lease
    with node_lease():asyncio.run(build(a))

if __name__=='__main__':main()
