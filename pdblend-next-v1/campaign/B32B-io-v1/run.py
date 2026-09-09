"""PDBlend-only B32B: preserve old four TP2 engines; test new four mixed.

Run with the frozen io-v1.1-runtime host PYTHONPATH and a verified --image ID.
No baseline execution, old result mutation, or container deletion is performed.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import aiohttp

from ecopadg.serving.campaign import node_lease
from ecopadg.serving.cell import run_cell

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
ENGINE_RELEASE=PROJECT/'releases/io-v1-runtime'
HOST_RELEASE=PROJECT/'releases/io-v1.1-runtime'
HISTORY=Path('/root/workspace/pdblend/new-results/campaigns/node-b-v9/quick32-v1')
OLD_NAMES=tuple('pdb-v2-b32q'+str(i) for i in range(4))
NEW_NAMES=tuple('pdb-next-b32q'+str(i) for i in range(4))
HTTP_BASE=24300
KV_BASE=28600
CONTROLLER_PORT=18380


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def write(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


async def command(*args,timeout=180):
    process=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)
    try: out,_=await asyncio.wait_for(process.communicate(),timeout)
    except BaseException:
        if process.returncode is None:process.kill();await process.wait()
        raise
    if process.returncode:raise RuntimeError(out.decode(errors='replace')[-4000:])
    return out.decode().strip()


async def request(session,url,*,payload=None,timeout=120):
    async with session.request('GET' if payload is None else 'POST',url,json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        if response.status!=200:raise RuntimeError(await response.text())
        return await response.json()


def engine_config(container):
    args=container['Config']['Cmd']
    path=Path(args[args.index('--config')+1] if '--config' in args else args[-1])
    return path,json.loads(path.read_text())


def require_idle(raw,label):
    if raw.get('error') or raw.get('runtime_error') or any(raw.get(key) for key in
            ('active','running','waiting','kv_allocations','transfer_allocations')):
        raise RuntimeError('engine is not safely idle: '+label)


async def deploy(session,image,status):
    if (ROOT/'previous-deployment.json').exists():
        raise RuntimeError('deployment snapshot already exists; use --stage cells or inspect the prior attempt')
    if await command('docker','image','inspect','--format','{{.Id}}',image)!=image:
        raise RuntimeError('requested image is not an immutable local image')
    previous=json.loads(await command('docker','inspect',*OLD_NAMES))
    write('previous-deployment.json',previous)
    write('gpu-before.json',{'output':await command('nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
        '--format=csv,noheader'),'at_s':time.time()})
    old_configs=[];before={};all_gpus=[]
    for container in previous:
        name=container['Name'].lstrip('/')
        if name not in OLD_NAMES or container['State']['Running'] is not True:
            raise RuntimeError('old deployment identity or running state differs')
        path,config=engine_config(container)
        gpus=[int(g) for item in container['Config']['Env'] if item.startswith('CUDA_VISIBLE_DEVICES=')
              for g in item.split('=',1)[1].split(',')]
        if config['tp']!=2 or len(gpus)!=2 or config.get('model')!='/models/Qwen2.5-32B-Instruct':
            raise RuntimeError('old engine is not the expected Qwen32B TP2 configuration')
        raw=await request(session,f"http://127.0.0.1:{config['port']}/runtime",timeout=5)
        require_idle(raw,name)
        before[name]=dict(config_path=str(path),config_sha256=digest(path),config=config,state=raw,gpus=gpus)
        old_configs.append((OLD_NAMES.index(name),config,gpus));all_gpus.extend(gpus)
    if sorted(all_gpus)!=list(range(8)):raise RuntimeError('four TP2 engines must cover exactly eight distinct GPUs')
    write('previous-runtime.json',before)
    peers={'nextb'+str(i):dict(host='127.0.0.1',tp=2,kv_port=KV_BASE+i*32) for i in range(4)}
    (ROOT/'runtime').mkdir(exist_ok=True)
    instances=[]
    for i,old,gpus in sorted(old_configs):
        config=dict(old)
        config.update(id='nextb'+str(i),role='mixed',port=HTTP_BASE+i,kv_port=KV_BASE+i*32,
            peers=peers,runtime_dir=str(ROOT/'runtime'),initial_generation=0,operation_timeout_s=120)
        write('engine-'+str(i)+'.json',config)
        instances.append(dict(id='nextb'+str(i),role='mixed',tp=2,gpus=gpus,port=HTTP_BASE+i,
            kv_port=KV_BASE+i*32,url=f'http://127.0.0.1:{HTTP_BASE+i}',container_name=NEW_NAMES[i]))
    write('deployment.json',dict(image=image,engine_source_release=str(ENGINE_RELEASE),
        controller_source_release=str(HOST_RELEASE),instances=instances,
        rollback_command='python3 run.py --stage rollback',old_containers_preserved=list(OLD_NAMES)))
    status['phase']='replacing_idle_pdb_engines';write('status.json',status)
    # Only stop the four inspected PDB engines. Their configuration and names survive.
    await command('docker','stop','-t','10',*OLD_NAMES)
    for i,instance in enumerate(instances):
        await command('docker','run','-d','--name',NEW_NAMES[i],'--gpus','all',
            '--network','host','--ipc','host','-e','CUDA_VISIBLE_DEVICES='+','.join(map(str,instance['gpus'])),
            '-e','NCCL_P2P_DISABLE=1','-e','NCCL_SHM_DISABLE=1','-e','NCCL_IB_DISABLE=1',
            '-e','NCCL_CUMEM_ENABLE=0','-e','NCCL_DEBUG=WARN','-e','VLLM_HOST_IP=127.0.0.1',
            '-e','PYTHONPATH='+str(ENGINE_RELEASE/'src'),
            '-v','/root/workspace:/root/workspace','-v','/root/workspace/models:/models:ro',
            image,'python3','-m','ecopadg.serving.engine','--config',str(ROOT/('engine-'+str(i)+'.json')))
    return instances


async def ready_and_smoke(session,instances,status):
    status['phase']='engine_loading';write('status.json',status)
    deadline=time.monotonic()+360;ready={};ack=[]
    while len(ready)<4:
        for instance in instances:
            if instance['id'] in ready:continue
            try:
                raw=await request(session,instance['url']+'/runtime',timeout=2)
                if raw.get('error'):raise RuntimeError(str(raw['error']))
                if raw.get('accepting') and raw.get('acknowledged_generation')==-1:
                    payload={k:raw[k] for k in ('role','mode','admit_prefill','admit_decode','generation')}
                    payload['generation']+=1
                    ack.append(dict(instance=instance['id'],payload=payload,
                        result=await request(session,instance['url']+'/control',payload=payload)))
                elif raw.get('accepting') and raw.get('acknowledged_generation')==raw.get('generation'):
                    require_idle(raw,instance['id']);ready[instance['id']]=raw
            except (aiohttp.ClientError,asyncio.TimeoutError):pass
        if time.monotonic()>deadline:raise RuntimeError('new B engine readiness deadline expired')
        if len(ready)<4:await asyncio.sleep(1)
    write('initial-control-ack.json',ack);write('ready.json',ready)
    status['phase']='correctness_smoke';write('status.json',status)
    smoke=[]
    for length in (128,7168):
        body=dict(prompt=([9707,1879,13]*(length//3+1))[:length],max_tokens=32,
            temperature=0,top_p=1.,seed=0,ignore_eos=True,stream=False)
        results=await asyncio.gather(*(request(session,i['url']+'/v1/completions',payload=body) for i in instances))
        for result in results:
            if (len(result.get('token_ids',[]))!=32 or result.get('usage',{}).get('completion_tokens')!=32
                    or result['token_ids']!=results[0].get('token_ids')):
                write('smoke-failure.json',dict(prompt_tokens=length,results=results))
                raise RuntimeError('TP2 smoke output incomplete or differs between equivalent instances')
        smoke.append(dict(prompt_tokens=length,results=results))
    barriers=[]
    for instance in instances:
        raw=await request(session,instance['url']+'/runtime',timeout=5)
        proof=await request(session,instance['url']+'/drain',
            payload={'expected_generation':raw['generation']},timeout=120)
        if (proof.get('drained') is not True or proof.get('generation')!=raw['generation']+1
                or proof.get('drain_proof_type')!='synchronous_put_owner_barrier'):
            raise RuntimeError('smoke drain barrier proof incomplete')
        barriers.append(dict(instance=instance['id'],proof=proof))
    write('smoke.json',dict(passed=True,requests=smoke,drain_barriers=barriers,
        scope='same-TP2 replica equality and prescribed output smoke; not an energy or baseline comparison'))


async def cells(datasets,status):
    instances=json.loads((ROOT/'deployment.json').read_text())['instances']
    refs={}
    for candidate in (HISTORY/'B32B-four-baseline-two-runs-v1/baseline-reference.json',
            HISTORY/'baseline-reference-r1.json'):
        if candidate.exists():refs[str(candidate)]=digest(candidate)
    for dataset in datasets:
        old_path=HISTORY/'B32B-allmixed4-r5'/(dataset+'.json')
        old=json.loads(old_path.read_text());config=dict(old)
        if old.get('strategy')!='pdblend-joint' or old.get('allow_pd') is not False:
            raise RuntimeError('historical source configuration is not PDB allmixed4')
        config.update(evaluation_protocol='evaluation-v3',request_timeout_s=120,slo_attainment_target=.9,
            allow_pd=False,dynamic_pools=False,slow_topology=False,instances=instances,port=CONTROLLER_PORT,
            engine_source_release=str(ENGINE_RELEASE),controller_source_release=str(HOST_RELEASE))
        # Preserve the old 211-token prior, frequency/role costs, parking and SLO.
        path=ROOT/(dataset+'.v1.1.config.json');write(path.name,config)
        trace_source=HISTORY/(dataset+'.trace.json');trace=ROOT/(dataset+'.trace.json')
        trace.write_bytes(trace_source.read_bytes())
        reference=dict(scope='PDBlend-only development; frozen historical baselines not rerun',
            historical_config=str(old_path),historical_config_sha256=digest(old_path),
            historical_protocol=old.get('evaluation_protocol','legacy-measurement-v2'),
            new_protocol='evaluation-v3',source_changed=True,engine_changed=True,
            arrival_basis_and_drain_changed=True,same_protocol_formal_comparison=False,
            trace_sha256=digest(trace),baseline_reference_hashes=refs)
        write(dataset+'.historical-reference.json',reference)
        status.update(phase='pdblend_cell',dataset=dataset);write('status.json',status)
        args=SimpleNamespace(config=path,trace=trace,out=ROOT/('cell-'+dataset+'-v1.1'),
            dataset=dataset,load='pilot',seed=11,split='development',strategy=None,freeze=None,
            mechanisms=None,slo_ttft_s=None,slo_tpot_s=None,timeout=120)
        result=await run_cell(args)
        status.setdefault('cells',{})[dataset]={k:result.get(k) for k in ('energy_j','slo_attainment',
            'work_complete','measurement_valid','completed','n_expected','runtime_error')}
        write('status.json',status)
        if any(digest(p)!=sha for p,sha in refs.items()):raise RuntimeError('historical baseline reference changed')
        if not result.get('measurement_valid'):raise RuntimeError('measurement invalid; retaining all raw: '+dataset)


async def rollback(session):
    previous=json.loads((ROOT/'previous-deployment.json').read_text())
    if {p['Name'].lstrip('/') for p in previous}!=set(OLD_NAMES):raise RuntimeError('rollback identity mismatch')
    running=set((await command('docker','ps','--format','{{.Names}}')).splitlines())
    for i,name in enumerate(NEW_NAMES):
        if name in running:
            require_idle(await request(session,f'http://127.0.0.1:{HTTP_BASE+i}/runtime',timeout=5),name)
            await command('docker','stop','-t','10',name)
    await command('docker','start',*OLD_NAMES)
    write('rollback.json',dict(at_s=time.time(),old_containers_started=list(OLD_NAMES),new_containers_deleted=False))


async def main(args):
    ROOT.mkdir(parents=True,exist_ok=True)
    status=dict(phase='starting',pid=os.getpid(),started_s=time.time(),complete=False,
        stage=args.stage,scope='PDBlend only',engine_source_release=str(ENGINE_RELEASE),
        controller_source_release=str(HOST_RELEASE))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
            if args.stage=='rollback':await rollback(session)
            else:
                if args.stage in ('all','deploy-smoke'):
                    if not args.image or not re.fullmatch(r'sha256:[0-9a-f]{64}',args.image):
                        raise ValueError('--image must be the actual immutable B image ID')
                    instances=await deploy(session,args.image,status)
                    await ready_and_smoke(session,instances,status)
                if args.stage in ('all','cells'):await cells(args.datasets,status)
        status.update(phase='finished',complete=True,finished_s=time.time())
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),finished_s=time.time())
        raise
    finally:write('status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image')
    parser.add_argument('--stage',choices=('all','deploy-smoke','cells','rollback'),default='all')
    parser.add_argument('--datasets',choices=('longbench','alpaca','sharegpt'),nargs='+',
        default=['longbench','alpaca','sharegpt'])
    args=parser.parse_args()
    with node_lease():asyncio.run(main(args))
