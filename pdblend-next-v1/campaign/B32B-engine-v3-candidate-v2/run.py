"""Isolated B TP2 v3 deployment candidate; only --phase build/deploy/validate mutate Docker/GPU state."""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import pathlib
import socket
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parent
WORK=ROOT.parents[1]
RELEASE=WORK/'releases/io-v3-runtime'
PREVIOUS=ROOT.parent/'B32B-capacity2-v1'
ORIGINAL=ROOT.parent/'B32B-io-v1'
BASE_IMAGE='sha256:f310d6341a999e7253332ca287c28d84f6d987ec51aec5f117d1f30b41afa577'
BASE_TAG='pdblend-next-b32-v3-base:frozen-f310d6341a99'
A_IMAGE='sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'
OLD_NAMES=['pdb-next-b32q'+str(i) for i in range(4)]
IDS=['nextv3b0','nextv3b1']
NAMES=['pdb-v2-'+i for i in IDS]
PORTS=[33500,33501]
KV_PORTS=[33700,33732]


def require(test,message):
    if not test:raise RuntimeError(message)


def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(pathlib.Path(path).read_text())


def terminal(status):
    return bool(status.get('finished_s') and status.get('phase') in ('finished','failed')
                and (status['phase']=='failed' or status.get('complete') is True))


def configs(template):
    require(template.get('model')=='/models/Qwen2.5-32B-Instruct','wrong B32 model')
    require(template.get('tp')==2,'B needs original TP2 template')
    peers={iid:dict(host='127.0.0.1',tp=2,kv_port=kv) for iid,kv in zip(IDS,KV_PORTS)}
    output=[]
    for index,iid in enumerate(IDS):
        c=dict(template,id=iid,tp=2,port=PORTS[index],kv_port=KV_PORTS[index],
            peers=peers,role='mixed',initial_generation=0,runtime_dir=str(ROOT/'runtime'),
            retained_weights=None,verify_recompute=False,verify_transport=False,
            validated_tp_pairs=[[2,2]],max_model_len=8192,max_num_batched_tokens=8192,max_num_seqs=32)
        c.pop('scheduler_budget',None) # runtime role validator starts with the unmodified startup budget.
        output.append(c)
    return output


def idle(raw,*,accepting=None):
    fields=('active','running','waiting','kv_allocations','transfer_allocations',
            'transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')
    require(all(k in raw for k in fields),'missing owner/transport residual fields')
    require(not any(raw[k] for k in fields),'live request/KV/transfer not drained')
    require(not raw.get('error') and not raw.get('runtime_error'),'engine quarantined')
    require(raw.get('transport_healthy') is True,'transport not healthy')
    require(raw.get('generation')==raw.get('acknowledged_generation'),'owner ACK differs')
    require(0<=time.time()-raw.get('timestamp',0)<=1,'stale owner snapshot')
    if accepting is not None:require(raw.get('accepting') is accepting,'wrong admission state')


def drain_proof(before,proof,tp=2):
    require(proof.get('drained') is True and proof.get('accepting') is False
            and proof.get('generation')==before['generation']+1
            and proof.get('drain_proof_type')=='synchronous_put_owner_barrier','no actual owner/rank drain')
    ranks=proof.get('transfers',[])
    require(len(ranks)==tp,'TP rank drain evidence missing')
    require(all(not any(r.get(k) for k in ('buffered_tensors','inflight_receives','inflight_sends','buffered_gpu_bytes','allocations'))
                and r.get('listener_alive') is True for r in ranks),'TP rank residue or dead listener')


def verify_files():
    manifest=read(ROOT/'candidate-manifest.json')
    for path,digest in manifest['files'].items():require(sha(ROOT/path)==digest,'candidate changed: '+path)
    require(sha(RELEASE/'manifest.json')==manifest['release_manifest_sha256'],'release identity changed')
    for path,digest in read(RELEASE/'manifest.json')['files'].items():
        require(sha(RELEASE/path)==digest,'release bytes changed: '+path)
    return manifest


async def command(*args,timeout=300):
    p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:out,_=await asyncio.wait_for(p.communicate(),timeout)
    except BaseException:
        if p.returncode is None:p.kill();await p.wait()
        raise
    row=dict(args=list(args),returncode=p.returncode,output=out.decode(errors='replace'),finished_s=time.time())
    with (ROOT/'commands.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    require(p.returncode==0,row['output'][-4000:])
    return row['output'].strip()


async def http(session,port,path,payload=None,timeout=45):
    import aiohttp
    at=time.time();method=session.post if payload is not None else session.get
    options=dict(timeout=aiohttp.ClientTimeout(total=timeout))
    if payload is not None:options['json']=payload
    async with method(f'http://127.0.0.1:{port}'+path,**options) as r:
        body=await r.text()
        with (ROOT/'http.jsonl').open('a') as f:f.write(json.dumps(dict(port=port,path=path,payload=payload,status=r.status,body=body,started_s=at,finished_s=time.time()))+'\n')
        require(r.status==200,body[-2000:])
        return json.loads(body)


async def ready(session,port,timeout=300):
    import aiohttp
    limit=time.monotonic()+timeout
    while True:
        try:
            raw=await http(session,port,'/runtime',timeout=2)
            require(not raw.get('error'),'engine failed to load: '+str(raw.get('error')))
            if raw.get('accepting') and raw.get('total_kv_tokens',0)>0:
                target=raw['generation']+1
                body=dict(generation=target,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
                if raw.get('scheduler_budget'):body['scheduler_budget']=raw['scheduler_budget']
                await http(session,port,'/control',body)
                raw=await http(session,port,'/runtime');idle(raw,accepting=True)
                require(raw['generation']==target,'initial real ACK missing')
                return raw
        except (aiohttp.ClientError,asyncio.TimeoutError):pass
        require(time.monotonic()<limit,'real readiness timed out')
        await asyncio.sleep(.25)


async def verify_live(session,index,image):
    raw=await http(session,PORTS[index],'/provenance')
    inspection=json.loads(await command('docker','inspect',NAMES[index]))[0]
    expected={str(RELEASE/path):value for path,value in read(RELEASE/'manifest.json')['files'].items()
              if path.startswith('src/ecopadg/serving/') and path.endswith('.py')}
    require(raw.get('source_files_at_import')==expected,'imported serving source mismatch')
    require(raw.get('instance_id')==IDS[index] and raw.get('tp')==2
            and raw.get('model')=='/models/Qwen2.5-32B-Instruct'
            and raw.get('cuda_visible_devices')==f'{2*index},{2*index+1}','wrong live B model/TP/GPU')
    require(inspection['Image']==image and inspection['State']['Running'],'wrong B image or dead container')
    require(nccl_environment(inspection)==read(ROOT/'transport-environment.json')['nccl_environment'],
            'new B container NCCL environment differs from old replicas')
    # Verify actual image patch bytes; this is CPU inspection inside the already running engine container.
    expected_patch=read(ROOT/'candidate-manifest.json')['image_patch_files']
    code='import hashlib,json,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in '+repr(list(expected_patch))+'}))'
    observed=json.loads(await command('docker','exec',NAMES[index],'python3','-c',code))
    require(observed==expected_patch,'loaded image lacks frozen B v2/v3 patch bytes')
    return dict(provenance=raw,container=inspection,image_patch_sha256=observed)


def nccl_environment(inspection):
    values={}
    for item in inspection.get('Config',{}).get('Env',[]):
        key,separator,value=item.partition('=')
        if key.startswith('NCCL_'):
            require(separator and key not in values,'duplicate or malformed NCCL environment')
            values[key]=value
    return values


def inherited_transport(inventory):
    rows={r['Name'].lstrip('/'):r for r in inventory}
    environments={name:nccl_environment(rows[name]) for name in OLD_NAMES[:2]}
    expected=environments[OLD_NAMES[0]]
    require(all(v==expected for v in environments.values()),'old B replicas use different NCCL environments')
    for key,value in dict(NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_CUMEM_ENABLE='0').items():
        require(expected.get(key)==value,'unexpected old transport setting: '+key)
    return dict(nccl_environment=expected,old_replica_environments=environments,
                policy='copy every actual NCCL_* value exactly; do not invent absent channel settings')


async def tag_base_image():
    base=json.loads(await command('docker','image','inspect',BASE_IMAGE))[0]
    require(base['Id']==BASE_IMAGE,'wrong B io-v1 immutable base')
    await command('docker','tag',BASE_IMAGE,BASE_TAG)
    tagged=json.loads(await command('docker','image','inspect',BASE_TAG))[0]
    require(tagged['Id']==BASE_IMAGE,'local B base tag does not resolve to the frozen image')
    return base,tagged


async def build():
    manifest=verify_files()
    require(not (ROOT/'image-receipt.json').exists(),'build already recorded; use new candidate')
    base,tagged=await tag_base_image()
    await command('docker','build','--pull=false','--build-arg','BASE_IMAGE='+BASE_TAG,
                  '--iidfile',str(ROOT/'image-id.txt'),str(ROOT/'build-context'),timeout=600)
    image=(ROOT/'image-id.txt').read_text().strip()
    require(image.startswith('sha256:') and image not in (BASE_IMAGE,A_IMAGE),'new B image required')
    inspection=json.loads(await command('docker','image','inspect',image))[0]
    write('image-receipt.json',dict(image=image,base_image=BASE_IMAGE,base_inspection=base,base_tag=BASE_TAG,base_tag_inspection=tagged,
        inspection=inspection,patches=manifest['image_patch_files'],recipe='B io-v1 + frozen v2 files then frozen v3 files'))


async def deploy():
    import aiohttp
    verify_files()
    require(not (ROOT/'deployment-status.json').exists(),'deployment already attempted; preserve evidence')
    require(terminal(read(PREVIOUS/'status.json')),'B2 task must be terminal before any deployment')
    image=read(ROOT/'image-receipt.json')['image']
    require(image not in (A_IMAGE,BASE_IMAGE),'B v3 image receipt required')
    status=dict(complete=False,phase='preflight',started_s=time.time(),image=image,baseline_reruns=False)
    write('deployment-status.json',status)
    attempted=[];stopped=[];drained_old=[]
    try:
        allids=(await command('docker','ps','-aq')).split()
        inventory=json.loads(await command('docker','inspect',*allids))
        write('all-containers.before.json',inventory)
        running={r['Name'].lstrip('/') for r in inventory if r['State']['Running']}
        require(running==set(OLD_NAMES[:2]),'unexpected running service; do not change allocation')
        transport=inherited_transport(inventory);write('transport-environment.json',transport)
        transport_args=[arg for key,value in sorted(transport['nccl_environment'].items()) for arg in ('-e',key+'='+value)]
        historical_ids=set();historical_ports=set()
        for row in inventory:
            args=row.get('Args',[])
            if '--config' not in args:continue
            path=pathlib.Path(args[args.index('--config')+1]);c=read(path)
            if 'id' not in c:continue
            historical_ids.add(c['id'])
            if 'port' in c:historical_ports.add(c['port'])
            if 'kv_port' in c:historical_ports.update(range(c['kv_port'],c['kv_port']+32))
        require(not set(IDS)&historical_ids,'new process IDs collide with historical NCCL identity')
        ports=PORTS+[p for k in KV_PORTS for p in range(k,k+32)]
        require(not set(ports)&historical_ports,'new ports reuse historical channels')
        for port in ports:
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        async with aiohttp.ClientSession(trust_env=False) as session:
            old={}
            for index in range(2):
                state=await http(session,24300+index,'/runtime');idle(state)
                drained_old.append(index)
                barrier=await http(session,24300+index,'/drain',dict(expected_generation=state['generation']))
                drain_proof(state,barrier);old[str(index)]=dict(state=state,barrier=barrier)
            write('old-owner-drain.json',old)
            for name in OLD_NAMES[:2]:
                stopped.append(name)
                await command('docker','stop','--time','30',name,timeout=120)
            require(not any(r['State']['Running'] for r in json.loads(await command('docker','inspect',*OLD_NAMES))),'old B process is still running')
            from ecopadg.measure.backends import PynvmlBackend
            hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
            def free():
                n=hardware._nvml;rows=[]
                for g in range(4):
                    h=hardware._handle(g);mem=n.nvmlDeviceGetMemoryInfo(h,version=n.nvmlMemory_v2)
                    rows.append(dict(gpu=g,used_bytes=mem.used,free_bytes=mem.free,reserved_bytes=mem.reserved,
                        memory_api='nvmlDeviceGetMemoryInfo_v2',
                        compute_processes=len(n.nvmlDeviceGetComputeRunningProcesses(h)),
                        graphics_processes=len(n.nvmlDeviceGetGraphicsRunningProcesses(h))))
                return rows
            memory=await asyncio.to_thread(free);write('target-gpus.free.json',memory)
            require(all(r['used_bytes']==r['compute_processes']==r['graphics_processes']==0 for r in memory),'target GPU retains a process/allocation')
            for index,c in enumerate(configs(read(ROOT/'engine-template.frozen.json'))):
                write(f'engine-{index}.json',c)
                attempted.append(NAMES[index])
                await command('docker','run','-d','--name',NAMES[index],'--gpus','all','--network','host','--ipc','host',
                    '-e',f'CUDA_VISIBLE_DEVICES={2*index},{2*index+1}',*transport_args,
                    '-e','PYTHONPATH='+str(RELEASE/'src'),'-e','VLLM_HOST_IP=127.0.0.1',
                    '-v','/root/workspace:/root/workspace','-v','/root/workspace/models:/models:ro',
                    image,'python3','-m','ecopadg.serving.engine','--config',str(ROOT/f'engine-{index}.json'))
            state={str(i):await ready(session,PORTS[i]) for i in range(2)}
            proof={str(i):await verify_live(session,i,image) for i in range(2)}
            write('ready.json',dict(state=state,proof=proof))
            topology={role:dict(id=IDS[i],tp=2,gpus=[2*i,2*i+1],port=PORTS[i],kv_port=KV_PORTS[i],container_name=NAMES[i])
                      for i,role in enumerate(('prefill','decode'))}
            write('topology.json',topology)
            after=json.loads(await command('docker','inspect',*allids,*NAMES));write('all-containers.after.json',after)
            require({r['Id'] for r in inventory}<={r['Id'] for r in after},'an old container was deleted')
            status.update(phase='ready_for_correctness_validation',complete=True,finished_s=time.time(),stopped_preserved=stopped)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc),stopped_preserved=stopped,finished_s=time.time())
        for name in attempted:
            try:
                rows=json.loads(await command('docker','inspect',name))
                if rows[0]['State']['Running']:await command('docker','stop','--time','30',name,timeout=120)
            except BaseException as cleanup:status.setdefault('cleanup_errors',[]).append(repr(cleanup))
        if not stopped:
            # A pre-stop guard failure must not leave a surviving old engine paused by our drain.
            async with aiohttp.ClientSession(trust_env=False) as recovery:
                for index in drained_old:
                    try:
                        raw=await http(recovery,24300+index,'/runtime')
                        await http(recovery,24300+index,'/control',dict(generation=raw['generation']+1,
                            role='mixed',mode='continuous',admit_prefill=True,admit_decode=True))
                        raw=await http(recovery,24300+index,'/runtime');idle(raw,accepting=True)
                        status.setdefault('prestop_old_restore',{})[str(index)]=raw
                    except BaseException as cleanup:status.setdefault('cleanup_errors',[]).append(repr(cleanup))
        status['rollback_note']='old stopped containers retained; do not restart old NCCL identities automatically; restoration requires a fresh-ID deployment review'
        raise
    finally:write('deployment-status.json',status)


async def validate():
    from validation import validate_all
    verify_files()
    require(read(ROOT/'deployment-status.json').get('complete') is True,'new B deployment not ready')
    await validate_all()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['check','build','deploy','validate'],default='check')
    args=parser.parse_args()
    if args.phase=='check':
        verify_files();print(json.dumps(dict(candidate_valid=True,gpu_execution=False,model='32B',tp=2,gpus=[[0,1],[2,3]],ports=PORTS,kv_ports=KV_PORTS)));return
    from ecopadg.serving.campaign import node_lease
    from ecopadg.serving.engine import EngineService
    require(pathlib.Path(inspect.getfile(EngineService)).resolve()==RELEASE/'src/ecopadg/serving/engine.py','host PYTHONPATH must use io-v3-runtime')
    with node_lease():asyncio.run({'build':build,'deploy':deploy,'validate':validate}[args.phase]())


if __name__=='__main__':main()
