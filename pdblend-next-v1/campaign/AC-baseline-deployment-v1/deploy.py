"""Frozen deployment/group binding for A/C historical baseline mechanisms.

prepare is CPU-only. deploy --run is the sole hardware mutation entry and must
hold a fresh exclusive node lease after predecessor checkpoint validation.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import time

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
DEADLINE=1788872770.0400891
IMAGE='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
META={'14b':dict(node='A',tag='a',resident_port=36000,kv_port=55000),
      '7b':dict(node='C',tag='c',resident_port=37000,kv_port=56000)}

def require(ok,why):
    if not ok:raise RuntimeError(why)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def stat(p):
    s=Path(p).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)

def verify_files(files):
    for p,h in files.items():require(sha(p)==h,'changed frozen file: '+p)

def package_check():
    if (ROOT/'manifest.json').exists():verify_files({str(ROOT/p):h for p,h in read(ROOT/'manifest.json')['files'].items()})

def source_process_alive(pid,binding,manifest):
    if type(pid) is not int or pid<=0:return False
    path=Path('/proc')/str(pid)/'cmdline'
    try:cmd=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except FileNotFoundError:return False
    return 'five-system-execution-' in cmd and (str(binding) in cmd or str(manifest) in cmd)

def terminal_group(binding_path,manifest_path,*,system='pdblend',datasets=None,check_processes=True):
    """Require real complete raw-backed main/scale CPs, not invocation count."""
    b=read(binding_path);m=read(manifest_path);out=Path(b['output'])
    require(b['protocol_id']==m['protocol_id']==PROTOCOL and b['system']==system,'predecessor identity differs')
    require(b['files'].get(str(Path(manifest_path).resolve()))==sha(manifest_path),'predecessor manifest not frozen')
    ds=set(datasets or ('alpaca','sharegpt','longbench'))
    rows=[r for r in m['cells'] if r['system']==system and r['dataset'] in ds]
    counts={p:sum(r['phase']==p for r in rows) for p in ('main','scale')}
    require(counts==dict(main=10*len(ds),scale=6*len(ds)),'predecessor declared group is not 10+6 cells per dataset')
    checkpoints={}
    for row in rows:
        p=out/'checkpoints'/(row['cell_id']+'.json');cp=read(p)
        require(cp['row']==row and cp.get('measurement_valid') is True,'incomplete/foreign predecessor checkpoint')
        require(sha(cp['receipt'])==cp['receipt_sha256'],'predecessor receipt changed')
        receipt=read(cp['receipt'])
        require(receipt.get('measurement_valid') is True and receipt.get('child_stopped') is True
            and receipt.get('clock_restore_complete') is True,'predecessor cleanup not complete')
        require(cp.get('artifacts'),'predecessor checkpoint lacks raw SHA references')
        verify_files(cp['artifacts']);checkpoints[str(p)]=sha(p)
        if row['phase']=='scale':
            ref=read(out/'checkpoints'/(row['reuse_main_cell_id']+'.json'))
            require(ref['row']['trace_sha256']==row['trace_sha256'],'predecessor scale trace reference differs')
    invs=[(p,read(p)) for p in (out/'invocations').glob('*.json')]
    for phase in ('main','scale'):
        candidates=[(p,x) for p,x in invs if x.get('system')==system and x.get('phase')==phase]
        require(candidates,'predecessor phase invocation missing')
        latest=max(candidates,key=lambda z:z[1].get('started_s',0))
        require(latest[1].get('complete') is True and latest[1].get('finished_s') and not latest[1].get('error'),
                'predecessor phase invocation not terminal/clean')
    if check_processes:
        require(not any(source_process_alive(x.get('pid'),binding_path,manifest_path) for _,x in invs),
                'predecessor runner process is still alive')
    return dict(system=system,datasets=sorted(ds),counts=counts,checkpoint_sha256=checkpoints,
        binding_sha256=sha(binding_path),manifest_sha256=sha(manifest_path))

def prepare(args):
    package_check();model=args.model;meta=META[model]
    require(args.layout=='resident' or model=='14b','only A has the historical heterogeneous DistServe layout')
    out=args.out.resolve();require(not out.exists(),'new deployment directory required')
    inventory=read(ROOT/'inputs'/(meta['node']+'-containers-before.json'))
    prefix='pdb-v2-cal0_' if model=='14b' else 'pdb-v2-c7quick'
    old={r['Name'].lstrip('/'):r for r in inventory if r['Name'].lstrip('/').startswith(prefix)}
    require(len(old)==8 and all(r['Image']==IMAGE and not r['State']['Running'] for r in old.values()),'historical eight-engine identity missing')
    reference=read(ROOT/'inputs/engine-config-index.json')[meta['node']]
    templates={int(k.rsplit('_',1)[-1]) if model=='14b' else int(k[-1]):read(ROOT/v['copied_path']) for k,v in reference.items()}
    historical_root=WORKSPACE/'campaign/AC-baseline-100s-preparation-v1'
    historical_files={str(historical_root/p):h for p,h in read(historical_root/'manifest.json')['files'].items()}
    historical_files[str(historical_root/'manifest.json')]=sha(historical_root/'manifest.json')
    verify_files(historical_files)
    historical=load('ac_historical_bind',historical_root/'bind.py')
    selected='longbench' if args.layout!='resident' else 'alpaca'
    baseline,_=historical.configuration(model,selected,'distserve' if args.layout!='resident' else 'mixed')
    out.mkdir(parents=True);new=[];configs={};old_by_gpu={int(next(v.split('=',1)[1] for v in r['Config']['Env'] if v.startswith('CUDA_VISIBLE_DEVICES='))):r for r in old.values()}
    suffix='r' if args.layout=='resident' else 'l';entry=ROOT/'engines'/meta['node']/'engine.py'
    port_base=meta['resident_port']+(100 if suffix=='l' else 0);kv_base=meta['kv_port']+(400 if suffix=='l' else 0)
    for index,i in enumerate(baseline['instances']):
        rid=f'base100{meta["tag"]}{suffix}{index}';oldc=old_by_gpu[i['gpus'][0]]
        cfg=copy.deepcopy(templates[i['gpus'][0]])
        if i['tp']==2:
            # This template is from the actual historical A LongBench TP2
            # deployment, not a synthesized TP1 capacity/retained-cache swap.
            cfg=read(ROOT/'inputs/A-distserve-longbench-tp2.json')
        require(cfg['model']==f'/models/Qwen2.5-{model.upper()}-Instruct' and cfg['max_model_len']==8192
            and cfg['max_num_seqs']==32 and cfg['max_num_batched_tokens']==8192,'historical engine work/budget differs')
        cfg.update(id=rid,tp=i['tp'],port=port_base+index,kv_port=kv_base+index*32,
            role=i['role'],runtime_dir=str(out/'runtime'),initial_generation=0)
        env=[v for v in oldc['Config']['Env'] if not v.startswith('CUDA_VISIBLE_DEVICES=')]
        env+=['CUDA_VISIBLE_DEVICES='+','.join(map(str,i['gpus'])),'PYTHONDONTWRITEBYTECODE=1']
        instance=dict(id=rid,tp=i['tp'],gpus=i['gpus'],role=i['role'],port=cfg['port'],kv_port=cfg['kv_port'],
            url=f'http://127.0.0.1:{cfg["port"]}',container_name='pdb-v2-'+rid,
            config=str(out/'engines'/(rid+'.json')),engine_entry=str(entry),image=IMAGE,
            environment=env,mounts=oldc['Mounts'],historical_container_id=oldc['Id'],historical_engine_id=templates[i['gpus'][0]]['id'],
            native_kind='legacy_sync_put',scheduler_cache_observed=False)
        new.append(instance);configs[rid]=cfg
    peers={i['id']:dict(host='127.0.0.1',tp=i['tp'],kv_port=i['kv_port']) for i in new}
    for i in new:
        configs[i['id']]['peers']=peers;write(i['config'],configs[i['id']])
    prior=read(args.pdb_binding)
    require(prior['model']==model and prior['system']=='pdblend','wrong source PDB binding')
    require(read(args.workloads)['model']==model,'wrong model workload source')
    required=[dict(binding=str(args.pdb_binding.resolve()),manifest=str(args.workloads.resolve()),system='pdblend',datasets=list(SLOS))]
    previous=args.previous_binding.resolve() if args.previous_binding else args.pdb_binding.resolve()
    if args.previous_binding:
        pb=read(previous);required.append(dict(binding=str(previous),manifest=str(args.workloads.resolve()),system=pb['system'],datasets=args.previous_dataset or list(SLOS)))
    files={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files.update(historical_files)
    for p in [args.pdb_binding,args.workloads,args.host/'manifest.json',previous,*[Path(i['config']) for i in new]]:files[str(p.resolve())]=sha(p)
    for p,h in read(args.host/'manifest.json')['files'].items():files[str(args.host/p)]=h
    for p,h in read(args.executor/'manifest.json')['files'].items():files[str(args.executor/p)]=h
    files[str(args.executor/'manifest.json')]=sha(args.executor/'manifest.json')
    record=dict(schema=1,protocol_id=PROTOCOL,model=model,layout=args.layout,hostname=prior['hostname'],deadline_s=DEADLINE,
        out=str(out),host_release=str(args.host.resolve()),executor_release=str(args.executor.resolve()),
        instances=new,required_predecessors=required,previous_binding=str(previous),pdb_binding=str(args.pdb_binding.resolve()),
        workloads=str(args.workloads.resolve()),image=IMAGE,files=files,
        supported_datasets=['longbench'] if suffix=='l' else list(SLOS),source_entry=str(entry),
        deployment_budget_s=720,cleanup_budget_s=120,does_not_certify_kv_output_correctness=True)
    write(out/'deployment.json',record);print(json.dumps(dict(deployment=str(out/'deployment.json'),sha256=sha(out/'deployment.json'),hardware_actions=False)))

SLOS={'alpaca':(1.,.1),'sharegpt':(5.,.15),'longbench':(15.,.2)}

async def command(*argv,timeout=60):
    p=await asyncio.create_subprocess_exec(*argv,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:data,_=await asyncio.wait_for(p.communicate(),timeout)
    except BaseException:
        if p.returncode is None:p.kill();await p.wait()
        raise
    require(p.returncode==0,data.decode(errors='replace')[-3000:]);return data.decode()

def docker_start_arguments(i):
    args=['docker','run','-d','--name',i['container_name'],'--gpus','all','--network','host','--ipc','host']
    for e in i['environment']:args+=['-e',e]
    for m in i['mounts']:
        require(m['Type']=='bind','unexpected non-bind historical mount')
        args+=['-v',m['Source']+':'+m['Destination']+('' if m.get('RW') else ':ro')]
    return args+[i['image'],'python3',i['engine_entry'],'--config',i['config']]

def free_gpu_snapshot(hardware):
    rows=[];nvml=hardware._nvml
    for gpu in range(8):
        handle=hardware._handle(gpu)
        mem=nvml.nvmlDeviceGetMemoryInfo(handle,version=nvml.nvmlMemory_v2)
        compute=nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        graphics=nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        rows.append(dict(gpu=gpu,used_bytes=mem.used,reserved_bytes=mem.reserved,free_bytes=mem.free,
            memory_api='nvmlDeviceGetMemoryInfo_v2',compute_pids=[p.pid for p in compute],graphics_pids=[p.pid for p in graphics]))
    return dict(observed_s=time.time(),gpus=rows)

async def await_power_ready(sampler,evidence_fn,timeout_s=5):
    until=time.time()+timeout_s
    while len(sampler.samples)<2 and time.time()<until and not sampler.error:await asyncio.sleep(.05)
    require(len(sampler.samples)>=2 and not sampler.error and
        evidence_fn(sampler.samples,sampler.power_source,sampler.power_metadata)['power_source_verified'],
        'two complete instant eight-GPU power frames are required')

async def stop_creation_intents(intents,deadline_s):
    """All unique owned names are attempted within one shared cleanup bound."""
    async def stop_one(intent):
        try:
            remaining=deadline_s-time.time();require(remaining>0,'owned-container cleanup deadline expired')
            async def inspect_and_stop():
                inspected=json.loads(await command('docker','inspect',intent['name'],timeout=min(5,remaining)))[0]
                intent['observed_container_id']=inspected['Id']
                await command('docker','stop','--time','10',intent['name'],timeout=min(15,max(.01,deadline_s-time.time())))
                intent['stopped']=True
            await asyncio.wait_for(inspect_and_stop(),remaining)
            return None
        except BaseException as exc:
            intent['cleanup_error']=repr(exc)
            return 'stop new failed '+intent['name']+': '+repr(exc)
    return [error for error in await asyncio.gather(*(stop_one(i) for i in intents)) if error]

async def ready(session,i,executor,until):
    """An unchanged owner must really run/ACK; initial -1 is not fabricated."""
    latest=None;initialized=False
    while time.time()<until:
        try:
            raw=await executor.http(session,i,'/runtime',timeout=2)
            if not initialized and raw.get('generation')==0 and raw.get('acknowledged_generation')==-1:
                provenance=await executor.http(session,i,'/provenance',timeout=3)
                cfg=read(i['config'])
                require(provenance.get('instance_id')==i['id'] and provenance.get('tp')==i['tp']
                    and provenance.get('model')==cfg['model'] and provenance.get('source_files_at_import')==
                    {str(p):sha(p) for p in Path(i['engine_entry']).parent.glob('*.py')},
                    'new engine identity differs before initial owner control')
                require(raw.get('id')==i['id'] and not raw.get('error') and not raw.get('runtime_error')
                    and raw.get('transport_healthy') is True and not any(raw.get(k) for k in
                    ('active','running','waiting','kv_allocations','transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives')),
                    'new engine is not idle before initial owner control')
                await executor.http(session,i,'/control',dict(generation=1,role=cfg['role'],mode='continuous',admit_prefill=True,admit_decode=True))
                initialized=True
                continue
            executor.idle(raw,i)
            require(raw.get('accepting') is True,'engine not accepting after startup')
            return raw
        except Exception as exc:latest=repr(exc);await asyncio.sleep(.2)
    raise TimeoutError('engine did not become truly ready: '+str(latest))

async def launch(args):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler,trapezoid_energy
    from ecopadg.serving.measurement import save_raw,power_evidence
    from ecopadg.metrics import clip_power_window
    spec=read(args.spec);verify_files(spec['files'])
    require(spec['hostname']==socket.gethostname(),'wrong physical node')
    require(time.time()+spec['deployment_budget_s']+spec['cleanup_budget_s']<DEADLINE,'insufficient original deadline for deployment and cleanup')
    proof=[terminal_group(r['binding'],r['manifest'],system=r['system'],datasets=r['datasets']) for r in spec['required_predecessors']]
    executor=load('deployment_five_system_executor',Path(spec['executor_release'])/'run.py')
    previous=read(spec['previous_binding']);executor.validate_binding(previous)
    out=Path(spec['out']);require(not (out/'deployment-receipt.json').exists(),'deployment attempt already has a receipt')
    result=dict(started_s=time.time(),complete=False,predecessor_proof=proof,created=[],creation_intents=[],stopped=[],errors=[],hardware_actions=False)
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
    async with aiohttp.ClientSession(trust_env=False) as session:
        result['previous_identity']=await executor.identity(session,previous)
        all_before=json.loads(await command('docker','inspect',*(await command('docker','ps','-aq')).split()))
        write(out/'containers.before.json',all_before)
        existing={x['Name'].lstrip('/') for x in all_before}
        require(not existing.intersection(i['container_name'] for i in spec['instances']),'new container name already exists')
        for i in spec['instances']:
            for port in [i['port'],*range(i['kv_port'],i['kv_port']+32)]:
                with socket.socket() as s:
                    # Probe availability without connecting to a possible NCCL
                    # bootstrap listener or altering its expected handshake.
                    try:s.bind(('127.0.0.1',port))
                    except OSError as exc:raise RuntimeError('new port is occupied: '+str(port)) from exc
        result['sampling_start_s']=time.time();sampler.start()
        try:
            await await_power_ready(sampler,power_evidence)
            require(time.time()+spec['deployment_budget_s']+spec['cleanup_budget_s']<DEADLINE,
                    'read-only preflight consumed the remaining deployment allowance')
            result['operation_start_s']=time.time();result['hardware_actions']=True
            restored=await asyncio.gather(*(executor.restore(session,i) for i in previous['instances']),return_exceptions=True)
            result['previous_native_restore']=[dict(error=repr(x)) if isinstance(x,BaseException) else x for x in restored]
            require(all(isinstance(x,dict) and x.get('complete') for x in restored),'previous native cleanup failed')
            for i in previous['instances']:
                await command('docker','stop','--time','30',i['container']['name'],timeout=40);result['stopped'].append(i['container']['name'])
            free_deadline=time.time()+20
            while True:
                available=await asyncio.to_thread(free_gpu_snapshot,hardware)
                result['gpus_after_stop']=available
                if all(r['used_bytes']==0 and not r['compute_pids'] and not r['graphics_pids'] for r in available['gpus']):break
                require(time.time()<free_deadline,'stopped engines did not free the eight GPUs')
                await asyncio.sleep(.2)
            for i in spec['instances']:
                result['creation_intents'].append(dict(id=i['id'],name=i['container_name'],issued_s=time.time()))
                write(out/'creation-intents'/(i['id']+'.json'),result['creation_intents'][-1])
                cid=(await command(*docker_start_arguments(i),timeout=30)).strip();result['created'].append(dict(id=i['id'],name=i['container_name'],container_id=cid))
            limit=min(result['operation_start_s']+spec['deployment_budget_s'],DEADLINE-spec['cleanup_budget_s'])
            startup=await asyncio.gather(*(ready(session,i,executor,limit) for i in spec['instances']),return_exceptions=True)
            result['startup']={i['id']:dict(error=repr(v)) if isinstance(v,BaseException) else v for i,v in zip(spec['instances'],startup)}
            require(not any(isinstance(x,BaseException) for x in startup),'one or more real engines failed readiness')
            result['new_provenance']={i['id']:await executor.http(session,i,'/provenance') for i in spec['instances']}
            result['new_native_restore']={i['id']:await executor.restore(session,i) for i in spec['instances']}
            require(all(x['complete'] for x in result['new_native_restore'].values()),'new native cleanup failed')
            result['complete']=True
        except BaseException as exc:
            result['errors'].append(repr(exc))
        finally:
            if not result['complete']:
                # A failure leaves new containers stopped and inspectable. It
                # never removes containers or automatically restarts old work.
                result['cleanup_deadline_s']=time.time()+spec['cleanup_budget_s']
                result['errors'].extend(await stop_creation_intents(result['creation_intents'],result['cleanup_deadline_s']))
            result['operation_end_s']=time.time()
            try:await asyncio.sleep(.1);await asyncio.to_thread(sampler.stop)
            except BaseException as exc:result['errors'].append('sampler stop: '+repr(exc))
            power=out/'deployment-power';power.mkdir(exist_ok=True)
            try:
                save_raw(power,[],sampler.samples,sampler.utilization_samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata)
                with (power/'clocks.csv').open('x') as f:
                    w=csv.writer(f);w.writerow(['t_s']+[f'gpu{i}_sm_mhz' for i in range(8)]);w.writerows((t,*v) for t,v in sampler.frequency_samples)
                result['power_evidence']=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
                result['all8_operation_energy_j']=trapezoid_energy(clip_power_window(sampler.samples,result.get('operation_start_s',result['sampling_start_s']),result['operation_end_s'],pad_s=0))
            except BaseException as exc:result['errors'].append('power evidence: '+repr(exc))
            try:write(out/'containers.after.json',json.loads(await command('docker','inspect',*(await command('docker','ps','-aq')).split())))
            except BaseException as exc:result['errors'].append('final inventory: '+repr(exc))
            result.update(finished_s=time.time(),sampling_error=sampler.error,
                measurement_valid=not result['errors'] and not sampler.error and result.get('power_evidence',{}).get('power_source_verified') is True,
                output_correctness_verified=False,scope='physical deployment and native cleanup only; serving correctness gate remains required')
            write(out/'deployment-receipt.json',result)
    require(result['complete'] and result['measurement_valid'],'failed deployment retained; no baseline work may start')
    print(json.dumps(dict(complete=True,receipt=str(out/'deployment-receipt.json'),energy_j=result['all8_operation_energy_j'])))

async def build_binding(args):
    """Read-only live binding; writes only a new baseline config/binding folder."""
    import aiohttp
    spec=read(args.spec);verify_files(spec['files']);receipt=read(args.receipt)
    require(receipt.get('complete') is True and receipt.get('measurement_valid') is True,
            'successful measured deployment receipt required')
    require(args.strategy in ('mixed','distserve','ecoserve','dynamollm','dynamollm-resident'),'unknown baseline')
    require(args.strategy!='dynamollm' or spec['model']=='14b','C has only the historically resident Dynamo implementation')
    require(args.strategy!='dynamollm-resident' or spec['model']=='7b','A historical Dynamo is the full implementation')
    datasets=args.dataset or list(SLOS)
    require(set(datasets)<=set(spec['supported_datasets']),'dataset not supported by this physical layout')
    require(spec['layout']=='resident' or args.strategy=='distserve','heterogeneous layout only implements DistServe')
    require(not (spec['model']=='14b' and args.strategy=='distserve' and spec['layout']=='resident' and 'longbench' in datasets),
            'A LongBench DistServe requires its historical heterogeneous layout')
    require(not args.out.exists(),'new binding directory required')
    executor=load('ac_baseline_binding_executor',Path(spec['executor_release'])/'run.py')
    source=read(spec['pdb_binding']);executor.validate_binding(source)
    original=load('ac_historical_config_bind',WORKSPACE/'campaign/AC-baseline-100s-preparation-v1/bind.py')
    files=dict(spec['files']);files.update(source['files'])
    files[str(args.spec.resolve())]=sha(args.spec);files[str(args.receipt.resolve())]=sha(args.receipt)
    topology=Path(spec['host_release'])/'src/ecopadg/serving/topology.py'
    if args.strategy=='dynamollm':
        require('observation_engine_entry' in topology.read_text() and 'observation_engine_sha256' in topology.read_text(),
                'baseline host lacks explicit hash-bound lifecycle entry')
    by_created={i['name']:i['container_id'] for i in receipt['created']}
    current=json.loads(await command('docker','inspect',*[i['container_name'] for i in spec['instances']]))
    require(set((await command('docker','ps','--format','{{.Names}}')).split())==set(by_created),'unexpected running container during binding')
    current_by={i['Name'].lstrip('/'):i for i in current};instances=[]
    async with aiohttp.ClientSession(trust_env=False) as session:
        for i in spec['instances']:
            c=current_by[i['container_name']]
            require(c['Id']==by_created[i['container_name']] and c['Image']==IMAGE and c['State']['Running'],
                    'deployment process identity differs')
            require(c['Config']['Cmd']==['python3',i['engine_entry'],'--config',i['config']],
                    'actual entry/config differs from the frozen launch')
            p=await executor.http(session,i,'/provenance');cfg=read(i['config'])
            require(p.get('instance_id')==i['id'] and p.get('tp')==i['tp'] and p.get('model')==cfg['model']
                and p.get('dtype')=='bfloat16' and p.get('max_model_len')==8192
                and p.get('cuda_visible_devices')==','.join(map(str,i['gpus']))
                and p.get('source_files_at_import')=={str(f):sha(f) for f in Path(i['engine_entry']).parent.glob('*.py')},
                    'actual new engine/model/source provenance differs')
            raw=await executor.wait_idle(session,i)
            require(raw.get('accepting') is True,'new baseline engine is not accepting')
            record={k:i[k] for k in ('id','tp','gpus','role','port','kv_port','url','native_kind','scheduler_cache_observed')}
            record.update(engine_config=i['config'],container=dict(name=i['container_name'],id=c['Id'],image=c['Image'],StartedAt=c['State']['StartedAt']),
                provenance={k:p[k] for k in ('instance_id','tp','model','dtype','max_model_len','cuda_visible_devices','source_files_at_import')})
            instances.append(record)
    args.out.mkdir(parents=True);configs={};notes={}
    for dataset in datasets:
        historical,_=original.configuration(spec['model'],dataset,args.strategy)
        roles={(i['tp'],tuple(i['gpus'])):i['role'] for i in historical['instances']}
        layout=[dict(id=i['id'],tp=i['tp'],gpus=i['gpus'],port=i['port'],kv_port=i['kv_port'],url=i['url'],
                     container_name=i['container_name'],role=roles[i['tp'],tuple(i['gpus'])]) for i in spec['instances']]
        dep=dict(model=spec['model'],historical_engine_observation_only=True,layouts={args.strategy+':'+dataset:layout},
                 topology_source_path_verified=args.strategy=='dynamollm')
        cfg,note=original.configuration(spec['model'],dataset,args.strategy,deployment=dep,out=args.output/'unused')
        cfg.update(comparison_system='dynamollm' if args.strategy.startswith('dynamollm') else args.strategy,
                   controller_source_release=spec['host_release'])
        if args.strategy=='dynamollm':
            template=read(spec['instances'][0]['config'])
            template.update(observation_engine_entry=spec['source_entry'],observation_engine_sha256=sha(spec['source_entry']))
            template_path=args.out/'engine-template.json'
            if not template_path.exists():write(template_path,template)
            cfg['topology'].update(runtime_dir=str(args.out/'dynamic-runtime'),image=IMAGE,engine_template=str(template_path))
            files[str(template_path.resolve())]=sha(template_path)
        path=args.out/'configs'/(dataset+'.json');write(path,cfg);configs[dataset]=str(path.resolve());files[str(path.resolve())]=sha(path);notes[dataset]=note
        def external(value,key=''):
            if isinstance(value,dict):
                for k,v in value.items():external(v,k)
            elif isinstance(value,list):
                for v in value:external(v,key)
            elif isinstance(value,str) and value.startswith('/') and key!='journal' and Path(value).is_file():files[value]=sha(value)
        external(cfg)
    canonical='dynamollm' if args.strategy.startswith('dynamollm') else args.strategy
    binding=dict(schema=1,protocol_id=PROTOCOL,model=spec['model'],system=canonical,implementation_variant=args.strategy,
        hostname=spec['hostname'],deadline_s=DEADLINE,host_release=spec['host_release'],output=str(args.output.resolve()),
        configs=configs,instances=instances,files=files,large_inputs=source.get('large_inputs',{}),window_s=100,seeds=[701],
        historical_policy=notes,deployment_receipt=str(args.receipt.resolve()),
        correctness_gate_required_before_performance=True,output_correctness_verified=False,formal_eligible=False)
    write(args.out/'binding.json',binding);executor.validate_binding(binding)
    print(json.dumps(dict(binding=str(args.out/'binding.json'),sha256=sha(args.out/'binding.json'),
        system=canonical,variant=args.strategy,datasets=datasets,hardware_actions=False,
        correctness_gate_still_required=True)))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--model',choices=META,required=True);a.add_argument('--layout',choices=('resident','distserve-longbench'),default='resident')
    for k in ('pdb-binding','workloads','host','out'):a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--executor',type=Path,default=WORKSPACE/'campaign/five-system-execution-v3')
    a.add_argument('--previous-binding',type=Path);a.add_argument('--previous-dataset',action='append',choices=SLOS)
    a=sub.add_parser('deploy');a.add_argument('--spec',type=Path,required=True);a.add_argument('--run',action='store_true')
    a=sub.add_parser('bind')
    for k in ('spec','receipt','out','output'):a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--strategy',required=True);a.add_argument('--dataset',action='append',choices=SLOS)
    args=p.parse_args()
    if args.action=='prepare':prepare(args);return
    spec=read(args.spec);package_check();verify_files(spec['files'])
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'independent launch must not inherit an active node lease FD')
    host=Path(spec['host_release']);sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    if args.action=='bind':
        from ecopadg.serving.campaign import node_lease
        with node_lease():asyncio.run(build_binding(args))
        return
    if not args.run:print(json.dumps(dict(spec_valid=True,hardware_actions=False)));return
    from ecopadg.serving.campaign import node_lease
    with node_lease():asyncio.run(launch(args))

if __name__=='__main__':main()
