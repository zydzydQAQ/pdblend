"""Frozen B32B two-TP2 budget8192 screening; isolated child, native v3 cleanup."""
from __future__ import annotations
import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
BINDING=json.loads((ROOT/'binding.json').read_text())
HOST=Path(BINDING['host_release'])
PROFILE=Path(BINDING['profile'])
ENGINE=ROOT.parents[1]/'releases/io-v3-runtime'
CONFIG=ROOT/'inputs/controller.fixed.json'
IMAGE='sha256:fd4ba34686c028ec6ba0ae17220f833b24c2f45f29535f735066f5a7a27004c2'
IDS=('nextv3b0','nextv3b1')
PORTS=(33500,33501)
NAMES=('pdb-v2-nextv3b0','pdb-v2-nextv3b1')
EVENT_ROOT=ROOT.parent/'B32B-engine-v3-candidate-v2/runtime'
RESIDUAL=('active','running','waiting','kv_allocations','transfer_allocations',
          'transfer_buffered_tensors','transfer_inflight_receives','transfer_inflight_sends')


def require(ok,reason):
    if not ok:raise RuntimeError(reason)


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(name,obj):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def stat_identity(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def package_check():
    manifest=read(ROOT/'package-manifest.json')
    for p,h in manifest['files'].items():require(sha(ROOT/p)==h,'package changed: '+p)
    require(sha(HOST/'manifest.json')==BINDING['host_manifest_sha256'] and sha(PROFILE)==BINDING['profile_sha256'],'bound host/profile changed')
    spec=read(ROOT/'runspec.json');source=read(ROOT/'inputs/source-sweep-manifest.json')
    require(sha(ROOT/'inputs/source-sweep-manifest.json')==spec['source_manifest_sha256'],'trace manifest changed')
    from contract import validate
    validate(spec,source,read(ROOT/'source-contract.json'),read,sha)
    cfg=read(CONFIG)
    require(cfg['strategy']=='pdblend-joint' and cfg['allow_pd'] is False and cfg['dynamic_pools'] is False
        and cfg['slow_topology'] is False and cfg.get('topology') is None,'must keep fixed mixed-only B2 strategy')
    require(cfg['evaluation_protocol']=='evaluation-v3' and cfg['power_mode']=='instant' and cfg['node_gpus']==list(range(8)),'wrong measurement protocol')
    require(cfg['slo_attainment_target']==.9 and cfg['slo_ttft_s']==5 and cfg['slo_tpot_s']==.1,'fixed SLO changed')
    require([i['id'] for i in cfg['instances']]==list(IDS) and [i['gpus'] for i in cfg['instances']]==[[0,1],[2,3]],'B2 instances changed')
    require(all(i['tp']==2 and i['role']=='mixed' for i in cfg['instances']),'wrong B2 roles/TP')
    require(cfg['controller_source_release']==str(HOST) and cfg['engine_source_release']==str(ENGINE),'wrong frozen releases')
    require(cfg['scheduler_budget_ablation']==dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32),'fixed budget changed')
    original=read(ROOT/'inputs/controller.source.json');original.update(controller_source_release=str(HOST),
        measurement_window_protocol='per-dataset-slo-fixed-window-v2',arrival_window_s=300,slo_scale=1,
        profiles=str(PROFILE))
    require(cfg==original,'policy differs from declared uniform fixed-window/decode-phase candidate')
    require(not cfg.get('prediction_cap_to_max_tokens') and not cfg.get('output_limit_aware_prediction')
        and cfg['output_prior']==211,'uniform prior/estimator differs')
    import fixed_queue
    fixed_queue.validate_spec(sys.modules[__name__],spec)
    return spec


def native_idle(raw,instance_id,*,accepting=None,tokens=None):
    require(raw.get('id')==instance_id,'wrong runtime instance')
    require(not raw.get('error') and not raw.get('runtime_error') and raw.get('transport_healthy') is True,'unhealthy v3 engine')
    require(all(k in raw for k in RESIDUAL) and not any(raw[k] for k in RESIDUAL),'request/KV/receive residue or missing observation')
    require(raw.get('generation')==raw.get('acknowledged_generation') and type(raw.get('generation')) is int,'actual owner ACK missing')
    now=time.time()
    require(all(type(raw.get(k)) in (int,float) and math.isfinite(raw[k]) and 0<=now-raw[k]<=1 for k in ('timestamp','transfer_observed_s')),'stale actual owner/transfer observation')
    caches=[io.get('controls',{}).get('runtime') for io in raw.get('scheduler_io',[])]
    require(len(caches)==1 and caches[0] and caches[0].get('generation')==raw['generation'] and caches[0].get('error') is None,'owner/cache ACK differs')
    if accepting is not None:require(raw.get('accepting') is accepting,'wrong admission state')
    generation=raw['generation']
    require(raw.get('acknowledged_generations')==[generation] and raw.get('observed_control_generation')==generation
        and raw.get('scheduler_budget_pending') is None,'pending or unobserved scheduler control')
    actual=raw.get('scheduler_budget_effective',{})
    require(actual.get('max_num_seqs')==32 and actual.get('max_num_batched_tokens') in (8192,2048)
        and (tokens is None or actual['max_num_batched_tokens']==tokens),'budget mismatch')
    require(raw.get('transfer_send_counters_observed') is True and raw.get('transfer_inflight_sends_observed') is True
        and raw.get('transfer_send_healthy') is True,'unobserved/unhealthy send counters')
    counts=[raw.get(k) for k in ('transfer_send_started','transfer_send_completed','transfer_send_failed')]
    require(all(type(c) is int and c>=0 for c in counts) and counts[0]==counts[1] and counts[2]==0,'unsettled or failed sends')


def native_barrier(before,proof):
    require(proof.get('drained') is True and proof.get('accepting') is False and proof.get('generation')==before['generation']+1
        and proof.get('drain_proof_type')=='synchronous_put_owner_barrier'
        and proof.get('send_counters_verified') is True,'missing actual v3 owner/rank barrier')
    ranks=proof.get('transfers');require(isinstance(ranks,list) and len(ranks)==2,'TP2 rank proof absent')
    for rank in ranks:
        require(all(k in rank for k in ('buffered_tensors','inflight_receives','inflight_sends','listener_alive'))
            and rank['buffered_tensors']==rank['inflight_receives']==rank['inflight_sends']==0
            and rank['listener_alive'] is True and rank.get('send_counters_observed') is True
            and rank.get('send_healthy') is True,'v3 rank residue/unhealthy/unobserved sender')
        counts=[rank.get(k) for k in ('send_started','send_completed','send_failed')]
        require(all(type(c) is int and c>=0 for c in counts) and counts[0]==counts[1] and counts[2]==0,'rank sends not settled')
        require(isinstance(rank.get('allocations'),dict) and not rank['allocations']
            and type(rank.get('buffered_gpu_bytes')) is int and rank['buffered_gpu_bytes']==0,'v3 rank KV residue/unobserved bytes')


async def command(*args,timeout=30):
    p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:out,_=await asyncio.wait_for(p.communicate(),timeout)
    except BaseException:
        if p.returncode is None:p.kill();await p.wait()
        raise
    require(p.returncode==0,out.decode(errors='replace')[-2000:]);return out.decode()


async def http(session,index,path,body=None,timeout=12):
    import aiohttp
    async with session.request('GET' if body is None else 'POST',f'http://127.0.0.1:{PORTS[index]}'+path,
            json=body,timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        text=await r.text();require(r.status==200,path+': '+text[:1000]);return json.loads(text)


def process_identity(row):
    return {**{k:row[k] for k in ('Id','Name','Image')},'StartedAt':row['State']['StartedAt'],'Running':row['State']['Running']}


async def inventory():
    ids=(await command('docker','ps','-aq')).split();require(ids,'no container inventory')
    return json.loads(await command('docker','inspect',*ids))


async def live(session,*,expected_inventory=None,accepting=True,tokens=8192):
    rows=await inventory();by_name={r['Name'].lstrip('/'):r for r in rows}
    original={r['Name'].lstrip('/'):r for r in read(ROOT/'inputs/containers.reference.json')}
    require({r['Name'].lstrip('/') for r in rows if r['State']['Running']}==set(NAMES),'unexpected running container')
    if expected_inventory is not None:
        require(sorted(map(process_identity,rows),key=lambda r:r['Id'])==expected_inventory,'a container changed/restarted since prepare')
    release=read(ENGINE/'manifest.json')
    expected_source={str(ENGINE/p):h for p,h in release['files'].items() if p.startswith('src/ecopadg/serving/') and p.endswith('.py')}
    reference=read(ROOT/'inputs/model-source.reference.json')
    gpu_rows=list(csv.reader((await command('nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits')).splitlines()))
    gpu_identity=[dict(index=int(r[0].strip()),uuid=r[1].strip(),name=r[2].strip()) for r in gpu_rows]
    require(gpu_identity==reference['node_gpu_identity'],'physical GPU identity changed')
    evidence=[]
    for n,name in enumerate(NAMES):
        row=by_name[name];prior=original[name]
        require(row['Id']==prior['Id'] and row['State']['StartedAt']==prior['State']['StartedAt'] and row['Image']==IMAGE,'original B2 process changed')
        p=await http(session,n,'/provenance')
        require(p.get('instance_id')==IDS[n] and p.get('tp')==2 and p.get('model')=='/models/Qwen2.5-32B-Instruct'
            and p.get('cuda_visible_devices')==f'{2*n},{2*n+1}','live model/TP/GPU/id mismatch')
        require(p.get('source_files_at_import')==expected_source,'live imported engine source changed')
        env={s.partition('=')[0]:s.partition('=')[2] for s in row['Config'].get('Env',[]) if '=' in s}
        require(env.get('CUDA_VISIBLE_DEVICES')==f'{2*n},{2*n+1}' and row['HostConfig'].get('DeviceRequests')==prior['HostConfig'].get('DeviceRequests'),'container GPU binding changed')
        require({k:v for k,v in env.items() if k.startswith('NCCL_')}==
            {s.partition('=')[0]:s.partition('=')[2] for s in prior['Config'].get('Env',[]) if s.startswith('NCCL_')},'communication environment changed')
        patches=reference['live_vllm_and_serving_source_sha256']
        script='import hashlib,json,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in '+repr(list(patches))+'}))'
        actual=json.loads(await command('docker','exec',name,'python3','-c',script))
        require(actual==patches,'loaded serving/vLLM patch source changed')
        changes=(await command('docker','diff',name)).splitlines()
        require(not any(line.split(' ',1)[-1].endswith('.py') and '/vllm/' in line for line in changes),'vLLM Python layer changed')
        r=await http(session,n,'/runtime');native_idle(r,IDS[n],accepting=accepting,tokens=tokens)
        require(r['role']=='mixed' and r['mode']=='continuous','B2 engine mode changed')
        evidence.append(dict(container=row,provenance=p,runtime=r))
    return dict(inventory=rows,instances=evidence,native_send_counter_support=True,gpu_identity=gpu_identity,
                drain_method='actual v3 synchronous PUT owner plus all TP2 rank send-counter barrier')


def dependencies(evidence):
    paths={};large=set()
    def add(path):
        p=Path(path);require(p.is_file(),'missing external input '+str(p));paths[str(p)]=sha(p)
    for release in (ENGINE,HOST):
        add(release/'manifest.json')
        for name,h in read(release/'manifest.json')['files'].items():
            add(release/name);require(paths[str(release/name)]==h,'frozen release mismatch')
    cfg=read(CONFIG)
    def visit(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif isinstance(value,str) and value.startswith('/') and key not in ('journal','engine_source_release','controller_source_release','host_source_release'):
            add(value)
    visit(cfg)
    # Freeze actual model and retained-weight bytes against the successful A observation.
    for path,h in read(ROOT/'inputs/model-source.reference.json')['model_files_sha256'].items():
        add(path);require(paths[path]==h,'model input differs from verified A source');large.add(path)
    for instance in evidence['instances']:
        args=instance['container']['Args'];require('--config' in args,'engine config argument missing')
        config_path=Path(args[args.index('--config')+1]);add(config_path);engine_cfg=read(config_path)
        require(engine_cfg['tp']==2 and engine_cfg['model']=='/models/Qwen2.5-32B-Instruct','wrong engine config model')
        expected=ROOT/'inputs'/('engine-'+str(engine_cfg['port']-33500)+'.json')
        require(paths[str(config_path)]==sha(expected),'original engine config changed')
        retained=engine_cfg.get('retained_weights')
        if retained:
            root=Path(retained);require(root.is_dir(),'retained weight directory absent')
            files=sorted(p for p in root.rglob('*') if p.is_file());require(files,'retained input files absent')
            for p in files:add(p);large.add(str(p))
    for path,digest in read(ROOT/'inputs/protected-history.json').items():
        add(path);require(paths[path]==digest,'historical B artifact changed: '+path)
    gates=read(ROOT/'inputs/correctness-source.json')
    for path,digest in gates['source_sha256'].items():
        add(path);require(paths[path]==digest,'correctness source changed: '+path)
    gate=read(gates['continuous_gate'])
    require(gate.get('passed') and gate.get('measurement_valid') and all(x['exact_equal'] for x in gate['cross_replica_outputs']),'scoped continuous gate not passed')
    require(read(gates['full_runtime_gate']).get('passed') is False,'original temporal failure lost')
    protected=read(ROOT/'inputs/protected-baselines.json')
    require(protected.get('model')=='32b' and len(protected['files'])==91,'wrong baseline protection evidence')
    for path,digest in protected['files'].items():
        add(path);require(paths[path]==digest,'protected old file changed '+path)
    for row in read(ROOT/'runspec.json')['cells']:add(row['trace'])
    add(ROOT.parent/'deadline-24h-v1/protocol.json')
    return dict(files=paths,large_inputs={p:dict(sha256=paths[p],stat=stat_identity(p)) for p in sorted(large)},
                protected_files=protected['files'])


def frozen_check(full=True):
    package_check();freeze=read(ROOT/'freeze.json')
    require(freeze['package_manifest_sha256']==sha(ROOT/'package-manifest.json'),'frozen package identity changed')
    for path,h in freeze['dependencies']['files'].items():
        if not full and path in freeze['dependencies']['large_inputs']:
            require(stat_identity(path)==freeze['dependencies']['large_inputs'][path]['stat'],'resident weight input changed during sweep')
        else:require(sha(path)==h,'frozen external file changed '+path)
    return freeze


async def prepare():
    import aiohttp
    package_check();require(not (ROOT/'freeze.json').exists(),'prepare already frozen; do not overwrite')
    async with aiohttp.ClientSession(trust_env=False) as session:
        evidence=await live(session)
        deps=await asyncio.to_thread(dependencies,evidence)
        after=await live(session,expected_inventory=sorted(map(process_identity,evidence['inventory']),key=lambda r:r['Id']))
        with socket.socket() as s:s.bind(('127.0.0.1',read(CONFIG)['port']))
    write('prepare.before.json',evidence);write('prepare.after.json',after)
    write('freeze.json',dict(schema=1,prepared_s=time.time(),no_control_actions=True,
        package_manifest_sha256=sha(ROOT/'package-manifest.json'),dependencies=deps,
        inventory=sorted(map(process_identity,evidence['inventory']),key=lambda r:r['Id']),
        engine_release=str(ENGINE),controller_release=str(HOST),image=IMAGE,
        policy_config_sha256=sha(CONFIG),slo_protocol='per-dataset-slo-v1',source_contract_sha256=sha(ROOT/'source-contract.json'),formal_eligible=False,saturation_verified=False))


async def cli(args):
    task=asyncio.current_task();interrupted=False
    def stop():
        nonlocal interrupted
        if not interrupted:interrupted=True;task.cancel()
    for sig in (signal.SIGINT,signal.SIGTERM):asyncio.get_running_loop().add_signal_handler(sig,stop)
    if args.action=='prepare':await prepare()
    else:
        import fixed_queue
        from execution import Cell
        await fixed_queue.sweep(sys.modules[__name__],Cell,phase=args.phase,max_cells=args.max_cells)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('check','prepare','run'),default='check')
    p.add_argument('--phase',choices=('probe','main','scale'),default='probe');p.add_argument('--max-cells',type=int,default=1);args=p.parse_args()
    if args.action=='check':package_check();print(json.dumps(dict(package_valid=True,cells=len(read(ROOT/'runspec.json')['cells']),gpu_executed=False)));return
    from ecopadg.serving.campaign import node_lease
    with node_lease():asyncio.run(cli(args))


if __name__=='__main__':main()
