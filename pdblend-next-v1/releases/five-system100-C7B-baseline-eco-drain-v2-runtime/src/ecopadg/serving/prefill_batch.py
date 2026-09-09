"""Measure real batched prefill and its bounded grouped KV handoff."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler,trapezoid_mean_power
from ecopadg.metrics import clip_power_window
from .backend import ClockOwner
from .campaign import node_lease
from .cli_async import run_async_cli, record_failure, finish_measurement
from .interconnect import InterconnectTopology
from .profiles import ProfilePoint
from .profiling import HardwareProfiler
from .measurement import power_evidence


TRANSFER_ENERGY_ACCOUNTING = 'p_send+d_send+d_import_nonoverlap_above_resident_v2'


def transfer_energy_components(watts,source,target,source_idle_w,target_idle_w,
                               send_window,import_window):
    """Incremental energy for one grouped handoff, with D overlap counted once.

    P's send event also covers destination transport receive/staging work.
    The later D import event covers consumption of those staged tensors.
    Windows use each endpoint's own resident baseline; unrelated GPUs and
    time between these windows do not become per-request transfer energy.
    """
    for start,end in (send_window,import_window):
        if not all(math.isfinite(t) for t in (start,end)) or end<=start:
            raise ValueError('finite, positive transfer energy windows required')
    a,z=send_window;b,y=import_window
    # Import can precede, overlap, contain, or follow the recorded send window.
    # Charge only its two possible pieces outside the already charged D send.
    remaining=[(start,end) for start,end in ((b,min(y,a)),(max(b,z),y)) if end>start]
    def energy(instance,idle,window):
        start,end=window
        return max(0,watts(instance,start,end)-idle)*(end-start)
    p_send=energy(source,source_idle_w,send_window)
    d_send=energy(target,target_idle_w,send_window)
    d_import=sum(energy(target,target_idle_w,window) for window in remaining)
    return dict(send_window_s=list(send_window),import_window_s=list(import_window),
        import_nonoverlap_windows_s=[list(window) for window in remaining],
        source_idle_w=source_idle_w,target_idle_w=target_idle_w,
        source_send_incremental_j=p_send,target_send_incremental_j=d_send,
        target_import_nonoverlap_incremental_j=d_import,
        incremental_j=p_send+d_send+d_import)


async def case(profiler,n,b):
    source,target=profiler.topology['prefill'],profiler.topology['decode']
    pstate=await profiler.call(source,'/runtime');dstate=await profiler.call(target,'/runtime')
    deadline=time.monotonic()+2
    while dstate.get('transfer_allocations'):
        if time.monotonic()>deadline: raise RuntimeError('previous transfer allocation has not drained')
        await asyncio.sleep(.02);dstate=await profiler.call(target,'/runtime')
    if any(s['running'] or s['waiting'] for s in (pstate,dstate)):
        raise ValueError('phase batch profiling requires drained engines')
    if (b*n>source.get('max_num_batched_tokens',8192) or b>source.get('max_num_seqs',32)
            or b*((n+16)//16)*16>min(pstate['free_kv_tokens'],dstate['free_kv_tokens'])
            or b*n*dstate['transfer_bytes_per_token']>dstate['free_transfer_bytes']):
        return dict(skipped=True,reason='measured KV/staging or configured phase batch capacity')
    paths=[profiler.runtime_dir/(source['id']+'.control.events.jsonl')]
    paths.extend(profiler.runtime_dir/(i['id']+'.control.json.kv.0.jsonl') for i in (source,target))
    offsets={str(p):p.stat().st_size if p.exists() else 0 for p in paths}
    nonces=[uuid.uuid4().hex for _ in range(b)]
    pids=[f"pdb:{nonce}:p:{source['id']}:{target['id']}" for nonce in nonces]
    dids=[f"pdb:{nonce}:d:{source['id']}:{target['id']}" for nonce in nonces]
    body=dict(prompt=([9707,1879,13]*(n//3+1))[:n],temperature=0,ignore_eos=True,stream=False)
    failed=False
    try:
        await profiler.control(source,role='prefill',mode='continuous',admit_prefill=False,admit_decode=True)
        await profiler.control(target,role='decode',mode='continuous',admit_prefill=True,admit_decode=False)
        profiler.tasks=[asyncio.create_task(profiler.call(source,'/v1/completions',
            dict(body,max_tokens=1),rid)) for rid in pids]
        deadline=time.monotonic()+30
        while (await profiler.call(source,'/runtime'))['waiting']!=b:
            if time.monotonic()>deadline or any(t.done() for t in profiler.tasks):
                raise RuntimeError('prefill batch was not queued behind the engine gate')
            await asyncio.sleep(.01)
        released=time.time();await profiler.control(source,admit_prefill=True)
        pout=await asyncio.gather(*profiler.tasks);profiler.tasks=[]
        for index,rid in enumerate(dids):
            profiler.tasks.append(asyncio.create_task(profiler.call(target,'/v1/completions',
                dict(body,max_tokens=2),rid)))
            await profiler.wait_allocated(target,rid,index+1)
        await profiler.control(target,admit_decode=True)
        dout=await asyncio.gather(*profiler.tasks)
        if any(len(p['token_ids'])!=1 or len(d['token_ids'])!=2 or p['token_ids'][0]!=d['token_ids'][0]
               for p,d in zip(pout,dout)):
            raise RuntimeError('batched prefill KV handoff lost or changed a token')
        def read(path):
            with path.open() as handle:
                handle.seek(offsets[str(path)])
                return [json.loads(line) for line in handle if line.strip()]
        events,send,receive=await asyncio.gather(*(asyncio.to_thread(read,path) for path in paths))
        steps=[e for e in events if set(e['request_ids']).intersection(pids) and e['tokens']]
        if len(steps)!=1 or steps[0]['prefill']!=b or steps[0]['decode']:
            raise RuntimeError('configured prefill batch was not executed as one real step')
        send=[e for e in send if set(e['request_ids']).intersection(pids)]
        receive=[e for e in receive if set(e['request_ids']).intersection(dids)]
        if (len(send)!=1 or set(send[0]['request_ids'])!=set(pids) or len(receive)!=b
                or {rid for e in receive for rid in e['request_ids']}!=set(dids)):
            raise RuntimeError('missing grouped transfer evidence')
        return dict(skipped=False,released_s=released,finished_s=time.time(),step=steps[0],
                    send=send,receive=receive,output_matches=True)
    except BaseException:
        failed=True;raise
    finally:
        owned=[*( (source,rid) for rid in pids),*( (target,rid) for rid in dids)] if failed else ()
        await profiler.cleanup((source,target),cancel_ids=owned)


def build(raw,digest,topology):
    if not raw.get('complete') or raw.get('sampling_error') or not raw.get('frequency_samples'):
        raise ValueError('complete phase measurements with hardware clock samples required')
    power_proof=power_evidence(raw.get('power_samples',[]),raw.get('power_source'),raw.get('power_metadata'))
    if not power_proof['power_source_verified']:
        raise ValueError('short transfer costs require verified eight-GPU instant power evidence')
    source,target=raw['topology']['prefill'],raw['topology']['decode']
    if any(raw.get('commanded_frequencies',{}).get(str(g),
            raw.get('commanded_frequencies',{}).get(g))!=frequency
            for instance,frequency in ((source,2520),(target,raw.get('decode_frequency_mhz',2520)))
            for g in instance['gpus']):
        raise ValueError('phase measurement clock command differs from profile frequency')
    def watts(instance,start,end):
        rows=clip_power_window(raw['power_samples'],start,end,pad_s=0)
        return trapezoid_mean_power([(t,[w[g] for g in instance['gpus']]) for t,w in rows])
    pi=watts(source,raw['idle_start_s'],raw['idle_end_s'])
    di=watts(target,raw['idle_start_s'],raw['idle_end_s'])
    points=[];links=[];energy_components=[]
    for r in raw['runs']:
        if r['skipped']: continue
        start=r['step']['started_s'];end=min(e['started_s'] for e in r['send'])
        power=watts(source,start,end)
        if end-start<.2 or power<pi: power=sum(raw['power_limit_w'][g] for g in source['gpus'])
        points.append(asdict(ProfilePoint('prefill',source['tp'],2520,r['input_tokens'],r['input_tokens']+1,
            r['batch'],end-start,0,power,pi,.1,1,digest,prefill_power_w=power)))
        intervals=[(min(e['started_s'] for e in r[key]),max(e['finished_s'] for e in r[key]))
                   for key in ('send','receive')]
        energy=transfer_energy_components(watts,source,target,pi,di,*intervals)
        energy_components.append(dict(input_tokens=r['input_tokens'],profile_batch=r['batch'],**energy))
        links.append(dict(source_tp=source['tp'],target_tp=target['tp'],max_input_tokens=r['input_tokens'],
            seconds_upper=1.1*sum(z-a for a,z in intervals),incremental_j=energy['incremental_j'],
            import_seconds_upper=1.1*(intervals[1][1]-intervals[1][0]),
            source_sha256=digest,validated=True,profile_batch=r['batch'],
            decode_frequency_mhz=raw['decode_frequency_mhz'],
            source_gpus=source['gpus'],target_gpus=target['gpus'],
            interconnect_class=topology.link_class(source['gpus'],target['gpus']),
            topology_sha256=topology.source_sha256))
    return dict(schema=2,measurement='hardware',model='Qwen2.5-14B-Instruct',points=points,
                purpose='actual one-step prefill batching',source_sha256=digest,
                power_source=raw['power_source'],power_source_verified=True,
                frequency_commands_verified=True,frequency_samples_source_sha256=digest,
                transfer_energy_accounting=TRANSFER_ENERGY_ACCOUNTING,
                transfer_energy_components=energy_components),links


async def measure(args):
    topology=json.loads(args.topology.read_text())
    if 'decode' not in topology: raise ValueError('two prepared instances required')
    # This dedicated tool has not inherited profiling's legacy average mode:
    # all new short handoff costs use the explicit instantaneous NVML field.
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    clocks=ClockOwner(hardware,range(8));sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
    raw=dict(schema=1,topology=topology,runs=[],complete=False,
             power_limit_w=[hardware.power_limit_w(g) for g in range(8)],
             decode_frequency_mhz=args.decode_frequency)
    args.out.mkdir(parents=True,exist_ok=False);sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        try:
            raw['engine_provenance']=await profiler.provenance()
            await clocks.set(sorted({g for i in topology.values() for g in i['gpus']}),2520,verify_rise=False)
            await clocks.set(topology['decode']['gpus'],args.decode_frequency,verify_rise=False)
            raw['commanded_frequencies']=dict(clocks.applied)
            await asyncio.sleep(.3);raw['idle_start_s']=time.time()
            await asyncio.sleep(.5);raw['idle_end_s']=time.time()
            if sampler.error or len(sampler.samples)<2:
                raise RuntimeError('instant transfer power preflight failed: '+str(sampler.error))
            for n in args.inputs:
                for b in args.batches:
                    result=await case(profiler,n,b)
                    result.update(input_tokens=n,batch=b);raw['runs'].append(result)
                    print(json.dumps(dict(input=n,batch=b,skipped=result['skipped'])),flush=True)
            raw['complete']=True
        except BaseException as exc:
            record_failure(raw,exc);raise
        finally:
            await finish_measurement(raw,args.out,sampler,clocks)
    if sampler.error: raise RuntimeError(sampler.error)
    digest=hashlib.sha256((args.out/'raw.json').read_bytes()).hexdigest()
    table,links=build(raw,digest,InterconnectTopology.parse(args.interconnect.read_text()))
    (args.out/'profiles.json').write_text(json.dumps(table,indent=2))
    (args.out/'transfers.json').write_text(json.dumps(links,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topology',type=Path,required=True)
    p.add_argument('--runtime-dir',type=Path,required=True)
    p.add_argument('--interconnect',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=int,nargs='+',default=[128,2048,7168])
    p.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--decode-frequency',type=int,choices=(900,1500,2100,2520),default=2520)
    args=p.parse_args()
    if min(args.inputs)<1 or max(args.inputs)+2>8192 or min(args.batches)<1:
        p.error('positive phase batches fitting the engine required')
    with node_lease(): run_async_cli(measure(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
