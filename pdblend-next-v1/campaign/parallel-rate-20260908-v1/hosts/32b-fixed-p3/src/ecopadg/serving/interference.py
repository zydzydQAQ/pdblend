"""Measure an arriving prefill's delay to already-running decode requests."""
import argparse
import asyncio
import itertools
import json
from pathlib import Path
import statistics
import time
import uuid

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from .backend import ClockOwner
from .campaign import node_lease
from .cli_async import run_async_cli, record_failure, finish_measurement
from .profiling import HardwareProfiler


def measured_grid(tables,tp,output_tokens,cross_inputs=()):
    grid={(p['frequency_mhz'],p['input_tokens'],p['batch']-1,
           max(1,p['context_tokens']-output_tokens)) for table in tables for p in table['points']
          if p['role']=='mixed' and p['tp']==tp and p['batch']>1}
    if cross_inputs:
        grid={(f,n,b,context) for f,_,b,context in grid for n in cross_inputs}
    return sorted(grid)


def interference_interval(events,background_ids,probe_id):
    prefill=[e for e in events if probe_id in e['request_ids'] and e['prefill']]
    if len(prefill)!=1: raise ValueError('requires one observed probe prefill')
    p=prefill[0]
    decode=[e for e in events if e['decode']
            and set(e['request_ids']).intersection(background_ids)]
    before=[e for e in decode if e['finished_s']<=p['started_s']]
    # Continuous vLLM can execute old decode tokens in the same step as the
    # new prefill. Those tokens arrive at that step's end, not one step later.
    after=[e for e in decode if e['started_s']>=p['started_s']]
    if len(before)<3 or not after:
        raise ValueError('probe must interrupt a live decode with before/after observations')
    before.sort(key=lambda e:e['finished_s']);after.sort(key=lambda e:e['started_s'])
    tail=before[-12:]
    spacing=[b['finished_s']-a['finished_s'] for a,b in zip(tail,tail[1:])]
    baseline=statistics.median(spacing)
    gap=after[0]['finished_s']-before[-1]['finished_s']
    return dict(prefill_s=p['finished_s']-p['started_s'],baseline_iteration_s=baseline,
                interrupted_token_interval_s=gap,incremental_delay_s=max(0,gap-baseline))


async def run_case(profiler,instance,path,n,b,context,out,probe_after_tokens=16):
    state=await profiler.call(instance,'/runtime')
    needed=b*((context+out+15)//16)*16+((n+16+15)//16)*16
    if state['running'] or state['waiting']: raise ValueError('interference case requires drained engine')
    if needed>state['free_kv_tokens']:
        return dict(skipped=True,reason='measured KV capacity')
    offset=path.stat().st_size
    def events():
        with path.open() as handle:
            handle.seek(offset)
            return [json.loads(line) for line in handle if line.strip()]
    background=[];probe=uuid.uuid4().hex;failed=False
    def body(length,output):
        return dict(prompt=([9707,1879,13]*(length//3+1))[:length],max_tokens=output,
                    temperature=0,ignore_eos=True,stream=False)
    started=time.time()
    try:
        await profiler.control(instance,role='mixed',mode='continuous',admit_prefill=True,admit_decode=False)
        for index in range(b):
            rid=uuid.uuid4().hex;background.append(rid)
            profiler.tasks.append(asyncio.create_task(profiler.call(instance,'/v1/completions',
                body(context,out),rid)))
            await profiler.wait_allocated(instance,rid,index+1)
        await profiler.control(instance,admit_decode=True)
        deadline=time.monotonic()+30
        while True:
            steps=await asyncio.to_thread(events)
            if sum(e['decode']==b and e['prefill']==0 for e in steps)>=probe_after_tokens: break
            if time.monotonic()>deadline or any(t.done() for t in profiler.tasks):
                raise RuntimeError('background decode did not reach the probe point')
            await asyncio.sleep(.01)
        profiler.tasks.append(asyncio.create_task(profiler.call(instance,'/v1/completions',body(n,16),probe)))
        outputs=await asyncio.gather(*profiler.tasks)
        if any(o['usage']['completion_tokens']!=count or len(o['token_ids'])!=count
               for o,count in zip(outputs,[out]*b+[16])):
            raise RuntimeError('interference case output workload mismatch')
        observed=await asyncio.to_thread(events)
        return dict(skipped=False,started_s=started,finished_s=time.time(),background_ids=background,
            probe_id=probe,events=observed,token_ids=[o['token_ids'] for o in outputs],
            **interference_interval(observed,background,probe))
    except BaseException:
        failed=True;raise
    finally:
        await profiler.cleanup((instance,),cancel_ids=[(instance,rid) for rid in [*background,probe]] if failed else ())


async def measure(args):
    topology=json.loads(args.topology.read_text());instance=topology['mixed']
    hardware=await asyncio.to_thread(PynvmlBackend)
    clocks=ClockOwner(hardware,range(8));sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
    args.out.mkdir(parents=True,exist_ok=False)
    raw=dict(schema=1,purpose='measured mixed prefill interference, not a serving comparison',
        topology=topology,background_context=args.context,background_output=args.output_tokens,
        probe_after_tokens=args.probe_after_tokens,runs=[],complete=False)
    sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        path=args.runtime_dir/(instance['id']+'.control.events.jsonl')
        try:
            raw['engine_provenance']=await profiler.provenance()
            if args.profile_grid:
                paths=[args.profile_grid,*args.extra_profile_grid]
                grid=measured_grid([json.loads(path.read_text()) for path in paths],instance['tp'],
                                   args.output_tokens,args.cross_inputs)
                raw['profile_grid']=str(args.profile_grid)
                raw['extra_profile_grid']=[str(path) for path in args.extra_profile_grid]
            else:
                grid=[(f,n,b,args.context) for f,n,b in itertools.product(args.frequencies,args.inputs,args.batches)]
            previous=None
            for f,n,b,context in grid:
                if f!=previous:
                    await clocks.set(instance['gpus'],f,verify_rise=False);previous=f
                result=await run_case(profiler,instance,path,n,b,context,args.output_tokens,
                                      args.probe_after_tokens)
                result.update(frequency_mhz=f,input_tokens=n,background_batch=b,tp=instance['tp'],
                              background_context=context,background_output=args.output_tokens,
                              commanded_frequencies=dict(clocks.applied))
                raw['runs'].append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in ('events','token_ids','background_ids')}),flush=True)
            raw['complete']=True
        except BaseException as exc:
            record_failure(raw,exc);raise
        finally:
            await finish_measurement(raw,args.out,sampler,clocks)
    if sampler.error: raise RuntimeError(sampler.error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topology',type=Path,required=True)
    p.add_argument('--runtime-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frequencies',type=int,nargs='+',default=[900,1500,2100,2520])
    p.add_argument('--inputs',type=int,nargs='+',default=[128,7168])
    p.add_argument('--batches',type=int,nargs='+',default=[1,4])
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--output-tokens',type=int,default=96)
    p.add_argument('--probe-after-tokens',type=int,default=16)
    p.add_argument('--profile-grid',type=Path,
        help='measure each multi-request mixed profile at its late context and exact background batch')
    p.add_argument('--extra-profile-grid',type=Path,nargs='*',default=[],
        help='include independently measured supplemental training tables in the same grid')
    p.add_argument('--cross-inputs',type=int,nargs='+',
        help='measure incoming lengths independently of resident decode context')
    args=p.parse_args()
    if (args.output_tokens<32 or min(args.inputs)<1 or min(args.batches)<1 or args.context<1
            or args.context+args.output_tokens>8192 or max(args.inputs)+16>8192):
        p.error('background must survive the probe and all requests must fit the engine')
    if not 3<=args.probe_after_tokens<args.output_tokens-8:
        p.error('probe needs earlier token intervals and an unfinished background decode')
    if args.cross_inputs and (min(args.cross_inputs)<1 or max(args.cross_inputs)+16>8192):
        p.error('cross-context probe inputs must fit the engine')
    with node_lease(): run_async_cli(measure(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
