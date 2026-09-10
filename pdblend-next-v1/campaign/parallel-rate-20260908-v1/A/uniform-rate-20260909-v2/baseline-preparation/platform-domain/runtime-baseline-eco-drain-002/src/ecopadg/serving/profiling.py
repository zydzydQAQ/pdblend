"""Hardware profiling of real context and decode batches on prepared engines.

Decode admission is temporarily held while real requests create their KV.
Each prefill/import finishes before the next starts, bounding transfer staging.
The resulting decode batch is released together. Holds are diagnostic setup,
never a serving result; every engine is restored even on failure.
"""
import argparse
import asyncio
import json
import math
from pathlib import Path
import time
import uuid

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from .backend import ClockOwner
from .campaign import node_lease
from .cli_async import run_async_cli, record_failure, finish_measurement, cancel_tasks, cleanup_timeout


def power_sample_interval(value):
    interval=float(value)
    if not math.isfinite(interval) or not .001<=interval<=1:
        raise ValueError('power sample interval must be finite and between 0.001 and 1 seconds')
    return interval


def hold_setup_limit(runs,frequency,n,b,layout):
    """Keep diagnostic KV assembly below the engine's 120 s token timeout.

    This is a harness limit, not a measured serving capacity. Use only this
    frequency's completed real setups; never label excluded batches as data.
    """
    samples=[(r['released_s']-r['started_s'])/r['batch'] for r in runs
             if not r['skipped'] and r['frequency_mhz']==frequency
             and r['input_tokens']==n and r['layout']==layout]
    if samples and b*max(samples)*1.1>100:
        return dict(skipped=True,reason='diagnostic held-batch setup timeout budget',
                    predicted_setup_s=b*max(samples)*1.1,setup_limit_s=100,
                    not_a_serving_capacity_measurement=True)
    return None


class HardwareProfiler:
    def __init__(self,session,topology,runtime_dir):
        self.session=session
        self.topology=topology
        self.runtime_dir=Path(runtime_dir)
        self.tasks=[]
        self.owned={}

    async def call(self,instance,path,body=None,rid=None):
        completion=path=='/v1/completions'
        if completion:
            rid=rid or uuid.uuid4().hex
            self.owned[rid]=instance
        kwargs={'json':body} if body is not None else {}
        if rid: kwargs['headers']={'X-Request-Id':rid}
        method=self.session.post if body is not None else self.session.get
        async with method(f"http://127.0.0.1:{instance['port']}"+path,**kwargs) as response:
            if response.status!=200: raise RuntimeError(await response.text())
            result=await response.json()
        if completion:self.owned.pop(rid,None)
        return result

    async def control(self,instance,**changes):
        state=await self.call(instance,'/runtime')
        if state.get('diagnostic_recompute') or state.get('diagnostic_transport'):
            raise ValueError('disable diagnostic verification before performance profiling')
        payload={key:state[key] for key in ('role','mode','admit_prefill','generation')}
        payload.update(admit_decode=state.get('admit_decode',True))
        payload.update(changes,generation=state['generation']+1)
        return await self.call(instance,'/control',payload)

    async def cleanup(self,instances,*,cancel_ids=()):
        """Independent bounded cancels and admission restore, even after an RPC failure."""
        errors=[]
        try:await cancel_tasks(self.tasks)
        except BaseException as exc:errors.append('local requests: '+repr(exc))
        self.tasks=[]
        owned={rid:instance for instance,rid in cancel_ids}
        owned.update(self.owned)
        async def attempt(label,action,timeout):
            try:await asyncio.wait_for(action(),cleanup_timeout(timeout))
            except BaseException as exc:errors.append(label+': '+repr(exc))
        await asyncio.gather(*(attempt(instance['id']+'/'+rid,
            lambda instance=instance,rid=rid:self.call(instance,'/cancel',dict(request_id=rid)),1.)
            for rid,instance in owned.items()))
        await asyncio.gather(*(attempt(instance['id']+' admission restore',
            lambda instance=instance:self.control(instance,admit_prefill=True,admit_decode=True),1.5)
            for instance in instances))
        if errors:raise RuntimeError('profiling request cleanup incomplete: '+'; '.join(errors))
        for rid in owned:self.owned.pop(rid,None)

    async def provenance(self):
        records=[]
        for instance in {i['id']:i for i in self.topology.values()}.values():
            record=await self.call(instance,'/provenance')
            process=await asyncio.create_subprocess_exec('docker','inspect','--format','{{.Image}}',
                'pdb-v2-'+instance['id'],stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            try: out,error=await asyncio.wait_for(process.communicate(),10)
            except BaseException:
                if process.returncode is None: process.kill();await process.wait()
                raise
            if process.returncode: raise RuntimeError('cannot identify profiling engine image')
            records.append(dict(record,image_id=out.decode().strip()))
        return records

    async def wait_allocated(self,instance,rid,expected_count):
        deadline=time.monotonic()+60
        while True:
            state=await self.call(instance,'/runtime')
            if rid in state['kv_allocations'] and state['running']==expected_count:
                return state
            if time.monotonic()>deadline or any(t.done() for t in self.tasks):
                failures=[repr(t.exception()) for t in self.tasks if t.done() and not t.cancelled()]
                raise RuntimeError('held real request did not retain its KV allocation: '+
                    json.dumps(dict(request_id=rid,expected=expected_count,running=state['running'],
                                    waiting=state['waiting'],task_failures=failures)))
            await asyncio.sleep(.005)

    async def run(self,n,b,output_tokens,layout):
        target=self.topology['mixed' if layout=='mixed' else 'decode']
        source=target if layout=='mixed' else self.topology['prefill']
        state=await self.call(target,'/runtime')
        reserved=((n+output_tokens+15)//16)*16*b
        if state['running'] or state['waiting']:
            raise ValueError('profiling requires drained instances')
        if reserved>state['free_kv_tokens']:
            return dict(skipped=True,reason='measured KV capacity',requested_tokens=reserved,
                        free_kv_tokens=state['free_kv_tokens'])
        if n>source.get('max_num_batched_tokens',8192) or b>target.get('max_num_seqs',32):
            return dict(skipped=True,reason='configured engine batch capacity')
        selected={i['id']:i for i in (source,target)}
        request_ids=[];owned=[];failed=False
        try:
            for instance in selected.values():
                role=('mixed' if layout=='mixed' else 'prefill' if instance is source else 'decode')
                await self.control(instance,role=role,mode='continuous',admit_prefill=True,
                                   admit_decode=instance is not target)
            paths={i:self.runtime_dir/(i+'.control.events.jsonl') for i in selected}
            offsets={i:p.stat().st_size for i,p in paths.items()}
            prompt=([9707,1879,13]*(n//3+1))[:n]
            started=time.time()
            for index in range(b):
                nonce=uuid.uuid4().hex
                rid=nonce if layout=='mixed' else f"pdb:{nonce}:d:{source['id']}:{target['id']}"
                # Register the consumer identity before its producer can send
                # staged KV, including cancellation between P and D startup.
                request_ids.append(rid);owned.append((target,rid))
                if layout=='pd':
                    producer=f"pdb:{nonce}:p:{source['id']}:{target['id']}"
                    owned.append((source,producer))
                    await self.call(source,'/v1/completions',dict(prompt=prompt,max_tokens=1,
                        temperature=0,ignore_eos=True,stream=False),producer)
                task=asyncio.create_task(self.call(target,'/v1/completions',dict(prompt=prompt,
                    max_tokens=output_tokens,temperature=0,ignore_eos=True,stream=False),rid))
                self.tasks.append(task)
                await self.wait_allocated(target,rid,index+1)
            held=await self.call(target,'/runtime')
            if set(held['kv_allocations'])!=set(request_ids):
                raise RuntimeError('held batch identity mismatch')
            released=time.time()
            await self.control(target,admit_decode=True)
            outputs=await asyncio.gather(*self.tasks)
            finished=time.time()
            if any(o['usage']['completion_tokens']!=output_tokens or len(o['token_ids'])!=output_tokens
                   for o in outputs):
                raise RuntimeError('profiling output workload mismatch')
            events=[]
            for instance,path in paths.items():
                def read(path=path,instance=instance):
                    with path.open() as handle:
                        handle.seek(offsets[instance])
                        return [dict(json.loads(line),instance=instance) for line in handle if line.strip()]
                events.extend(await asyncio.to_thread(read))
            decode=[e for e in events if e['instance']==target['id'] and e['decode']==b
                    and e['prefill']==0 and e['tokens']>0]
            if len(decode)!=output_tokens-1:
                raise RuntimeError(f'actual decode batch did not execute all steps: {len(decode)}')
            return dict(skipped=False,started_s=started,released_s=released,finished_s=finished,
                events=events,token_ids=[o['token_ids'] for o in outputs],
                usages=[o['usage'] for o in outputs],held_kv_tokens=held['kv_allocations'],
                prefill_batch=1,decode_batch=b)
        except BaseException:
            failed=True;raise
        finally:
            await self.cleanup(selected.values(),cancel_ids=owned if failed else ())


async def profile(args):
    # Check direct callers as well as argparse before opening any GPU backend.
    interval=power_sample_interval(getattr(args,'power_sample_interval_s',.02))
    topology=json.loads(args.topology.read_text())
    instances={i['id']:i for i in topology.values()}
    mode=getattr(args,'power_mode','average')
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode=mode)
    # NVML's enforced limit is a measured device setting, only a conservative
    # upper bound when short phase power cannot be resolved by the sampler.
    limits=[hardware.power_limit_w(g) for g in range(8)]
    raw=dict(schema=3,measurement='hardware',purpose='held-KV operator profiling',
        topology=topology,runtime_dir=str(args.runtime_dir),power_limit_w=limits,runs=[],complete=False,
        output_tokens=args.output_tokens,profiling_holds_are_not_serving_results=True,
        requested_power_sample_interval_s=interval)
    args.out.mkdir(parents=True,exist_ok=False)
    clocks=ClockOwner(hardware,range(8))
    # Host polling cadence is not the NVML sensor's independent update period.
    sampler=PowerSampler(range(8),interval=interval,backend=hardware,sample_clocks=True)
    sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        try:
            raw['engine_provenance']=await profiler.provenance()
            if 'decode' in topology:
                await profiler.call(topology['prefill'],'/prepare-peers',dict(peers=[topology['decode']['id']]))
            for f in args.frequencies:
                await clocks.set(sorted({g for i in instances.values() for g in i['gpus']}),f,verify_rise=False)
                await asyncio.sleep(.5)
                idle_start=time.time();await asyncio.sleep(.5);idle_end=time.time()
                for n in args.inputs:
                    for b in args.batches:
                        for trial in range(args.repeats):
                            reference=None
                            for layout in (('mixed','pd') if 'decode' in topology else ('mixed',)):
                                result=hold_setup_limit(raw['runs'],f,n,b,layout)
                                if result is None:
                                    result=await profiler.run(n,b,args.output_tokens,layout)
                                result.update(frequency_mhz=f,input_tokens=n,batch=b,trial=trial,layout=layout,
                                    output_tokens=args.output_tokens,idle_start_s=idle_start,idle_end_s=idle_end,
                                    commanded_frequencies=dict(clocks.applied))
                                if not result['skipped']:
                                    if layout=='mixed': reference=result['token_ids']
                                    elif len({i['tp'] for i in topology.values()})==1 and result['token_ids']!=reference:
                                        raise RuntimeError('same-TP PD output differs from matched held mixed reference')
                                raw['runs'].append(result)
                                print(json.dumps(dict(frequency=f,input=n,batch=b,trial=trial,
                                    layout=layout,skipped=result['skipped'])),flush=True)
                await asyncio.to_thread((args.out/'progress.json').write_text,
                    json.dumps(dict(frequency_complete=f,runs=len(raw['runs']))))
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
    p.add_argument('--inputs',type=int,nargs='+',default=[2048,4096,7168])
    p.add_argument('--batches',type=int,nargs='+',default=[1,4,8,16,32])
    p.add_argument('--frequencies',type=int,nargs='+',default=[900,1500,2100,2520])
    p.add_argument('--output-tokens',type=int,default=512)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--power-mode',choices=('average','instant'),default='average')
    p.add_argument('--power-sample-interval-s',type=power_sample_interval,default=.02,
        help='host power polling interval in seconds; does not specify sensor refresh rate')
    args=p.parse_args()
    if (args.output_tokens<2 or args.repeats<1 or min(args.inputs)<1 or min(args.batches)<1
            or max(args.inputs)+args.output_tokens>8192):
        p.error('positive real workloads must fit the 8192-token engine limit')
    with node_lease(): run_async_cli(profile(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
