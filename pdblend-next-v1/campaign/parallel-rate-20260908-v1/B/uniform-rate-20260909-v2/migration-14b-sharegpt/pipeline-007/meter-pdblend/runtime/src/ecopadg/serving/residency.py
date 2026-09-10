"""Measure loaded idle/parked power and verify a real idle clock wakeup."""
import argparse
import asyncio
import json
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
from .profiling import HardwareProfiler


async def measure(args):
    topology=json.loads(args.topology.read_text())
    instances=tuple({i['id']:i for i in topology.values()}.values())
    gpus=sorted({g for i in instances for g in i['gpus']})
    hardware=await asyncio.to_thread(PynvmlBackend)
    clocks=ClockOwner(hardware,range(8));sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
    raw=dict(purpose='resident idle power and real clock wakeup; not serving energy comparison',
             topology=topology,windows=[],complete=False)
    args.out.mkdir(parents=True,exist_ok=False);sampler.start()
    task=None;owned=[];target=topology['mixed'];changed=False;failed=False
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        try:
            raw['engine_provenance']=await profiler.provenance()
            for instance in instances:
                state=await profiler.call(instance,'/runtime')
                if state['running'] or state['waiting'] or state.get('active') or state['transfer_allocations']:
                    raise ValueError('resident power measurements require drained instances')
            for frequency in (900,1500,2100,2520,None):
                if frequency is None: await clocks.park(gpus)
                else: await clocks.set(gpus,frequency,verify_rise=False)
                await asyncio.sleep(2)
                started=time.time();await asyncio.sleep(3);finished=time.time()
                raw['windows'].append(dict(frequency_mhz=frequency,parked=frequency is None,
                    started_s=started,finished_s=finished,commanded=dict(clocks.applied)))
            changed=True
            await profiler.control(target,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
            body=dict(prompt=[9707,1879,13]*42+[13,13],max_tokens=32,temperature=0,ignore_eos=True,stream=False)
            await clocks.set(target['gpus'],2520,verify_rise=False)
            reference_id=uuid.uuid4().hex;owned.append((target,reference_id))
            reference=await profiler.call(target,'/v1/completions',body,reference_id)
            await clocks.park(target['gpus']);await asyncio.sleep(2)
            before=time.time();await clocks.set(target['gpus'],1500)
            raw['wakeup']=dict(set_started_s=before,set_finished_s=time.time(),
                               deferred_gpus=list(clocks.deferred))
            request_id=uuid.uuid4().hex;owned.append((target,request_id))
            task=asyncio.create_task(profiler.call(target,'/v1/completions',body,request_id));profiler.tasks=[task]
            while not task.done():
                await clocks.verify_deferred();await asyncio.sleep(.02)
            output=await task
            await clocks.verify_deferred()
            passed=(output['token_ids']==reference['token_ids'] and len(output['token_ids'])==32
                    and all(clocks.applied[g]==1500 for g in target['gpus'])
                    and not set(target['gpus']).intersection(clocks.deferred)
                    and not any(v['at_s']>=before for v in clocks.fallbacks.values()))
            raw['wakeup'].update(passed=passed,finished_s=time.time(),commanded=dict(clocks.applied),
                fallbacks=dict(clocks.fallbacks),output_tokens=len(output['token_ids']),
                outputs_match=output['token_ids']==reference['token_ids'])
            if not passed: raise RuntimeError('idle wakeup frequency or output verification failed')
            raw['complete']=True
        except BaseException as exc:
            failed=True;record_failure(raw,exc);raise
        finally:
            cleanup_error=None
            try:await profiler.cleanup((target,) if changed else (),cancel_ids=owned if failed else ())
            except BaseException as exc:cleanup_error=exc;record_failure(raw,exc)
            def derive():
                raw['residency']=[]
                for window in raw['windows']:
                    samples=clip_power_window(sampler.samples,window['started_s'],window['finished_s'],pad_s=0)
                    raw['residency'].extend(dict(instance_id=i['id'],tp=i['tp'],gpus=i['gpus'],
                        frequency_mhz=window['frequency_mhz'],parked=window['parked'],
                        watts=trapezoid_mean_power([(t,[w[g] for g in i['gpus']]) for t,w in samples])) for i in instances)
            await finish_measurement(raw,args.out,sampler,clocks,derive=derive)
            if cleanup_error:raise cleanup_error
    if sampler.error: raise RuntimeError(sampler.error)
    print(json.dumps(dict(complete=raw['complete'],residency=raw['residency'],wakeup=raw['wakeup'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topology',type=Path,required=True)
    p.add_argument('--runtime-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with node_lease(): run_async_cli(measure(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
