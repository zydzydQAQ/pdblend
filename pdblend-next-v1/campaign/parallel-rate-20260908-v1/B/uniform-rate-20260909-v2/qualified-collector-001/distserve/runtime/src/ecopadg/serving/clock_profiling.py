"""Measure clock settling and all-eight-GPU switching energy under real decode."""
import argparse
import asyncio
import itertools
import json
from pathlib import Path
import time
import uuid

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler,trapezoid_energy
from ecopadg.metrics import clip_power_window
from .backend import ClockOwner
from .campaign import node_lease
from .cli_async import run_async_cli, record_failure, finish_measurement
from .evidence import sha256
from .profiling import HardwareProfiler


async def measure(args):
    topology=json.loads(args.topology.read_text());instance=topology['mixed']
    args.out.mkdir(parents=True,exist_ok=False)
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode=getattr(args,'power_mode','average'))
    clocks=ClockOwner(hardware,range(8))
    sampler=PowerSampler(range(8),interval=.01,backend=hardware,sample_clocks=True)
    raw=dict(complete=False,topology=topology,switches=[],purpose='real decode clock transition costs, all 8 GPUs')
    task=None;owned=[];changed=False;failed=False;sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        async def settled(frequency):
            deadline=time.monotonic()+4;consecutive=0;observed=[]
            while consecutive<3:
                if task and task.done(): raise RuntimeError('decode finished before clock measurement')
                values=await asyncio.to_thread(lambda:[hardware.current_freq(g) for g in instance['gpus']])
                observed.append((time.time(),values))
                consecutive=consecutive+1 if all(abs(v-frequency)<=15 for v in values) else 0
                if time.monotonic()>deadline: raise RuntimeError('requested clock did not settle under real decode')
                if consecutive<3: await asyncio.sleep(.02)
            return observed
        try:
            raw['engine_provenance']=await profiler.provenance()
            state=await profiler.call(instance,'/runtime')
            if any(state.get(k) for k in ('active','running','waiting','transfer_allocations')):
                raise ValueError('clock profile requires drained instances')
            changed=True
            await profiler.control(instance,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
            await clocks.set(instance['gpus'],2520,verify_rise=False)
            body=dict(prompt=[9707,1879,13]*42+[13,13],max_tokens=32,temperature=0,ignore_eos=True,stream=False)
            reference_id=uuid.uuid4().hex;owned.append((instance,reference_id))
            reference=await profiler.call(instance,'/v1/completions',body,reference_id)
            request_id=uuid.uuid4().hex;owned.append((instance,request_id))
            task=asyncio.create_task(profiler.call(instance,'/v1/completions',dict(body,max_tokens=1024),request_id))
            profiler.tasks=[task]
            # The short prompt finishes well before the first measured switch.
            await asyncio.sleep(.5)
            for source,target in itertools.permutations((900,1500,2100,2520),2):
                await clocks.set(instance['gpus'],source,verify_rise=False)
                await settled(source);await asyncio.sleep(.2)
                started=time.time()
                await clocks.set(instance['gpus'],target,verify_rise=False)
                observed=await settled(target);finished=time.time()
                raw['switches'].append(dict(tp=instance['tp'],source_mhz=source,target_mhz=target,
                    started_s=started,finished_s=finished,observed_sm_mhz=observed,
                    commanded_frequencies=dict(clocks.applied)))
            output=await task
            if (len(output['token_ids'])!=output['usage']['completion_tokens'] or len(output['token_ids'])!=1024
                    or output['token_ids'][:32]!=reference['token_ids']):
                raise RuntimeError('clock changes corrupted the prescribed output workload')
            raw.update(complete=True,output_tokens=1024,prefix_matches_reference=True)
        except BaseException as exc:
            failed=True;record_failure(raw,exc);raise
        finally:
            cleanup_error=None
            try:await profiler.cleanup((instance,) if changed else (),cancel_ids=owned if failed else ())
            except BaseException as exc:cleanup_error=exc;record_failure(raw,exc)
            def derive():
                for switch in raw['switches']:
                    switch['energy_j']=trapezoid_energy(clip_power_window(sampler.samples,
                        switch['started_s'],switch['finished_s'],pad_s=0))
            await finish_measurement(raw,args.out,sampler,clocks,derive=derive)
            if cleanup_error:raise cleanup_error
    if not raw['complete'] or sampler.error: raise RuntimeError('clock profiling incomplete')
    digest=sha256(args.out/'raw.json')
    costs=[dict(tp=r['tp'],source_mhz=r['source_mhz'],target_mhz=r['target_mhz'],
        duration_upper_s=1.2*(r['finished_s']-r['started_s']),energy_upper_j=1.2*r['energy_j'],source_sha256=digest)
        for r in raw['switches']]
    (args.out/'frequency_costs.json').write_text(json.dumps(costs,indent=2))
    print(json.dumps(dict(complete=True,costs=costs)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--power-mode',choices=('average','instant'),default='average')
    args=parser.parse_args()
    with node_lease(): run_async_cli(measure(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
