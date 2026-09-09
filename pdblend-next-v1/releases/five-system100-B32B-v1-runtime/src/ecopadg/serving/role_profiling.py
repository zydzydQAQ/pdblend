"""Lightweight resident role costs; existing runtime validation proves correctness.

No generation requests are sent. Each of the six directed role changes is
measured twice on one explicitly prepared TP1/2/4 instance. Whole-node instant
power is retained, while admission uses a separately labeled power-limit
energy bound for these short control intervals.
"""
import argparse
import asyncio
import itertools
import json
import math
from pathlib import Path
import signal
import time

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler,trapezoid_energy
from ecopadg.metrics import clip_power_window
from .backend import ClockOwner
from .campaign import node_lease
from .evidence import sha256
from .measurement import power_evidence
from .profiling import HardwareProfiler


ROLES=('mixed','prefill','decode')
CONTROL=('role','mode','admit_prefill','admit_decode')
EMPTY=('active','running','waiting','kv_allocations','transfer_allocations')
PURPOSE='resident role cost calibration'
ENERGY_BOUND='maximum of 1.2 times observed instant joules and eight-GPU enforced power limits times duration upper bound'


def verify_change(before,after,desired):
    if (any(before.get(k) or after.get(k) for k in EMPTY)
            or any(after.get(k)!=v for k,v in desired.items())
            or after.get('generation')!=before.get('generation',-2)+1
            or after.get('acknowledged_generation')!=after.get('generation')
            or not after.get('accepting') or after.get('runtime_error') or after.get('error')):
        raise RuntimeError('role change lacks drained KV, exact next generation or engine acknowledgement')


def costs(raw,digest):
    """Rebuild costs from real confirmations, not a pre-written passed flag."""
    if (not raw.get('complete') or not raw.get('passed') or raw.get('errors')
            or raw.get('sampling_error') or raw.get('purpose')!=PURPOSE
            or not power_evidence(raw.get('power_samples',[]),raw.get('power_source'),
                                  raw.get('power_metadata'))['power_source_verified']):
        raise ValueError('complete instant resident role evidence required')
    limits=raw.get('power_limit_w',[])
    if len(limits)!=8 or any(not math.isfinite(x) or x<=0 for x in limits):
        raise ValueError('eight measured enforced power limits required')
    instance=raw['instance'];tp=instance['tp']
    if tp not in (1,2,4) or len(instance['gpus'])!=tp:
        raise ValueError('only explicitly measured TP1/2/4 resident role costs supported')
    if raw.get('provenance_before')!=raw.get('provenance_after') or not raw.get('provenance_before'):
        raise ValueError('resident engine source, image or process changed')
    provenance=raw['provenance_before']
    if (len(provenance)!=1 or provenance[0].get('instance_id')!=instance['id']
            or provenance[0].get('tp')!=tp
            or provenance[0].get('cuda_visible_devices')!=','.join(map(str,instance['gpus']))):
        raise ValueError('role cost TP or physical instance differs from its real engine')
    restore=raw['restoration'];verify_change(restore['before'],restore['after'],raw['original_control'])
    groups={key:[] for key in itertools.permutations(ROLES,2)}
    if len(raw['switches'])!=12: raise ValueError('six directed changes with exactly two repetitions required')
    for switch in raw['switches']:
        key=(switch['source_role'],switch['target_role'])
        if key not in groups or switch['before']['role']!=key[0] or switch['tp']!=tp:
            raise ValueError('role observation identity mismatch')
        verify_change(switch['before'],switch['after'],dict(role=key[1],mode='continuous',admit_prefill=True,admit_decode=True))
        duration=switch['finished_s']-switch['started_s']
        if not math.isfinite(duration) or duration<=0: raise ValueError('invalid role control duration')
        energy=trapezoid_energy(clip_power_window(raw['power_samples'],switch['started_s'],switch['finished_s'],pad_s=0))
        if not math.isclose(energy,switch['energy_j'],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('role sampled energy differs from its raw instant integral')
        groups[key].append((duration,energy,switch['trial']))
    result=[]
    for (source,target),values in groups.items():
        if len(values)!=2 or {v[2] for v in values}!={0,1}: raise ValueError('missing independent role repetitions')
        duration=1.2*max(v[0] for v in values)
        result.append(dict(tp=tp,source_role=source,target_role=target,time_upper_s=duration,
            energy_upper_j=max(1.2*max(v[1] for v in values),sum(limits)*duration),source_sha256=digest))
    return result


async def measure(args):
    topology=json.loads(args.topology.read_text())
    instance=next((i for i in topology.values() if i['id']==args.instance),None) if args.instance else topology.get('prefill')
    if not instance or instance['tp'] not in (1,2,4): raise ValueError('explicit prepared TP1/2/4 instance required')
    args.out.mkdir(parents=True,exist_ok=False)
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant')
    limits=await asyncio.to_thread(lambda:[hardware.power_limit_w(g) for g in range(8)])
    clocks=ClockOwner(hardware,range(8));sampler=PowerSampler(range(8),interval=.01,backend=hardware,sample_clocks=True)
    raw=dict(schema=1,purpose=PURPOSE,instance=instance,switches=[],complete=False,passed=False,errors=[],
        power_limit_w=limits,energy_upper_method=ENERGY_BOUND,
        energy_j_semantics='diagnostic trapezoid integral of instantaneous samples, not exact short-window energy',
        correctness_scope='generation acknowledgement and empty KV only; complete runtime failure tests remain separate')
    sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10),trust_env=False) as session:
        profiler=HardwareProfiler(session,{instance['id']:instance},args.runtime_dir)
        async def change(desired):
            before=await profiler.call(instance,'/runtime')
            if any(before.get(k) for k in EMPTY): raise RuntimeError('role profiling requires a drained instance')
            started=time.time();await profiler.control(instance,**desired);finished=time.time()
            after=await profiler.call(instance,'/runtime');verify_change(before,after,desired)
            return dict(before=before,after=after,started_s=started,finished_s=finished)
        try:
            original=await profiler.call(instance,'/runtime');raw['original_control']={k:original[k] for k in CONTROL}
            raw['provenance_before']=await profiler.provenance()
            await clocks.set(instance['gpus'],2520,verify_rise=False)
            await asyncio.sleep(.2)
            if sampler.error or len(sampler.samples)<2: raise RuntimeError('instant role power preflight failed')
            for source,target in itertools.permutations(ROLES,2):
                for trial in (0,1):
                    await change(dict(role=source,mode='continuous',admit_prefill=True,admit_decode=True))
                    result=await change(dict(role=target,mode='continuous',admit_prefill=True,admit_decode=True))
                    raw['switches'].append(dict(result,tp=instance['tp'],source_role=source,target_role=target,trial=trial))
                    await asyncio.sleep(.025)
            raw['complete']=True
        except BaseException as exc:
            raw['errors'].append(type(exc).__name__+': '+str(exc))
        finally:
            try:
                if raw.get('original_control'): raw['restoration']=await change(raw['original_control'])
                raw['provenance_after']=await profiler.provenance()
            except BaseException as exc: raw['errors'].append('restoration: '+repr(exc))
            await asyncio.sleep(.025);await asyncio.to_thread(sampler.stop)
            try: await clocks.close()
            except BaseException as exc: raw['errors'].append('clock restoration: '+repr(exc))
            raw.update(power_samples=sampler.samples,power_source=sampler.power_source,power_metadata=sampler.power_metadata,
                sampling_error=sampler.error,frequency_samples=sampler.frequency_samples)
            raw['power_source_verified']=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)['power_source_verified']
            raw['passed']=(raw['complete'] and not raw['errors'] and not sampler.error and raw['power_source_verified']
                           and raw.get('provenance_before')==raw.get('provenance_after'))
            for switch in raw['switches']:
                try: switch['energy_j']=trapezoid_energy(clip_power_window(sampler.samples,switch['started_s'],switch['finished_s'],pad_s=0))
                except Exception as exc:
                    raw['errors'].append('partial power window: '+repr(exc));raw['passed']=False
            await asyncio.to_thread((args.out/'raw.json').write_text,json.dumps(raw,indent=2,allow_nan=False))
    result=costs(raw,sha256(args.out/'raw.json'))
    (args.out/'role_costs.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return dict(passed=True,costs=len(result),tp=instance['tp'],power_source_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',type=Path,required=True);parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--instance')
    args=parser.parse_args()
    async def cancellable():
        loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
        try:return await measure(args)
        finally:loop.remove_signal_handler(signal.SIGTERM)
    with node_lease():result=asyncio.run(cancellable())
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
