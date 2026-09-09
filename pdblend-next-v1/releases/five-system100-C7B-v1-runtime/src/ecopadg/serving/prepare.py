"""Prepare an explicitly listed experimental layout and report startup energy.

The manifest names every removed and added instance. Only this experiment's
container prefix and the authorized eight-card node are accepted. This is
between-run preparation; in-run changes use the transactional topology backend.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler,trapezoid_energy
from ecopadg.metrics import clip_power_window
from .backend import ClockOwner,HttpEngineBackend
from .topology import DockerLifecycle,InstanceSpec,TopologyManager,validate_layout
from .campaign import node_lease


async def prepare(manifest,out):
    def spec(x):
        return InstanceSpec(x.get('instance_id',x.get('id')),x['tp'],tuple(x['gpus']),x['port'],
            x['kv_port'],x.get('role','mixed'),x.get('generation',0))
    added=[spec(x) for x in manifest['add']]
    removed=[spec(x) for x in manifest.get('remove',[])]
    kept=[spec(x) for x in manifest.get('keep',[])]
    validate_layout(added+kept,range(8));validate_layout(removed+kept,range(8))
    template=json.loads(Path(manifest['engine_template']).read_text())
    template.update(manifest.get('engine_overrides',{}))
    lifecycle=DockerLifecycle(out/'lifecycle',manifest['image'],template)
    hardware=await asyncio.to_thread(PynvmlBackend)
    clocks=ClockOwner(hardware,range(8));sampler=PowerSampler(range(8),interval=.02,backend=hardware)
    events=[];result={};sampler.start();started=None;engine_ready=None;finished=None
    class Journal:
        async def emit(self,event): events.append(event)
    async def no_commit(*args): raise RuntimeError('preparation cannot commit serving routes')
    async with aiohttp.ClientSession(trust_env=False) as session:
        backend=HttpEngineBackend([s.endpoint() for s in added+kept],session,clocks)
        manager=TopologyManager(backend,lifecycle,added+kept,range(8),Journal(),freeze=no_commit,commit=no_commit)
        try:
            await asyncio.sleep(.1);started=time.time()
            for existing in kept:
                state=await manager.ready(existing)
                if state['running'] or state['waiting'] or state.get('active'):
                    raise RuntimeError('between-run preparation requires drained retained instances')
            for old in removed: await lifecycle.stop(old)
            if added: await clocks.set(sorted({g for s in added for g in s.gpus}),2520)
            peers={s.instance_id:dict(host='127.0.0.1',tp=s.tp,kv_port=s.kv_port) for s in added+kept}
            async def start(spec):
                at=time.time()
                await lifecycle.start(spec,peers,manifest.get('retained_weights'))
                state=await manager.ready(spec)
                return dict(spec=asdict(spec),startup_duration_s=time.time()-at,
                            measured_kv_capacity=state['total_kv_tokens'],
                            measured_runtime_capacity=dict(tp=spec.tp,kv_tokens=state['total_kv_tokens'],
                                transfer_buffer_bytes=state['free_transfer_bytes']+sum(state['transfer_allocations'].values()),
                                transfer_bytes_per_token=state['transfer_bytes_per_token']))
            outcomes=await asyncio.gather(*(start(s) for s in added),return_exceptions=True)
            errors=[repr(x) for x in outcomes if isinstance(x,BaseException)]
            result.update(instances=[x for x in outcomes if not isinstance(x,BaseException)],errors=errors)
            if errors: raise RuntimeError('layout preparation failed: '+str(errors))
            # A phase barrier makes startup and warm-up energy disjoint. Each
            # phase still prepares independent instances concurrently.
            engine_ready=time.time()
            async def warmup(spec):
                at=time.time()
                validation=await manager.verify(spec)
                return dict(instance_id=spec.instance_id,validation=validation,
                            warmup_duration_s=time.time()-at)
            warmed=await asyncio.gather(*(warmup(s) for s in added),return_exceptions=True)
            warmup_errors=[repr(x) for x in warmed if isinstance(x,BaseException)]
            result['errors'].extend(warmup_errors)
            if warmup_errors: raise RuntimeError('layout warm-up failed: '+str(warmup_errors))
            by_id={x['instance_id']:x for x in warmed}
            for value in result['instances']:
                warm=by_id[value['spec']['instance_id']]
                value.update(validation=warm['validation'],warmup_duration_s=warm['warmup_duration_s'])
            for existing in kept:
                for new in added:
                    await manager.request(existing,'/register-peer',dict(id=new.instance_id,peer=peers[new.instance_id]))
            final=added+kept
            for index,spec in enumerate(final):
                if final[index+1:]:
                    await manager.request(spec,'/prepare-peers',dict(peers=[s.instance_id for s in final[index+1:]]))
            finished=time.time()
            result['complete']=True
        finally:
            finished=finished or time.time()
            await asyncio.sleep(.1);await asyncio.to_thread(sampler.stop)
            await clocks.close()
            result.update(purpose='between-run startup and output-work/peer warm-up; not serving energy',
                manifest=manifest,started_s=started,finished_s=finished,
                engine_ready_s=engine_ready,
                startup_duration_s=engine_ready-started if engine_ready and started else None,
                warmup_duration_s=finished-engine_ready if engine_ready else None,
                startup_energy_j=trapezoid_energy(clip_power_window(sampler.samples,started,engine_ready,pad_s=0))
                    if engine_ready and started else None,
                warmup_energy_j=trapezoid_energy(clip_power_window(sampler.samples,engine_ready,finished,pad_s=0))
                    if engine_ready else None,
                duration_s=finished-started if started else None,
                energy_j=trapezoid_energy(clip_power_window(sampler.samples,started,finished,pad_s=0)) if started else None,
                power_samples=sampler.samples,sampling_error=sampler.error)
            await asyncio.to_thread((out/'startup.json').write_text,json.dumps(result))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with node_lease():
        args.out.mkdir(parents=True,exist_ok=False)
        result=asyncio.run(prepare(json.loads(args.manifest.read_text()),args.out))
    print(json.dumps({k:v for k,v in result.items() if k not in ('power_samples','manifest')}))


if __name__=='__main__': main()
