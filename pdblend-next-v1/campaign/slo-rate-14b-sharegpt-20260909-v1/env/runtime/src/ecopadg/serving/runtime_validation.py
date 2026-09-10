"""Real role/version/cancellation checks and measured resident switch costs."""
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
from .cli_async import run_async_cli, record_failure, finish_measurement, cleanup_timeout, write_json
from .evidence import sha256
from .profiling import HardwareProfiler


CONTROL_FIELDS = ('role', 'mode', 'admit_prefill', 'admit_decode')


def verify_role_rollback(before, after, attempted_generation):
    """Check the engine's versioned rollback, including scheduler confirmation.

    A failed commit consumes the attempted version. Its rollback must use the
    next version so a delayed command cannot reinstate the rejected role.
    """
    if attempted_generation != before['generation'] + 1:
        raise RuntimeError('rollback check did not submit the next generation')
    if any(after.get(key) != before[key] for key in CONTROL_FIELDS):
        raise RuntimeError('failed role change did not restore control state')
    expected_generation = attempted_generation + 1
    if (after.get('generation') != expected_generation
            or after.get('acknowledged_generation') != expected_generation):
        raise RuntimeError('rollback generation was not confirmed by the engine')
    if after.get('runtime_error') or after.get('error') or not after.get('accepting'):
        raise RuntimeError('rolled-back engine did not resume healthy admission')
    fields = CONTROL_FIELDS + ('generation', 'acknowledged_generation')
    return dict(before={key: before.get(key) for key in fields},
                after={key: after.get(key) for key in fields},
                attempted_generation=attempted_generation,
                expected_rollback_generation=expected_generation)


async def validate(args):
    topology=json.loads(args.topology.read_text());a,b=topology['prefill'],topology['decode']
    if a['tp']!=b['tp']: raise ValueError('role validation needs two instances of the same TP')
    args.out.mkdir(parents=True,exist_ok=False)
    hardware=await asyncio.to_thread(PynvmlBackend);clocks=ClockOwner(hardware,range(8))
    sampler=PowerSampler(range(8),interval=.01,backend=hardware,sample_clocks=True)
    raw=dict(complete=False,passed=False,topology=topology,checks={},switches=[],
             purpose='forced resident role and failure-path validation; not energy optimization evidence')
    sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        original={};pending_transfers={}
        async def drained(instance):
            deadline=time.monotonic()+10
            while True:
                state=await profiler.call(instance,'/runtime')
                if not any(state.get(k) for k in ('active','running','waiting','kv_allocations','transfer_allocations')):
                    return state
                if time.monotonic()>deadline: raise RuntimeError('request or KV did not drain')
                await asyncio.sleep(.01)
        async def reject(instance,path,body,rid=None):
            headers={'X-Request-Id':rid} if rid else {}
            async with session.post(f"http://127.0.0.1:{instance['port']}"+path,json=body,headers=headers) as response:
                text=await response.text()
                if response.status not in (400,409):
                    raise RuntimeError(f'expected explicit rejection, received {response.status}: {text}')
                return dict(status=response.status,reason=text)
        body=dict(prompt=[9707,1879,13]*32,max_tokens=64,temperature=0,ignore_eos=True,stream=False)
        try:
            raw['engine_provenance']=await profiler.provenance()
            await clocks.set(sorted(set(a['gpus']+b['gpus'])),2520,verify_rise=False)
            for instance in (a,b):
                original[instance['id']]=await drained(instance)
                await profiler.control(instance,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
            reference=await profiler.call(a,'/v1/completions',body)
            comparison=await profiler.call(b,'/v1/completions',body)
            if reference['token_ids']!=comparison['token_ids'] or len(reference['token_ids'])!=64:
                raise RuntimeError('same-TP ordinary references differ')
            for source,target in itertools.permutations(('mixed','prefill','decode'),2):
                for trial in range(3):
                    await profiler.control(a,role=source)
                    await asyncio.sleep(.03)
                    start=time.time();ack=await profiler.control(a,role=target);end=time.time()
                    raw['switches'].append(dict(source_role=source,target_role=target,tp=a['tp'],trial=trial,
                        started_s=start,finished_s=end,generation=ack['generation']))
            await profiler.control(a,role='prefill');await profiler.control(b,role='decode')
            await profiler.call(a,'/prepare-peers',dict(peers=[b['id']]))
            nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{a['id']}:{b['id']}";did=f"pdb:{nonce}:d:{a['id']}:{b['id']}"
            pending_transfers[did]=b
            await profiler.call(a,'/v1/completions',dict(body,max_tokens=1),pid)
            raw['checks']['duplicate_prefill']=await reject(a,'/v1/completions',dict(body,max_tokens=1),pid)
            before=await profiler.call(b,'/runtime')
            proposal={k:before[k] for k in ('role','mode','admit_prefill','admit_decode','generation')}
            proposal.update(role='mixed',generation=before['generation']+1)
            raw['checks']['residual_kv_role_rejected']=await reject(b,'/control',proposal)
            after=await profiler.call(b,'/runtime')
            raw['checks']['residual_kv_role_rollback']=verify_role_rollback(before,after,proposal['generation'])
            raw['checks']['rejected_role_replay']=await reject(b,'/control',proposal)
            replayed=await profiler.call(b,'/runtime')
            verify_role_rollback(before,replayed,proposal['generation'])
            output=await profiler.call(b,'/v1/completions',body,did)
            pending_transfers.pop(did)
            if output['token_ids']!=reference['token_ids']: raise RuntimeError('role-switched PD output differs')
            raw['checks']['pd_output_tokens_equal']=True
            raw['checks']['duplicate_decode']=await reject(b,'/v1/completions',body,did)
            # A completed producer can leave real received tensors without a
            # consumer. Cancellation must drop those tensors on every rank.
            nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{a['id']}:{b['id']}";did=f"pdb:{nonce}:d:{a['id']}:{b['id']}"
            pending_transfers[did]=b
            await profiler.call(a,'/v1/completions',dict(body,max_tokens=1),pid)
            cancelled=await profiler.call(b,'/cancel',dict(request_id=did))
            if any(s['buffered_tensors'] for s in cancelled['transfers']): raise RuntimeError('cancel leaked received tensors')
            await drained(b);pending_transfers.pop(did);raw['checks']['cancel_drained_all_ranks']=True
            await profiler.control(b,role='mixed')
            state=await profiler.call(b,'/runtime')
            command={k:state[k] for k in ('generation','role','mode','admit_prefill','admit_decode')}
            raw['checks']['out_of_order_version']=await reject(b,'/control',dict(command,generation=state['generation']+2))
            if await profiler.call(b,'/control',command)!=command: raise RuntimeError('idempotent acknowledgement differs')
            raw['checks']['idempotent_acknowledgement']=True
            async with session.post(f"http://127.0.0.1:{b['port']}/v1/completions",
                    json=dict(body,max_tokens=512,stream=True)) as response:
                if response.status!=200: raise RuntimeError('disconnect request did not start')
                await response.content.readline()
            await drained(b);raw['checks']['client_disconnect_drained']=True
            reference_after=await profiler.call(b,'/v1/completions',body)
            if reference_after['token_ids']!=reference['token_ids']: raise RuntimeError('failure path corrupted next request')
            raw['checks']['next_request_unchanged']=True
            longer=dict(body,prompt=body['prompt']*2)
            second_reference=await profiler.call(b,'/v1/completions',longer)
            timeline=args.runtime_dir/(b['id']+'.control.events.jsonl');offset=timeline.stat().st_size
            first_id=uuid.uuid4().hex;second_id=uuid.uuid4().hex
            try:
                await profiler.control(b,mode='temporal',admit_prefill=True)
                first=asyncio.create_task(profiler.call(b,'/v1/completions',body,first_id))
                profiler.tasks=[first]
                await profiler.wait_allocated(b,first_id,1)
                await profiler.control(b,admit_prefill=False)
                second=asyncio.create_task(profiler.call(b,'/v1/completions',longer,second_id))
                profiler.tasks.append(second)
                await asyncio.sleep(.1)
                held=await profiler.call(b,'/runtime')
                if second_id in held['kv_allocations'] or held['waiting']<1:
                    raise RuntimeError('prefill executed during the decode window')
                await profiler.control(b,admit_prefill=True)
                outputs=await asyncio.gather(*profiler.tasks)
                if [o['token_ids'] for o in outputs]!=[reference['token_ids'],second_reference['token_ids']]:
                    raise RuntimeError('temporal execution changed output tokens')
            finally:
                await profiler.cleanup((b,))
            def read_events():
                with timeline.open() as handle:
                    handle.seek(offset)
                    return [json.loads(line) for line in handle if line.strip()]
            events=await asyncio.to_thread(read_events)
            if (not any(e['prefill'] for e in events) or not any(e['decode'] for e in events)
                    or any(e['prefill'] and e['decode'] for e in events)):
                raise RuntimeError('engine timeline does not establish temporal exclusion')
            raw['temporal_events']=events
            raw['checks']['temporal_executed_steps']=sum(e['tokens']>0 for e in events)
            raw['checks']['temporal_overlap_steps']=0
            after=await profiler.provenance()
            if after!=raw['engine_provenance']: raise RuntimeError('resident transition restarted or changed an engine')
            raw['checks']['same_resident_processes']=True
            raw.update(complete=True,passed=True)
        except BaseException as exc:
            record_failure(raw,exc);raise
        finally:
            try:
                try:write_json(args.out/'raw.json',dict(raw,complete=False,passed=False,cleanup_complete=False))
                except BaseException as exc:
                    raw.setdefault('cleanup_errors',[]).append('partial raw: '+repr(exc));record_failure(raw,exc)
                try:await profiler.cleanup((),cancel_ids=[(instance,rid) for rid,instance in pending_transfers.items()])
                except BaseException as exc:
                    raw.setdefault('cleanup_errors',[]).append(repr(exc));record_failure(raw,exc)
                async def restore(instance):
                    state=original.get(instance['id'])
                    if state is None:return
                    async def work():
                        await drained(instance)
                        await profiler.control(instance,**{k:state[k] for k in CONTROL_FIELDS})
                        restored=await drained(instance)
                        if (any(restored.get(k)!=state[k] for k in CONTROL_FIELDS)
                                or restored.get('generation')!=restored.get('acknowledged_generation')):
                            raise RuntimeError('original controls were not acknowledged')
                    try:await asyncio.wait_for(work(),cleanup_timeout(2.))
                    except BaseException as exc:
                        raw.setdefault('restore_errors',[]).append(instance['id']+': '+repr(exc));record_failure(raw,exc)
                await asyncio.gather(*(restore(instance) for instance in (a,b)))
            except BaseException as exc:
                record_failure(raw,exc);raise
            finally:
                def derive():
                    for switch in raw['switches']:
                        switch['energy_j']=trapezoid_energy(clip_power_window(sampler.samples,switch['started_s'],switch['finished_s'],pad_s=0))
                await finish_measurement(raw,args.out,sampler,clocks,derive=derive)
    if not raw['passed']: raise RuntimeError('runtime validation failed')
    costs=[];digest=sha256(args.out/'raw.json')
    for source,target in itertools.permutations(('mixed','prefill','decode'),2):
        values=[s for s in raw['switches'] if (s['source_role'],s['target_role'])==(source,target)]
        costs.append(dict(tp=a['tp'],source_role=source,target_role=target,
            time_upper_s=1.2*max(s['finished_s']-s['started_s'] for s in values),
            energy_upper_j=1.2*max(s['energy_j'] for s in values),source_sha256=digest))
    (args.out/'role_costs.json').write_text(json.dumps(costs,indent=2))
    print(json.dumps(dict(passed=True,checks=raw['checks'],role_costs=costs)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with node_lease(): run_async_cli(validate(args),failure_path=args.out/'interrupted.json')


if __name__=='__main__': main()
