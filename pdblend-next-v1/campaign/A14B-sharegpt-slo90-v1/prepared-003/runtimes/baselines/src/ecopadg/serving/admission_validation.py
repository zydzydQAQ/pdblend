"""Real mixed-instance KV boundary and recovery; no capacity is fabricated.

The temporary decode hold only assembles real KV. Production backend snapshots
are read after decode resumes, so the hold itself cannot explain rejection.
The first phase checks mixed admission. A second phase checks the actual
DistServe P->D path and reserves target KV/staging before starting its producer.
No GPU clock is written. All owned requests and original controls are restored.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time
import uuid

import aiohttp

from .campaign_followup_setup import read,write
from .calibration import implementation_sources
from .calibration_setup import current_engine_sources,docker,physical,spec
from .backend import HttpEngineBackend
from .distserve import DistServeScheduler
from .evidence import freeze_files,sha256,validate_freeze
from .mechanism_evidence import engine_provenance
from .planner import JointPlanner,TransferCost
from .interconnect import InterconnectTopology
from .profiles import ProfileStore
from .profiling import HardwareProfiler
from .state import StateManager
from .types import RequestBudget


INPUT=7168
OUTPUT=16
HELD_OUTPUT=512
MAX_HELD=6
CONTROL=('role','mode','admit_prefill','admit_decode')


def reservation(input_tokens,output_tokens):return ((input_tokens+output_tokens+15)//16)*16


def can_add_hold(raw,count):
    if raw.get('error') or raw.get('runtime_error') or raw.get('waiting'):
        raise ValueError('engine is unhealthy or still scheduling its previous held prefill')
    return count<MAX_HELD and raw['free_kv_tokens']>=reservation(INPUT,1)


def capacity_rejection(snapshot,request,planner,now):
    if len(snapshot.instances)!=1:raise ValueError('one explicitly owned real instance required')
    instance=snapshot.instances[0]
    if (not instance.accepting or now-instance.timestamp_s>1 or instance.running<1
            or instance.free_kv_tokens>=reservation(request.input_tokens,request.output_limit)
            or not instance.kv_allocations):
        raise ValueError('rejection must use a live accepting instance with genuinely insufficient measured KV')
    plan=planner.plan(snapshot,(request,),now=now,joint=False)
    if plan.feasible or plan.routes:raise RuntimeError('planner admitted a request exceeding actual remaining KV')
    return plan


def spatial_rejection(snapshot,request,policy,source,target,now):
    states={i.instance_id:i for i in snapshot.instances};p,d=states[source],states[target]
    if (not p.accepting or not d.accepting or (p.role,d.role)!=('prefill','decode')
            or any(now-i.timestamp_s>1 for i in (p,d)) or not d.kv_allocations
            or d.free_kv_tokens>=reservation(request.input_tokens,request.output_limit)
            or p.free_kv_tokens<reservation(request.input_tokens,1)
            or d.running>=policy.limits['decode']):
        raise ValueError('actual P/D capacity boundary is not isolated from holds or stage batch limits')
    plan=policy.plan(snapshot,(request,),now=now)
    if plan.feasible or plan.routes:raise RuntimeError('DistServe admitted insufficient real target KV')
    return plan


async def reserve_then_produce(manager,plan,budget,producer):
    """The real shared reservation must succeed before any producer request."""
    reserved=await manager.reserve(plan,time.time(),budget)
    if not reserved:raise RuntimeError('cannot forward a producer without confirmed target reservation')
    target=next(i for i in manager.snapshot.instances if i.instance_id==plan.routes[0].decode_id)
    observation=dict(at_s=time.time(),reserved_kv_tokens=target.reserved_kv_tokens,
        reserved_transfer_bytes=target.reserved_transfer_bytes,snapshot=asdict(manager.snapshot))
    if (target.reserved_kv_tokens<plan.routes[0].reserve_tokens or
            target.reserved_transfer_bytes<plan.routes[0].transfer_reserve_bytes):
        raise RuntimeError('target KV or transfer staging reservation was incomplete')
    output=await producer()
    return observation,output


async def drained(profiler,instance,timeout=20):
    deadline=time.monotonic()+timeout
    while True:
        raw=await profiler.call(instance,'/runtime')
        if not any(raw.get(k) for k in ('active','running','waiting','kv_allocations','transfer_allocations')):
            return raw
        if time.monotonic()>deadline:raise TimeoutError('owned validation requests did not drain')
        await asyncio.sleep(.02)


async def staging_drained(profiler,instance,timeout=5):
    deadline=time.monotonic()+timeout
    while True:
        raw=await profiler.call(instance,'/runtime')
        if not raw.get('transfer_allocations'):return raw
        if time.monotonic()>deadline:raise TimeoutError('previous imported KV staging did not release')
        await asyncio.sleep(.01)


async def cancel_requests(profiler,owned,tasks,*,rpc_timeout=10.,task_timeout=2.):
    """A failed remote cancel must never skip other RPCs or local task cleanup."""
    pending=dict(owned);report=dict(request_ids=list(pending),results={},errors=[],at_s=time.time())
    async def cancel(rid,instance):
        try:
            result=await asyncio.wait_for(profiler.call(instance,'/cancel',dict(request_id=rid)),rpc_timeout)
            if result.get('cancelled')!=rid:raise RuntimeError('cancel acknowledgement has a different request ID')
            report['results'][rid]=result;owned.pop(rid,None)
        except BaseException as exc:
            # Keep unconfirmed ownership so final cleanup can retry it.
            error=f'{instance["id"]}/{rid}: {type(exc).__name__}: {exc}'
            report['results'][rid]=dict(error=error);report['errors'].append(error)
    try:
        await asyncio.gather(*(cancel(rid,instance) for rid,instance in pending.items()))
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        if tasks:
            done,left=await asyncio.wait(tasks,timeout=task_timeout)
            # Retrieve expected cancellation/request errors, even if the remote
            # RPC failed. Do not wait unboundedly for a noncooperative task.
            await asyncio.gather(*done,return_exceptions=True)
            if left:
                report['errors'].append(f'{len(left)} local request tasks did not cancel within the cleanup bound')
                for task in left:
                    task.cancel()
                    task.add_done_callback(lambda t:None if t.cancelled() else t.exception())
    return report


async def cleanup_instances(profiler,owned,tasks,originals,*,deadline,rpc_timeout=10.,restore_timeout=25.):
    """Bound all hardware cleanup; restore independent instances concurrently."""
    result=dict(cancellation=None,restored={},errors=[],started_s=time.time())
    remaining=lambda:max(.001,deadline-time.monotonic())
    try:
        result['cancellation']=await cancel_requests(profiler,owned,tasks,
            rpc_timeout=min(rpc_timeout,remaining()/3),task_timeout=min(2.,remaining()/3))
        result['errors'].extend(result['cancellation']['errors'])
    except BaseException as exc:
        result['errors'].append('request cleanup: '+type(exc).__name__+': '+str(exc))
    async def restore(instance,original):
        if original is None:return
        identifier=instance['id']
        async def work():
            await drained(profiler,instance,timeout=min(8.,remaining()))
            await profiler.control(instance,**{k:original[k] for k in CONTROL})
            state=await drained(profiler,instance,timeout=min(8.,remaining()))
            if (any(state.get(k)!=original[k] for k in CONTROL) or state.get('generation')!=state.get('acknowledged_generation')
                    or state.get('runtime_error') or state.get('error')):
                raise RuntimeError('original controls did not restore with a healthy generation acknowledgement')
            result['restored'][identifier]=state
        try:
            await asyncio.wait_for(work(),min(restore_timeout,remaining()))
        except BaseException as exc:
            result['errors'].append(identifier+' restoration: '+type(exc).__name__+': '+str(exc))
    # No restore exception can prevent the other instance's attempt.
    await asyncio.gather(*(restore(instance,original) for instance,original in originals))
    result['finished_s']=time.time();return result


async def event_file(instance,campaign_root):
    values=json.loads(await docker('inspect','pdb-v2-'+instance['id']))
    if len(values)!=1:raise ValueError('unexpected validation container identity')
    command=values[0]['Config']['Cmd'];config=read(command[command.index('--config')+1])
    actual=dict(config,gpus=list(map(int,dict(v.split('=',1) for v in values[0]['Config']['Env'])['CUDA_VISIBLE_DEVICES'].split(','))))
    if physical(spec(actual))!=physical(spec(instance)):raise ValueError('validation container differs from explicit physical instance')
    path=Path(config['runtime_dir']).resolve()/(instance['id']+'.control.events.jsonl')
    if not path.is_relative_to(Path(campaign_root).resolve()):raise ValueError('engine timeline outside this campaign')
    return path


def verify_raw(raw):
    if raw.get('errors') or raw.get('cleanup_errors') or not raw.get('original_control') or not raw.get('restoration'):
        raise ValueError('KV validation or control restoration incomplete')
    original=raw['original_control'];final=raw['restoration']
    if any(final.get(k)!=original[k] for k in CONTROL) or final.get('generation')!=final.get('acknowledged_generation'):
        raise ValueError('original controls were not acknowledged after cleanup')
    if any(final.get(k) for k in ('running','waiting','active','kv_allocations','transfer_allocations')):
        raise ValueError('validation left live requests or KV behind')
    before=raw['reject_before'];after=raw['reject_after'];probe=raw['probe_request_id']
    need=reservation(INPUT,OUTPUT)
    if (not before.get('accepting') or not before.get('admit_decode') or before['free_kv_tokens']>=need
            or not before['kv_allocations'] or probe in after['kv_allocations']
            or set(before['kv_allocations'])!=set(after['kv_allocations'])
            or raw['rejected_plan']['feasible'] or raw['rejected_plan']['routes']
            or raw['rejected_reserved'] or raw['reservations_after_reject']):
        raise ValueError('real KV rejection was not established independently of the diagnostic hold')
    if not raw['recovered_plan']['feasible'] or not raw['recovered_reserved']:
        raise ValueError('same request was not admissible after the real KV drained')
    reference=raw['reference']['token_ids'];output=raw['recovered_output']
    if len(reference)!=OUTPUT or output.get('token_ids')!=reference or output.get('usage',{}).get('completion_tokens')!=OUTPUT:
        raise ValueError('recovered output differs from ordinary execution')
    ids={h['request_id'] for h in raw['holds']};seen=set()
    if not ids or len(ids)>MAX_HELD or raw['clocks_written']:
        raise ValueError('invalid hold count or unexpected clock mutation')
    for event in raw['held_events']:
        if event.get('prefill',0)>0 and event.get('tokens',0)>0:seen.update(set(event['request_ids'])&ids)
    if seen!=ids:raise ValueError('not every held request has a real executed prefill')
    if any(h['after']['free_kv_tokens']<0 or h['request_id'] not in h['after']['kv_allocations'] for h in raw['holds']):
        raise ValueError('held requests lack actual nonoverflowing KV allocations')
    result=dict(real_mixed_kv_boundary=True,shared_joint_planner_kv_guard=True,
        no_new_request_allocation=True,cancel_drained=True,recovered_output_correct=True,
        distserve_pd_boundary=False,
        note='mixed hardware boundary on production backend snapshots; existing decode blocks may grow during the brief resume')
    if 'pd' in raw:result.update(verify_pd(raw['pd']))
    return result


def verify_pd(pd):
    d=pd['reject_before'];after=pd['reject_after'];need=reservation(INPUT,OUTPUT)
    if (not d.get('accepting') or not d.get('admit_decode') or d['role']!='decode'
            or d['free_kv_tokens']>=need or not d['kv_allocations']
            or set(d['kv_allocations'])!=set(after['kv_allocations'])
            or pd['rejected_plan']['feasible'] or pd['rejected_reserved'] or pd['reservations_after_reject']):
        raise ValueError('actual DistServe decode-target KV rejection incomplete')
    if any(pd['new_prefill_id'] in e['request_ids'] and e['started_s']<=pd['rejected_until_s'] for e in pd['producer_events']):
        raise ValueError('a rejected producer was forwarded')
    holds=pd['holds'];ids={h['request_id'] for h in holds};pids={h['producer_id'] for h in holds}
    actual={rid for e in pd['producer_events'] if e.get('prefill') and e.get('tokens') for rid in e['request_ids']}
    if (not 1<=len(holds)<=MAX_HELD or not pids<=actual or set(d['kv_allocations'])!=ids
            or any(h['request_id'] not in h['after']['kv_allocations'] or h['after']['free_kv_tokens']<0 for h in holds)):
        raise ValueError('PD boundary lacks actual producer execution and retained target KV')
    plan=pd['recovered_plan'];route=plan['routes'][0];reserved=pd['target_reservation']
    if (not plan['feasible'] or route['prefill_id']==route['decode_id'] or route['reserve_tokens']<need
            or reserved['reserved_kv_tokens']<route['reserve_tokens']
            or route['transfer_reserve_bytes']<=0 or reserved['reserved_transfer_bytes']<route['transfer_reserve_bytes']):
        raise ValueError('DistServe target KV/staging was not reserved before producer execution')
    steps=[e for e in pd['producer_events'] if pd['new_prefill_id'] in e['request_ids'] and e.get('prefill') and e.get('tokens')]
    if not steps or min(e['started_s'] for e in steps)<reserved['at_s']:
        raise ValueError('real producer timeline predates the target reservation')
    if (pd['recovered_output'].get('token_ids')!=pd['reference_tokens'] or len(pd['reference_tokens'])!=OUTPUT
            or pd['recovered_output'].get('usage',{}).get('completion_tokens')!=OUTPUT):
        raise ValueError('reserved DistServe PD output differs from ordinary reference')
    original=pd['producer_original'];final=pd['producer_restoration']
    if any(final.get(k)!=original[k] for k in CONTROL) or final.get('generation')!=final.get('acknowledged_generation'):
        raise ValueError('producer original controls were not restored')
    if any(final.get(k) for k in ('active','running','waiting','kv_allocations','transfer_allocations')):
        raise ValueError('producer did not drain')
    return dict(distserve_pd_boundary=True,target_kv_reserved_before_producer=True,
        target_staging_reserved_before_producer=True,rejected_producer_not_forwarded=True)


async def validate(plan,out):
    validation_started=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    followup=read(plan['followup']);cal=read(followup['calibration_template'])
    smoke=read(Path(plan['smoke_out'])/'setup.json')
    instance=dict(smoke['instances'][0]);instance['role']='mixed'
    source=dict(smoke['instances'][1]);source['role']='mixed'
    if (any(i['tp']!=1 or not set(i['gpus'])<=set(range(8)) for i in (instance,source))
            or set(instance['gpus'])&set(source['gpus'])):raise ValueError('two explicit nonoverlapping TP1 node instances required')
    raw=dict(passed=False,complete=False,formal_eligible=False,capacity_certified=False,
        purpose='real mixed and DistServe P-to-D KV-boundary correctness; not performance evidence',
        errors=[],cleanup_errors=[],holds=[],clocks_written=False,instance=instance,
        source_files=freeze_files(implementation_sources()),profiles=cal['profiles'],profile_sha256=sha256(cal['profiles']))
    tasks=[];owned={};profiler=None;original=None;producer_original=None;path=None;offset=0;source_path=None;source_offset=0
    prompt=([9707,1879,13]*(INPUT//3+1))[:INPUT]
    body=dict(prompt=prompt,max_tokens=OUTPUT,temperature=0,top_p=1.,seed=0,ignore_eos=True,stream=False)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        profiler=HardwareProfiler(session,{'mixed':instance,'prefill':source},smoke['runtime_dir'])
        backend=HttpEngineBackend([instance],session)
        async def request(rid,payload,where=instance):
            owned[rid]=where
            result=await profiler.call(where,'/v1/completions',payload,rid)
            owned.pop(rid,None)
            return result
        async def cancel_owned():
            report=await cancel_requests(profiler,owned,tasks)
            raw.setdefault('cancellations',[]).append(report)
            results=await asyncio.gather(drained(profiler,source),drained(profiler,instance),return_exceptions=True)
            report['errors'].extend(type(r).__name__+': '+str(r) for r in results if isinstance(r,BaseException))
            if report['errors']:raise RuntimeError('owned request cleanup failed: '+'; '.join(report['errors']))
            return results[1]
        async def work():
            nonlocal original,path,offset,producer_original,source_path,source_offset
            raw['provenance_before']=await profiler.provenance();engine_provenance(raw['provenance_before'],cal['image'])
            original=await drained(profiler,instance);raw['original_control']=original
            path=await event_file(instance,plan['campaign_root']);offset=path.stat().st_size
            await profiler.control(instance,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
            raw['reference']=await request('reference-'+uuid.uuid4().hex,body)
            await drained(profiler,instance)
            await profiler.control(instance,admit_decode=False)
            while True:
                state=await profiler.call(instance,'/runtime')
                if state['free_kv_tokens']<reservation(INPUT,OUTPUT):break
                if not can_add_hold(state,len(tasks)):
                    raise RuntimeError('six real held requests did not reach the measured KV boundary')
                rid='held-'+uuid.uuid4().hex
                task=asyncio.create_task(request(rid,dict(body,max_tokens=HELD_OUTPUT)));tasks.append(task)
                profiler.tasks=tasks
                after=await asyncio.wait_for(profiler.wait_allocated(instance,rid,len(tasks)),25)
                raw['holds'].append(dict(request_id=rid,before=state,after=after))
            # The production backend deliberately treats a decode hold as
            # unsafe admission. Resume first; never override accepting/free KV.
            await profiler.control(instance,admit_decode=True)
            snapshot=await backend.read_state();now=time.time()
            raw['reject_snapshot']=asdict(snapshot);raw['reject_before']=dict(backend.last[instance['id']])
            probe='probe-'+uuid.uuid4().hex;raw['probe_request_id']=probe
            budget=RequestBudget(probe,now,INPUT,256,120.,1.,output_limit=OUTPUT)
            planner=JointPlanner(ProfileStore.load(cal['profiles']),allow_pd=False,dvfs=False)
            rejected=capacity_rejection(snapshot,budget,planner,now);raw['rejected_plan']=asdict(rejected)
            state=StateManager(snapshot);raw['rejected_reserved']=await state.reserve(rejected,time.time(),budget)
            raw['reservations_after_reject']=dict(state.reservations)
            await asyncio.sleep(.02);raw['reject_after']=await profiler.call(instance,'/runtime')
            # Existing requests may emit tokens/extend their own blocks. The
            # rejected ID is never forwarded and creates no new allocation.
            raw['existing_decode_kv_delta']=sum(raw['reject_after']['kv_allocations'].values())-sum(raw['reject_before']['kv_allocations'].values())
            await profiler.control(instance,admit_decode=False)
            raw['after_cancel']=await cancel_owned()
            await profiler.control(instance,admit_decode=True)
            snapshot=await backend.read_state();now=time.time()
            budget=RequestBudget(probe,now,INPUT,256,120.,1.,output_limit=OUTPUT)
            recovered=planner.plan(snapshot,(budget,),now=now,joint=False);raw['recovered_plan']=asdict(recovered)
            state=StateManager(snapshot);raw['recovered_reserved']=await state.reserve(recovered,time.time(),budget)
            if not raw['recovered_reserved']:raise RuntimeError('drained same request remains inadmissible')
            raw['recovered_output']=await request(probe,body);await state.release(probe)
            await drained(profiler,instance)
            producer_original=await drained(profiler,source)
            source_path=await event_file(source,plan['campaign_root']);source_offset=source_path.stat().st_size
            pd=dict(holds=[],reference_tokens=raw['reference']['token_ids'],producer_original=producer_original)
            raw['pd']=pd
            await profiler.control(source,role='prefill',mode='continuous',admit_prefill=True,admit_decode=True)
            await profiler.control(instance,role='decode',mode='continuous',admit_prefill=True,admit_decode=False)
            await profiler.call(source,'/prepare-peers',dict(peers=[instance['id']]))
            phase_tasks=[]
            while True:
                before=await staging_drained(profiler,instance)
                if before['free_kv_tokens']<reservation(INPUT,OUTPUT):break
                if not can_add_hold(before,len(phase_tasks)):raise RuntimeError('six true PD held requests did not reach target KV boundary')
                if before['free_transfer_bytes']<INPUT*before['transfer_bytes_per_token']:
                    raise RuntimeError('previous PD staging did not release before the next producer')
                nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{source['id']}:{instance['id']}";did=f"pdb:{nonce}:d:{source['id']}:{instance['id']}"
                owned[did]=instance
                await request(pid,dict(body,max_tokens=1),source)
                task=asyncio.create_task(request(did,dict(body,max_tokens=HELD_OUTPUT)))
                tasks.append(task);phase_tasks.append(task);profiler.tasks=phase_tasks
                after=await asyncio.wait_for(profiler.wait_allocated(instance,did,len(phase_tasks)),25)
                pd['holds'].append(dict(request_id=did,producer_id=pid,before=before,after=after))
            await profiler.control(instance,admit_decode=True)
            pd_backend=HttpEngineBackend([dict(source,role='prefill'),dict(instance,role='decode')],session)
            snapshot=await pd_backend.read_state();now=time.time()
            nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{source['id']}:{instance['id']}";did=f"pdb:{nonce}:d:{source['id']}:{instance['id']}"
            pd.update(logical_request_id=nonce,new_prefill_id=pid,new_decode_id=did,reject_snapshot=asdict(snapshot),
                reject_before=dict(pd_backend.last[instance['id']]))
            budget=RequestBudget(nonce,now,INPUT,256,120.,1.,output_limit=OUTPUT)
            topology=InterconnectTopology.parse(Path(cal['interconnect']).read_text())
            spatial=DistServeScheduler(planner.profiles,[TransferCost(**v) for v in read(cal['transfers'])['links']],
                prefill_batch=32,decode_batch=32,topology=topology)
            rejected=spatial_rejection(snapshot,budget,spatial,source['id'],instance['id'],now)
            state=StateManager(snapshot);pd['rejected_reserved']=await state.reserve(rejected,time.time(),budget)
            pd.update(rejected_plan=asdict(rejected),reservations_after_reject=dict(state.reservations),
                reject_after=await profiler.call(instance,'/runtime'),rejected_until_s=time.time())
            await profiler.control(instance,admit_decode=False);pd['after_cancel']=await cancel_owned()
            await profiler.control(instance,admit_decode=True)
            snapshot=await pd_backend.read_state();now=time.time()
            budget=RequestBudget(nonce,now,INPUT,256,120.,1.,output_limit=OUTPUT)
            recovered=spatial.plan(snapshot,(budget,),now=now);state=StateManager(snapshot)
            pd['recovered_plan']=asdict(recovered)
            owned[did]=instance
            async def producer():return await request(pid,dict(body,max_tokens=1),source)
            pd['target_reservation'],pd['prefill_output']=await reserve_then_produce(state,recovered,budget,producer)
            pd['received_before_decode']=await profiler.call(instance,'/runtime')
            pd['recovered_output']=await request(did,body)
            await state.release(nonce);await drained(profiler,instance)
        try:
            await asyncio.wait_for(work(),175)
        except BaseException as exc:
            raw['errors'].append(type(exc).__name__+': '+str(exc))
        finally:
            write(out/'raw.json',raw)
            cleanup_deadline=min(time.monotonic()+55.,validation_started+230.)
            try:
                cleanup=await asyncio.wait_for(cleanup_instances(profiler,owned,tasks,
                    [(instance,original),(source,producer_original)],deadline=cleanup_deadline),
                    max(.001,cleanup_deadline-time.monotonic()))
                raw['cleanup']=cleanup;raw['cleanup_errors'].extend(cleanup['errors'])
                if instance['id'] in cleanup['restored']:raw['restoration']=cleanup['restored'][instance['id']]
                if source['id'] in cleanup['restored']:raw['pd']['producer_restoration']=cleanup['restored'][source['id']]
            except BaseException as exc:
                raw['cleanup_errors'].append('bounded restoration: '+type(exc).__name__+': '+str(exc))
            # Persist partial restoration results before read-only evidence
            # collection, which shares the same cleanup deadline.
            write(out/'raw.json',raw)
            async def collect_cleanup_evidence():
                if path is not None:
                    def timeline():
                        with path.open() as handle:
                            handle.seek(offset)
                            return [json.loads(line) for line in handle if line.strip()]
                    raw['held_events']=await asyncio.to_thread(timeline)
                    raw['held_emitted_tokens']={h['request_id']:sum(1 for e in raw['held_events']
                        if e.get('tokens',0)>0 and h['request_id'] in e['request_ids']) for h in raw['holds']}
                if source_path is not None:
                    def producer_timeline():
                        with source_path.open() as handle:
                            handle.seek(source_offset)
                            return [json.loads(line) for line in handle if line.strip()]
                    raw['pd']['producer_events']=await asyncio.to_thread(producer_timeline)
                    raw['pd']['held_emitted_tokens']={h['request_id']:1+sum(1 for e in raw['held_events']
                        if e.get('decode',0)>0 and e.get('tokens',0)>0 and h['request_id'] in e['request_ids']) for h in raw['pd']['holds']}
                raw['provenance_after']=await profiler.provenance()
                await asyncio.to_thread(engine_provenance,raw['provenance_after'],cal['image'])
                changed=await asyncio.to_thread(lambda:validate_freeze(raw['source_files']) or
                    sha256(cal['profiles'])!=raw['profile_sha256'])
                if changed:
                    raise RuntimeError('implementation or profiles changed during admission validation')
            try:
                await asyncio.wait_for(collect_cleanup_evidence(),max(.001,cleanup_deadline-time.monotonic()))
            except BaseException as exc:raw['cleanup_errors'].append('bounded evidence: '+type(exc).__name__+': '+str(exc))
            try:raw['checks']=verify_raw(raw);raw.update(passed=True,complete=True)
            except (ValueError,KeyError) as exc:raw['errors'].append(str(exc))
            write(out/'raw.json',raw)
    if not raw['passed']:raise RuntimeError('real KV admission validation incomplete; inspect '+str(out/'raw.json'))
    return raw


def main():
    from .campaign import node_lease
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    async def cancellable():
        loop=asyncio.get_running_loop();loop.add_signal_handler(signal.SIGTERM,asyncio.current_task().cancel)
        try:return await validate(read(args.manifest),args.out)
        finally:loop.remove_signal_handler(signal.SIGTERM)
    with node_lease():result=asyncio.run(cancellable())
    print(json.dumps(dict(passed=result['passed'],checks=result['checks'])))


if __name__=='__main__':main()
