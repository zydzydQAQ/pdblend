"""Measured TP1 frequency and legacy-cancellation qualification after cold restore."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import time
from types import SimpleNamespace
import uuid

import cold_restore as c


def clock_window(samples, gpus, target, start, end):
    assert end > start
    rows = [(t, f) for t, f in samples if start <= t <= end]
    assert len(rows) >= 2 and rows[0][0]-start <= .25 and end-rows[-1][0] <= .25
    assert max(b[0]-a[0] for a,b in zip(rows, rows[1:])) <= .25
    assert all(len(f) == 8 and all(type(v) in (int,float) for v in f) for _,f in rows)
    assert all(abs(f[g]-target) <= 15 for _,f in rows for g in gpus)
    return dict(gpus=gpus, target_mhz=target, start_s=start, end_s=end, samples=len(rows),
                minimum_mhz=min(f[g] for _,f in rows for g in gpus),
                maximum_mhz=max(f[g] for _,f in rows for g in gpus), passed=True)


def legacy_transfer_rows(rows):
    assert isinstance(rows,list) and len(rows)==1, 'one actual TP1 reply required'
    row=rows[0]
    for key in ('buffered_tensors','buffered_gpu_bytes','inflight_receives'):
        assert type(row.get(key)) is int and row[key]==0, 'legacy cancel residue: '+key
    assert row.get('listener_alive') is True and row.get('allocations') == {}


def legacy_idle(raw, instance):
    assert raw['id']==instance['id']
    for key in ('active','running','waiting'):
        assert type(raw.get(key)) is int and raw[key]==0
    assert raw.get('kv_allocations')=={} and raw.get('transfer_allocations')=={}
    assert not raw.get('error') and not raw.get('runtime_error')
    assert raw['generation'] == raw['acknowledged_generation'] and raw['generation']>0
    caches=[v.get('controls',{}).get('runtime') for v in raw.get('scheduler_io',[])]
    assert len(caches)==instance.get('scheduler_cache_count',1)
    assert all(v and v.get('generation')==raw['generation'] and not v.get('error') for v in caches)


def validate(spec, restoration=None):
    assert spec['schema']=='C7B-frequency-legacy-cancel-spec-v1'
    for path,digest in spec['files'].items():
        assert c.sha(path)==digest, path
    cold=c.checked(spec['cold_spec']); c.validate(cold)
    assert spec['frequencies_mhz']==[900,1500,2100,2520]
    if restoration is None:
        return None,None
    state=c.checked(restoration)
    assert state['spec']==spec['cold_spec'] and state['complete'] and not state.get('error')
    assert state['finished_s'] and not state['node_lease_held']
    binding=c.checked(state['binding']); ordinary=c.checked(state['ordinary'])
    assert ordinary['passed'] and binding['hostname']==c.HOSTNAME
    assert binding['instances']==c.checked(state['binding'])['instances']
    return binding,ordinary


async def cancel_case(common, session, instance):
    import aiohttp
    rid='C7B-ascending-cancel-'+uuid.uuid4().hex
    limit=time.time()+30
    async def issue():
        body=dict(prompt=([9707,1879,13]*43)[:128],max_tokens=512,temperature=0,
                  top_p=1,seed=0,ignore_eos=True,stream=False)
        async with session.post(instance['url']+'/v1/completions',json=body,
                headers={'X-Request-Id':rid},timeout=aiohttp.ClientTimeout(total=30)) as response:
            return dict(status=response.status,body=await response.text(),finished_s=time.time())
    child=asyncio.create_task(issue()); cancelled=False
    try:
        while time.time()<limit:
            before=await common.http(session,instance,'/runtime',timeout=1)
            if before.get('active') and before.get('running') and before.get('kv_allocations',{}).get(rid,0)>0:
                break
            assert not child.done(), 'controlled request finished before own running KV proof'
            await asyncio.sleep(.05)
        else:
            raise TimeoutError('own running KV never observed')
        reply=await common.http(session,instance,'/cancel',dict(request_id=rid))
        cancelled=True
        assert reply.get('cancelled')==rid
        legacy_transfer_rows(reply.get('transfers'))
        response=await asyncio.wait_for(child,max(.1,limit-time.time()))
        assert response['status']>=400 and 'cancel' in response['body'].lower()
        settled=await common.wait_idle(session,instance,seconds=max(.1,limit-time.time()))
        legacy_idle(settled,instance)
        return dict(request_id=rid,before=before,cancelled=reply,response=response,settled=settled,
                    verified=True,proof_kind='legacy_sync_put; no V3 sender-counter claim')
    finally:
        try:
            if not cancelled:
                await common.http(session,instance,'/cancel',dict(request_id=rid))
        finally:
            if not child.done():
                child.cancel()
            await asyncio.gather(child,return_exceptions=True)


async def execute(spec, restoration, out, state):
    binding,ordinary=validate(spec,restoration)
    assert socket.gethostname()==c.HOSTNAME and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    cold=c.checked(spec['cold_spec'])
    helper=c.load(cold['restore_executor']['path'],'C_qualification_original_executor')
    common=helper.load_common(binding['host_release'])
    c.load(spec['capacity_executor']['path'],'capacity_executor')
    meter_module=c.load(spec['capacity_backend']['path'],'C_qualification_meter')
    stream_module=c.load(spec['stream']['path'],'C_qualification_frozen_stream')
    import aiohttp
    from ecopadg.serving.backend import ClockOwner
    from ecopadg.serving.campaign import node_lease
    expected={r['instance_id']:r['response']['token_ids'] for r in ordinary['replies'] if r['prompt_length']==128}
    meter=owner=None; streams=[]
    with node_lease():
        state['node_lease_held']=True; c.save(out/'status.json',state)
        async with aiohttp.ClientSession(trust_env=False) as session:
            try:
                common.validate_binding(binding)
                c.save(out/'identity.before.json',await common.identity(session,binding))
                meter=await meter_module.TransitionMeter(out/'power').start()
                owner=await asyncio.to_thread(ClockOwner,meter.sampler.backend,tuple(range(8)))
                for instance in binding['instances']:
                    stream=stream_module.NaturalStream();stream.session=session
                    stream.args=SimpleNamespace(port=instance['port'],seed=0);stream.issued=set()
                    stream.stream_journal=(out/(instance['id']+'.stream.jsonl')).open('x')
                    stream.result_journal=(out/(instance['id']+'.requests.jsonl')).open('x')
                    streams.append((instance,stream))
                    for frequency in spec['frequencies_mhz']:
                        await common.resume(session,instance)
                        write=await owner.set(tuple(instance['gpus']),frequency,verify_rise=False)
                        warmup=await stream.request(128,64)
                        assert warmup['success'] and warmup['output_token_ids']==expected[instance['id']]
                        await common.wait_idle(session,instance)
                        row=await stream.request(128,64)
                        assert row['success'] and row['output_token_ids']==expected[instance['id']]
                        assert row['usage']['prompt_tokens']==128 and row['usage']['completion_tokens']==64
                        native=await common.wait_idle(session,instance);legacy_idle(native,instance)
                        clocks=clock_window(meter.sampler.frequency_samples,instance['gpus'],frequency,
                                            row['token_received_s'][0],row['token_received_s'][-1])
                        state['frequency_cases'].append(dict(instance_id=instance['id'],frequency_mhz=frequency,
                            warmup=warmup,request=row,native_after=native,clock_write=write,loaded_clock=clocks))
                        c.save(out/'status.json',state)
                    state['cancellations'].append(dict(instance_id=instance['id'],
                        evidence=await cancel_case(common,session,instance)))
                    c.save(out/'status.json',state)
                state['passed']=True
            finally:
                for instance,stream in streams:
                    for rid in stream.issued:
                        try:
                            await common.http(session,instance,'/cancel',dict(request_id=rid))
                        except BaseException as exc:
                            state['cleanup_errors'].append('owned cancellation: '+repr(exc))
                    stream.stream_journal.close();stream.result_journal.close()
                restored=await asyncio.gather(*(common.restore(session,i) for i in binding['instances']),return_exceptions=True)
                state['restoration']={i['id']:dict(error=repr(x)) if isinstance(x,BaseException) else x
                                      for i,x in zip(binding['instances'],restored)}
                if owner:
                    try:
                        await asyncio.wait_for(owner.close(),15);state['clock_restore_complete']=True
                    except BaseException as exc:
                        state['cleanup_errors'].append('clock restore: '+repr(exc))
                if meter:
                    state['measurement']=await meter.finish()
                c.save(out/'identity.after.json',await common.identity(session,binding))
                state['passed']=bool(state['passed'] and not state['cleanup_errors'] and state.get('clock_restore_complete')
                    and all(x.get('complete') for x in state['restoration'].values())
                    and state.get('measurement',{}).get('measurement_valid'))
        state['node_lease_held']=False
    assert state['passed'], 'fresh qualification failed; no performance successor'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=c.HERE/'qualification-spec.json')
    parser.add_argument('--restoration',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();spec=c.read(args.spec)
    restoration=c.ref(args.restoration) if args.restoration else None
    validate(spec,restoration)
    if not args.run:
        print(json.dumps(dict(passed=True,cpu_only=True,gpu_work_started=False,requires_fresh_restore=True)));return
    assert restoration and args.out and not args.out.exists()
    args.out.mkdir(parents=True)
    state=dict(schema='C7B-frequency-legacy-cancel-status-v1',pid=os.getpid(),started_s=time.time(),
               spec=c.ref(args.spec),cold_restoration=restoration,passed=False,node_lease_held=False,
               frequency_cases=[],cancellations=[],cleanup_errors=[])
    async def controlled():
        task=asyncio.current_task()
        for sig in (signal.SIGINT,signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig,task.cancel)
        await execute(spec,restoration,args.out,state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(passed=False,error=repr(exc));raise
    finally:
        state.update(finished_s=time.time(),node_lease_held=False);c.save(args.out/'status.json',state)
    cold=c.checked(restoration)
    qualification=dict(schema='C7B-saved-frequency-legacy-qualification-v1',spec=c.ref(args.spec),
        status=c.ref(args.out/'status.json'),cold_restoration=restoration,binding=cold['binding'],ordinary=cold['ordinary'],
        files={str(p):c.sha(p) for p in args.out.rglob('*') if p.is_file()},source_files=spec['files'],
        performance_started=False)
    c.save(args.out/'qualified.json',qualification)
    verifier=c.load(c.HERE/'verify_qualification.py','C_independent_saved_qualification')
    print(json.dumps(verifier.verify(c.ref(args.out/'qualified.json'))))


if __name__=='__main__':
    main()
