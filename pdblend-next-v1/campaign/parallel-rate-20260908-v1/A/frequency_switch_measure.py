"""GPU6 measured 2400 transitions, owned natural decode, original native cleanup."""
import asyncio
import json
import time
from pathlib import Path

def require(ok, message):
    if not ok:
        raise ValueError(message)

def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)

def pairs():
    return [(source, target) for lower in (900, 1500, 2100)
            for source, target in ((lower, 2400), (2400, lower))]

async def observe(probe, clocks, hardware, out):
    raw = dict(schema='A2400-real-clock-transitions-v1', complete=False, requests=[], switches=[],
               target_gpus=[6], tp=1, repeats=3, prescribed_reference_output=32,
               prescribed_active_output=1024, automatic_retries=False, profile_publication_allowed=False)
    path=out/'switches.raw.json'
    task=None
    event_path=Path(probe.o.configured_engines()['nextv3a6']['config']['runtime_dir'])/'nextv3a6.control.events.jsonl'
    offset=None
    async def settled(target):
        start=time.monotonic()
        rows=[]
        consecutive=0
        while consecutive<3:
            require(task is not None and not task.done(), 'owned decode ended before actual clock observation')
            values=await asyncio.to_thread(lambda: [hardware.current_freq(6)])
            now=time.time()
            rows.append(dict(at_s=now, actual_sm_mhz=values))
            consecutive=consecutive+1 if all(abs(v-target)<=15 for v in values) else 0
            require(time.monotonic()-start<4, 'actual target failed to settle within4s under owned decode')
            if consecutive<3:
                await asyncio.sleep(.02)
        return rows
    try:
        raw['runtime_before']=await probe.idle()
        raw['budget_transition']=await probe.o.helpers().set_one(probe.o.parent(),probe.session,0,2048)
        raw['runtime_service']=raw['budget_transition']['after']
        probe.o.check_ack(raw['runtime_service'],startup=True,tokens=2048)
        await clocks.set([6],2400,verify_rise=False)
        raw['idle_residency_start_s']=time.time()
        await asyncio.sleep(5)
        raw['idle_residency_end_s']=time.time()
        raw['runtime_after_idle']=await probe.idle()
        require(clocks.applied.get(6)==2400, 'idle lock command differs')
        reference=await probe.request(128,32)
        raw['requests'].append(reference)
        require(reference['success'] and reference['done_marker'] and len(reference['output_token_ids'])==32,
                'full real short reference required')
        offset=event_path.stat().st_size
        raw['active_start_s']=time.time()
        task=asyncio.create_task(probe.request(128,1024))
        await asyncio.sleep(1)
        for repeat in range(1,4):
            for source,target in pairs():
                await clocks.set([6],source,verify_rise=False)
                source_observed=await settled(source)
                await asyncio.sleep(.2)
                state=await probe.http('/runtime')
                require(state.get('active')==1 and state.get('running')==1 and not state.get('waiting'),
                        'exact one real owned active decode required')
                start=time.time()
                await clocks.set([6],target,verify_rise=False)
                observed=await settled(target)
                end=time.time()
                raw['switches'].append(dict(repeat=repeat,tp=1,source_mhz=source,target_mhz=target,
                    started_s=start,finished_s=end,source_observed=source_observed,
                    target_observed=observed,actual_runtime_before=state,
                    applied_after=dict(clocks.applied)))
                save(path,raw)
        output=await task
        raw['active_end_s']=time.time()
        raw['requests'].append(output)
        require(output['success'] and output['done_marker'] and len(output['output_token_ids'])==1024
                and output['usage']['completion_tokens']==1024 and output['output_token_ids'][:32]==reference['output_token_ids'],
                'complete output identity changed across actual frequency transitions')
        raw['runtime_after_requests']=await probe.idle()
        raw['complete']=True
    except BaseException as exc:
        raw['error']=repr(exc)
        if task is not None and not task.done():
            for rid in probe.issued:
                try:
                    await probe.http('/cancel',dict(request_id=rid))
                except BaseException as error:
                    raw.setdefault('cancel_errors',[]).append(repr(error))
            result=await asyncio.gather(task,return_exceptions=True)
            raw['requests'].extend(r if isinstance(r,dict) else dict(success=False,error=repr(r)) for r in result)
    finally:
        try:
            raw['cleanup']=await probe.restore()
        except BaseException as exc:
            raw['cleanup']=dict(complete=False,error=repr(exc))
        raw['finished_s']=time.time()
        if offset is not None:
            with event_path.open('rb') as handle:
                handle.seek(offset)
                (out/'switches.events.jsonl').write_bytes(handle.read())
        save(path,raw)
    require(raw['complete'] and raw['cleanup']['complete'], 'actual transition work or cleanup failed')
    return raw

def derive(raw,power,clocks):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    require(raw['complete'] and raw['cleanup']['complete'] and len(raw['switches'])==18,
            'all three bidirectional transition repeats and complete output required')
    rows=[]
    for switch in raw['switches']:
        require(switch['source_mhz']!=switch['target_mhz'] and switch['applied_after'].get('6')==switch['target_mhz'],
                'actual target command differs')
        for key,target in (('source_observed',switch['source_mhz']),('target_observed',switch['target_mhz'])):
            observed=switch[key]
            require(len(observed)>=3 and all(abs(row['actual_sm_mhz'][0]-target)<=15 for row in observed[-3:]),
                    'three consecutive actual target observations required')
        window=clip_power_window(power,switch['started_s'],switch['finished_s'],pad_s=0)
        require(all(len(values)==8 for _,values in window), 'complete eight GPU switching power required')
        rows.append(dict(switch, all_eight_gpu_energy_j=trapezoid_energy(window)))
    start,end=raw['idle_residency_start_s'],raw['idle_residency_end_s']
    window=clip_power_window(power,start,end,pad_s=0)
    target=trapezoid_energy([(t,[values[6]]) for t,values in window])
    actual=[v[6] for t,v in clocks if start<=t<=end]
    require(len(actual)>=3 and all(len(v)==8 for t,v in clocks if start<=t<=end), 'idle actual clock rows missing')
    return dict(valid=True,profile_publication_allowed=False,switches=rows,
        idle_residency=dict(duration_s=end-start,all_eight_gpu_energy_j=trapezoid_energy(window),
            target_gpu6_mean_w=target/(end-start),locked_command_mhz=2400,
            actual_sm_min_mhz=min(actual),actual_sm_max_mhz=max(actual),
            legitimate_idle_Pstate_may_run_below_locked_command=True),
        full_reference_and_decode_work_complete=True,empirical_transition_envelopes_only=True)
