"""Six measured directed transitions during complete owned decode on one GPU."""
import asyncio,json,math,time
from pathlib import Path
import validate as v

def pairs():return [(a,b) for low in (900,1500,2100) for a,b in ((low,2400),(2400,low))]
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

async def observe(common,session,instance,probe,clocks,hardware,out):
    gpu=instance['gpus'][0];raw=dict(complete=False,gpu=gpu,instance_id=instance['id'],requests=[],switches=[],error=None)
    task=None;offset=None;cfg=v.read(instance['engine_config']);event_path=Path(cfg['runtime_dir'])/(instance['id']+'.control.events.jsonl')
    async def settled(target):
        beginning=time.monotonic();rows=[];consecutive=0
        while consecutive<3:
            v.require(task is not None and not task.done(),'owned decode ended before measured clock transition')
            values=await asyncio.to_thread(lambda:[hardware.current_freq(gpu)])
            rows.append(dict(at_s=time.time(),actual_sm_mhz=values))
            consecutive=consecutive+1 if all(abs(x-target)<=15 for x in values) else 0
            v.require(time.monotonic()-beginning<4,'actual clock failed to settle within original four-second observation budget')
            if consecutive<3:await asyncio.sleep(.02)
        return rows
    try:
        raw['budget_transition']=await common.resume(session,instance,2048)
        await clocks.set([gpu],2400,verify_rise=False)
        raw['reference']=await probe.request(128,32)
        v.require(raw['reference']['success'] and len(raw['reference']['output_token_ids'])==32,'actual numerical reference incomplete')
        raw['runtime_before']=await common.wait_idle(session,instance);offset=event_path.stat().st_size
        raw['started_s']=time.time();task=asyncio.create_task(probe.request(128,1024));probe.tasks=[task]
        await asyncio.sleep(1)
        for source,target in pairs():
            await clocks.set([gpu],source,verify_rise=False);source_observed=await settled(source);await asyncio.sleep(.2)
            state=await common.http(session,instance,'/runtime')
            v.require(state['active']==state['running']==1 and state['waiting']==0,'one real active owned request required during switching')
            start=time.time();await clocks.set([gpu],target,verify_rise=False);target_observed=await settled(target);end=time.time()
            raw['switches'].append(dict(source_mhz=source,target_mhz=target,started_s=start,finished_s=end,
                source_observed=source_observed,target_observed=target_observed,actual_runtime_before=state,applied_after=dict(clocks.applied)))
            save(out/'raw.json',raw)
        row=await task;raw['requests']=[row]
        v.require(row['success'] and row.get('done_marker') and len(row['output_token_ids'])==1024
                  and row['usage']['completion_tokens']==1024 and row['output_token_ids'][:32]==raw['reference']['output_token_ids'],
                  'actual full output changed or failed during measured switching')
        raw['runtime_after_requests']=await common.wait_idle(session,instance);raw['complete']=True
    except BaseException as exc:
        raw['error']=repr(exc)
        if task is not None and not task.done():
            for rid in probe.issued:
                try:await common.http(session,instance,'/cancel',dict(request_id=rid))
                except BaseException as error:raw.setdefault('cancel_errors',[]).append(repr(error))
            rows=await asyncio.gather(task,return_exceptions=True)
            raw['requests']=[r if isinstance(r,dict) else dict(success=False,error=repr(r)) for r in rows]
    finally:
        try:raw['cleanup']=await asyncio.wait_for(common.restore(session,instance),120)
        except BaseException as exc:raw['cleanup']=dict(complete=False,error=repr(exc))
        raw['finished_s']=time.time()
        if offset is not None:
            with event_path.open('rb') as f:f.seek(offset);(out/'events.jsonl').write_bytes(f.read())
            raw['events']=v.ref(out/'events.jsonl')
        save(out/'raw.json',raw)
    v.require(raw['complete'] and raw['cleanup']['complete'] and not raw['error'],'actual transition work or cleanup failed')
    return raw

def derive(raw,power,clocks):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    native=v.load(v.PARENT,'actual2400_switch_native')
    v.require(raw['complete'] is True and not raw['error'] and len(raw['requests'])==1
              and [(x['source_mhz'],x['target_mhz']) for x in raw['switches']]==pairs(),'complete six directed observations required')
    gpu=raw['gpu'];row=raw['requests'][0]
    v.require(row['success'] and row.get('done_marker') and row['http_status']==200
              and row['prompt_token_ids']==([9707,1879,13]*43)[:128] and row['requested_output_tokens']==1024
              and len(row['output_token_ids'])==len(row['token_received_s'])==1024
              and row['usage']['prompt_tokens']==128 and row['usage']['completion_tokens']==1024
              and row['output_token_ids'][:32]==raw['reference']['output_token_ids'], 'transition numerical work incomplete')
    for state in (raw['runtime_before'],raw['runtime_after_requests']):native.native_saved(state,raw['instance_id'],2048)
    native.cleanup_saved(raw['cleanup'],raw['instance_id'])
    events=[json.loads(line) for line in Path(raw['events']['path']).read_text().splitlines() if line]
    v.require(v.sha(raw['events']['path'])==raw['events']['sha256'],'changed switch native stream')
    compatible=dict(raw,spec=dict(batch_size=1,input_tokens=128,output_tokens=1024),measurement_start_s=raw['started_s'],measurement_end_s=raw['finished_s'])
    selected=events
    if any(not e.get('request_ids') for e in events):
        selected,_=v.load(v.HERE/'terminal.py','actual2400_switch_terminal').trailing_empty(compatible,events)
    # The surrounding complete measurement independently verifies the actual
    # scheduler/logger source pair before this reconstruction is registered.
    context=v.load(v.HERE/'context_export.py','actual2400_switch_context')
    context.reconstruct(compatible,selected,generation=raw['runtime_before']['generation'],token_budget=2048,source_order_verified=True)
    result=[]
    for switch in raw['switches']:
        a,b=switch['started_s'],switch['finished_s'];source,target=switch['source_mhz'],switch['target_mhz']
        v.require(raw['started_s']<=row['dispatch_s']<a<b<row['stream_end_s']<=raw['finished_s']
                  and switch['applied_after'].get(str(gpu),switch['applied_after'].get(gpu))==target,'switch escaped actual complete decode')
        v.require(switch['actual_runtime_before']['id']==raw['instance_id'] and switch['actual_runtime_before']['active']==1
                  and switch['actual_runtime_before']['waiting']==0,'switch was not under real native load')
        for key,frequency in [('source_observed',source),('target_observed',target)]:
            observed=switch[key]
            v.require(len(observed)>=3 and all(math.isfinite(x['at_s']) and len(x['actual_sm_mhz'])==1 for x in observed)
                      and all(y['at_s']>x['at_s'] for x,y in zip(observed,observed[1:]))
                      and all(abs(x['actual_sm_mhz'][0]-frequency)<=15 for x in observed[-3:]),'three real target observations required')
        v.require(switch['source_observed'][-1]['at_s']<=a and a<=switch['target_observed'][0]['at_s']
                  and switch['target_observed'][-1]['at_s']<=b,'actual transition read chronology differs')
        w=clip_power_window(power,a,b,pad_s=0)
        v.require(all(len(values)==8 and all(math.isfinite(x) and x>=0 for x in values) for _,values in w),'all8 real switch energy missing')
        result.append(dict(switch,gpu=gpu,tp=1,actual_duration_s=b-a,all8_energy_j=trapezoid_energy(w),
                           empirical_not_future_guarantee=True))
    return dict(passed=True,gpu=gpu,switches=result,full_numerical_work=True,native_cleanup_complete=True)
