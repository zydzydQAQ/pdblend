"""Explicit warmup followed by source-ordered actual2400 pure-prefill samples."""
import asyncio
import json
from pathlib import Path
import time

def require(ok,message):
    if not ok:
        raise ValueError(message)

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)

def point_specs():
    return [dict(point_id=f'pure-prefill-in{length}-out64-f2400-repeat{repeat}',
        input_tokens=length,output_tokens=64,batch_size=1,clock_command_mhz=2400,repeat=repeat,
        explicit_warmup_input=128,explicit_warmup_output=32,hidden_warmup=False)
        for length in (128,2048,7168) for repeat in range(1,4)]

async def observe(probe,clocks,meter,out,point_done):
    event_path=Path(probe.o.configured_engines()['nextv3a6']['config']['runtime_dir'])/'nextv3a6.control.events.jsonl'
    await probe.idle()
    transition=await probe.o.helpers().set_one(probe.o.parent(),probe.session,0,2048)
    probe.o.check_ack(transition['after'],startup=True,tokens=2048)
    await clocks.set([6],2400,verify_rise=False)
    for spec in point_specs():
        path=out/'results'/spec['point_id']
        path.mkdir()
        raw=dict(schema='A2400-explicit-pure-prefill-v1',spec=spec,complete=False,
            native_generation=transition['after']['generation'],budget_transition=transition,
            profile_publication_allowed=False,all_warmup_energy_in_whole_operation=True)
        try:
            raw['declared_warmup']=await probe.request(128,32)
            warmup=raw['declared_warmup']
            require(warmup['success'] and warmup['done_marker'] and len(warmup['output_token_ids'])==32,
                    'complete declared warmup required')
            raw['runtime_before']=await probe.idle()
            offset=event_path.stat().st_size
            raw['measurement_start_s']=time.time()
            raw['request']=await probe.request(spec['input_tokens'],spec['output_tokens'])
            request=raw['request']
            require(request['success'] and request['done_marker'] and len(request['output_token_ids'])==64
                    and request['usage']['prompt_tokens']==spec['input_tokens']
                    and request['usage']['completion_tokens']==64,'complete prescribed pure-prefill request required')
            raw['runtime_after']=await probe.idle()
            raw['measurement_end_s']=time.time()
            with event_path.open('rb') as handle:
                handle.seek(offset)
                (path/'events.jsonl').write_bytes(handle.read())
            events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
            raw['complete']=True
            await asyncio.sleep(.06)
            raw['immediate_phase_evidence']=derive(raw,events,meter.sampler.samples,meter.sampler.frequency_samples)
        except BaseException as exc:
            raw['error']=repr(exc)
        finally:
            save(path/'raw.json',raw)
            point_done(raw,path)
        require(not raw.get('error') and raw.get('immediate_phase_evidence',{}).get('valid'),
                'actual pure-prefill evidence failed; later points stop')

def derive(raw,events,power,clocks):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    require(raw['complete'],'incomplete prescribed request')
    request=raw['request'];rid=request['request_id'];spec=raw['spec']
    require(request['prompt_token_ids']==([9707,1879,13]*(spec['input_tokens']//3+1))[:spec['input_tokens']],
            'actual input tokens changed')
    previous=None;prefills=[];prefill_tokens=0;decode_tokens=0
    for event in events:
        ids=event['request_ids'];start,end=event['started_s'],event['finished_s']
        require(set(ids)<={rid} and len(ids)==len(set(ids)) and event['generation']==raw['native_generation']
            and event['role']=='mixed' and event['mode']=='continuous' and 0<=event['tokens']<=2048
            and raw['measurement_start_s']<=start<end<=raw['measurement_end_s']
            and (previous is None or start>=previous-1e-6),'actual owner phase identity/order differs')
        previous=end
        require(event['prefill']+event['decode']==len(ids),'phase identity mismatch')
        if event['prefill']:
            require(event['prefill']==1 and event['decode']==0 and ids==[rid],
                    'pure-prefill phase has foreign decode interference')
            prefills.append(event);prefill_tokens+=event['tokens']
        else:
            require(event['tokens']==event['decode'],'pure decode accounting mismatch')
            decode_tokens+=event['decode']
    require(prefill_tokens==spec['input_tokens'] and decode_tokens==63 and prefills,
            'full actual prefill and complete decode tail required')
    active=[v[6] for t,v in clocks if any(e['started_s']<=t<=e['finished_s'] for e in prefills)]
    require(len(active)>=2 and all(abs(f-2400)<=15 for f in active),
            'actual pure-prefill clock must remain2400 within15 throughout all owner steps')
    start,end=prefills[0]['started_s'],prefills[-1]['finished_s']
    window=clip_power_window(power,start,end,pad_s=0)
    require(all(len(v)==8 for _,v in window),'all-eight power missing')
    target=trapezoid_energy([(t,[v[6]]) for t,v in window])
    return dict(valid=True,input_tokens=spec['input_tokens'],batch_size=1,frequency_mhz=2400,
        prefill_duration_s=end-start,prefill_target_power_w=target/(end-start),
        prefill_all_eight_gpu_j=trapezoid_energy(window),owner_prefill_steps=len(prefills),
        actual_sm_min_mhz=min(active),actual_sm_max_mhz=max(active),actual_clock_samples=len(active),
        complete_output_tokens=64,profile_publication_allowed=False,
        scope='actual pure prefill only; no decode batch or cross-context interference inferred')
