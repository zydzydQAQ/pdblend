"""Native DistServe stage windows and independent holdout, with restart files.

Two resident engines measure P and D on disjoint groups. The shared wave only
coordinates interference probes; all samples, shape selection and fits here
belong exclusively to DistServe.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace
from urllib.request import Request,urlopen
import uuid

from pdblend.bench.metering import Gpus
from pdblend.engine.launcher import Fleet
from pdblend.measure.power import trapezoid_mean_power
from pdblend.profile.wave import ProfileWave,atomic_json
from pdblend_runtime.cleanup import cleanup_owned
from pdblend_runtime.probe import NativeSpec
from ..native_profile import DIST_FREQS,call
from ..resident_campaign import model_load_lock,verify_endpoints,warmup_endpoints,drain_endpoints
from .stage_surface import fit_surface,measured_events


TRAIN_SHAPES=((16,),(128,),(512,),(2048,),(4096,),(7168,),
              (16,16),(4096,4096),(16,7168),(128,4096),
              (128,128,128,128),(2048,2048,2048,2048),
              (16,128,512,2048),(16,)*8,(1024,)*8)
HOLDOUT_SHAPES=((64,),(1024,),(3072,),(6144,),(64,512),(256,3584),
                (64,128,256,512),(64,)*8)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs_for(model,tp,gpus,base_port):
    expected=2 if model.endswith('32B-Instruct') else 1
    if tp!=expected or len(gpus)!=2*tp or len(set(gpus))!=len(gpus):
        raise ValueError('initial independent native profile requires TP1/TP1/TP2 resident pairs')
    return [NativeSpec('dist-profile-'+role,tuple(gpus[i*tp:(i+1)*tp]),base_port+16*i,model,tp=tp,
        max_num_seqs=32,extra_args=('--enforce-eager','--worker-cls',
                                   'pdblend_baselines.distserve.stage_worker.StageProfileWorker'))
            for i,role in enumerate(('prefill','decode'))]


def plan_points():
    return [dict(frequency_mhz=f,role=role,purpose=purpose,lengths=list(lengths),repeats=3)
            for f in DIST_FREQS for role in ('prefill','decode')
            for purpose,shapes in (('training',TRAIN_SHAPES),('holdout',HOLDOUT_SHAPES))
            for lengths in shapes]


def load_point_plan(path, model, tp):
    value=json.loads(Path(path).read_text())
    if (value.get('schema')!='distserve-targeted-stage-plan-v1' or value.get('system')!='distserve'
            or value.get('model_id')!=model or value.get('tp')!=tp or value.get('pp')!=1
            or value.get('selection_split') not in ('calibration','tuning')
            or value.get('evaluation_used_for_selection') is not False):
        raise ValueError('independent calibration/tuning-only stage point plan required')
    points=value.get('points',[])
    if not points or len({json.dumps(p,sort_keys=True) for p in points})!=len(points):
        raise ValueError('nonempty unique explicit stage points required')
    for point in points:
        if (set(point)!={'frequency_mhz','role','purpose','lengths','repeats'}
                or point['frequency_mhz'] not in DIST_FREQS or point['role'] not in ('prefill','decode')
                or point['purpose'] not in ('training','holdout') or point['repeats']!=3
                or not point['lengths'] or len(point['lengths'])>32
                or any(type(n)is not int or not 1<=n<=7936 for n in point['lengths'])):
            raise ValueError('targeted stage point violates frequency/repeat/shape protocol')
    return points


def verify_inputs(args):
    expected=json.loads(args.input_manifest.read_text())
    source_path=Path(os.environ['PDBLEND_SOURCE_MANIFEST'])
    bindings=dict(point_plan=sha(args.point_plan),source_manifest=sha(source_path),
                  model_verification=sha(Path(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT'])))
    if bindings!=expected['exact_inputs_sha256']:
        raise ValueError('frozen targeted stage input checksum differs')
    source=json.loads(source_path.read_text())
    source_sha=hashlib.sha256(json.dumps(source['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (source_sha!=source['source_sha256'] or source_sha!=expected['source_sha256']
            or source_sha!=os.environ['PDBLEND_SOURCE_SHA256']
            or expected['image_digest']!=os.environ['PDBLEND_IMAGE_ID']):
        raise ValueError('frozen targeted stage source/image differs')
    root=Path(__file__).resolve().parents[2]
    if any(sha(root/name)!=checksum for name,checksum in source['files'].items()):
        raise ValueError('frozen targeted stage execution source bytes differ')
    return expected


def _stream(url,payload,state,lock):
    request=Request(url+'/baseline/generate',data=json.dumps(payload).encode(),method='POST',
                    headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    with lock:
        state.update(request_id=payload['request_id'],input_tokens=len(payload['prompt']),
                     requested_output_tokens=payload['max_tokens'],submitted_s=time.time(),
                     events=[],stream_complete=False,done_marker=False)
    try:
        with urlopen(request,timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if not data:continue
                if data==b'[DONE]':
                    with lock:state['done_marker']=True
                    break
                event=json.loads(data)
                with lock:
                    tokens=event.get('token_ids');stamp=time.time()
                    if (state.get('finished') or event.get('error') or not isinstance(tokens,list)
                            or any(type(token)is not int or token<0 for token in tokens)
                            or event.get('token_index')!=state['tokens']+len(tokens)
                            or state['tokens']+len(tokens)>payload['max_tokens']):
                        raise ValueError('native stage client token stream is invalid or post-terminal')
                    state['tokens']+=len(tokens)
                    state['finished']=event.get('finished') is True
                    state['events'].append(dict(token_ids=tokens,token_index=event['token_index'],
                        finished=event.get('finished'),finish_reason=event.get('finish_reason'),
                        usage=event.get('usage'),received_s=stamp))
                    if state['finished']:
                        state['terminal_s']=stamp;state['terminal_usage']=event.get('usage')
                        state['finish_reason']=event.get('finish_reason')
            with lock:
                if not state['done_marker']:
                    raise ValueError('native stage stream lacks terminal event or DONE marker')
                if state.get('cleanup_cancel_requested_s') is not None and (
                        not state['finished'] or state['tokens']!=payload['max_tokens']):
                    state['cleanup_stream_closed']=True
                elif not state['finished'] or state['tokens']!=payload['max_tokens']:
                    raise ValueError('native stage stream terminated outside requested output budget')
                else:state['stream_complete']=True
    except Exception as exc:
        with lock:state.update(error=repr(exc),error_at_s=time.time())
    finally:
        with lock:state['finished_s']=time.time()


class ContextWindowExhausted(RuntimeError):
    """Exact natural context exhaustion is unsupported, never a fit sample."""


def check_decode_terminal(states,futures):
    if any(s.get('error') for s in states.values()):
        raise RuntimeError('native decode request failed during settled measurement')
    finished=[rid for rid,s in states.items() if s.get('finished')]
    if not finished:return
    for (rid,row),future in zip(states.items(),futures):
        if rid in finished:future.result(timeout=2.)
    if any(s.get('error') for s in states.values()):
        raise RuntimeError('native decode request failed during settled measurement')
    for rid in finished:
        row=states[rid];prompt=row.get('input_tokens');budget=row.get('requested_output_tokens')
        usage=row.get('terminal_usage',{}) or {}
        if (type(prompt)is not int or type(budget)is not int or prompt+budget!=8192
                or row.get('tokens')!=budget or row.get('stream_complete') is not True
                or row.get('done_marker') is not True or row.get('finish_reason')!='length'
                or usage.get('prompt_tokens')!=prompt or usage.get('completion_tokens')!=budget):
            raise RuntimeError('decode terminated without exact natural 8192-context completion')
    raise ContextWindowExhausted('context_window_exhausted')


def _submit_batch(url,lengths,role,pool,prefix):
    lock=threading.Lock();states={};futures=[]
    call(url,'POST','/baseline/control',dict(admit_prefill=False,admit_decode=False))
    try:
        for i,length in enumerate(lengths):
            rid=f'{prefix}-{i}';states[rid]=dict(tokens=0,finished=False)
            payload=dict(request_id=rid,prompt=list(range(100,100+length)),max_tokens=1 if role=='prefill' else 8192-length,
                         seed=9701,temperature=0,ignore_eos=True)
            futures.append(pool.submit(_stream,url,payload,states[rid],lock))
        deadline=time.monotonic()+60
        while not set(states)<=set(call(url,'GET','/baseline/state')['all_queue']):
            if any(s.get('error') for s in states.values()):raise RuntimeError('native stage request failed at admission')
            if time.monotonic()>deadline:raise TimeoutError('native stage admission barrier timed out')
            time.sleep(.02)
    finally:call(url,'POST','/baseline/control',dict(admit_prefill=True,admit_decode=True))
    return states,futures


def window(spec,meter,point,path,*,before_measure=None):
    """Arm measurement while idle, then settle active decode before >=5s service.

    The native server rejects scope changes with queued requests. CUDA samples
    include startup/settle steps; only the recorded service interval is fitted.
    """
    path=Path(path)
    if path.exists():return json.loads(path.read_text())
    url=spec.base_url;role=point['role'];lengths=point['lengths'];prefix='dist-stage-'+uuid.uuid4().hex
    capability=call(url,'GET','/baseline/capability');capacity=capability['state']
    if (len(lengths)>capacity['max_num_seqs'] or sum(lengths)+len(lengths)*1024>capacity['total_kv_tokens']*.9
            or role=='prefill' and sum(lengths)>capacity['max_num_batched_tokens']):
        result=dict(schema='distserve-stage-window-v1',status='unsupported_engine',point=point,
                    reason='actual native batch/token/KV capacity',capability=capability)
        atomic_json(path,result);return result
    clock=call(url,'POST','/baseline/clock',dict(frequency_mhz=point['frequency_mhz']))
    sampler=meter.sampler(spec.gpus,interval_s=.1)
    result=dict(schema='distserve-stage-window-v1',status='failed',point=point,capability=capability,
                clock_receipt=clock,window_id=prefix,cleanup_errors=[],settle_s=2.,required_measure_s=5.)
    states={};pool=ThreadPoolExecutor(max_workers=len(lengths))
    try:
        result['measurement_start']=call(url,'POST','/baseline/measurement/start',
                                         dict(system='distserve',scope='runner'))
        result['measurement_started_s']=time.time()
        if result['measurement_start'].get('acknowledged') is not True:
            raise RuntimeError('native stage measurement start was not acknowledged')
        if role=='decode':
            states,futures=_submit_batch(url,lengths,role,pool,prefix)
            deadline=time.monotonic()+120
            while any(s['tokens']==0 for s in states.values()):
                if any(s.get('error') for s in states.values()):raise RuntimeError('native decode request failed before settle')
                if time.monotonic()>deadline:raise TimeoutError('native decode did not become active')
                time.sleep(.02)
            result['decode_active_s']=time.time()
        result['settle_started_s']=time.time()
        time.sleep(2)
        result['settle_finished_s']=time.time()
        if role=='decode':check_decode_terminal(states,futures)
        if before_measure:before_measure()
        result['start_s']=time.time();sampler.start()
        if role=='decode':
            time.sleep(5)
            check_decode_terminal(states,futures)
        else:
            count=0
            while time.time()-result['start_s']<5:
                current,futures=_submit_batch(url,lengths,role,pool,prefix+'-'+str(count))
                states.update(current)
                for future in futures:future.result()
                if any(s.get('error') or not s.get('finished') for s in current.values()):
                    raise RuntimeError('native full-prefill workload failed')
                count+=1
        result['end_s']=time.time();sampler.stop()
        result['sample']=call(url,'GET','/baseline/measurement/samples')
        events=[r for r in measured_events(result['sample'],tp=spec.tp)
                if r['role']==role and result['start_s']<=r['at_s']<=result['end_s']]
        if not events or role=='decode' and len(events)<8:raise RuntimeError('insufficient actual CUDA stage steps')
        result.update(status='measured',decode_steps=len(events) if role=='decode' else 0,full_prefill_calls=len(events) if role=='prefill' else 0)
    except ContextWindowExhausted:
        result.update(status='unsupported_engine',reason='context_window_exhausted',
            observed_end_s=time.time(),exhausted_requests=[rid for rid,s in states.items() if s.get('finished')],
            termination_policy='exact natural requested-output completion at 8192 context; excluded from fitting')
        if 'start_s' in result:result['end_s']=result['observed_end_s']
    except Exception as exc:result['error']=repr(exc)
    finally:
        sampler.stop()
        for rid,state in states.items():
            if not state.get('finished'):
                try:
                    state['cleanup_cancel_requested_s']=time.time()
                    state['cleanup_cancel_receipt']=call(url,'POST','/baseline/cancel',dict(request_id=rid))
                    if state['cleanup_cancel_receipt'].get('acknowledged') is not True:
                        raise RuntimeError('native stage cleanup cancel unacknowledged')
                except Exception as exc:result['cleanup_errors'].append(repr(exc))
        pool.shutdown(wait=True,cancel_futures=True)
        try:
            if result.get('measurement_start',{}).get('acknowledged') is True and 'sample' not in result:
                result['sample']=call(url,'GET','/baseline/measurement/samples')
            call(url,'POST','/baseline/measurement/stop',{})
            result['drain']=call(url,'POST','/baseline/drain',dict(timeout_s=30))
            call(url,'POST','/baseline/control',dict(accepting=True,admit_prefill=True,admit_decode=True))
        except Exception as exc:result['cleanup_errors'].append(repr(exc))
        result.update(power_samples=sampler.samples,frequency_samples=sampler.frequency_samples,
                      power_metadata=sampler.power_metadata,sampler_error=sampler.error,
                      client_requests=list(states.values()))
        # A clean DONE after our own acknowledged cancellation is separate
        # from a terminal workload. Network errors/timeouts are never hidden.
        errors=[s for s in states.values() if s.get('error') or (
            s.get('cleanup_stream_closed') and s.get('cleanup_cancel_receipt',{}).get('acknowledged') is not True)]
        if errors:
            result.update(status='failed',error='native client request failed outside acknowledged cleanup cancellation')
        if result['cleanup_errors'] or sampler.error:result['status']='failed'
        atomic_json(path,result)
    return result


def rows_from_window(raw):
    if raw.get('status')!='measured':return []
    point=raw['point'];role=point['role'];tp=raw['capability']['tp']
    if raw['settle_s']<2 or raw['end_s']-raw['start_s']<5 or raw['sampler_error'] or raw['cleanup_errors']:
        raise ValueError('stage measurement timing/cleanup protocol failed')
    samples=[row for row in raw['power_samples'] if raw['start_s']<=row[0]<=raw['end_s']]
    clocks=[row for row in raw['frequency_samples'] if raw['start_s']<=row[0]<=raw['end_s']]
    if len(samples)<2 or not clocks or any(abs(f-point['frequency_mhz'])>30 for _,freqs in clocks for f in freqs):
        raise ValueError('stage window power/actual-frequency samples missing or out of range')
    events=[row for row in measured_events(raw['sample'],tp=tp)
            if row['role']==role and raw['start_s']<=row['at_s']<=raw['end_s']]
    if not events or role=='decode' and len(events)<8:raise ValueError('stage window lacks required CUDA steps')
    power=trapezoid_mean_power(samples)
    return [dict(role=role,frequency_mhz=point['frequency_mhz'],purpose=point['purpose'],
                 lengths=row['lengths'],latency_ms=row['latency_ms'],power_w=power,
                 window_id=raw['window_id']) for row in events]


class NativeStageWave(ProfileWave):
    async def probe(self,profiler,phase):
        ready={};loop=asyncio.get_running_loop()
        async def before(i,repeat):
            ready.setdefault(repeat,set()).add(i)
            marker=f'parallel-window-{repeat}-ready'
            if len(ready[repeat])==len(profiler.specs):self.write(marker,dict(time=time.time(),instances=sorted(ready[repeat])))
            await self.wait(marker)
        async def one(index,spec):
            repeats=[]
            for repeat in range(3):
                point=dict(frequency_mhz=2100,role='decode',lengths=[1024]*8,purpose='interference',repeat=repeat)
                callback=(lambda:asyncio.run_coroutine_threadsafe(before(index,repeat),loop).result(timeout=self.timeout_s)) if phase=='parallel' else None
                raw=await asyncio.to_thread(window,spec,profiler.meter,point,
                    profiler.out_dir/'samples'/f'interference-{phase}-{index}-{repeat}.json',before_measure=callback)
                rows=rows_from_window(raw)
                if not rows:raise RuntimeError('native interference probe failed')
                repeats.append(dict(start_s=raw['start_s'],end_s=raw['end_s'],step_seconds=statistics.median(r['latency_ms'] for r in rows)/1000,
                                    power_w=rows[0]['power_w']))
            return dict(step_seconds=statistics.median(r['step_seconds'] for r in repeats),power_w=statistics.median(r['power_w'] for r in repeats),repeats=repeats)
        if phase=='isolated':
            instances=[]
            for i,spec in enumerate(profiler.specs):instances.append(await one(i,spec))
        else:instances=await asyncio.gather(*(one(i,s) for i,s in enumerate(profiler.specs)))
        return dict(instances=instances,gpu_uuids=profiler.raw['environment']['gpu_uuids'])


async def collect_owned(args,specs):
    meter=Gpus(args.gpus,power_mode='instant');fleet=Fleet(specs,args.out/'logs');outer=meter.sampler(interval_s=.1)
    result=dict(status='failed',complete=False,formal_eligible=False,hardware_executed=False,cleanup_errors=[])
    ports=None
    try:
        if os.environ.get('PDBLEND_DIST_PORT_GUARD')=='1':
            from .stage_ports import PortReservations
            ports=PortReservations(os.environ['PDBLEND_PROFILE_WAVE'],os.environ['PDBLEND_PROFILE_MEMBER'],
                specs,args.out,environment_path=os.environ['PDBLEND_CONCURRENCY_ENVIRONMENT'])
            result['startup_ports']=ports.reserve()
        outer.start()
        with model_load_lock():
            for spec in specs:
                if ports is not None:ports.before_engine_start(spec)
                fleet[spec.instance_id].start();fleet[spec.instance_id].wait_ready(timeout_s=600)
                if ports is not None:ports.after_engine_ready(spec,fleet[spec.instance_id].process.pid)
        caps=await verify_endpoints(specs);await warmup_endpoints(specs,'dist-stage');result['hardware_executed']=True
        root=os.environ.get('PDBLEND_PROFILE_WAVE');member=os.environ.get('PDBLEND_PROFILE_MEMBER')
        if not root or not member:raise ValueError('coordinated native DistServe interference wave is required')
        wave=NativeStageWave(Path(root),member,timeout_s=7200)
        raw=dict(environment=dict(gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(',')))
        shim=SimpleNamespace(specs=specs,meter=meter,out_dir=args.out,raw=raw,
            profile_key=SimpleNamespace(as_dict=lambda:dict(system='distserve',model_id=args.model,tp=args.tp,pp=1)),
            _checkpoint=lambda:atomic_json(args.out/'progress.json',raw))
        await wave.qualify_external(shim)
        async def role_loop(spec,role):
            receipts=[]
            for point in args.points:
                if point['role']!=role:continue
                key=hashlib.sha256(json.dumps(point,sort_keys=True).encode()).hexdigest()[:20]
                for repeat in range(point['repeats']):
                    path=args.out/'samples'/f'{key}-{repeat}.json'
                    row=await asyncio.to_thread(window,spec,meter,dict(point,repeat=repeat),path)
                    receipts.append(dict(path=str(path.relative_to(args.out)),sha256=sha(path),status=row['status']))
                    if row['status']=='failed':raise RuntimeError('native DistServe point failed: '+str(path))
            return receipts
        async with wave.measurement():
            if wave.parallel:
                groups=await asyncio.gather(*(role_loop(spec,role) for spec,role in zip(specs,('prefill','decode'))))
            else:
                # The isolated probe measured one engine at a time. A failed
                # parallel check therefore serializes both local roles too.
                groups=[]
                for spec,role in zip(specs,('prefill','decode')):
                    groups.append(await role_loop(spec,role))
        bindings=[r for group in groups for r in group];training=[];holdout=[]
        for binding in bindings:
            for row in rows_from_window(json.loads((args.out/binding['path']).read_text())):
                (training if row['purpose']=='training' else holdout).append(row)
        external=args.out/'samples/external-interference.json';evidence=json.loads(external.read_text())
        # Serial fallback has a controlled cohort too. Its window ordering is
        # enforced by ProfileWave; it is not a claim that parallelism passed.
        qualified=evidence.get('cross_job') is True and (evidence.get('passed') is True or evidence.get('fallback')=='serial_cohort')
        receipt=dict(passed=qualified,external_interference_path=str(external.relative_to(args.out)),external_interference_sha256=sha(external),
                     cross_job=evidence.get('cross_job'),measured_mode='parallel' if wave.parallel else 'serial_cohort',
                     point_protocol='three independent >=5s windows; >=2s decode settle; actual clock, power and rank timing')
        qualification=args.out/'measurement-qualification.json';atomic_json(qualification,receipt)
        cap=caps[specs[0].instance_id]
        identity=dict(system='distserve',model_id=args.model,tp=args.tp,pp=1,capacity_tokens=cap['state']['total_kv_tokens'],
                      **{k:cap[k] for k in ('model_hash','tokenizer_hash','engine_revision','image_digest','source_revision')})
        surface=fit_surface(training,holdout,identity=identity,raw_bindings=bindings,
            measurement_qualification=dict(**receipt,receipt_path=str(qualification.relative_to(args.out)),receipt_sha256=sha(qualification)))
        atomic_json(args.out/'surface.json',surface)
        result.update(status='passed',complete=True,calibration_passed=surface['qualified'],
            observation='sampling_complete; formal coverage and calibration are separate gates',
            surface_path=str(args.out/'surface.json'),surface_sha256=sha(args.out/'surface.json'),raw_windows=len(bindings),
            unsupported_windows=sum(r['status']=='unsupported_engine' for r in bindings),
            final_drains=await drain_endpoints(specs))
    except BaseException as exc:
        result['error']=repr(exc)
        if ports is not None:
            try:ports.failure(exc)
            except BaseException as evidence_error:result['port_evidence_error']=repr(evidence_error)
        if 'wave' in locals():wave.write('error',dict(error=repr(exc)))
    finally:
        if ports is not None:
            try:ports.close()
            except BaseException as release_error:result['port_release_error']=repr(release_error)
        result['cleanup_errors']=cleanup_owned(fleet,meter,outer)
        if result['cleanup_errors'] or result.get('port_release_error'):result.update(status='failed',complete=False)
        atomic_json(args.out/'completion.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','gpus'):p.add_argument('--'+name,required=True)
    p.add_argument('--tp',type=int,required=True);p.add_argument('--base-port',type=int,required=True)
    p.add_argument('--point-plan',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(argv);args.gpus=[int(v) for v in args.gpus.split(',')];args.out.mkdir(parents=True,exist_ok=True)
    specs=specs_for(args.model,args.tp,args.gpus,args.base_port)
    binding=verify_inputs(args)
    args.points=load_point_plan(args.point_plan,args.model,args.tp)
    if args.preflight_only:
        result=dict(status='cpu_preflight_passed',hardware_executed=False,formal_eligible=False,
            points=len(args.points),windows=sum(p['repeats'] for p in args.points),source=binding,
            specs=[dict(id=s.instance_id,command=s.command(),gpus=s.gpus) for s in specs],
            point_plan_sha256=sha(args.point_plan),
            protocol=dict(frequencies=DIST_FREQS,repeats=3,settle_s=2,measure_s=5,parallel_interference_limit=.05))
        if os.environ.get('PDBLEND_DIST_PORT_GUARD')=='1':
            from .stage_ports import inventory,CONTRACT
            result['startup_ports']=dict(contract=CONTRACT,actual_static_ports=inventory(specs),
                reservations_executed=False,cohort_lease_binding_pending=True)
        atomic_json(args.out/'preflight.json',result)
    else:result=asyncio.run(collect_owned(args,specs))
    print(json.dumps(dict(status=result['status'],complete=result.get('complete',False))))
    return 0 if result['status'] in ('cpu_preflight_passed','passed') else 2


if __name__=='__main__':raise SystemExit(main())
