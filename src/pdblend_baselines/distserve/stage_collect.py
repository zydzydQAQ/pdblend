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


def _stream(url,payload,state,lock):
    request=Request(url+'/baseline/generate',data=json.dumps(payload).encode(),method='POST',
                    headers={'Content-Type':'application/json','Accept':'text/event-stream'})
    try:
        with urlopen(request,timeout=180) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if not data or data==b'[DONE]':continue
                event=json.loads(data)
                with lock:
                    state['tokens']+=len(event.get('token_ids',[]))
                    state['finished']=bool(event.get('finished'))
    except Exception as exc:
        with lock:state['error']=repr(exc)


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
    """A real >=5s active window; decoder is already running before 2s settle."""
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
        if role=='decode':
            states,futures=_submit_batch(url,lengths,role,pool,prefix)
            deadline=time.monotonic()+120
            while any(s['tokens']==0 for s in states.values()):
                if any(s.get('error') for s in states.values()):raise RuntimeError('native decode request failed before settle')
                if time.monotonic()>deadline:raise TimeoutError('native decode did not become active')
                time.sleep(.02)
        time.sleep(2)
        if before_measure:before_measure()
        call(url,'POST','/baseline/measurement/start',dict(system='distserve',scope='runner'))
        result['start_s']=time.time();sampler.start()
        if role=='decode':
            time.sleep(5)
            if any(s.get('finished') or s.get('error') for s in states.values()):
                raise RuntimeError('decode terminated before complete settled measurement window')
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
    except Exception as exc:result['error']=repr(exc)
    finally:
        sampler.stop()
        for rid,state in states.items():
            if not state.get('finished'):
                try:call(url,'POST','/baseline/cancel',dict(request_id=rid))
                except Exception as exc:result['cleanup_errors'].append(repr(exc))
        pool.shutdown(wait=True,cancel_futures=True)
        try:
            call(url,'POST','/baseline/measurement/stop',{})
            result['drain']=call(url,'POST','/baseline/drain',dict(timeout_s=30))
            call(url,'POST','/baseline/control',dict(accepting=True,admit_prefill=True,admit_decode=True))
        except Exception as exc:result['cleanup_errors'].append(repr(exc))
        result.update(power_samples=sampler.samples,frequency_samples=sampler.frequency_samples,
                      power_metadata=sampler.power_metadata,sampler_error=sampler.error)
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
    try:
        outer.start()
        with model_load_lock():
            for spec in specs:
                fleet[spec.instance_id].start();fleet[spec.instance_id].wait_ready(timeout_s=600)
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
            for point in plan_points():
                if point['role']!=role:continue
                key=hashlib.sha256(json.dumps(point,sort_keys=True).encode()).hexdigest()[:20]
                for repeat in range(point['repeats']):
                    path=args.out/'samples'/f'{key}-{repeat}.json'
                    row=await asyncio.to_thread(window,spec,meter,dict(point,repeat=repeat),path)
                    receipts.append(dict(path=str(path),sha256=sha(path),status=row['status']))
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
            for row in rows_from_window(json.loads(Path(binding['path']).read_text())):
                (training if row['purpose']=='training' else holdout).append(row)
        external=args.out/'samples/external-interference.json';evidence=json.loads(external.read_text())
        # Serial fallback has a controlled cohort too. Its window ordering is
        # enforced by ProfileWave; it is not a claim that parallelism passed.
        qualified=evidence.get('cross_job') is True and (evidence.get('passed') is True or evidence.get('fallback')=='serial_cohort')
        receipt=dict(passed=qualified,external_interference_path=str(external),external_interference_sha256=sha(external),
                     cross_job=evidence.get('cross_job'),measured_mode='parallel' if wave.parallel else 'serial_cohort',
                     point_protocol='three independent >=5s windows; >=2s decode settle; actual clock, power and rank timing')
        qualification=args.out/'measurement-qualification.json';atomic_json(qualification,receipt)
        cap=caps[specs[0].instance_id]
        identity=dict(system='distserve',model_id=args.model,tp=args.tp,pp=1,capacity_tokens=cap['state']['total_kv_tokens'],
                      **{k:cap[k] for k in ('model_hash','tokenizer_hash','engine_revision','image_digest','source_revision')})
        surface=fit_surface(training,holdout,identity=identity,raw_bindings=bindings,
            measurement_qualification=dict(**receipt,receipt_path=str(qualification),receipt_sha256=sha(qualification)))
        atomic_json(args.out/'surface.json',surface)
        result.update(status='collected',complete=True,calibration_passed=surface['qualified'],
            surface_path=str(args.out/'surface.json'),surface_sha256=sha(args.out/'surface.json'),raw_windows=len(bindings),
            unsupported_windows=sum(r['status']=='unsupported_engine' for r in bindings),
            final_drains=await drain_endpoints(specs))
    except BaseException as exc:
        result['error']=repr(exc)
        if 'wave' in locals():wave.write('error',dict(error=repr(exc)))
    finally:
        result['cleanup_errors']=cleanup_owned(fleet,meter,outer)
        if result['cleanup_errors']:result.update(status='failed',complete=False)
        atomic_json(args.out/'completion.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','gpus'):p.add_argument('--'+name,required=True)
    p.add_argument('--tp',type=int,required=True);p.add_argument('--base-port',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(argv);args.gpus=[int(v) for v in args.gpus.split(',')];args.out.mkdir(parents=True,exist_ok=True)
    specs=specs_for(args.model,args.tp,args.gpus,args.base_port)
    if args.preflight_only:
        result=dict(status='cpu_preflight_passed',hardware_executed=False,formal_eligible=False,
            points=len(plan_points()),windows=sum(p['repeats'] for p in plan_points()),
            specs=[dict(id=s.instance_id,command=s.command(),gpus=s.gpus) for s in specs],
            training_shapes=TRAIN_SHAPES,holdout_shapes=HOLDOUT_SHAPES,
            protocol=dict(frequencies=DIST_FREQS,repeats=3,settle_s=2,measure_s=5,parallel_interference_limit=.05))
        atomic_json(args.out/'preflight.json',result)
    else:result=asyncio.run(collect_owned(args,specs))
    print(json.dumps(dict(status=result['status'],complete=result.get('complete',False))))
    return 0 if result['status'] in ('cpu_preflight_passed','collected') else 2


if __name__=='__main__':raise SystemExit(main())
