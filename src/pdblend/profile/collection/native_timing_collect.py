"""PDblend-owned native CUDA timing windows on one immutable resident fleet."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import statistics
import time
import uuid

import aiohttp
from .native_timing_plan import binding,read_bound,validate_plan,digest
from .native_timing_audit import audit_window,fit_component,need
from .native_frequency_domain import (plan_frequencies, point_fields, with_domain,
                                      validate_collection_inputs, require_same_domain)
from .wave import atomic_json

WORKER='pdblend.profile.collection.native_timing_worker.PDNativeTimingWorker'


def validate_timing_plan(plan):
    if plan.get('schema')=='pdblend-native-timing-plan/v2':
        from .native_timing_plan_v2 import validate_plan as validate_v2
        return validate_v2(plan)
    return validate_plan(plan)


def resident_specs(args,plan):
    """Partition the owned eight physical devices into model-owned replicas."""
    from pdblend_runtime.probe import NativeSpec
    tp=plan.get('tp',1)
    need(type(tp) is int and tp in (1,2) and len(args.gpus)==len(set(args.gpus))==8,
         'native timing requires one distinct eight-GPU fleet')
    need(plan['model_id']==Path(args.model).name and plan.get('pp',1)==1,
         'native timing model/topology differs')
    if plan.get('schema')=='pdblend-native-timing-plan/v2':
        from .native_timing_plan_v2 import MODEL_TP
        need(MODEL_TP.get(plan['model_id'])==tp and plan['resident_instances']==8//tp,
             'v2 native timing model-owned inventory differs')
    else:need(tp==1,'v1 timing retains its TP1 inventory')
    return [NativeSpec('pd-timing-'+str(i),tuple(args.gpus[i*tp:(i+1)*tp]),
        args.base_port+i*4,args.model,tp=tp,max_num_seqs=32,
        extra_args=('--enforce-eager','--worker-cls',WORKER)) for i in range(8//tp)]


async def capacity_before_window(spec,point,path,plan,identity):
    """Record fresh capacity before any request or measurement is submitted."""
    from pdblend_runtime.probe import call
    from .native_timing_capacity import capacity_decision,unsupported_capacity_record
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        drain=await call(session,spec.base_url,'/baseline/drain',dict(timeout_s=30));drain_s=time.time()
        capability=await call(session,spec.base_url,'/baseline/capability');capability_s=time.time()
    receipts=dict(identity=identity,capability=capability,capability_received_s=capability_s,
        drain=drain,drain_received_s=drain_s,observed_s=time.time())
    decision=capacity_decision(plan,point,**receipts)
    if decision['supported']:return None
    raw=unsupported_capacity_record(plan,point,**receipts)
    path=Path(path)
    need(not path.exists(),'new native capacity receipt required')
    atomic_json(path,raw)
    return raw


def adopt_warmup_generations(specs, fleet, receipts):
    """Use real warmup epochs for runtime control and subsequent reloads."""
    by_id={r['instance_id']:r for r in receipts}
    need(len(by_id)==len(receipts)==len(specs) and set(by_id)=={s.instance_id for s in specs},
         'warmup generation inventory differs')
    updated=[]
    for spec in specs:
        row=by_id[spec.instance_id];generation=row.get('generation');ack=row.get('control',{})
        need(type(generation) is int and generation>spec.generation
             and ack.get('acknowledged') is True and ack.get('generation')==generation,
             'warmup generation lacks its actual native ACK')
        updated.append(replace(spec,generation=generation))
    need(len({s.generation for s in updated})==1,'communicating native peers must share one warmup epoch')
    for spec in updated:fleet[spec.instance_id].spec=spec
    return updated


async def verify_compute_empty(fleet, meter, gpu_uuids, *, timeout_s=60.):
    """Keep this lease until physical CUDA processes have actually disappeared."""
    report=dict(passed=False,started_s=time.time(),observations=[])
    try:
        need(len(meter.gpus)==len(set(meter.gpus))==len(gpu_uuids)==len(set(gpu_uuids))==8,
             'cleanup must cover all eight distinct physical GPUs')
        need(all(not i.alive() for i in fleet.instances.values()),'owned server process remains alive')
        deadline=time.monotonic()+timeout_s
        while True:
            devices=[]
            for gpu,expected in zip(meter.gpus,gpu_uuids):
                uuid=meter.backend.gpu_uuid(gpu)
                need(uuid==expected,'cleanup physical UUID binding differs')
                pids=[int(p.pid) for p in meter.backend._nvml.nvmlDeviceGetComputeRunningProcesses(
                    meter.backend._handle(gpu))]
                devices.append(dict(gpu=gpu,gpu_uuid=uuid,compute_pids=pids))
            report['observations'].append(dict(at_s=time.time(),devices=devices))
            if all(not d['compute_pids'] for d in devices):
                report['passed']=True;break
            if time.monotonic()>=deadline:raise TimeoutError('CUDA processes remain on the owned lease')
            await asyncio.sleep(.1)
    except BaseException as exc:report['error']=repr(exc)
    report['finished_s']=time.time()
    return report


async def gather_owned(awaitables):
    tasks=[asyncio.create_task(a) for a in awaitables]
    try:return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise


def measurement_barrier(count):
    # Python 3.10 in the pinned container has no asyncio.Barrier.
    event=asyncio.Event();ready=0
    async def wait():
        nonlocal ready
        ready+=1
        if ready==count:event.set()
        await event.wait()
    return wait


async def _request(session,url,payload,row):
    from pdblend.results.journal import payload_receipt
    events=[];row['submitted_s']=time.time()
    try:
        async with session.post(url+'/baseline/generate',json=payload) as response:
            if response.status!=200:raise RuntimeError(await response.text())
            async for line in response.content:
                if not line.startswith(b'data:'):continue
                data=line[5:].strip()
                if data==b'[DONE]':break
                event=json.loads(data);stamp=time.time()
                events.append(dict(token_ids=event.get('token_ids',[]),token_index=event.get('token_index'),
                                   finished=event.get('finished'),received_s=stamp))
                row['seen_tokens']=sum(len(e['token_ids']) for e in events)
                row.setdefault('first_token_s',stamp)
    except BaseException as exc:
        row['error']=repr(exc);raise
    finally:
        row['finished_s']=time.time();row['events']=events
        row.update(payload_receipt(events,journal_path='embedded:events',request_id=payload['request_id']))


async def burst(session,url,point,prefix,clients):
    """Hold decode until every identical-length prefill has completed.

    The gate aligns real decode contexts; no fake lengths or CUDA samples.
    """
    from pdblend_runtime.probe import call
    tasks=[];current=[]
    await call(session,url,'/baseline/control',dict(accepting=True,admit_prefill=False,admit_decode=False))
    try:
        rng=random.Random(point['seed'])
        prompt=[rng.randrange(100,60000) for _ in range(point['prompt_tokens'])]
        for i in range(point['batch']):
            rid=prefix+'-'+str(i);row=dict(request_id=rid,seen_tokens=0);clients.append(row);current.append(row)
            payload=dict(request_id=rid,prompt=prompt,max_tokens=point['output_tokens'],
                         seed=point['seed'],temperature=0,ignore_eos=True)
            tasks.append(asyncio.create_task(_request(session,url,payload,row)))
        deadline=time.monotonic()+120
        while not {r['request_id'] for r in current}<=set((await call(session,url,'/baseline/state'))['all_queue']):
            if any(t.done() for t in tasks):raise RuntimeError('request failed before admission barrier')
            if time.monotonic()>deadline:raise TimeoutError('native admission barrier expired')
            await asyncio.sleep(.01)
        await call(session,url,'/baseline/control',dict(admit_prefill=True,admit_decode=False))
        while any(r['seen_tokens']==0 for r in current):
            if any(r.get('error') for r in current):raise RuntimeError('prefill failed before decode barrier')
            if time.monotonic()>deadline:raise TimeoutError('native prefill barrier expired')
            await asyncio.sleep(.01)
        await call(session,url,'/baseline/control',dict(admit_prefill=True,admit_decode=True))
        await asyncio.wait_for(asyncio.gather(*tasks),120)
        need(all(r.get('terminal') and r.get('completion_tokens')==point['output_tokens'] for r in current),
             'native burst output budget incomplete')
    finally:
        await call(session,url,'/baseline/control',dict(admit_prefill=True,admit_decode=True))
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        for row in current:
            if not row.get('terminal'):
                await call(session,url,'/baseline/cancel',dict(request_id=row['request_id']))


async def window(spec,meter,point,path,*,before_measure=None):
    from pdblend_runtime.probe import call
    path=Path(path)
    if path.exists():raise FileExistsError('new native timing window required')
    raw=dict(schema='pdblend-native-timing-window-v1',system='pdblend',scope='native_cuda_timing_component_only',
             status='failed',point=point,window_id='pd-native-'+uuid.uuid4().hex,cleanup_errors=[],client_requests=[])
    sampler=meter.sampler(spec.gpus,interval_s=.1);url=spec.base_url
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        try:
            raw['capability']=await call(session,url,'/baseline/capability')
            capacity=raw['capability']['state']
            need(point['batch']<=capacity['max_num_seqs'] and point['batch']*(point['prompt_tokens']+point['output_tokens'])<capacity['total_kv_tokens']*.9,
                 'requested batch/context exceeds actual native KV capacity')
            raw['clock_receipt']=await call(session,url,'/baseline/clock',dict(frequency_mhz=point['frequency_mhz']))
            await call(session,url,'/baseline/drain',dict(timeout_s=30))
            await call(session,url,'/baseline/control',dict(accepting=True,admit_prefill=True,admit_decode=True))
            raw['measurement_start']=await call(session,url,'/baseline/measurement/start',dict(system='pdblend',scope='runner'))
            raw['measurement_started_s']=time.time()
            need(raw['measurement_start'].get('acknowledged') is True,'idle measurement arm lacks ACK')
            raw['settle_started_s']=time.time();count=0
            while time.time()-raw['settle_started_s']<2.:
                await burst(session,url,point,raw['window_id']+'-'+str(count),raw['client_requests']);count+=1
            raw['settle_finished_s']=time.time()
            if before_measure:await before_measure()
            sampler.start();raw['start_s']=time.time()
            while time.time()-raw['start_s']<5.:
                await burst(session,url,point,raw['window_id']+'-'+str(count),raw['client_requests']);count+=1
            raw['end_s']=time.time();sampler.stop()
            raw['sample']=await call(session,url,'/baseline/measurement/samples')
            raw['status']='measured'
        except BaseException as exc:raw['error']=repr(exc)
        finally:
            sampler.stop()
            try:
                raw['measurement_stop']=await call(session,url,'/baseline/measurement/stop',{})
                raw['drain']=await call(session,url,'/baseline/drain',dict(timeout_s=30))
                await call(session,url,'/baseline/control',dict(accepting=True,admit_prefill=True,admit_decode=True))
            except BaseException as exc:raw['cleanup_errors'].append(repr(exc))
            raw.update(power_samples=sampler.samples,frequency_samples=sampler.frequency_samples,
                       power_metadata=sampler.power_metadata,sampler_error=sampler.error,
                       power_scope='prefill_decode_burst_auxiliary_not_pure_decode_power',formal_eligible=False)
            if raw['cleanup_errors'] or sampler.error:raw['status']='failed'
            atomic_json(path,raw)
    return raw


def preflight(args):
    from pdblend.source_inventory import source_root
    plan=validate_timing_plan(json.loads(args.point_plan.read_text()));expected=json.loads(args.input_manifest.read_text())
    input_schema='pdblend-native-timing-inputs/v2' if plan.get('schema')=='pdblend-native-timing-plan/v2' else 'pdblend-native-timing-inputs-v1'
    need(expected.get('schema')==input_schema and expected.get('model_id')==plan['model_id']
         and expected.get('system')=='pdblend','native timing invocation schema/model identity differs')
    validate_frequency_invocation(args, plan, expected)
    source_path=Path(os.environ['PDBLEND_SOURCE_MANIFEST']);source=json.loads(source_path.read_text())
    need(expected['point_plan']==binding(args.point_plan) and expected['source_manifest']==binding(source_path), 'frozen invocation input differs')
    need(digest(source['files'])==source['source_sha256']==os.environ['PDBLEND_SOURCE_SHA256']
         and expected['image_digest']==os.environ['PDBLEND_IMAGE_ID'],'actual source/image differs')
    for name,sha in source['files'].items():need(binding(source_root()/name)['sha256']==sha,'source bytes differ: '+name)
    need(expected['model_verification']==binding(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']), 'model receipt differs')
    resident_specs(args,plan)
    need(expected.get('collect_runtime', False) is getattr(args, 'collect_runtime', False),
         'resident runtime invocation differs from frozen inputs')
    pilot_path = getattr(args, 'power_pilot_plan', None)
    need(expected.get('power_pilot_plan') == (binding(pilot_path) if pilot_path else None),
         'resident power pilot invocation differs from frozen inputs')
    if getattr(args, 'collect_runtime', False):
        from .native_runtime_collect import build_runtime_plan
        need(expected.get('runtime_plan') == build_runtime_plan(plan.get('frequency_domain_ref')),
             'runtime calibration/holdout split differs from frozen inputs')
    if pilot_path:
        from .native_power_plan import validate_plan as validate_power_plan
        from .native_power_collect import collect_power_pilot
        pilot = validate_power_plan(json.loads(pilot_path.read_text()))
        need(pilot['model_id'] == plan['model_id'] and pilot['query_ledger'] == plan['query_ledger'],
             'power pilot and timing must use the same model-owned non-evaluation queries')
    validate_cycle_invocation(args, plan, expected)
    validate_layout_invocation(args, plan, expected)
    validate_phase_order(args, plan, expected)
    return plan,expected


def validate_frequency_invocation(args, plan, expected):
    return validate_collection_inputs(plan, expected,
        collect_runtime=getattr(args,'collect_runtime',False),
        power_pilot=bool(getattr(args,'power_pilot_plan',None)),
        request_cycles=bool(getattr(args,'request_cycle_plan',None)),
        layout_energy=bool(getattr(args,'layout_energy_plan',None)))


def resident_phase_order(*, collect_runtime=False, power_pilot=False, request_cycles=False,
                         layout_energy=False, timing_first=False):
    phases = ['runtime'] if collect_runtime else []
    supplements = (['power_pilot'] if power_pilot else []) + (['request_cycles'] if request_cycles else [])
    phases += ['timing'] + supplements if timing_first else supplements + ['timing']
    return phases + (['layout_energy'] if layout_energy else [])


def invocation_phase_order(args):
    return resident_phase_order(collect_runtime=getattr(args, 'collect_runtime', False),
        power_pilot=bool(getattr(args, 'power_pilot_plan', None)),
        request_cycles=bool(getattr(args, 'request_cycle_plan', None)),
        layout_energy=bool(getattr(args, 'layout_energy_plan', None)),
        timing_first=getattr(args, 'timing_first', False))


def validate_phase_order(args, plan, expected):
    first = getattr(args, 'timing_first', False)
    need(type(first) is bool and expected.get('timing_first', False) is first,
         'resident phase order flag differs from frozen inputs')
    need(not first or plan.get('schema') == 'pdblend-native-timing-plan/v2',
         'independent timing stage requires the v2 point plan')
    order = invocation_phase_order(args)
    need(expected.get('phase_order', order if not first else None) == order,
         'resident phase order differs from frozen inputs')
    return order


def begin_phase(report, name):
    events = report.setdefault('phase_events', [])
    need(not events or events[-1]['status'] != 'running', 'previous resident phase is still running')
    report['active_phase'] = name
    events.append(dict(phase=name, started_s=time.time(), status='running'))


def finish_phase(report, error=None):
    event = report['phase_events'][-1]
    need(event['phase'] == report['active_phase'] and event['status'] == 'running',
         'resident phase completion does not match its start')
    event.update(finished_s=time.time(), status='failed' if error is not None else 'passed')
    if error is not None:
        event['error'] = repr(error)
    return event['finished_s']


def validate_cycle_invocation(args, plan, expected):
    path = getattr(args, 'request_cycle_plan', None)
    need(expected.get('request_cycle_plan') == (binding(path) if path else None),
         'resident request-cycle invocation differs from frozen inputs')
    if path is None:
        return None
    from .native_serving_cycles import validate_cycle_plan
    cycle = validate_cycle_plan(read_bound(expected['request_cycle_plan']))
    need(all(cycle.get(key) == plan.get(key) for key in ('model_id', 'tp', 'pp', 'query_ledger')),
         'request-cycle and timing model/topology/query identity differs')
    provenance = plan.get('query_provenance', plan.get('query_bindings'))
    if provenance is not None:
        need(cycle['query_provenance'] == provenance,
             'request-cycle tuning provenance differs from timing')
    return cycle


def validate_layout_invocation(args, plan, expected):
    path = getattr(args, 'layout_energy_plan', None)
    need(expected.get('layout_energy_plan') == (binding(path) if path else None),
         'resident layout-energy invocation differs from frozen inputs')
    if path is None:
        return None
    need(getattr(args, 'collect_runtime', False) is True
         and plan.get('schema') == 'pdblend-native-timing-plan/v2'
         and not getattr(args, 'power_pilot_plan', None)
         and not getattr(args, 'request_cycle_plan', None),
         'layout-energy requires runtime and v2 timing, without legacy pilot/cycles')
    from .native_layout_energy import validate_layout_plan
    layout = validate_layout_plan(read_bound(expected['layout_energy_plan']))
    require_same_domain(plan,layout)
    need(all(layout.get(key) == plan.get(key) for key in ('model_id', 'tp', 'pp', 'query_ledger'))
         and layout['query_provenance'] == plan.get('query_provenance'),
         'layout-energy model/topology/query provenance differs from timing')
    return layout


async def collect_cycle_supplement(args, specs, fleet, meter, sampler, report):
    """Use the loaded inventory; operational recovery and fit quality are separate."""
    from .native_serving_cycles import collect_and_fit_serving_cycles
    from pdblend_baselines.resident_campaign import verify_endpoints, drain_endpoints
    result = await collect_and_fit_serving_cycles(
        specs, fleet, meter, sampler, args.out/'request-cycles',
        gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(','),
        plan=read_bound(binding(args.request_cycle_plan)))
    report['resident_request_cycles'] = binding(args.out/'request-cycles/completion.json')
    need(result.get('ready_for_timing') is True and result.get('safe_restore_passed') is True
         and result.get('operational_failure') is False,
         'request-cycle supplement did not safely restore the resident timing inventory')
    report['post_cycle_capabilities'] = await verify_endpoints(specs)
    report['post_cycle_drains'] = await drain_endpoints(specs)


async def collect_energy_supplements(args, specs, fleet, meter, sampler, report):
    from pdblend_baselines.resident_campaign import verify_endpoints, drain_endpoints
    if getattr(args, 'power_pilot_plan', None):
        from .native_power_collect import collect_power_pilot
        begin_phase(report, 'power_pilot')
        pilot = await collect_power_pilot(specs, fleet, meter, sampler, args.out/'power-pilot',
            gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(','),
            plan=json.loads(args.power_pilot_plan.read_text()))
        report['resident_power_pilot'] = binding(args.out/'power-pilot/completion.json')
        need(pilot.get('ready_for_timing') is True,
             'power pilot did not safely restore the resident timing inventory')
        report['post_pilot_capabilities'] = await verify_endpoints(specs)
        report['post_pilot_drains'] = await drain_endpoints(specs)
        finish_phase(report)
    if getattr(args, 'request_cycle_plan', None):
        begin_phase(report, 'request_cycles')
        await collect_cycle_supplement(args, specs, fleet, meter, sampler, report)
        finish_phase(report)


async def collect_layout_supplement(args, specs, fleet, meter, sampler, report):
    """Qualify the pre-layout timing stage without releasing its live Fleet."""
    from .native_layout_stage import capture_resident_timing
    from .native_layout_energy import collect_and_replay_layout_energy
    from pdblend.profile.query.native_composition import IDENTITY
    from pdblend.profile.query.native_layout_profile import (freeze_layout_timing_selection,
                                                            LayoutQualificationUnavailable)
    from pdblend_baselines.resident_campaign import verify_endpoints, drain_endpoints
    if report.get('component_qualified') is not True:
        report['resident_layout_energy_status'] = dict(status='blocked',
            reason='native_timing_independent_holdout_unqualified',
            hardware_executed=False,formal_eligible=False)
        return
    stage_ref = capture_resident_timing(report, input_manifest_ref=binding(args.input_manifest),
        attempt_manifest_ref=binding(args.out.parent/'manifest.json'), specs=specs, fleet=fleet,
        out=args.out/'resident-timing-stage.json')
    stage_field = 'resident_layout_timing_stage' if getattr(args, 'timing_first', False) else 'resident_timing_stage'
    report[stage_field] = stage_ref
    inputs = read_bound(binding(args.input_manifest))
    fitted = read_bound(report['timing_component'])
    actual_identity = fitted['component']['identity']
    selection_identity={key:actual_identity[key] for key in IDENTITY}
    if 'frequency_domain' in actual_identity:
        selection_identity=with_domain(selection_identity,actual_identity['frequency_domain'])
    require_same_domain(actual_identity,selection_identity)
    try:
        timing_ref = freeze_layout_timing_selection(
            identity=selection_identity, timing_ref=stage_ref,
            runtime_ref=report['resident_runtime'], serving_source_manifest=inputs['source_manifest'],
            calibration_source_manifests=[inputs['source_manifest']], out=args.out/'layout-timing-selection.json')
    except LayoutQualificationUnavailable as exc:
        report['resident_layout_energy_status'] = dict(status='blocked',reason=str(exc),
            gate=exc.gate,evidence=exc.evidence,hardware_executed=False,formal_eligible=False)
        return
    report['layout_timing_selection'] = timing_ref
    result = await collect_and_replay_layout_energy(specs, fleet, meter, sampler,
        args.out/'layout-energy', gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(','),
        plan=read_bound(binding(args.layout_energy_plan)), timing_profile_ref=timing_ref)
    report['resident_layout_energy'] = binding(args.out/'layout-energy/completion.json')
    need(result.get('ready_for_next') is True and result.get('safe_restore_passed') is True
         and result.get('operational_failure') is False,
         'layout-energy supplement failed its collection or safe restoration boundary')
    report['post_layout_capabilities'] = await verify_endpoints(specs)
    report['post_layout_drains'] = await drain_endpoints(specs)


async def collect(args,plan):
    from pdblend.engine.launcher import Fleet
    from pdblend.bench.metering import Gpus
    from pdblend_runtime.cleanup import cleanup_owned
    from pdblend_baselines.resident_campaign import verify_endpoints,drain_endpoints,warmup_endpoints,model_load_lock
    if 'frequency_domain' in plan:
        validate_frequency_invocation(args,plan,json.loads(args.input_manifest.read_text()))
    specs=resident_specs(args,plan)
    v2=plan.get('schema')=='pdblend-native-timing-plan/v2'
    meter=Gpus(args.gpus,power_mode='instant');sampler=meter.sampler(interval_s=.1);fleet=Fleet(specs,args.out/'logs')
    report=dict(schema='pdblend-native-timing-collection/v2' if v2 else 'pdblend-native-timing-collection-v1',system='pdblend',status='failed',complete=False,
        formal_eligible=False,energy_comparable=False,hardware_executed=False,raw_bindings=[],cleanup_errors=[],
        phase_order=invocation_phase_order(args),active_phase='startup')
    if v2:report.update(point_plan=binding(args.point_plan),capacity_policy=plan['capacity_policy'],window_owners=[])
    if 'frequency_domain' in plan:
        report.update({key:plan[key] for key in ('model_id','tp','pp','frequency_domain_ref',
                                                'frequency_domain','frequency_domain_sha256')})
    try:
        begin_phase(report, 'startup')
        sampler.start()
        with model_load_lock():
            for spec in specs:
                fleet[spec.instance_id].start();fleet[spec.instance_id].wait_ready(timeout_s=900)
        report['pre_warmup_capabilities']=await verify_endpoints(specs)
        report['warmup']=await warmup_endpoints(specs,'pd-native-timing')
        specs=adopt_warmup_generations(specs,fleet,report['warmup'])
        caps=await verify_endpoints(specs)
        report['post_warmup_drains']=await drain_endpoints(specs)
        report.update(hardware_executed=True,capabilities=caps,
            actual_launch=[dict(spec=asdict(s),argv=s.command(),environment={k:s.environment().get(k) for k in
                ('CUDA_VISIBLE_DEVICES','VLLM_USE_V1','NCCL_CUMEM_ENABLE','NCCL_IB_DISABLE','NCCL_P2P_DISABLE')}) for s in specs])
        finish_phase(report)
        if getattr(args, 'collect_runtime', False):
            from .native_runtime_collect import collect_runtime
            begin_phase(report, 'runtime')
            runtime_options = (dict(frequency_domain_ref=plan['frequency_domain_ref'],include_transfer=False)
                               if 'frequency_domain' in plan else {})
            runtime = await collect_runtime(specs, fleet, meter, sampler, args.out/'runtime',
                gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(','),**runtime_options)
            report['resident_runtime'] = binding(args.out/'runtime/completion.json')
            need(runtime.get('ready_for_timing') is True,
                 'runtime supplement did not safely restore the resident timing inventory')
            # A physical off/wake is part of the algorithm's measured mechanism.
            # Re-query all replicas after it; no cached generation/KV is reused.
            report['post_runtime_capabilities'] = await verify_endpoints(specs)
            report['post_runtime_drains'] = await drain_endpoints(specs)
            finish_phase(report)
        if not getattr(args, 'timing_first', False):
            await collect_energy_supplements(args, specs, fleet, meter, sampler, report)
        begin_phase(report, 'timing')
        identities={s.instance_id:{k:caps[s.instance_id][k] for k in
            ('model_id','model_hash','tokenizer_hash','engine_revision','source_revision','image_digest','tp','pp','gpu_uuids')} for s in specs}
        async def sample(spec,point,path,barrier=None):
            raw=await window(spec,meter,point,path,before_measure=barrier)
            rows=audit_window(raw,identity=identities[spec.instance_id])
            watts=[sum(v) for t,v in raw['power_samples'] if raw['start_s']<=t<raw['end_s']]
            need(watts,'auxiliary interference power missing')
            return raw,rows,dict(latency_ms=statistics.median(r['latency_ms'] for r in rows),
                                 power_w=statistics.mean(watts),start_s=raw['start_s'],end_s=raw['end_s'])
        isolated={};parallel={};checks=[]
        for frequency in plan_frequencies(plan):
            # Isolated and concurrent observations must have the same idle
            # peer clocks. Set the whole inventory before either measurement.
            boundary=dict(frequency_mhz=frequency,started_s=time.time(),status='incomplete',clocks=[],observations=[])
            report.setdefault('interference_peer_states',[]).append(boundary)
            peer_drains=await drain_endpoints(specs);boundary['drains']=peer_drains
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
                from pdblend_runtime.probe import call
                peer_clocks=boundary['clocks']
                for spec in specs:
                    clock=await call(session,spec.base_url,'/baseline/clock',dict(frequency_mhz=frequency))
                    need(clock.get('acknowledged') is True and clock.get('success') is True
                         and clock.get('requested_frequency_mhz')==frequency,
                         'interference peer clock lacks requested-frequency ACK')
                    peer_clocks.append(dict(instance_id=spec.instance_id,clock=clock))
            frequency_observations=boundary['observations'];deadline=time.monotonic()+5
            while True:
                observed=[meter.current_freq(g) for g in args.gpus]
                frequency_observations.append(dict(at_s=time.time(),frequencies_mhz=observed))
                if all(abs(f-frequency)<=15 for f in observed):break
                need(time.monotonic()<deadline,'interference peer clocks failed to stabilize')
                await asyncio.sleep(.05)
            settle_started_s=time.time();await asyncio.sleep(2.)
            boundary.update(settle_started_s=settle_started_s,observed_s=time.time(),status='passed')
            for repeat in range(3):
                point=dict(role='decode',batch=8,prompt_tokens=1024,output_tokens=64,
                           frequency_mhz=frequency,purpose='interference',seed=9701,repeat=repeat,
                           **point_fields(plan))
                for spec in specs:
                    _,_,summary=await sample(spec,point,args.out/'interference'/f'{frequency}-{repeat}-{spec.instance_id}-isolated.json')
                    isolated[(frequency,repeat,spec.instance_id)]=summary
                barrier=measurement_barrier(len(specs))
                values=await gather_owned(sample(s,point,args.out/'interference'/f'{frequency}-{repeat}-{s.instance_id}-parallel.json',
                    barrier) for s in specs)
                overlap=min(v[2]['end_s'] for v in values)-max(v[2]['start_s'] for v in values)
                for spec,(_,_,summary) in zip(specs,values):
                    old=isolated[(frequency,repeat,spec.instance_id)];parallel[(frequency,repeat,spec.instance_id)]=summary
                    errors={k:abs(summary[k]/old[k]-1) for k in ('latency_ms','power_w')}
                    checks.append(dict(instance_id=spec.instance_id,frequency_mhz=frequency,repeat=repeat,
                        relative_errors=errors,common_window_s=overlap,passed=overlap>=5 and max(errors.values())<=.05))
        concurrent=all(c['passed'] for c in checks)
        qualification=dict(qualified=True,parallel_qualified=concurrent,mode='parallel' if concurrent else 'serial_resident_fallback',
            limit=.05,checks=checks,exclusive_fleet_gpu_uuids=os.environ['PDBLEND_GPU_UUIDS'].split(','),energy_comparable=False)
        atomic_json(args.out/'measurement-qualification.json',qualification)
        training=[];holdout=[];v2_windows=[]
        async def one(spec,points):
            for point in points:
                for repeat in range(point['repeats']):
                    raw_point=dict(point,repeat=repeat);path=args.out/'samples'/f'{digest(point)[:20]}-{repeat}.json'
                    raw=None
                    if v2:raw=await capacity_before_window(spec,raw_point,path,plan,identities[spec.instance_id])
                    if raw is None:raw,rows,_=await sample(spec,raw_point,path)
                    else:rows=[]
                    ref=binding(path);report['raw_bindings'].append(ref)
                    if v2:
                        owner=dict(instance_id=spec.instance_id,raw=ref)
                        v2_windows.append(owner);report['window_owners'].append(owner)
                    else:(training if point['purpose']=='training' else holdout).extend(rows)
            return True
        if concurrent:await gather_owned(one(s,plan['points'][i::len(specs)]) for i,s in enumerate(specs))
        else:
            for i,spec in enumerate(specs):await one(spec,plan['points'][i::len(specs)])
        identity={k:v for k,v in identities[specs[0].instance_id].items() if k!='gpu_uuids'}
        if 'frequency_domain' in plan:identity=with_domain(identity,plan['frequency_domain'])
        qualify=dict(qualification,receipt=binding(args.out/'measurement-qualification.json'))
        if v2:
            from .native_timing_capacity import partition_windows,fit_measured_partition
            partition=partition_windows(plan,(dict(instance_id=r['instance_id'],raw=read_bound(r['raw']))
                for r in v2_windows),identities=identities)
            fitted=fit_measured_partition(partition,identity=dict(system='pdblend',**identity),
                raw_bindings=report['raw_bindings'],measurement_qualification=qualify,limits=plan['holdout_limits'])
            atomic_json(args.out/'window-partition.json',partition)
            report.update(window_partition=binding(args.out/'window-partition.json'),
                measurement_qualification=binding(args.out/'measurement-qualification.json'),
                unsupported_windows=len(partition['unsupported']),measured_windows=len(partition['measured']))
            component=fitted
        else:component=fit_component(training,holdout,identity=dict(system='pdblend',**identity),raw_bindings=report['raw_bindings'],
            measurement_qualification=qualify,limits=plan['holdout_limits'])
        atomic_json(args.out/'timing-component.json',component)
        report.update(status='passed',complete=True,component_qualified=component['component_qualified'],
            timing_component=binding(args.out/'timing-component.json'),final_drains=await drain_endpoints(specs),
            remaining_gates=plan['required_remaining'])
        report['timing_completed_s'] = finish_phase(report)
        if getattr(args, 'timing_first', False):
            from .native_timing_stage import capture_timing_stage
            begin_phase(report, 'timing_snapshot')
            report['resident_timing_stage'] = capture_timing_stage(report,
                input_manifest_ref=binding(args.input_manifest),
                attempt_manifest_ref=binding(args.out.parent/'manifest.json'),
                specs=specs, fleet=fleet, out=args.out/'timing-stage.json')
            finish_phase(report)
            await collect_energy_supplements(args, specs, fleet, meter, sampler, report)
        if getattr(args, 'layout_energy_plan', None):
            begin_phase(report, 'layout_energy')
            await collect_layout_supplement(args, specs, fleet, meter, sampler, report)
            finish_phase(report)
        report['active_phase'] = 'completed'
    except BaseException as exc:
        if report.get('phase_events') and report['phase_events'][-1]['status'] == 'running':
            finish_phase(report, exc)
        # A supplement may preserve a failure receipt before raising. Bind it
        # without turning its existence into a success or qualification claim.
        phase_receipts = {'power_pilot': ('power-pilot', 'resident_power_pilot'),
            'request_cycles': ('request-cycles', 'resident_request_cycles'),
            'layout_energy': ('layout-energy', 'resident_layout_energy')}
        if report['active_phase'] in phase_receipts:
            directory, field = phase_receipts[report['active_phase']]
            receipt = args.out/directory/'completion.json'
            if receipt.is_file():
                report[field] = binding(receipt)
        report.update(error=repr(exc),status='failed',complete=False,failed_phase=report['active_phase'])
    finally:
        report['cleanup_errors']=cleanup_owned(fleet,meter,sampler)
        report['actual_engine_starts']={s.instance_id:[dict(e) for e in fleet[s.instance_id].events
            if e.get('kind')=='start'] for s in specs}
        report['engine_loads']=sum(len(v) for v in report['actual_engine_starts'].values())
        report['hardware_executed']=bool(report['engine_loads'])
        report['physical_cleanup']=await verify_compute_empty(fleet,meter,os.environ['PDBLEND_GPU_UUIDS'].split(','))
        if not report['physical_cleanup']['passed']:
            report['cleanup_errors'].append(dict(component='physical_compute_cleanup',
                error=report['physical_cleanup'].get('error','unverified cleanup')))
        if report['cleanup_errors']:report.update(status='failed',complete=False)
        atomic_json(args.out/'completion.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('model','gpus'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--base-port',type=int,required=True)
    for key in ('point-plan','input-manifest','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--collect-runtime',action='store_true',
                        help='Measure runtime components on this same resident fleet before timing')
    parser.add_argument('--timing-first',action='store_true',
                        help='Bind v2 timing before independent power/request-cycle supplements')
    parser.add_argument('--power-pilot-plan',type=Path,
                        help='Bound pure-power feasibility pilot on the same resident fleet')
    parser.add_argument('--request-cycle-plan',type=Path,
                        help='Bound request-cycle training and independent holdout on the same fleet')
    parser.add_argument('--layout-energy-plan',type=Path,
                        help='Opt-in native32 Poisson layout component after runtime and timing')
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args(argv)
    args.gpus=[int(v) for v in args.gpus.split(',')];plan,inputs=preflight(args)
    args.out.mkdir(parents=True,exist_ok=False)
    if args.preflight_only:
        report=dict(status='cpu_preflight_passed',hardware_executed=False,formal_eligible=False,
                    points=len(plan['points']),windows=sum(p['repeats'] for p in plan['points']),inputs=inputs)
        atomic_json(args.out/'preflight.json',report)
    else:report=asyncio.run(collect(args,plan))
    print(json.dumps(dict(status=report['status'],complete=report.get('complete',False))))
    return 0 if report['status'] in ('passed','cpu_preflight_passed') else 2


if __name__=='__main__':raise SystemExit(main())
