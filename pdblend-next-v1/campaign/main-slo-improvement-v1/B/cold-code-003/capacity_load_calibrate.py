"""Declared development load/physical-capacity calibration and 900s validation.

This deliberately bypasses the 100-second comparison protocol. Layout evidence
is empirical, domain-specific and cannot itself authorize production planning.
Every physical action uses the measured executor's explicit calibration API.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_executor import durable, fixed, require, sha


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_trace(templates, phases, seed, domain_sha256):
    """Independent Poisson phases; retain every exact token prompt in the trace.

    Templates are declared development inputs, never sampled from future formal
    arrivals. Reuse the identical resulting trace for matched layout comparisons.
    """
    require(templates and phases and len(domain_sha256) == 64, 'trace inputs required')
    rng = random.Random(seed)
    requests, prompts, offset = [], [], 0.
    for phase in phases:
        duration, rate = float(phase['duration_s']), float(phase['rate_rps'])
        require(math.isfinite(duration) and duration > 0 and math.isfinite(rate) and rate > 0,
                'positive finite declared phase required')
        at = 0.
        while at < duration:
            item = templates[len(requests) % len(templates)]
            prompt = item['prompt']
            require(isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt),
                    'exact tokenized development prompts required')
            require(type(item['output_len']) is int and item['output_len'] > 0,
                    'positive output limit required')
            requests.append(dict(arrival_s=offset+at, prompt_len=len(prompt),
                                 output_len=item['output_len'], phase=phase['name']))
            prompts.append(list(prompt))
            at += rng.expovariate(rate)
        offset += duration
    return dict(schema='capacity-development-trace-v1', split='development',
        formal_eligible=False, duration_s=offset, seed=seed, demand_domain_sha256=domain_sha256,
        phases=phases, requests=requests, prompts=prompts, n_requests=len(requests),
        generation='independent Poisson segments, fixed cyclic declared prompt shapes')


def validate_trace(trace, domain_sha256, *, duration=None):
    require(trace.get('schema') == 'capacity-development-trace-v1'
        and trace.get('split') == 'development' and trace.get('formal_eligible') is False
        and trace.get('demand_domain_sha256') == domain_sha256, 'development domain trace mismatch')
    reqs, prompts = trace['requests'], trace['prompts']
    require(reqs and len(reqs) == len(prompts) == trace['n_requests'], 'trace cardinality differs')
    span = trace['duration_s']
    require(span > 0 and (duration is None or span == duration), 'declared trace duration differs')
    offsets = [r['arrival_s'] for r in reqs]
    require(offsets[0] == 0 and offsets == sorted(offsets)
        and all(math.isfinite(v) and 0 <= v < span for v in offsets), 'arrival domain invalid')
    require(all(len(p) == r['prompt_len'] and r['output_len'] > 0
        for r,p in zip(reqs,prompts)), 'shape declaration differs')


def validate_spec(spec):
    require(spec['schema'] == 'capacity-load-calibration-spec-v1' and spec['authorized'] is True
        and spec['automatic_retries'] is False, 'explicit bounded development declaration required')
    require(spec['files'] and all(sha(p) == h for p,h in spec['files'].items()), 'calibration source changed')
    original, binding, config = fixed(spec['original_binding']), fixed(spec['capacity_binding']), fixed(spec['config'])
    require(len(original['instances']) == 2 and config['instances'] == original['instances'],
            'same exact initial two instances required')
    require(set(config['node_gpus']) == set(range(8)) and config.get('allow_pd') is False
        and not config.get('transfers') and not config.get('slow_topology')
        and not config.get('dynamic_pools'), 'whole-node independent mixed execution required')
    require(config.get('measurement_window_protocol') is None, 'development calibration is not a 100s comparison')
    require(spec['deadline_s'] == binding['deadline_s'], 'same absolute deadline required')
    if spec['mode'] == 'layout_calibration':
        require(config.get('capacity_integration_v1') is not True and len(spec['cycles']) == 3,
                'exactly three explicit cycles; automatic capacity disabled')
        traces = []
        for cycle in spec['cycles']:
            require(set(cycle) == {'low', 'high2', 'high3', 'under_load', 'restore', 'remove'}, 'complete cycle required')
            for key in ('low','high2','high3','under_load'):
                trace = fixed(cycle[key]); validate_trace(trace, spec['demand_domain_sha256'])
                require(trace['duration_s'] >= 60, 'sustained layout observation must be at least60s')
                traces.append(cycle[key]['sha256'])
            for key in ('restore','remove'):
                action = fixed(cycle[key])
                require(action['operation'] == key and action['deadline_s'] == spec['deadline_s']
                    and action['gpus'] == spec['gpus'], 'physical action declaration differs')
        require(len({c['low']['sha256'] for c in spec['cycles']}) == 3
            and len({c['high2']['sha256'] for c in spec['cycles']}) == 3
            and len({c['high3']['sha256'] for c in spec['cycles']}) == 3,
            'three independently generated low/high repeat traces required')
    else:
        require(spec['mode'] == 'qualification900' and spec['arm'] in ('fixed2','dynamic')
            and (config.get('capacity_integration_v1') is True) == (spec['arm']=='dynamic'),
                '900s arm must match fixed-two or calibrated dynamic runtime')
        trace = fixed(spec['trace']); validate_trace(trace,spec['demand_domain_sha256'],duration=900)
        require([p['name'] for p in trace['phases']] == ['low','high','low']
            and all(p['duration_s'] == 300 for p in trace['phases']), 'exact 300/300/300 low/high/low required')
    return original, binding, config


def phase_metrics(rows, trace, measurement, native_idle):
    n = len(trace['requests'])
    complete = len(rows) == n and all(r['success'] and r['token_ids_verified']
        and r['generated_tokens'] == r['output_len'] for r in rows)
    good = sum(bool(r['slo_ok']) for r in rows)
    duration = measurement['measurement_end_s'] - measurement['measurement_start_s']
    return dict(n_expected=n, n_rows=len(rows), n_good=good, work_complete=bool(complete and native_idle),
        slo_attainment=good/n, offered_rate_rps=n/trace['duration_s'],
        goodput_measurement_rps=good/duration, throughput_measurement_rps=sum(bool(r['success']) for r in rows)/duration,
        energy_j=measurement['energy_j'], mean_whole_node_power_w=measurement['energy_j']/duration,
        empirical_sustainable_sample=bool(complete and native_idle and good/n >= .9),
        bounds_are_empirical_not_hard_guarantees=True)


async def native_idle(controller, deadline):
    from ecopadg.serving.completion_policy import engine_residual
    while time.time() < deadline:
        if not controller.active and not controller.request_tasks:
            await controller.refresh()
            raws = await asyncio.gather(*(controller.backend.json(i,'/runtime') for i in controller.backend.instances))
            if raws and not any(engine_residual(raw,time.time()) for raw in raws):
                return True
        await asyncio.sleep(.05)
    raise TimeoutError('actual request/native workload has not drained')


async def measure_phase(controller, service, trace_ref, out, spec, *, action=None):
    from benchmarks.scripts import bench_vllm as bench
    from benchmarks.scripts.bench_vllm import run_trace, bench_rows, EVALUATION_V3
    from ecopadg.serving.measurement import finite_json
    from capacity_backend import TransitionMeter
    trace = fixed(trace_ref)
    meter = await TransitionMeter(out/'raw').start()
    before = [dict(id=i['id'],gpus=i['gpus']) for i in controller.backend.instances.values()]
    work = physical = None
    partial = {}
    original_send = bench.send_request
    async def tracked_send(*args,**kwargs):
        sink = kwargs.get('_result_sink')
        if sink is not None:
            # The unchanged client continuously updates this dictionary after
            # each received event, including failed/partially streamed requests.
            partial['sink'] = sink
        return await original_send(*args,**kwargs)
    bench.send_request = tracked_send
    result = dict(schema='capacity-load-measurement-v1', phase=out.name, trace=trace_ref,
        demand_domain_sha256=spec['demand_domain_sha256'], resident_before=before, complete=False,
        source=dict(original_binding=spec['original_binding'],capacity_binding=spec['capacity_binding'],
                    config=spec['config'],host_manifest=ref(Path(spec['host_release'])/'manifest.json')))
    try:
        work = asyncio.create_task(run_trace(trace, spec['api_base'], spec['served_model'],
                                             evaluation_protocol=EVALUATION_V3))
        if action:
            await asyncio.sleep(spec.get('cold_start_after_s',10.))
            require(not work.done() and bool(controller.active), 'under-load cold start requires actual active requests')
            result['active_at_cold_start'] = len(controller.active)
            physical = asyncio.create_task(service.executor.calibrate('restore',tuple(spec['gpus']),declaration=action))
        outputs, elapsed = await asyncio.shield(work)
        rows = bench_rows(trace,outputs,controller.config['slo_ttft_s'],controller.config['slo_tpot_s'])
        epoch = outputs[0]['planned_arrival_s']
        await asyncio.sleep(max(0.,epoch+trace['duration_s']-time.time(),trace['duration_s']-elapsed))
        if physical:
            result['physical'] = await asyncio.shield(physical)
        await native_idle(controller,min(spec['deadline_s']-120,epoch+trace['duration_s']+120))
        if spec['mode'] == 'qualification900':
            result['dynamic_drain'] = await controller.finish_measurement(
                min(spec['deadline_s']-60,epoch+trace['duration_s']+120))
            require(result['dynamic_drain'].get('drain_complete') is True,
                    '900s dynamic native cleanup incomplete')
        durable(out/'requests.json',rows)
        result.update(native_idle=True, complete=True, actual_arrival_epoch_s=epoch,
            measured_arrival_duration_s=trace['duration_s'],
            resident_after=[dict(id=i['id'],gpus=i['gpus']) for i in controller.backend.instances.values()])
        result['resident_groups'] = sorted(i['gpus'] for i in before) if result['resident_after']==before else None
    except BaseException as exc:
        result['error'] = repr(exc)
        # A signal or failed action does not abandon already-dispatched requests.
        for pending in (work,physical):
            if pending is not None and not pending.done():
                try:
                    await asyncio.wait_for(asyncio.shield(pending),max(.001,min(120.,spec['deadline_s']-180-time.time())))
                except BaseException as cleanup:
                    result.setdefault('cleanup_errors',[]).append(repr(cleanup))
        raise
    finally:
        if work is not None and not work.done():
            work.cancel()
            await asyncio.gather(work,return_exceptions=True)
        bench.send_request = original_send
        durable(out/'partial-client-results.json',finite_json(partial.get('sink',{})))
        measured = await meter.finish()
        result['raw_measurement'] = measured['receipt']
        if result['complete']:
            result.update(phase_metrics(rows,trace,measured,result['native_idle']))
        result['artifacts'] = {str(p):sha(p) for p in out.iterdir() if p.is_file()}
        durable(out/'result.json',result)
    require(measured['measurement_valid'],'whole-node phase power invalid')
    return result


async def execute(spec,out):
    import aiohttp
    from aiohttp import web
    from urllib.parse import urlparse
    from ecopadg.serving.runtime import Controller
    from capacity_runtime import CapacityService
    from capacity_backend import TransitionMeter
    original,binding,config = validate_spec(spec)
    common = load(spec['common_executor']['path'],'capacity_load_original_common')
    require(sha(spec['common_executor']['path']) == spec['common_executor']['sha256'],'identity verifier changed')
    config = copy.deepcopy(config)
    config.update(journal=str(out/'control.jsonl'),capacity_inventory_path=str(out/'inventory.json'))
    durable(out/'runtime-config.json',config)
    controller = Controller(config)
    runner = web.AppRunner(controller.application())
    service = None
    state = dict(schema='capacity-load-status-v1',pid=os.getpid(),started_s=time.time(),
                 phase='initializing',complete=False,production_ready=False,completed=[],automatic_retries=False)
    stopping = False
    def stop():
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM,signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig,stop)
    def update(**values):
        state.update(values,updated_s=time.time());durable(out/'status.json',state)
    def boundary(name,reserve):
        require(not stopping and not Path(spec['stop_path']).exists(),'STOP preserves remaining declaration')
        require(time.time()+reserve+120 < spec['deadline_s'],'remaining work and cleanup exceed absolute deadline')
        update(phase=name)
    update()
    operation_meter = await TransitionMeter(out/'full-operation-power').start()
    dispatched=[]
    ownership=(out/'engine-dispatch.jsonl').open('x',buffering=1)
    original_request=aiohttp.ClientSession._request
    async def owned_request(session,method,url,**kwargs):
        parsed=urlparse(str(url))
        instances=list(getattr(getattr(controller,'backend',None),'instances',{}).values())
        if service is not None:
            instances=list(service.inventory.value['known_instances'].values())
        target=next((i for i in instances if i.get('port')==parsed.port),None)
        if method.upper()=='POST' and parsed.path=='/v1/completions' and target:
            rid=(kwargs.get('headers') or {}).get('X-Request-Id')
            require(isinstance(rid,str) and 0<len(rid)<=256,'owned native request ID required')
            record=dict(instance_id=target['id'],port=parsed.port,request_id=rid,at_s=time.time())
            ownership.write(json.dumps(record)+'\n');ownership.flush();os.fsync(ownership.fileno())
            dispatched.append(record)
        return await original_request(session,method,url,**kwargs)
    aiohttp.ClientSession._request=owned_request
    try:
        common.validate_binding(original)
        await runner.setup()
        await web.TCPSite(runner,'127.0.0.1',config['port']).start()
        durable(out/'identity.before.json',await common.identity(controller.session,original))
        service = (getattr(controller,'capacity_service',None) if spec['mode']=='qualification900'
                   else CapacityService(controller,binding,require_calibration=False))
        require(service is not None or spec.get('arm')=='fixed2','dynamic lifecycle integration missing')
        if spec['mode'] == 'layout_calibration':
            for index,cycle in enumerate(spec['cycles'],1):
                for layout,key in ((2,'low'),(2,'high2'),(None,'under_load'),(3,'high3'),(3,'low')):
                    name=f'cycle-{index}-{key}-layout{layout or "2to3"}'
                    trace=fixed(cycle[key]);boundary(name,trace['duration_s']+360)
                    if layout:
                        require(len(controller.backend.instances)==layout,'measured layout differs')
                    result=await measure_phase(controller,service,cycle[key],out/name,spec,
                        action=cycle['restore'] if layout is None else None)
                    state['completed'].append(ref(out/name/'result.json'));update()
                    require(result['work_complete'],'incomplete measured work prevents successor calibration')
                boundary(f'cycle-{index}-remove',360)
                extra=[i for i in controller.backend.instances.values() if i['id'] not in service.inventory.value['initial_ids']]
                require(len(extra)==1,'exactly one calibration-created replica required')
                result=await service.executor.calibrate('remove',tuple(spec['gpus']),remove_id=extra[0]['id'],declaration=cycle['remove'])
                durable(out/f'cycle-{index}-remove.json',result)
        else:
            boundary('qualification900',1020)
            await measure_phase(controller,service,spec['trace'],out/'qualification900',spec)
        await native_idle(controller,min(time.time()+120,spec['deadline_s']-60))
        if service is not None:
            await service.finish_to_initial()
        durable(out/'identity.after.json',await common.identity(controller.session,original))
        update(phase='measured_complete',complete=True,empirical_only=True)
    except BaseException as exc:
        update(phase='needs_attention',error=repr(exc))
        raise
    finally:
        errors=[]
        if service is not None:
            try:
                await service.quiesce()
                await native_idle(controller,min(time.time()+120,spec['deadline_s']-30))
                await service.finish_to_initial()
            except BaseException as exc:
                errors.append('capacity cleanup: '+repr(exc))
        try:
            await runner.cleanup()
        except BaseException as exc:
            errors.append('controller cleanup: '+repr(exc))
        # Restore the retained engines' exact ordinary budget/role after their
        # own controller and every owned added process have stopped serving.
        if not errors:
            async with aiohttp.ClientSession(trust_env=False) as session:
                try:
                    await asyncio.gather(*(common.resume(session,i,i.get('restore_budget_tokens')) for i in original['instances']))
                    await common.identity(session,original)
                except BaseException as exc:
                    errors.append('original restoration: '+repr(exc))
        aiohttp.ClientSession._request=original_request
        ownership.close()
        outer=await operation_meter.finish()
        update(cleanup_complete=not errors,cleanup_errors=errors,finished_s=time.time(),
            full_operation_measurement=outer['receipt'],dispatched_native_requests=len(dispatched),
            energy_accounting='full-operation energy includes setup/restoration and overlaps all phase/transition energy; do not add them')
        require(not errors,'calibration cleanup requires attention')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    spec=fixed(dict(path=str(args.spec),sha256=args.spec_sha256))
    original,_,_=validate_spec(spec)
    host=Path(spec['host_release'])
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    os.environ['PYTHONPATH']=':'.join(sys.path[:4])
    if not args.run:
        print(json.dumps(dict(cpu_only=True,mode=spec['mode'],files_verified=len(spec['files']),hardware_actions=False)))
        return
    from ecopadg.serving.campaign import node_lease
    require('PDBLEND_NODE_LOCK_FD' not in os.environ,'development calibration needs a fresh exclusive node lease')
    require(not args.out.exists(),'new independent calibration output required')
    with node_lease():
        args.out.mkdir(parents=True)
        durable(args.out/'spec-reference.json',ref(args.spec))
        asyncio.run(execute(spec,args.out))


if __name__=='__main__':
    main()
