"""Run one GPU measurement on explicitly prepared, qualified resident engines.

The driver never stops or deletes containers. Use the shared node lease, verify
native drain, and restore the roles observed before the experiment.
"""
import argparse
import asyncio
import contextlib
import csv
import json
import socket
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from benchmarks.scripts.bench_vllm import bench_rows, send_request
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from ecopadg.serving.campaign import node_lease
from ecopadg.serving.measurement import save_raw
from ecopadg.serving.runtime import Controller

from .artifacts import object_hash, read_json, sha256, source_files, write_json
from .gpu_observe import instrument
from .native_logs import begin_capture, finish_capture
from .preflight import engine_identity, inspect_engines, quiescent, verify_freeze


class DeadlineSession:
    def __init__(self, session, timeout_s):
        self.session, self.timeout_s = session, timeout_s

    def post(self, *args, **kwargs):
        kwargs['timeout'] = aiohttp.ClientTimeout(total=self.timeout_s)
        return self.session.post(*args, **kwargs)


async def replay(trace, api_base, model, *, start_s=None, start_mono=None, progress=None):
    """Open-loop arrivals, with a separate complete deadline for every request."""
    start_s = time.time() if start_s is None else start_s
    start_mono = time.perf_counter() if start_mono is None else start_mono
    progress = progress if progress is not None else {'terminal': 0}
    async with aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(limit=0),
                                    timeout=aiohttp.ClientTimeout(total=None)) as session:
        async def worker(index, request):
            scheduled = start_mono + request['arrival_s']
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            remaining = request['timeout_s'] - max(0, time.perf_counter() - scheduled)
            # An event-loop stall does not grant an extra request lifetime.
            client = DeadlineSession(session, max(remaining, .000001))
            result = await send_request(client, api_base, model, trace['prompts'][index],
                request['output_len'], arrival_s=start_s+request['arrival_s'],
                arrival_monotonic=scheduled, request_id=str(index))
            result['declared_timeout_s'] = request['timeout_s']
            result['scheduled_arrival_s'] = start_s + request['arrival_s']
            progress['terminal'] += 1
            return result
        tasks = [asyncio.create_task(worker(i, request)) for i, request in enumerate(trace['requests'])]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def native_states(config):
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=5)) as session:
        states = {}
        for instance in config['instances']:
            async with session.get(instance['url']+'/runtime') as response:
                response.raise_for_status()
                states[instance['id']] = await response.json()
        return states


async def wait_drained(config, controller=None, *, timeout_s=30):
    deadline = time.monotonic() + timeout_s
    while True:
        states = await native_states(config)
        if all(quiescent(s) for s in states.values()) and (controller is None or not controller.active):
            return states
        if time.monotonic() >= deadline:
            raise RuntimeError('native requests or KV did not drain within the declared deadline')
        await asyncio.sleep(.1)


async def restore_roles(config, original):
    await wait_drained(config)
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=30)) as session:
        for instance in config['instances']:
            async with session.get(instance['url']+'/runtime') as response:
                response.raise_for_status()
                current = await response.json()
            desired = {key: original[instance['id']].get(key, True if key.startswith('admit_') else 'continuous')
                       for key in ('role', 'mode', 'admit_prefill', 'admit_decode')}
            if any(current.get(key) != value for key, value in desired.items()):
                async with session.post(instance['url']+'/control',
                        json=dict(desired, generation=current['generation']+1)) as response:
                    response.raise_for_status()
                    await response.read()
    await wait_drained(config)


def validate_inputs(config, trace, manifest, *, diagnostic=False):
    from .protocol import POOL_COUNTS, validate_row
    validate_row(manifest)
    if not trace.get('requests') or len(trace['requests']) != len(trace.get('prompts', [])):
        raise ValueError('a nonempty complete immutable trace is required')
    allocated = config['allocated_gpu_ids']
    if allocated != manifest['allocated_gpu_ids'] or len(allocated) != manifest['n_gpus']:
        raise ValueError('manifest and runtime allocation differ')
    if len(set(allocated)) != len(allocated) or not set(allocated) <= set(range(8)):
        raise ValueError('invalid GPU subset')
    if any(i['tp'] != 1 or len(i['gpus']) != 1 for i in config['instances']):
        raise ValueError('all instances must use physical TP1')
    if sorted(g for i in config['instances'] for g in i['gpus']) != sorted(allocated):
        raise ValueError('physical instance layout differs from allocated GPU budget')
    if config.get('slow_topology') or config.get('dynamic_pools') or config.get('allow_unprofiled_fallback'):
        raise ValueError('fixed-layout scaling forbids reconfiguration and unprofiled fallback')
    if config.get('manage_clocks', True) is not True:
        raise ValueError('GPU comparisons require measured hardware clock control')
    system = manifest['system']
    strategies = {'pdblend': 'pdblend-joint', 'mixed': 'mixed', 'fixed_pd': 'distserve'}
    if config['strategy'] != strategies[system] or config.get('comparison_system') != system:
        raise ValueError('actual strategy differs from declared system')
    roles = [i['role'] for i in config['instances']]
    if system == 'pdblend':
        m,p,d = POOL_COUNTS[manifest['n_gpus']]
        if roles != ['mixed']*m+['prefill']*p+['decode']*d or config.get('allow_pd') is not True or config.get('dvfs') is not True:
            raise ValueError('selective PD requires the frozen three-pool layout and DVFS')
    elif system == 'mixed' and (set(roles) != {'mixed'} or config.get('dvfs')):
        raise ValueError('Mixed baseline must use only fixed-frequency mixed engines')
    elif system == 'fixed_pd':
        if set(roles) != {'prefill','decode'} or config.get('dvfs'):
            raise ValueError('fixed PD baseline must use separate fixed-frequency P/D pools')
    if diagnostic and manifest['stage'] != 'diagnostic':
        raise ValueError('diagnostic execution must be explicitly labelled diagnostic')
    if not diagnostic:
        for field in ('dataset','n_gpus','seed','rate_rps','stage'):
            if trace.get(field) != manifest[field]:
                raise ValueError('trace identity differs: '+field)
    if not diagnostic and float(manifest['arrival_window_s']) != 600:
        raise ValueError('formal and pilot GPU observations require the frozen 600-second arrival window')
    previous = -1
    context_limit = config.get('max_model_len', 8192)
    if type(context_limit) is not int or context_limit <= 0:
        raise ValueError('explicit model context limit must be positive')
    for request in trace['requests']:
        if (type(request.get('prompt_len')) is not int or request['prompt_len'] <= 0
                or type(request.get('output_len')) is not int
                or request['prompt_len']+request['output_len'] > context_limit):
            raise ValueError('request exceeds the fixed per-instance model context limit')
        offset = float(request['arrival_s'])
        if not previous <= offset < manifest['arrival_window_s']:
            raise ValueError('arrival order/window mismatch')
        previous = offset
        expected = max(120., config['slo_ttft_s'] + (request['output_len']-1)*config['slo_tpot_s']+30.)
        if request['timeout_s'] != expected or request['output_len'] < 2:
            raise ValueError('per-request timeout/work differs from the protocol')
    for key in ('slo_ttft_s', 'slo_tpot_s'):
        if config[key] != manifest[key]:
            raise ValueError('effective SLO differs from manifest')


async def run_cell(config, trace, manifest, out, *, freeze=None, diagnostic=False):
    from .audit import audit_run
    validate_inputs(config, trace, manifest, diagnostic=diagnostic)
    if not diagnostic:
        errors = verify_freeze(freeze or {})
        if errors:
            raise ValueError('formal input gate closed: ' + '; '.join(errors[:5]))
        records = await inspect_engines(config)
        for record in records:
            if record['errors'] or engine_identity(record) != freeze['engine_identities'].get(record['instance_id']):
                raise ValueError('live qualified engine identity changed')
        from .protocol import build_config
        expected = build_config(read_json(freeze['config_path']),system=manifest['system'],dataset=manifest['dataset'],
            allocated_gpu_ids=manifest['allocated_gpu_ids'],fixed_pd_p_count=config.get('fixed_pd_p_count'),
            max_frequency_mhz=2520)
        for field in set(expected) | set(config):
            if field not in ('warmup_trace','journal') and config.get(field) != expected.get(field):
                raise ValueError('runtime config differs from frozen input: '+field)
        from .workload import build_trace, load_pool
        pool = load_pool(freeze['pools'][manifest['dataset']])
        kwargs = {k:manifest[k] for k in ('dataset','n_gpus','seed','rate_rps')}
        if trace != build_trace(pool,stage=manifest['stage'],**kwargs):
            raise ValueError('measurement trace is not the frozen generator output')
        if config.get('warmup_trace') != build_trace(pool,stage='warmup',**kwargs):
            raise ValueError('warmup trace is not independently generated from the frozen pool')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(config, journal=str(out/'control.jsonl'),
        engine_request_timeout_s=max(r['timeout_s'] for r in trace['requests'])+30)
    manifest = dict(manifest, scope='gpu_serving', host_id=socket.gethostname(),
        native_kv_capture_required=True,
        formal_eligible=not diagnostic and manifest['stage'] in ('capacity', 'weak'),
        source_hashes={str(p): sha256(p) for p in source_files()},
        profile_sha256=sha256(config['profiles']),
        source_config_sha256=(freeze or {}).get('source_config_sha256', object_hash(config)))
    write_json(out/'trace.json', trace)
    manifest['trace_sha256'] = sha256(out/'trace.json')
    write_json(out/'manifest.json', manifest)
    write_json(out/'runtime_config.json', config)
    if freeze:
        write_json(out/'freeze.json', freeze)
    original = await wait_drained(config)
    write_json(out/'native.before.json', original)
    kv_cursors = begin_capture(config)
    controller = Controller(config)
    timing_records = instrument(controller)
    runner = web.AppRunner(controller.application(), access_log=None, handler_cancellation=True)
    sampler = None
    sampling = False
    backlog_task = None
    measurement = dict(declared_formal=manifest['formal_eligible'], initial_quiescent=True,
        terminal_quiescent=False, hardware_qualification_verified=not diagnostic,
        source_freeze_verified=not diagnostic, errors=[])
    rows = []
    try:
        await runner.setup()
        port = config.get('port', 18080)
        await web.TCPSite(runner, '127.0.0.1', port).start()
        api = f'http://127.0.0.1:{port}'
        warm_duration = 2. if diagnostic else 120.
        warm_start = time.monotonic()
        warm = config.get('warmup_trace')
        if warm is None and not diagnostic:
            raise ValueError('formal GPU measurements need an independently generated warmup trace')
        if warm is None:
            warm = dict(trace, requests=[dict(r, arrival_s=i*.2) for i, r in enumerate(trace['requests'][:min(8,len(trace['requests']))])],
                        prompts=trace['prompts'][:min(8,len(trace['requests']))])
        warm_outputs = await replay(warm, api, config.get('model_name','Qwen2.5-14B-Instruct'))
        await asyncio.sleep(max(0, warm_duration-(time.monotonic()-warm_start)))
        await wait_drained(config, controller, timeout_s=max((r['timeout_s'] for r in warm['requests']), default=120.))
        write_json(out/'warmup.json', dict(duration_s=time.monotonic()-warm_start, outputs=warm_outputs,
                                        excluded_from_primary=True))
        await controller.journal.flush()
        timing_records.clear()
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        sampler = PowerSampler(range(8), interval=.02, backend=hardware)
        sampler.start()
        sampling = True
        await asyncio.sleep(.15)
        if sampler.error or len(sampler.samples) < 2:
            raise RuntimeError('instant eight-GPU power preflight failed')
        start, mono = time.time(), time.perf_counter()
        measurement.update(start_s=start, arrival_end_s=start+manifest['arrival_window_s'])
        progress = {'terminal': 0}
        backlog = []
        finished = asyncio.Event()

        async def observe_backlog():
            while not finished.is_set():
                now = time.time()
                offered = sum(start+r['arrival_s'] <= now for r in trace['requests'])
                backlog.append(dict(t_s=now, pending=max(0,offered-progress['terminal']),
                    admission_pending=controller.pending.qsize(), active=len(controller.active)))
                try:
                    await asyncio.wait_for(finished.wait(), 1.)
                except asyncio.TimeoutError:
                    pass

        backlog_task = asyncio.create_task(observe_backlog())
        outputs = await replay(trace, api, config.get('model_name','Qwen2.5-14B-Instruct'),
                               start_s=start, start_mono=mono, progress=progress)
        write_json(out/'outputs.json', outputs)
        rows = bench_rows(trace, outputs, config['slo_ttft_s'], config['slo_tpot_s'])
        await asyncio.sleep(max(0, measurement['arrival_end_s']-time.time()))
        terminal = await wait_drained(config, controller,
                                     timeout_s=max(r['timeout_s'] for r in trace['requests']))
        await controller.quiesce_controls()
        measurement.update(end_s=time.time(), terminal_quiescent=True)
        finished.set()
        await backlog_task
        write_json(out/'native.after.json', terminal)
        with (out/'backlog.jsonl').open('x') as handle:
            for row in backlog:
                handle.write(json.dumps(row)+'\n')
        await asyncio.sleep(.1)
        if controller.failure:
            raise RuntimeError(controller.failure)
        if freeze and verify_freeze(freeze):
            measurement['source_freeze_verified'] = False
            raise RuntimeError('source freeze changed during measurement')
    except BaseException as exc:
        measurement['errors'].append(type(exc).__name__+': '+str(exc))
        measurement.setdefault('end_s', time.time())
        raise
    finally:
        if backlog_task and not backlog_task.done():
            backlog_task.cancel()
            await asyncio.gather(backlog_task, return_exceptions=True)
        if sampling:
            await asyncio.to_thread(sampler.stop)
            measurement['sampling_error'] = sampler.error
            if rows:
                await asyncio.to_thread(save_raw, out, rows, sampler.samples, sampler.utilization_samples,
                    power_source=sampler.power_source, power_metadata=sampler.power_metadata)
            else:
                write_json(out/'incomplete-power.json', dict(samples=sampler.samples,
                    metadata=sampler.power_metadata, power_source=sampler.power_source))
        cleanup_errors = []
        try:
            await runner.cleanup()
        except BaseException as exc:
            cleanup_errors.append('controller cleanup: '+repr(exc))
        try:
            await restore_roles(config, original)
        except BaseException as exc:
            cleanup_errors.append('native role restoration: '+repr(exc))
        measurement['cleanup_errors'] = cleanup_errors
        try:
            finish_capture(kv_cursors, out)
        except Exception as exc:
            measurement['errors'].append('native KV log capture: '+str(exc))
        if cleanup_errors:
            measurement['terminal_quiescent'] = False
        write_json(out/'measurement.json', measurement)
        with (out/'planning.jsonl').open('x') as handle:
            for record in timing_records:
                handle.write(json.dumps(record)+'\n')
    summary = audit_run(out)
    write_json(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    with node_lease():
        result = asyncio.run(run_cell(read_json(args.config), read_json(args.trace), read_json(args.manifest),
            args.out, freeze=read_json(args.freeze) if args.freeze else None, diagnostic=args.diagnostic))
    print(json.dumps({k:result.get(k) for k in ('manifest_path','measurement_valid','formal_eligible',
        'offered_requests','good_requests','energy_allocated_j','energy_node8_j','audit_errors')},ensure_ascii=False))


if __name__ == '__main__':
    main()
