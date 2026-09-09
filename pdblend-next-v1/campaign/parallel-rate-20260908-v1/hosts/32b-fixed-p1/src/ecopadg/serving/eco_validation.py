"""Forced, real EcoServe rolling windows and macro split/merge correctness.

Uses four explicitly configured, already running instances. This is mechanism
evidence, never evidence of autonomous adaptation, capacity, or energy savings.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import re
import signal
import time
import traceback
from urllib.parse import urlparse
import uuid

import aiohttp
from aiohttp import web
from benchmarks.scripts.bench_vllm import send_request

from .backend import HttpEngineBackend
from .campaign import node_lease
from .controller import Controller
from .evidence import freeze_files, sha256, validate_freeze
from .profiles import ProfileStore
from .profiling import HardwareProfiler


CONTROL_FIELDS = ('role', 'mode', 'admit_prefill', 'admit_decode')
PURPOSE = 'forced real EcoServe rolling windows and macro split/merge; not autonomous or energy evidence'


def validation_config(config, out, short_input, long_input, output_tokens):
    """Select four supplied instances; do not invent endpoints or a profile."""
    if len(config.get('instances', [])) < 4:
        raise ValueError('at least four explicitly configured existing instances required')
    if not 0 < short_input < long_input or output_tokens < 32 or long_input + output_tokens > 8192:
        raise ValueError('interleaved short/long requests must fit the engine context')
    instances = [dict(i) for i in config['instances'][:4]]
    ids, gpus = set(), set()
    for i in instances:
        identifier = i['id']
        endpoint = urlparse(i['url'])
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in ids
                or endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost', '::1')
                or not endpoint.port or endpoint.path not in ('', '/')
                or i['tp'] not in (1, 2, 4, 8) or len(i['gpus']) != i['tp']
                or len(set(i['gpus'])) != i['tp'] or set(i['gpus']) & gpus
                or not set(i['gpus']) <= set(range(8))):
            raise ValueError('distinct, local instances on disjoint GPUs required')
        ids.add(identifier); gpus.update(i['gpus'])
        i.update(port=endpoint.port, role='mixed')
    profiles = ProfileStore.load(config['profiles'])
    for i in instances:
        for n in (short_input, long_input):
            if profiles.lookup('mixed', i['tp'], 2520, n, n + 1, 1) is None:
                raise ValueError('provided measured profile does not cover the validation prefill')
    result = dict(config, strategy='ecoserve', instances=instances,
        journal=str(out / 'control.jsonl'), eco_initial_instances=3,
        eco_macro_lower=2, eco_macro_upper=3, eco_scale_period_s=3600,
        manage_clocks=True, park_idle=False, prepare_peers=False,
        node_gpus=list(range(8)), output_prior=output_tokens)
    # This test changes resident macro membership only, never physical layout.
    result.pop('topology', None)
    return result


def verify_execution(events, routes, kv_observations):
    """Engine steps and physical KV observations must agree with every route."""
    active = [e for e in events if e.get('tokens', 0) > 0]
    if not active or any(e['prefill'] and e['decode'] for e in active):
        raise RuntimeError('engine timeline does not establish phase exclusion')
    phases = {rid: set() for rid in routes}
    for event in active:
        if event.get('mode') != 'temporal' or event.get('role') != 'mixed':
            raise RuntimeError('work executed outside the real temporal engine mode')
        for rid in event['request_ids']:
            if rid not in routes or routes[rid] != event['instance']:
                raise RuntimeError('request executed on an unassigned instance or moved its KV')
            phases[rid].add('prefill' if event['prefill'] else 'decode')
    if not phases or any(p != {'prefill', 'decode'} for p in phases.values()):
        raise RuntimeError('every routed request needs actual prefill and decode steps')
    seen = set()
    for observation in kv_observations:
        for instance, requests in observation['owners'].items():
            for rid in requests:
                if routes.get(rid) != instance:
                    raise RuntimeError('request KV appeared on an unassigned instance')
                seen.add(rid)
    if set(routes) - seen:
        raise RuntimeError('missing physical KV observation for a routed request')
    return dict(actual_steps=len(active), overlap_steps=0, requests_with_stationary_kv=len(seen))


def read_events(runtime_dir, offsets):
    events = []
    for identifier, offset in offsets.items():
        with (runtime_dir / f'{identifier}.control.events.jsonl').open() as handle:
            handle.seek(offset)
            events.extend(dict(json.loads(line), instance=identifier) for line in handle if line.strip())
    return events


async def drain(backend, identifier, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        state = await backend.json(identifier, '/runtime')
        if not any(state.get(k) for k in ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations')):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError('engine did not drain: ' + identifier)
        await asyncio.sleep(.02)


async def restore(backend, original):
    errors = []
    for identifier, state in original.items():
        try:
            current = await drain(backend, identifier)
            wanted = {k: state[k] for k in CONTROL_FIELDS}
            if any(current.get(k) != v for k, v in wanted.items()):
                await backend.json(identifier, '/control', dict(wanted, generation=current['generation'] + 1))
            restored = await backend.json(identifier, '/runtime')
            if (any(restored.get(k) != v for k, v in wanted.items())
                    or restored.get('acknowledged_generation') != restored['generation']):
                raise RuntimeError('restored control state was not acknowledged')
        except Exception:
            errors.append(dict(instance=identifier, traceback=traceback.format_exc()))
    return errors


async def validate(args):
    args.out.mkdir(parents=True, exist_ok=False)
    raw = dict(complete=False, passed=False, purpose=PURPOSE, changes=[], kv_observations=[],
               reference={}, outputs=[], errors=[], cleanup_errors=[])
    runner = controller = backend = None
    original, requests, reference_ids, offsets = {}, {}, [], {}
    failure = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout), trust_env=False) as session:
        try:
            config = validation_config(json.loads(args.config.read_text()), args.out,
                                       args.short_input, args.long_input, args.output_tokens)
            raw['config'] = config
            raw['input_config_sha256'] = sha256(args.config)
            raw['profile_sha256'] = sha256(config['profiles'])
            raw['source_files'] = await asyncio.to_thread(freeze_files,
                [*Path(__file__).parent.glob('*.py'), Path(send_request.__code__.co_filename)])
            instances = config['instances']; ids = [i['id'] for i in instances]
            backend = HttpEngineBackend(instances, session)
            profiler = HardwareProfiler(session, {i['id']: i for i in instances}, args.runtime_dir)
            raw['engine_provenance'] = await profiler.provenance()
            if any(len({e[k] for e in raw['engine_provenance']}) != 1
                   for k in ('model', 'engine_version', 'image_id')):
                raise RuntimeError('validation instances use different models, versions, or images')
            engine_sources = {next((value for path, value in e['source_files_at_import'].items()
                if path.endswith('/serving/engine.py')), None) for e in raw['engine_provenance']}
            if engine_sources != {sha256(Path(__file__).with_name('engine.py'))}:
                raise RuntimeError('resident engines do not use the current engine implementation')
            # Capture every original state before the first role mutation.
            for identifier in ids:
                original[identifier] = await drain(backend, identifier)
            for i in instances:
                await profiler.control(i, role='mixed', mode='continuous', admit_prefill=True, admit_decode=True)
            lengths = (args.short_input, args.long_input, args.short_input, args.long_input)
            prompts = [([9707, 1879, 13] * (n // 3 + 1))[:n] for n in lengths]
            async def reference(i):
                values = {}
                for n, prompt in zip(lengths[:2], prompts[:2]):
                    rid = uuid.uuid4().hex; reference_ids.append(rid)
                    response = await profiler.call(i, '/v1/completions', dict(prompt=prompt,
                        max_tokens=args.output_tokens, temperature=0, top_p=1., seed=0,
                        ignore_eos=True, stream=False), rid)
                    if len(response['token_ids']) != args.output_tokens or response['usage']['completion_tokens'] != args.output_tokens:
                        raise RuntimeError('ordinary reference did not complete its output workload')
                    values[str(n)] = response['token_ids']
                    reference_ids.remove(rid)
                    raw['reference'].setdefault(i['id'], {})[str(n)] = response['token_ids']
                return i['id'], values
            reference_tasks = [asyncio.create_task(reference(i)) for i in instances]
            try:
                raw['reference'] = dict(await asyncio.gather(*reference_tasks))
            finally:
                for task in reference_tasks:
                    if not task.done(): task.cancel()
                await asyncio.gather(*reference_tasks, return_exceptions=True)
            controller = Controller(config)
            runner = web.AppRunner(controller.application(), shutdown_timeout=2)
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', args.port).start()
            api = f'http://127.0.0.1:{runner.addresses[0][1]}'
            raw['controller_url'] = api
            offsets = {i: (args.runtime_dir / f'{i}.control.events.jsonl').stat().st_size for i in ids}

            async def observe():
                states = await asyncio.gather(*(backend.json(i, '/runtime') for i in ids))
                item = dict(at_s=time.time(), owners={i: sorted(s['kv_allocations']) for i, s in zip(ids, states)})
                raw['kv_observations'].append(item)
                return states, item

            def launch(index):
                requests[index] = asyncio.create_task(send_request(session, api,
                    config.get('served_model', config.get('model_name', 'Qwen2.5-14B-Instruct')),
                    prompts[index], args.output_tokens, request_id=str(index)))

            async def live(index):
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    if controller.failure: raise RuntimeError(controller.failure)
                    for rid, active in tuple(controller.active.items()):
                        if active['client_request_id'] == str(index) and active['budget'].emitted:
                            _, physical = await observe()
                            target = active['route'].decode_id
                            if rid in physical['owners'][target]: return rid, target
                    if requests[index].done():
                        raise RuntimeError('request ended before the live-KV checkpoint: ' + str(requests[index].result()))
                    await asyncio.sleep(.01)
                raise TimeoutError('request did not reach live decode')

            async def membership(operation):
                async with controller.action_lock:
                    await controller.refresh()
                    states, before_kv = await observe()
                    if not any(before_kv['owners'].values()):
                        raise RuntimeError('macro transition must overlap real live KV')
                    before = list(controller.eco_scheduler.groups)
                    if operation == 'split':
                        controller.eco_scheduler.add_instance(ids[3]); removed = None
                        expected = [2, 2]
                    else:
                        removed = controller.eco_scheduler.remove_idle_instance(controller.state.snapshot)
                        expected = [3]
                        if removed is None or any(states[ids.index(removed)].get(k) for k in
                                ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations')):
                            raise RuntimeError('macro removal did not select a drained member')
                    if sorted(map(len, controller.eco_scheduler.groups)) != expected:
                        raise RuntimeError('unexpected macro split/merge membership')
                    plan = controller.eco_scheduler.membership_plan(controller.state.snapshot, time.time())
                    await controller.backend.execute(plan)
                    if not await controller.backend.confirm(plan): raise RuntimeError('macro window plan unconfirmed')
                    await controller.state.apply_windows(plan.windows)
                    after, after_kv = await observe()
                    raw['changes'].append(dict(operation=operation, removed=removed, before=before,
                        after=list(controller.eco_scheduler.groups), plan=asdict(plan),
                        before_kv=before_kv, after_kv=after_kv,
                        removed_state=states[ids.index(removed)] if removed else None,
                        acknowledgements=[{k: s.get(k) for k in (*CONTROL_FIELDS, 'generation', 'acknowledged_generation')}
                                          for s in after], at_s=time.time()))

            launch(0)
            _, first = await live(0)
            # Force the cyclic admission branch while its current member keeps
            # decoding. No queued token or prefill is delayed at this proxy.
            await controller.freeze_topology([first], True)
            try:
                launch(1); _, second = await live(1)
                group = controller.eco_scheduler.groups[0]
                if second != group[(group.index(first) + 1) % len(group)]:
                    raise RuntimeError('rolling activation did not select the next macro member')
                raw['rolling'] = dict(first=first, second=second, group=list(group),
                    trigger='temporary new-admission pause on current member; its decode continues')
            finally:
                await controller.freeze_topology([first], False)
            await membership('split')
            launch(2); await live(2)
            await membership('merge')
            launch(3); await live(3)
            raw['outputs'] = list(await asyncio.gather(*(requests[i] for i in range(4))))
            await controller.quiesce_controls()
            await controller.journal.flush()
            journal = [json.loads(line) for line in (args.out / 'control.jsonl').read_text().splitlines() if line]
            admissions = [e for e in journal if e['kind'] == 'admission']
            if len(admissions) != 4: raise RuntimeError('missing or duplicate admission evidence')
            raw['admissions'] = admissions
            routes = {e['request_id']: e['plan']['routes'][0]['decode_id'] for e in admissions}
            clients = {e['client_request_id']: e for e in admissions}
            rolling_plan = clients['1']['plan']['windows']
            if not ({(first, False), (second, True)} <= {(a['instance_id'], a['admit_prefill']) for a in rolling_plan}):
                raise RuntimeError('rolling admission did not execute both window transitions')
            matches = [o['success'] and o['generated_tokens'] == args.output_tokens and
                o['token_ids'] == raw['reference'][routes[clients[str(i)]['request_id']]][str(lengths[i])]
                for i, o in enumerate(raw['outputs'])]
            if not all(matches): raise RuntimeError('EcoServe output tokens differ from ordinary execution')
            raw['output_matches'] = matches
            raw['events'] = await asyncio.to_thread(read_events, args.runtime_dir, offsets)
            raw['checks'] = verify_execution(raw['events'], routes, raw['kv_observations'])
            raw['checks'].update(engine_temporal_exclusion=True, rolling_activation=True,
                                 macro_split_merge=True, output_correctness=True)
            raw['engine_provenance_after'] = await profiler.provenance()
            if raw['engine_provenance_after'] != raw['engine_provenance']:
                raise RuntimeError('macro changes replaced or modified a resident engine')
            if validate_freeze(raw['source_files']) or sha256(config['profiles']) != raw['profile_sha256']:
                raise RuntimeError('implementation or profile changed during validation')
            raw.update(complete=True, passed=True)
        except BaseException as exc:
            failure = exc
            raw['errors'].append(traceback.format_exc())
        finally:
            owned = list(reference_ids)
            if controller is not None: owned += list(controller.active)
            journal_path = args.out / 'control.jsonl'
            if journal_path.exists():
                try:
                    journal_text = await asyncio.to_thread(journal_path.read_text)
                    owned += [e['request_id'] for line in journal_text.splitlines() if line
                              for e in [json.loads(line)] if e.get('kind') == 'admission']
                except Exception: raw['cleanup_errors'].append(traceback.format_exc())
            for task in requests.values():
                if not task.done(): task.cancel()
            if requests: await asyncio.gather(*requests.values(), return_exceptions=True)
            raw['partial_outputs'] = {str(i): task.result() for i, task in requests.items()
                if not task.cancelled() and task.exception() is None}
            if backend:
                for rid in dict.fromkeys(owned):
                    try:
                        errors = await backend.cancel(rid)
                        if errors: raw['cleanup_errors'].append(dict(request_id=rid, errors=errors))
                    except Exception: raw['cleanup_errors'].append(traceback.format_exc())
            if runner:
                try: await runner.cleanup()
                except Exception: raw['cleanup_errors'].append(traceback.format_exc())
            owner = getattr(controller, 'clock_owner', None)
            if owner and any(not handle.closed for handle in owner.files):
                try: await owner.close()
                except Exception: raw['cleanup_errors'].append(traceback.format_exc())
            if backend: raw['cleanup_errors'].extend(await restore(backend, original))
            if offsets and 'events' not in raw:
                try: raw['events'] = await asyncio.to_thread(read_events, args.runtime_dir, offsets)
                except Exception: raw['cleanup_errors'].append(traceback.format_exc())
            if raw['cleanup_errors']: raw['passed'] = False
            await asyncio.to_thread((args.out / 'raw.json').write_text, json.dumps(raw, indent=2, allow_nan=False))
    if failure is not None: raise RuntimeError('EcoServe mechanism validation failed; see raw.json') from failure
    if not raw['passed']: raise RuntimeError('EcoServe validation cleanup failed; see raw.json')
    print(json.dumps(dict(passed=True, purpose=PURPOSE, checks=raw['checks'])), flush=True)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--short-input', type=int, default=128)
    parser.add_argument('--long-input', type=int, default=1024)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--port', type=int, default=0, help='temporary controller port; zero chooses a free local port')
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 <= args.port < 65536: parser.error('positive timeout and valid local port required')
    async def run():
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
        try: return await validate(args)
        finally: loop.remove_signal_handler(signal.SIGTERM)
    with node_lease(): asyncio.run(run())


if __name__ == '__main__': main()
