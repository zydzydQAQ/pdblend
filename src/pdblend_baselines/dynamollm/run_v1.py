"""Execute independent DynamoLLM against the pinned native V1 serving bridge.

The trace is supplied as an immutable artifact, allowing all systems to consume
exactly the shared seed-701 request schedule. A successful functional/primitive
run is development evidence; it never grants complete reproduction or energy
comparison eligibility.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
import random
from pathlib import Path
import sys
import time

from .deployment import SubprocessLifecycle, save, sha
from .policy import PERIODS
from .reconfiguration import Transition
from .runtime import DynamoController
from .transport import V1Transport
from .validation import COMPARISON_DURATIONS, MODELS, preflight
from pdblend.results.journal import CompactJournal, file_sha256


async def collect_functional_profile_stage(config, output):
    """Collect missing predictor envelope from resident instances.

    This path is deliberately opt-in.  Each resident instance owns half of
    the frequency cells, and the collectors only use its already-running HTTP
    endpoint; they never start or stop the engine.  The resulting own profiles
    are merged before DynamoController is constructed.
    """
    stage = config.get('functional_profile_stage')
    if not isinstance(stage, dict) or stage.get('enabled') is not True:
        return None
    if config.get('mode', 'functional') != 'functional':
        raise ValueError('functional profile stage is only allowed in functional mode')
    instances = config.get('instances', [])
    if len(instances) != 2:
        raise ValueError('functional profile stage requires exactly two resident instances')
    groups = stage.get('frequency_groups', [[900, 1200, 1500], [1800, 2100, 2520]])
    if len(groups) != 2 or any(not group for group in groups):
        raise ValueError('functional profile stage requires two nonempty frequency groups')
    if any(len(group) != len(set(group)) for group in groups) or set(groups[0]) & set(groups[1]):
        raise ValueError('functional profile frequency groups must be unique and disjoint')
    if set(sum(groups, [])) != {900, 1200, 1500, 1800, 2100, 2520}:
        raise ValueError('functional profile stage must cover exactly the six native frequencies')
    if any(instance.get('tp') != instances[0].get('tp') for instance in instances):
        raise ValueError('functional profile stage instances must use the same TP')
    receipt_path = stage.get('prediction_receipt')
    if not receipt_path:
        raise ValueError('functional profile stage requires predictor-bound prediction_receipt')
    prediction_receipt = json.loads(Path(receipt_path).read_text())
    if prediction_receipt.get('model_id') != config.get('model_id') or prediction_receipt.get('trace_sha256') != stage.get('trace_sha256'):
        raise ValueError('functional profile prediction receipt model/trace identity differs')
    if sha(Path(config['trace'])) != stage.get('trace_sha256'):
        raise ValueError('functional profile trace checksum differs from the bound receipt')
    predictor_manifest = Path(config['dynamo_predictor_dir']) / 'manifest.json'
    if prediction_receipt.get('predictor_manifest_sha256') != sha(predictor_manifest):
        raise ValueError('functional profile predictor manifest checksum differs')
    trace = json.loads(Path(config['trace']).read_text())
    predictions, requests = prediction_receipt.get('predictions', []), trace.get('requests', [])
    if (prediction_receipt.get('seed') != 701 or trace.get('seed') != 701
            or not requests or len(predictions) != len(requests)
            or any(row.get('request_index') != index
                   or row.get('input_tokens') != len(request['prompt'])
                   or row.get('trace_max_tokens') != request['max_tokens']
                   or type(row.get('predicted_output')) is not int or row['predicted_output'] < 1
                   for index, (row, request) in enumerate(zip(predictions, requests)))):
        raise ValueError('functional profile prediction receipt must bind every trace request')
    if sha(Path(receipt_path)) != stage.get('prediction_receipt_sha256'):
        raise ValueError('functional profile prediction receipt checksum differs')
    predicted = {row['predicted_output'] for row in predictions}
    if not predicted or predicted != set(stage.get('outputs', [])):
        raise ValueError('functional profile outputs are not exactly predictor-bound')
    model = Path(config['model_path'])
    output_root = output / stage.get('output_dir', 'functional-profile-stage')
    output_root.mkdir(parents=True, exist_ok=True)
    processes = []
    timeout = float(stage.get('collector_timeout_s', 1200.))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('finite positive profile collection deadline required')
    try:
        for index, (instance, frequencies) in enumerate(zip(instances, groups)):
            if len(instance.get('gpus', [])) != int(instance.get('tp', 0)):
                raise ValueError('profile stage instance GPU list must equal TP')
            profile_out = output_root / ('instance-' + str(index))
            command = [sys.executable, '-m', 'pdblend_baselines.dynamollm.profile_v1',
                       '--model', str(model), '--gpus', ','.join(map(str, instance['gpus'])),
                       '--tp', str(instance['tp']), '--base-port', str(instance['port']),
                       '--existing-url', str(instance['url']), '--instance-id', str(instance.get('id', index)),
                       '--out', str(profile_out), '--inputs', '128,512,2048',
                       '--outputs', ','.join(map(str, stage['outputs'])),
                       '--batches', '1', '--freqs', ','.join(map(str, frequencies)),
                       '--settle', str(stage.get('settle_s', 2)), '--measure', str(stage.get('measure_s', 5))]
            processes.append(await asyncio.create_subprocess_exec(*command))
        async def wait_success(process):
            status = await process.wait()
            if status != 0:
                raise RuntimeError('functional profile collector failed: ' + repr(status))
        await asyncio.wait_for(asyncio.gather(*(wait_success(process) for process in processes)), timeout=timeout)
    finally:
        if processes:
            await _stop_profile_collectors(processes,
                terminate_timeout_s=float(stage.get('terminate_timeout_s', 10)), kill_timeout_s=5)
    from .prepare_v1 import merge_own_profiles
    profiles = [output_root / ('instance-' + str(i)) / 'profile.json' for i in range(2)]
    merged = merge_own_profiles(profiles, model_id=config['model_id'])
    merged_path = output_root / 'merged-profile.json'
    save(merged_path, merged)
    return dict(path=str(merged_path.resolve()), profiles=[str(p.resolve()) for p in profiles],
                frequencies=groups, points=len(merged['points']), formal_eligible=False,
                hardware_qualified=False)


class Journal:
    def __init__(self, path):
        self.path = Path(str(path)+'.gz') if Path(path).suffix != '.gz' else Path(path)
        self.rows = []
        self.power_samples = 0
        self.power_errors = 0
        self.file = CompactJournal(self.path)

    def __call__(self, event, **fields):
        row = dict(event=event, at_s=time.time(), **fields)
        self.file.write(row)
        if isinstance(row.get('payload'), dict):
            row = dict(row, payload={key:value for key,value in row['payload'].items()
                                    if key not in ('text','choices')})
        # High-frequency power samples live in the durable stream, not RAM.
        if event != 'dynamo_power':
            self.rows.append(row)
        elif 'error' in row:
            self.power_errors += 1
        else:
            self.power_samples += 1

    def close(self):
        self.file.close()

    def checkpoint(self):
        self.file.checkpoint()


async def _stop_profile_collectors(processes, *, terminate_timeout_s=10, kill_timeout_s=5):
    """Bounded cleanup for clock-owning profile subprocesses."""
    for process in processes:
        if process.returncode is None:
            try: process.terminate()
            except ProcessLookupError: pass
    results = await asyncio.gather(*(
        asyncio.wait_for(process.wait(), timeout=terminate_timeout_s)
        for process in processes), return_exceptions=True)
    for process, result in zip(processes, results):
        if isinstance(result, asyncio.TimeoutError) and process.returncode is None:
            try: process.kill()
            except ProcessLookupError: pass
    await asyncio.gather(*(
        asyncio.wait_for(process.wait(), timeout=kill_timeout_s)
        for process in processes), return_exceptions=True)
    if any(process.returncode is None for process in processes):
        raise RuntimeError('profile collector did not exit after TERM/KILL deadlines')


def load_trace(path, duration_s):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get('seed') != 701:
        raise ValueError('shared trace requires explicit seed 701 manifest')
    result = []
    previous = -1
    for index, row in enumerate(value['requests']):
        arrival = row.get('arrival_s', row.get('at_s'))
        prompt = row.get('prompt')
        maximum = row.get('max_tokens', row.get('output_tokens'))
        if (type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < previous
                or not 0 <= arrival < duration_s or not isinstance(prompt, list) or not prompt
                or any(type(token) is not int or token < 0 for token in prompt)
                or type(maximum) is not int or not 1 <= maximum <= 512
                or len(prompt) + maximum > 8192):
            raise ValueError('invalid shared trace request at index ' + str(index))
        previous = arrival
        result.append(dict(request_id='dynamo-701-' + str(index), arrival_s=arrival,
                           prompt=prompt, max_tokens=maximum, source=row.get('source', '')))
    if not result:
        raise ValueError('nonempty shared trace required')
    return result


def qualify(rows, outcomes, *, mode, duration_s):
    routes = {row.get('request_id') for row in rows if row['event'] == 'dynamo_route'}
    request_ids = {row['request_id'] for row in outcomes}
    passed = bool(outcomes) and request_ids <= routes and all(row.get('ok') is True for row in outcomes)
    failures = [] if passed else ['request_output_or_real_route_missing']
    transitions = [row for row in rows if row['event'] == 'dynamo_transition' and row.get('phase') == 'complete']
    if mode == 'primitive' and not transitions:
        failures.append('real_weight_transition_missing')
    if mode == 'full':
        if duration_s < 1890:
            failures.append('original_period_window_missing')
        for operation in PERIODS:
            if not any(row['event'] == 'dynamo_control_epoch' and row.get('operation') == operation
                       and row.get('executed') is True and row.get('initialization') is not True for row in rows):
                failures.append('real_' + operation + '_action_missing')
        if not transitions:
            failures.append('real_weight_transition_missing')
    if mode == 'comparison' and duration_s not in COMPARISON_DURATIONS:
        failures.append('comparison_window_requires_150_or_300_seconds')
    return dict(status='passed' if not failures else 'inconclusive', complete=not failures,
                failures=failures,
                scope='development_' + mode, formal_eligible=False, hardware_qualified=False,
                energy_comparable=False, complete_reproduction=False, periods_s=dict(PERIODS),
                seed=701, requests=len(outcomes), routed_requests=len(routes & request_ids),
                successful_requests=sum(row.get('ok') is True for row in outcomes),
                completed_transitions=len(transitions),
                short_window_dynamic_tp_benefit_claim=False)


async def execute_on_resident(config, trace, *, session, output, duration_s=150,
                              mode='comparison', receipt=None):
    """Run a fresh controller on a caller-owned independent Dynamo session.

    Session refusal never falls back silently. Close it and call ``execute``
    with a new output directory for an independently owned launch instead.
    """
    return await session.execute_window(trace, config=config, output=output,
        duration_s=duration_s, mode=mode, receipt=receipt)


async def execute(config, trace, *, output, duration_s, mode, receipt):
    from .telemetry import GroupTelemetry
    from .relay import StagedGpuTopologyHooks
    journal = Journal(output / 'events.jsonl')
    telemetry = transport = lifecycle = controller = None
    outcomes = []
    tasks = []
    cleanup_errors = []
    failure = None
    stage_receipt = None
    try:
        telemetry = GroupTelemetry(config['node_gpus'], journal)
        telemetry.start()
        transport = V1Transport(config['instances'], telemetry.clock, journal)
        await transport.start()
        lifecycle = SubprocessLifecycle(config, transport, journal, output / 'engines')
        for spec in config['instances']:
            # Model loads are deliberately staggered; sampling/requests run in parallel.
            await lifecycle.start(spec)
        # Optional development-only coverage collection runs against the
        # resident endpoints before the controller loads its profile.  The
        # default functional/primitive/full paths are byte-for-byte unchanged.
        if config.get('functional_profile_stage', {}).get('enabled') is True:
            stage_receipt = await collect_functional_profile_stage(config, output)
            config['profiles'] = stage_receipt['path']
            journal('dynamo_functional_profile_stage', **stage_receipt)
        # Match the shared development warmup seed without reading workload
        # labels or charging warmup as part of the request service window.
        warm_rng = random.Random(9701)
        warm_prompt = [warm_rng.randint(1000, 60000) for _ in range(128)]
        for iid in transport.instances:
            tokens = []
            async for event in transport.stream(iid, dict(prompt=warm_prompt, max_tokens=16,
                    seed=9701, request_id='dynamo-warmup-' + iid, temperature=0, ignore_eos=True)):
                tokens.extend(event.get('token_ids', []))
            if len(tokens) != 16:
                raise RuntimeError('Dynamo engine warmup token accounting failed')
            journal('dynamo_warmup', instance_id=iid, seed=9701, input_tokens=128, output_tokens=len(tokens))
        if mode != 'functional':
            goldens = {int(tp): value for tp, value in config['goldens'].items()}
            transport.dynamo_topology = StagedGpuTopologyHooks(transport, lifecycle, journal,
                goldens=goldens, store_port=config.get('store_port', config['base_port']+80))
        controller = DynamoController(config, transport, journal)
        await controller.startup()
        started = time.monotonic()
        journal('dynamo_service_window_start', duration_s=duration_s,
                gpu_uuids=telemetry.uuids, mode=mode, profile_sha256=controller.profiles.fingerprint)

        async def request(row):
            await asyncio.sleep(max(0, started + row['arrival_s'] - time.monotonic()))
            outcome = dict(request_id=row['request_id'], submitted_s=time.time(), token_ids=[], ok=False)
            stream = controller.handle(dict(prompt=row['prompt'], max_tokens=row['max_tokens'],
                                            temperature=0, seed=701, ignore_eos=True), row['request_id'])
            try:
                terminal = False
                async for event in stream:
                    journal('dynamo_sse', request_id=row['request_id'], payload=event)
                    ids = event.get('token_ids', [])
                    if ids:
                        outcome.setdefault('first_token_s', time.time())
                        outcome['token_ids'].extend(ids)
                    terminal |= any(c.get('finish_reason') is not None for c in event.get('choices', []))
                outcome['ok'] = terminal and len(outcome['token_ids']) == row['max_tokens']
            except Exception as exc:
                outcome['error'] = repr(exc)
            finally:
                await stream.aclose()
                outcome['finished_s'] = time.time()
                ids=outcome.pop('token_ids')
                outcome.update(completion_tokens=len(ids),journal_path=journal.path.name,
                    token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
                outcomes.append(outcome)
                journal('dynamo_outcome', **outcome)

        tasks = [asyncio.create_task(request(row)) for row in trace]

        async def primitive():
            await asyncio.sleep(float(config.get('primitive_at_s', 10)))
            spec = config['primitive']
            sources = [controller.replicas[iid] for iid in spec['source_ids']]
            transition = Transition('primitive-701-' + str(time.time_ns()),
                tuple(i.instance_id for i in sources), tuple(i.gpus for i in sources),
                tuple(tuple(g) for g in spec['target_layout']), overlap_memory_qualified=False,
                timeout_s=config.get('transition_timeout_s', 600),
                target_shapes=tuple(spec['target_shapes']))
            async def commit(prepared):
                # Staged hooks have already inserted verified target replicas.
                async with controller.lock:
                    for iid in transition.source_ids:
                        controller.replicas.pop(iid, None)
            return await controller.transitions.execute(transition, commit=commit)

        if mode == 'primitive':
            tasks.append(asyncio.create_task(primitive()))
        await asyncio.sleep(duration_s)
        await asyncio.wait_for(asyncio.gather(*tasks), config.get('request_drain_timeout_s', 600))
        journal('dynamo_service_window_end', elapsed_s=time.monotonic()-started)
    except BaseException as exc:
        failure = repr(exc)
        journal('dynamo_run_failure', error=failure)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # A forcibly terminated profile collector cannot run its NVML finally;
        # mark the owned GPUs changed so GroupTelemetry.close() resets clocks.
        if failure and telemetry is not None:
            try:
                telemetry.clock(config['node_gpus'], 900)
            except BaseException as exc:
                cleanup_errors.append(dict(component='profile_stage_clock_recovery', error=repr(exc)))
        for name, resource in [('controller', controller), ('lifecycle', lifecycle),
                               ('telemetry', telemetry), ('transport', transport)]:
            if resource is not None:
                try:
                    await resource.close()
                except BaseException as exc:
                    cleanup_errors.append(dict(component=name, error=repr(exc)))
        result = qualify(journal.rows, outcomes, mode=mode, duration_s=duration_s)
        if journal.power_samples < 2:
            result['failures'].append('group_power_samples_missing')
            result['status'] = 'inconclusive'
        if failure or cleanup_errors:
            result.update(status='failed', complete=False, error=failure, cleanup_errors=cleanup_errors)
        result.update(system='dynamollm', model_id=config['model_id'],
                      own_cleanup_complete=not cleanup_errors,
                      provenance=receipt['evidence'], duration_s=duration_s, warmup_seed=9701,
                      group_power_samples=journal.power_samples, group_power_errors=journal.power_errors,
                      gpu_uuids=telemetry.uuids if telemetry else {},
                      stationary_weight_bytes=0, original_weight_retention_implemented=False)
        result.update(journal_path=journal.path.name,raw_schema='pdblend-journal-v1')
        if stage_receipt is not None:
            result['functional_profile_stage'] = stage_receipt
        journal.close()
        result['events_sha256']=file_sha256(journal.path)
        save(output / 'outcomes.json', outcomes)
        save(output / 'completion.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=100)
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--mode', choices=('functional', 'primitive', 'full', 'comparison'), default='functional')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('positive finite --duration required')
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in ('completion.json','events.jsonl','events.jsonl.gz')):
        raise FileExistsError('refusing to overwrite a Dynamo execution artifact')
    config = json.loads(args.config.read_text())
    # Queue workers bind a lease-local HTTP base port at launch.  Keeping this
    # substitution in the runtime makes the frozen config portable while
    # preserving deterministic +2 rank/instance offsets.
    lease_port = os.environ.get('PDBLEND_LEASE_PORT')
    if lease_port is not None:
        try:
            lease_port = int(lease_port)
        except ValueError:
            parser.error('PDBLEND_LEASE_PORT must be an integer')
        config['base_port'] = lease_port
        for index, instance in enumerate(config.get('instances', [])):
            instance['port'] = lease_port + 2 * index
            instance['url'] = 'http://127.0.0.1:' + str(instance['port'])
        config['target_port'] = lease_port + 32
        config['store_port'] = lease_port + 80
    config['legal_tp'] = list(MODELS.get(config.get('model_id'), ()))
    config['mode'] = args.mode
    if config.get('functional_profile_stage', {}).get('enabled') is True and args.mode != 'functional':
        parser.error('functional_profile_stage is only valid for --mode functional')
    config['tokenizer'] = config.get('model_path')
    config['dynamo_require_full_mechanisms'] = args.mode in ('full', 'comparison')
    config.setdefault('dynamo_reference_tp', 4)
    config.setdefault('base_port', 16000)
    receipt = preflight(config, mode=args.mode, duration_s=args.duration, seed=args.seed)
    receipt['evidence'].update(config_sha256=sha(args.config),
        image_id=os.environ.get('PDBLEND_IMAGE_ID'), source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'))
    trace = None
    if not args.preflight_only:
        try:
            trace_path = args.trace or Path(config['trace'])
            trace = load_trace(trace_path, args.duration)
            config['trace'] = str(trace_path)
            receipt['evidence']['trace_sha256'] = sha(trace_path)
        except (KeyError, OSError, ValueError, TypeError) as exc:
            receipt['missing_evidence']['trace'] = str(exc)
            receipt.update(ready=False, status='inconclusive')
    save(args.out / 'preflight.json', receipt)
    if not receipt['ready'] or args.preflight_only:
        save(args.out / 'completion.json', dict(receipt, scope='preflight_only',
                                               hardware_actions_started=False))
        print(json.dumps(receipt))
        return 0 if receipt['ready'] else 2
    save(args.out / 'config.json', config)
    save(args.out / 'trace.json', dict(seed=701, requests=trace))
    result = asyncio.run(execute(config, trace, output=args.out, duration_s=args.duration,
                                 mode=args.mode, receipt=receipt))
    print(json.dumps(result))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
