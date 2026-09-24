"""Independent Dynamo mixed-instance measurement on real V1 workers.

Only this collector's native/SSE/power samples can create its PaperProfiles
table. It measures the chosen envelope, resumes completed immutable cells, and
does not load any PDBlend performance/capacity model. Profiles remain development
artifacts until a separate coverage/provenance review admits the envelope.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from .deployment import SubprocessLifecycle, save, sha
from .predictor import model_identity
from .run_v1 import Journal
from .telemetry import GroupTelemetry
from .transport import V1Transport
from .validation import MODELS

FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)


def capacity_limit(point, state):
    """A native inventory refusal is evidence of unsupported geometry, not a fit."""
    capacity = state.get('total_kv_tokens')
    if type(capacity) is not int or capacity <= 0:
        raise ValueError('actual positive native KV capacity required before profile submission')
    required = point['batch'] * (point['input_tokens'] + point['output_tokens'])
    return dict(supported=required <= capacity, required_kv_tokens=required,
                actual_total_kv_tokens=capacity, policy_max_num_seqs_unchanged=True,
                measured=False, formal_eligible=False)


def measurement_points(args, model_id):
    """Explicit sparse missing cells; never expand them to a Cartesian rerun."""
    path = getattr(args, 'points_file', None)
    if path:
        plan = json.loads(Path(path).read_text())
        if (plan.get('schema') != 'dynamo-missing-profile-cells-v1'
                or plan.get('system') != 'dynamollm' or plan.get('model_id') != model_id
                or plan.get('tp') != args.tp or plan.get('pp') != 1
                or plan.get('fit_from_holdout') is not False):
            raise ValueError('independent sparse profile plan identity differs')
        points = plan.get('points', [])
    else:
        points = [dict(frequency_mhz=f, input_tokens=n, output_tokens=o, batch=b)
                  for f,n,o,b in product(args.freqs,args.inputs,args.outputs,args.batches)]
    keys=('frequency_mhz','input_tokens','output_tokens','batch')
    if not points or any(set(p)!=set(keys) or any(type(p[k]) is not int for k in keys)
            or p['frequency_mhz'] not in FREQUENCIES or not 1<=p['input_tokens']<=7168
            or not 2<=p['output_tokens']<=512 or not 1<=p['batch']<=256
            or p['input_tokens']+p['output_tokens']>8192 for p in points):
        raise ValueError('invalid explicitly measured Dynamo workload cells')
    ordered = sorted(points,key=lambda p:tuple(p[k] for k in keys))
    if len({tuple(p[k] for k in keys) for p in ordered})!=len(ordered):
        raise ValueError('duplicate sparse profile cell')
    return ordered


def key_for(identity, point):
    return hashlib.sha256(json.dumps(dict(identity=identity, point=point), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def reduce_window(window, *, tp, batch, frequency):
    """Fail closed on native/rank/power gaps; no synthetic time substitutes."""
    if window['settle_s'] < 2 or window['finished_s'] - window['started_s'] < 5:
        raise ValueError('decode windows require >=2s settle and >=5s measurement')
    ranks = window['native']['ranks']
    if len(ranks) != tp or {row['rank'] for row in ranks} != set(range(tp)):
        raise ValueError('native all-rank coverage missing')
    counts = []
    cuda = []
    for rank in ranks:
        decode = [row for row in rank['samples'] if row.get('role') == 'decode' and row.get('batch') == batch]
        if len(decode) < 8:
            raise ValueError('fewer than 8 actual decode steps at requested batch')
        for row in rank['samples']:
            if (row.get('system') != 'dynamollm' or row.get('measurement_scope') != 'runner'
                    or row.get('tp') != tp or row.get('pp') != 1 or row.get('failed')
                    or not math.isfinite(row.get('gpu_elapsed_ms', 0)) or row['gpu_elapsed_ms'] <= 0):
                raise ValueError('invalid independent native CUDA sample')
        counts.append(len(decode))
        cuda.append(statistics.median(row['gpu_elapsed_ms'] for row in decode))
    gpus = window['gpu_ids']
    if len(set(gpus)) != tp:
        raise ValueError('power GPU group differs from requested TP')
    powers = []
    for gpu in gpus:
        rows = [row for row in window['power'] if row.get('gpu') == gpu and 'error' not in row
                and window['started_s'] <= row['timestamp'] <= window['finished_s']]
        if len(rows) < 2 or not any('frequency_mhz' in row for row in rows):
            raise ValueError('at least 2 group power and 1 frequency samples per GPU required')
        # Idle clocks between bursts are recorded but cannot stand in for an
        # observed active locked-clock sample in the requested frequency bin.
        if not any(abs(row['frequency_mhz'] - frequency) <= max(30, .05*frequency) for row in rows):
            raise ValueError('requested frequency never observed during window')
        if any(row.get('source') != 'nvml:field:186:scope:0:mW' or row.get('power_w', 0) <= 0 for row in rows):
            raise ValueError('independent instantaneous power receipt missing')
        powers.append(statistics.mean(row['power_w'] for row in rows))
    requests = window['requests']
    if not requests or any(row.get('ok') is not True or row['tokens'] < 2 for row in requests):
        raise ValueError('actual complete SSE request evidence required')
    iterations = [(row['finished_s']-row['first_token_s'])/(row['tokens']-1) for row in requests]
    iteration = statistics.median(iterations)
    ttft = statistics.median(row['first_token_s']-row['submitted_s'] for row in requests)
    if not math.isfinite(iteration) or iteration <= 0 or ttft <= iteration:
        raise ValueError('Dynamo measured TTFT/iteration decomposition not positive')
    # This is the independent historical Dynamo observation estimator. Native
    # CUDA samples validate actual decode geometry, not a replacement for TTFT.
    return dict(prefill_s=(ttft-iteration)/batch, iteration_s=iteration,
                power_w=sum(powers), ttft_s=ttft, decode_steps=min(counts),
                native_decode_cuda_ms=max(cuda), measurement='hardware')


def fit_cell(windows, holdout, *, tp, point):
    if len(windows) < 3:
        raise ValueError('three independent decode window repeats required')
    estimates = [reduce_window(row, tp=tp, batch=point['batch'], frequency=point['frequency_mhz']) for row in windows]
    estimate = {key: statistics.median(row[key] for row in estimates)
                for key in ('prefill_s', 'iteration_s', 'power_w')}
    observed = reduce_window(holdout, tp=tp, batch=point['batch'], frequency=point['frequency_mhz'])
    prediction = point['batch']*estimate['prefill_s']+estimate['iteration_s']
    errors = dict(ttft=abs(prediction/observed['ttft_s']-1),
                  iteration=abs(estimate['iteration_s']/observed['iteration_s']-1),
                  power=abs(estimate['power_w']/observed['power_w']-1))
    return dict(**estimate, holdout_errors=errors, holdout_passed=max(errors.values()) <= .10,
                decode_steps=[row['decode_steps'] for row in estimates], repeats=len(windows))


async def measure_window(transport, telemetry, iid, point, *, repeat, settle_s, measure_s, ownership=None,
                         before_measure=None):
    receipt = await transport.json(iid, '/baseline/measurement/start', dict(scope='runner', system='dynamollm'))
    if receipt.get('acknowledged') is not True:
        raise RuntimeError('independent measurement scope was not acknowledged')
    if ownership is not None:
        ownership['started'] = True
    requests = []
    native = {rank: [] for rank in range(transport.instances[iid]['tp'])}
    burst = 0
    async def run_burst():
        async def request(index):
            rng = random.Random(701+point['input_tokens']+repeat*100000+burst*128+index)
            prompt = [rng.randint(1000, 60000) for _ in range(point['input_tokens'])]
            rid = f'dynamo-profile-{point["frequency_mhz"]}-{repeat}-{burst}-{index}-{time.time_ns()}'
            row = dict(request_id=rid, submitted_s=time.time(), tokens=0, ok=False)
            async for event in transport.stream(iid, dict(prompt=prompt, max_tokens=point['output_tokens'],
                    request_id=rid, seed=701, temperature=0, ignore_eos=True)):
                if event.get('token_ids'):
                    row.setdefault('first_token_s', time.time())
                    row['tokens'] += len(event['token_ids'])
                if event.get('finished'):
                    row['ok'] = True
            row['finished_s'] = time.time()
            row['ok'] &= row['tokens'] == point['output_tokens']
            return row
        return await asyncio.gather(*(request(index) for index in range(point['batch'])))
    # Settle under the measured workload, not an idle GPU sleep. Warm-up
    # outputs/events are retained separately and excluded from fit inputs.
    settled = time.monotonic()
    warmup=[]
    while time.monotonic()-settled < settle_s:
        warmup.extend(await run_burst())
        burst += 1
    if before_measure is not None:
        # Stay under the same workload while the other members reach their
        # barrier. An idle wait would invalidate the just-completed settling.
        boundary = asyncio.create_task(before_measure())
        try:
            while not boundary.done():
                warmup.extend(await run_burst())
                burst += 1
                await asyncio.sleep(0)
            await boundary
        finally:
            if not boundary.done():
                boundary.cancel()
                await asyncio.gather(boundary, return_exceptions=True)
    warmup_native=await transport.json(iid, '/baseline/measurement/samples', method='GET')
    actual_settle = time.monotonic()-settled
    started = time.time()
    while True:
        requests.extend(await run_burst())
        rows = await transport.json(iid, '/baseline/measurement/samples', method='GET')
        if {row['rank'] for row in rows['ranks']} != set(native):
            raise RuntimeError('incomplete CUDA rank samples')
        for row in rows['ranks']:
            native[row['rank']].extend(row['samples'])
        burst += 1
        steps = [sum(row['role'] == 'decode' and row['batch'] == point['batch'] for row in samples)
                 for samples in native.values()]
        if time.time()-started >= measure_s and min(steps) >= 8:
            break
        if time.time()-started > max(60, measure_s*4):
            raise RuntimeError('requested batch cannot produce qualifying native decode steps')
    finished = time.time()
    return dict(point=point, repeat=repeat, settle_s=actual_settle, started_s=started, finished_s=finished,
                warmup_requests=warmup, warmup_native=warmup_native,
                requests=requests, native={'ranks': [dict(rank=rank, samples=rows) for rank, rows in native.items()]},
                gpu_ids=transport.instances[iid]['gpus'],
                power=[row for row in telemetry.readings if started <= row['timestamp'] <= finished])


def completed_cells(output, identity):
    result = {}
    for path in sorted((output/'cells').glob('*.json')):
        value = json.loads(path.read_text())
        if value.get('identity') != identity:
            raise ValueError('resume profile source/model/image identity changed')
        if path.stem != key_for(identity, value['point']):
            raise ValueError('profile cell filename identity mismatch')
        for name, digest in value.get('artifacts', {}).items():
            if sha(output/name) != digest:
                raise ValueError('completed profile raw sample checksum differs')
            if name.endswith(('-repeat0.json', '-repeat1.json', '-repeat2.json', '-holdout.json')):
                from .profile_epochs import validate_window_qualification
                verified = validate_window_qualification(json.loads((output/name).read_text()), output)
                if identity.get('sampling_protocol') == 'cohort-epochs-v1' and verified is None:
                    raise ValueError('completed epoch profile repeat lacks qualification')
        capability = output/value.get('capability_path', 'capability.json')
        if not capability.is_file() or sha(capability) != value.get('capability_sha256'):
            raise ValueError('completed profile native capability checksum differs')
        if value.get('fit', {}).get('holdout_passed') is True:
            result[path.stem] = (path, value)
    return result


def freeze_capability(output, capability):
    """Keep every cell's native identity receipt immutable across restarts."""
    raw = (json.dumps(capability, indent=2, allow_nan=False)+'\n').encode()
    digest = hashlib.sha256(raw).hexdigest()
    relative = 'capabilities/'+digest+'.json'
    path = output/relative
    if path.exists():
        if sha(path) != digest:
            raise ValueError('immutable native capability receipt was modified')
    else:
        save(path, capability)
    # Legacy cells bind this filename. Never overwrite it when the next
    # engine reports a new timestamp, generation, or other live state.
    if not (output/'capability.json').exists():
        save(output/'capability.json', capability)
    save(output/'current-capability-reference.json', dict(path=relative, sha256=digest))
    return relative, digest


async def collect_golden(transport, iid, *, output, capability, tp):
    """Save a real same-TP baseline for subsequent dummy-target verification."""
    rng = random.Random(701)
    prompt = [rng.randint(1000, 60000) for _ in range(128)]
    tokens, events = [], []
    async for event in transport.stream(iid, dict(prompt=prompt, max_tokens=16, seed=701,
            request_id='dynamo-profile-golden-'+str(time.time_ns()), ignore_eos=True)):
        tokens.extend(event.get('token_ids', []))
        events.append(event)
    if len(tokens) != 16 or not events or not events[-1].get('finished'):
        raise RuntimeError('same-TP golden output incomplete')
    raw = output/'golden-raw.json'
    save(raw, dict(capability=capability, prompt=prompt, token_ids=tokens, seed=701, events=events))
    golden = {key: capability[key] for key in ('model_id', 'engine_revision', 'model_hash',
        'tokenizer_hash', 'image_digest', 'source_revision', 'verification_receipt_sha256') if key in capability}
    golden.update(tp=tp, prompt=prompt, token_ids=tokens, seed=701,
                  source_path=str(raw.resolve()), source_sha256=sha(raw), formal_eligible=False)
    save(output/'goldens.json', {str(tp): golden})
    return golden


async def collect(args):
    output = args.out
    identity = model_identity(args.model)
    if args.tp not in MODELS.get(identity['model'], ()) or len(args.gpus) != args.tp:
        raise ValueError('Dynamo collector requires one legal PP1 instance and exactly TP GPUs')
    selected_points = measurement_points(args, identity['model'])
    metadata = dict(system='dynamollm', model_id=identity['model'], model_identity=identity,
        tp=args.tp, pp=1, engine_revision='vllm-0.10.1.1', source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'),
        image_digest=os.environ.get('PDBLEND_IMAGE_ID'), seed=701)
    if getattr(args, 'points_file', None):
        metadata['measurement_plan_sha256'] = sha(args.points_file)
    previous = {}
    if (output/'profile.json').exists() and not args.resume:
        raise FileExistsError('use --resume to retain independently measured cells')
    output.mkdir(parents=True, exist_ok=True)
    journal = Journal(output/('events-'+str(time.time_ns())+'.jsonl'))
    telemetry = transport = lifecycle = epochs = None
    iid = getattr(args, 'instance_id', 'dynamo-profile')
    existing_url = getattr(args, 'existing_url', None)
    epoch_root = getattr(args, 'sampling_epoch_root', None) or os.environ.get('PDBLEND_SAMPLING_EPOCH_ROOT')
    epoch_member = getattr(args, 'sampling_member', None) or os.environ.get('PDBLEND_PROFILE_MEMBER')
    if bool(epoch_root) != bool(epoch_member):
        raise ValueError('sampling epoch root and unique member must be provided together')
    if getattr(args, 'require_sampling_epochs', False) and not epoch_root:
        raise ValueError('this profile job requires a frozen shared sampling cohort')
    if epoch_root and existing_url:
        raise ValueError('sampling epoch retirement requires ownership of the native model lifecycle')
    if epoch_root:
        metadata['sampling_protocol'] = 'cohort-epochs-v1'
    identity_verified = False
    measurement_ownership = {'started': False}
    failures = []
    try:
        telemetry = GroupTelemetry(args.gpus, journal)
        metadata['gpu_uuids'] = {str(gpu): uuid for gpu, uuid in telemetry.uuids.items()}
        if args.resume:
            previous = completed_cells(output, metadata)
        telemetry.start()
        spec = dict(id=iid, instance_id=iid, tp=args.tp, gpus=args.gpus, port=args.base_port,
                    url=existing_url or 'http://127.0.0.1:'+str(args.base_port), generation=0, shape='LL')
        config = dict(model_id=identity['model'], model_path=str(args.model), legal_tp=list(MODELS[identity['model']]),
                      node_gpus=args.gpus, base_port=args.base_port, instances=[spec],
                      max_num_seqs=max(16, max(p['batch'] for p in selected_points)))
        transport = V1Transport([spec], telemetry.clock, journal)
        await transport.start()
        if not existing_url:
            lifecycle = SubprocessLifecycle(config, transport, journal, output/'engines')
            await lifecycle.start(spec)
        capability = await transport.json(iid, '/baseline/capability', method='GET')
        if (capability.get('model_id') != identity['model'] or capability.get('engine_revision') != 'vllm-0.10.1.1'
                or capability.get('tp') != args.tp or capability.get('pp') != 1):
            raise RuntimeError('independent native profiler identity mismatch')
        if set(capability.get('gpu_uuids', [])) != set(telemetry.uuids.values()):
            raise RuntimeError('native profile endpoint differs from allocated GPU group')
        if (capability.get('source_revision') != metadata['source_sha256'] or
                capability.get('image_digest') != metadata['image_digest']):
            raise RuntimeError('native profile endpoint source/image identity differs')
        transport.instances[iid]['generation']=capability['state']['generation']
        identity_verified = True
        capability_path, capability_sha256 = freeze_capability(output, capability)
        await collect_golden(transport, iid, output=output, capability=capability, tp=args.tp)
        if epoch_root:
            from .profile_epochs import DynamoEpochs
            epochs = DynamoEpochs(epoch_root, epoch_member, output, metadata,
                                  transport, telemetry, iid, measurement_ownership)
            await epochs.ready()
        async def measured_repeat(point, repeat):
            if epochs is not None:
                return await epochs.measure(point, repeat, settle_s=args.settle, measure_s=args.measure)
            return await measure_window(transport, telemetry, iid, point, repeat=repeat,
                settle_s=args.settle, measure_s=args.measure, ownership=measurement_ownership)
        # Frequency-major order retains the engine and limits clock transitions.
        for point in selected_points:
            frequency, n, o, batch = (point[k] for k in ('frequency_mhz','input_tokens','output_tokens','batch'))
            key = key_for(metadata, point)
            if key in previous:
                continue
            artifacts, windows = {}, []
            try:
                state = await transport.state(iid)
                limit = capacity_limit(point, state)
                if not limit['supported']:
                    path = output/'unsupported'/f'{key}.json'
                    save(path, dict(point=point, capacity=limit, native_state=state,
                        observed_s=time.time(), identity=metadata, reason='unsupported_memory'))
                    failures.append(dict(point=point, reason='unsupported_memory', capacity=limit,
                        evidence_path=str(path.resolve()), evidence_sha256=sha(path)))
                    save(output/'failures.json', failures)
                    continue
                telemetry.clock(args.gpus, frequency)
                for repeat in range(3):
                    window = await measured_repeat(point, repeat)
                    path = output/'raw'/f'{key}-repeat{repeat}.json'
                    save(path, window)
                    artifacts[str(path.relative_to(output))] = sha(path)
                    windows.append(window)
                # Freeze fit inputs before collecting a separate holdout window.
                frozen = output/'raw'/f'{key}-fit-inputs.json'
                save(frozen, dict(artifacts=artifacts, frozen_at_s=time.time(), point=point))
                artifacts[str(frozen.relative_to(output))] = sha(frozen)
                holdout = await measured_repeat(point, 3)
                holdout_path = output/'raw'/f'{key}-holdout.json'
                save(holdout_path, holdout)
                artifacts[str(holdout_path.relative_to(output))] = sha(holdout_path)
                fitted = fit_cell(windows, holdout, tp=args.tp, point=point)
                cell = dict(identity=metadata, point=point, artifacts=artifacts, fit=fitted,
                    capability_path=capability_path, capability_sha256=capability_sha256, gpu_uuids=telemetry.uuids)
                path = output/'cells'/f'{key}.json'
                save(path, cell)
                if fitted['holdout_passed']:
                    previous[key] = (path, cell)
                else:
                    failures.append(dict(point=point, reason='holdout_error', errors=fitted['holdout_errors']))
            except Exception as exc:
                failures.append(dict(point=point, reason=repr(exc)))
                save(output/'failures.json', failures)
                # No fresh point after an uncertain engine/stream state.
                current = await transport.state(iid)
                if current.get('active') or current.get('running') or current.get('waiting'):
                    raise RuntimeError('profile cell failed with live engine work') from exc
        if args.label_corpus_root:
            from .label_v1 import collect_labels
            await collect_labels(transport, iid, model_path=args.model, corpus_root=args.label_corpus_root,
                                 output=output/'labels', per_dataset=args.label_samples, seed=701)
    except BaseException as exc:
        failures.append(dict(reason='run_failure', error=repr(exc)))
        if epochs is not None:
            epochs.fail(exc)
        raise
    finally:
        cleanup = []
        if epochs is not None:
            try:
                # A completed member must meet peers at a window boundary
                # before it changes the physical concurrency layout.
                await epochs.retire()
                from .native_hooks import aggregate_drain
                receipt = await transport.json(iid, '/baseline/dynamollm/drain', dict(timeout_s=30))
                state = await transport.state(iid)
                drained = aggregate_drain(state, receipt['ranks'], tp=args.tp,
                                           generation=transport.instances[iid]['generation'])
                save(output/'epoch-drain.json', dict(status='passed', complete=True,
                    receipt=receipt, verified=drained))
            except BaseException as exc:
                epochs.fail(exc)
                cleanup.append(dict(component='epoch_drain', error=repr(exc)))
        if transport is not None and existing_url and identity_verified and measurement_ownership['started']:
            try:
                await transport.json(iid, '/baseline/measurement/stop', {})
            except BaseException as exc:
                cleanup.append(dict(component='native_measurement', error=repr(exc)))
        for name, resource in [('lifecycle', lifecycle), ('telemetry', telemetry), ('transport', transport)]:
            if resource is not None:
                try:
                    await resource.close()
                except BaseException as exc:
                    cleanup.append(dict(component=name, error=repr(exc)))
        if epochs is not None:
            if not cleanup and epochs.retirement_permitted:
                epochs.released()
                save(output/'epoch-release.json', dict(status='passed', complete=True, engines_closed=True,
                    clock_cleanup_complete=True, released_s=time.time(), member=epoch_member))
            else:
                epochs.fail(RuntimeError('Dynamo native profile cleanup was not completely verified'))
        journal.close()
        points = []
        for path, cell in previous.values():
            point, fitted = cell['point'], cell['fit']
            points.append(dict(role='mixed', tp=args.tp, pp=1, frequency_mhz=point['frequency_mhz'],
                input_tokens=point['input_tokens'], context_tokens=point['input_tokens']+point['output_tokens'],
                batch=point['batch'], **{k:fitted[k] for k in ('prefill_s', 'iteration_s', 'power_w')},
                samples=3, source_sha256=sha(path), source_profile_path=str(path.resolve()),
                source_profile_sha256=sha(path)))
        profile = dict(schema=2, measurement='hardware', coordinate_system='input_output_batch',
            independent_profile=True, strict_provenance=True, **metadata, model=identity['model'], points=points,
            formal_eligible=False, hardware_qualified=False,
            estimator='own SSE (median TTFT - iteration)/batch and steady SSE iteration; native CUDA geometry checked',
            provenance={'model_identity': identity, 'gpu_uuids': telemetry.uuids if telemetry else {}},
            coverage=dict(frequencies=sorted({point['frequency_mhz'] for point in points}),
                all_six_frequencies=set(point['frequency_mhz'] for point in points)==set(FREQUENCIES),
                missing_points=failures, no_extrapolation=True))
        save(output/'profile.json', profile)
        save(output/'completion.json', dict(status='passed' if points and not failures and not cleanup else 'inconclusive',
            complete=bool(points and not failures and not cleanup),
            points=len(points), missing_points=failures, cleanup_errors=cleanup,
            unsupported_points=[row for row in failures if row.get('reason') == 'unsupported_memory'],
            formal_eligible=False, energy_comparable=False, hardware_qualified=False))
    return profile


def integers(text):
    values = [int(value) for value in text.split(',')]
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError('unique nonnegative integers required')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpus', type=integers, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--base-port', type=int, default=17000)
    parser.add_argument('--existing-url', help='reuse an identity-checked resident native instance')
    parser.add_argument('--instance-id', default='dynamo-profile')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=integers, default=[128, 512, 2048])
    parser.add_argument('--outputs', type=integers, default=[16, 512])
    parser.add_argument('--batches', type=integers, default=[1])
    parser.add_argument('--freqs', type=integers, default=list(FREQUENCIES))
    parser.add_argument('--points-file', type=Path,
                        help='immutable own-system explicit missing cells; suppress Cartesian grid')
    parser.add_argument('--settle', type=float, default=2)
    parser.add_argument('--measure', type=float, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sampling-epoch-root', type=Path)
    parser.add_argument('--sampling-member')
    parser.add_argument('--require-sampling-epochs', action='store_true')
    parser.add_argument('--job-config', type=Path,
                        help='revalidate frozen independent job inputs before any GPU operation')
    parser.add_argument('--label-corpus-root', type=Path,
                        help='optional future natural-completion labels; NOT used by fixed-work campaign')
    parser.add_argument('--label-samples', type=int, default=32)
    args = parser.parse_args(argv)
    if args.job_config:
        from .asset_preflight import check
        config = json.loads(args.job_config.read_text())
        check(config)
        if (config['kind'] != 'profile' or Path(config['model_path']) != args.model
                or config['tp'] != args.tp or config['gpus'] != args.gpus
                or Path(config['points_file']) != args.points_file
                or (config.get('sampling_epoch_root') and
                    (str(args.sampling_epoch_root or os.environ.get('PDBLEND_SAMPLING_EPOCH_ROOT'))
                        != config['sampling_epoch_root']
                     or (args.sampling_member or os.environ.get('PDBLEND_PROFILE_MEMBER'))
                        != config['sampling_member'] or not args.require_sampling_epochs))):
            parser.error('runtime profile arguments differ from frozen independent job')
    if (args.settle < 2 or args.measure < 5 or not set(args.freqs) <= set(FREQUENCIES)
            or min(args.inputs) < 1 or min(args.outputs) < 2 or max(args.outputs) > 512
            or min(args.batches) < 1 or max(args.inputs)+max(args.outputs) > 8192):
        parser.error('invalid workload/frequency/sampling envelope')
    result = asyncio.run(collect(args))
    print(json.dumps(dict(points=len(result['points']), coverage=result['coverage'])))
    return 0 if json.loads((args.out/'completion.json').read_text())['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
