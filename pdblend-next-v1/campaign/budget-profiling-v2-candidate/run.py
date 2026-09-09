"""PDB-only natural whole-batch microprofiles; explicit, small operating-point runs."""
import argparse
import asyncio
import csv
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import time
import uuid

import aiohttp

from evidence import ack, derive, digest, lookup, require, verify_restored


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def host_path(path, mounts):
    path = PurePosixPath(path)
    for mount in sorted(mounts, key=lambda m: len(m['Destination']), reverse=True):
        root = PurePosixPath(mount['Destination'])
        if mount.get('Type') == 'bind' and path.is_relative_to(root):
            return Path(mount['Source']) / path.relative_to(root)
    raise ValueError('model path has no explicit host bind mount: ' + str(path))


def model_hashes(roots):
    result = {}
    for root in roots:
        require(root.is_dir(), 'model/retained-weight directory missing: ' + str(root))
        files = sorted(path for path in root.rglob('*') if path.is_file()
                       and not path.name.endswith('.lock') and '__pycache__' not in path.parts)
        require(files, 'empty model identity root')
        for path in files:
            result[str(path)] = file_hash(path)
    require(any(Path(path).suffix in ('.safetensors', '.bin', '.pt') for path in result),
            'model byte identity contains no weight payload')
    return result


def verify_gpu_binding(inspection, provenance, target_gpus):
    requests = inspection['HostConfig'].get('DeviceRequests') or []
    devices = [device for request in requests for device in (request.get('DeviceIDs') or [])]
    accepted = {str(gpu) for gpu in target_gpus}
    env = dict(item.split('=', 1) for item in (inspection['Config'].get('Env') or []) if '=' in item)
    visible = provenance.get('cuda_visible_devices')
    all_devices = any(request.get('Count') == -1 for request in requests)
    bound_by_visible = (all_devices and visible == env.get('CUDA_VISIBLE_DEVICES')
                        and set(str(visible).split(',')) == accepted)
    require(set(devices) == accepted or bound_by_visible,
            'container GPU binding differs from measured target GPU indices')
    return visible


async def command(*args):
    process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    require(process.returncode == 0, 'command failed: ' + stderr.decode(errors='replace')[:500])
    return stdout.decode()


class Profiler:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(args.engine_config.read_text())
        require(self.config['port'] == args.port and self.config['tp'] == len(args.target_gpus),
                'target GPU count/port differs from engine config')
        self.issued = set()
        self.tasks = []
        self.http_log = (args.out / 'http.jsonl').open('x', buffering=1)

    async def http(self, route, payload=None):
        method = 'GET' if payload is None else 'POST'
        record = dict(route=route, method=method, request=payload, started_s=time.time())
        try:
            async with self.session.request(method, f'http://127.0.0.1:{self.args.port}{route}',
                    json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                text = await response.text()
                try:
                    body = json.loads(text)
                except ValueError:
                    body = text
                record.update(status=response.status, response=body, finished_s=time.time())
                require(response.status == 200, route + ' rejected: ' + text[:300])
                return body
        finally:
            self.http_log.write(json.dumps(record, allow_nan=False) + '\n')

    async def idle(self):
        deadline = time.monotonic() + 40
        while True:
            state = await self.http('/runtime')
            require(not state.get('error'), 'engine failed: ' + str(state.get('error')))
            if all(state.get(key) == 0 for key in ('active', 'running', 'waiting')):
                return state
            require(time.monotonic() < deadline, 'instance failed to become idle')
            await asyncio.sleep(.02)

    async def control(self, tokens, seqs):
        before = await self.idle()
        payload = dict(generation=before['generation'] + 1, role='mixed', mode='continuous',
            admit_prefill=True, admit_decode=True, scheduler_budget=dict(schema_version=1,
                max_num_batched_tokens=tokens, max_num_seqs=seqs))
        await self.http('/control', payload)
        state = await self.http('/runtime')
        ack(state, payload['generation'], tokens, seqs)
        return state

    async def identity(self):
        inspection = json.loads(await command('docker', 'inspect', self.args.container))[0]
        require(inspection['State']['Running'], 'profile container is not running')
        provenance = await self.http('/provenance')
        require(provenance.get('instance_id') == self.config['id'] and provenance.get('tp') == self.config['tp']
            and provenance.get('model') == self.config['model'], 'live model/config identity differs')
        require(provenance.get('source_files_at_import'), 'loaded serving source identity absent')
        source_paths = list(provenance['source_files_at_import'])
        script = '''import hashlib,importlib.util,json,pathlib,sys
paths=[pathlib.Path(p) for p in json.loads(sys.argv[1])]
root=pathlib.Path(importlib.util.find_spec("vllm").origin).parent
paths += [root/p for p in ("pdblend_budget.py","pdblend_runtime.py","pdblend_io.py","core/scheduler.py")]
print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))'''
        live_sources = json.loads(await command('docker', 'exec', self.args.container, 'python3', '-c',
                                               script, json.dumps(source_paths)))
        require(all(live_sources.get(path) == sha for path, sha in provenance['source_files_at_import'].items()),
                'loaded serving source changed on disk')
        changes = (await command('docker', 'diff', self.args.container)).splitlines()
        require(not any(line.split(' ', 1)[-1].endswith('.py') and '/vllm/' in line for line in changes),
                'vLLM Python source modified in running container layer')
        roots = [host_path(self.config['model'], inspection['Mounts'])]
        if self.config.get('retained_weights'):
            roots.append(host_path(self.config['retained_weights'], inspection['Mounts']))
        weights = await asyncio.to_thread(model_hashes, roots)
        visible = verify_gpu_binding(inspection, provenance, self.args.target_gpus)
        from ecopadg.measure import backends, power
        from ecopadg.serving import measurement
        import evidence
        running_ids = (await command('docker', 'ps', '-q')).split()
        running_containers = json.loads(await command('docker', 'inspect', *running_ids)) if running_ids else []
        residents = sorted([dict(id=item['Id'], image=item['Image'],
            cuda_visible_devices=next((value.split('=', 1)[1] for value in item['Config'].get('Env', [])
                                       if value.startswith('CUDA_VISIBLE_DEVICES=')), None),
            device_requests=item['HostConfig']['DeviceRequests']) for item in running_containers
            if item['HostConfig'].get('DeviceRequests')], key=lambda item: item['id'])
        nvml = self.hardware._nvml
        def text(value):
            return value.decode() if isinstance(value, bytes) else value
        gpu_identity = [dict(index=gpu, uuid=text(nvml.nvmlDeviceGetUUID(self.hardware._handle(gpu))),
            name=text(nvml.nvmlDeviceGetName(self.hardware._handle(gpu)))) for gpu in range(8)]
        return dict(container_id=inspection['Id'], engine_image=inspection['Image'],
            container_started_at=inspection['State']['StartedAt'], container_pid=inspection['State']['Pid'],
            cuda_visible_devices=visible, docker_device_requests=inspection['HostConfig'].get('DeviceRequests', []),
            model=provenance['model'], tp=provenance['tp'], dtype=provenance['dtype'],
            engine_version=provenance['engine_version'], max_model_len=provenance['max_model_len'],
            model_files_sha256=weights, engine_config_sha256=file_hash(self.args.engine_config),
            source_files_at_import=provenance['source_files_at_import'], live_vllm_and_serving_source_sha256=live_sources,
            measurement_source_sha256={str(Path(module.__file__)): file_hash(Path(module.__file__))
                for module in (backends, power, measurement, evidence)},
            node_gpu_identity=gpu_identity, gpu_container_residency=residents,
            driver_version=text(nvml.nvmlSystemGetDriverVersion()),
            tool_sha256=file_hash(Path(__file__)), target_gpus=self.args.target_gpus,
            identity_semantics='full model/retained-weight file hashes and live source hashes checked before and after campaign')

    async def request(self, input_length, output_length, gate=None, release=None, arrival_offset_s=0.):
        row = dict(request_id='pdb-profile-' + uuid.uuid4().hex, success=False,
            prompt_token_ids=([9707, 1879, 13] * (input_length // 3 + 1))[:input_length],
            output_token_ids=[], token_received_s=[], stream_events=[], requested_output_tokens=output_length,
            token_timing_semantics='client epoch receipt; token IDs sharing a frame share its timestamp')
        self.issued.add(row['request_id'])
        if gate is not None:
            await gate.wait()
            require(release is not None, 'timed request lacks batch release clock')
            row['planned_arrival_s'] = release['epoch_s'] + arrival_offset_s
            remaining = release['monotonic_s'] + arrival_offset_s - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
        row['dispatch_s'] = time.time()
        row.setdefault('planned_arrival_s', row['dispatch_s'])
        row['arrival_offset_s'] = arrival_offset_s
        row['dispatch_delay_s'] = row['dispatch_s'] - row['planned_arrival_s']
        body = dict(prompt=row['prompt_token_ids'], max_tokens=output_length, temperature=0,
                    top_p=1, seed=self.args.seed, ignore_eos=True, stream=True)
        try:
            async with self.session.post(f'http://127.0.0.1:{self.args.port}/v1/completions',
                    json=body, headers={'X-Request-Id': row['request_id']},
                    timeout=aiohttp.ClientTimeout(total=120)) as response:
                row['http_status'] = response.status
                require(response.status == 200, 'completion failed: ' + await response.text()
                        if response.status != 200 else '')
                async for line in response.content:
                    line = line.decode().strip()
                    if not line.startswith('data: '):
                        continue
                    data, received = line[6:], time.time()
                    if data == '[DONE]':
                        row['done_marker'] = True
                        break
                    event = json.loads(data)
                    row['stream_events'].append(dict(received_s=received, event=event))
                    require(not event.get('error'), 'stream failed: ' + str(event.get('error')))
                    ids = event.get('token_ids', [])
                    require(all(type(token) is int for token in ids), 'noninteger token identity')
                    row['output_token_ids'].extend(ids)
                    row['token_received_s'].extend([received] * len(ids))
                    require(event.get('token_index') == len(row['output_token_ids']), 'token index gap/duplicate')
                    if event.get('usage'):
                        row['usage'] = event['usage']
                row['stream_end_s'] = time.time()
                row['success'] = bool(row.get('done_marker'))
        except BaseException as exc:
            row.update(error=repr(exc), stream_end_s=time.time())
        return row

    async def measure(self, spec, destination):
        from ecopadg.measure.power import PowerSampler
        destination.mkdir()
        raw = dict(schema_version=1, system='pdblend', profile_kind='whole_batch', spec=spec,
                   requests=[], energy_boundary='batch release through owner drain; excludes warmup/control/clock setup')
        sampler = PowerSampler(range(8), interval=.02, backend=self.hardware, sample_clocks=True)
        self.issued.clear(); self.tasks = []
        try:
            state = await self.control(spec['budget_tokens'], spec['max_num_seqs'])
            await self.clocks.set(self.args.target_gpus, spec['clock_command_mhz'], verify_rise=False)
            raw['warmup'] = await self.request(128, 64)
            require(raw['warmup'].get('success'), 'natural warmup failed')
            state = await self.idle()
            raw.update(generation=state['generation'], runtime_before=state)
            event_path = self.args.runtime_dir / (self.config['id'] + '.control.events.jsonl')
            offset = event_path.stat().st_size
            sampler.start()
            deadline = time.monotonic() + 2
            while not sampler.samples:
                require(not sampler.error and time.monotonic() < deadline, 'power sampler did not start')
                await asyncio.sleep(.005)
            gate = asyncio.Event()
            release = {}
            self.tasks = [asyncio.create_task(self.request(n, output, gate, release, offset))
                for n, output, offset in zip(spec['input_lengths'], spec['output_lengths'], spec['arrival_offsets_s'])]
            await asyncio.sleep(0)
            raw['measurement_start_s'] = release['epoch_s'] = time.time()
            release['monotonic_s'] = time.monotonic()
            gate.set()
            raw['requests'] = await asyncio.gather(*self.tasks)
            raw['client_end_s'] = max(row['stream_end_s'] for row in raw['requests'])
            raw['runtime_after_requests'] = await self.idle()
            raw['drain'] = await self.http('/drain', dict(expected_generation=raw['generation']))
            raw['drain_response_s'] = time.time()
            raw['runtime_drained'] = await self.http('/runtime')
            raw['measurement_end_s'] = time.time()
            deadline = time.monotonic() + 2
            while not sampler.samples or sampler.samples[-1][0] < raw['measurement_end_s']:
                require(not sampler.error and time.monotonic() < deadline, 'power does not cover drain tail')
                await asyncio.sleep(.005)
            with event_path.open('rb') as handle:
                handle.seek(offset)
                events = handle.read()
            (destination / 'events.jsonl').write_bytes(events)
        except BaseException as exc:
            raw['error'] = repr(exc)
            for request_id in self.issued:
                try:
                    await self.http('/cancel', dict(request_id=request_id))
                except Exception as error:
                    raw.setdefault('cleanup_errors', []).append(repr(error))
            if self.tasks:
                raw['requests'] = await asyncio.gather(*self.tasks)
            try:
                state = await self.idle()
                raw['failure_drain'] = await self.http('/drain', dict(expected_generation=state['generation']))
            except Exception as error:
                raw.setdefault('cleanup_errors', []).append(repr(error))
        finally:
            sampler.stop()
            raw.update(sampling_error=sampler.error, power_source=sampler.power_source)
            write(destination / 'raw.json', raw)
            with (destination / 'power.csv').open('w', newline='') as handle:
                writer = csv.writer(handle); writer.writerow(['t_s'] + [f'gpu{gpu}_w' for gpu in range(8)])
                writer.writerows([t, *values] for t, values in sampler.samples)
            write(destination / 'power-metadata.json', sampler.power_metadata)
            write(destination / 'clocks.json', sampler.frequency_samples)
            write(destination / 'utilization.json', sampler.utilization_samples)
        return raw


def read_point(path):
    raw = json.loads((path / 'raw.json').read_text())
    with (path / 'power.csv').open(newline='') as handle:
        power = [(float(row['t_s']), [float(row[f'gpu{gpu}_w']) for gpu in range(8)])
                 for row in csv.DictReader(handle)]
    metadata = json.loads((path / 'power-metadata.json').read_text())
    clocks = json.loads((path / 'clocks.json').read_text())
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines() if line]
    profile = derive(raw, power, metadata, clocks, events)
    profile['artifact_sha256'] = {name: file_hash(path / name) for name in
        ('raw.json', 'power.csv', 'power-metadata.json', 'clocks.json', 'events.jsonl', 'utilization.json')}
    return profile


def patterns(text):
    values = [int(value) for value in text.split(',')]
    require(values and all(value > 0 for value in values), 'nonempty positive token pattern required')
    return values


def offsets(text):
    import math
    values = [float(value) for value in text.split(',')]
    require(values and all(math.isfinite(value) and value >= 0 for value in values)
            and min(values) == 0., 'offsets must be finite, nonnegative, and start at zero')
    return values


async def run(args):
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.backend import ClockOwner
    args.out.mkdir(parents=True, exist_ok=False)
    profiler = Profiler(args)
    specs = []
    for pattern, batch, frequency, tokens, repeat in itertools.product(
            args.input_patterns, args.batches, args.frequencies, args.budgets, range(args.repeats)):
        inputs = [pattern[index % len(pattern)] for index in range(batch)]
        outputs = [args.output_pattern[index % len(args.output_pattern)] for index in range(batch)]
        require(len(args.arrival_offsets) in (1, batch), 'arrival offset count must be one or match batch size')
        arrival_offsets = args.arrival_offsets * batch if len(args.arrival_offsets) == 1 else args.arrival_offsets
        require(all(n + output <= profiler.config.get('max_model_len', 8192) for n, output in zip(inputs, outputs)),
                'input plus output exceeds unchanged context limit')
        specs.append(dict(batch_size=batch, input_lengths=inputs, output_lengths=outputs,
            clock_command_mhz=frequency, budget_tokens=tokens, max_num_seqs=32,
            target_gpus=args.target_gpus, tp=profiler.config['tp'], temperature=0, seed=args.seed,
            arrival_offsets_s=arrival_offsets, preexisting_decode_required=args.require_preexisting_decode,
            arrival_lateness_limit_s=args.arrival_lateness_limit_s,
            arrival_trigger='fixed_open_loop_offset; preexisting decode checked from tokens and owner steps'))
    report = dict(system='pdblend', profile_kind='whole_batch', complete=False, points=[], planned_specs=specs)
    write(args.out / 'campaign.json', report)
    before = after = initial = None
    try:
        profiler.hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        profiler.clocks = ClockOwner(profiler.hardware, args.target_gpus)
    except BaseException as exc:
        report['error'] = repr(exc)
        write(args.out / 'campaign.json', report)
        profiler.http_log.close()
        return False
    async with aiohttp.ClientSession(trust_env=False) as profiler.session:
        try:
            initial = await profiler.idle()
            before = await profiler.identity()
            write(args.out / 'identity-before.json', before)
            for index, spec in enumerate(specs):
                path = args.out / f'point-{index:04d}'
                raw = await profiler.measure(spec, path)
                report['points'].append(dict(path=str(path), error=raw.get('error'), spec=spec))
                write(args.out / 'campaign.json', report)
                if raw.get('error'):
                    break
        except BaseException as exc:
            report['error'] = repr(exc)
        finally:
            try:
                state = await profiler.idle()
                if initial:
                    payload = {key: initial[key] for key in ('role', 'mode', 'admit_prefill', 'admit_decode')}
                    payload.update(generation=state['generation'] + 1, scheduler_budget=dict(schema_version=1,
                        **initial['scheduler_budget_effective']))
                    await profiler.http('/control', payload)
                    report['restored_runtime'] = await profiler.http('/runtime')
                    verify_restored(report['restored_runtime'], initial, payload['generation'])
            except BaseException as exc:
                report['cleanup_error'] = repr(exc)
            try:
                await profiler.clocks.close()
                report['clock_locks_reset'] = True
            except BaseException as exc:
                report['clock_cleanup_error'] = repr(exc)
            try:
                after = await profiler.identity()
                write(args.out / 'identity-after.json', after)
            except BaseException as exc:
                report['identity_error'] = repr(exc)
            profiler.http_log.close()
    profiles = []
    for point in report['points']:
        path = Path(point['path']); raw = json.loads((path / 'raw.json').read_text())
        raw.update(identity_before=before, identity_after=after)
        write(path / 'raw.json', raw)
        try:
            require(not raw.get('error'), 'point execution failed')
            profile = read_point(path)
            write(path / 'profile.json', profile)
            point.update(valid=True, profile_sha256=file_hash(path / 'profile.json'))
            point['interference_valid'] = profile['interference_valid']
            profiles.append(str(path / 'profile.json'))
        except Exception as exc:
            point.update(valid=False, validation_error=repr(exc))
    report['complete'] = (len(report['points']) == len(specs) and all(point.get('valid') for point in report['points'])
        and not any(report.get(key) for key in ('error', 'cleanup_error', 'clock_cleanup_error', 'identity_error')))
    report['interference_complete'] = (report['complete'] and all(point.get('interference_valid')
        for point in report['points'])) if args.require_preexisting_decode else None
    write(args.out / 'catalog.json', dict(system='pdblend', profile_kind='whole_batch',
        complete=report['complete'], interference_complete=report['interference_complete'], profiles=profiles,
        planned_points=len(specs), measured_points=len(report['points'])))
    write(args.out / 'campaign.json', report)
    print(json.dumps(dict(complete=report['complete'], interference_complete=report['interference_complete'],
                         out=str(args.out), valid_profiles=len(profiles))))
    return report['complete'] and report['interference_complete'] is not False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    measure = commands.add_parser('run')
    for name in ('engine-config', 'runtime-dir', 'out'):
        measure.add_argument('--' + name, type=Path, required=True)
    measure.add_argument('--port', type=int, required=True)
    measure.add_argument('--container', required=True)
    measure.add_argument('--target-gpus', type=int, nargs='+', required=True)
    measure.add_argument('--input-patterns', type=patterns, nargs='+', default=[[128, 7168]])
    measure.add_argument('--output-pattern', type=patterns, default=[64, 512])
    measure.add_argument('--arrival-offsets', type=offsets, default=[0.])
    measure.add_argument('--require-preexisting-decode', action='store_true')
    measure.add_argument('--arrival-lateness-limit-s', type=float)
    measure.add_argument('--batches', type=int, nargs='+', choices=[1, 2, 4], default=[2])
    measure.add_argument('--frequencies', type=int, nargs='+', choices=[900, 1500, 2100, 2520], default=[1500])
    measure.add_argument('--budgets', type=int, nargs='+', choices=[8192, 2048, 1024], required=True)
    measure.add_argument('--seed', type=int, default=0)
    measure.add_argument('--repeats', type=int, default=1)
    validate = commands.add_parser('validate'); validate.add_argument('point', type=Path)
    find = commands.add_parser('lookup')
    find.add_argument('--catalog', type=Path, required=True)
    find.add_argument('--exact-key', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        import math
        require(args.repeats > 0 and args.target_gpus and len(set(args.target_gpus)) == len(args.target_gpus)
                and all(gpu in range(8) for gpu in args.target_gpus), 'invalid repeats/target GPU indices')
        require(args.arrival_lateness_limit_s is None or math.isfinite(args.arrival_lateness_limit_s)
                and args.arrival_lateness_limit_s >= 0, 'lateness limit must be finite and nonnegative')
        if args.require_preexisting_decode:
            require(max(args.arrival_offsets) > 0 and args.arrival_lateness_limit_s is not None,
                    'interference requires explicit positive offsets and a declared lateness limit')
        from ecopadg.serving.campaign import node_lease
        with node_lease():
            raise SystemExit(0 if asyncio.run(run(args)) else 1)
    if args.command == 'validate':
        actual = read_point(args.point)
        require(actual == json.loads((args.point / 'profile.json').read_text()), 'profile summary/artifact mismatch')
        print(json.dumps(actual, indent=2))
    else:
        catalog = json.loads(args.catalog.read_text())
        require(catalog.get('system') == 'pdblend' and catalog.get('profile_kind') == 'whole_batch', 'wrong catalog type')
        profiles = []
        for path in catalog['profiles']:
            actual = read_point(Path(path).parent)
            require(actual == json.loads(Path(path).read_text()), 'profile summary/artifact mismatch')
            profiles.append(actual)
        print(json.dumps(lookup(profiles, json.loads(args.exact_key.read_text())), indent=2))


if __name__ == '__main__':
    main()
