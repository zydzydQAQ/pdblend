"""Measured, lease-owned deployment on the first eligible host.

No GPU action occurs during import/preparation or without explicit ``run=True``.
Containers from earlier tasks are retained. New task containers have independent
names and are stopped, never removed, on failure or final restoration.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('slo90_deploy_adapter', HERE / 'runtime_adapter.py')
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
read, write, require = adapter.read, adapter.write, adapter.require
REPO = adapter.REPO
PDB_IMAGE = 'sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'
BASELINE_IMAGE = 'sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'
RETAINED = Path('/root/workspace/pdblend/new-results/campaigns/three-pool-v2/weights/aeabfaf47f4941e6ba56d5e20d27d055')
PARENT_BASELINE = REPO / 'campaign/AC-baseline-deployment-prepared-v1/A-resident/deployment.json'
PARENT_ORDINARY = REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py'
LOCK_PATH = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_lease(lease=None):
    if lease is None:
        require('PDBLEND_NODE_LOCK_FD' in os.environ, 'caller must pass the actual node lease')
        descriptor = int(os.environ['PDBLEND_NODE_LOCK_FD'])
    else:
        descriptor = lease if isinstance(lease, int) else lease.fileno()
    require(os.path.samefile('/proc/self/fd/' + str(descriptor), LOCK_PATH), 'foreign lease descriptor')
    info = Path('/proc/self/fdinfo/' + str(descriptor)).read_text()
    require(any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line for line in info.splitlines()),
            'descriptor does not already hold the exclusive node lease')
    return descriptor


def assets_needed(frozen_release):
    """CPU staging inventory. Retained-weight load journals are mutable outputs."""
    release = read(frozen_release) if isinstance(frozen_release, (str, Path)) else frozen_release
    model = Path(release.get('model_root', '/root/workspace/models')) / 'Qwen2.5-14B-Instruct'
    retained = Path(release.get('retained_weights', RETAINED))
    return dict(model_directory=str(model), retained_weights=str(retained),
        retained_files=[str(retained / name) for name in ('manifest.json', 'rank-0.json', 'rank-0.safetensors')],
        pdb_engine_source=str(Path(release.get('engine_source_release', REPO / 'releases/io-v3-runtime'))),
        baseline_engine_entry=str(Path(release.get('baseline_engine_entry',
            REPO / 'campaign/AC-baseline-deployment-v1/engines/A/engine.py'))),
        baseline_pythonpath=str(release.get('baseline_engine_pythonpath', '/root/workspace/pdblend/src')))


def freeze_assets(release):
    """Verify retained rank bytes once before capturing target-local identities."""
    assets = assets_needed(release)
    retained = Path(assets['retained_weights'])
    manifest = read(retained / 'manifest.json')
    require(manifest.get('complete') is True and manifest.get('tp') == 1 and
            len(manifest.get('ranks', [])) == 1, 'actual TP1 retained-weight manifest required')
    rank = manifest['ranks'][0]
    require(rank['file'] == 'rank-0.safetensors', 'unexpected retained rank identity')
    path = retained / rank['file']
    before = adapter.stat_identity(path)
    # An explicit verified target-local asset cache avoids hashing the same
    # 29.5 GB again when switching to the baseline stage on this host.
    cached = release.get('verified_large_inputs', {}).get(str(path))
    if cached and cached.get('stat') == before and cached.get('sha256') == rank['sha256']:
        actual_sha = cached['sha256']
    else:
        actual_sha = sha(path)
    require(actual_sha == rank['sha256'] and adapter.stat_identity(path) == before,
            'retained model bytes differ or changed while verified')
    files = {str(retained / name): sha(retained / name) for name in ('manifest.json', 'rank-0.json')}
    model = Path(assets['model_directory'])
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
        require((model / name).is_file(), 'model/tokenizer staging incomplete: ' + name)
        files[str(model / name)] = sha(model / name)
    # The retained loader hashes hf_config.to_json_string(), including its
    # normalized metadata, rather than raw config.json bytes. CPU preflight
    # binds the already-qualified raw model/manifest/rank bytes; the unchanged
    # native RetainedWeightLoader repeats its canonical identity and parameter
    # checks during actual startup. Do not substitute a different hash here.
    reference = release.get('required_inputs')
    require(reference and sha(reference['path']) == reference['sha256'],
            'qualified source model/retained input declaration required')
    required = read(reference['path'])['files']
    for input_path, digest in {**files, str(path): actual_sha}.items():
        proof = required.get(input_path)
        require(proof and proof['sha256'] == digest and proof['size'] == Path(input_path).stat().st_size,
                'actual model/retained bytes differ from qualified inputs: ' + input_path)
    require(rank.get('model_config_sha256') == manifest['model_config_sha256'] and
            read(retained / 'rank-0.json').get('model_config_sha256') == manifest['model_config_sha256'],
            'retained rank metadata canonical model identity differs')
    files[reference['path']] = reference['sha256']
    return assets, files, {str(path): dict(sha256=actual_sha, stat=before)}


def prepare_spec(stage, frozen_release, hostname, idle_proof, previous_binding=None):
    """Create new engine configs from the qualified 14B templates; CPU only."""
    require(stage in ('pdblend', 'baselines'), 'unknown deployment stage')
    release_path = Path(frozen_release).resolve() if isinstance(frozen_release, (str, Path)) else None
    release = read(release_path) if release_path else copy.deepcopy(frozen_release)
    out = Path(release['deployment_root']).resolve() / stage
    require(not out.exists(), 'fresh deployment spec directory required')
    assets, files, large_inputs = freeze_assets(release)
    host = Path(release['host_releases'][stage]).resolve()
    common = Path(release['common_dir']).resolve()
    adapter.checked_manifest(host)
    adapter.checked_manifest(common)
    idle_proof = Path(idle_proof).resolve()
    require(idle_proof.is_file(), 'explicit supervisor idle/terminal evidence required')
    source = Path(assets['pdb_engine_source'])
    baseline_entry = Path(assets['baseline_engine_entry'])
    baseline = read(PARENT_BASELINE)
    runtime = out / 'engine-runtime'
    name_prefix = release.get('container_prefix', 'slo90-14b')
    require(name_prefix.startswith('slo90-') and all(c.isalnum() or c in '-_' for c in name_prefix),
            'independent task container prefix required')
    instances = []
    for j in ([6, 7] if stage == 'pdblend' else range(8)):
        if stage == 'pdblend':
            original_path = REPO / f'campaign/A14B-engine-v3/engine-{j}.json'
            cfg = read(original_path)
            iid = f'nextv3a{j}'
            env = ['VLLM_HOST_IP=127.0.0.1', f'CUDA_VISIBLE_DEVICES={j}', 'NCCL_P2P_DISABLE=1',
                'NCCL_SHM_DISABLE=1', 'NCCL_IB_DISABLE=1', 'NCCL_CUMEM_ENABLE=0', 'NCCL_DEBUG=WARN',
                f'PYTHONPATH={source / "src"}', 'PDBLEND_ASYNC_IO=1', 'PDBLEND_ENGINE_TIMING=1',
                'PYTHONDONTWRITEBYTECODE=1', 'PYTHONUNBUFFERED=1']
            command = ['python3', '-m', 'ecopadg.serving.engine']
            source_files = {str(p): sha(p) for p in sorted((source / 'src/ecopadg/serving').glob('*.py'))}
            image, kind = PDB_IMAGE, 'v3'
        else:
            original = baseline['instances'][j]
            original_path = Path(original['config'])
            cfg = read(original_path)
            iid = f'base100ar{j}'
            env = [e for e in original['environment'] if not e.startswith('PYTHONPATH=')]
            env.append('PYTHONPATH=' + assets['baseline_pythonpath'])
            command = ['python3', str(baseline_entry)]
            source_files = {str(p): sha(p) for p in sorted(baseline_entry.parent.glob('*.py'))}
            image, kind = BASELINE_IMAGE, 'legacy_sync_put'
        require(cfg['id'] == iid and cfg['model'] == '/models/Qwen2.5-14B-Instruct' and cfg['tp'] == 1,
                'qualified engine template identity changed')
        cfg['runtime_dir'] = str(runtime)
        cfg['retained_weights'] = assets['retained_weights']
        config_path = out / 'engines' / (iid + '.json')
        write(config_path, cfg)
        command += ['--config', str(config_path)]
        files[str(original_path)] = sha(original_path)
        files[str(config_path)] = sha(config_path)
        files.update(source_files)
        instance = dict(id=iid, tp=1, gpus=[j], role='mixed', port=cfg['port'], kv_port=cfg['kv_port'],
            url='http://127.0.0.1:' + str(cfg['port']), config=str(config_path), engine_config=str(config_path),
            container_name=name_prefix + '-' + stage + '-' + iid, image=image, command=command,
            environment=env, mounts=[dict(Type='bind', Source='/root/workspace', Destination='/root/workspace', RW=True),
                dict(Type='bind', Source=str(Path(release.get('model_root', '/root/workspace/models')).resolve()),
                     Destination='/models', RW=False)],
            native_kind=kind, scheduler_cache_observed=stage == 'pdblend',
            expected_provenance=dict(instance_id=iid, model=cfg['model'], tp=1, dtype='bfloat16',
                max_model_len=8192, cuda_visible_devices=str(j), source_files_at_import=source_files))
        if stage == 'pdblend':
            instance.update(scheduler_cache_count=1, service_budget_tokens=2048, restore_budget_tokens=8192)
        instances.append(instance)
    for directory in (host, common):
        manifest = adapter.checked_manifest(directory)
        files.update({str(directory / name): digest for name, digest in manifest['files'].items()})
        files[str(directory / 'manifest.json')] = sha(directory / 'manifest.json')
    for path in (idle_proof, Path(__file__).resolve(), HERE / 'runtime_adapter.py',
                 *((release_path,) if release_path else ())):
        files[str(path)] = sha(path)
    if stage == 'pdblend':
        files[str(PARENT_ORDINARY)] = sha(PARENT_ORDINARY)
    if previous_binding is not None:
        previous_binding = str(Path(previous_binding).resolve())
        files[previous_binding] = sha(previous_binding)
    # The legacy observation script imports its host Python package separately
    # from the serving controller runtime; bind this independent source too.
    if stage == 'baselines':
        package = Path(assets['baseline_pythonpath'])
        require(package.is_dir(), 'baseline native Python source has not been staged')
        files.update({str(p): sha(p) for p in sorted(package.rglob('*.py'))})
    spec = dict(schema=1, protocol_id=adapter.PROTOCOL, model='14b', stage=stage, hostname=hostname,
        host_release=str(host), common_dir=str(common), runtime_dir=str(runtime),
        source_entry=str(baseline_entry if stage == 'baselines' else source / 'src/ecopadg/serving/engine.py'),
        instances=instances, files=files, large_inputs=large_inputs, cpu_staging_inputs=assets,
        previous_binding=previous_binding, idle_proof=reference(idle_proof),
        deployment_budget_s=720, cleanup_budget_s=120, campaign_deadline_s=None,
        output_correctness_verified=False, preserve_previous_containers=True, remove_containers=False)
    validate_spec(spec, check_host=False)
    write(out / 'spec.json', spec)
    return reference(out / 'spec.json')


def validate_spec(spec, *, check_host=True):
    require(spec['protocol_id'] == adapter.PROTOCOL and spec['model'] == '14b' and
            spec['stage'] in ('pdblend', 'baselines'), 'wrong experiment deployment scope')
    if check_host:
        require(spec['hostname'] == socket.gethostname(), 'deployment belongs to another host')
    expected = [('nextv3a6', [6]), ('nextv3a7', [7])] if spec['stage'] == 'pdblend' else [
        ('base100ar' + str(j), [j]) for j in range(8)]
    require([(i['id'], i['gpus']) for i in spec['instances']] == expected, 'declared deployment layout differs')
    image = PDB_IMAGE if spec['stage'] == 'pdblend' else BASELINE_IMAGE
    names = [i['container_name'] for i in spec['instances']]
    require(len(names) == len(set(names)) and all(n.startswith('slo90-') for n in names), 'unique owned names required')
    require(spec['deployment_budget_s'] == 720 and spec['cleanup_budget_s'] == 120 and
            spec['campaign_deadline_s'] is None and spec['remove_containers'] is False,
            'bounded deployment/cleanup and retained containers required')
    for path, digest in spec['files'].items():
        require(sha(path) == digest, 'frozen deployment source changed: ' + path)
    for path, item in spec['large_inputs'].items():
        require(adapter.stat_identity(path) == item['stat'], 'target-local large asset changed: ' + path)
    for instance in spec['instances']:
        cfg = read(instance['config'])
        require(instance['image'] == image and instance['tp'] == 1 and cfg['id'] == instance['id'] and
                cfg['model'] == '/models/Qwen2.5-14B-Instruct' and cfg['tp'] == 1 and
                cfg['max_model_len'] == 8192 and cfg['max_num_seqs'] == 32 and cfg['max_num_batched_tokens'] == 8192,
                'qualified engine geometry/model/image differs')
        require(instance['expected_provenance']['source_files_at_import'], 'actual import source freeze required')


def docker_start_arguments(instance):
    args = ['docker', 'run', '-d', '--name', instance['container_name'], '--label',
            'com.openai.slo90=' + adapter.PROTOCOL, '--gpus', 'all', '--network', 'host', '--ipc', 'host']
    for value in instance['environment']:
        args += ['-e', value]
    for mount in instance['mounts']:
        require(mount['Type'] == 'bind', 'only explicit original bind mounts are allowed')
        args += ['-v', mount['Source'] + ':' + mount['Destination'] + ('' if mount['RW'] else ':ro')]
    return args + [instance['image']] + instance['command']


async def command(argv, records, timeout=40):
    record = dict(argv=argv, started_s=time.time())
    records.append(record)
    process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    record['pid'] = process.pid
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
        record.update(returncode=process.returncode, output=output.decode(errors='replace'))
        require(process.returncode == 0, 'command failed: ' + record['output'][-2500:])
        return record['output']
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        record['finished_s'] = time.time()


async def inventory(records):
    names = (await command(['docker', 'ps', '-aq'], records)).split()
    return json.loads(await command(['docker', 'inspect', *names], records)) if names else []


def verify_inventory(previous, rows, target_instances):
    running = {c['Name'].lstrip('/'): c for c in rows if c['State']['Running']}
    expected = {i['container']['name']: i for i in (previous or {}).get('instances', [])}
    require(set(running) == set(expected), 'unexpected running container; idle ownership changed')
    for name, instance in expected.items():
        actual, bound = running[name], instance['container']
        require(actual['Id'] == bound['id'] and actual['Image'] == bound['image'] and
                actual['State']['StartedAt'] == bound['StartedAt'], 'predecessor process changed')
    existing_names = {c['Name'].lstrip('/') for c in rows}
    require(not existing_names & {i['container_name'] for i in target_instances},
            'new task container name already exists; never replace prior evidence')
    return list(running.values())


async def capture_previous(session, common, rows):
    """Read actual idle residents on the chosen host; never copy another host's binding."""
    instances = []
    source_files = {}
    for container in rows:
        if not container['State']['Running']:
            continue
        argv = container['Config']['Cmd']
        require('--config' in argv and argv.index('--config') + 1 < len(argv),
                'unrecognized running process; supervisor must resolve ownership')
        path = Path(argv[argv.index('--config') + 1]).resolve()
        require(path.is_relative_to('/root/workspace') and path.is_file(), 'actual predecessor config unavailable')
        cfg = read(path)
        env = dict(value.split('=', 1) for value in container['Config']['Env'] if '=' in value)
        visible = env.get('CUDA_VISIBLE_DEVICES', '')
        require(visible and all(x.isdigit() for x in visible.split(',')), 'predecessor GPU placement unobserved')
        instance = dict(id=cfg['id'], tp=cfg['tp'], gpus=[int(x) for x in visible.split(',')],
            role=cfg.get('role', 'mixed'), port=cfg['port'], kv_port=cfg['kv_port'],
            url='http://127.0.0.1:' + str(cfg['port']), engine_config=str(path))
        require(len(instance['gpus']) == instance['tp'], 'actual predecessor TP/GPU placement differs')
        provenance = await common.http(session, instance, '/provenance')
        require(provenance.get('instance_id') == cfg['id'] and provenance.get('model') == cfg['model']
                and provenance.get('tp') == cfg['tp'] and provenance.get('cuda_visible_devices') == visible,
                'actual predecessor model/config/TP differs')
        imported = provenance.get('source_files_at_import')
        require(isinstance(imported, dict) and imported, 'actual predecessor source provenance absent')
        for source, digest in imported.items():
            require(sha(source) == digest, 'predecessor imported source changed')
            source_files[source] = digest
        raw = await common.http(session, instance, '/runtime')
        kind = 'v3' if raw.get('transfer_send_counters_observed') is True else 'legacy_sync_put'
        caches = raw.get('scheduler_io') or []
        instance.update(native_kind=kind, scheduler_cache_observed=bool(kind == 'v3' and caches),
            container=dict(name=container['Name'].lstrip('/'), id=container['Id'], image=container['Image'],
                           StartedAt=container['State']['StartedAt']),
            provenance=provenance, host_pid=container['State']['Pid'])
        if instance['scheduler_cache_observed']:
            instance['scheduler_cache_count'] = len(caches)
        if kind == 'v3':
            instance['restore_budget_tokens'] = cfg['max_num_batched_tokens']
        # This is a read-only idle test, not a drain of an active predecessor.
        common.idle(raw, instance)
        require(raw.get('generation') == raw.get('acknowledged_generation'), 'predecessor ACK absent')
        source_files[str(path)] = sha(path)
        instances.append(instance)
    require(instances and len({i['id'] for i in instances}) == len(instances), 'no unique idle predecessors captured')
    return dict(schema=1, hostname=socket.gethostname(), instances=instances, files=source_files,
                purpose='actual idle pre-experiment residents retained for restoration', captured_s=time.time())


def free_gpu_snapshot(hardware):
    nvml = hardware._nvml
    rows = []
    for gpu in range(8):
        handle = hardware._handle(gpu)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
        rows.append(dict(gpu=gpu, used_bytes=memory.used, free_bytes=memory.free,
            compute_pids=[p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)],
            graphics_pids=[p.pid for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]))
    return dict(observed_s=time.time(), gpus=rows)


async def wait_free(hardware, timeout=25):
    until = time.monotonic() + timeout
    while True:
        observed = await asyncio.to_thread(free_gpu_snapshot, hardware)
        if all(r['used_bytes'] == 0 and not r['compute_pids'] and not r['graphics_pids'] for r in observed['gpus']):
            return observed
        require(time.monotonic() < until, 'the eight GPUs did not become empty; foreign process or residue')
        await asyncio.sleep(.2)


async def ready(session, instance, common, deadline):
    last = None
    while time.monotonic() < deadline:
        try:
            provenance = await common.http(session, instance, '/provenance', timeout=3)
            require(all(provenance.get(k) == v for k, v in instance['expected_provenance'].items()),
                    'actual engine import/model/TP provenance differs')
            require(type(provenance.get('pid')) is int and provenance['pid'] > 0, 'actual engine PID missing')
            raw = await common.http(session, instance, '/runtime', timeout=3)
            require(raw.get('id') == instance['id'] and not raw.get('error') and not raw.get('runtime_error'),
                    'actual initial owner is unhealthy')
            if raw.get('generation') == 0 and raw.get('acknowledged_generation') == -1:
                require(all(k in raw and not raw[k] for k in
                    ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations')),
                    'new engine already has work')
                await common.http(session, instance, '/control', dict(generation=1, role='mixed',
                    mode='continuous', admit_prefill=True, admit_decode=True))
                continue
            common.idle(raw, instance)
            require(raw.get('accepting') is True and raw['generation'] == raw.get('acknowledged_generation'),
                    'actual accepting owner ACK missing')
            return dict(provenance=provenance, runtime=raw)
        except Exception as exc:
            last = repr(exc)
            await asyncio.sleep(.2)
    raise TimeoutError('engine readiness failed: ' + str(last))


def actual_binding(spec, rows, observed):
    by_name = {r['Name'].lstrip('/'): r for r in rows}
    require({name for name, r in by_name.items() if r['State']['Running']} ==
            {i['container_name'] for i in spec['instances']}, 'unexpected running engine after deployment')
    instances = []
    for declared in spec['instances']:
        actual = by_name[declared['container_name']]
        require(actual['Image'] == declared['image'] and actual['Config']['Cmd'] == declared['command']
                and actual['State']['Running'] and actual['State']['Pid'] > 0, 'actual container execution differs')
        provenance = observed[declared['id']]['provenance']
        require(all(provenance.get(k) == v for k, v in declared['expected_provenance'].items()),
                'actual imported provenance differs')
        record = {key: copy.deepcopy(declared[key]) for key in
            ('id', 'tp', 'gpus', 'role', 'port', 'kv_port', 'url', 'native_kind', 'scheduler_cache_observed', 'engine_config')}
        for key in ('scheduler_cache_count', 'service_budget_tokens', 'restore_budget_tokens'):
            if key in declared:
                record[key] = declared[key]
        record.update(container=dict(name=declared['container_name'], id=actual['Id'], image=actual['Image'],
                                     StartedAt=actual['State']['StartedAt']),
                      host_pid=actual['State']['Pid'], provenance={key: provenance[key] for key in
                        (*declared['expected_provenance'], 'pid')})
        instances.append(record)
    return dict(schema=1, protocol_id=adapter.PROTOCOL, model='14b', system='pdblend' if spec['stage'] == 'pdblend' else 'mixed',
        hostname=spec['hostname'], host_release=spec['host_release'], executor=str(Path(spec['common_dir']) / 'run.py'),
        deadline_s=None, campaign_lifecycle=adapter.LIFECYCLE, instances=instances,
        files=copy.deepcopy(spec['files']), large_inputs=copy.deepcopy(spec['large_inputs']),
        configs={}, output_correctness_verified=False, correctness_gate_required_before_performance=True)


class Measurement:
    """All-eight operation energy, including error cleanup, kept separately."""
    def __init__(self, out, hardware, result):
        from ecopadg.measure.power import PowerSampler
        self.out, self.hardware, self.result = Path(out), hardware, result
        self.sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)

    async def start(self):
        from ecopadg.serving.measurement import power_evidence
        self.sampler.start()
        until = time.monotonic() + 5
        while len(self.sampler.samples) < 2:
            require(not self.sampler.error and time.monotonic() < until, 'all-eight power preflight failed')
            await asyncio.sleep(.02)
        require(power_evidence(self.sampler.samples, self.sampler.power_source,
            self.sampler.power_metadata)['power_source_verified'], 'actual NVML instant source unverified')
        self.result['operation_start_s'] = time.time()

    async def finish(self):
        from ecopadg.serving.measurement import save_raw, power_evidence
        from ecopadg.measure.power import trapezoid_energy
        from ecopadg.metrics import clip_power_window
        self.result['operation_end_s'] = time.time()
        try:
            await asyncio.sleep(.12)
            await asyncio.to_thread(self.sampler.stop)
            power = self.out / 'power'
            power.mkdir()
            save_raw(power, [], self.sampler.samples, self.sampler.utilization_samples,
                power_source=self.sampler.power_source, power_metadata=self.sampler.power_metadata)
            with (power / 'clocks.csv').open('x', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['t_s'] + [f'gpu{j}_sm_mhz' for j in range(8)])
                writer.writerows((t, *values) for t, values in self.sampler.frequency_samples)
            self.result['power_evidence'] = power_evidence(self.sampler.samples, self.sampler.power_source,
                                                         self.sampler.power_metadata)
            start, end = self.result.get('operation_start_s'), self.result['operation_end_s']
            samples = self.sampler.samples
            require(start is not None and samples[0][0] <= start < end <= samples[-1][0], 'full energy window not bracketed')
            clocks = self.sampler.frequency_samples
            require(clocks and clocks[0][0] <= start and clocks[-1][0] >= end and
                    all(len(values) == 8 and all(math.isfinite(x) and x > 0 for x in values) for _, values in clocks),
                    'full all-eight clock observation missing')
            self.result['all8_operation_energy_j'] = trapezoid_energy(clip_power_window(samples, start, end, pad_s=0))
            self.result['full_operation_energy_j'] = self.result['all8_operation_energy_j']
        except BaseException as exc:
            self.result['errors'].append('measurement: ' + repr(exc))
        self.result.update(measured_gpu_count=8, sampling_error=self.sampler.error,
            measurement_valid=bool(not self.result['errors'] and not self.sampler.error and
                self.result.get('power_evidence', {}).get('power_source_verified') and
                self.result.get('all8_operation_energy_j') is not None), finished_s=time.time(),
            energy_scope='separate all-eight deployment/qualification/restoration operation; do not add overlapping windows')


async def execute(spec_path, out, *, run=False, lease=None):
    """Deploy a stage after the supervisor's idle check, under its actual lease."""
    spec = read(spec_path)
    validate_spec(spec)
    if not run:
        return dict(cpu_only=True, hardware_actions=False, spec_valid=True)
    require_lease(lease)
    out = Path(out).resolve()
    require(not out.exists(), 'fresh deployment attempt output required')
    out.mkdir(parents=True)
    common = adapter.load_runtime(spec['host_release'], spec['common_dir'])
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.backend import ClockOwner
    import aiohttp
    previous = read(spec['previous_binding']) if spec.get('previous_binding') else None
    result = dict(started_s=time.time(), complete=False, measurement_valid=False, errors=[], commands=[],
        created=[], creation_intents=[], stopped=[], spec=reference(spec_path), original_running=[],
        original_binding=previous, hardware_actions=False, output_correctness_verified=False)
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    meter = Measurement(out, hardware, result)
    mutation_started = False
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            before = await inventory(result['commands'])
            if previous is None and any(c['State']['Running'] for c in before):
                previous = await capture_previous(session, common, before)
                result['original_binding'] = previous
                write(out / 'captured-previous-binding.json', previous)
            result['original_running'] = verify_inventory(previous, before, spec['instances'])
            write(out / 'containers.before.json', before)
            if previous:
                await common.identity(session, previous)
            else:
                result['gpus_before'] = await wait_free(hardware, timeout=1)
            await meter.start()
            mutation_started = result['hardware_actions'] = True
            deadline = time.monotonic() + spec['deployment_budget_s']
            if previous:
                restored = await asyncio.wait_for(asyncio.gather(
                    *(common.restore(session, i) for i in previous['instances']), return_exceptions=True),
                    min(120, max(.01, deadline - time.monotonic())))
                result['previous_native_restore'] = [dict(error=repr(r)) if isinstance(r, BaseException) else r for r in restored]
                require(all(isinstance(r, dict) and r.get('complete') for r in restored), 'previous native cleanup failed')
                for instance in previous['instances']:
                    await command(['docker', 'stop', '--time', '30', instance['container']['id']], result['commands'])
                    result['stopped'].append(instance['container']['id'])
            result['gpus_after_stop'] = await wait_free(hardware)
            # The new 14B engines may use the same ports as the now-stopped
            # predecessors. Probe only after native cleanup and stop, and keep
            # failed setup energy in this operation's measured window.
            for instance in spec['instances']:
                for port in (instance['port'], *range(instance['kv_port'], instance['kv_port'] + 2 * instance['tp'])):
                    with socket.socket() as probe:
                        probe.bind(('127.0.0.1', port))
            Path(spec['runtime_dir']).mkdir(parents=True, exist_ok=False)
            for instance in spec['instances']:
                require(time.monotonic() < deadline, 'deployment budget exhausted')
                intent = dict(id=instance['id'], name=instance['container_name'], issued_s=time.time())
                result['creation_intents'].append(intent)
                write(out / 'creation-intents' / (instance['id'] + '.json'), intent)
                cid = (await command(docker_start_arguments(instance), result['commands'],
                                     timeout=max(.01, min(35, deadline - time.monotonic())))).strip()
                result['created'].append(dict(id=instance['id'], name=instance['container_name'], container_id=cid))
            observed_rows = await asyncio.wait_for(asyncio.gather(
                *(ready(session, i, common, deadline) for i in spec['instances'])),
                max(.01, deadline - time.monotonic()))
            observed = dict(zip((i['id'] for i in spec['instances']), observed_rows))
            result['startup'] = observed
            after = await inventory(result['commands'])
            write(out / 'containers.after.json', after)
            base = actual_binding(spec, after, observed)
            result['new_provenance'] = {i['id']: i['provenance'] for i in base['instances']}
            native_rows = await asyncio.wait_for(asyncio.gather(
                *(common.restore(session, i) for i in base['instances']), return_exceptions=True),
                max(.01, min(120, deadline - time.monotonic())))
            result['new_native_restore'] = {i['id']: dict(complete=False, error=repr(value))
                if isinstance(value, BaseException) else value for i, value in zip(base['instances'], native_rows)}
            require(all(v.get('complete') for v in result['new_native_restore'].values()), 'new native cleanup failed')
            await common.identity(session, base)
            write(out / 'binding-base.json', base)
            result['binding_base'] = reference(out / 'binding-base.json')
            result['complete'] = True
    except BaseException as exc:
        result['errors'].append(repr(exc))
    finally:
        if mutation_started and not result['complete']:
            cleanup_end = time.monotonic() + spec['cleanup_budget_s']
            # Only inspect names for which this operation persisted a creation
            # intent. Never stop a predecessor or an unrelated new claimant.
            for intent in result['creation_intents']:
                try:
                    inspected = json.loads(await command(['docker', 'inspect', intent['name']], result['commands'], timeout=5))[0]
                    require(inspected.get('Config', {}).get('Labels', {}).get('com.openai.slo90') == adapter.PROTOCOL,
                            'creation name was taken by a foreign container')
                    await command(['docker', 'stop', '--time', '10', inspected['Id']], result['commands'],
                                  timeout=max(.01, min(20, cleanup_end - time.monotonic())))
                except BaseException as exc:
                    result['errors'].append('stop owned creation: ' + repr(exc))
        if mutation_started:
            try:
                clocks = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
                await asyncio.wait_for(clocks.close(), 10)
                result['clock_restore_complete'] = True
            except BaseException as exc:
                result['errors'].append('clock cleanup: ' + repr(exc))
        await meter.finish()
        result['measurement_valid'] = bool(result['measurement_valid'] and result['complete'])
        write(out / 'deployment-receipt.json', result)
    require(result['complete'] and result['measurement_valid'], 'deployment failure retained: ' + str(result['errors']))
    return result


async def measured_ordinary(common, binding, out, *, lease=None):
    """Run the original deterministic PDB gate with full all-eight metering."""
    require_lease(lease)
    out = Path(out).resolve()
    require(not out.exists(), 'fresh PDB correctness output required')
    out.mkdir(parents=True)
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.backend import ClockOwner
    import aiohttp
    original = load(PARENT_ORDINARY, 'slo90_original_ordinary')
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    result = dict(passed=False, complete=False, errors=[], started_s=time.time(),
                  original_gate=reference(PARENT_ORDINARY))
    meter = Measurement(out, hardware, result)
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            common.validate_binding(binding)
            write(out / 'identity.before.json', await common.identity(session, binding))
            await meter.start()
            result['ordinary'] = await asyncio.wait_for(original.ordinary_gate(common, session, binding, out / 'ordinary'), 600)
            result['native_cleanup_complete'] = bool(result['ordinary'].get('passed') and
                all(isinstance(r, dict) and r.get('complete') for r in result['ordinary'].get('restoration', [])) and
                len(result['ordinary'].get('restoration', [])) == len(binding['instances']))
            require(result['native_cleanup_complete'], 'original PDB gate native cleanup incomplete')
            write(out / 'identity.after.json', await common.identity(session, binding))
            result['passed'] = result['complete'] = True
    except BaseException as exc:
        result['errors'].append(repr(exc))
    finally:
        try:
            clocks = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
            await asyncio.wait_for(clocks.close(), 10)
            result['clock_restore_complete'] = True
        except BaseException as exc:
            result['errors'].append('clock cleanup: ' + repr(exc))
        await meter.finish()
        result['passed'] = bool(result['passed'] and result['measurement_valid'] and not result['errors'])
        write(out / 'status.json', result)
    require(result['passed'], 'PDB ordinary correctness/measurement failed: ' + str(result['errors']))
    return result


def _stage_receipt(root, stage):
    choices = [root / stage / 'deployment-receipt.json', root / stage / 'deployment/deployment-receipt.json']
    existing = [p for p in choices if p.is_file()]
    if not existing:
        return None
    require(all(read(p) == read(existing[0]) for p in existing), 'stage receipt alias differs')
    return existing[0]


def verify_original_container(actual, original):
    for field in ('Id', 'Image', 'Name', 'Path', 'Args', 'Config', 'HostConfig'):
        require(actual.get(field) == original.get(field), 'retained original container changed: ' + field)
    normalized = lambda c: sorted(json.dumps(m, sort_keys=True) for m in c.get('Mounts', []))
    require(normalized(actual) == normalized(original), 'retained original mounts changed')


async def restore_original(release_path, out=None, *, run=False, lease=None):
    """Stop owned task containers and restore the first stage's original residents.

    Restoration is one separately metered operation. Failure outputs are kept;
    a caller must choose a new explicit ``out`` for a later recovery attempt.
    """
    release = read(release_path) if isinstance(release_path, (str, Path)) else copy.deepcopy(release_path)
    root = Path(release['deployment_root']).resolve()
    first_path = _stage_receipt(root, 'pdblend')
    if not run:
        return dict(cpu_only=True, hardware_actions=False, original_receipt=str(first_path) if first_path else None)
    require_lease(lease)
    out = Path(out).resolve() if out else root / 'restoration-001'
    require(not out.exists(), 'fresh restoration attempt directory required')
    out.mkdir(parents=True)
    if first_path is None:
        result = dict(complete=True, measurement_valid=True, restored=True, hardware_actions=False,
                      no_deployment_attempt=True, errors=[], finished_s=time.time())
        write(out / 'restoration-receipt.json', result)
        return result
    first = read(first_path)
    original_rows = first.get('original_running', [])
    original_binding = first.get('original_binding')
    require(bool(original_rows) == bool(original_binding), 'original ownership snapshot incomplete')
    specs, owned = {}, {}
    for stage in ('pdblend', 'baselines'):
        path = _stage_receipt(root, stage)
        if path is None:
            continue
        receipt = read(path)
        spec_path = Path(receipt['spec']['path'])
        require(sha(spec_path) == receipt['spec']['sha256'], 'deployment declaration changed before restoration')
        spec = read(spec_path)
        require(spec['hostname'] == socket.gethostname(), 'restoration is on another host')
        specs[stage] = spec
        created = {row['name']: row['container_id'] for row in receipt.get('created', [])}
        # Include persisted run intents in case docker created the owned
        # container but the response was interrupted before recording its ID.
        for declared in spec['instances']:
            if declared['container_name'] in {r['name'] for r in receipt.get('creation_intents', [])}:
                owned[declared['container_name']] = (declared, created.get(declared['container_name']))
    stage = 'baselines' if 'baselines' in specs else 'pdblend'
    common = adapter.load_runtime(release['host_releases'][stage], release['common_dir'])
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.backend import ClockOwner
    import aiohttp
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    result = dict(started_s=time.time(), complete=False, restored=False, measurement_valid=False,
        errors=[], commands=[], stopped=[], restarted=[], hardware_actions=False,
        original_receipt=reference(first_path), preserve_all_containers=True)
    meter = Measurement(out, hardware, result)
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            before = await inventory(result['commands'])
            by_name = {r['Name'].lstrip('/'): r for r in before}
            originals = {r['Name'].lstrip('/'): r for r in original_rows}
            require({name for name, c in by_name.items() if c['State']['Running']} <= set(owned) | set(originals),
                    'foreign running container appeared; restoration refuses to interfere')
            for name, old in originals.items():
                require(name in by_name, 'original retained container was removed externally')
                verify_original_container(by_name[name], old)
            for name, (declared, cid) in owned.items():
                if name not in by_name:
                    require(cid is None, 'owned created container was removed externally')
                    continue
                actual = by_name[name]
                require((cid is None or actual['Id'] == cid) and actual['Image'] == declared['image'] and
                        actual.get('Config', {}).get('Labels', {}).get('com.openai.slo90') == adapter.PROTOCOL,
                        'owned container identity or ownership label changed')
            write(out / 'containers.before.json', before)
            await meter.start()
            result['hardware_actions'] = True
            async def stop_owned(name, declared):
                actual = by_name.get(name)
                if actual is None or not actual['State']['Running']:
                    return
                try:
                    provenance = await common.http(session, declared, '/provenance')
                    require(all(provenance.get(k) == v for k, v in declared['expected_provenance'].items()),
                            'owned source/model changed before restoration')
                    restoration = await asyncio.wait_for(common.restore(session, declared), 65)
                    result.setdefault('owned_native_cleanup', {})[name] = restoration
                    require(restoration.get('complete'), 'owned task native cleanup failed')
                except BaseException as exc:
                    # An unhealthy engine must not prevent recovery of the
                    # previous idle environment. Its independently verified
                    # owned container is stopped while native failures remain
                    # visible and keep this receipt measurement-invalid.
                    result['errors'].append('owned native cleanup ' + name + ': ' + repr(exc))
                finally:
                    try:
                        await command(['docker', 'stop', '--time', '30', actual['Id']], result['commands'])
                        result['stopped'].append(actual['Id'])
                    except BaseException as exc:
                        result['errors'].append('owned container stop ' + name + ': ' + repr(exc))
            await asyncio.gather(*(stop_owned(name, declared) for name, (declared, _) in owned.items()))
            # Existing originals may already be running after a partial earlier
            # attempt. Do not stop them or insist their resident memory is zero.
            if not any(by_name[name]['State']['Running'] for name in originals):
                result['gpus_before_restore'] = await wait_free(hardware)
            deadline = time.monotonic() + 720
            # A retained container's mounted source may change independently
            # of its Docker metadata. Check the captured files before starting
            # it, after stopping our owned task containers.
            for path, digest in (original_binding or {}).get('files', {}).items():
                require(sha(path) == digest, 'original mounted source changed; refuse restart: ' + path)
            for name, old in originals.items():
                if not by_name[name]['State']['Running']:
                    try:
                        await command(['docker', 'start', old['Id']], result['commands'], timeout=30)
                        result['restarted'].append(old['Id'])
                    except BaseException as exc:
                        result['errors'].append('restart original ' + name + ': ' + repr(exc))
            restored_instances = []
            after = await inventory(result['commands'])
            after_by = {c['Name'].lstrip('/'): c for c in after}
            for original in (original_binding or {}).get('instances', []):
                try:
                    name = original['container']['name']
                    actual = after_by[name]
                    verify_original_container(actual, originals[name])
                    require(actual['State']['Running'] and actual['State']['Pid'] > 0, 'original engine did not restart')
                    expected = {k: v for k, v in original['provenance'].items() if k != 'pid'}
                    declared = dict(original, expected_provenance=expected)
                    observed = await ready(session, declared, common, deadline)
                    resumed = await asyncio.wait_for(common.restore(session, original),
                        max(.01, min(90, deadline - time.monotonic())))
                    require(resumed.get('complete'), 'restored original native ACK/cleanup failed')
                    fresh = copy.deepcopy(original)
                    fresh['container']['StartedAt'] = actual['State']['StartedAt']
                    fresh.update(provenance=observed['provenance'], host_pid=actual['State']['Pid'])
                    restored_instances.append(fresh)
                except BaseException as exc:
                    result['errors'].append('verify restored original ' + original['id'] + ': ' + repr(exc))
            final_rows = await inventory(result['commands'])
            require({c['Name'].lstrip('/') for c in final_rows if c['State']['Running']} == set(originals),
                    'final running set differs from original idle state')
            write(out / 'containers.after.json', final_rows)
            require(len(restored_instances) == len(originals), 'one or more restored originals lack complete identity/native proof')
            if original_binding:
                restored_binding = copy.deepcopy(original_binding)
                restored_binding['instances'] = restored_instances
                write(out / 'restored-original-binding.json', restored_binding)
                result['restored_binding'] = reference(out / 'restored-original-binding.json')
            else:
                result['gpus_final'] = await wait_free(hardware)
            result['complete'] = result['restored'] = True
    except BaseException as exc:
        result['errors'].append(repr(exc))
    finally:
        if result['hardware_actions']:
            try:
                clocks = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
                await asyncio.wait_for(clocks.close(), 10)
                result['clock_restore_complete'] = True
            except BaseException as exc:
                result['errors'].append('clock cleanup: ' + repr(exc))
        await meter.finish()
        result['measurement_valid'] = bool(result['measurement_valid'] and result['complete'])
        write(out / 'restoration-receipt.json', result)
    require(result['complete'] and result['measurement_valid'], 'restoration failure retained: ' + str(result['errors']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    deploy = sub.add_parser('deploy')
    deploy.add_argument('--spec', required=True, type=Path)
    deploy.add_argument('--out', required=True, type=Path)
    deploy.add_argument('--run', action='store_true')
    restore = sub.add_parser('restore')
    restore.add_argument('--release', required=True, type=Path)
    restore.add_argument('--out', type=Path)
    restore.add_argument('--run', action='store_true')
    args = parser.parse_args()
    result = asyncio.run(execute(args.spec, args.out, run=args.run) if args.action == 'deploy' else
                         restore_original(args.release, args.out, run=args.run))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
