"""Measurement-only installation and per-invocation sampler ownership."""
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'measurement reference changed')
    return json.loads(Path(reference['path']).read_text())


def references(binding):
    host_path = str(Path(binding['host_release']) / 'manifest.json')
    host = dict(path=host_path, sha256=binding['files'][host_path])
    adapter = binding['isolated_power_adapter']
    need(binding['files'].get(adapter['path']) == adapter['sha256'], 'adapter manifest is not bound')
    for path, digest in checked(adapter)['files'].items():
        need(binding['files'].get(path) == digest and sha(path) == digest, 'adapter source is not bound')
    checked(host)
    return host, adapter


def install(root, host, adapter):
    manifest = checked(host)
    host_root = Path(host['path']).parent
    for name, digest in manifest['files'].items():
        need(sha(host_root / name) == digest, 'actual host source changed')
    definition = checked(adapter)
    for path, digest in definition['files'].items():
        need(sha(path) == digest, 'isolated measurement source changed')
    directory = Path(adapter['path']).parent
    sys.path.insert(0, str(directory))
    module = importlib.import_module('isolated_sampler')
    need(Path(module.__file__).resolve() == (directory / 'isolated_sampler.py').resolve(),
         'wrong isolated sampler import')
    module.install(root, host, adapter)
    sampler_type = module.IsolatedPowerSampler
    if not getattr(sampler_type, '_fixed100_raw_snapshot_hook', False):
        original_stop = sampler_type.stop

        def stop_with_raw(sampler):
            original_stop(sampler)
            if sampler._directory is None:
                return
            need(not sampler._thread.is_alive(), 'sampler raw snapshot requires worker and reader exit')
            value = dict(schema='isolated-sampler-terminal-raw-v1', power_source=sampler.power_source,
                         samples=sampler.samples, metadata=sampler.power_metadata,
                         utilization=sampler.utilization_samples, frequency=sampler.frequency_samples,
                         sampling_error=sampler.error)
            data = (json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n').encode()
            path = sampler._directory / 'final-raw.json'
            if path.exists():
                need(path.read_bytes() == data, 'terminal sampler raw changed after stop')
            else:
                with path.open('xb') as handle:
                    handle.write(data)

        sampler_type.stop = stop_with_raw
        sampler_type._fixed100_raw_snapshot_hook = True
    return module


def directories(root):
    root = Path(root)
    return {p for p in root.glob('sampler-*') if p.is_dir()}


def completed_artifacts(roots, host, adapter):
    artifacts = {}
    need(roots, 'no owned isolated sampler evidence')
    for directory in sorted(roots):
        spec = json.loads((directory / 'spec.json').read_text())
        launch = json.loads((directory / 'launch.json').read_text())
        receipt = json.loads((directory / 'receipt.json').read_text())
        need(spec['host_manifest'] == host and spec['adapter_manifest'] == adapter
             and spec['gpus'] == list(range(8)) and spec['interval'] == .02
             and spec['read_only'] is True, 'isolated sampler scope changed')
        need(launch['pid'] == receipt['pid'] and launch['read_only'] is True
             and receipt['complete'] is True and receipt['child_exited'] is True
             and receipt['reader_stopped'] is True and receipt['returncode'] == 0
             and receipt['error'] is None, 'isolated sampler did not finish cleanly')
        terminal = receipt['terminal']
        need(terminal['pid'] == receipt['pid'] and terminal['sampler_thread_stopped'] is True
             and terminal['error'] is None and receipt['raw_sample_count'] >= 2
             and terminal['emitted'] == receipt['raw_sample_count']
             and terminal['counts']['samples'] == receipt['raw_sample_count']
             and terminal['counts']['metadata'] == receipt['raw_sample_count'],
             'isolated sampler terminal is incomplete')
        raw = json.loads((directory / 'final-raw.json').read_text())
        need(raw['sampling_error'] is None and raw['power_source'].get('mode') == 'instant'
             and len(raw['samples']) == receipt['raw_sample_count']
             and len(raw['metadata']) == terminal['counts']['metadata']
             and len(raw['utilization']) == terminal['counts']['utilization']
             and len(raw['frequency']) == terminal['counts']['frequency'],
             'isolated sampler terminal raw is incomplete')
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                artifacts[str(path)] = sha(path)
    return artifacts


def sampler_references(roots):
    return [dict(directory=str(directory),
                 raw=dict(path=str(directory / 'final-raw.json'), sha256=sha(directory / 'final-raw.json')),
                 receipt=dict(path=str(directory / 'receipt.json'), sha256=sha(directory / 'receipt.json')),
                 spec=dict(path=str(directory / 'spec.json'), sha256=sha(directory / 'spec.json')),
                 launch=dict(path=str(directory / 'launch.json'), sha256=sha(directory / 'launch.json')))
            for directory in sorted(roots)]


async def prepared_primary(args, cell, adapter):
    """Start only the read-only observer before the unchanged cell/controller."""
    from ecopadg.measure.backends import PynvmlBackend
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    sampler = adapter.IsolatedPowerSampler(range(8), interval=.02, backend=hardware)
    original_factory = cell.PowerSampler
    need(original_factory is adapter.IsolatedPowerSampler, 'cell cached the old sampler')
    consumed = False

    def already_started():
        need(sampler._process is not None and sampler._process.poll() is None
             and not sampler.error and len(sampler.samples) >= 2,
             'prepared observer is no longer ready')

    def factory(gpus, interval=.02, backend=None, clock=time.time, sample_clocks=False):
        nonlocal consumed
        need(not consumed, 'primary observer may be consumed once only')
        need(list(gpus) == list(range(8)) and interval == .02 and clock is time.time
             and sample_clocks is False and backend is not None
             and backend.power_source == sampler.power_source,
             'prepared primary observer scope changed')
        already_started()
        consumed = True
        sampler.start = already_started
        return sampler

    try:
        sampler.start()
        await asyncio.to_thread(sampler.wait_ready)
        cell.PowerSampler = factory
        result = await cell.run_cell(args)
        need(consumed, 'cell did not consume its prepared observer')
        return result
    finally:
        cell.PowerSampler = original_factory
        await asyncio.to_thread(sampler.stop)
        need(not sampler.error and not sampler._thread.is_alive(),
             'prepared primary observer did not stop cleanly')
