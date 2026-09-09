"""Small PDB-only TP2 batch4/8 observations; never edits serving profiles."""
import argparse
import asyncio
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
HELPER = ROOT.parent / 'budget-profiling-v2-candidate'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check():
    m = read(ROOT / 'manifest.json')
    for name, digest in m['files'].items():
        require(sha(ROOT / name) == digest, 'package changed: ' + name)
    for name, digest in m['helper_files'].items():
        require(sha(HELPER / name) == digest, 'frozen profiler changed: ' + name)


def protect():
    files = read(ROOT / 'protected-baselines.json')['files']
    require(len(files) == 91, 'B baseline preservation set differs')
    for name, digest in files.items():
        require(sha(name) == digest, 'historical baseline changed: ' + name)
    return files


def validate_identity(actual, expected):
    for key, value in expected.items():
        require(actual.get(key) == value, 'live B identity differs: ' + key)


def validate_tp2_observation(raw, events):
    ranks = raw['drain']['transfers']
    require(len(ranks) == 2 and all(r.get('buffered_gpu_bytes') == 0 for r in ranks),
            'both actual TP2 rank drains required')
    for state in (raw['runtime_before'], raw['runtime_after_requests'], raw['runtime_drained']):
        require(len(state['scheduler_io']) == 1, 'actual TP2 scheduler owner cache observation required')
        require(state['acknowledged_generations'] == [state['generation']],
                'actual TP2 scheduler owner ACK required')
    ids = {r['request_id'] for r in raw['requests']}
    batch = raw['spec']['batch_size']
    steps = [e for e in events if e.get('prefill') == 0 and e.get('decode') == batch
             and set(e.get('request_ids', [])) == ids]
    require(len(steps) >= 64, 'requested concurrent decode batch was not sustained for64 steps')
    return dict(actual_batch=batch, full_batch_decode_steps=len(steps),
                earliest_step_s=min(e['started_s'] for e in steps),
                latest_step_s=max(e['finished_s'] for e in steps), tp2_rank_count=2)


def arguments():
    return SimpleNamespace(
        engine_config=ROOT.parent/'B32B-engine-v3-candidate-v2/engine-0.json',
        runtime_dir=ROOT.parent/'B32B-engine-v3-candidate-v2/runtime', out=ROOT/'results',
        port=33500, container='pdb-v2-nextv3b0', target_gpus=[0, 1],
        input_patterns=[[512]], output_pattern=[256], batches=[4, 8],
        frequencies=[2520, 1500], budgets=[8192], repeats=3, seed=0,
        arrival_offsets=[0.], require_preexisting_decode=False, arrival_lateness_limit_s=None)


async def run():
    sys.path.insert(0, str(HELPER))
    spec = importlib.util.spec_from_file_location('frozen_b_batch_profiler', HELPER/'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import evidence
    require(Path(evidence.__file__).resolve() == HELPER/'evidence.py', 'wrong frozen evidence helper')
    expected = read(ROOT/'expected-identity.json')

    class VerifiedProfiler(module.Profiler):
        async def identity(self):
            actual = await super().identity()
            validate_identity(actual, expected)
            return actual

        async def measure(self, point, path):
            raw = await super().measure(point, path)
            if not raw.get('error'):
                try:
                    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines() if line]
                    raw['batch_coverage_observation'] = validate_tp2_observation(raw, events)
                except Exception as exc:
                    raw['error'] = repr(exc)
                module.write(path/'raw.json', raw)
            return raw

    module.Profiler = VerifiedProfiler
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
    from ecopadg.metrics import clip_power_window
    require(not (ROOT/'operation.json').exists() and not arguments().out.exists(),
            'observation already attempted; retain originals')
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
    operation = dict(complete=False, energy_scope='all8 GPU boards including identity, warmup, controls, failure and native cleanup; inner batch energies are overlapping subwindows')
    try:
        sampler.start()
        deadline = time.monotonic() + 3
        while not sampler.samples:
            require(not sampler.error and time.monotonic() < deadline, 'outer sampler did not start')
            await asyncio.sleep(.01)
        operation['start_s'] = time.time()
        operation['complete'] = await module.run(arguments())
    except BaseException as exc:
        operation['error'] = repr(exc)
        raise
    finally:
        operation['end_s'] = time.time()
        deadline = time.monotonic() + 3
        while sampler.samples and sampler.samples[-1][0] < operation['end_s'] and not sampler.error and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        sampler.stop()
        operation['sampling_error'] = sampler.error
        operation['power_source'] = sampler.power_source
        with (ROOT/'operation-power.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['t_s'] + [f'gpu{i}_w' for i in range(8)])
            w.writerows([t, *values] for t, values in sampler.samples)
        write(ROOT/'operation-power-metadata.json', sampler.power_metadata)
        operation['power_coverage_valid'] = bool('start_s' in operation and not sampler.error
            and len(sampler.samples) >= 2 and sampler.samples[0][0] <= operation['start_s']
            and sampler.samples[-1][0] >= operation['end_s'])
        if operation['power_coverage_valid']:
            operation['energy_j'] = trapezoid_energy(clip_power_window(
                sampler.samples, operation['start_s'], operation['end_s'], pad_s=0))
        else:
            operation['observed_raw_energy_j'] = trapezoid_energy(sampler.samples)
        write(ROOT/'operation.json', operation)
    require(operation['power_coverage_valid'], 'outer energy does not cover entire operation')
    return operation['complete']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); args = p.parse_args()
    check()
    if args.check:
        print(json.dumps(dict(package_valid=True, planned_points=12, gpu_executed=False)))
        return
    from ecopadg.serving.campaign import node_lease
    with node_lease():
        before = protect()
        try:
            success = asyncio.run(run())
        finally:
            check(); after = protect()
            require(before == after, 'protected baseline set changed')
            write(ROOT/'baseline-preservation.json', dict(verified=True, files=after))
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
