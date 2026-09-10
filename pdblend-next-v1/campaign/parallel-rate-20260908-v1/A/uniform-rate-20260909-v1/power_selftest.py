"""Fresh new-A measurement-only qualification using the original isolated sampler."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
A = HERE.parent
ROOT = A.parent
HOST = ROOT / 'hosts/14b-capacity-p12'
METER = HERE / 'meter-runtime'
IDENTITY = A / 'ascending-newnode-20260909-v1/node-identity.json'
ADAPTER = HERE / 'isolated-power/manifest.json'
HOOKS = METER / 'sampler_hooks.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_identity(identity, hostname, rows):
    assert identity['node'] == 'Anew20260909'
    assert hostname == identity['actual_hostname'] == 'iZwz9274emxme9019d2sjgZ', 'wrong physical node'
    expected = [(r['index'], r['uuid']) for r in identity['GPUs']]
    assert len(rows) == 8 and [(r['index'], r['uuid']) for r in rows] == expected, 'wrong actual GPU identity'


def source_check():
    checked = {}
    for path, relative in [(HOST / 'manifest.json', HOST), (METER / 'manifest.json', None), (ADAPTER, None)]:
        manifest = read(path)
        for name, digest in manifest['files'].items():
            file = relative / name if relative else Path(name)
            assert sha(file) == digest, 'changed qualification source: ' + str(file)
            checked[str(file)] = digest
        checked[str(path)] = sha(path)
    for path in [IDENTITY, HOOKS, Path(__file__), METER / 'meter_evidence.py']:
        checked[str(path)] = sha(path)
    return checked


def actual_identity():
    raw = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True).stdout
    rows = [dict(index=int(line.split(',')[0]), uuid=line.split(',')[1].strip())
            for line in raw.strip().splitlines()]
    hostname = socket.gethostname()
    validate_identity(read(IDENTITY), hostname, rows)
    return dict(hostname=hostname, GPUs=rows, captured_s=time.time())


async def measure(out):
    from capacity_backend import TransitionMeter
    meter = await TransitionMeter(out / 'raw').start()
    try:
        await asyncio.sleep(8)
    finally:
        result = await meter.finish()
    assert result['measurement_valid'] is True
    assert result['gpu_indices'] == list(range(8)) and result['energy_j'] > 0
    assert result['duration_s'] >= 8 and result['power_observer_stopped']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    sources = source_check()
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, checked_files=len(sources),
                              GPU_work_started=False, physical_qualification_granted=False)))
        return
    assert args.out and not args.out.exists(), 'fresh explicit output directory required'
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh process must acquire own node lease'
    sys.path[:0] = [str(METER), str(HOST / 'src'), str(HOST), '/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.campaign import node_lease
    from meter_evidence import install
    with node_lease():
        identity = actual_identity()
        # A held node lease is required even though this probe only observes GPUs.
        args.out.mkdir(parents=True)
        save(args.out / 'source-manifest.json', sources)
        save(args.out / 'identity.json', identity)
        finish = install(args.out / 'isolated-samplers', ref(HOST / 'manifest.json'), ref(ADAPTER), ref(HOOKS))
        state = dict(schema='new-A-isolated-power-qualification-v1', node='Anew20260909', model='14b',
                     passed=False, hostname=identity['hostname'], identity=ref(args.out / 'identity.json'),
                     node_lease_held=True, read_only=True, hardware_writes=False, new_requests=0,
                     host_manifest=ref(HOST / 'manifest.json'), source_manifest=ref(args.out / 'source-manifest.json'),
                     source=ref(__file__), started_s=time.time())
        try:
            state['result'] = asyncio.run(measure(args.out))
        except BaseException as exc:
            state['error'] = repr(exc)
            raise
        finally:
            try:
                finish()
                terminal_path = args.out / 'isolated-observers-terminal.json'
                terminal = read(terminal_path)
                state['terminal'] = ref(terminal_path)
                state['passed'] = bool(not state.get('error') and state.get('result', {}).get('measurement_valid')
                                       and terminal['complete'])
            except BaseException as exc:
                state['cleanup_error'] = repr(exc)
            state['finished_s'] = time.time()
            save(args.out / 'validation.json', state)
        assert state['passed'], 'fresh isolated sampler qualification failed'
        print(json.dumps(dict(validation=ref(args.out / 'validation.json'), passed=True,
                              hostname=identity['hostname'], energy_j=state['result']['energy_j'])))


if __name__ == '__main__':
    main()
