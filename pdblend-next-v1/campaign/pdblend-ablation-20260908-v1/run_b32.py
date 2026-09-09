"""Operational B watcher; waits for a deeply verified qualified baseline192 gate."""
import argparse
import asyncio
import fcntl
from pathlib import Path

import run as shared_run
import readiness_b32

ROOT = Path(__file__).resolve().parent


def verify_package(root=ROOT):
    path = root / 'execution-manifest-b32.json'
    shared_run.e.require(path.is_file(), 'B execution manifest is required')
    value = shared_run.e.read(path)
    shared_run.e.require(value.get('model') == '32b' and value.get('schema') == 1,
                         'wrong B execution manifest')
    mandatory = {'run.py', 'execution.py', 'readiness.py', 'run_b32.py', 'readiness_b32.py',
                 'declarations/core.cells.json', 'declarations/manifest.json'}
    shared_run.e.require(mandatory <= set(value['files']), 'B execution manifest omitted a required source')
    for name, digest in value['files'].items():
        file = Path(name) if Path(name).is_absolute() else root / name
        shared_run.e.require(shared_run.e.sha(file) == digest, 'B execution source changed: ' + str(file))
    for name, digest in value['dependencies'].items():
        shared_run.e.require(shared_run.e.sha(name) == digest, 'B serving dependency changed: ' + name)
    shared_run.e.require(not any('/scale-only-continuation-v3/' in name
        for key in ('dependencies', 'gate_dependencies') for name in value[key]),
        'B manifest must not depend on the A/C scale contract')
    # Baseline verification sources may be staged while this watcher is pending.
    # Their exact hashes are mandatory when an actual publication appears.
    return value


def install():
    shared_run.verify_package = verify_package
    shared_run.readiness.inspect_readiness = readiness_b32.inspect_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('32b',), default='32b')
    parser.add_argument('--declarations', type=Path, default=ROOT / 'declarations')
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-s', type=float, default=30)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    shared_run.e.require(5 <= args.poll_s <= 60, 'poll interval must be 5..60 seconds')
    install()
    with (ROOT / 'watcher-32b.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(shared_run.supervise(args))


if __name__ == '__main__':
    main()
