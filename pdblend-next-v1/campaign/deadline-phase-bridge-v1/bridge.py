"""Run an already frozen PDB main queue, then its declared SLO-scale queue.

The existing runners own all GPU actions, deadlines, energy and native cleanup.
This process never retries a cell, changes a declaration, or controls hardware.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def scale_ready(root, status):
    require(not (root / 'STOP').exists(), 'STOP remains active; scale not launched')
    require(status.get('selected_phase') == 'main' and status.get('complete') is True
            and status.get('phase') in ('finished', 'stopped_by_deadline')
            and status.get('baseline_preservation_verified') is True,
            'main did not terminate cleanly; scale not launched')
    spec = read(root / 'runspec.json')
    refs = {r['reuse_main_cell_id']: r['trace_sha256'] for r in spec['cells']
            if r.get('phase') == 'scale'}
    require(len(refs) == 18, 'expected eighteen declared scale-one references')
    available = {}
    for path in (root / 'checkpoints' / 'main').glob('*.json'):
        record = read(path)
        require(record['cell_id'] not in available, 'duplicate main checkpoint')
        available[record['cell_id']] = record
    for cell_id, digest in refs.items():
        record = available.get(cell_id, {})
        require(record.get('phase') == 'main' and record.get('measurement_valid') is True
                and record.get('source_trace_sha256') == digest,
                'missing valid same-work scale-one reference: ' + cell_id)
    # The frozen scale runner checks the full immutable artifact/receipt chain.
    return sorted(refs)


def execute(root, output, expected_manifest):
    require(not output.exists(), 'bridge attempt exists; preserve it')
    require(hashlib.sha256((root / 'package-manifest.json').read_bytes()).hexdigest()
            == expected_manifest, 'wrong frozen package')
    output.mkdir(parents=True)
    state = dict(schema=1, started_s=time.time(), pid=os.getpid(), package=str(root),
                 package_manifest_sha256=expected_manifest, phase='starting', complete=False,
                 baseline_execution=False, automatic_retries=False)
    child = None
    interrupted = False

    def save():
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temp.replace(output / 'status.json')

    def stop(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    save()
    try:
        for phase, count in (('main', 60), ('scale', 36)):
            require(not interrupted and not (root / 'STOP').exists(), 'stop requested')
            require(hashlib.sha256((root / 'package-manifest.json').read_bytes()).hexdigest()
                    == expected_manifest, 'package identity changed')
            before = set((root / 'invocations').glob('*.json'))
            with (output / (phase + '.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(root / 'run.py'),
                    '--phase', 'run', '--part', phase, '--max-cells', str(count)],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT)
                state.update(phase=phase, child_pid=child.pid)
                save()
                code = child.wait()
            state[phase + '_exitcode'] = code
            require(code == 0 and not interrupted, phase + ' runner failed or interrupted')
            fresh = set((root / 'invocations').glob('*.json')) - before
            require(len(fresh) == 1, 'ambiguous phase invocation')
            invocation = fresh.pop()
            result = read(invocation)
            state[phase + '_invocation'] = str(invocation)
            state[phase + '_checkpointed'] = result.get('checkpointed_cells', 0)
            state[phase + '_execution_complete'] = result.get('selected_phase_execution_complete', False)
            if phase == 'main':
                state['scale_one_references'] = scale_ready(root, result)
            else:
                require(result.get('complete') is True and result.get('phase') in
                        ('finished', 'stopped_by_deadline') and
                        result.get('baseline_preservation_verified') is True,
                        'scale did not terminate cleanly')
            save()
        state.update(phase='finished', complete=True)
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            # The frozen runner owns cancellation and its bounded native cleanup.
            # Never kill it merely because this supervisor had a logging error.
            state['cleanup_child_exitcode'] = child.wait()
        state['finished_s'] = time.time()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    execute(args.package.resolve(), args.output.resolve(), args.manifest_sha256)
