"""Supervise the frozen main and scale queues without controlling engines."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import os


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def verify_main_references(manifest, binding, system):
    rows = [r for r in manifest['cells'] if r['phase'] == 'main' and r['system'] == system]
    root = Path(binding['output'])
    if (root / 'STOP').exists(): raise RuntimeError('boundary stop requested')
    if len(rows) != 30: raise RuntimeError('expected thirty paired main cells')
    for row in rows:
        cp = read(root / 'checkpoints' / (row['cell_id'] + '.json'))
        if cp['row'] != row or cp.get('measurement_valid') is not True:
            raise RuntimeError('main checkpoint does not match declared experiment')
        if sha(cp['receipt']) != cp['receipt_sha256'] or read(cp['receipt'])['measurement_valid'] is not True:
            raise RuntimeError('main receipt changed or invalid')
        for path, digest in cp['artifacts'].items():
            if sha(path) != digest: raise RuntimeError('main artifact changed: ' + path)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--system', required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), started_s=time.time(), complete=False, phase='starting',
        system=a.system, manifest_sha256=sha(a.manifest), binding_sha256=sha(a.binding),
        runner_sha256=sha(a.runner), bridge_sha256=sha(__file__))
    def save():
        tmp = a.out / 'status.tmp'; tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(a.out / 'status.json')
    save()
    try:
        for phase, count in (('main', 30), ('scale', 18)):
            for path, key in ((a.manifest, 'manifest_sha256'), (a.binding, 'binding_sha256'), (a.runner, 'runner_sha256')):
                if sha(path) != state[key]: raise RuntimeError('supervised input changed')
            if phase == 'scale':
                state['verified_main_references'] = verify_main_references(read(a.manifest), read(a.binding), a.system)
            with (a.out / (phase + '.log')).open('xb') as log:
                child = subprocess.Popen([sys.executable, '-u', str(a.runner), '--manifest', str(a.manifest),
                    '--binding', str(a.binding), '--system', a.system, '--phase', phase,
                    '--max-cells', str(count), '--run'], stdout=log, stderr=subprocess.STDOUT)
                state.update(phase=phase, child_pid=child.pid); save()
                code = child.wait()
            state[phase + '_exitcode'] = code
            if code != 0: raise RuntimeError(phase + ' queue failed; its raw evidence is retained')
        state.update(phase='finished', complete=True)
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        state['finished_s'] = time.time(); save()


if __name__ == '__main__': main()
