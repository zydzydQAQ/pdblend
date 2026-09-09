"""Wait without a lease, then freeze and execute only the predeclared final P8 cells."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from run import checked, need, read, ref, sha

HERE = Path(__file__).resolve().parent
A = HERE.parent
OUT = A / 'final-p8-continuation-001'
PREDECESSOR = A / 'p8-qualification900-dynamic-001'


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    need(not OUT.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh CPU-only waiter required')
    OUT.mkdir()
    state = dict(pid=os.getpid(), started_s=time.time(), phase='waiting900_without_lease',
                 complete=False, node_lease_held=False, steps=[])
    pins = {str(HERE / name): sha(HERE / name) for name in
            ('run.py', 'prepare.py', 'continue.py', 'qualification_contract.py', 'work-declaration.json', 'cpu-validation.json')}
    write(OUT / 'source-manifest.json', pins)

    def update(**value):
        state.update(value, updated_s=time.time()); write(OUT / 'status.json', state)

    def check():
        need(not (A / 'STOP_FINAL_P8').exists(), 'STOP blocks continuation')
        need(all(sha(p) == h for p, h in pins.items()), 'declared final candidate changed')

    def run(name, args):
        check(); update(phase=name)
        with (OUT / (name + '.log')).open('xb') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid); code = child.wait()
        state['steps'].append(dict(name=name, exitcode=code)); state.pop('child_pid', None)
        update(); need(code == 0, 'stage failed; no automatic retry: ' + name)

    update()
    try:
        while True:
            check(); predecessor = read(PREDECESSOR / 'status.json')
            if not alive(predecessor['pid']):
                break
            time.sleep(3)
        need(predecessor['complete'] and predecessor['phase'] == 'measured_complete'
             and predecessor['cleanup_complete'] and not predecessor.get('error')
             and not predecessor.get('cleanup_errors'), 'actual P8 candidate900 failed or not cleaned')
        run('qualify', [sys.executable, str(HERE / 'qualification_contract.py'),
                        '--out', str(OUT / 'qualification.json')])
        q = read(OUT / 'qualification.json')
        need(q['passed'] and q['actual_growth_and_return'], 'actual P8 lifecycle evidence required')
        run('freeze', [sys.executable, str(HERE / 'prepare.py')])
        release_path = A / 'final-p8-release-001/release.json'
        write(OUT / 'release-reference.json', ref(release_path))
        args = [sys.executable, '-u', str(HERE / 'run.py'), '--release', str(release_path),
                '--out', str(A / 'final-p8-001')]
        run('validate', args)
        run('measure', [*args, '--run'])
        final = read(A / 'final-p8-001/status.json')
        need(final['complete'] and not final.get('failed') and not final.get('error')
             and final['node_lease_held'] is False and not alive(final['pid']),
             'formal queue failed or remains active')
        update(phase='complete', complete=True, formal_status=ref(A / 'final-p8-001/status.json'))
    except BaseException as exc:
        update(phase='stopped_failure', error=repr(exc), complete=False)
        raise
    finally:
        update(finished_s=time.time())


if __name__ == '__main__':
    main()
