"""Launch this campaign once, after both CPU and native readiness checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import slo_support as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', choices=('A', 'C'), required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    node = args.node
    expected = {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}[node]
    p.need(socket.gethostname() == expected, 'wrong physical host')
    node_dir = p.HERE / node
    ready_path = node_dir / 'pdb-ready.json'
    ready = p.read(ready_path)
    p.need(ready.get('complete') and ready['node'] == node, 'native environment is not ready')
    runtime_paths = ready.get('runtime_pythonpath', [])
    p.need(isinstance(runtime_paths, list) and runtime_paths, 'explicit runtime dependency path required')
    sys.path[:0] = runtime_paths
    child_env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=':'.join(runtime_paths))
    os.environ['PYTHONPATH'] = child_env['PYTHONPATH']
    proof = p.load(ready['qualification_validator'], 'slo_launch_native_verifier').verify(ready['qualification'])
    p.need(proof.get('passed') and proof.get('independently_recomputed') and proof['binding'] == ready['binding'],
           'fresh physical qualification did not replay')
    command_path = p.HERE / 'env/baseline' / ('command-' + node + '.json')
    command = p.read(command_path)
    p.need(command['node'] == node, 'foreign baseline transition')
    for filename, digest in command['files'].items():
        p.need(p.sha(filename) == digest, 'baseline package changed')
    lock = (node_dir / 'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    p.need(not (node_dir / 'run-001').exists() and not (node_dir / 'launch-001.json').exists(), 'already launched')
    argv = [sys.executable, '-B', str(p.HERE / 'node_run.py'), '--node', node,
            '--handoff', str(ready_path), '--baseline-command', str(command_path),
            '--out', str(node_dir / 'run-001'), '--run']
    result = dict(schema='slo-rate-node-launch-v1', node=node, expected_hostname=expected,
        prepared_s=time.time(), argv=argv, handoff=p.ref(ready_path), baseline_command=p.ref(command_path),
        cpu_source_files={str(f): p.sha(f) for f in p.HERE.glob('*.py')}, run_requested=args.run)
    if args.run:
        log_path = node_dir / 'node-run-001.log'
        with log_path.open('xb') as log:
            # Release only our queue lock; the child obtains it and every GPU
            # operation separately acquires the shared physical node lease.
            lock.close()
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=p.HERE, start_new_session=True,
                env=child_env)
            result.update(pid=process.pid, process_identity=p.process_identity(process.pid),
                          launched_s=time.time(), log=str(log_path))
        p.save(node_dir / 'launch-001.json', result)
    else:
        lock.close()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
