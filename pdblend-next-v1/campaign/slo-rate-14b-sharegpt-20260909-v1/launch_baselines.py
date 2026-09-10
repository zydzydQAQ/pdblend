"""Launch the reviewed baseline-only continuation after setup is terminal."""
import argparse
import fcntl
import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import resume_baselines as resume
import slo_support as p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', choices=('A', 'C'), required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    node = args.node
    p.need(socket.gethostname() == {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}[node],
           'wrong physical host')
    node_dir = p.HERE / node
    setup = node_dir / ('baseline-qualification-002' if node == 'A' else 'baseline-002')
    setup_status = p.read(setup / 'status.json')
    p.need(setup_status.get('complete') and setup_status.get('finished_s') and
           not setup_status.get('node_lease_held') and not p.active_owner(setup_status),
           'fresh baseline setup must have passed and exited')
    resume.children_terminal(setup_status)
    cfg = SimpleNamespace(node=node, predecessor=node_dir / 'run-001/status.json',
        baseline_ready=setup / 'ready.json', repair=setup / 'repair.json', out=node_dir / 'run-002',
        boundary=node_dir / 'pdb-boundary-002.json' if node == 'C' else None)
    lock = (node_dir / 'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    resume.validate(cfg)
    p.need(not cfg.out.exists() and not (node_dir / 'launch-002.json').exists(), 'already resumed')
    argv = [sys.executable, '-B', str(p.HERE / 'resume_baselines.py'), '--node', node,
        '--predecessor', str(cfg.predecessor), '--baseline-ready', str(cfg.baseline_ready),
        '--repair', str(cfg.repair), '--out', str(cfg.out), '--run']
    if cfg.boundary:
        argv += ['--boundary', str(cfg.boundary)]
    result = dict(schema='slo-rate-baseline-continuation-launch-v1', node=node,
        prepared_s=time.time(), argv=argv, qualification_terminal=p.ref(setup / 'status.json'),
        ready=p.ref(cfg.baseline_ready), repair=p.ref(cfg.repair), source=p.ref(__file__),
        resume_source=p.ref(resume.__file__), run_requested=args.run)
    if args.run:
        if node == 'C':
            pause = node_dir / 'STOP'
            archive = node_dir / 'pause-before-baselines.json'
            pause_record = p.read(pause)
            p.need(pause_record.get('do_not_interrupt_current_measurement') is True and
                   pause_record.get('node') == 'C' and not archive.exists(), 'foreign pause request')
            pause.rename(archive)
            result['completed_pause'] = p.ref(archive)
        else:
            p.need(not (node_dir / 'STOP').exists(), 'operator pause remains active')
        with (node_dir / 'node-run-002.log').open('xb') as log:
            lock.close()
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=p.HERE, start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            result.update(pid=process.pid, identity=p.process_identity(process.pid), launched_s=time.time())
        p.save(node_dir / 'launch-002.json', result)
    else:
        lock.close()
    import json
    print(json.dumps(result))


if __name__ == '__main__':
    main()
