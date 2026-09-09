"""Hold one node lease across the explicitly staged Gamma campaign commands."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
LOCK = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
stop = False

def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def main():
    global stop
    state = dict(pid=os.getpid(), hostname=socket.gethostname(), started_s=time.time(),
                 phase='acquiring', node_lease_held=False, completed=[], failed=[])
    ROOT.joinpath('queue').mkdir(exist_ok=True)
    ROOT.joinpath('queue-results').mkdir(exist_ok=True)
    def save(**values):
        state.update(values, updated_s=time.time())
        write(ROOT / 'owner-status.json', state)
    def request_stop(*args):
        global stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with LOCK.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(phase='preparing', node_lease_held=True)
        try:
            while not stop and not (ROOT / 'STOP_OWNER').exists():
                pending = [p for p in sorted((ROOT / 'queue').glob('*.json'))
                           if not (ROOT / 'queue-results' / p.name).exists()]
                if not pending:
                    if (ROOT / 'QUEUE_COMPLETE').exists():
                        save(phase='complete')
                        break
                    time.sleep(1)
                    continue
                job_path = pending[0]
                job = json.loads(job_path.read_text())
                assert isinstance(job['argv'], list) and all(isinstance(x, str) for x in job['argv'])
                assert not state['failed'], 'failed stage requires explicit owner restart'
                save(phase='running', current_job=job_path.name)
                started = time.time()
                environment = dict(os.environ, PDBLEND_NODE_LOCK_FD=str(lease.fileno()),
                                   PYTHONDONTWRITEBYTECODE='1')
                with (ROOT / 'queue-results' / (job_path.stem + '.log')).open('xb') as log:
                    child = subprocess.Popen(job['argv'], cwd=str(ROOT), env=environment,
                                             stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lease.fileno(),))
                    save(child_pid=child.pid)
                    while child.poll() is None:
                        # Finish the active bounded command before releasing its lease.
                        time.sleep(1)
                result = dict(job=job, pid=child.pid, started_s=started, finished_s=time.time(),
                              exitcode=child.returncode)
                write(ROOT / 'queue-results' / job_path.name, result)
                state.pop('child_pid', None)
                if child.returncode:
                    state['failed'].append(job_path.name)
                    save(phase='failed')
                    break
                state['completed'].append(job_path.name)
                save(phase='preparing', current_job=None)
        finally:
            save(node_lease_held=False, finished_s=time.time())

if __name__ == '__main__':
    main()
