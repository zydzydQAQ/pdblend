"""Durable CPU-only B priority successor; physical work belongs to a pinned child."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEADLINE = 1788872770.0400891


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fixed(reference):
    require(digest(reference['path']) == reference['sha256'], 'changed reference: ' + reference['path'])
    return read(reference['path'])


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def validate_handoff(path):
    ready = read(path)
    require(ready.get('schema') == 'B-measured-baseline-return-ready-v2' and ready.get('ready') is True,
            'measured baseline handoff required')
    require(not alive(ready['coordinator_pid']), 'original repeats owner must exit')
    for key in ('original_baseline_binding', 'fresh_restoration_binding', 'fresh_inventory',
                'measured_restore_status', 'prefix_proof', 'repeats_status'):
        fixed(ready[key])
    repeat = fixed(ready['repeats_status'])
    require(repeat.get('complete') and len(repeat.get('completed', [])) == 16
            and not repeat.get('failed'), 'all sixteen original repeats must finish')
    restore = fixed(ready['measured_restore_status'])
    require(restore.get('complete') and restore.get('clock_restore_complete')
            and not restore.get('errors') and not restore.get('sampling_error')
            and restore['power_evidence']['power_source_verified'], 'restoration invalid')
    return ready


def validate_spec(path):
    spec = read(path)
    require(spec.get('schema') == 'B-improvement-child-v1' and spec.get('approved') is True,
            'approved successor declaration required')
    require(spec.get('deadline_s') == DEADLINE and spec.get('model') == '32b', 'wrong deadline/model')
    require(spec.get('automatic_retries') is False and spec.get('acquires_fresh_node_lease') is True,
            'fresh lease and no automatic retries required')
    require(isinstance(spec.get('argv'), list) and all(isinstance(a, str) for a in spec['argv'])
            and len(spec['argv']) >= 3, 'explicit argv required')
    require(spec.get('files') and all(digest(p) == h for p, h in spec['files'].items()),
            'child source hash changed')
    require(spec['argv'][0] in ('/usr/bin/python3', sys.executable), 'explicit Python child required')
    scripts = [a for a in spec['argv'][1:] if a.endswith('.py')]
    require(scripts and scripts[0] in spec['files'], 'entry source must be pinned')
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=ROOT / 'approved-child.json')
    p.add_argument('--self-sha256', required=True)
    args = p.parse_args()
    require(digest(__file__) == args.self_sha256, 'queue source changed')
    require(not args.out.exists(), 'new queue output required')
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'queue must not inherit a node lease')
    args.out.mkdir(parents=True)
    state = dict(schema='B-improvement-priority-queue-v1', pid=os.getpid(), started_s=time.time(),
                 phase='waiting_original16_and_baseline_return', complete=False,
                 hardware_actions_in_parent=False, automatic_retries=False,
                 ready_path=str(args.ready), child_spec_path=str(args.spec),
                 old_tasks_blocked_until_improvement_terminal=True)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def update(**value):
        state.update(value, updated_s=time.time())
        write(args.out / 'status.json', state)
    def boundary():
        require(not stopped and not (ROOT / 'STOP').exists(), 'queue stopped at boundary')
        require(time.time() < DEADLINE, 'same-day deadline exhausted')
        require(digest(__file__) == args.self_sha256, 'queue source changed')
    with (ROOT / 'priority-watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        update()
        try:
            while not args.ready.exists() or alive(read(args.ready)['coordinator_pid']):
                boundary()
                predecessor = args.ready.parent / 'status.json'
                if predecessor.exists():
                    d = read(predecessor)
                    require(d.get('phase') != 'failed', 'original repeats/restoration failed')
                    q = d.get('repeats', {})
                    update(predecessor_phase=d.get('phase'), original_completed=len(q.get('completed', [])),
                           original_failed=len(q.get('failed', [])))
                time.sleep(3)
            boundary()
            ready = validate_handoff(args.ready)
            update(phase='waiting_approved_improvement_child', original_completed=16,
                   handoff=dict(path=str(args.ready), sha256=digest(args.ready)), baseline_ready=ready)
            while not args.spec.exists():
                boundary()
                time.sleep(3)
            boundary()
            spec = validate_spec(args.spec)
            validate_handoff(args.ready)
            write(args.out / 'accepted-child.json', dict(reference=dict(path=str(args.spec), sha256=digest(args.spec)),
                                                       specification=spec))
            with (args.out / 'child.log').open('xb') as log:
                child = subprocess.Popen(spec['argv'], stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                update(phase='improvement_child_running', child_pid=child.pid, child_argv=spec['argv'])
                while child.poll() is None:
                    # Child owns all physical work, deadline, and cleanup. A queue stop
                    # never signals a currently running measurement.
                    time.sleep(3)
                require(child.returncode == 0, 'improvement child failed; inspect retained evidence')
            update(phase='improvement_child_complete', complete=True, child_exitcode=child.returncode,
                   old_tasks_resume_requires_fresh_explicit_handoff=True)
        except BaseException as exc:
            update(phase='needs_attention', error=repr(exc))
            raise
        finally:
            update(finished_s=time.time())


if __name__ == '__main__':
    main()
