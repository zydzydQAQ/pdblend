"""Read-only scale observer; publishes metadata only on a verified step change."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
C = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
        after = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        if not argv or after[0] == 'Z' or after[19] != stat[19]:
            return None  # Child may exit between the first stat and cmdline read.
        return dict(pid=pid, argv=argv, start_ticks=int(after[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def snapshot(status_path):
    status = read(status_path)
    assert status['model'] == '32b' and socket.gethostname() == 'iZwz9i5bte3xkpmcoes3t2Z'
    steps = status.get('steps', [])
    if not steps:
        return None
    step = steps[-1]
    child = process(step['pid'])
    if not step.get('complete'):
        if child is None:
            return None  # Parent has not yet reaped/published the exit.
        assert child['argv'] == step['argv'], 'actual child argv changed'
    else:
        assert child is None and isinstance(step.get('exitcode'), int), 'unverified terminal child'
    argv = step['argv']
    assert argv[2] == str(C / 'scale-only-continuation-B32B-v1/scale_driver.py') and '--run' in argv
    bp = Path(argv[argv.index('--binding') + 1])
    b = read(bp)
    assert b['hostname'] == socket.gethostname() and b['model'] == status['model']
    assert b['protocol_id'] == status['protocol_id'] and b['deadline_s'] == status['deadline_s']
    for path in b['configs'].values():
        assert b['files'][path] == sha(path), 'actual controller configuration changed'
    invs = []
    for path in (Path(b['output']) / 'invocations').glob('*.json'):
        inv = read(path)
        if inv.get('binding_sha256') == sha(bp) and inv.get('started_s', 0) >= step['started_s']:
            invs.append((path, inv))
    inv = min(invs, key=lambda x: x[1]['started_s']) if invs else None
    row = None
    if inv:
        assert inv[1]['phase'] == 'scale' and inv[1]['system'] == b['system']
        rows = read(status['source_manifest'])['cells']
        row = next((r for r in rows if r['cell_id'] == inv[1].get('current_cell')), None)
        if row:
            assert row['phase'] == 'scale' and row['system'] == b['system']
    if read(status_path) != status:
        return None
    return status, step, child, bp, b, inv, row


def publish(status_path, result):
    s, step, child, bp, b, inv, row = result
    index = C / 'current-experiment.json'
    previous = read(index)
    history = C / 'current-experiment-history' / (str(time.time_ns()) + '-model-scale')
    history.mkdir(parents=True)
    for name in ('current-experiment.json', 'CURRENT_EXPERIMENT.md'):
        p = C / name
        if p.exists():
            (history / name).write_bytes(p.read_bytes())
    if child:
        assert process(child['pid']) == child, 'child changed before publication'
    value = dict(previous)
    strategy = read(next(iter(b['configs'].values())))['strategy']
    value.update(schema=5, written_s=time.time(), hostname=b['hostname'], model=b['model'],
                 active_phase='scale', active_system=b['system'], scale_execution_allowed=True,
                 implementation_variant=strategy,
                 scale_release_scope='B32B actual150; explicit native trajectory qualification',
                 queue_pid=step['pid'], queue_running=child is not None,
                 queue_exitcode=step.get('exitcode'), queue_actual_argv=step['argv'],
                 queue_proc_start_ticks=None if child is None else child['start_ticks'],
                 supervisor_status=str(status_path), supervisor_pid=s['pid'],
                 binding=dict(path=str(bp), sha256=sha(bp)), host_release=b['host_release'],
                 actual_controller_configs=b['configs'], selected_datasets=step['argv'][step['argv'].index('--dataset')+1::2],
                 historical_index=str(history), current_queue_error=s.get('error'),
                 workload_manifest=dict(path=s['source_manifest'], sha256=s['source_sha256']),
                 execution_manifest=b.get('execution_manifest'), active_cell=row,
                 producer_invocation=None if inv is None else dict(path=str(inv[0]), sha256=sha(inv[0])),
                 observation_scope='Actual child process and source read; no synthetic telemetry heartbeat',
                 baseline_policy='Original main policy, original scale rows, seed701/100s, true B150 before scale')
    for key in ('queue_part', 'execution_subset_main_cells', 'original_main_cells_retained',
                'bridge', 'producer_observation', 'controller_host_gpu_validated'):
        value.pop(key, None)
    write(index, value)
    (C / 'CURRENT_EXPERIMENT.md').write_text(
        f'{b["model"]} {b["system"]} SLO scale. Queue running: {child is not None}.\n\n'
        'B32B completed150 main points; scale follows its qualified model release.\n'
        'Seed701,100s arrivals, original0.5/2 scales, all8GPU energy and120s request/drain.\n\n'
        f'Actual binding: {bp}\nActual supervisor: {status_path}\n'
        f'Current cell: {None if row is None else row["cell_id"]}\n')
    write(history / 'publication.json', dict(observer_pid=os.getpid(), status=str(status_path),
                                            actual_step=step, index_sha256=sha(index)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    last = None
    with (C / 'scale-index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = snapshot(args.status)
            if result:
                s, step, child, bp, b, inv, row = result
                signature = (step['pid'], step.get('complete'), None if row is None else row['cell_id'],
                             s.get('complete'), s.get('phase'), s.get('error'))
                if signature != last:
                    publish(args.status, result)
                    last = signature
                if (s.get('complete') or s.get('finished_s')) and process(s['pid']) is None:
                    return
            time.sleep(5)


if __name__ == '__main__':
    main()
