"""Persistent B-only successor; one node owner, no retries after unknown failure."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_control_v2 as c
p, R = c.p, c.R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--declaration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = p.read(args.declaration)
    for path, digest in spec['files'].items():
        assert p.sha(path) == digest, path
    assert spec['schema'] == 'B32B-ascending-baseline-continuation-v1'
    assert spec['systems'] == ['mixed', 'distserve', 'dynamollm', 'ecoserve']
    assert spec['maximum_new_baseline_runs'] == 8 and spec['unknown_failure_stops_successor'] is True
    assert spec['pdb_release'] == p.ref(HERE / 'pdb-release-002/release.json')
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(schema='B32B-ascending-baseline-pipeline-status-v1', pid=os.getpid(), started_s=time.time(),
        declaration=p.ref(args.declaration), phase='waiting_PDB_same_rate_repeats', complete=False, node_lease_held=False,
        children=[], completed_systems=[], stop_requested=False)
    child = None
    def save():
        state['updated_s'] = time.time()
        p.save(args.out / 'status.json', state)
    def stop(*_):
        if not state['stop_requested']:
            state['stop_requested'] = True
            save()
            if child is not None and child.poll() is None:
                child.terminate()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    def run(name, argv):
        nonlocal child
        assert not state['stop_requested'] and not (HERE / 'STOP').exists()
        for path, digest in spec['files'].items():
            assert p.sha(path) == digest, path
        state['phase'] = name
        with (args.out / (name + '.log')).open('xb') as f:
            child = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)
            record = dict(phase=name, pid=child.pid, argv=argv, started_s=time.time())
            state['children'].append(record)
            save()
            code = child.wait()
            record.update(exitcode=code, finished_s=time.time())
            save()
        assert not state['stop_requested'] and code == 0, name + ' stopped/failed; no automatic retry'
        child = None
    save()
    try:
        while True:
            assert not state['stop_requested'] and not (HERE / 'STOP').exists()
            path = HERE / 'pdb-performance-001/status.json'
            if path.exists():
                s = p.read(path)
                if s.get('finished_s') and not c.r.alive(s['pid']):
                    assert s['complete'] and not s['failed'] and not s['node_lease_held']
                    break
            time.sleep(2)
        c.terminal()
        for action in ('restore', 'gate', 'qualify'):
            run(action, [sys.executable, '-B', str(HERE / 'baseline_control_v2.py'), action])
        run('freeze_baselines', [sys.executable, '-B', str(HERE / 'prepare_baselines_v2.py')])
        releases = p.read(HERE / 'baseline-releases-001.json')
        for system in spec['systems']:
            out = HERE / ('baseline-' + system + '-performance-001')
            run('measure_' + system, [sys.executable, '-B', str(HERE / 'run_cells_v1.py'),
                '--release', releases[system]['path'], '--out', str(out), '--run'])
            s = p.read(out / 'status.json')
            assert s['complete'] and not s['failed'] and not s['node_lease_held'] and not c.r.alive(s['pid'])
            assert len(s['observed_checkpoints']) == len(s['completed']) == 2
            state['completed_systems'].append(dict(system=system, status=p.ref(out / 'status.json'),
                                                 checkpoints=s['observed_checkpoints']))
            save()
        state.update(complete=True, phase='all_eight_baseline_runs_complete',
                     native_and_clock_cleanup_verified_each_cell=True)
    except BaseException as exc:
        state.update(error=repr(exc), phase='stopped_for_diagnosis', complete=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        state.update(finished_s=time.time(), node_lease_held=False)
        save()


if __name__ == '__main__':
    main()
