"""New one-attempt successor for eight unattempted B32B Alpaca 4.5 baselines.

Default mode reads and validates CPU evidence only. --run is deliberately
exclusive: any existing pipeline/partial phase requires explicit diagnosis.
"""
import argparse
import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from adapters import HERE, OLD, control, historical, p, source_contract
from cold_restore import helper, host_processes, cold_snapshot, restore_cold

SYSTEMS = ('mixed', 'distserve', 'dynamollm', 'ecoserve')
PIPELINE = HERE / 'pipeline-001'


def validate():
    manifest, boundary, _ = source_contract()
    target = p.read(control.ORIGINAL)
    assert p.sha(control.ORIGINAL) == '87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119'
    assert socket.gethostname() == target['hostname'] == 'iZwz9i5bte3xkpmcoes3t2Z'
    for instance in target['instances']:
        for path, digest in instance['provenance']['source_files_at_import'].items():
            assert p.sha(path) == digest, path
        cfg = p.read(instance['engine_config'])
        assert cfg['id'] == instance['id'] and cfg['tp'] == 2 and cfg['max_model_len'] == 8192
        assert p.sha(instance['engine_config']) == target['files'][instance['engine_config']]
    return dict(passed=True, cpu_only=True, source_files=len(manifest['files']),
                historical_declaration_files=len(p.checked(manifest['original_declaration'])['files']),
                historical_pdb_repeats_recomputed=2, boundary_rate_rps=boundary['cap_rate_rps'],
                remaining_baseline_cells=boundary['required_new_baseline_cell_ids'],
                original_attempt_preserved=True, host_pid_and_gpu_checks_deferred_to_run=True,
                pipeline_already_exists=PIPELINE.exists())


async def cold_stage():
    manifest, boundary, _ = source_contract()
    assert not (HERE / 'baseline-boundary-001.json').exists()
    p.save(HERE / 'baseline-boundary-001.json', boundary)
    target = copy.deepcopy(p.read(control.ORIGINAL))
    target.update(deadline_s=None, campaign_lifecycle='until_declared_complete_v1')
    target['files'].update(manifest['files'])
    target['files'][str(HERE / 'manifest.json')] = p.sha(HERE / 'manifest.json')
    common = helper.load_common(target['host_release'])
    from ecopadg.serving.campaign import node_lease
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    with node_lease():
        snapshot = cold_snapshot(target)
        await restore_cold(common, target, snapshot, control.RESTORE)


def stage_cold():
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await cold_stage()
    asyncio.run(controlled())


def fresh_outputs_required():
    paths = [PIPELINE, control.RESTORE, control.QUAL, HERE / 'baseline-boundary-001.json',
             HERE / 'baseline-releases-001.json']
    for system in SYSTEMS:
        paths.extend(HERE / ('baseline-' + system + '-' + kind + '-001') for kind in ('performance', 'release'))
    assert all(not path.exists() for path in paths), 'new attempt output already exists; preserve evidence and diagnose; never automatically retry'


def execute():
    validate()
    host_processes()
    # The owner lock prevents duplicate successors between GPU phases. Each
    # GPU phase also acquires the original global node-experiment.lock itself.
    with (HERE / 'successor-owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh_outputs_required()
        PIPELINE.mkdir()
        state = dict(schema='B32B-cold-resume-status-v1', pid=os.getpid(), started_s=time.time(),
                     startticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
                     phase='preflight', complete=False, children=[], completed_systems=[],
                     observed_checkpoints=[], stop_requested=False, node_lease_held=False,
                     manifest=p.ref(HERE / 'manifest.json'), predecessor=p.ref(OLD / 'baseline-pipeline-001/status.json'),
                     earlier_failed_successor=p.read(HERE / 'manifest.json')['failed_successor'])
        child = None

        def save():
            state['updated_s'] = time.time()
            p.save(PIPELINE / 'status.json', state)

        def stop(*_):
            if not state['stop_requested']:
                state['stop_requested'] = True
                save()
                if child is not None and child.poll() is None:
                    child.terminate()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)

        def run(phase, argv):
            nonlocal child
            assert not state['stop_requested']
            source_contract()
            state['phase'] = phase
            with (PIPELINE / (phase + '.log')).open('xb') as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                record = dict(phase=phase, pid=child.pid, argv=argv, started_s=time.time())
                state['children'].append(record)
                save()
                record.update(exitcode=child.wait(), finished_s=time.time())
                save()
            assert not state['stop_requested'] and record['exitcode'] == 0, phase + ' failed/stopped; no automatic retry'
            child = None

        save()
        try:
            # This CPU probe checks lock availability before creating any GPU work.
            target = p.read(control.ORIGINAL)
            helper.load_common(target['host_release'])
            from ecopadg.serving.campaign import node_lease
            assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
            with node_lease():
                p.save(PIPELINE / 'preflight.json', cold_snapshot(target))
            run('cold_restore', [sys.executable, '-B', str(Path(__file__).resolve()), '--run', '--cold-stage'])
            for action in ('gate', 'qualify'):
                run(action, [sys.executable, '-B', str(HERE / 'baseline_control_v2.py'), action, '--run'])
            run('freeze', [sys.executable, '-B', str(HERE / 'prepare_baselines_v2.py'), '--run'])
            releases = p.read(HERE / 'baseline-releases-001.json')
            for system in SYSTEMS:
                out = HERE / ('baseline-' + system + '-performance-001')
                run('measure_' + system, [sys.executable, '-B', str(HERE / 'run_cells_v1.py'),
                    '--release', releases[system]['path'], '--out', str(out), '--run'])
                result = p.read(out / 'status.json')
                assert result['complete'] and not result['failed'] and not result['node_lease_held']
                assert result['finished_s'] and not control.r.alive(result['pid'])
                assert len(result['observed_checkpoints']) == len(result['completed']) == 2
                state['completed_systems'].append(dict(system=system, status=p.ref(out / 'status.json')))
                state['observed_checkpoints'].extend(result['observed_checkpoints'])
                save()
            assert len(state['observed_checkpoints']) == 8
            state.update(complete=True, phase='all_eight_baseline_runs_complete')
        except BaseException as exc:
            state.update(error=repr(exc), phase='stopped_for_diagnosis')
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            state.update(finished_s=time.time(), node_lease_held=False)
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--cold-stage', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cold_stage:
        assert args.run, 'cold stage requires --run'
        validate()
        stage_cold()
    elif args.run:
        execute()
    else:
        print(json.dumps(validate(), indent=2))
