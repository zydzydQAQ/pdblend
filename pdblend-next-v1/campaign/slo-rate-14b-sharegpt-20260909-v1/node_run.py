"""One host's ascending PDB scan followed by the four paired baseline grids."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import slo_support as p
import contract


def read_observations(path):
    if not Path(path).exists():
        return []
    observations = p.read(path)
    for observation in observations:
        raw = p.checked(observation['audit_reference'])
        for key, value in raw.items():
            p.need(observation.get(key) == value, 'observation differs from its immutable audit: ' + key)
    return observations


async def child(argv, log_path, state, status_path, env=None):
    with Path(log_path).open('ab') as log:
        proc = await asyncio.create_subprocess_exec(*map(str, argv), stdout=log, stderr=asyncio.subprocess.STDOUT,
                                                     env=env, start_new_session=True)
        state['child'] = dict(pid=proc.pid, startticks=p.process_identity(proc.pid)['startticks'],
                              argv=list(map(str, argv)), log=str(log_path), started_s=time.time())
        p.save(status_path, state)
        try:
            code = await proc.wait()
        except BaseException:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 180)
                except asyncio.TimeoutError:
                    # A baseline child can be forwarding cancellation through
                    # a qualifier to a native gate with a 120-second cleanup.
                    # Killing its parent would not prove its descendants exited.
                    state['unsettled_child'] = dict(state['child'], cancellation_requested=True,
                        cleanup_wait_s=180, physical_lease_release_not_certified=True)
            raise
        finally:
            state['child'].update(exitcode=proc.returncode, observed_s=time.time())
            if proc.returncode is not None:
                state['child']['finished_s'] = time.time()
            p.save(status_path, state)
    return code


def environment(handoff):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if handoff.get('runtime_pythonpath'):
        paths = handoff['runtime_pythonpath']
        env['PYTHONPATH'] = ':'.join(paths) if isinstance(paths, list) else paths
    return env


async def one(row, handoff, observations, node_dir, run_dir, state):
    status_path = run_dir / 'status.json'
    observation_path = node_dir / 'observations.json'
    sequence = len(state['attempts']) + 1
    cell_root = run_dir / ('cell-' + str(sequence).zfill(4))
    cell_root.mkdir()
    p.save(cell_root / 'observations.before.json', observations)
    argv = [sys.executable, '-B', p.HERE / 'run_cell.py', 'prepare', '--node', state['node'],
            '--rate', row['rate_rps_decimal'], '--system', row['system'], '--repeat', row['repeat'],
            '--qualification', handoff['qualification']['path'], '--validator', handoff['qualification_validator']['path'],
            '--observations', cell_root / 'observations.before.json', '--out', cell_root / 'release']
    if handoff.get('measurement_executor'):
        argv += ['--measurement-executor', handoff['measurement_executor']['path']]
    state.update(phase='preparing_cell', current_cell=row['cell_id'])
    code = await child(argv, cell_root / 'prepare.log', state, status_path, environment(handoff))
    p.need(code == 0, 'cell preparation failed; no measurement submitted: ' + str(cell_root / 'prepare.log'))
    state['attempts'].append(dict(row=row, path=str(cell_root / 'measurement'), started_s=time.time()))
    state['phase'] = 'measuring_' + row['system']
    code = await child([sys.executable, '-B', p.HERE / 'run_cell.py', 'execute',
        '--release', cell_root / 'release/release.json', '--out', cell_root / 'measurement', '--run'],
        cell_root / 'measurement.log', state, status_path, environment(handoff))
    cell_status_path = cell_root / 'measurement/status.json'
    p.need(cell_status_path.exists(), 'measurement exited without durable status')
    cell_state = p.read(cell_status_path)
    state['last_cell_status'] = p.ref(cell_status_path)
    if code == 0 and cell_state.get('complete'):
        audit_reference = cell_state['audit_reference']
        observation = p.checked(audit_reference)
        observation.update(audit_reference=audit_reference, engineering_attempt=1)
    else:
        observation = dict(row, measurement_valid=False, service_terminal_valid=False,
            strict_slo_recomputed=False, independently_recomputed=False, work_complete=None,
            unknown_error_count=1, engineering_attempt=1, error=cell_state.get('error', 'measurement child failed'),
            checkpoint=cell_state.get('checkpoint'), cleanup_complete=cell_state.get('cleanup_complete', False))
        failure_path = cell_root / 'measurement/engineering-failure.json'
        p.save(failure_path, observation)
        observation['audit_reference'] = p.ref(failure_path)
        state.update(phase='engineering_diagnosis', engineering_fault=observation,
                     automatic_retry=False, complete=False)
    observations.append(observation)
    p.save(observation_path, observations)
    p.save(status_path, state)
    return code == 0 and cell_state.get('complete')


async def execute(args, state):
    node_dir = p.HERE / args.node; run_dir = args.out.resolve(); status_path = run_dir / 'status.json'
    p.need(socket.gethostname() == {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}[args.node],
           'wrong physical node')
    ready = p.checked(p.ref(args.handoff))
    p.need(ready.get('complete') and ready['node'] == args.node, 'PDB environment not ready')
    for key in ('qualification', 'qualification_validator', 'binding'):
        p.need(p.sha(ready[key]['path']) == ready[key]['sha256'], 'handoff reference changed')
    observations = read_observations(node_dir / 'observations.json')
    p.need(not observations, 'new supervisor expects no prior measurements; use explicit reviewed resume')
    while True:
        if (node_dir / 'STOP').exists():
            state['phase'] = 'stopped_at_boundary'; return
        decision = contract.evaluate_group(args.node, observations)
        state['decision'] = decision; p.save(status_path, state)
        if decision['status'] == 'cap_confirmed':
            break
        p.need(decision['status'] in ('measure_pdblend', 'confirm_boundary'), 'PDB requires engineering diagnosis')
        if not await one(decision['next_tasks'][0], ready, observations, node_dir, run_dir, state):
            return
    cell_state = p.checked(state['last_cell_status'])
    p.need(cell_state.get('finished_s') and not cell_state.get('node_lease_held') and not p.active_owner(cell_state),
           'PDB child still owns execution')
    boundary = dict(schema='slo-rate-pdb-boundary-v1', node=args.node, model='14b', datasets=['sharegpt'],
        campaign_id=contract.CAMPAIGN, dataset='sharegpt', pdb_boundary_complete=True,
        binding=ready['binding'], last_cell_status=state['last_cell_status'],
        scope='pdblend', complete=True, finished_s=time.time(), node_lease_held=False,
        pid=cell_state['pid'], startticks=cell_state['startticks'], source_cell_status=state['last_cell_status'],
        supervisor_pid=os.getpid(), decision=decision, observations=[o['audit_reference'] for o in observations],
        observation_values=observations,
        handoff=ready, physical_owner_is_exited_measurement_child=True)
    p.save(run_dir / 'pdb-boundary.json', boundary)
    state.update(phase='awaiting_baseline_preparation', pdb_boundary=p.ref(run_dir / 'pdb-boundary.json'))
    p.save(status_path, state)
    # This CPU command descriptor can arrive while the PDB sweep is in flight.
    # It names a qualified transition, never grants permission to a historical queue.
    while not args.baseline_command.exists():
        if (node_dir / 'STOP').exists():
            state['phase'] = 'stopped_at_boundary'; return
        await asyncio.sleep(5)
    command = p.read(args.baseline_command)
    p.need(command['node'] == args.node and command.get('schema') == 'slo-rate-baseline-command-v1', 'wrong baseline command')
    for path, digest in command['files'].items():
        p.need(p.sha(path) == digest, 'baseline transition source changed')
    argv = [str(x).replace('{pdb_boundary}', str(run_dir / 'pdb-boundary.json')) for x in command['argv']]
    state['phase'] = 'preparing_baselines'
    code = await child(argv, run_dir / 'baseline-preparation.log', state, status_path, environment(command))
    p.need(code == 0, 'baseline preparation failed')
    baseline_ready = p.read(command['handoffs_path'])
    for system in contract.SYSTEMS[1:]:
        handoff = baseline_ready[system]
        while True:
            if (node_dir / 'STOP').exists():
                state['phase'] = 'stopped_at_boundary'; return
            decision = contract.evaluate_group(args.node, observations)
            state['decision'] = decision; p.save(status_path, state)
            tasks = [r for r in decision['baseline_tasks'] if r['system'] == system]
            if not tasks:
                break
            row = min(tasks, key=lambda r: r['rate_rps'])
            if not await one(row, handoff, observations, node_dir, run_dir, state):
                return
    decision = contract.evaluate_group(args.node, observations)
    p.need(decision['complete'], 'five-system grid is incomplete')
    state.update(complete=True, phase='complete', decision=decision, five_system_complete=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', choices=('A', 'C'), required=True)
    ap.add_argument('--handoff', type=Path, required=True)
    ap.add_argument('--baseline-command', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    p.need(args.run, 'explicit --run is required')
    args.out = args.out.resolve()
    p.need(not args.out.exists(), 'fresh supervisor directory required')
    lock = (p.HERE / args.node / 'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir(parents=True)
    state = dict(schema='slo-rate-node-status-v1', node=args.node, pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        protocol=p.ref(p.HERE / 'protocol.json'), complete=False, node_lease_held=False,
        phase='starting', attempts=[])
    p.save(args.out / 'status.json', state)
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(args, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(error=repr(exc), phase='engineering_diagnosis')
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(args.out / 'status.json', state)
        lock.close()


if __name__ == '__main__':
    main()
