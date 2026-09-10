"""Five-system missing-cell continuation after the declared PDB boundary."""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
contract = p.load(Path(__file__).parent / 'contract_capacity_v1.py', 'C_capacity_contract')


async def child(argv, log, state, out):
    with Path(log).open('xb') as stream:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=asyncio.subprocess.STDOUT,
                                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        state['child'] = dict(pid=proc.pid, argv=argv, started_s=time.time(),
                              startticks=p.process_identity(proc.pid)['startticks'])
        p.save(out / 'status.json', state)
        try:
            code = await proc.wait()
        except BaseException:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            raise
        state['child'].update(exitcode=code, finished_s=time.time())
        p.save(out / 'status.json', state)
        p.need(code == 0, 'child failed; preserve this attempt and stop')


def observations(state, dataset):
    return [r for r in state['observations'] if p.checked(r)['dataset'] == dataset]


def decision(plan, state, dataset):
    group = contract.resolve_group(state['declaration'], plan['model'], dataset, actual_host=plan['node'])
    return contract.select_group(group, [p.checked(r) for r in observations(state, dataset)])


async def execute(plan, state, out):
    for system in plan['system_order']:
        handoff = p.checked(plan['handoffs'][system])
        p.need(handoff['node'] == plan['node'] and handoff['model'] == plan['model']
               and handoff['system'] == system, 'wrong handoff identity')
        for dataset in plan['datasets']:
            while True:
                selected = decision(plan, state, dataset)
                state['group_decisions'][dataset] = selected
                p.save(out / 'status.json', state)
                if selected['phase'] == 'complete':
                    break
                p.need(selected['phase'] == 'baselines', 'PDB boundary missing or engineering failure needs diagnosis')
                tasks = [t for t in selected['baseline_tasks'] if t['action'] == 'execute' and t['row']['system'] == system]
                if not tasks:
                    break
                tasks.sort(key=lambda t:(t['row']['rate_rps'], t['row'].get('measurement_purpose', 'normal'), t['row']['repeat']))
                row = tasks[0]['row']
                p.need(not any(Path(x).exists() for x in plan['stop_paths']), 'stop requested at cell boundary')
                name = f'{len(state["attempts"])+1:04d}-{system}-{dataset}-r{row["rate_rps"]}-{row.get("measurement_purpose", "normal")}'
                attempt = out / name
                predecessors = list(handoff.get('predecessors', []))
                if state.get('last_cell_status'):
                    predecessors.append(state['last_cell_status'])
                kwargs = dict(declaration=state['declaration'], qualification=handoff['qualification'],
                    qualification_validator=handoff['qualification_validator'], node=plan['node'], model=plan['model'],
                    dataset=dataset, rate=row['rate_rps'], system=system, out=str(attempt / 'release'),
                    scheduling_observations=observations(state, dataset), predecessors=predecessors,
                    repeats=[row['repeat']], measurement_purpose=row.get('measurement_purpose','normal'),
                    stop_paths=plan['stop_paths'], extra_files=[state['plan'], p.ref(__file__)],
                    cell_auditor=plan['cell_auditor'], cell_auditor_dependencies=plan['cell_auditor_dependencies'])
                p.save(attempt / 'prepare-request.json', dict(rows=[row], kwargs=kwargs))
                state['attempts'].append(dict(name=name, cell_id=row['cell_id'], started_s=time.time()))
                state.update(phase=system, current_dataset=dataset, current_rate_rps=row['rate_rps'])
                await child([sys.executable, '-B', str(Path(__file__).parent / 'prepare_request_v4.py'), '--request',
                    str(attempt / 'prepare-request.json'), '--out', str(attempt / 'release-reference.json')],
                    attempt / 'prepare.log', state, out)
                ref = p.read(attempt / 'release-reference.json')
                await child([sys.executable, '-B', str(Path(__file__).parent / 'run_cells_v3.py'), '--release', ref['path'],
                    '--out', str(attempt / 'measurement'), '--run'], attempt / 'measurement.log', state, out)
                terminal = p.ref(attempt / 'measurement/status.json')
                result = p.checked(terminal)
                p.need(result['complete'] and not result.get('error') and not result['failed']
                       and not result['node_lease_held'] and not p.active_owner(result), 'measurement did not finish cleanly')
                state['observations'].extend(result['observations'])
                state['last_cell_status'] = terminal
                state['attempts'][-1].update(complete=True, status=terminal, finished_s=time.time())
                p.save(out / 'status.json', state)
    for dataset in plan['datasets']:
        final = decision(plan, state, dataset)
        p.need(final['phase'] == 'complete' and final['five_system_complete'], 'five-system group incomplete')
        state['group_decisions'][dataset] = final
    state.update(complete=True, phase='complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan_ref = p.ref(args.plan)
    plan = p.checked(plan_ref)
    p.need(plan['scope'] == 'five_systems' and socket.gethostname() == plan['expected_hostname'], 'wrong host/scope')
    contract.load_declaration(plan['declaration'])
    if plan.get('previous_pipeline'):
        previous = p.checked(plan['previous_pipeline'])
        p.need(previous['finished_s'] and not previous['node_lease_held'] and not p.active_owner(previous), 'previous supervisor remains active')
        p.need(previous.get('error') == "ValueError('child failed; preserve this attempt and stop')", 'unexpected predecessor failure')
        recovered = p.checked(plan['reconstructed_observation'])
        terminal = p.checked(plan['last_cell_status'])
        p.need(terminal['finished_s'] and not terminal['node_lease_held'] and not p.active_owner(terminal)
               and len(terminal['observed_checkpoints']) == 1 and len(terminal['failed']) == 1
               and terminal['error'] == "ValueError('baseline incomplete work is not an explicit isolated request deadline')",
               'unknown physical or audit failure')
        checkpoint = terminal['observed_checkpoints'][0]
        p.need(recovered['checkpoint'] == checkpoint and recovered['failure_class'] == 'independently_diagnosed_capacity_rejection'
               and recovered['independently_recomputed'] is True and contract.acceptable_baseline(recovered),
               'reconstructed capacity rejection lacks independent proof')
        fresh = p.load(plan['cell_auditor'], 'C_recovered_capacity_audit').audit(checkpoint)
        for key in ('cell_id', 'checkpoint', 'measurement_valid', 'work_complete', 'n_expected', 'completed_work_requests',
                    'slo_attainment', 'energy_j', 'ttft_avg_s', 'tpot_avg_s', 'token_throughput_tps', 'gpu_util', 'request_timeouts'):
            p.need(fresh[key] == recovered[key], 'reconstructed physical metric differs: ' + key)
        p.need(previous['child']['exitcode'] == 1 and previous['child']['pid'] == terminal['pid']
               and previous['child']['startticks'] == terminal['startticks'], 'wrong failed child identity')
        p.need(plan['initial_observations'] == previous['observations'] + [plan['reconstructed_observation']],
               'resume must retain every prior observation and exactly the audited missing result')
        p.need(terminal['failed'][0]['cell_id'] == recovered['cell_id'], 'failed cell differs from derived observation')
    if not args.run:
        print('CPU plan passed')
        return
    p.need(not args.out.exists(), 'new pipeline output required')
    args.out.mkdir(parents=True)
    lease = Path(plan['supervisor_lock']).open('a+')
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(schema='uniform-v2-five-system-pipeline-status', plan=plan_ref, pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(), node=plan['node'],
        model=plan['model'], datasets=plan['datasets'], declaration=plan['declaration'],
        declaration_contract=p.ref(Path(__file__).parent / 'contract_capacity_v1.py'), scope='five_systems',
        complete=False, node_lease_held=False, phase='starting', observations=list(plan.get('initial_observations', [])), attempts=[], group_decisions={})
    if plan.get('last_cell_status'):
        state['last_cell_status'] = plan['last_cell_status']
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(plan, state, args.out)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(args.out / 'status.json', state)
        lease.close()


if __name__ == '__main__':
    main()
