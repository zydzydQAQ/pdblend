"""Execute five declared normal cells while retaining one diagnosed metric gap."""
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
U = ROOT / 'B/uniform-rate-20260909-v2'
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
import contract


def select(plan, observations, dataset):
    group = contract.resolve_group(plan['declaration'], '32b', dataset, actual_host='B')
    return contract.select_group(group, [p.checked(r) for r in observations if p.checked(r)['dataset'] == dataset])


def validate(plan):
    p.need(plan['node'] == 'B' and plan['model'] == '32b' and plan['scope'] == 'five_systems', 'wrong scope')
    p.need(socket.gethostname() == plan['expected_hostname'], 'wrong node')
    previous = p.checked(plan['previous_pipeline'])
    p.need(previous['error'] == "ValueError('PDB boundary missing or engineering failure needs diagnosis')"
           and previous['finished_s'] and not previous['node_lease_held'] and not p.active_owner(previous), 'unexpected previous owner')
    p.need(plan['initial_observations'] == previous['observations'] and plan['last_cell_status'] == previous['last_cell_status'], 'prior evidence omitted')
    last = p.checked(plan['last_cell_status'])
    p.need(last['complete'] and not last['failed'] and not last.get('error') and last['finished_s']
           and not last['node_lease_held'] and not p.active_owner(last), 'previous measurement is not clean')
    excluded = plan['metric_gap_observations']
    p.need(len(excluded) == 1 and excluded[0] in previous['observations'], 'only declared observed metric gap may be isolated')
    gap = p.checked(excluded[0])
    p.need(gap['measurement_purpose'] == 'metric_supplement' and gap['system'] == 'dynamollm'
           and gap['dataset'] == 'sharegpt' and gap['rate_rps'] == 1.5 and gap['repeat'] == 1
           and gap['token_throughput_is_exact'] is False and contract.acceptable_baseline(gap), 'unknown metric gap')
    actual = select(plan, previous['observations'], 'sharegpt')
    p.need(actual['phase'] == 'diagnosis' and actual['engineering_fault_cell_ids'] == [gap['cell_id']], 'additional unhandled fault')
    expected = []
    scheduling = [r for r in previous['observations'] if r not in excluded]
    for dataset in plan['datasets']:
        decision = select(plan, scheduling, dataset)
        p.need(decision['phase'] in ('baselines', 'complete'), 'PDB boundary or independent fault blocks normal work')
        expected += [t['row'] for t in decision.get('baseline_tasks', []) if t['action'] == 'execute'
                     and t['row']['measurement_purpose'] == 'normal']
    expected.sort(key=lambda r:(plan['datasets'].index(r['dataset']), r['rate_rps']))
    p.need(plan['rows'] == expected and len(expected) == 5
           and all(r['system'] == 'ecoserve' and r['repeat'] == 1 for r in expected), 'explicit normal tail changed')
    return previous


async def child(argv, log, state, out):
    with log.open('xb') as stream:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=asyncio.subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        state['child'] = dict(pid=proc.pid, startticks=p.process_identity(proc.pid)['startticks'], argv=argv, started_s=time.time())
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
        p.need(code == 0, 'normal tail child failed; preserve attempt')


async def execute(plan, state, out):
    handoff = p.checked(plan['handoff'])
    for row in plan['rows']:
        p.need(not any(Path(x).exists() for x in plan['stop_paths']), 'stop requested at cell boundary')
        attempt = out / row['cell_id']
        observations = [r for r in state['observations'] if r not in plan['metric_gap_observations']
                        and p.checked(r)['dataset'] == row['dataset']]
        kwargs = dict(declaration=plan['declaration'], qualification=handoff['qualification'],
            qualification_validator=handoff['qualification_validator'], node='B', model='32b',
            dataset=row['dataset'], rate=row['rate_rps'], system='ecoserve', out=str(attempt / 'release'),
            scheduling_observations=observations, predecessors=[*handoff['predecessors'], state['last_cell_status']],
            repeats=[1], measurement_purpose='normal', stop_paths=plan['stop_paths'],
            extra_files=[state['plan'], p.ref(__file__), *plan['metric_gap_observations']])
        p.save(attempt / 'prepare-request.json', dict(rows=[row], kwargs=kwargs))
        state.update(phase='ecoserve', current_dataset=row['dataset'], current_rate_rps=row['rate_rps'])
        state['attempts'].append(dict(cell_id=row['cell_id'], started_s=time.time()))
        await child([sys.executable, '-B', str(ROOT / 'C/uniform-rate-20260909-v2/prepare_request_v3.py'),
            '--request', str(attempt / 'prepare-request.json'), '--out', str(attempt / 'release-reference.json')], attempt / 'prepare.log', state, out)
        release = p.read(attempt / 'release-reference.json')
        await child([sys.executable, '-B', str(p.HERE / 'run_cells.py'), '--release', release['path'],
            '--out', str(attempt / 'measurement'), '--run'], attempt / 'measurement.log', state, out)
        terminal = p.ref(attempt / 'measurement/status.json')
        result = p.checked(terminal)
        p.need(result['complete'] and not result['failed'] and not result.get('error') and result['finished_s']
               and not result['node_lease_held'] and not p.active_owner(result), 'tail measurement is not clean')
        state['observations'].extend(result['observations'])
        state['last_cell_status'] = terminal
        state['attempts'][-1].update(complete=True, status=terminal, finished_s=time.time())
        p.save(out / 'status.json', state)
    state['group_decisions'] = {d:select(plan, state['observations'], d) for d in plan['datasets']}
    state.update(measurements_complete=True, complete=False, phase='awaiting_metric_closure',
        completion_note='All five normal tail cells finished; the original metric gap remains and must be independently closed before full scope completion.')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--run', action='store_true')
    args = a.parse_args()
    plan_ref = p.ref(args.plan)
    plan = p.checked(plan_ref)
    validate(plan)
    if not args.run:
        print('Explicit five-normal-cell tail passed CPU validation')
        return
    p.need(not args.out.exists(), 'fresh output required')
    args.out.mkdir(parents=True)
    lease = Path(plan['supervisor_lock']).open('a+')
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(schema='uniform-v2-five-system-pipeline-status', node='B', model='32b', datasets=plan['datasets'],
        scope='five_systems', declaration=plan['declaration'], plan=plan_ref, started_s=time.time(),
        pid=os.getpid(), startticks=p.process_identity(os.getpid())['startticks'], complete=False,
        node_lease_held=False, phase='starting', observations=list(plan['initial_observations']), attempts=[],
        last_cell_status=plan['last_cell_status'], metric_gap_observations=plan['metric_gap_observations'])
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
