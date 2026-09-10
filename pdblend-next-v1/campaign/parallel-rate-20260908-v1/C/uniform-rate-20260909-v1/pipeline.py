"""Sequential per-node dispatcher; exact grid decisions and immutable attempts."""
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

import support as p
import prepare_release


def group_observations(state, dataset, *, pdb_only=True):
    result = []
    for reference in state['observations']:
        value = p.checked(reference)
        if value['dataset'] == dataset and (not pdb_only or value['system'] == 'pdblend'):
            result.append(reference)
    return result


def resolve_group(plan, state, contract, dataset):
    group = contract.resolve_group(state['declaration'], plan['model'], dataset, actual_host=plan['node'])
    dynamic = [p.checked(r) for r in plan.get('dynamic_reuse_observations', [])
               if p.checked(r)['dataset'] == dataset]
    return contract.apply_audited_reuse(group, dynamic) if dynamic else group


async def child_command(argv, log, state, status):
    with Path(log).open('xb') as stream:
        child = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=asyncio.subprocess.STDOUT,
                                                     start_new_session=True)
        state['child'] = dict(pid=child.pid, argv=argv, started_s=time.time())
        p.save(status, state)
        try:
            code = await child.wait()
        except BaseException:
            if child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 130)
                except asyncio.TimeoutError:
                    # A still-live measurement is never treated as cleaned up.
                    state['child_cleanup_unresolved'] = True
            raise
        state['child'].update(exitcode=code, finished_s=time.time())
        p.save(status, state)
        p.need(code == 0, 'child failed; preserve its attempt and diagnose before any successor')


async def system_stage(plan, system, out, state):
    """A stage handoff pins its qualified binding and optional measured setup.

    If a producer command is configured, it is invoked once and must acquire
    the original node lock for hardware. Otherwise the dispatcher waits for
    the exact planned handoff path while retaining no hardware lock.
    """
    handoff_path = Path(plan['system_handoffs'][system])
    producer = plan.get('stage_producers', {}).get(system)
    request_path = plan.get('stage_requests', {}).get(system)
    if not handoff_path.exists() and producer is None and request_path:
        while not Path(request_path).exists() and not handoff_path.exists():
            p.need(not any(Path(path).exists() for path in plan['stop_paths']), 'stop requested at stage boundary')
            state.update(phase='waiting_for_' + system + '_stage', waiting_for=request_path)
            p.save(out / 'status.json', state)
            await asyncio.sleep(5)
        if not handoff_path.exists():
            source = p.ref(request_path)
            producer = p.checked(source)
            state.setdefault('stage_requests', {})[system] = source
    if not handoff_path.exists() and producer:
        for reference in producer['sources']:
            p.need(p.sha(reference['path']) == reference['sha256'], 'stage producer source changed')
        state['phase'] = 'prepare_' + system
        await child_command(producer['argv'], out / ('prepare-' + system + '.log'), state, out / 'status.json')
    while not handoff_path.exists():
        p.need(not any(Path(path).exists() for path in plan['stop_paths']), 'stop requested at stage boundary')
        state.update(phase='waiting_for_' + system + '_qualification', waiting_for=str(handoff_path))
        p.save(out / 'status.json', state)
        await asyncio.sleep(5)
    reference = p.ref(handoff_path)
    handoff = p.checked(reference)
    p.need(handoff['node'] == plan['node'] and handoff['model'] == plan['model']
           and handoff['system'] == system, 'stage handoff identity differs')
    verification = out / ('verify-' + system + '.json')
    await child_command([sys.executable, '-B', str(p.HERE / 'validate_handoff.py'),
                         '--handoff', str(handoff_path), '--out', str(verification)],
                         out / ('verify-' + system + '.log'), state, out / 'status.json')
    q = p.read(verification)
    b = p.checked(q['binding'])
    p.need(b['hostname'] == plan['expected_hostname'] and b['system'] == system,
           'handoff actual binding differs')
    state.setdefault('system_handoffs', {})[system] = reference
    return handoff


async def run_rate(plan, handoff, selected, dataset, state, out):
    p.need(not any(Path(path).exists() for path in plan['stop_paths']), 'stop requested at rate boundary')
    rows = [task['row'] for task in selected]
    rate, system = rows[0]['rate_rps'], rows[0]['system']
    rate_name = str(rate).replace('.', 'p')
    name = f'{len(state["attempts"])+1:04d}-{system}-{dataset}-r{rate_name}'
    attempt = out / name
    state['attempts'].append(dict(name=name, system=system, dataset=dataset, rate_rps=rate,
                                  cells=[r['cell_id'] for r in rows], started_s=time.time()))
    state.update(phase=system, current_dataset=dataset, current_rate_rps=rate)
    p.save(out / 'status.json', state)
    observations = group_observations(state, dataset, pdb_only=False)
    predecessors = list(handoff.get('predecessors', []))
    if state.get('last_cell_status'):
        predecessors.append(state['last_cell_status'])
    kwargs = dict(declaration=state['declaration'], qualification=handoff['qualification'],
        qualification_validator=handoff['qualification_validator'], node=plan['node'], model=plan['model'],
        dataset=dataset, rate=rate, system=system, out=str(attempt / 'release'),
        scheduling_observations=observations, predecessors=predecessors,
        repeats=tuple(r['repeat'] for r in rows), measurement_executor=handoff.get('measurement_executor'),
        extra_files=handoff.get('extra_files', []), stop_paths=plan['stop_paths'])
    kwargs['dynamic_reuse_observations'] = [r for r in plan.get('dynamic_reuse_observations', [])
                                            if p.checked(r)['dataset'] == dataset]
    request = attempt / 'prepare-request.json'
    result = attempt / 'release-reference.json'
    p.save(request, dict(prepare_adapter=handoff.get('prepare_adapter'), rows=rows, kwargs=kwargs))
    await child_command([sys.executable, '-B', str(p.HERE / 'prepare_request.py'),
                         '--request', str(request), '--out', str(result)],
                         attempt / 'prepare.log', state, out / 'status.json')
    release = p.read(result)
    await child_command([sys.executable, '-B', str(p.HERE / 'run_cells.py'), '--release', release['path'],
                         '--out', str(attempt / 'measurement'), '--run'], attempt / 'measurement.log',
                         state, out / 'status.json')
    terminal = p.ref(attempt / 'measurement/status.json')
    saved = p.checked(terminal)
    p.need(saved['complete'] and not saved.get('error') and not saved['failed']
           and not saved['node_lease_held'] and saved['finished_s'] and not p.active_owner(saved),
           'cell stage did not terminate cleanly')
    state['observations'].extend(saved['observations'])
    state['last_cell_status'] = terminal
    state['attempts'][-1].update(finished_s=time.time(), status=terminal, complete=True)
    p.save(out / 'status.json', state)


async def execute(plan, out, state):
    contract = p.load(plan['contract'], 'uniform_pipeline_contract')
    for dataset in plan['dataset_order']:
        while True:
            group = resolve_group(plan, state, contract, dataset)
            observations = [p.checked(r) for r in group_observations(state, dataset, pdb_only=False)]
            decision = contract.select_group(group, observations)
            state.setdefault('group_decisions', {})[dataset] = decision
            p.save(out / 'status.json', state)
            if decision['phase'] in ('baselines', 'complete'):
                break
            p.need(decision['phase'] != 'diagnosis', 'PDB engineering failure requires diagnosis')
            if decision['phase'] == 'extension_declaration_required':
                generator = p.load(plan['generator'], 'uniform_pipeline_generator')
                destination = out / ('extension-' + dataset + '-' + decision['next_rate_rps_decimal'].replace('.', 'p'))
                extended = generator.append_point(state['declaration'], plan['model'], dataset,
                                                   decision['next_rate_rps_decimal'], destination)
                state['declaration'] = p.ref(destination / 'declaration.json')
                state.setdefault('extensions', []).append(state['declaration'])
                continue
            p.need(decision['phase'] == 'pdblend', 'unknown PDB scheduling phase')
            if 'pdblend' not in state.get('system_handoffs', {}):
                pdb = await system_stage(plan, 'pdblend', out, state)
            else:
                pdb = p.checked(state['system_handoffs']['pdblend'])
            tasks = decision['next_tasks']
            if pdb.get('per_cell_release'):
                tasks = tasks[:1]
            await run_rate(plan, pdb, tasks, dataset, state, out)
    for system in plan['baseline_system_order']:
        handoff = None
        for dataset in plan['dataset_order']:
            group = resolve_group(plan, state, contract, dataset)
            decision = contract.select_group(group, [p.checked(r) for r in group_observations(state, dataset, pdb_only=False)])
            if decision['phase'] == 'complete':
                continue
            p.need(decision['phase'] == 'baselines', 'baseline phase lost PDB cap')
            done = {p.checked(r)['cell_id'] for r in group_observations(state, dataset, pdb_only=False)}
            tasks = [t for t in decision['baseline_tasks'] if t['action'] == 'execute'
                     and t['row']['system'] == system and t['cell_id'] not in done]
            by_rate = {}
            for task in tasks:
                by_rate.setdefault(task['row']['rate_rps'], []).append(task)
            for rate in sorted(by_rate):
                if handoff is None:
                    handoff = await system_stage(plan, system, out, state)
                await run_rate(plan, handoff, sorted(by_rate[rate], key=lambda t: t['repeat']), dataset, state, out)
        state.setdefault('completed_systems', []).append(system)
        p.save(out / 'status.json', state)
    for dataset in plan['dataset_order']:
        group = resolve_group(plan, state, contract, dataset)
        final = contract.select_group(group, [p.checked(r) for r in group_observations(state, dataset, pdb_only=False)])
        p.need(final['phase'] == 'complete', 'cannot finish with missing/faulted reached baseline pairs')
        state['group_decisions'][dataset] = final
    state.update(complete=True, phase='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan_ref = p.ref(args.plan)
    plan = p.checked(plan_ref)
    p.need(plan['schema'] == 'uniform-rate-node-pipeline-v1', 'unknown node pipeline')
    p.need(plan['dataset_order'] == ['alpaca', 'sharegpt', 'longbench']
           and plan['baseline_system_order'] == ['mixed', 'distserve', 'dynamollm', 'ecoserve'], 'phase order differs')
    p.load(plan['contract'], 'uniform_preflight_contract').load_declaration(plan['declaration'])
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True)))
        return
    p.need(socket.gethostname() == plan['expected_hostname'], 'wrong actual node')
    p.need(not args.out.exists(), 'fresh pipeline directory required; no automatic replay')
    args.out.mkdir(parents=True)
    owner = Path(plan['supervisor_lock']).open('a+')
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(schema='uniform-rate-node-pipeline-status-v1', plan=plan_ref, pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        declaration=plan['declaration'], node=plan['node'], model=plan['model'], complete=False,
        node_lease_held=False, phase='starting', attempts=[], observations=list(plan.get('initial_observations', [])))
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(plan, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(args.out / 'status.json', state)
        owner.close()


if __name__ == '__main__':
    main()
