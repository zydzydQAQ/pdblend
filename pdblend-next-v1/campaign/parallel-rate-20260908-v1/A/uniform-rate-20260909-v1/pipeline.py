"""New-A PDB-only dispatch: fixed SG/LB, fresh dynamic qualification, Alpaca."""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'C/uniform-rate-20260909-v1'))
import support as p

shared = p.load(ROOT / 'C/uniform-rate-20260909-v1/pipeline.py', 'newA_shared_pipeline')


async def stage(plan, dataset, state, out):
    definition = plan['dataset_stages'][dataset]
    handoff_path = Path(definition['handoff'])
    if not handoff_path.exists():
        producer = definition.get('producer')
        if not producer and definition.get('producer_release'):
            release = Path(definition['producer_release'])
            # CPU preparation may finish while the preceding fixed datasets run.
            # The published release starts the actual producer automatically.
            while not release.exists():
                p.need(not any(Path(x).exists() for x in plan['stop_paths']), 'stop requested')
                state.update(phase='awaiting_dynamic_producer_release', waiting_for=str(release))
                p.save(out / 'status.json', state)
                await asyncio.sleep(5)
            producer = p.checked(p.ref(release))
            state['dynamic_producer_release'] = p.ref(release)
        p.need(producer is not None, 'automatic qualification producer missing')
        for reference in producer['sources']:
            p.need(p.sha(reference['path']) == reference['sha256'], 'qualification producer source changed')
        state.update(phase='qualifying_' + dataset)
        await shared.child_command(producer['argv'], out / ('qualify-' + dataset + '.log'), state, out / 'status.json')
    p.need(handoff_path.exists(), 'producer returned without an independently qualified handoff')
    reference = p.ref(handoff_path)
    handoff = p.checked(reference)
    p.need((handoff['node'], handoff['model'], handoff['system']) == ('Anew20260909', '14b', 'pdblend'),
           'qualification handoff identity differs')
    verification = out / ('verify-' + dataset + '.json')
    await shared.child_command([sys.executable, '-B', str(p.HERE / 'validate_handoff.py'),
        '--handoff', str(handoff_path), '--out', str(verification)],
        out / ('verify-' + dataset + '.log'), state, out / 'status.json')
    result = p.read(verification)
    binding = p.checked(result['binding'])
    p.need(binding['hostname'] == plan['expected_hostname'] and dataset in binding['configs'], 'qualified binding mismatch')
    state.setdefault('dataset_handoffs', {})[dataset] = reference
    return handoff


async def execute(plan, out, state):
    contract = p.load(plan['contract'], 'newA_uniform_contract')
    for dataset in plan['dataset_order']:
        handoff = None
        while True:
            group = shared.resolve_group(plan, state, contract, dataset)
            decision = contract.select_group(group, [p.checked(r) for r in shared.group_observations(state, dataset)])
            state.setdefault('group_decisions', {})[dataset] = decision
            p.save(out / 'status.json', state)
            if decision['phase'] in ('baselines', 'complete'):
                state.setdefault('completed_pdblend_datasets', []).append(dataset)
                break
            p.need(decision['phase'] != 'diagnosis', 'engineering failure requires diagnosis, not a capacity boundary')
            if decision['phase'] == 'extension_declaration_required':
                generator = p.load(plan['generator'], 'newA_uniform_generator')
                destination = out / ('extension-' + dataset + '-' + decision['next_rate_rps_decimal'].replace('.', 'p'))
                generator.append_point(state['declaration'], '14b', dataset, decision['next_rate_rps_decimal'], destination)
                state['declaration'] = p.ref(destination / 'declaration.json')
                state.setdefault('extensions', []).append(state['declaration'])
                continue
            p.need(decision['phase'] == 'pdblend', 'unknown scheduling phase')
            if handoff is None:
                handoff = await stage(plan, dataset, state, out)
            tasks = decision['next_tasks'][:1] if handoff.get('per_cell_release') else decision['next_tasks']
            await shared.run_rate(plan, handoff, tasks, dataset, state, out)
    p.need(state['completed_pdblend_datasets'] == plan['dataset_order'], 'PDB group incomplete')
    state.update(complete=True, phase='complete_pdblend_only', baseline_work_started=False,
                 all_five_system_contract_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = p.checked(p.ref(args.plan))
    p.need(plan['schema'] == 'new-A-uniform-pdblend-only-pipeline-v1', 'wrong plan')
    p.need(plan['dataset_order'] == ['sharegpt', 'longbench', 'alpaca'] and
           (plan['node'], plan['model']) == ('Anew20260909', '14b'), 'wrong node or dataset order')
    for path, digest in plan['files'].items():
        p.need(p.sha(path) == digest, 'dispatcher source changed')
    p.load(plan['contract'], 'newA_preflight_contract').load_declaration(plan['declaration'])
    if not args.run:
        print('CPU declaration and dispatcher checks passed')
        return
    p.need(socket.gethostname() == plan['expected_hostname'] and not args.out.exists(), 'wrong host or reused output')
    owner = Path(plan['supervisor_lock']).open('a+')
    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir(parents=True)
    state = dict(schema='uniform-rate-node-pipeline-status-v1', plan=p.ref(args.plan), pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        declaration=plan['declaration'], node=plan['node'], model=plan['model'], complete=False,
        node_lease_held=False, phase='starting', attempts=[], observations=[])
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(plan, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(error=repr(exc), phase='stopped_failure')
        raise
    finally:
        state['finished_s'] = time.time()
        p.save(args.out / 'status.json', state)
        owner.close()


if __name__ == '__main__':
    main()
