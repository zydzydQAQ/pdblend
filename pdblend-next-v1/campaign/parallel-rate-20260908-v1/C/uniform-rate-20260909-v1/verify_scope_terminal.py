"""Independently verify a completed PDB-only scope and the released node."""
import argparse
import asyncio
import fcntl
from pathlib import Path
import time
import support as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    p.need(not args.out.exists(), 'fresh terminal verification required')
    args.out.mkdir(parents=True)
    state_ref = p.ref(args.pipeline / 'status.json')
    state = p.checked(state_ref)
    plan = p.checked(state['plan'])
    p.need(state['finished_s'] and not p.active_owner(state)
           and 'stop requested at stage boundary' in state.get('error', ''),
           'supervisor has not reached the expected scope boundary')
    completion_ref = p.ref(args.completion)
    completion = p.checked(completion_ref)
    p.need(completion['pdb_complete'] and completion['baseline_dispatched'] is False
           and completion['plan'] == state['plan'] and completion['observations'] == state['observations'],
           'scope receipt differs from actual completed observations')
    p.need(all(a['system'] == 'pdblend' and a['complete'] for a in state['attempts']),
           'unfinished or non-PDB measurement attempt')
    terminal = p.checked(state['last_cell_status'])
    p.need(terminal['complete'] and not terminal['failed'] and not terminal['node_lease_held']
           and terminal['finished_s'] and not p.active_owner(terminal), 'last measurement remains active')
    contract = p.load(plan['contract'], 'terminal_uniform_contract')
    observations = [p.checked(r) for r in state['observations']]
    decisions = {}
    caps = {}
    for dataset in plan['dataset_order']:
        group = contract.resolve_group(state['declaration'], plan['model'], dataset, actual_host=plan['node'])
        dynamic = [p.checked(r) for r in plan.get('dynamic_reuse_observations', [])
                   if p.checked(r)['dataset'] == dataset]
        if dynamic:
            group = contract.apply_audited_reuse(group, dynamic)
        selected = [o for o in observations if o['dataset'] == dataset]
        decision = contract.select_group(group, selected)
        p.need(decision['phase'] in ('baselines', 'complete') and decision['decision']['cap_observed']
               and not decision['decision']['engineering_fault_cell_ids'], 'incomplete/faulted PDB scope')
        decisions[dataset] = decision
        cap = decision['cap_rate_rps']
        rows = [o for o in selected if o['rate_rps'] == cap]
        p.need(len(rows) == 2 and all(o['measurement_valid'] and o['work_complete'] for o in rows),
               'two actual complete cap repetitions required')
        caps[dataset] = dict(rate_rps=cap, slo_attainment=[o['slo_attainment'] for o in rows],
                            reached_grid_positions=len(decision['eligible_position_ids']))
    handoff = p.checked(state['system_handoffs']['pdblend'])
    validator = p.load(handoff['qualification_validator'], 'terminal_native_qualification')
    proof = validator.verify(handoff['qualification'])
    p.need(proof['passed'] and proof['independently_recomputed'], 'qualified source no longer valid')
    binding = p.checked(proof['binding'])
    common = p.load(p.ROOT / 'common/execution-until-complete-v1/run.py', 'terminal_original_identity')
    import aiohttp
    async def identity():
        async with aiohttp.ClientSession(trust_env=False) as session:
            return await common.identity(session, binding)
    lock_path = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
    with lock_path.open('r+') as lock, Path(plan['supervisor_lock']).open('r+') as supervisor:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(supervisor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        native = asyncio.run(identity())
        p.save(args.out / 'native-terminal.json', native)
    guard = p.HERE / 'pdblend-scope-guard-001/status.json'
    p.save(args.out / 'status.json', dict(schema='uniform-PDB-scope-independent-terminal-verification-v1',
        complete=True, passed=True, independently_recomputed=True, node=plan['node'], model=plan['model'],
        pdb_complete=True, baseline_dispatched=False, five_system_complete=False,
        caps=caps, group_decisions=decisions, observations=state['observations'],
        supervisor_terminal=state_ref, last_measurement=state['last_cell_status'],
        scope_completion=completion_ref, native_terminal=p.ref(args.out / 'native-terminal.json'),
        qualification=handoff['qualification'], qualification_validator=handoff['qualification_validator'],
        node_lock_released=True, supervisor_lock_released=True, node_lock_path=str(lock_path),
        redundant_guard_receipt=p.ref(guard),
        redundant_guard_disposition='The separately installed CPU scope producer wrote STOP first; original guard failure retained.',
        verifier=p.ref(__file__), finished_s=time.time()))


if __name__ == '__main__':
    main()
