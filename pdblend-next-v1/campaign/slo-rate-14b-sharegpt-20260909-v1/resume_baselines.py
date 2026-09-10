"""Resume only the unmeasured baseline grid after a diagnosed setup repair.

All previous observations and the confirmed PDB boundary remain immutable.
This entry does not retry a failed or valid performance measurement.
"""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import signal
import socket
import sys
import time

import slo_support as p
import contract
import node_run as original


def children_terminal(state):
    for key in ('child', 'unsettled_child'):
        child = state.get(key)
        if child:
            p.need(not p.active_owner(child) and child.get('finished_s') and
                   not child.get('physical_lease_release_not_certified'),
                   'previous descendant cleanup is not certified: ' + key)


def validate(args):
    previous = p.checked(p.ref(args.predecessor))
    p.need(previous['node'] == args.node and previous.get('finished_s') and
           not previous.get('node_lease_held') and not p.active_owner(previous),
           'previous supervisor must have exited')
    children_terminal(previous)
    failed_setup = (previous.get('phase') == 'engineering_diagnosis' and
                    'baseline preparation failed' in previous.get('error', ''))
    paused_setup = previous.get('phase') == 'stopped_at_boundary'
    p.need(failed_setup or paused_setup, 'only a terminal baseline setup interruption is supported')
    last = p.checked(previous['last_cell_status'])
    p.need(last.get('complete') and last.get('cleanup_complete') and last.get('finished_s')
           and not last.get('node_lease_held') and not p.active_owner(last),
           'last performance measurement must be valid and fully terminal')
    observations = original.read_observations(p.HERE / args.node / 'observations.json')
    p.need(all(o.get('measurement_valid') and o.get('engineering_attempt') == 1 for o in observations),
           'this resume cannot replace performance attempts')
    decision = contract.evaluate_group(args.node, observations)
    p.need(decision['status'] == 'cap_confirmed' and not decision.get('complete'),
           'confirmed PDB boundary and unfinished baseline grid required')
    p.need(all(o['system'] == 'pdblend' for o in observations),
           'this setup recovery must precede all baseline performance measurements')
    boundary_ref = p.ref(args.boundary) if args.boundary else previous['pdb_boundary']
    boundary = p.checked(boundary_ref)
    p.need(boundary['node'] == args.node and boundary['observation_values'] == observations and
           boundary['decision'] == decision and boundary['last_cell_status'] == previous['last_cell_status'],
           'confirmed boundary changed')
    repair = p.checked(p.ref(args.repair))
    p.need(repair.get('node') == args.node and repair.get('implementation_unchanged') is True and
           repair.get('diagnosis'), 'explicit repair evidence required')
    if failed_setup:
        failed = p.checked(repair['failed_setup_status'])
        p.need(failed.get('finished_s') and not failed.get('node_lease_held') and not p.active_owner(failed),
               'failed setup owner is still active')
        children_terminal(failed)
    else:
        p.need(repair.get('preventive_setup_correction') is True and
               boundary.get('paused_predecessor') == p.ref(args.predecessor),
               'preventive setup correction must retain the exact paused predecessor')
    ready = p.checked(p.ref(args.baseline_ready))
    p.need(set(ready) == set(contract.SYSTEMS[1:]), 'all four qualified baseline handoffs required')
    for handoff in ready.values():
        for key in ('qualification', 'qualification_validator', 'binding', 'measurement_executor'):
            p.need(p.sha(handoff[key]['path']) == handoff[key]['sha256'], 'repaired handoff changed')
    return previous, observations, ready, boundary_ref


async def execute(args, state, observations, ready):
    for system in contract.SYSTEMS[1:]:
        handoff = ready[system]
        while True:
            if (p.HERE / args.node / 'STOP').exists():
                state['phase'] = 'stopped_at_boundary'
                return
            decision = contract.evaluate_group(args.node, observations)
            state['decision'] = decision
            p.save(args.out / 'status.json', state)
            tasks = [r for r in decision.get('baseline_tasks', []) if r['system'] == system]
            if not tasks:
                break
            for key in ('qualification', 'qualification_validator', 'binding', 'measurement_executor'):
                p.need(p.sha(handoff[key]['path']) == handoff[key]['sha256'], 'handoff reference drift')
            if not await original.one(min(tasks, key=lambda r: r['rate_rps']), handoff, observations,
                                      p.HERE / args.node, args.out, state):
                return
    decision = contract.evaluate_group(args.node, observations)
    p.need(decision['complete'], 'five-system grid remains incomplete')
    state.update(complete=True, five_system_complete=True, phase='complete', decision=decision)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', choices=('A', 'C'), required=True)
    for name in ('predecessor', 'baseline-ready', 'repair', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--boundary', type=Path)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    p.need(socket.gethostname() == {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}[args.node],
           'wrong physical node')
    p.need('PDBLEND_NODE_LOCK_FD' not in os.environ, 'inherited hardware lease forbidden')
    lock = (p.HERE / args.node / 'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous, observations, ready, boundary_ref = validate(args)
    if not args.run:
        print('CPU resume preflight passed; no performance measurement submitted')
        return
    args.out = args.out.resolve()
    p.need(args.out.is_relative_to(p.HERE / args.node) and not args.out.exists(), 'fresh node output required')
    args.out.mkdir()
    state = dict(schema='slo-rate-node-status-v1', node=args.node, pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        protocol=p.ref(p.HERE / 'protocol.json'), complete=False, node_lease_held=False,
        phase='resuming_baselines', attempts=[], prior_observation_count=len(observations),
        predecessor=p.ref(args.predecessor), repair=p.ref(args.repair), baseline_ready=p.ref(args.baseline_ready),
        pdb_boundary=boundary_ref, source=p.ref(__file__), original_executor=p.ref(original.__file__),
        unchanged_observations=[o['audit_reference'] for o in observations])
    p.save(args.out / 'status.json', state)
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(args, state, observations, ready)
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
