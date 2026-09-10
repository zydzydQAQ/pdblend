"""Publish a confirmed PDB boundary from a supervisor paused between cells."""
import argparse
from pathlib import Path
import socket
import time

import contract
import node_run
import slo_support as p
from resume_baselines import children_terminal


def build(node, predecessor, handoff, out):
    p.need(socket.gethostname() == {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}[node],
           'wrong physical host')
    prior = p.checked(p.ref(predecessor))
    p.need(prior['node'] == node and prior['phase'] == 'stopped_at_boundary' and prior.get('finished_s')
           and not prior.get('node_lease_held') and not p.active_owner(prior), 'supervisor must be terminal and paused')
    children_terminal(prior)
    last = p.checked(prior['last_cell_status'])
    p.need(last.get('complete') and last.get('cleanup_complete') and last.get('finished_s')
           and not last.get('node_lease_held') and not p.active_owner(last), 'last measurement must be terminal and clean')
    observations = node_run.read_observations(p.HERE / node / 'observations.json')
    p.need(all(o['system'] == 'pdblend' and o.get('measurement_valid') and o.get('engineering_attempt') == 1
               for o in observations), 'only unchanged valid PDB measurements allowed')
    decision = contract.evaluate_group(node, observations)
    p.need(decision['status'] == 'cap_confirmed', 'PDB boundary is not confirmed')
    ready = p.checked(p.ref(handoff))
    p.need(ready.get('complete') and ready['node'] == node, 'PDB handoff mismatch')
    p.checked(ready['binding'])
    p.need(all(o['binding'] == ready['binding'] for o in observations),
           'paused boundary must retain the binding actually measured')
    boundary = dict(schema='slo-rate-pdb-boundary-v1', node=node, model='14b', datasets=['sharegpt'],
        campaign_id=contract.CAMPAIGN, dataset='sharegpt', pdb_boundary_complete=True,
        binding=ready['binding'], last_cell_status=prior['last_cell_status'], scope='pdblend', complete=True,
        finished_s=time.time(), node_lease_held=False, pid=last['pid'], startticks=last['startticks'],
        source_cell_status=prior['last_cell_status'], supervisor_pid=prior['pid'],
        decision=decision, observations=[o['audit_reference'] for o in observations], observation_values=observations,
        handoff=ready, physical_owner_is_exited_measurement_child=True,
        paused_predecessor=p.ref(predecessor), boundary_producer=p.ref(__file__))
    p.need(not Path(out).exists(), 'fresh boundary artifact required')
    p.save(out, boundary)
    return p.ref(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node', choices=('A', 'C'), required=True)
    for name in ('predecessor', 'handoff', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    print(build(args.node, args.predecessor, args.handoff, args.out))
