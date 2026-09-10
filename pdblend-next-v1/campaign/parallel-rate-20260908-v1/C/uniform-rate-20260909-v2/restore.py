"""Measured retained baseline restart with fresh node ownership and evidence."""
import argparse
import asyncio
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
SHARED = ROOT / 'C/uniform-rate-20260909-v1'
sys.path.insert(0, str(SHARED))
import support as p


def terminal(spec):
    p.need(socket.gethostname() == spec['hostname'], 'wrong physical node')
    state = p.checked(spec['predecessor'])
    p.need(state['complete'] and state.get('pdb_complete') is True, 'PDB scope incomplete')
    supervisor = p.checked(spec['supervisor'])
    p.need(supervisor['finished_s'] and not p.active_owner(supervisor)
           and not supervisor.get('node_lease_held'), 'previous supervisor remains active')
    p.need(not Path(spec['stop_path']).exists(), 'new queue stop requested')
    for name, ref in spec['sources'].items():
        p.need(p.sha(ref['path']) == ref['sha256'], 'restart source differs: ' + name)
    return state


async def run(spec, out):
    terminal(spec)
    p.need(not out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh restart/lease required')
    parent = p.checked(spec['target'])
    previous = p.checked(spec['previous'])
    helper = p.load(spec['sources']['restore'], 'v2_original_retained_restore')
    common = helper.load_common(parent['host_release'])
    from ecopadg.serving.campaign import node_lease
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    status = dict(schema='uniform-v2-retained-baseline-owner-v1', node=spec['node'], model=spec['model'],
                  pid=os.getpid(), startticks=p.process_identity(os.getpid())['startticks'],
                  started_s=time.time(), complete=False, node_lease_held=False, spec=p.ref(args.spec))
    owner_path = out.parent / (out.name + '-owner.json')
    p.need(not owner_path.exists(), 'restart owner already exists')
    p.save(owner_path, status)
    guard = None
    try:
        with node_lease():
            terminal(spec)
            status['node_lease_held'] = True
            p.save(owner_path, status)
            common.validate_binding(parent)
            common.validate_binding(previous)
            if spec['node'] == 'C':
                ports = {i[k] for i in parent['instances'] for k in ('port', 'kv_port')}
                guard_module = p.load(spec['sources']['ports'], 'v2_C_original_startup_ports')
                guard = guard_module.Guard(ports)
            with guard if guard else contextlib.nullcontext():
                binding = await helper.restore_core(common, parent, previous, out)
            restored = p.read(out / 'status.json')
            p.need(restored['complete'] and restored['correctness']['passed']
                   and restored['clock_restore_complete'] and not restored['errors'], 'retained restart failed')
            status.update(complete=True, binding=p.ref(out / 'binding.json'), restoration=p.ref(out / 'status.json'))
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        if guard is not None:
            p.save(out.parent / (out.name + '-ports.json'), guard.state)
        status.update(node_lease_held=False, finished_s=time.time())
        p.save(owner_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = p.checked(p.ref(args.spec))
    terminal(spec)
    if args.run:
        asyncio.run(run(spec, args.out))
    else:
        print(json.dumps(dict(passed=True, hardware_actions=False, node=spec['node'])))
