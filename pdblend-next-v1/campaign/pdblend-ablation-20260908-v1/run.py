"""Authorized ablation continuation; baseline supervisors always have priority."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import execution as e
import readiness

ROOT = Path(__file__).resolve().parent


def previous_baseline_binding(model):
    index = e.read(ROOT.parent / 'current-experiment.json')
    ref = index['binding']
    e.require(e.sha(ref['path']) == ref['sha256'], 'last baseline binding hash changed')
    b = e.read(ref['path'])
    e.require(b['model'] == model and b['hostname'] == socket.gethostname()
              and b['system'] in readiness.BASELINES, 'last deployment is not the bound baseline')
    return b, ref


def selected_cells(declarations, model):
    values = e.read(declarations / 'core.cells.json')['cells']
    cells = [c for c in values if c['model'] == model]
    e.require(len(cells) == (76 if model == '14b' else 18), 'wrong fixed model experiment count')
    # Keep every point's Full and ablation arms together; both repeat orders
    # are copied from the frozen declarations. Prioritize service-risk evidence.
    order = ([("sharegpt", 1.5), ("sharegpt", 2.0), ("alpaca", 9.0), ("longbench", 1.0),
              ("alpaca", .6), ("sharegpt", .4), ("longbench", .2),
              ("alpaca", 3.6), ("sharegpt", 1.0), ("longbench", .5)] if model == '14b'
             else [('alpaca', 9 if model == '7b' else 2.5),
                   ('sharegpt', 2 if model == '7b' else .6),
                   ('longbench', 1.5 if model == '7b' else .3)])
    rank = {key: i for i, key in enumerate(order)}
    return sorted(cells, key=lambda c: (rank[(c['dataset'], c['rate_rps'])], c['sequence']))


def verify_package(root=ROOT):
    manifest = root / 'execution-manifest.json'
    e.require(manifest.is_file(), 'CPU-reviewed execution manifest required')
    m = e.read(manifest)
    for path, digest in m['files'].items():
        p = Path(path) if Path(path).is_absolute() else root / path
        e.require(e.sha(p) == digest, 'execution package changed: ' + str(p))
    for path, digest in m['dependencies'].items():
        e.require(e.sha(path) == digest, 'measurement dependency changed: ' + path)
    return m


async def execute_once(args, observation, state):
    cells = selected_cells(args.declarations, args.model)
    for cell in cells:
        e.check_cell(cell)
    parent_path = cells[0]['source_binding']['path']
    parent = e.read(parent_path)
    common = e.load_common(parent['host_release'])
    from ecopadg.serving.campaign import node_lease
    e.require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'ablation must acquire a fresh node lease')
    with node_lease():
        locked = await asyncio.to_thread(readiness.inspect_readiness, args.model)
        e.require(locked['ready'], 'baseline readiness changed after acquiring lease: ' + str(locked['reasons']))
        verify_package()
        previous, previous_ref = previous_baseline_binding(args.model)
        attempt = args.attempt
        e.require(not attempt.exists(), 'attempt exists; no silent re-execution')
        attempt.mkdir(parents=True)
        e.write(attempt / 'baseline-readiness.json', locked)
        e.write(attempt / 'previous-baseline-binding.json', dict(reference=previous_ref, binding=previous))
        e.write(attempt / 'declaration-order.json', cells)
        state.update(phase='restoring_original_pdb', attempt=str(attempt), started_gpu_s=time.time())
        e.write(args.status, state)
        restored = await e.restore_core(common, parent, previous, attempt / 'restoration')
        state.update(phase='core_ablation', restored_binding=str(attempt / 'restoration/binding.json'))
        e.write(args.status, state)
        def update(run_state):
            combined = dict(run_state, model=args.model, phase='core_ablation',
                            baseline_first=True, fresh_node_lease_held=True,
                            start_cutoff_s=e.START_CUTOFF, pid=os.getpid())
            e.write(attempt / 'status.json', combined)
            state.update(current_cell=run_state.get('current_cell'),
                         completed=len(run_state['completed']), attempted=len(run_state['attempted']))
            e.write(args.status, state)
        result = await e.run_cells(common, restored, cells, attempt, ROOT / 'STOP', update)
        state.update(phase='core_finished' if result['complete'] else 'core_stopped',
                     core_complete=result['complete'], result=result,
                     selective_complete=False,
                     selective_status='requires_independent_calibration_and_kv_gate' if args.model == '14b' else 'not_declared_for_model')
        e.write(args.status, state)
        return result


async def supervise(args):
    verify_package()
    require_model = readiness.SETTINGS[args.model]['hostname']
    e.require(socket.gethostname() == require_model, 'run this watcher on its actual model node')
    e.require(not args.status.exists(), 'watch status already exists; choose a new observer path')
    state = dict(model=args.model, pid=os.getpid(), started_s=time.time(), phase='waiting_for_baselines',
                 complete=False, core_complete=False, selective_complete=False,
                 start_cutoff_s=e.START_CUTOFF, delivery_cutoff_s=e.DELIVERY_CUTOFF,
                 hardware_started=False, permission='explicit user implementation order, 2026-09-08')
    e.write(args.status, state)
    owner_task = asyncio.current_task()
    interrupted = False
    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            owner_task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    try:
        while time.time() < e.START_CUTOFF:
            if (ROOT / 'STOP').exists():
                state.update(phase='stopped_before_gpu', stop_reason='campaign_stop')
                break
            observation = await asyncio.to_thread(readiness.inspect_readiness, args.model)
            state.update(checked_s=time.time(), readiness=observation)
            e.write(args.status, state)
            if observation['ready']:
                if not args.run:
                    state['phase'] = 'ready_read_only'
                    break
                try:
                    result = await execute_once(args, observation, state)
                    state['hardware_started'] = True
                    break
                except BlockingIOError:
                    state['lease_busy'] = True
                    e.write(args.status, state)
            if not args.watch:
                break
            await asyncio.sleep(args.poll_s)
        else:
            state.update(phase='start_cutoff_reached', stop_reason='18:00_no_new_experiments')
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        restoration = args.attempt / 'restoration/status.json'
        if restoration.exists():
            state['hardware_started'] = e.read(restoration).get('measurement_start_s') is not None
        state.update(finished_s=time.time(), complete=state.get('core_complete', False)
                     and (args.model != '14b' or state.get('selective_complete', False)))
        e.write(args.status, state)
    print(json.dumps({k: v for k, v in state.items() if k not in ('readiness', 'result')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('14b', '7b', '32b'), required=True)
    p.add_argument('--declarations', type=Path, default=ROOT / 'declarations')
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--poll-s', type=float, default=30)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    e.require(5 <= args.poll_s <= 60, 'poll interval must be 5..60 seconds')
    lock_path = ROOT / ('watcher-' + args.model + '.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(supervise(args))


if __name__ == '__main__':
    main()
