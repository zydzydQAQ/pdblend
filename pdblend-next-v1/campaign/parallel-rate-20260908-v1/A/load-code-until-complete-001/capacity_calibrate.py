"""Run three explicitly declared cold add/stop cycles under a fresh node lease.

This produces raw transition measurements. It does not certify layout capacity,
matched-load energy savings, warm cache recovery or production readiness.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_executor import durable, fixed, require, sha


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


async def run(spec, out):
    import aiohttp
    from ecopadg.serving.runtime import Controller
    from ecopadg.serving.backend import HttpEngineBackend, ClockOwner
    from ecopadg.measure.backends import PynvmlBackend
    from capacity_runtime import CapacityService
    original = fixed(spec['original_binding'])
    binding = fixed(spec['capacity_binding'])
    for p, h in original['files'].items():
        require(sha(p) == h, 'original current source changed: '+p)
    for p, value in original['large_inputs'].items():
        stat = Path(p).stat()
        wanted = value['stat']
        require(stat.st_size == wanted['size'] and stat.st_mtime_ns == wanted['mtime_ns']
                and stat.st_ino == wanted['inode'] and stat.st_dev == wanted['device'],
                'previously hashed model/trace identity changed')
    config = dict(strategy='pdblend-joint', instances=original['instances'], allow_pd=False, dvfs=False,
        park_idle=False, slow_topology=False, dynamic_pools=False, profiles=spec['profiles']['path'],
        journal=str(out/'unused-controller-journal.jsonl'), node_gpus=list(range(8)),
        capacity_inventory_path=str(out/'inventory.json'), output_prior=spec['output_prior'])
    fixed_profile = spec['profiles']
    require(sha(fixed_profile['path']) == fixed_profile['sha256'], 'original profile changed')
    controller = Controller(config)
    state = dict(schema='capacity-cold-calibration-status-v1', pid=os.getpid(), started_s=time.time(),
                 phase='preflight', complete=False, production_ready=False,
                 automatic_retries=False, completed=[], failed=[], original_binding=spec['original_binding'])
    def update(**value):
        state.update(value, updated_s=time.time())
        durable(out/'status.json', state)
    update()
    task = asyncio.current_task()
    interrupted = False
    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    clocks = await asyncio.to_thread(ClockOwner, hardware, range(8))
    service = None
    session = aiohttp.ClientSession(trust_env=False)
    try:
        controller.session = session
        controller.backend = HttpEngineBackend(original['instances'], session, clocks)
        await controller.refresh()
        require(all(controller.backend.last[i['id']].get('active') == 0 for i in original['instances']),
                'original engines are not idle')
        # Preserve each exact retained original process throughout calibration.
        common = load(spec['common_executor']['path'], 'capacity_original_execution')
        require(sha(spec['common_executor']['path']) == spec['common_executor']['sha256'],
                'original identity verifier changed')
        durable(out/'identity.before.json', await common.identity(session, original))
        await clocks.set([g for i in original['instances'] for g in i['gpus']], 2520)
        unused = sorted(set(range(8))-{g for i in original['instances'] for g in i['gpus']})
        await clocks.park(unused, dict(clocks.epochs))
        service = CapacityService(controller, binding, require_calibration=False)
        for index, pair in enumerate(spec['cycles'], 1):
            require(spec['deadline_s'] is None and spec['campaign_lifecycle']=='until_declared_complete_v1',
                'total deadline was cancelled; preserve only local work/cleanup bounds')
            require(not (Path(spec['stop_path'])).exists(), 'STOP prevents successor calibration cycle')
            update(phase='cold_restore', current_cycle=index)
            result = await service.executor.calibrate('restore', tuple(spec['gpus']), declaration=pair['restore'])
            durable(out/f'cycle-{index}-restore.json', result)
            added = result['instance_id']
            update(phase='physical_stop', current_instance_id=added)
            result = await service.executor.calibrate('remove', tuple(spec['gpus']),
                                                     remove_id=added, declaration=pair['remove'])
            durable(out/f'cycle-{index}-remove.json', result)
            state['completed'].append(index)
            update()
        await service.finish_to_initial()
        durable(out/'identity.after.json', await common.identity(session, original))
        update(phase='measured_complete', complete=True,
               layout_capacity_calibrated=False, matched_load_savings_calibrated=False,
               warm_restore_calibrated=False, production_ready=False)
    except BaseException as exc:
        state['failed'].append(repr(exc))
        update(phase='needs_attention', error=repr(exc))
        if service is not None:
            try:
                await service.quiesce()
                await service.finish_to_initial()
            except BaseException as cleanup:
                update(cleanup_error=repr(cleanup))
        raise
    finally:
        await session.close()
        await clocks.close()
        await controller.planning_executor.close()
        update(clock_restore_complete=True, finished_s=time.time())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    spec = fixed(dict(path=str(args.spec), sha256=args.spec_sha256))
    require(spec['schema'] == 'capacity-cold-calibration-spec-v1' and spec['authorized'] is True
            and spec['automatic_retries'] is False and len(spec['cycles']) == 3,
            'exact three-cycle development calibration declaration required')
    require(spec['files'] and all(sha(p) == h for p,h in spec['files'].items()), 'physical package changed')
    require(not args.out.exists(), 'new physical calibration output required')
    original = fixed(spec['original_binding'])
    host = Path(original['host_release'])
    sys.path[:0] = [str(host/'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    os.environ['PYTHONPATH'] = ':'.join(sys.path[:3])
    if not args.run:
        print(json.dumps(dict(cpu_only=True, files_verified=len(spec['files']), hardware_actions=False)))
        return
    from ecopadg.serving.campaign import node_lease
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'calibration requires its own fresh node lease')
    with node_lease():
        args.out.mkdir(parents=True)
        asyncio.run(run(spec, args.out))


if __name__ == '__main__':
    main()
