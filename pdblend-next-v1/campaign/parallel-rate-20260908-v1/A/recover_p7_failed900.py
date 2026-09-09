"""Owned terminal recovery only; preserve the failed P7 measurement unchanged."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

A = Path(__file__).resolve().parent
R = A.parent
HOST = R / 'hosts/14b-capacity-p7'
sys.path[:0] = [str(HOST), str(HOST / 'src'), '/root/workspace/pdblend/.runtime-deps']
from capacity_executor import fixed, durable, require
from capacity_certificate import ref
from capacity_backend import TransitionMeter
from ecopadg.serving.backend import ClockOwner
from ecopadg.serving.campaign import node_lease
from ecopadg.measure.backends import PynvmlBackend

OUT = A / 'p7-failed900-recovery-001'
OLD = A / 'p7-qualification900-dynamic-001'


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def clocks():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,clocks.current.sm,clocks.applications.graphics,pstate,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)


async def recover():
    import aiohttp
    status = fixed(ref(OLD / 'status.json'))
    require(status['pid'] == 1645667 and not alive(status['pid']) and not status['complete']
            and status['cleanup_errors'] == ['controller cleanup: TimeoutError()'],
            'only the exact exited P7 failed cleanup is authorized here')
    spec = fixed(ref(A / 'p7-qualification900-inputs-001/spec.json'))
    require(ref(A / 'p7-qualification900-inputs-001/spec.json')['sha256'] ==
            'b8fe7d7822544cb797fd25e3949b628921ba46b19d90e36c49e8bb79cb2b13b1', 'actual failed spec changed')
    base = fixed(spec['original_binding'])
    inventory = fixed(ref(OLD / 'inventory.json'))
    require(inventory['complete'] and not inventory['transition_inflight']
            and inventory['active_instances'] == base['instances'], 'capacity rollback must have restored exact original2')
    s = importlib.util.spec_from_file_location('p7_terminal_recovery_common', R / 'common/execution-until-complete-v1/run.py')
    common = importlib.util.module_from_spec(s)
    s.loader.exec_module(common)
    state = dict(schema='A-P7-failed900-owned-terminal-recovery-v1', pid=os.getpid(), started_s=time.time(),
        complete=False, original_failure_unchanged=True, original_status=ref(OLD / 'status.json'),
        source=ref(__file__), node_lease_held=True, experiment_requests_sent=0)
    durable(OUT / 'status.json', state)
    meter = await TransitionMeter(OUT / 'raw').start()
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            before = await common.identity(session, base)
            original = fixed(ref(OLD / 'identity.before.json'))
            by_id = {x['container']['Id']: x for x in original}
            for row in before:
                prior = by_id[row['container']['Id']]
                require(row['provenance'] == prior['provenance']
                        and all(row['container']['State'][k] == prior['container']['State'][k] for k in ('Pid', 'StartedAt')),
                        'retained engines must have the exact old process/provenance')
            durable(OUT / 'identity.before.json', before)
            durable(OUT / 'clocks.before.json', dict(observed_s=time.time(), csv=clocks()))
            hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
            owner = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
            await asyncio.wait_for(owner.close(), 10)
            state['all_eight_clock_reset_complete'] = True
            replies = await asyncio.wait_for(asyncio.gather(*(common.resume(session, i, i['restore_budget_tokens']) for i in base['instances'])), 120)
            durable(OUT / 'native-resume.json', replies)
            after = await common.identity(session, base)
            durable(OUT / 'identity.after.json', after)
            durable(OUT / 'clocks.after.json', dict(observed_s=time.time(), csv=clocks()))
            state.update(complete=True, native_original_budget_restored=True, exact_original2=True)
    except BaseException as exc:
        state.update(error=repr(exc), complete=False)
        raise
    finally:
        measured = await meter.finish()
        state.update(measurement=measured['receipt'], measurement_valid=measured['measurement_valid'],
                     finished_s=time.time(), node_lease_held=False)
        if not measured['measurement_valid']:
            state['complete'] = False
        durable(OUT / 'status.json', state)
        require(state['complete'], 'fresh owned recovery did not complete')


if __name__ == '__main__':
    require(not OUT.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh output and own short lease required')
    with node_lease():
        OUT.mkdir()
        asyncio.run(recover())
