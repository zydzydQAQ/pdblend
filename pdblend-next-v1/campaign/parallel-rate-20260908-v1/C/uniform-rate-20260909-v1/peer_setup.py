"""Reapply the exact historical in-memory C peer registration after restart."""
import argparse
import asyncio
import os
from pathlib import Path
import signal
import socket
import time

import support as p


async def execute(base, previous, out, state):
    verifier = p.load(p.HERE / 'verify_qualification.py', 'C_peer_base_qualification')
    q = verifier.verify(base)
    binding = p.checked(q['binding'])
    prior = p.checked(previous)
    p.need(prior['finished_s'] and not prior['node_lease_held'] and not p.active_owner(prior), 'old owner active')
    p.need(len(prior['observed_checkpoints']) == 1 and not prior['observations'], 'unexpected earlier performance')
    checkpoint = p.checked(prior['observed_checkpoints'][0])
    operation = Path(checkpoint['receipt']['path']).parent
    receipt = p.checked(checkpoint['receipt'])
    p.need(receipt['child_stopped'] and receipt['clock_restore_complete'] and not receipt['outer_cleanup_errors']
           and all(x['complete'] for x in receipt['restoration'].values()), 'failed setup not cleaned')
    p.need('nextc7 /prepare-peers: 400 unknown or local peer' in (operation / 'child.log').read_text(),
           'unknown startup failure')
    for path in (operation / 'dispatch.jsonl', operation.parents[1] / 'cells' / checkpoint['row']['cell_id'] / 'bench.csv'):
        p.need(not path.exists() or not path.read_bytes().strip(), 'request evidence exists; cannot classify pre-arrival setup')
    p.need(not (operation / 'actual-epoch.json').exists(), 'arrival epoch already selected')
    state['failure_diagnosis'] = dict(checkpoint=prior['observed_checkpoints'][0],
        classification='lost_historical_in_memory_peer_after_restart_before_any_arrival',
        no_performance_request_issued=True, setup_energy_j=receipt['full_operation_energy_j'])
    original = p.ref(p.ROOT.parent / 'C7B-retained-peer-repair-v1/actual-001/spec.json')
    plan = p.checked(original)
    source = next(i for i in binding['instances'] if i['id'] == plan['register']['instance'])
    target = next(i for i in binding['instances'] if i['id'] == plan['register']['body']['id'])
    p.need(plan['register']['body']['peer'] == dict(host='127.0.0.1', tp=target['tp'], kv_port=target['kv_port']),
           'historical peer does not name current exact target')
    p.need(socket.gethostname() == binding['hostname'] and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'wrong owner context')
    helper = p.load(p.ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'C_peer_original_operation')
    common = helper.load_common(binding['host_release'])
    p.load(Path(binding['host_release']) / 'capacity_executor.py', 'capacity_executor')
    meter_module = p.load(Path(binding['host_release']) / 'capacity_backend.py', 'C_peer_original_meter')
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        async with aiohttp.ClientSession(trust_env=False) as session:
            p.save(out / 'identity.before.json', await common.identity(session, binding))
            meter = await meter_module.TransitionMeter(out / 'power').start()
            try:
                state['registered'] = await common.http(session, source, '/register-peer', plan['register']['body'])
                p.need(state['registered'] == dict(id=target['id'], registered=[dict(id=target['id'], rank=0)]),
                       'native registration acknowledgement differs')
                state['prepared'] = await common.http(session, source, '/prepare-peers', dict(peers=[target['id']]))
                p.need(state['prepared'] == dict(ready=[dict(peers=[target['id']], rank=0)]),
                       'native peer readiness acknowledgement differs')
                state['complete'] = True
            finally:
                restored = await asyncio.gather(*(common.restore(session, i) for i in binding['instances']), return_exceptions=True)
                state['restoration'] = {i['id']: dict(complete=False, error=repr(x)) if isinstance(x, BaseException) else x
                                        for i, x in zip(binding['instances'], restored)}
                state['measurement'] = await meter.finish()
                p.save(out / 'identity.after.json', await common.identity(session, binding))
                state['complete'] = bool(state['complete'] and all(x['complete'] for x in state['restoration'].values())
                                         and state['measurement']['measurement_valid'])
        state['node_lease_held'] = False
    p.need(state['complete'], 'peer setup failed')
    state.update(binding=q['binding'], base_qualification=base, historical_peer_plan=original,
                 same_policy_and_host_source=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    p.need(not args.out.exists(), 'fresh explicit setup attempt required')
    args.out.mkdir(parents=True)
    state = dict(schema='C-restarted-retained-peer-setup-v1', pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        complete=False, node_lease_held=False, serving_measurements_started=False)
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(p.ref(args.base), p.ref(args.previous), args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
