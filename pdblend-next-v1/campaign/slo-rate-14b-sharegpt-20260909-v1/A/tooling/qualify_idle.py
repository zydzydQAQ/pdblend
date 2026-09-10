"""Qualify the existing P12 idle-domain recovery on both migration-B native engines."""
import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import bootstrap as b
import power_selftest as p
import qualify_fixed as q
import verify_fixed


async def execute(spec, out, state):
    import aiohttp
    from aiohttp import web
    from ecopadg.serving.campaign import node_lease
    from ecopadg.serving.runtime import Controller
    from capacity_backend import TransitionMeter
    from meter_evidence import install
    from stream import NaturalStream
    prior = verify_fixed.verify(spec['previous_qualification'])
    previous_binding = b.checked(prior['binding'])
    common = b.load(spec['common_executor']['path'], 'migrationB_idle_common')
    with node_lease():
        p.actual_identity()
        state.update(node_lease_held=True, started_s=time.time())
        out.mkdir(parents=True)
        configs = {}
        binding = copy.deepcopy(previous_binding)
        for dataset, path in previous_binding['configs'].items():
            config = p.read(path)
            config['idle_domain_reacquire_v1'] = True
            config['idle_domain_reacquire_timeout_s'] = spec['idle_timeout_s']
            config['journal'] = str(out / 'unused.jsonl')
            target = out / 'configs' / (dataset + '.json')
            b.save(target, config)
            configs[dataset] = str(target)
            binding['files'][str(target)] = p.sha(target)
        binding.update(configs=configs)
        binding['files'].update(prior['files'])
        binding['files'].update(spec['files'])
        b.save(out / 'binding.json', binding)
        state['binding'] = p.ref(out / 'binding.json')
        b.save(out / 'status.json', state)
        finish = install(out / 'isolated-samplers', p.ref(p.HOST / 'manifest.json'), p.ref(p.ADAPTER), p.ref(p.HOOKS))
        meter = await TransitionMeter(out / 'power').start()
        async with aiohttp.ClientSession(trust_env=False) as session:
            try:
                b.save(out / 'identity.before.json', await common.identity(session, binding))
                references = b.checked(b.checked(spec['previous_qualification'])['status'])['reference_cases']
                for instance in binding['instances']:
                    native = await common.wait_idle(session, instance)
                    await common.resume(session, instance, 2048)
                    cfg = copy.deepcopy(p.read(configs['sharegpt']))
                    cfg.update(instances=[instance], port=34960, prepare_peers=False,
                               journal=str(out / (instance['id'] + '.control.jsonl')))
                    b.save(out / (instance['id'] + '.test-config.json'), cfg)
                    controller = Controller(cfg)
                    runner = web.AppRunner(controller.application())
                    stream = NaturalStream()
                    stream.session, stream.args, stream.issued = session, SimpleNamespace(port=34960, seed=0), set()
                    stream.stream_journal = (out / (instance['id'] + '.stream.jsonl')).open('x')
                    stream.result_journal = (out / (instance['id'] + '.requests.jsonl')).open('x')
                    expected = next(r['request']['output_token_ids'] for r in references
                                    if r['instance_id'] == instance['id'] and r['input_length'] == 128)
                    try:
                        await runner.setup()
                        await web.TCPSite(runner, '127.0.0.1', 34960).start()
                        for cycle in range(3):
                            if cycle:
                                await common.wait_idle(session, instance)
                                # Exercise the natural park/reset transition with its original .5s grace.
                                await asyncio.sleep(3)
                                clock = await asyncio.to_thread(controller.backend.clocks.hardware.current_freq, instance['gpus'][0])
                                assert clock > 2115 and instance['id'] in controller.backend.parked
                                assert controller.backend.clocks.applied.get(instance['gpus'][0]) is None
                                native = await common.wait_idle(session, instance)
                            else:
                                clock = await asyncio.to_thread(controller.backend.clocks.hardware.current_freq, instance['gpus'][0])
                            state.update(phase='idle_probe', current=dict(instance_id=instance['id'], cycle=cycle))
                            b.save(out / 'status.json', state)
                            row = await stream.request(128, 64)
                            q.stream_check(row, 128, expected)
                            assert row['token_received_s'][0] - row['dispatch_s'] < cfg['slo_ttft_s']
                            after = await common.wait_idle(session, instance)
                            state['probes'].append(dict(instance_id=instance['id'], cycle=cycle,
                                native_before=native, pre_request_clock_mhz=clock, request=row, native_after=after))
                            b.save(out / 'status.json', state)
                        proof = await controller.finish_measurement(time.time() + 30)
                        assert proof['drain_complete']
                        state.setdefault('controller_drains', {})[instance['id']] = proof
                    finally:
                        await runner.cleanup()
                        stream.stream_journal.close()
                        stream.result_journal.close()
                        restored = await common.restore(session, instance)
                        assert restored['complete']
                        state.setdefault('restoration', {})[instance['id']] = restored
                state.update(passed=True, phase='qualified')
            finally:
                state['measurement'] = await meter.finish()
                finish()
                b.save(out / 'identity.after.json', await common.identity(session, binding))
                state['node_lease_held'] = False
                b.save(out / 'status.json', state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    spec = p.read(args.spec)
    assert all(p.sha(path) == digest for path, digest in spec['files'].items())
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True)))
        return
    assert not args.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    sys.path[:0] = [str(p.METER), str(p.HOST / 'src'), str(p.HOST), '/root/workspace/pdblend/.runtime-deps']
    state = dict(schema='migration-B-idle-domain-qualification-status-v1', node=p.NODE, model='14b',
                 spec=p.ref(args.spec), pid=os.getpid(), passed=False, node_lease_held=False, probes=[])
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(spec, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(passed=False, error=repr(exc), phase='stopped_failure')
        raise
    finally:
        if args.out.exists():
            state.update(finished_s=time.time(), node_lease_held=False)
            b.save(args.out / 'status.json', state)
    assert state['passed'] and state['measurement']['measurement_valid']
    b.save(args.out / 'qualified.json', dict(schema='migration-B-fixed14B-idle-qualification-v1',
        previous_qualification=spec['previous_qualification'], previous_validator=p.ref(p.HERE / 'verify_fixed.py'),
        binding=state['binding'], status=p.ref(args.out / 'status.json'), spec=p.ref(args.spec),
        source_files=spec['files'], files={str(path): p.sha(path) for path in args.out.rglob('*') if path.is_file()}))
    print(json.dumps(dict(qualification=p.ref(args.out / 'qualified.json'), passed=True)))


if __name__ == '__main__':
    main()
