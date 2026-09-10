"""Qualify every candidate TP1 shape and native cancellation on migration-B engines."""
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


def clock_window(samples, gpus, target, start, end):
    rows = [(t, f) for t, f in samples if start <= t <= end]
    assert end > start and len(rows) >= 2
    assert rows[0][0] - start <= .25 and end - rows[-1][0] <= .25
    assert max(y[0] - x[0] for x, y in zip(rows, rows[1:])) <= .25
    assert all(len(f) == 8 and all(abs(f[g] - target) <= 15 for g in gpus) for _, f in rows)
    return dict(passed=True, gpus=gpus, frequency_mhz=target, start_s=start, end_s=end, samples=len(rows))


def shapes(profile):
    return sorted({(v['frequency_mhz'], v['input_tokens'], v['batch']) for v in profile['points']
                   if v['tp'] == 1 and v['role'] == 'mixed'})


def validate(spec):
    assert spec['schema'] == 'migration-B-fixed14B-qualification-spec-v1'
    assert all(p.sha(path) == digest for path, digest in spec['files'].items())
    import restore
    assert restore.audit(spec['bootstrap'])['passed']
    bootstrap = b.checked(spec['bootstrap'])
    assert bootstrap['complete'] and bootstrap['ordinary_passed'] and not bootstrap.get('error')
    assert bootstrap['setup_measurement']['measurement_valid']
    assert bootstrap['hostname'] == p.EXPECTED_HOSTNAME and not bootstrap['node_lease_held']
    original = copy.deepcopy(b.checked(spec['source_config_template']))
    original.update(slo_ttft_s=5.,slo_tpot_s=.15)
    assert b.checked(spec['config_templates']['sharegpt']) == original
    assert spec['authorized_slo'] == dict(ttft_s=5.,tpot_s=.15)
    profile = b.checked(spec['profile'])
    assert [list(v) for v in shapes(profile)] == spec['shapes'] and len(spec['shapes']) == 41 and sorted({s[0] for s in spec['shapes']}) == [900, 1500, 2100]
    assert spec['old_node_qualifications_inherited'] is False
    return bootstrap


def stream_check(row, length, expected):
    assert row['success'] and row['http_status'] == 200 and row['done_marker'] and not row.get('error')
    assert row['prompt_token_ids'] == ([9707, 1879, 13] * (length // 3 + 1))[:length]
    assert row['output_token_ids'] == expected and len(expected) == 64
    assert row['usage']['prompt_tokens'] == length and row['usage']['completion_tokens'] == 64
    assert len(row['token_received_s']) == 64 and row['token_received_s'] == sorted(row['token_received_s'])


def make_binding(spec, bootstrap, out, common):
    configs = {}
    files = dict(spec['files'])
    manifest = p.read(p.HOST / 'manifest.json')
    files.update({str(p.HOST / path): digest for path, digest in manifest['files'].items()})
    for dataset, reference in spec['config_templates'].items():
        cfg = copy.deepcopy(b.checked(reference))
        cfg['instances'] = copy.deepcopy(bootstrap['instances'])
        cfg['port'] = 34950
        cfg['journal'] = str(out / 'unused-controller.jsonl')
        cfg['host_source_release'] = cfg['controller_source_release'] = str(p.HOST)
        cfg['profiles'] = spec['profile']['path']
        # Actual per-dataset SLO is set by the unchanged fixed100 child from its exact declared row.
        target = out / 'configs' / (dataset + '.json')
        b.save(target, cfg)
        configs[dataset] = str(target)
        files[str(target)] = p.sha(target)
    model_manifest = b.checked(spec['model_manifest'])
    large = {}
    for item in model_manifest['files']:
        path = Path(model_manifest['model_root']) / item['name']
        assert path.stat().st_size == item['bytes']
        large[str(path)] = dict(sha256=item['sha256'], stat=common.stat_identity(path))
    return dict(schema=1, protocol_id=common.PROTOCOL, model='14b', system='pdblend',
                hostname=bootstrap['hostname'], deadline_s=None, campaign_lifecycle=common.CAMPAIGN_LIFECYCLE,
                host_release=str(p.HOST), output=str(out / 'future-results'), configs=configs,
                instances=bootstrap['instances'], files=files, large_inputs=large, window_s=100,
                seeds=[701], formal_eligible=False, new_physical_node=False, node=p.NODE, old_node_qualifications_inherited=False,
                independent_capacity_qualification_granted=False,
                inherited_profile_costs_retained_as_candidates=True)


async def execute(spec, out, state):
    bootstrap = validate(spec)
    common = b.load(spec['common_executor']['path'], 'migrationB_fixed_qualification_common')
    stream_module = b.load(spec['stream']['path'], 'migrationB_fixed_natural_stream')
    from ecopadg.serving.campaign import node_lease
    from ecopadg.serving.backend import ClockOwner
    from capacity_backend import TransitionMeter, PinnedDockerBackend
    from meter_evidence import install
    import aiohttp
    with node_lease():
        p.actual_identity()
        out.mkdir(parents=True)
        binding = make_binding(spec, bootstrap, out, common)
        b.save(out / 'binding.json', binding)
        state.update(binding=p.ref(out / 'binding.json'), node_lease_held=True, started_s=time.time())
        b.save(out / 'status.json', state)
        common.validate_binding(binding)
        finish = install(out / 'isolated-samplers', p.ref(p.HOST / 'manifest.json'), p.ref(p.ADAPTER), p.ref(p.HOOKS))
        meter = owner = None
        streams = []
        async with aiohttp.ClientSession(trust_env=False) as session:
            try:
                b.save(out / 'identity.before.json', await common.identity(session, binding))
                meter = await TransitionMeter(out / 'power').start()
                owner = await asyncio.to_thread(ClockOwner, meter.sampler.backend, tuple(range(8)))
                for instance in binding['instances']:
                    stream = stream_module.NaturalStream()
                    stream.session = session
                    stream.args = SimpleNamespace(port=instance['port'], seed=0)
                    stream.issued = set()
                    stream.stream_journal = (out / (instance['id'] + '.stream.jsonl')).open('x')
                    stream.result_journal = (out / (instance['id'] + '.requests.jsonl')).open('x')
                    streams.append((instance, stream))
                    await common.resume(session, instance, 2048)
                    await owner.set(tuple(instance['gpus']), 2100, verify_rise=False)
                    refs = {}
                    for length in sorted({v[1] for v in spec['shapes']}):
                        row = await stream.request(length, 64)
                        expected = row['output_token_ids']
                        stream_check(row, length, expected)
                        old = next((x['response']['token_ids'] for x in p.read(bootstrap['ordinary']['path'])
                                    if x['instance_id'] == instance['id'] and x['prompt_length'] == length), None)
                        if old is not None:
                            assert expected == old
                        refs[length] = expected
                        state['reference_cases'].append(dict(instance_id=instance['id'], input_length=length, request=row))
                        await common.wait_idle(session, instance)
                    b.save(out / 'status.json', state)
                    current_frequency = None
                    for frequency, length, batch in spec['shapes']:
                        if frequency != current_frequency:
                            await owner.set(tuple(instance['gpus']), frequency, verify_rise=False)
                            warmup = await stream.request(128, 64)
                            stream_check(warmup, 128, refs[128])
                            await common.wait_idle(session, instance)
                            current_frequency = frequency
                        state.update(phase='profile_shapes', current_case=dict(instance_id=instance['id'],
                                     frequency=frequency, input_length=length, batch=batch))
                        b.save(out / 'status.json', state)
                        rows = await asyncio.gather(*(stream.request(length, 64) for _ in range(batch)))
                        for row in rows:
                            stream_check(row, length, refs[length])
                        native = await common.wait_idle(session, instance)
                        loaded = [clock_window(meter.sampler.frequency_samples, instance['gpus'], frequency,
                                               row['token_received_s'][0], row['token_received_s'][-1]) for row in rows]
                        state['shape_cases'].append(dict(instance_id=instance['id'], frequency=frequency,
                            input_length=length, batch=batch, requests=rows, native_after=native, loaded_clocks=loaded))
                        b.save(out / 'status.json', state)
                    events = []
                    def event(kind, **values):
                        events.append(dict(kind=kind, **values))
                        b.save(out / (instance['id'] + '.cancel-events.json'), events)
                    proxy = SimpleNamespace(controller=SimpleNamespace(session=session), inventory=SimpleNamespace(event=event),
                                            oracles=b.checked(spec['numerical_reference']))
                    async def request(target, path, payload=None, *, limit, cap=35):
                        return await PinnedDockerBackend.request(proxy, target, path, payload, limit=limit, cap=cap)
                    proxy.request = request
                    evidence = await PinnedDockerBackend.verify_cancel(proxy, instance, time.time() + 35)
                    state['cancellations'].append(dict(instance_id=instance['id'], evidence=evidence))
                state.update(passed=True, phase='qualified')
            finally:
                for instance, stream in streams:
                    for rid in stream.issued:
                        try:
                            await common.http(session, instance, '/cancel', dict(request_id=rid))
                        except BaseException as exc:
                            state['cleanup_errors'].append('own cancel: ' + repr(exc))
                    stream.stream_journal.close()
                    stream.result_journal.close()
                restored = await asyncio.gather(*(common.restore(session, i) for i in binding['instances']), return_exceptions=True)
                state['restoration'] = {i['id']: dict(error=repr(x)) if isinstance(x, BaseException) else x
                                        for i, x in zip(binding['instances'], restored)}
                if owner:
                    try:
                        await owner.close()
                        state['clock_restore_complete'] = True
                    except BaseException as exc:
                        state['cleanup_errors'].append('clock restore: ' + repr(exc))
                if meter:
                    state['measurement'] = await meter.finish()
                finish()
                b.save(out / 'identity.after.json', await common.identity(session, binding))
                state['passed'] = bool(state['passed'] and not state['cleanup_errors'] and state.get('clock_restore_complete')
                                       and all(x.get('complete') for x in state['restoration'].values())
                                       and state.get('measurement', {}).get('measurement_valid'))
        state['node_lease_held'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=p.HERE / 'fixed-qualification-spec.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = p.read(args.spec)
    validate(spec)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, shape_count=82, GPU_work_started=False)))
        return
    assert args.out and not args.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    sys.path[:0] = [str(p.METER), str(p.HOST / 'src'), str(p.HOST), '/root/workspace/pdblend/.runtime-deps']
    state = dict(schema='migration-B-fixed14B-qualification-status-v1', node=p.NODE, model='14b',
                 spec=p.ref(args.spec), pid=os.getpid(), passed=False, node_lease_held=False,
                 shape_cases=[], reference_cases=[], cancellations=[], cleanup_errors=[])
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
    assert state['passed']
    result = dict(schema='migration-B-fixed14B-qualification-v1', status=p.ref(args.out / 'status.json'),
                  binding=state['binding'], spec=p.ref(args.spec), source_files=spec['files'],
                  files={str(path): p.sha(path) for path in args.out.rglob('*') if path.is_file()},
                  datasets=['sharegpt'], dynamic_capacity_qualification=False)
    b.save(args.out / 'qualified.json', result)
    print(json.dumps(dict(qualification=p.ref(args.out / 'qualified.json'), passed=True)))


if __name__ == '__main__':
    main()
