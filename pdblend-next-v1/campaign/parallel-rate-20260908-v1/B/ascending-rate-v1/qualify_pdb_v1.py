"""Retained B32B source/profile refresh: original full-output and cancel primitives."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
REPO = R.parents[1]


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


p = load(HERE / 'restore_pdb_v1.py', 'ascending_B_restore_tools')


def clock_window(samples, gpus, target, start, end):
    assert end > start
    rows = [(t, f) for t, f in samples if start <= t <= end]
    assert len(rows) >= 2, 'missing loaded SM samples'
    assert rows[0][0] - start <= .25 and end - rows[-1][0] <= .25
    assert max(b[0] - a[0] for a, b in zip(rows, rows[1:])) <= .25
    assert all(len(f) == 8 and all(type(x) in (int, float) for x in f) for _, f in rows)
    assert all(abs(f[g] - target) <= 15 for _, f in rows for g in gpus), 'loaded TP member outside original frequency domain'
    return dict(gpus=gpus, target_mhz=target, start_s=start, end_s=end,
                samples=len(rows), minimum_mhz=min(f[g] for _, f in rows for g in gpus),
                maximum_mhz=max(f[g] for _, f in rows for g in gpus), passed=True)


def validate(spec):
    assert spec['schema'] == 'B32B-ascending-frequency-cancel-qualification-v1'
    for path, digest in spec['files'].items():
        assert p.sha(path) == digest, path
    restore = p.checked(spec['restoration'])
    assert restore['complete'] and not restore['node_lease_held']
    b = p.checked(spec['binding'])
    assert b['model'] == '32b' and b['system'] == 'pdblend'
    assert [(i['tp'], i['gpus'], i['service_budget_tokens']) for i in b['instances']] == [(2, [0, 1], 8192), (2, [2, 3], 8192)]
    assert spec['frequencies_mhz'] == [1500, 2520] and spec['input_tokens'] == 128 and spec['output_tokens'] == 64
    assert spec['profile']['sha256'] == '8ac6b01af2696e7cddac72c0aca97994747a082325df79e6c1347eaf419d3b31'
    profile = p.checked(spec['profile'])
    assert {(x['tp'], x['frequency_mhz']) for x in profile['points']} == {(2, 1500), (2, 2520)}
    assert p.checked(spec['ordinary'])['passed']
    return b


async def execute(spec, out, state):
    binding = validate(spec)
    assert socket.gethostname() == binding['hostname'] and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    assert not Path('/proc', str(p.checked(spec['restoration'])['pid'])).exists(), 'restore owner still running'
    helper = load(R / 'B/baseline-return-after-external-source-v1/execution.py', 'ascending_B_original_operation')
    common = helper.load_common(Path(spec['host_manifest']['path']).parent)
    # Explicitly load the original bounded cancellation and meter modules;
    # the P12 serving source is not substituted into these evidence primitives.
    load(Path(spec['capacity_executor']['path']), 'capacity_executor')
    cb = load(Path(spec['capacity_backend']['path']), 'ascending_B_original_cancel')
    sm = load(Path(spec['stream']['path']), 'ascending_B_original_stream')
    import aiohttp
    from ecopadg.serving.backend import ClockOwner
    from ecopadg.serving.campaign import node_lease
    ordinary = p.checked(spec['ordinary'])
    expected = [x['response']['token_ids'] for x in ordinary['replies'] if x['prompt_length'] == 128]
    assert expected and all(x == expected[0] for x in expected)
    sampler = None
    owner = None
    streams = []
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        async with aiohttp.ClientSession(trust_env=False) as session:
            try:
                common.validate_binding(binding)
                p.save(out / 'identity.before.json', await common.identity(session, binding))
                sampler = await cb.TransitionMeter(out / 'power').start()
                owner = await asyncio.to_thread(ClockOwner, sampler.sampler.backend, tuple(range(8)))
                for i in binding['instances']:
                    stream = sm.NaturalStream()
                    stream.session = session
                    stream.args = SimpleNamespace(port=i['port'], seed=0)
                    stream.issued = set()
                    stream.stream_journal = (out / (i['id'] + '.stream.jsonl')).open('x')
                    stream.result_journal = (out / (i['id'] + '.requests.jsonl')).open('x')
                    streams.append((i, stream))
                    for frequency in spec['frequencies_mhz']:
                        state['current'] = dict(instance=i['id'], frequency=frequency)
                        p.save(out / 'status.json', state)
                        control = await common.resume(session, i, i['service_budget_tokens'])
                        write = await owner.set(tuple(i['gpus']), frequency, verify_rise=False)
                        warmup = await stream.request(128, 64)
                        assert warmup['success'] and warmup['output_token_ids'] == expected[0], 'warmup output mismatch'
                        await common.wait_idle(session, i)
                        row = await stream.request(128, 64)
                        assert row['success'] and row['output_token_ids'] == expected[0]
                        assert row['usage']['prompt_tokens'] == 128 and row['usage']['completion_tokens'] == 64
                        native = await common.wait_idle(session, i)
                        clocks = clock_window(sampler.sampler.frequency_samples, i['gpus'], frequency,
                                              row['token_received_s'][0], row['token_received_s'][-1])
                        state['frequency_cases'].append(dict(instance_id=i['id'], control=control, clock_write=write,
                            warmup=warmup, request=row, native_after=native, loaded_clock=clocks))
                        p.save(out / 'status.json', state)
                    class Adapter:
                        controller = SimpleNamespace(session=session)
                        oracles = {'cases': [dict(prompt_length=128, prompt=([9707, 1879, 13] * 43)[:128])]}
                        class Inventory:
                            def event(self, event, **values):
                                with (out / 'cancel-events.jsonl').open('a') as f:
                                    f.write(json.dumps(dict(event=event, at_s=time.time(), **values)) + '\n')
                        inventory = Inventory()
                        async def request(self, instance, path, body=None, *, limit, cap=10):
                            remaining = min(cap, limit - time.time())
                            assert remaining > 0
                            return await common.http(session, instance, path, body, timeout=remaining)
                    cancellation = await cb.PinnedDockerBackend.verify_cancel(Adapter(), i, time.time() + 30)
                    assert cancellation['verified']
                    state['cancellations'].append(dict(instance_id=i['id'], evidence=cancellation))
                    p.save(out / 'status.json', state)
                state['passed'] = True
            finally:
                # Cancel only exact request IDs issued by this gate, then use
                # the original all-rank barrier and scheduler-budget restore.
                for i, stream in streams:
                    for rid in stream.issued:
                        try:
                            await common.http(session, i, '/cancel', dict(request_id=rid))
                        except BaseException as exc:
                            state['cleanup_errors'].append('owned cancel: ' + repr(exc))
                    stream.stream_journal.close()
                    stream.result_journal.close()
                restored = await asyncio.gather(*(common.restore(session, i) for i in binding['instances']), return_exceptions=True)
                state['restoration'] = {i['id']: dict(error=repr(v)) if isinstance(v, BaseException) else v
                                        for i, v in zip(binding['instances'], restored)}
                if owner:
                    try:
                        await asyncio.wait_for(owner.close(), 15)
                        state['clock_restore_complete'] = True
                    except BaseException as exc:
                        state['cleanup_errors'].append('clock restore: ' + repr(exc))
                if sampler:
                    state['measurement'] = await sampler.finish()
                try:
                    p.save(out / 'identity.after.json', await common.identity(session, binding))
                except BaseException as exc:
                    state['cleanup_errors'].append('identity after: ' + repr(exc))
                state['passed'] = bool(state['passed'] and not state['cleanup_errors'] and state.get('clock_restore_complete')
                    and all(v.get('complete') for v in state['restoration'].values())
                    and state.get('measurement', {}).get('measurement_valid'))
                p.save(out / 'status.json', state)
        assert state['passed'], 'qualification failed; original evidence retained'
    state['node_lease_held'] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    spec = p.read(args.spec)
    validate(spec)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True)))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(schema='B32B-ascending-frequency-cancel-result-v1', pid=os.getpid(), started_s=time.time(),
                 spec=p.ref(args.spec), passed=False, node_lease_held=False, frequency_cases=[], cancellations=[], cleanup_errors=[])
    async def controlled():
        task = asyncio.current_task()
        stop = False
        def cancel():
            nonlocal stop
            if not stop:
                stop = True
                task.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, cancel)
        await execute(spec, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(error=repr(exc), passed=False)
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
