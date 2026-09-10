"""Cold-start the unchanged 14B native implementation on the new physical node."""
import argparse
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import power_selftest as p


def checked(reference):
    assert p.sha(reference['path']) == reference['sha256'], 'changed source: ' + reference['path']
    return p.read(reference['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(spec):
    from inputs import validate_bootstrap
    return validate_bootstrap(spec)


async def execute(spec, out, state):
    common = load(spec['common_executor']['path'], 'newA_bootstrap_common')
    from ecopadg.serving.campaign import node_lease
    from meter_evidence import install
    from capacity_backend import TransitionMeter
    import aiohttp
    owned = []
    with node_lease():
        state['identity'] = p.actual_identity()
        assert not (await common.command('docker', 'ps', '--format', '{{.ID}}')).strip(), 'another deployment owns this node'
        assert not (await common.command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits')).strip(), 'another GPU process owns this node'
        actual_image = (await common.command('docker', 'image', 'inspect', spec['image'], '--format', '{{.Id}}')).strip()
        assert actual_image == spec['image']
        for instance in spec['instances']:
            for port in (instance['port'], *range(instance['kv_port'], instance['kv_port'] + 16)):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', port))
        out.mkdir(parents=True)
        state.update(node_lease_held=True, started_s=time.time(), phase='cold_start')
        save(out / 'status.json', state)
        finish = install(out / 'isolated-samplers', p.ref(p.HOST / 'manifest.json'), p.ref(p.ADAPTER), p.ref(p.HOOKS))
        meter = await TransitionMeter(out / 'setup-power').start()
        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                for plan in spec['instances']:
                    instance = copy.deepcopy(plan)
                    cfg = copy.deepcopy(checked(spec['engine_template']))
                    cfg.update(id=instance['id'], port=instance['port'], kv_port=instance['kv_port'],
                               peers={r['id']: dict(host='127.0.0.1', tp=1, kv_port=r['kv_port']) for r in spec['instances']},
                               runtime_dir=str(out / 'native' / instance['id']), retained_weights=None,
                               scheduler_budget=dict(schema_version=1, max_num_batched_tokens=2048, max_num_seqs=32))
                    cfg_path = out / ('engine-' + instance['id'] + '.json')
                    save(cfg_path, cfg)
                    env = dict(spec['environment'], CUDA_VISIBLE_DEVICES=str(instance['gpus'][0]))
                    argv = ['docker', 'run', '-d', '--name', instance['container_name'], '--gpus', 'all',
                            '--network', 'host', '--ipc', 'host', '--security-opt', 'label=disable',
                            '--label', 'pdblend.slo14.owner=A']
                    for key, value in env.items():
                        argv += ['-e', key + '=' + value]
                    argv += ['-v', '/root/workspace:/root/workspace', '-v', '/root/workspace/models:/models:ro',
                             spec['image'], 'python3', '-m', spec['engine_module'], '--config', str(cfg_path)]
                    cid = (await common.command(*argv)).strip()
                    owned.append(dict(id=instance['id'], container_id=cid, container_name=instance['container_name']))
                    save(out / 'owned-containers.json', owned)
                    actual = json.loads(await common.command('docker', 'inspect', cid))[0]
                    instance.update(container=dict(id=cid, name=instance['container_name'], image=actual['Image'],
                                                   StartedAt=actual['State']['StartedAt']), host_pid=actual['State']['Pid'])
                    state['instances'].append(instance)
                    save(out / 'status.json', state)
                for instance in state['instances']:
                    deadline = time.monotonic() + 300
                    while True:
                        try:
                            raw = await common.http(session, instance, '/runtime', timeout=1)
                            common.idle(raw, instance)
                            assert raw.get('transport_healthy') is True and raw.get('total_kv_tokens', 0) > 0
                            break
                        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, AssertionError):
                            assert time.monotonic() < deadline, 'native engine readiness timeout'
                            await asyncio.sleep(.25)
                    provenance = await common.http(session, instance, '/provenance')
                    expected = dict(spec['expected_provenance'], instance_id=instance['id'],
                                    cuda_visible_devices=str(instance['gpus'][0]))
                    assert all(provenance.get(key) == value for key, value in expected.items()), 'native source/provenance differs'
                    instance['provenance'] = provenance
                    save(out / 'status.json', state)
                oracle = checked(spec['numerical_reference'])
                replies = []
                for instance in state['instances']:
                    for case in oracle['cases']:
                        rid = 'newA-ordinary-' + instance['id'] + '-' + str(case['prompt_length'])
                        request = dict(prompt=case['prompt'], max_tokens=case['max_tokens'], ignore_eos=True,
                                       temperature=0, stream=False, request_id=rid)
                        reply = await common.http(session, instance, '/v1/completions', request, timeout=90)
                        replies.append(dict(instance_id=instance['id'], request_id=rid, prompt_length=case['prompt_length'],
                                            expected_token_ids=case['token_ids'], response=reply))
                        save(out / 'ordinary-replies.json', replies)
                        assert reply.get('token_ids') == case['token_ids'], 'fresh ordinary output differs from numerical reference'
                        assert reply['usage']['prompt_tokens'] == case['prompt_length']
                        assert reply['usage']['completion_tokens'] == case['max_tokens']
                        await common.wait_idle(session, instance)
                restorations = await asyncio.gather(*(common.restore(session, instance) for instance in state['instances']))
                save(out / 'restoration.json', restorations)
                assert all(row['complete'] for row in restorations)
                state.update(ordinary_passed=True, complete=True, phase='native_ready',
                             ordinary=p.ref(out / 'ordinary-replies.json'), restoration=p.ref(out / 'restoration.json'),
                             frequency_profile_capacity_qualification_pending=True,
                             serving_measurements_started=False)
        except BaseException as exc:
            state.update(error=repr(exc), complete=False, phase='stopped_failure')
            failures = []
            for instance in owned:
                try:
                    await common.command('docker', 'stop', '--time', '10', instance['container_id'])
                except BaseException as cleanup:
                    failures.append(repr(cleanup))
            state['cleanup_errors'] = failures
            raise
        finally:
            try:
                state['setup_measurement'] = await meter.finish()
                finish()
            except BaseException as exc:
                state.update(measurement_error=repr(exc), complete=False)
            state.update(finished_s=time.time(), node_lease_held=False)
            save(out / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=p.HERE / 'bootstrap-spec.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = validate(p.read(args.spec))
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, GPU_work_started=False)))
        return
    assert args.out and not args.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    sys.path[:0] = [str(p.METER), str(p.HOST / 'src'), str(p.HOST), '/root/workspace/pdblend/.runtime-deps']
    state = dict(schema='new-A-native14B-bootstrap-status-v1', node=p.NODE, hostname=socket.gethostname(),
                 spec=p.ref(args.spec), pid=os.getpid(), instances=[], complete=False, node_lease_held=False)
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(spec, args.out, state)
    asyncio.run(controlled())
    assert state['complete'] and state['setup_measurement']['measurement_valid']
    print(json.dumps(dict(status=p.ref(args.out / 'status.json'), native_ready=True,
                          formal_qualification_granted=False)))


if __name__ == '__main__':
    main()
