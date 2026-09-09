"""Explicit B14B-to-retained-B32B handoff using the original measured restore."""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

R = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(value):
    assert sha(value['path']) == value['sha256'], value['path']
    return read(value['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def validate(spec):
    assert spec['schema'] == 'B-ascending-retained32B-restore-v1'
    assert spec['authorized'] is True and spec['node'] == 'B'
    assert spec['hostname'] == 'iZwz9i5bte3xkpmcoes3t2Z'
    for path, digest in spec['files'].items():
        assert sha(path) == digest, path
    parent, previous = checked(spec['target_binding']), checked(spec['previous_binding'])
    assert parent['model'] == '32b' and previous['model'] == '14b'
    assert parent['system'] == previous['system'] == 'pdblend'
    assert [(i['id'], i['tp'], i['gpus']) for i in parent['instances']] == [
        ('nextv3b0', 2, [0, 1]), ('nextv3b1', 2, [2, 3])]
    assert all(i['native_kind'] == 'v3' and i['service_budget_tokens'] == 8192
               and i['restore_budget_tokens'] == 8192 for i in parent['instances'])
    source = checked(spec['performance_release'])
    assert source['host_manifest']['sha256'] == '69688ede41eb7bd6503d4e8e5fd9ddca42931f7fd75a36c5f53af88b73bc28b2'
    config = checked(source['configs']['fixed2']['alpaca'])
    assert 'distributed-14b' not in config['profiles']
    assert config.get('max_service_frequency_mhz', 2520) == 2520
    profile = checked(source['profile_refs']['alpaca'])
    assert {(p['tp'], p['frequency_mhz']) for p in profile['points']} == {(2, 1500), (2, 2520)}
    for key in ('prior_pipeline_status', 'prior_runner_status'):
        prior = checked(spec[key])
        assert prior.get('finished_s') is not None and prior.get('node_lease_held') is False
    return parent, previous


async def execute(args, state):
    spec = read(args.spec)
    parent, previous = validate(spec)
    assert socket.gethostname() == spec['hostname']
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    for key in ('prior_pipeline_status', 'prior_runner_status'):
        assert not Path('/proc', str(checked(spec[key])['pid'])).exists(), 'prior measured owner is still live'
    entry = Path(spec['restore_executor']['path'])
    imported = importlib.util.spec_from_file_location('ascending_B_original_restore', entry)
    restore = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(restore)
    common = restore.load_common(Path(spec['new_host_manifest']['path']).parent)
    from ecopadg.serving.campaign import node_lease
    parent = copy.deepcopy(parent)
    parent.update(deadline_s=None, campaign_lifecycle='until_declared_complete_v1')
    with node_lease():
        state.update(phase='restoring_original_32b', node_lease_held=True)
        save(args.out / 'status.json', state)
        actual = await restore.restore_core(common, parent, previous, args.out / 'restoration')
        status = read(args.out / 'restoration/status.json')
        assert status['complete'] is True and status['correctness']['passed'] is True
        assert not status['errors'] and status['clock_restore_complete'] is True
        save(args.out / 'binding-restored.json', actual)
        state.update(phase='restored_32b_ordinary_passed', complete=True,
            binding=ref(args.out / 'binding-restored.json'),
            ordinary_qualification=ref(args.out / 'restoration/correctness/status.json'),
            restoration=ref(args.out / 'restoration/status.json'),
            previous_14b_containers_retained=True,
            serving_performance_started=False,
            frequency_and_cancel_gate_still_required=True)
    state['node_lease_held'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    validate(read(args.spec))
    if not args.run:
        print(json.dumps(dict(cpu_only=True, passed=True, no_hardware_actions=True)))
        return
    assert not args.out.exists(), 'immutable new restore attempt required'
    args.out.mkdir(parents=True)
    state = dict(schema='B-ascending-restore-status-v1', pid=os.getpid(), started_s=time.time(),
        complete=False, node_lease_held=False, spec=ref(args.spec), automatic_retry=False)
    async def controlled():
        task = asyncio.current_task()
        requested = False
        def stop():
            nonlocal requested
            if not requested:
                requested = True
                state['stop_requested'] = True
                task.cancel()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        await execute(args, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
