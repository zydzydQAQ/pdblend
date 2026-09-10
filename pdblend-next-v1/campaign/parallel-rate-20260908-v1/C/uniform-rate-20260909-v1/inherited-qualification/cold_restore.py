"""C7B cold recovery of the exact retained P4 TP1 pair, with measured cleanup.

Default mode checks frozen files only. --run is required for hardware actions.
"""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HOSTNAME = 'iZwz9gfq11hx1sbob59yrgZ'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return read(reference['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(spec):
    assert spec['schema'] == 'C7B-ascending-cold-recovery-v1'
    assert spec['node'] == 'C' and spec['hostname'] == HOSTNAME
    for path, digest in spec['files'].items():
        assert sha(path) == digest, path
    target = checked(spec['target_binding'])
    assert target['hostname'] == HOSTNAME and target['model'] == '7b' and target['system'] == 'pdblend'
    assert [(i['id'], i['tp'], i['gpus'], i['native_kind']) for i in target['instances']] == [
        ('nextc7', 1, [7], 'legacy_sync_put'),
        ('pdbcap-12aee0f43ea3429bb4ca3f4d3f3570bb', 1, [6], 'legacy_sync_put')]
    original = checked(spec['original_release'])
    assert target['configs'] == {k:v['path'] for k,v in original['configs']['fixed2'].items()}
    for cfgpath in target['configs'].values():
        cfg = read(cfgpath)
        assert not cfg.get('allow_pd') and not cfg.get('dynamic_pools')
        assert all(cfg.get(flag, False) is False for flag in (
            'capacity_integration_v1', 'idle_domain_reacquire_v1', 'clock_failure_fresh_confirmation_v1'))
        assert 'idle_domain_reacquire_timeout_s' not in cfg
    off = checked(spec['off_source_equivalence'])
    assert off['passed'] and off['independently_recomputed']
    model = next(x for x in off['models'] if x['model'] == '7b')
    assert model['P12_manifest'] == spec['host_manifest']
    assert spec['frequencies_mhz'] == [900, 1500, 2100, 2520]
    return target


def container_equivalence(historical, actual, dns):
    # Runtime PID/timestamps necessarily change at restart; static launch and
    # namespace configuration must remain identical, except explicit empty DNS.
    for key in ('Id', 'Name', 'Image', 'Path', 'Args', 'Config', 'Mounts'):
        assert actual[key] == historical[key], 'retained container static field changed: ' + key
    assert actual['State']['Running'] is False and actual['State']['Pid'] == 0
    return dns.hostconfig_equivalence(historical['HostConfig'], actual['HostConfig'])


async def execute(spec, out, state):
    target = validate(spec)
    assert socket.gethostname() == HOSTNAME and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    helper = load(spec['restore_executor']['path'], 'C_ascending_original_restore')
    common = helper.load_common(target['host_release'])
    dns = load(spec['docker_equivalence']['path'], 'C_ascending_explicit_dns')
    from ecopadg.serving.campaign import node_lease
    original_identity = common.identity

    async def identity(session, binding):
        if binding['instances']:
            return await original_identity(session, binding)
        assert binding['schema'] == 'C7B-observed-empty-predecessor-v1'
        running = await common.command('docker', 'ps', '--no-trunc', '--format', '{{.ID}}')
        processes = await common.command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits')
        assert not running.strip() and not processes.strip(), 'cold recovery requires actual empty node'
        return dict(schema='C7B-live-empty-node-observation-v1', captured_s=time.time(),
                    running_containers=[], compute_processes=[], actual_observation=True)
    common.identity = identity
    previous = dict(schema='C7B-observed-empty-predecessor-v1', protocol_id=common.PROTOCOL,
                    hostname=HOSTNAME, model='7b', system='pdblend', deadline_s=None,
                    campaign_lifecycle=common.CAMPAIGN_LIFECYCLE, instances=[], configs={}, files={}, large_inputs={})
    with node_lease():
        state['node_lease_held'] = True
        save(out / 'status.json', state)
        # Static checks finish before the original meter or any docker start.
        ids = [i['container']['id'] for i in target['instances']]
        actual = json.loads(await common.command('docker', 'inspect', *ids))
        historical = {x['Id']:x for x in checked(spec['historical_inventory'])['expected_inventory']}
        save(out / 'containers.before.json', actual)
        assert {x['Id'] for x in actual} == set(ids)
        proof = {x['Id']:container_equivalence(historical[x['Id']], x, dns) for x in actual}
        save(out / 'static-equivalence.json', proof)
        for instance in target['instances']:
            with socket.socket() as probe:
                probe.settimeout(0.2)
                assert probe.connect_ex(('127.0.0.1', instance['port'])) != 0, 'retained endpoint port already occupied'
        binding = await helper.restore_core(common, target, previous, out / 'measured-restore')
        restored = read(out / 'measured-restore/status.json')
        assert restored['complete'] and restored['correctness']['passed'] and restored['clock_restore_complete']
        assert not restored['errors']
        binding['fresh_ascending_binding'] = True
        binding['frequency_and_cancel_qualification_pending'] = True
        save(out / 'binding.json', binding)
        state.update(complete=True, binding=ref(out / 'binding.json'),
                     ordinary=ref(out / 'measured-restore/correctness/status.json'),
                     restoration=ref(out / 'measured-restore/status.json'),
                     frequency_and_cancel_qualification_pending=True,
                     serving_measurements_started=False)
    state['node_lease_held'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=HERE/'cold-spec.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = read(args.spec)
    validate(spec)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, gpu_work_started=False,
                              frequency_and_cancel_qualification_pending=True)))
        return
    assert args.out and not args.out.exists(), 'fresh explicit output directory required'
    args.out.mkdir(parents=True)
    state = dict(schema='C7B-cold-recovery-status-v1', pid=os.getpid(), started_s=time.time(),
                 spec=ref(args.spec), complete=False, node_lease_held=False, automatic_retry=False)
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(spec, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
