"""Fresh-process deployment and correctness gates for one leased SLO90 stage.

The CLI executes only with an inherited, already held node lease. Preparation
and measurement failures are retained and never publish runnable bindings.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROTOCOL = 'a14b-sharegpt-slo90-v1'
SYSTEMS = ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked_ref(reference):
    path = Path(reference['path']).resolve()
    require(digest(path) == reference['sha256'], 'frozen input changed: ' + str(path))
    return path


def load(name):
    spec = importlib.util.spec_from_file_location('slo90_worker_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dependencies():
    return SimpleNamespace(adapter=load('runtime_adapter'), baseline=load('baseline_stage'),
                           deploy=load('deploy'), lease_check=load('dispatcher').verify_owned_fd)


def inherited_lease(lease, check):
    if lease is None:
        require('PDBLEND_NODE_LOCK_FD' in os.environ, 'already held inherited node lease required')
        lease = int(os.environ['PDBLEND_NODE_LOCK_FD'])
    descriptor = lease if type(lease) is int else lease.fileno()
    check(descriptor)
    return descriptor


def verify_inputs(release, adapter):
    """Verify source locks and all model/cache bytes before any deployment."""
    require(release.get('protocol_id') == PROTOCOL, 'wrong frozen release protocol')
    require(set(release['configs']) == set(SYSTEMS), 'all five prepared configurations required')
    for reference in release['configs'].values():
        checked_ref(reference)
    source_lock = checked_ref(release['source_lock'])
    for path, expected in read(source_lock)['files'].items():
        require(digest(path) == expected, 'source input changed: ' + path)
    for path in (*release['host_releases'].values(), release['common_dir'], release['gate_code']):
        require(adapter.checked_manifest(path)['protocol_id'] == PROTOCOL, 'wrong runtime protocol')
    reference = release.get('required_inputs')
    inputs = checked_ref(reference) if reference else HERE / 'asset-staging/required-inputs.json'
    records = read(inputs)['files']
    require(records and all(Path(path).is_absolute() for path in records), 'absolute large inputs required')
    identities = {}
    for path, expected in records.items():
        before = adapter.stat_identity(path)
        require(before['size'] == expected['size'] and digest(path) == expected['sha256'],
                'model or retained cache content changed: ' + path)
        after = adapter.stat_identity(path)
        require(before == after, 'large input changed while hashing: ' + path)
        identities[path] = dict(sha256=expected['sha256'], stat=after)
    return Path(inputs).resolve(), identities


def remap_pdb_config(original, actual_instances, host_release):
    """Keep the frozen policy, replacing only actual deployment locations."""
    cfg = copy.deepcopy(original)
    require(cfg.get('measurement_window_protocol') == PROTOCOL and
            cfg['strategy'].startswith('pdblend'), 'prepared PDB configuration required')
    require([i['id'] for i in cfg['instances']] == [i['id'] for i in actual_instances],
            'actual PDB instance IDs differ from frozen policy')
    for old, actual in zip(cfg['instances'], actual_instances):
        require((old['tp'], old['gpus']) == (actual['tp'], actual['gpus']),
                'actual PDB physical layout differs from frozen policy')
        for key in ('url', 'port', 'kv_port'):
            old[key] = actual[key]
        old['container_name'] = actual['container']['name']
    cfg['controller_source_release'] = str(Path(host_release).resolve())
    return cfg


def passed_gate(status):
    require(status.get('complete') is True and status.get('passed') is True and status.get('measurement_valid') is True,
            'fresh measured correctness gate did not pass')
    require(status.get('native_cleanup_complete') is True and status.get('clock_restore_complete') is True,
            'correctness gate lacks complete native and clock cleanup')


def confirm_large_inputs(binding, identities, adapter):
    require(set(binding.get('large_inputs', {})) == set(identities), 'binding omitted model/cache inputs')
    for path, identity in identities.items():
        require(adapter.stat_identity(path) == identity['stat'] == binding['large_inputs'][path]['stat'],
                'model/cache input changed after deployment verification: ' + path)


async def execute_stage(stage, release_path, idle_proof, previous_binding=None, *, lease=None, deps=None):
    require(stage in ('pdblend', 'baselines'), 'unknown deployment stage')
    deps = deps or dependencies()
    descriptor = inherited_lease(lease, deps.lease_check)
    release_path, idle_proof = Path(release_path).resolve(), Path(idle_proof).resolve()
    release = read(release_path)
    target = Path(release['deployment_root']).resolve() / stage
    require(not target.exists(), 'fresh stage destination required; review retained failure before retry')
    result = dict(schema=1, protocol_id=PROTOCOL, stage=stage, hostname=socket.gethostname(),
                  release=ref(release_path), idle_proof=ref(idle_proof), started_s=time.time(),
                  complete=False, measurement_valid=False, bindings={})
    try:
        if stage == 'baselines':
            require(previous_binding is not None, 'actual previous PDB binding required for baseline transition')
            previous = read(previous_binding)
            require(previous.get('hostname') == result['hostname'] and previous.get('system') == 'pdblend',
                    'previous binding is not this host PDB stage')
        else:
            require(previous_binding is None, 'first PDB deployment must start from the idle host')
        inputs_manifest, identities = verify_inputs(release, deps.adapter)
        spec_ref = deps.deploy.prepare_spec(stage, release_path, result['hostname'], idle_proof,
                                            previous_binding=previous_binding)
        spec_path = checked_ref(spec_ref)
        spec = read(spec_path)
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'large-input-verification.json', dict(manifest=ref(inputs_manifest),
              verified_s=time.time(), files=identities))
        result['spec'] = spec_ref
        deploy_out = target / 'deployment'
        try:
            receipt = await deps.deploy.execute(spec_path, deploy_out, run=True, lease=descriptor)
        finally:
            # Keep the original receipt byte-identical at the supervisor's
            # stable recovery path, including an invalid/partial deployment.
            receipt_path = deploy_out / 'deployment-receipt.json'
            if receipt_path.exists():
                stable_receipt = target / 'deployment-receipt.json'
                with stable_receipt.open('xb') as stream:
                    stream.write(receipt_path.read_bytes())
                result['deployment_receipt'] = ref(stable_receipt)
        require(read(receipt_path) == receipt and receipt.get('complete') is True and
                receipt.get('measurement_valid') is True, 'actual deployment did not pass')
        base_path = deploy_out / 'binding-base.json'
        base = read(base_path)
        require(base.get('model') == '14b' and base.get('hostname') == result['hostname'],
                'deployment returned foreign actual engine identity')
        host_release = release['host_releases'][stage]
        evidence = (release_path, idle_proof, spec_path, inputs_manifest,
                    target / 'large-input-verification.json', stable_receipt,
                    *map(Path, release.get('evidence', [])), Path(__file__).resolve())
        gate = target / ('ordinary-gate' if stage == 'pdblend' else 'mechanism-gate')
        if stage == 'pdblend':
            require(base.get('system') == 'pdblend', 'PDB deployment returned wrong system')
            original_path = checked_ref(release['configs']['pdblend'])
            cfg = remap_pdb_config(read(original_path), base['instances'], host_release)
            config_path = target / 'policy/sharegpt.json'
            write(config_path, cfg)
            write(target / 'policy/provenance.json', dict(parent=ref(original_path),
                  actual_base=ref(base_path), only_actual_locations_changed=True))
            bootstrap = deps.adapter.make_binding(base_path, host_release=host_release, config=config_path,
                common_dir=release['common_dir'], output=target / 'results', evidence=evidence,
                frozen_inputs=(target / 'policy/provenance.json',), large_input_paths=identities,
                retain_parent_files=True)
        else:
            deps.baseline.validate_actual_layout(base)
            cfg_ref = deps.baseline.prepare_policy('mixed', base['instances'], target / 'bootstrap-policy',
                host_release=host_release, engine_entry=spec['source_entry'])
            bootstrap = deps.adapter.make_binding(base_path, host_release=host_release, config=cfg_ref,
                common_dir=release['common_dir'], output=target / 'gate-results', evidence=evidence,
                frozen_inputs=(target / 'bootstrap-policy/policy-provenance.json',), large_input_paths=identities,
                retain_parent_files=True)
        bootstrap.update(output_correctness_verified=False, correctness_gate_required_before_performance=True)
        bootstrap_path = target / 'bootstrap-binding.json'
        write(bootstrap_path, bootstrap)
        confirm_large_inputs(bootstrap, identities, deps.adapter)
        common = deps.adapter.load_runtime(host_release, release['common_dir'])
        common.validate_binding(bootstrap)
        if stage == 'pdblend':
            status = await deps.deploy.measured_ordinary(common, bootstrap, gate, lease=descriptor)
        else:
            status = await deps.baseline.qualify(bootstrap_path, gate_code=release['gate_code'],
                out=gate, runtime_dir=spec['runtime_dir'], lease=descriptor)
        passed_gate(status)
        require(read(gate / 'status.json') == status, 'gate status differs from retained measurement')
        result['gate'] = ref(gate / 'status.json')
        if stage == 'pdblend':
            gate_files = tuple(path for path in gate.rglob('*') if path.is_file())
            binding = deps.adapter.make_binding(base_path, host_release=host_release, config=config_path,
                common_dir=release['common_dir'], output=target / 'results',
                evidence=(*evidence, bootstrap_path, target / 'policy/provenance.json'),
                frozen_inputs=gate_files, large_input_paths=identities, retain_parent_files=True)
            binding.update(output_correctness_verified=True, correctness_gate_required_before_performance=False,
                           correctness_evidence=str(gate), deployment_receipt=str(stable_receipt))
            confirm_large_inputs(binding, identities, deps.adapter)
            common.validate_binding(binding)
            path = target / 'bindings/pdblend/binding.json'
            write(path, binding)
            bindings = {'pdblend': ref(path)}
        else:
            bindings = deps.baseline.build_baseline_bindings(base_path, gate=gate,
                gate_code=release['gate_code'], host_release=host_release, common_dir=release['common_dir'],
                out=target / 'bindings', engine_entry=spec['source_entry'],
                deployment_receipt=stable_receipt, evidence=(*evidence, bootstrap_path),
                large_input_paths=identities)
            require(set(bindings) == set(SYSTEMS[1:]), 'all four fresh baseline bindings required')
            for reference in bindings.values():
                binding = read(checked_ref(reference))
                confirm_large_inputs(binding, identities, deps.adapter)
                common.validate_binding(binding)
        deps.lease_check(descriptor)
        result.update(measurement_valid=True, bindings=bindings)
    except Exception as exc:
        result.update(error=dict(type=type(exc).__name__, message=str(exc)), bindings={})
    finally:
        gate_path = target / ('ordinary-gate' if stage == 'pdblend' else 'mechanism-gate') / 'status.json'
        if gate_path.exists():
            result['gate'] = ref(gate_path)
        result.update(complete=True, finished_s=time.time())
        write(target / 'stage-result.json', result)
    return result


async def restore(release_path, *, lease=None, deps=None):
    deps = deps or dependencies()
    descriptor = inherited_lease(lease, deps.lease_check)
    release_path = Path(release_path).resolve()
    release = read(release_path)
    require(release.get('protocol_id') == PROTOCOL, 'wrong frozen release protocol')
    target = Path(release['deployment_root']).resolve() / 'restore-result.json'
    require(not target.exists(), 'restoration already attempted; retain its evidence for review')
    result = dict(schema=1, protocol_id=PROTOCOL, hostname=socket.gethostname(),
                  release=ref(release_path), started_s=time.time(), complete=False,
                  measurement_valid=False, restored=[])
    try:
        # The deployment owner restores only receipt-recorded containers and
        # original clock/native state, and retains its measured recovery data.
        status = await deps.deploy.restore_original(release_path, run=True, lease=descriptor)
        result['restoration'] = status
        result['restored'] = status.get('restored', [])
        result['measurement_valid'] = status.get('complete') is True and status.get('measurement_valid') is True
        deps.lease_check(descriptor)
    except Exception as exc:
        result.update(measurement_valid=False, error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        receipt = Path(release['deployment_root']) / 'restoration-001/restoration-receipt.json'
        if receipt.exists():
            result['restoration_receipt'] = ref(receipt)
        result.update(complete=True, finished_s=time.time())
        write(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage')
    stage.add_argument('--stage', choices=('pdblend', 'baselines'), required=True)
    stage.add_argument('--release', required=True)
    stage.add_argument('--idle-proof', required=True)
    stage.add_argument('--previous-binding')
    recovery = commands.add_parser('restore')
    recovery.add_argument('--release', required=True)
    args = parser.parse_args(argv)
    if args.command == 'stage':
        result = asyncio.run(execute_stage(args.stage, args.release, args.idle_proof, args.previous_binding))
    else:
        result = asyncio.run(restore(args.release))
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 0 if result.get('complete') is True and result.get('measurement_valid') is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
