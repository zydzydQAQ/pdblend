"""CPU-only fresh EcoServe qualification after a measured retained restart.

The independent native oracle and original 27-request gate readers are frozen.
Only process freshness is re-established here; old failed exact flags stay false.
No engine, clock, lease, network or performance operations are performed.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
C = Path('/root/workspace/pdblend-next-v1/campaign')
QUAL = C / 'B32B-temporal-qualification-v2'
BINDER = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1/B/eco-drain-binder-v1')
PROTOCOL = 'legacy-temporal-default-trajectory-exact-v2'
DEADLINE = None
ORIGINALS = {
    str(C / 'B32B-ecoserve-qualified-main-v1/binding.json'):
        '87dbf43fcf8076dc1b4bc588e14fad717a2d2b3eeb83cae1514f3fe2d078d119',
    str(C / 'B32B-main-to-scale-handoff-v2/attempt-001/scale-bindings/ecoserve/binding.json'):
        '1b1b4280ce8dd4b8cd919cf5a1706aa9229eb94776fd1ef0af5b4cc1c9ee423f',
}
PINS = {
    str(QUAL / 'manifest.json'): 'fe69a81abe3cddd789dd91041faddbae2dc6f82bada79cb7375ad93542dc6633',
    str(QUAL / 'qualification.py'): '693d7b47ae5473c6fae0e0b21b3a6083b7599980a0873f22271c4fc350ffd1a1',
    str(BINDER / 'manifest.json'): '817d4c9416e2912487346c17dde7e881b20ba5d05f66bd25b498ad3f754d5598',
    str(BINDER / 'bind.py'): 'e56b05c59c22dd6473050a1bc2d8ea739e55e8be04d3ee02af99c1a59e05f109',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def pinned(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'explicit path/SHA reference required')
    require(Path(value['path']).is_absolute() and sha(value['path']) == value['sha256'],
            'input SHA differs: ' + value['path'])
    return read(value['path'])


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sources():
    for path, digest in PINS.items():
        require(sha(path) == digest, 'frozen qualifier/binder source changed')
    b = load('priority_restored_frozen_binder', BINDER / 'bind.py')
    b.package_check()
    q = load('priority_restored_frozen_qualifier', QUAL / 'qualification.py')
    files = q.check_package()
    files.update(PINS)
    g, o, power = q.modules()
    return b, q, g, o, power, files


def instance_policy(instance):
    """Only real process identities may change across a retained restart."""
    value = copy.deepcopy(instance)
    value.pop('host_pid', None)
    value['container'].pop('StartedAt')
    value['provenance'].pop('pid')
    return value


def same_policy(old, fresh):
    require(len(old) == len(fresh) == 4, 'four original TP2 instances required')
    require([instance_policy(i) for i in old] == [instance_policy(i) for i in fresh],
            'retained source/config/geometry/route/role policy changed')
    for instance in fresh:
        require(type(instance['provenance'].get('pid')) is int and instance['provenance']['pid'] > 0,
                'actual imported engine PID required')


def merge_files(target, mapping):
    for path, digest in mapping.items():
        require(target.get(path, digest) == digest, 'conflicting source hash: ' + path)
        target[path] = digest


def verify_files(files, oracle_module):
    for path, digest in files.items():
        require(oracle_module.sha(path) == digest, 'frozen input changed: ' + path)


def restoration_energy(directory, status):
    reader_path = C / 'AC-baseline-binding-v2/gate_evidence.py'
    require(sha(reader_path) == 'b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6',
            'frozen raw energy reader changed')
    reader = load('priority_restored_energy_reader', reader_path)
    with (directory / 'power/power.csv').open() as handle:
        raw = list(csv.DictReader(handle))
    samples = [(float(row['t_s']), [float(row[f'gpu{i}_w']) for i in range(8)]) for row in raw]
    require(all(math.isfinite(t) and all(math.isfinite(w) and w >= 0 for w in values) for t, values in samples),
            'restoration raw power has non-finite or negative samples')
    actual = reader.integrate(samples, status['measurement_start_s'], status['measurement_end_s'])
    require(math.isclose(actual, status['setup_and_correctness_energy_j'], rel_tol=1e-8, abs_tol=1e-6),
            'restoration energy does not match all-eight raw measurement')
    return actual


def restore_contract(bootstrap, original, inventory_ref, binder, power):
    evidence = bootstrap.get('restored_priority')
    require(isinstance(evidence, dict) and set(evidence) == {'binding', 'status', 'inventory'},
            'explicit measured priority restoration references required')
    require(evidence['inventory'] == inventory_ref, 'fresh inventory differs from measured restoration')
    raw = pinned(evidence['binding'])
    status = pinned(evidence['status'])
    inventory = pinned(inventory_ref)
    directory = Path(evidence['status']['path']).parent
    require(Path(evidence['binding']['path']) == directory / 'binding.json'
            and Path(inventory_ref['path']) == directory / 'containers.after.json',
            'fresh identities must come from the same measured restore operation')
    same_policy(original['instances'], raw['instances'])
    same_policy(raw['instances'], bootstrap['instances'])
    indexed = binder.inventory_contract(bootstrap, inventory)
    old_inventory = {r['Id']: r for r in pinned(dict(path=original['identity_file'],
        sha256=original['files'][original['identity_file']]))}
    for cid, actual in indexed.items():
        previous = old_inventory[cid]
        require(actual['State']['Pid'] != previous['State']['Pid'],
                'restored host PID did not change from the original process')
        for key in ('Id', 'Name', 'Image', 'Path', 'Args', 'Config', 'HostConfig'):
            require(actual.get(key) == previous.get(key), 'retained Docker execution settings changed: ' + key)
        normalize = lambda row: sorted(json.dumps(v, sort_keys=True) for v in row.get('Mounts', []))
        require(normalize(actual) == normalize(previous), 'retained Docker mounts changed')
    for old, actual, bound in zip(original['instances'], raw['instances'], bootstrap['instances']):
        container = indexed[actual['container']['id']]
        require(actual['container'] == bound['container']
                and actual['provenance'] == bound['provenance']
                and actual.get('host_pid') == container['State']['Pid'],
                'restored host PID/StartedAt/provenance is not the fresh bootstrap')
        require(old['container']['StartedAt'] != actual['container']['StartedAt'],
                'a fresh restart was not observed for every original container')
    for key in ('hostname', 'model', 'large_inputs', 'window_s', 'seeds'):
        require(original.get(key) == raw.get(key) == bootstrap.get(key), 'restored model/protocol changed: ' + key)
    new_host = Path('/root/workspace/pdblend-next-v1/releases/five-system100-B32B-baseline-eco-drain-v1-runtime')
    manifest = read(new_host/'manifest.json')
    require(sha(new_host/'manifest.json') == 'edea3effe708fa2ff312648e793b462a08877d359ed938e49bbd5987c1265e20', 'new Eco guard host changed')
    old_host = Path(manifest['parent_release'])
    require(original['host_release'] == raw['host_release'] == str(old_host)
            and bootstrap['host_release'] == str(new_host), 'explicit original-to-guard host lineage required')
    old_manifest = read(old_host/'manifest.json')
    require(sha(old_host/'manifest.json') == manifest['parent_manifest_sha256']
            and set(manifest['files']) == set(old_manifest['files']), 'guard parent source differs')
    require([n for n in manifest['files'] if manifest['files'][n] != old_manifest['files'][n]]
            == ['src/ecopadg/serving/runtime.py'], 'only Eco window guard may change')
    verify_files({str(new_host/n):v for n,v in manifest['files'].items()}, type('Hasher', (), {'sha':staticmethod(sha)}))
    require(raw.get('configs') == original['configs'], 'restored original policy config references changed')
    require(status.get('complete') is True and status.get('clock_restore_complete') is True
            and not status.get('error') and not status.get('errors') and not status.get('sampling_error')
            and status.get('power_evidence', {}).get('power_source_verified') is True,
            'fresh measured restore or clock cleanup is incomplete')
    correctness = status.get('correctness', {})
    require(correctness.get('passed') is True and not correctness.get('error')
            and not correctness.get('cleanup_errors') and not correctness.get('owned')
            and len(correctness.get('restoration', [])) == 4
            and all(row.get('complete') is True for row in correctness['restoration']),
            'fresh ordinary correctness/native cleanup failed')
    replies = correctness.get('replies', [])
    expected = {(i['id'], n) for i in raw['instances'] for n in (128, 7168)}
    require(len(replies) == 8 and {(r['instance_id'], r['prompt_length']) for r in replies} == expected,
            'full eight-request short/long restoration gate required')
    require(len({r['request_id'] for r in replies}) == 8, 'restoration request IDs are not unique')
    for length in (128, 7168):
        rows = [r['response'] for r in replies if r['prompt_length'] == length]
        for row in rows:
            tokens = row.get('token_ids')
            require(isinstance(tokens, list) and len(tokens) == 64 and all(type(t) is int and t >= 0 for t in tokens)
                    and row.get('usage', {}).get('prompt_tokens') == length
                    and row['usage'].get('completion_tokens') == 64, 'restoration output work differs')
        require(all(r['token_ids'] == rows[0]['token_ids'] for r in rows), 'restoration replica output mismatch')
    require(status['started_s'] <= status['measurement_start_s'] < correctness['started_s']
            <= correctness['finished_s'] <= status['measurement_end_s'] <= status['finished_s'],
            'restoration measurement time boundaries differ')
    energy = status.get('setup_and_correctness_energy_j')
    require(type(energy) in (int, float) and math.isfinite(energy) and energy > 0,
            'measured restoration energy missing')
    files = {str(p.resolve()): sha(p) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(power.audit_raw(directory / 'power', files).get('power_source_verified') is True,
            'restoration eight-GPU raw power source invalid')
    restoration_energy(directory, status)
    return status, inventory, files


def audit_inputs(bootstrap_ref, gate_dir, original_performance_ref, fresh_inventory_ref):
    binder, qualifier, shape, oracle_module, power, files = sources()
    require(ORIGINALS.get(original_performance_ref.get('path')) == original_performance_ref.get('sha256'),
            'unregistered original performance binding')
    original = pinned(original_performance_ref)
    bootstrap = pinned(bootstrap_ref)
    binder.bootstrap_contract(bootstrap)
    same_policy(original['instances'], bootstrap['instances'])
    identity_path = bootstrap.get('identity_file')
    require(identity_path and bootstrap['files'].get(identity_path) == sha(identity_path)
            and read(identity_path) == pinned(fresh_inventory_ref),
            'bootstrap identity alias must be frozen and exactly equal to the original restored physical inventory')
    restore, inventory, restore_files = restore_contract(bootstrap, original, fresh_inventory_ref, binder, power)
    oracle_ref = original['oracle']
    oracle = pinned(oracle_ref)
    actual_oracle = oracle_module.verify(oracle['inputs'])
    require(actual_oracle == oracle and oracle.get('kind') == 'registered-native-default-temporal-oracle'
            and oracle.get('protocol_id') == PROTOCOL, 'registered native oracle raw revalidation failed')
    historical = oracle_module.read(oracle_module.ATTEMPT / 'results/restored-bootstrap.binding.json')
    same_policy(historical['instances'], bootstrap['instances'])
    gate = Path(gate_dir).resolve()
    status = read(gate / 'status.json')
    require(actual_oracle['physical_evidence']['full_operation_end_s'] <= restore['started_s']
            and restore['finished_s'] <= status['started_s'] <= status['measurement_start_s']
            < status['measurement_end_s'] <= status['finished_s'],
            'fresh 27-request gate must follow the complete measured restoration')
    inspected = shape.inspect_fresh_gate(gate, bootstrap, oracle['reference_tokens_by_label'], power.load())
    temporal, original_gate = inspected['temporal'], inspected['original_gate']
    require(original_gate['verified']['ordinary'] and original_gate['verified']['pd']
            and temporal['temporal_native_trajectory_exact'], 'fresh original checks/native trajectory failed')
    for mapping in (bootstrap['files'], original['files'], actual_oracle['files'], power.SOURCES,
                    inspected['files'], restore_files):
        merge_files(files, mapping)
    for source_ref in (bootstrap_ref, original_performance_ref, fresh_inventory_ref, oracle_ref):
        merge_files(files, {source_ref['path']: source_ref['sha256']})
    files[str(Path(__file__).resolve())] = sha(__file__)
    qualification = dict(schema=2, kind='temporal-default-trajectory-qualification', protocol_id=PROTOCOL,
        passed=True, eligible_systems={'ecoserve': True},
        verified={name: True for name in binder.NEEDED}, original_mechanism_gate=original_gate['verified'],
        legacy_single_vs_pair_exact=temporal['legacy_single_vs_pair_exact'],
        legacy_first_differences=temporal['legacy_first_differences'], legacy_failure_preserved=True,
        inputs=dict(gate_dir=str(gate), binding_canonical_sha256=canonical(bootstrap),
            oracle_canonical_sha256=canonical(oracle), oracle_inputs=actual_oracle['inputs']),
        temporal=temporal, physical_evidence=original_gate['physical'], files=files,
        restored_priority=copy.deepcopy(bootstrap['restored_priority']),
        identity_relation='restarted retained original four TP2 containers; fresh measured gate on all new processes',
        verified_identity_scope='recorded fresh processes; the caller must check live identities under its own lease',
        numerical_scope='unchanged registered native trajectory exact; not general numerical correctness',
        performance_outcomes_certified=False)
    _, _, _, legacy = binder.qualification_contract(qualification, bootstrap, gate, inventory)
    verify_files(files, oracle_module)
    return original, bootstrap, qualification, inventory, legacy, oracle_module


def build_binding(original, bootstrap, qualification, inventory, legacy, refs, output_root):
    """Deterministic output assembly shared by creation and later CPU re-audit."""
    out = Path(output_root).resolve()
    require(HERE in out.parents, 'new qualification output must be under priority-v2/B')
    files = dict(qualification['files'])
    generated = {out / 'identity.json': encoded(inventory), out / 'qualification.json': encoded(qualification)}
    configs = {}
    for dataset, source in original['configs'].items():
        require(original['files'].get(source) == sha(source), 'original policy config changed')
        path = out / 'configs' / (dataset + '.json')
        generated[path] = Path(source).read_bytes()
        configs[dataset] = str(path)
    for path, data in generated.items():
        files[str(path)] = hashlib.sha256(data).hexdigest()
    result = copy.deepcopy(original)
    for key in ('fresh_ablation_binding', 'experiment_scope', 'executor_wrapper'):
        result.pop(key, None)
    result.update(host_release=bootstrap['host_release'],deadline_s=None,campaign_lifecycle='until_declared_complete_v1',instances=copy.deepcopy(bootstrap['instances']), configs=configs, output=str(out / 'results'),
        files=files, identity_file=str(out / 'identity.json'), correctness_evidence=qualification['inputs']['gate_dir'],
        qualification=dict(path=str(out / 'qualification.json'), sha256=files[str(out / 'qualification.json')]),
        qualifier_source=ref(__file__), qualified_bootstrap=copy.deepcopy(refs['bootstrap']),
        output_correctness_verified=True, correctness_gate_required_before_performance=False,
        correctness_protocol_id=PROTOCOL, legacy_output_correctness_verified=False, legacy_single_vs_pair_exact=False,
        legacy_exact_evidence=legacy, restored_priority=copy.deepcopy(bootstrap['restored_priority']),
        priority_qualification_inputs=refs,
        identity_relation=qualification['identity_relation'], historical_binding_only=False,
        mechanism_proof=dict(required=['ordinary', 'temporal_native_trajectory_exact'],
            verified=copy.deepcopy(qualification['verified']), legacy_verified=qualification['original_mechanism_gate'],
            overall_runtime_gate_passed=False, qualified_under_explicit_protocol=PROTOCOL,
            original_failure_preserved=True))
    return result, generated


def qualify(bootstrap_ref, gate_dir, original_performance_ref, fresh_inventory_ref, output_root):
    """Append-only CPU qualification; returns the fresh binding {path, sha256}."""
    out = Path(output_root).resolve()
    require(not out.exists(), 'new output directory required; no overwrite or silent repeat')
    original, bootstrap, qualification, inventory, legacy, oracle = audit_inputs(
        bootstrap_ref, gate_dir, original_performance_ref, fresh_inventory_ref)
    refs = dict(bootstrap=bootstrap_ref, gate_dir=str(Path(gate_dir).resolve()),
                original_performance=original_performance_ref, fresh_inventory=fresh_inventory_ref)
    binding, generated = build_binding(original, bootstrap, qualification, inventory, legacy, refs, out)
    verify_files(qualification['files'], oracle)
    out.mkdir(parents=True)
    for path, data in generated.items():
        write_new(path, data)
    verify_files(binding['files'], oracle)
    write_new(out / 'binding.json', encoded(binding))
    binding_ref = ref(out / 'binding.json')
    write_new(out / 'binding-receipt.json', encoded(dict(complete=True, binding=binding_ref,
        gpu_executed=False, original_gate_passed=False, legacy_temporal_exact=False,
        actual_live_recheck_still_required=True, performance_outcomes_certified=False)))
    return binding_ref


def audit_binding(binding_ref):
    """Re-read all completed raw evidence; no live checks and no output writes."""
    if not isinstance(binding_ref, dict):
        binding_ref = ref(binding_ref)
    binding = pinned(binding_ref)
    refs = binding['priority_qualification_inputs']
    original, bootstrap, qualification, inventory, legacy, oracle = audit_inputs(
        refs['bootstrap'], refs['gate_dir'], refs['original_performance'], refs['fresh_inventory'])
    expected, generated = build_binding(original, bootstrap, qualification, inventory, legacy, refs,
                                        Path(binding_ref['path']).parent)
    require(binding == expected, 'fresh performance binding differs from raw revalidation')
    for path, data in generated.items():
        require(path.read_bytes() == data, 'derived qualification/config/inventory changed: ' + str(path))
    verify_files(binding['files'], oracle)
    return dict(complete=True, passed=True, binding=binding_ref, gpu_executed=False,
                actual_live_recheck_still_required=True, original_temporal_exact=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'qualify', 'audit'])
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--spec-sha256')
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--binding-sha256')
    args = parser.parse_args()
    if args.command == 'check':
        sources()
        result = dict(cpu_only=True, gpu_executed=False)
    elif args.command == 'audit':
        result = audit_binding(dict(path=str(args.binding.resolve()), sha256=args.binding_sha256))
    else:
        spec = pinned(dict(path=str(args.spec.resolve()), sha256=args.spec_sha256))
        result = qualify(**spec)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
