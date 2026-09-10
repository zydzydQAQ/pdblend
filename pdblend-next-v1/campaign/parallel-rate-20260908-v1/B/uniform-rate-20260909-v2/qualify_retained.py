"""Pure raw requalification after restarting an already-qualified B32B Eco host.

The original native oracle, all27 gate, identity and power rules are unchanged.
The stopped containers.before inventory replaces the old deployment-specific
cold_start_preflight field; all retained source/config values must be identical.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_PATH = ROOT / 'B/ascending-resume-20260909-v2/qualify_eco_drained_baseline_v2.py'
sys.path.insert(0, str(BASE_PATH.parent))

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

BASE = load(BASE_PATH, '_retained_B_original_eco_auditor')
require, sha, read, ref, pinned = BASE.require, BASE.sha, BASE.read, BASE.ref, BASE.pinned
same_policy, verify_files = BASE.same_policy, BASE.verify_files
hostconfig_equivalence = BASE.hostconfig_equivalence
restoration_energy = BASE.restoration_energy
PARENT = dict(path=str(ROOT / 'B/ascending-resume-20260909-v2/baseline-qualification-001/ecoserve/binding.json'),
              sha256='777a679ff160be12d430fa26bbff7f624e044d4c6d3c4248a28471eed37b8de6')
GATE = HERE / 'native-qualification-001'
RESTORE = HERE / 'baseline-restoration-001'
OUTPUT = HERE / 'native-qualification-retained-002'

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
        for key in ('Id', 'Name', 'Image', 'Path', 'Args', 'Config'):
            require(actual.get(key) == previous.get(key), 'retained Docker execution settings changed: ' + key)
        hostconfig_equivalence(previous['HostConfig'], actual['HostConfig'])
        normalize = lambda row: sorted(json.dumps(v, sort_keys=True) for v in row.get('Mounts', []))
        require(normalize(actual) == normalize(previous), 'retained Docker mounts changed')
    # This restart already targets the qualified Eco guard runtime. The actual
    # stopped inventory supplies the cold-start proof directly; no old field is synthesized.
    cold_rows = read(directory / 'containers.before.json')
    cold = {row['Id']: row for row in cold_rows}
    require(set(cold) == set(old_inventory), 'retained stopped target identity set differs')
    for cid, previous in old_inventory.items():
        actual = cold[cid]
        require(actual['State']['Running'] is False and actual['State']['Pid'] == 0,
                'target was not stopped before retained restart')
        for key in ('Id', 'Name', 'Image', 'Path', 'Args', 'Config'):
            require(actual.get(key) == previous.get(key), 'stopped retained execution changed: ' + key)
        hostconfig_equivalence(previous['HostConfig'], actual['HostConfig'])
        normalize = lambda row: sorted(json.dumps(v, sort_keys=True) for v in row.get('Mounts', []))
        require(normalize(actual) == normalize(previous), 'stopped retained Docker mounts changed')
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
    require(original['host_release'] == raw['host_release'] == bootstrap['host_release'],
            'already-qualified Eco guard host changed during retained restart')
    host = Path(original['host_release'])
    manifest = read(host / 'manifest.json')
    require(sha(host / 'manifest.json') == 'edea3effe708fa2ff312648e793b462a08877d359ed938e49bbd5987c1265e20',
            'qualified Eco guard runtime changed')
    verify_files({str(host / name): digest for name, digest in manifest['files'].items()},
                 type('Hasher', (), {'sha': staticmethod(sha)}))
    require(raw.get('configs') == original['configs'], 'retained qualified policy configuration changed')
    for path in original['configs'].values():
        require(original['files'].get(path) == sha(path), 'qualified policy configuration bytes changed')
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
    files[str(Path(__file__).resolve())] = sha(__file__)
    return status, inventory, files


def invocation():
    saved = read(GATE / 'gate-invocation.json')
    gate = read(GATE / 'original27/status.json')
    boot = read(GATE / 'bootstrap.json')
    require(saved['bootstrap'] == ref(GATE / 'bootstrap.json') and not saved['stop_requested']
            and saved['exitcode'] in (0, 1), 'fresh gate invocation stopped or changed')
    require(saved['started_s'] <= gate['started_s'] < gate['finished_s'] <= saved['finished_s'],
            'fresh gate time membership differs')
    require(saved['host_manifest'] == ref(Path(boot['host_release']) / 'manifest.json')
            and saved['gate_source'] == ref(ROOT / 'B/baseline-gate-until-complete-v1/validate.py'),
            'fresh gate source differs')
    require(gate['complete'] and gate['measurement_valid'] and gate['native_cleanup_complete']
            and gate['clock_restore_complete'] and not gate['cleanup_errors'] and not gate.get('sampling_error'),
            'fresh native gate measurement or cleanup incomplete')
    return boot


def qualifier():
    parent = pinned(PARENT)
    require(parent['qualifier_source'] == ref(BASE_PATH), 'qualified parent auditor source changed')
    require(BASE.audit_binding(PARENT)['passed'], 'parent qualification did not independently revalidate')
    invocation()
    module = load(BASE_PATH, '_retained_B_fresh_eco_auditor')
    module.HERE = HERE
    module.ORIGINALS = dict(module.ORIGINALS, **{PARENT['path']: PARENT['sha256']})
    module.restore_contract = restore_contract
    original_build = module.build_binding

    def build(*args, **kwargs):
        result, generated = original_build(*args, **kwargs)
        result.update(qualifier_source=ref(__file__), retained_qualified_parent=PARENT,
                      retained_restore_semantics='same qualified guard host and policy; fresh process identities and actual stopped inventory')
        return result, generated
    module.build_binding = build
    return module


def other_binding(system, destination):
    control = load(ROOT / 'B/ascending-rate-v1/baseline_control_v2.py', '_retained_B_original_control')
    original = load(ROOT / 'B/baseline_control_after_external_v2.py', '_retained_B_original_policy')
    helper = load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', '_retained_B_runtime_loader')
    historical = read(control.OLD_BINDINGS[system])
    helper.load_common(historical['host_release'])
    result = original.bind_other(system, ref(original.POLICIES[system]), ref(GATE / 'bootstrap.json'),
                                 GATE / 'original27', destination)
    require(result['configs'] == historical['configs'], 'original baseline strategy/configuration changed')
    result['host_release'] = historical['host_release']
    host = Path(result['host_release'])
    result['files'].update({str(host / path): digest for path, digest in read(host / 'manifest.json')['files'].items()})
    for path in (host / 'manifest.json', control.OLD_BINDINGS[system], Path(control.__file__),
                 Path(__file__).resolve(), GATE / 'gate-invocation.json'):
        result['files'][str(path)] = sha(path)
    result['ascending_native_qualification'] = dict(original_selected_binding=ref(control.OLD_BINDINGS[system]),
        bootstrap=ref(GATE / 'bootstrap.json'), fresh_gate=ref(GATE / 'original27/status.json'),
        native_engine_sources_unchanged=True, controller_source_and_policy_unchanged=True,
        gate_only_native_mechanisms_not_performance=True)
    return result


def write(path, value):
    BASE.write_new(Path(path), BASE.encoded(value))


def qualify(out=OUTPUT):
    out = Path(out).resolve()
    require(not out.exists(), 'new append-only qualification output required')
    q = qualifier()
    refs = {'ecoserve': q.qualify(ref(GATE / 'bootstrap.json'), GATE / 'original27', PARENT,
                                ref(RESTORE / 'containers.after.json'), out / 'ecoserve')}
    require(q.audit_binding(refs['ecoserve'])['passed'], 'fresh Eco raw revalidation failed')
    for system in ('mixed', 'distserve', 'dynamollm'):
        binding = other_binding(system, out / system)
        for path, digest in binding['files'].items():
            require(sha(path) == digest, 'derived baseline source changed: ' + path)
        write(out / system / 'binding.json', binding)
        refs[system] = ref(out / system / 'binding.json')
    qualification_refs = {}
    for system, binding_ref in refs.items():
        record = dict(schema='uniform-v2-retained-native-qualification', passed=True, node='B', model='32b',
            system=system, binding=binding_ref, original_qualified_eco_parent=PARENT,
            bootstrap=ref(GATE / 'bootstrap.json'), gate=ref(GATE / 'original27/status.json'),
            gate_invocation=ref(GATE / 'gate-invocation.json'),
            restoration=ref(RESTORE / 'status.json'), qualifier=ref(__file__),
            independently_recomputed=True, legacy_temporal_false_preserved=True,
            GPU_work_started_by_this_qualifier=False)
        write(out / system / 'qualified.json', record)
        qualification_refs[system] = ref(out / system / 'qualified.json')
        verify(qualification_refs[system])
    write(out / 'bindings.json', refs)
    write(out / 'qualifications.json', qualification_refs)
    return dict(passed=True, bindings=refs, qualifications=qualification_refs, cpu_only=True)


def audit_binding(binding_ref):
    binding = pinned(binding_ref)
    invocation()
    if binding['system'] == 'ecoserve':
        result = qualifier().audit_binding(binding_ref)
        require(result['passed'], 'fresh Eco raw revalidation failed')
    else:
        expected = other_binding(binding['system'], Path(binding_ref['path']).parent)
        require(binding == expected, 'fresh baseline differs from independent reconstruction')
        for path, digest in binding['files'].items():
            require(sha(path) == digest, 'baseline frozen dependency differs: ' + path)
    return dict(passed=True, independently_recomputed=True, binding=binding_ref, node='B', model='32b',
                system=binding['system'], host_manifest=ref(Path(binding['host_release']) / 'manifest.json'),
                files=binding['files'], legacy_temporal_false_preserved=True, cpu_only=True)


def verify(reference):
    value = pinned(reference)
    require(value['schema'] == 'uniform-v2-retained-native-qualification'
            and value['qualifier'] == ref(__file__) and value['node'] == 'B' and value['model'] == '32b',
            'wrong retained qualification identity/source')
    require(value['bootstrap'] == ref(GATE / 'bootstrap.json') and value['gate'] == ref(GATE / 'original27/status.json')
            and value['restoration'] == ref(RESTORE / 'status.json')
            and value['gate_invocation'] == ref(GATE / 'gate-invocation.json'), 'qualified evidence changed')
    result = audit_binding(value['binding'])
    require(result['system'] == value['system'], 'qualification system differs')
    result['qualification'] = reference
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    result = verify(ref(args.verify)) if args.verify else qualify(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
