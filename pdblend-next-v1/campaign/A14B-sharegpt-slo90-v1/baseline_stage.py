"""Fresh eight-resident qualification and unchanged 14B baseline policy binding.

Engine deployment is owned by the campaign supervisor. The preparation helpers
are CPU-only; ``qualify`` performs real measured correctness requests and must
only be called by the process holding the actual host's exclusive node lease.
No hostname, deployment receipt, or previous campaign completion is invented.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('slo90_baseline_runtime_adapter', HERE / 'runtime_adapter.py')
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
read, write, sha, require = adapter.read, adapter.write, adapter.sha, adapter.require
PARENT_GATE = adapter.REPO / 'campaign/AC-legacy-resident-correctness-v1'
PARENT_AUDIT = adapter.REPO / 'campaign/AC-baseline-binding-v2/gate_evidence.py'
OLD_BINDINGS = adapter.REPO / 'campaign/A14B-scale-stage-v1/resident-prepared-001'
BASELINE_IMAGE = 'sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform_gate(source, common_dir):
    """Preserve every actual gate and its bounded work/cleanup measurement."""
    common_path = Path(common_dir).resolve() / 'run.py'
    source = adapter.once(source, "COMMON=ROOT.parent/'five-system-execution-v2/run.py'",
                          'COMMON=Path(' + repr(str(common_path)) + ')')
    source = adapter.once(source,
        "COMMON_SHA='ddc634e0b826d1873ed0bb7e3bd9088ba1412476725d8ec1e371414ccce54ad2'",
        'COMMON_SHA=' + repr(sha(common_path)))
    source = adapter.once(source,
        "require(binding.get('model') in ('7b','14b'),'this gate is the A/C eight-TP1 resident scope')",
        "require(binding.get('model') == '14b','this gate is the 14B eight-TP1 resident scope')")
    source = adapter.once(source,
        "    require(time.time()+510<m.GLOBAL_DEADLINE,'insufficient global time for gate plus cleanup')\n", '')
    source = adapter.once(source,
        'cleanup_end=time.monotonic()+max(0,min(90,m.GLOBAL_DEADLINE-time.time()))',
        'cleanup_end=time.monotonic()+90')
    ast.parse(source)
    return source


def prepare_gate(out, common_dir):
    out, common_dir = Path(out).resolve(), Path(common_dir).resolve()
    require(not out.exists(), 'fresh correctness source destination required')
    adapter.checked_manifest(common_dir)
    out.mkdir(parents=True)
    (out / 'validate.py').write_text(transform_gate((PARENT_GATE / 'validate.py').read_text(), common_dir))
    shutil.copyfile(PARENT_GATE / 'checks.py', out / 'checks.py')
    shutil.copyfile(PARENT_AUDIT, out / 'gate_evidence.py')
    manifest = dict(schema=1, protocol_id=adapter.PROTOCOL, common=adapter.ref(common_dir / 'manifest.json'),
        parent_files={str(p): sha(p) for p in (PARENT_GATE / 'validate.py', PARENT_GATE / 'checks.py', PARENT_AUDIT)},
        checks_byte_identical=True, gate_audit_byte_identical=True,
        no_campaign_deadline=True, work_timeout_s=390, cleanup_timeout_s=90,
        files={name: sha(out / name) for name in ('validate.py', 'checks.py', 'gate_evidence.py')})
    write(out / 'manifest.json', manifest)
    return manifest


def validate_actual_layout(binding):
    require(binding.get('model') == '14b', '14B actual binding required')
    instances = binding.get('instances', [])
    require([(i['tp'], i['gpus']) for i in instances] == [(1, [j]) for j in range(8)],
            'eight actual TP1 GPUs 0..7 required')
    require([i['id'] for i in instances] == ['base100ar' + str(j) for j in range(8)],
            'use the declared 14B baseline instance names to retain policy IDs')
    for instance in instances:
        require(instance['native_kind'] == 'legacy_sync_put' and
                instance['container']['image'] == BASELINE_IMAGE and
                instance['provenance']['model'] == '/models/Qwen2.5-14B-Instruct',
                'actual 14B native/source identity required')
        require('service_budget_tokens' not in instance and 'restore_budget_tokens' not in instance,
                'legacy residents must not receive v3 budgets')


async def qualify(binding, *, gate_code, out, runtime_dir, lease=None):
    """Caller already acquired the lease and has bootstrapped a mixed binding."""
    binding = read(binding) if isinstance(binding, (str, Path)) else binding
    validate_actual_layout(binding)
    if lease is None:
        require('PDBLEND_NODE_LOCK_FD' in os.environ, 'caller must pass its actual exclusive node lease')
        descriptor = int(os.environ['PDBLEND_NODE_LOCK_FD'])
    else:
        descriptor = lease if isinstance(lease, int) else lease.fileno()
    lock_path = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
    require(os.path.samefile('/proc/self/fd/' + str(descriptor), lock_path), 'foreign node lease descriptor')
    fdinfo = Path('/proc/self/fdinfo/' + str(descriptor)).read_text()
    require(any(line.startswith('lock:') and ' FLOCK ' in line and ' WRITE ' in line
                for line in fdinfo.splitlines()), 'caller does not hold an exclusive node lease')
    adapter.load_runtime(binding['host_release'], Path(binding['executor']).parent)
    gate_code = Path(gate_code).resolve()
    adapter.checked_manifest(gate_code)
    # Load the copied original Checks under its legacy import name, avoiding
    # accidental import of another campaign's module of the same generic name.
    checks = load(gate_code / 'checks.py', 'slo90_legacy_checks')
    previous = sys.modules.get('checks')
    sys.modules['checks'] = checks
    try:
        module = load(gate_code / 'validate.py', 'slo90_legacy_gate')
    finally:
        if previous is None:
            sys.modules.pop('checks', None)
        else:
            sys.modules['checks'] = previous
    return await module.execute(SimpleNamespace(out=Path(out).resolve(), runtime_dir=Path(runtime_dir).resolve()), binding)


def prepare_policy(system, instances, out, *, host_release, engine_entry=None):
    """Freeze historical ShareGPT policy with only actual deployment locations."""
    require(system in adapter.SYSTEMS[1:], 'one of the four declared baselines required')
    out, host_release = Path(out).resolve(), Path(host_release).resolve()
    old_binding = read(OLD_BINDINGS / system / 'binding.json')
    old_path = Path(old_binding['configs']['sharegpt'])
    original = read(old_path)
    require([i['id'] for i in original['instances']] == [i['id'] for i in instances],
            'baseline IDs differ from original policy; explicit reviewed remapping required')
    require(not out.exists(), 'fresh policy destination required')
    out.mkdir(parents=True)
    cfg = copy.deepcopy(original)
    for old, actual in zip(cfg['instances'], instances):
        require((old['tp'], old['gpus']) == (actual['tp'], actual['gpus']), 'policy physical layout changed')
        for key in ('url', 'port', 'kv_port'):
            old[key] = actual[key]
        old['container_name'] = actual['container']['name']
        # Retain old per-system roles, notably DistServe's 2P + 6D split.
    cfg['measurement_window_protocol'] = adapter.PROTOCOL
    cfg['controller_source_release'] = str(host_release)
    if system == 'dynamollm':
        require(engine_entry is not None, 'actual observation engine entry required for full Dynamo lifecycle')
        engine_entry = Path(engine_entry).resolve()
        engine_pythonpath = engine_entry.parent.parent / 'src'
        require(engine_pythonpath.is_dir(), 'frozen observation engine Python source directory missing')
        names = {instance['id']: instance['container']['name'] for instance in instances}
        prefixes = {name[:-len(iid)] for iid, name in names.items() if name.endswith(iid)}
        require(len(prefixes) == 1 and len(names) == len(instances)
                and all(name.endswith(iid) for iid, name in names.items()),
                'one actual task prefix must identify every baseline container')
        prefix = next(iter(prefixes))
        require(prefix.startswith('slo90-') and prefix.endswith('-'), 'Dynamo may only own new task containers')
        template = read(instances[0]['engine_config'])
        template.update(observation_engine_entry=str(engine_entry), observation_engine_sha256=sha(engine_entry),
                        observation_engine_pythonpath=str(engine_pythonpath),
                        observation_container_prefix=prefix, observation_container_names=names)
        write(out / 'engine-template.json', template)
        cfg['topology'].update(runtime_dir=str(out / 'dynamic-runtime'), image=BASELINE_IMAGE,
                               engine_template=str(out / 'engine-template.json'))
    write(out / 'sharegpt.json', cfg)
    allowed = {'measurement_window_protocol', 'controller_source_release', 'instances', 'topology'}
    require({k: v for k, v in original.items() if k not in allowed} ==
            {k: v for k, v in cfg.items() if k not in allowed}, 'baseline policy changed')
    write(out / 'policy-provenance.json', dict(parent=adapter.ref(old_path), system=system,
        original_strategy=original['strategy'], new_config=adapter.ref(out / 'sharegpt.json'),
        deployment_locations_updated=True, request_policy_and_per_system_roles_unchanged=True))
    return adapter.ref(out / 'sharegpt.json')


def build_baseline_bindings(base, *, gate, gate_code, host_release, common_dir, out,
                            engine_entry, deployment_receipt, evidence=(), large_input_paths=()):
    """Read actual raw correctness evidence before publishing any runnable binding."""
    base = read(base) if isinstance(base, (str, Path)) else copy.deepcopy(base)
    validate_actual_layout(base)
    out, gate, gate_code = Path(out).resolve(), Path(gate).resolve(), Path(gate_code).resolve()
    require(not out.exists(), 'fresh baseline binding package required')
    deployed = read(deployment_receipt)
    require(deployed.get('complete') is True and deployed.get('measurement_valid') is True,
            'actual measured deployment must have passed')
    created = {row['name']: row['container_id'] for row in deployed['created']}
    require(created == {i['container']['name']: i['container']['id'] for i in base['instances']},
            'deployment receipt does not describe the actual baseline residents')
    gate_manifest = adapter.checked_manifest(gate_code)
    adapter.load_runtime(host_release, common_dir)
    audit = load(gate_code / 'gate_evidence.py', 'slo90_gate_audit')
    from ecopadg.serving.measurement import power_evidence
    proofs = {}
    raw_files = {}
    for system in adapter.SYSTEMS[1:]:
        proofs[system], raw = audit.audit(gate, base['instances'], system, power_evidence)
        raw_files.update(raw)
    require(all(read(gate / 'status.json')['mechanism_gate'].values()),
            'all ordinary, PD and temporal gates must pass before this baseline stage')
    out.mkdir(parents=True)
    paths = {}
    for system in adapter.SYSTEMS[1:]:
        target = out / system
        cfg = prepare_policy(system, base['instances'], target / 'policy',
                             host_release=host_release, engine_entry=engine_entry)
        actual = copy.deepcopy(base)
        actual.update(system=system, implementation_variant=system)
        binding = adapter.make_binding(actual, host_release=host_release, config=cfg,
            common_dir=common_dir, output=target / 'results',
            large_input_paths=large_input_paths,
            retain_parent_files=True,
            evidence=(*evidence, Path(__file__).resolve(), deployment_receipt,
                      gate_code / 'manifest.json', target / 'policy/policy-provenance.json'),
            frozen_inputs=(*raw_files, *(gate_code / name for name in gate_manifest['files'])))
        binding.update(correctness_evidence=str(gate), deployment_receipt=str(Path(deployment_receipt).resolve()),
                       mechanism_proof=proofs[system], output_correctness_verified=True,
                       correctness_gate_required_before_performance=False)
        write(target / 'binding.json', binding)
        paths[system] = adapter.ref(target / 'binding.json')
    write(out / 'index.json', dict(protocol_id=adapter.PROTOCOL, host=base['hostname'],
        all_fresh_mechanisms_passed=True, bindings=paths))
    return paths
