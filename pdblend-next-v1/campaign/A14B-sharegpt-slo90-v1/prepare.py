"""CPU-only immutable source, policy and input preparation for all five systems."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import baseline_stage
import protocol
import runtime_adapter as a
import trace_source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUALIFICATION = REPO / 'campaign/parallel-rate-20260908-v1/A/final-p4-audit.json'
PDB_CONFIG = a.PDB_RELEASE.parent / 'configs/sharegpt.json'


def stable_version(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tree_files(directory):
    return [p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.suffix not in ('.pyc', '.pyo')]


def verify_prepared_versions(release):
    """A seal cannot silently give changed prepared sources an old version ID."""
    common = Path(release['common_dir'])
    a.checked_manifest(common)
    a.checked_manifest(Path(release['gate_code']))
    common_files = {str(path): a.sha(path) for path in tree_files(common)}
    for system in protocol.SYSTEMS:
        runtime = Path(release['host_releases']['pdblend' if system == 'pdblend' else 'baselines'])
        a.checked_manifest(runtime)
        engine = Path(release['engine_source_release']) if system == 'pdblend' else Path(release['baseline_engine_pythonpath']).parent
        cfg = release['configs'][system]
        a.require(a.sha(cfg['path']) == cfg['sha256'], 'prepared configuration changed: ' + system)
        files = {str(path): a.sha(path) for root in (runtime, engine) for path in tree_files(root)}
        files.update(common_files)
        files[cfg['path']] = cfg['sha256']
        files.update({str(path): a.sha(path) for path in a.configuration_inputs(a.read(cfg['path']))})
        a.require(stable_version(files) == release['versions'][system],
                  'prepared source changed under existing version: ' + system)
    a.require(a.sha(release['source_lock']['path']) == release['source_lock']['sha256'], 'preparation source-lock changed')
    a.require(a.sha(release['model_manifest']['path']) == release['model_manifest']['sha256'], 'model inventory changed')
    if release.get('required_inputs'):
        a.require(a.sha(release['required_inputs']['path']) == release['required_inputs']['sha256'], 'required input inventory changed')


def copy_manifest(parent, target):
    manifest = a.checked_manifest(parent)
    target.mkdir(parents=True)
    for name in manifest['files']:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('relative frozen source required')
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, destination)
    a.write(target / 'manifest.json', dict(parent=a.ref(parent / 'manifest.json'),
        files={str(p.relative_to(target)): a.sha(p) for p in tree_files(target)}))


def prepare(out):
    out = Path(out).resolve()
    a.require(not out.exists(), 'new preparation directory required')
    # Verify the selected passed ShareGPT source before writing anything. The
    # newer load-dependent Alpaca candidate is not a ShareGPT qualification.
    old_release = a.read(a.PDB_RELEASE)
    a.require(old_release.get('approved') is True and old_release.get('model') == '14b'
              and Path(old_release['host_release']).resolve() == a.PDB_PARENT.resolve(),
              'selected PDB source is not the approved 14B parent')
    for name, digest in old_release['files'].items():
        a.require(a.sha(name) == digest, 'qualified parent changed: ' + name)
    a.require(QUALIFICATION.is_file(), 'final measured p4 audit required')
    source = trace_source.load_source()
    out.mkdir(parents=True)
    runtime_pdb, runtime_base = out / 'runtimes/pdblend', out / 'runtimes/baselines'
    a.prepare_host(a.PDB_PARENT, runtime_pdb)
    a.prepare_host(a.BASELINE_PARENT, runtime_base)
    common = out / 'common'
    a.prepare_common(common)
    gate = out / 'gate'
    baseline_stage.prepare_gate(gate, common)
    configs = {'pdblend': a.prepare_config(PDB_CONFIG, out / 'configs/pdblend.json')}
    external = {Path(ref['path']) for ref in source['references']}
    external |= {a.PDB_RELEASE, PDB_CONFIG, QUALIFICATION}
    external.add(REPO / 'campaign/AC-baseline-deployment-prepared-v1/A-resident/deployment.json')
    external |= a.configuration_inputs(a.read(PDB_CONFIG))
    for system in protocol.BASELINES:
        old_binding = baseline_stage.OLD_BINDINGS / system / 'binding.json'
        config_path = Path(a.read(old_binding)['configs']['sharegpt'])
        configs[system] = a.prepare_config(config_path, out / ('configs/' + system + '.json'))
        external |= {old_binding, config_path}
        external |= a.configuration_inputs(a.read(config_path))
    # Engine source is isolated from ongoing campaigns. Its actual import
    # provenance is re-observed by deployment and bound on the selected host.
    engine_source = out / 'sources/io-v3-runtime'
    copy_manifest(REPO / 'releases/io-v3-runtime', engine_source)
    baseline_source = out / 'sources/baseline-native'
    original_entry = REPO / 'campaign/AC-baseline-deployment-v1/engines/A/engine.py'
    for path in original_entry.parent.glob('*.py'):
        dest = baseline_source / 'engines' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    native_src = Path('/root/workspace/pdblend/src')
    for path in tree_files(native_src):
        dest = baseline_source / 'src' / path.relative_to(native_src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for path in (REPO / 'campaign/A14B-engine-v3').glob('engine-*.json'):
        if path.name in ('engine-6.json', 'engine-7.json'):
            external.add(path)
    for path in (REPO / 'campaign/AC-baseline-deployment-prepared-v1/A-resident/engines').glob('base100ar*.json'):
        external.add(path)
    qualification = Path(old_release['profile_refs']['sharegpt']['path']) if 'path' in old_release['profile_refs']['sharegpt'] else None
    if qualification:
        external.add(qualification)
    profile_gate = REPO / 'campaign/main-slo-improvement-v1/A/long-batch6-profile-001/qualification.json'
    external.add(profile_gate)
    ordinary_source = REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py'
    external.add(ordinary_source)
    source_lock = {str(path.resolve()): a.sha(path) for path in sorted(external)}
    a.write(out / 'source-lock.json', dict(created_s=time.time(), files=source_lock,
        qualification_selection='latest measured ShareGPT-qualified p4 fixed2 before this preparation',
        qualification_audit=a.ref(QUALIFICATION), no_historical_measurements_reused=True))
    common_files = {str(path): a.sha(path) for path in tree_files(common)}
    versions = {}
    for system in protocol.SYSTEMS:
        runtime = runtime_pdb if system == 'pdblend' else runtime_base
        engine = engine_source if system == 'pdblend' else baseline_source
        files = {str(path): a.sha(path) for root in (runtime, engine) for path in tree_files(root)}
        files.update(common_files)
        files.update({configs[system]['path']: configs[system]['sha256']})
        files.update({str(path): a.sha(path) for path in a.configuration_inputs(a.read(configs[system]['path']))})
        versions[system] = stable_version(files)
    release = dict(schema=1, protocol_id=protocol.PROTOCOL, created_s=time.time(),
        deployment_root=str(HERE / 'execution/deployment'), common_dir=str(common),
        host_releases={'pdblend': str(runtime_pdb), 'baselines': str(runtime_base)},
        gate_code=str(gate), configs=configs, pdb_config=configs['pdblend']['path'],
        engine_source_release=str(engine_source), baseline_engine_entry=str(baseline_source / 'engines/engine.py'),
        baseline_engine_pythonpath=str(baseline_source / 'src'), versions=versions,
        evidence=[str(a.PDB_RELEASE), str(QUALIFICATION), str(profile_gate), str(out / 'source-lock.json')],
        model_root='/root/workspace/models', source_lock=a.ref(out / 'source-lock.json'),
        container_prefix='slo90-14b', model_manifest=a.ref(HERE / 'asset-staging/C/model-source.json'),
        required_inputs=a.ref(HERE / 'asset-staging/required-inputs.json'))
    a.write(out / 'release.json', release)
    return release


def seal(release_path):
    """Seal only after all execution modules and their integration checks pass."""
    release_path = Path(release_path).resolve()
    release = a.read(release_path)
    verify_prepared_versions(release)
    source = a.read(release['source_lock']['path'])['files']
    for path, digest in source.items():
        a.require(a.sha(path) == digest, 'source changed since preparation: ' + path)
    files = dict(source)
    files.update({str(p): a.sha(p) for p in tree_files(release_path.parent)})
    files.update({str(p): a.sha(p) for p in HERE.glob('*.py')})
    files[release['model_manifest']['path']] = release['model_manifest']['sha256']
    files[release['required_inputs']['path']] = release['required_inputs']['sha256']
    files[str(HERE / 'README.md')] = a.sha(HERE / 'README.md')
    files[str(HERE / 'EXPERIMENT.md')] = a.sha(HERE / 'EXPERIMENT.md')
    a.write(HERE / 'package.json', dict(protocol_id=protocol.PROTOCOL, created_s=time.time(),
        release=a.ref(release_path), files=files, version_ids=release['versions']))
    return dict(files=len(files), bytes=sum(Path(path).stat().st_size for path in files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--seal', type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.seal) if args.seal else {'versions': prepare(args.out)['versions']}))
