"""CPU tests for immutable preparation/version identity."""
from pathlib import Path

import pytest

import prepare


def fixture_release(tmp_path):
    roots = {}
    for name in ('common', 'gate', 'pdb-runtime', 'base-runtime'):
        root = tmp_path / name
        root.mkdir()
        (root / 'code.py').write_text('# CPU fixture source\n')
        prepare.a.write(root / 'manifest.json', dict(files={'code.py': prepare.a.sha(root / 'code.py')}))
        roots[name] = root
    pdb_engine, base_engine = tmp_path / 'pdb-engine', tmp_path / 'base-engine'
    pdb_engine.mkdir()
    (pdb_engine / 'engine.py').write_text('# CPU engine fixture\n')
    (base_engine / 'src').mkdir(parents=True)
    (base_engine / 'src/native.py').write_text('# CPU native fixture\n')
    configs = {}
    for system in prepare.protocol.SYSTEMS:
        path = tmp_path / (system + '.json')
        prepare.a.write(path, {})
        configs[system] = prepare.a.ref(path)
    lock, model = tmp_path / 'source-lock.json', tmp_path / 'model.json'
    prepare.a.write(lock, {'files': {}})
    prepare.a.write(model, {'cpu_only': True})
    release = dict(common_dir=str(roots['common']), gate_code=str(roots['gate']),
        host_releases={'pdblend': str(roots['pdb-runtime']), 'baselines': str(roots['base-runtime'])},
        engine_source_release=str(pdb_engine), baseline_engine_pythonpath=str(base_engine / 'src'),
        configs=configs, source_lock=prepare.a.ref(lock), model_manifest=prepare.a.ref(model), versions={})
    common_files = {str(path): prepare.a.sha(path) for path in prepare.tree_files(roots['common'])}
    for system in prepare.protocol.SYSTEMS:
        runtime = roots['pdb-runtime' if system == 'pdblend' else 'base-runtime']
        engine = pdb_engine if system == 'pdblend' else base_engine
        files = {str(path): prepare.a.sha(path) for root in (runtime, engine) for path in prepare.tree_files(root)}
        files.update(common_files)
        files[configs[system]['path']] = configs[system]['sha256']
        release['versions'][system] = prepare.stable_version(files)
    return release


def test_seal_rejects_source_mutation_under_existing_version(tmp_path):
    release = fixture_release(tmp_path)
    prepare.verify_prepared_versions(release)
    (Path(release['baseline_engine_pythonpath']) / 'native.py').write_text('# changed after preparation\n')
    with pytest.raises(ValueError, match='prepared source changed under existing version'):
        prepare.verify_prepared_versions(release)


def test_seal_rejects_changed_configuration_reference(tmp_path):
    release = fixture_release(tmp_path)
    Path(release['configs']['pdblend']['path']).write_text('{"changed":true}')
    with pytest.raises(ValueError, match='prepared configuration changed'):
        prepare.verify_prepared_versions(release)
