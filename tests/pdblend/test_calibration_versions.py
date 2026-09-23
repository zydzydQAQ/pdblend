import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('calibration_versions',ROOT/'scripts/2026-09-23_calibration_versions.py')
versions=importlib.util.module_from_spec(spec);spec.loader.exec_module(versions)


def test_source_binding_detects_changed_code_and_path_escape(tmp_path):
    source=tmp_path/'identity';source.mkdir();module=source/'module.py';module.write_text('original')
    manifest=dict(source_sha256=source.name,files={'module.py':versions.sha(module)})
    (source/'manifest.json').write_text(json.dumps(manifest))
    assert versions.verify_source(source)=='identity'
    module.write_text('changed')
    with pytest.raises(ValueError,match='implementation changed'):
        versions.verify_source(source)
    manifest['files']={'../outside.py':'x'};(source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='implementation changed'):
        versions.verify_source(source)


def test_version_cannot_replace_existing_evidence(tmp_path):
    path=tmp_path/'registry.json';versions.immutable(path,dict(complete=False))
    versions.immutable(path,dict(complete=False))
    with pytest.raises(ValueError,match='immutable calibration'):
        versions.immutable(path,dict(complete=True))
    assert versions.read(path)==dict(complete=False)


def test_sampling_completeness_and_sha_are_separate_gates(tmp_path):
    with pytest.raises(ValueError,match='incomplete'):
        versions.require_complete(dict(status='passed',complete=False),'power')
    path=tmp_path/'raw.json';path.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        versions.require_checksum(path,'wrong')


def test_real_frozen_power_and_timing_revalidation():
    """Use actual frozen CPU audit code and real archived raw samples, no GPUs."""
    if not versions.SOURCE.is_dir():
        pytest.skip('archived campaign source unavailable')
    registry=versions.build_registry(versions.read(versions.QUEUE))
    assert len(registry['versions'])==2
    seven,large=registry['versions']
    assert seven['power']['fresh_windows']==72 and seven['effective_timing_passed']
    assert large['timing']['original_timing_passed'] is False
    assert large['effective_timing_passed'] is True
    assert large['timing']['fresh_windows']==54
    assert large['timing']['fresh_long_context_qualified'] is False
    assert large['timing']['fresh_batches']==[24,32,48]
    assert large['consumer']['planner_registry_wired'] is False
    assert all(not row['formal_eligible'] and not row['full_profile_qualified'] for row in registry['versions'])
