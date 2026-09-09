"""CPU-only host orchestration checks; no deployment or hardware is invoked."""
import copy
import socket

import pytest

import host_executor as host


def stage_fixture(tmp_path, name='pdblend'):
    release = tmp_path / 'release.json'
    host.runner.save(release, {'deployment_root': str(tmp_path / 'deployment')})
    refs = {}
    for system in (['pdblend'] if name == 'pdblend' else host.protocol.BASELINES):
        path = tmp_path / (system + '-binding.json')
        host.runner.save(path, dict(protocol_id=host.protocol.PROTOCOL, system=system, hostname=socket.gethostname()))
        refs[system] = dict(path=str(path), sha256=host.trace_source.file_sha(path))
    result = dict(complete=True, measurement_valid=True, protocol_id=host.protocol.PROTOCOL,
                  stage=name, hostname=socket.gethostname(), bindings=refs,
                  release=dict(path=str(release), sha256=host.trace_source.file_sha(release)))
    return release, result


def test_cached_stage_requires_exact_release_host_and_complete_bindings(tmp_path):
    release, result = stage_fixture(tmp_path, 'baselines')
    host.validate_stage_result(result, release, 'baselines')
    for field in ('hostname', 'protocol_id'):
        bad = dict(result, **{field: 'another'})
        with pytest.raises(ValueError):
            host.validate_stage_result(bad, release, 'baselines')
    bad = copy.deepcopy(result)
    del bad['bindings']['mixed']
    with pytest.raises(ValueError, match='incomplete system bindings'):
        host.validate_stage_result(bad, release, 'baselines')


def test_restoration_marker_blocks_stale_cached_bindings(tmp_path):
    release, result = stage_fixture(tmp_path)
    root = tmp_path / 'deployment'
    host.runner.save(root / 'pdblend/stage-result.json', result)
    host.runner.save(root / 'restore-result.json', {'complete': True})
    with pytest.raises(ValueError, match='old stage bindings cannot be reused'):
        host.stage(release, 'pdblend', tmp_path / 'proof.json', 99)


def test_resume_rechecks_busy_host_before_package_or_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'HERE', tmp_path)
    monkeypatch.setattr(host.dispatcher, 'NODE_LOCK', tmp_path / 'node.lock')
    monkeypatch.setattr(host.dispatcher, 'probe_local', lambda **kwargs: {'eligible_for_owned_start': False})
    monkeypatch.setattr(host.dispatcher, 'current_ancestry', lambda: [])
    monkeypatch.setattr(host, 'verify_package', lambda: pytest.fail('busy host must be declined first'))
    assert host.run('cpu-only-resume', resume=True) == 3
    result = host.runner.read(tmp_path / 'claims/cpu-only-resume/status.json')
    assert result['status'] == 'declined' and not result['gpu_actions']
