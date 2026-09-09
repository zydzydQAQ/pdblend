"""Deployment CPU checks use real templates and simulated hardware interfaces."""
import asyncio
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slo90_deployment_tested', HERE / 'deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def container(name='previous', cid='original-id', *, running=True):
    return dict(Name='/' + name, Id=cid, Image='original-image',
                State=dict(Running=running, StartedAt='old-start', Pid=123),
                Config={}, HostConfig={}, Mounts=[])


def previous():
    return dict(instances=[dict(container=dict(name='previous', id='original-id', image='original-image', StartedAt='old-start'))])


def test_inventory_rejects_foreign_running_and_never_replaces_existing_name():
    own = [dict(container_name='slo90-new')]
    assert deploy.verify_inventory(previous(), [container()], own)[0]['Id'] == 'original-id'
    with pytest.raises(ValueError, match='unexpected running'):
        deploy.verify_inventory(previous(), [container(), container('foreign', 'foreign')], own)
    with pytest.raises(ValueError, match='already exists'):
        deploy.verify_inventory(previous(), [container(), container('slo90-new', 'old-run', running=False)], own)
    with pytest.raises(ValueError, match='process changed'):
        deploy.verify_inventory(previous(), [container(cid='recreated')], own)


def test_docker_launch_preserves_evidence_and_uses_exact_independent_name():
    instance = dict(container_name='slo90-14b-pdblend-nextv3a6', environment=['CUDA_VISIBLE_DEVICES=6'],
        mounts=[dict(Type='bind', Source='/source', Destination='/models', RW=False)],
        image=deploy.PDB_IMAGE, command=['python3', '-m', 'ecopadg.serving.engine', '--config', '/frozen.json'])
    argv = deploy.docker_start_arguments(instance)
    assert '--rm' not in argv and 'rm' not in argv
    assert argv[argv.index('--name') + 1] == instance['container_name']
    assert argv[-6:] == [deploy.PDB_IMAGE] + instance['command']
    assert '/source:/models:ro' in argv
    assert 'com.openai.slo90=' + deploy.adapter.PROTOCOL in argv


def test_lease_requires_actual_target_file(tmp_path, monkeypatch):
    lock = tmp_path / 'node.lock'
    other = tmp_path / 'other.lock'
    lock.touch()
    other.touch()
    monkeypatch.setattr(deploy, 'LOCK_PATH', lock)
    with lock.open('a') as stream:
        with pytest.raises(ValueError, match='does not already hold'):
            deploy.require_lease(stream)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert deploy.require_lease(stream) == stream.fileno()
    with other.open('a') as stream:
        with pytest.raises(ValueError, match='foreign lease'):
            deploy.require_lease(stream)


@pytest.mark.parametrize('stage,count', [('pdblend', 2), ('baselines', 8)])
def test_prepare_uses_qualified_templates_and_isolates_runtime(stage, count, tmp_path, monkeypatch):
    roots = {}
    for name in ('pdblend', 'baselines', 'common'):
        root = tmp_path / name
        root.mkdir()
        deploy.write(root / 'manifest.json', dict(files={}))
        roots[name] = str(root)
    release = dict(deployment_root=str(tmp_path / 'deployment'), common_dir=roots['common'],
        host_releases={key: roots[key] for key in ('pdblend', 'baselines')})
    asset = deploy.assets_needed(release)
    monkeypatch.setattr(deploy, 'freeze_assets', lambda _: (asset, {}, {}))
    idle = tmp_path / 'idle-proof.json'
    deploy.write(idle, dict(eligible=True, host='actual-target'))
    result = deploy.prepare_spec(stage, release, 'actual-target', idle)
    frozen = deploy.read(result['path'])
    assert len(frozen['instances']) == count
    assert frozen['hostname'] == 'actual-target'
    for instance in frozen['instances']:
        cfg = deploy.read(instance['config'])
        assert cfg['runtime_dir'] == str(tmp_path / 'deployment' / stage / 'engine-runtime')
        assert cfg['retained_weights'] == str(deploy.RETAINED)
        assert cfg['max_num_batched_tokens'] == 8192 and cfg['max_model_len'] == 8192
        assert instance['container_name'].startswith('slo90-14b-' + stage + '-')
        assert instance['expected_provenance']['source_files_at_import']
    with pytest.raises(ValueError, match='fresh deployment spec'):
        deploy.prepare_spec(stage, release, 'actual-target', idle)


def test_cpu_execute_does_not_acquire_lease_or_import_hardware(tmp_path, monkeypatch):
    path = tmp_path / 'spec.json'
    deploy.write(path, dict(cpu_fixture=True))
    monkeypatch.setattr(deploy, 'validate_spec', lambda _: None)
    monkeypatch.setattr(deploy, 'require_lease', lambda _: pytest.fail('must not acquire a lease'))
    result = asyncio.run(deploy.execute(path, tmp_path / 'out', run=False))
    assert result == dict(cpu_only=True, hardware_actions=False, spec_valid=True)
    assert not (tmp_path / 'out').exists()


def test_capture_reads_actual_predecessor_and_rejects_live_work():
    path = deploy.REPO / 'campaign/A14B-engine-v3/engine-6.json'
    cfg = deploy.read(path)
    row = container()
    row['Config'] = dict(Cmd=['python3', '-m', 'ecopadg.serving.engine', '--config', str(path)],
                         Env=['CUDA_VISIBLE_DEVICES=6'])
    source = HERE / 'deploy.py'
    provenance = dict(instance_id='nextv3a6', tp=1, model=cfg['model'], cuda_visible_devices='6',
                      pid=987, source_files_at_import={str(source): deploy.sha(source)})
    class Common:
        active = 0
        async def http(self, session, instance, route):
            return provenance if route == '/provenance' else dict(id='nextv3a6', generation=9,
                acknowledged_generation=9, transfer_send_counters_observed=True, scheduler_io=[], active=self.active)
        def idle(self, raw, instance):
            if raw['active']:
                raise RuntimeError('active predecessor')
    common = Common()
    result = asyncio.run(deploy.capture_previous(None, common, [row]))
    assert result['instances'][0]['container']['id'] == 'original-id'
    assert result['instances'][0]['provenance']['pid'] == 987
    assert result['instances'][0]['native_kind'] == 'v3'
    common.active = 1
    with pytest.raises(RuntimeError, match='active predecessor'):
        asyncio.run(deploy.capture_previous(None, common, [row]))


def test_retained_original_identity_checks_allow_only_runtime_restart_fields():
    original = container()
    current = copy.deepcopy(original)
    current['State'].update(Pid=456, StartedAt='new-start')
    deploy.verify_original_container(current, original)
    current['Image'] = 'different-image'
    with pytest.raises(ValueError, match='Image'):
        deploy.verify_original_container(current, original)


def test_restore_cpu_check_never_mutates_or_claims_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, 'require_lease', lambda _: pytest.fail('must not acquire a lease'))
    result = asyncio.run(deploy.restore_original(dict(deployment_root=str(tmp_path)), run=False))
    assert result['hardware_actions'] is False
    assert 'restored' not in result
    assert not (tmp_path / 'restoration-001').exists()


def test_receipt_alias_must_be_identical(tmp_path):
    deploy.write(tmp_path / 'pdblend/deployment-receipt.json', {'one': 1})
    deploy.write(tmp_path / 'pdblend/deployment/deployment-receipt.json', {'one': 2})
    with pytest.raises(ValueError, match='alias differs'):
        deploy._stage_receipt(tmp_path, 'pdblend')
