import asyncio
import copy
import importlib.util
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slo90_stage_worker_test', HERE / 'stage_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev)


@pytest.fixture
def setup(tmp_path):
    calls = []
    model = tmp_path / 'model.safetensors'
    model.write_bytes(b'actual model fixture')
    retained = tmp_path / 'rank-0.safetensors'
    retained.write_bytes(b'actual retained cache fixture')
    inputs = tmp_path / 'required-inputs.json'
    worker.write(inputs, dict(files={str(path): dict(size=path.stat().st_size, sha256=worker.digest(path))
                                    for path in (model, retained)}))
    source = tmp_path / 'source-lock.json'
    worker.write(source, dict(files={str(inputs): worker.digest(inputs)}))
    original = dict(strategy='pdblend-joint', measurement_window_protocol=worker.PROTOCOL,
                    policy={'deadline': 0.02, 'max_pending': 256}, instances=[dict(id='pdb6', tp=1,
                    gpus=[6], role='mixed', url='http://old:1', port=1, kv_port=2, container_name='old')])
    config = tmp_path / 'original-pdb.json'
    worker.write(config, original)
    release_path = tmp_path / 'release.json'
    release = dict(protocol_id=worker.PROTOCOL, deployment_root=str(tmp_path / 'deployment'),
                   source_lock=worker.ref(source), required_inputs=worker.ref(inputs),
                   configs={system: worker.ref(config) for system in worker.SYSTEMS},
                   host_releases={stage: str(tmp_path / stage) for stage in ('pdblend', 'baselines')},
                   common_dir=str(tmp_path / 'common'), gate_code=str(tmp_path / 'gate'), evidence=[])
    worker.write(release_path, release)
    proof = tmp_path / 'idle-proof.json'
    worker.write(proof, dict(hostname=socket.gethostname(), eligible_for_owned_start=True))
    actual = dict(id='pdb6', tp=1, gpus=[6], url='http://new:100', port=100, kv_port=101,
                  container=dict(name='campaign-pdb6', id='actual-container-id'))
    state = dict(deployment_valid=True, gate_valid=True, cleanup_valid=True, actual=actual,
                 original=original, model=model, baseline_large=True)

    def lease_check(descriptor):
        assert descriptor == 42
        calls.append('lease')

    def make_binding(base, **kwargs):
        calls.append('binding')
        assert kwargs['retain_parent_files'] is True
        base = worker.read(base) if isinstance(base, (str, Path)) else copy.deepcopy(base)
        base['large_inputs'] = {str(path): dict(stat=stat(path)) for path in kwargs['large_input_paths']}
        base['host_release'] = kwargs['host_release']
        base['executor'] = str(Path(kwargs['common_dir']) / 'run.py')
        base['files'] = {str(path): worker.digest(path) for path in kwargs.get('evidence', [])}
        return base

    def validate(binding):
        assert set(binding['large_inputs']) == {str(model), str(retained)}
        calls.append('validate')

    def prepare_spec(stage, release, hostname, idle_proof, previous_binding=None):
        calls.append('prepare-' + stage)
        path = tmp_path / 'deployment' / stage / 'spec.json'
        worker.write(path, dict(stage=stage, hostname=hostname, runtime_dir=str(tmp_path / 'native'),
                                source_entry=str(tmp_path / 'engine.py')))
        return worker.ref(path)

    async def execute(spec_path, out, *, run, lease):
        assert run is True and lease == 42 and not Path(out).exists()
        calls.append('deploy')
        stage = worker.read(spec_path)['stage']
        status = dict(complete=True, measurement_valid=state['deployment_valid'], created=[])
        worker.write(Path(out) / 'deployment-receipt.json', status)
        worker.write(Path(out) / 'binding-base.json', dict(model='14b', hostname=socket.gethostname(),
              system='pdblend' if stage == 'pdblend' else 'mixed', instances=[state['actual']]))
        return status

    async def gate(common, binding, out, *, lease):
        assert lease == 42
        calls.append('gate')
        status = dict(complete=True, passed=state['gate_valid'], measurement_valid=state['gate_valid'],
                      native_cleanup_complete=state['cleanup_valid'], clock_restore_complete=True)
        worker.write(Path(out) / 'status.json', status)
        worker.write(Path(out) / 'raw-request.json', dict(output=[1, 2, 3]))
        return status

    def prepare_policy(system, instances, out, **kwargs):
        assert system == 'mixed'
        calls.append('mixed-bootstrap-policy')
        worker.write(Path(out) / 'sharegpt.json', dict(instances=instances))
        worker.write(Path(out) / 'policy-provenance.json', dict(policy_unchanged=True))
        return worker.ref(Path(out) / 'sharegpt.json')

    async def qualify(binding, *, gate_code, out, runtime_dir, lease):
        return await gate(None, binding, out, lease=lease)

    def build(base, *, out, large_input_paths, **kwargs):
        calls.append('build-four')
        result = {}
        for system in worker.SYSTEMS[1:]:
            binding = worker.read(base)
            binding.update(system=system, output_correctness_verified=True,
                large_inputs={str(path): dict(stat=stat(path)) for path in large_input_paths}
                             if state['baseline_large'] else {})
            path = Path(out) / system / 'binding.json'
            worker.write(path, binding)
            result[system] = worker.ref(path)
        return result

    async def restore_original(release, *, run, lease):
        assert run and lease == 42
        calls.append('restore')
        return dict(complete=True, measurement_valid=True, restored=['original-container-id'])

    deps = SimpleNamespace(lease_check=lease_check,
        adapter=SimpleNamespace(checked_manifest=lambda p: dict(protocol_id=worker.PROTOCOL),
            stat_identity=stat, make_binding=make_binding,
            load_runtime=lambda *a: SimpleNamespace(validate_binding=validate)),
        deploy=SimpleNamespace(prepare_spec=prepare_spec, execute=execute, measured_ordinary=gate,
                               restore_original=restore_original),
        baseline=SimpleNamespace(validate_actual_layout=lambda b: None, prepare_policy=prepare_policy,
                                 qualify=qualify, build_baseline_bindings=build))
    return SimpleNamespace(path=release_path, release=release, proof=proof, deps=deps, calls=calls,
                           state=state, root=tmp_path, config=config, inputs=inputs)


def run_stage(setup, stage='pdblend', previous=None):
    return asyncio.run(worker.execute_stage(stage, setup.path, setup.proof, previous,
                                           lease=42, deps=setup.deps))


def test_actual_pdb_gate_then_frozen_binding_with_every_large_input(setup):
    result = run_stage(setup)
    assert result['complete'] and result['measurement_valid']
    assert set(result['bindings']) == {'pdblend'}
    assert setup.calls.index('deploy') < setup.calls.index('gate') < len(setup.calls) - 1
    base = Path(setup.release['deployment_root']) / 'pdblend'
    assert (base / 'deployment-receipt.json').read_bytes() == (base / 'deployment/deployment-receipt.json').read_bytes()
    binding = worker.read(worker.checked_ref(result['bindings']['pdblend']))
    assert binding['output_correctness_verified'] is True
    assert len(binding['large_inputs']) == 2
    assert worker.read(setup.config) == setup.state['original']
    remapped = worker.read(base / 'policy/sharegpt.json')
    assert remapped['policy'] == setup.state['original']['policy']
    assert remapped['instances'][0]['container_name'] == 'campaign-pdb6'
    assert remapped['instances'][0]['role'] == 'mixed'


@pytest.mark.parametrize('failure', ['deployment_valid', 'gate_valid', 'cleanup_valid'])
def test_failed_deployment_or_measured_cleanup_never_publishes_binding(setup, failure):
    setup.state[failure] = False
    result = run_stage(setup)
    assert result['complete'] and not result['measurement_valid'] and not result['bindings']
    assert 'error' in result
    assert not (Path(setup.release['deployment_root']) / 'pdblend/bindings').exists()
    if failure == 'deployment_valid':
        assert 'gate' not in setup.calls


def test_model_content_change_rejected_before_any_engine_action(setup):
    setup.state['model'].write_bytes(b'changed model fixture')
    result = run_stage(setup)
    assert not result['measurement_valid']
    assert 'deploy' not in setup.calls and 'prepare-pdblend' not in setup.calls
    assert 'model or retained cache content changed' in result['error']['message']


def test_actual_layout_cannot_change_frozen_pdb_gpu_selection(setup):
    setup.state['actual']['gpus'] = [5]
    result = run_stage(setup)
    assert not result['measurement_valid']
    assert 'physical layout' in result['error']['message']
    assert 'gate' not in setup.calls


def test_baselines_share_fresh_gate_and_all_four_keep_model_cache_freeze(setup):
    previous = setup.root / 'pdb-binding.json'
    worker.write(previous, dict(hostname=socket.gethostname(), system='pdblend'))
    result = run_stage(setup, 'baselines', previous)
    assert result['measurement_valid']
    assert set(result['bindings']) == set(worker.SYSTEMS[1:])
    assert setup.calls.index('gate') < setup.calls.index('build-four')
    for reference in result['bindings'].values():
        assert len(worker.read(reference['path'])['large_inputs']) == 2


def test_baseline_missing_large_inputs_invalidates_stage_even_after_gate(setup):
    previous = setup.root / 'pdb-binding.json'
    worker.write(previous, dict(hostname=socket.gethostname(), system='pdblend'))
    setup.state['baseline_large'] = False
    result = run_stage(setup, 'baselines', previous)
    assert not result['measurement_valid'] and result['bindings'] == {}
    assert 'omitted model/cache' in result['error']['message']


def test_baselines_require_real_previous_same_host_binding(setup):
    result = run_stage(setup, 'baselines')
    assert not result['measurement_valid'] and 'deploy' not in setup.calls


def test_no_inherited_lease_has_no_side_effects(setup, monkeypatch):
    monkeypatch.delenv('PDBLEND_NODE_LOCK_FD', raising=False)
    with pytest.raises(ValueError, match='inherited node lease'):
        asyncio.run(worker.execute_stage('pdblend', setup.path, setup.proof, deps=setup.deps))
    assert not Path(setup.release['deployment_root']).exists()


def test_retained_stage_is_not_automatically_retried(setup):
    setup.state['gate_valid'] = False
    run_stage(setup)
    before = list(setup.calls)
    with pytest.raises(ValueError, match='fresh stage destination'):
        run_stage(setup)
    assert setup.calls == before + ['lease']


def test_restore_uses_actual_deployment_owner_and_keeps_stable_result(setup):
    result = asyncio.run(worker.restore(setup.path, lease=42, deps=setup.deps))
    assert result['measurement_valid'] and result['restored'] == ['original-container-id']
    assert worker.read(Path(setup.release['deployment_root']) / 'restore-result.json') == result
    with pytest.raises(ValueError, match='already attempted'):
        asyncio.run(worker.restore(setup.path, lease=42, deps=setup.deps))
