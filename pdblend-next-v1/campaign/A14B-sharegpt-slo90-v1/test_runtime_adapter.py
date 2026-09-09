"""CPU checks: scope gates and source equivalence, never GPU actions."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slo90_runtime_adapter_tested', HERE / 'runtime_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def gate(source):
    tree = ast.parse(source)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_fixed_window'
            or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('FIXED_WINDOW_PROTOCOL', 'COMPARISON_SYSTEMS', 'WINDOW_STRATEGIES', 'DATASET_SLOS') for t in n.targets)]
    namespace = dict(EVALUATION_V3='evaluation-v3', REQUEST_TIMEOUT_S=120, math=__import__('math'))
    exec(compile(ast.Module(body=keep, type_ignores=[]), '<scope-test>', 'exec'), namespace)
    return namespace['configure_fixed_window']


def inputs(scale=.5):
    args = SimpleNamespace(split='development', dataset='sharegpt', seed=701,
                           slo_scale=scale, slo_ttft_s=5 * scale, slo_tpot_s=.15 * scale)
    cfg = dict(measurement_window_protocol=adapter.PROTOCOL, evaluation_protocol='evaluation-v3',
               strategy='pdblend-joint', arrival_window_s=100)
    trace = dict(model='14b', protocol_id=adapter.PROTOCOL, measurement_schema=3,
                 arrival_window_s=100, duration_s=100, split='development', dataset='sharegpt',
                 seed=701, comparison_systems=list(adapter.SYSTEMS), request_hard_timeout_s=120,
                 post_window_drain_allowance_s=120, rate_rps=99, n_requests=1,
                 requests=[dict(arrival_s=0)], prompts=['full prompt'])
    return args, cfg, trace


@pytest.mark.parametrize('parent', [adapter.PDB_PARENT, adapter.BASELINE_PARENT])
def test_only_fixed_window_function_changes(parent):
    source = (parent / adapter.CELL).read_text()
    before, after = functions(source), functions(adapter.transform_cell(source))
    assert {k for k in before if before[k] != after[k]} == {'configure_fixed_window'}


@pytest.mark.parametrize('scale', [.5, 2.])
def test_effective_thresholds_and_arbitrary_rate(scale):
    configure = gate(adapter.transform_cell((adapter.PDB_PARENT / adapter.CELL).read_text()))
    args, cfg, trace = inputs(scale)
    window = configure(args, cfg, trace)
    assert window['effective_slo_s'] == dict(ttft=5 * scale, tpot=.15 * scale)
    assert cfg['slo_attainment_target'] == .9


@pytest.mark.parametrize('change', [dict(model='7b'), dict(dataset='alpaca'), dict(seed=1701),
    dict(rate_rps=0), dict(rate_rps=float('inf')), dict(request_hard_timeout_s=121),
    dict(post_window_drain_allowance_s=121), dict(protocol_id=adapter.OLD_PROTOCOL)])
def test_bad_trace_scope_rejected(change):
    configure = gate(adapter.transform_cell((adapter.PDB_PARENT / adapter.CELL).read_text()))
    args, cfg, trace = inputs()
    trace.update(change)
    with pytest.raises(ValueError):
        configure(args, cfg, trace)


def test_scale_one_rejected():
    configure = gate(adapter.transform_cell((adapter.PDB_PARENT / adapter.CELL).read_text()))
    with pytest.raises(ValueError):
        configure(*inputs(1.))


def test_common_measurement_cleanup_and_identity_byte_ast_identical():
    source = (adapter.COMMON_PARENT / 'run.py').read_text()
    before, after = functions(source), functions(adapter.transform_common(source))
    assert {k for k in before if before[k] != after[k]} == {'validate_binding', 'sweep'}
    assert before['run_one'] == after['run_one']


def test_baseline_dynamic_launch_binds_prepared_pythonpath(tmp_path):
    source = (adapter.BASELINE_PARENT / adapter.TOPOLOGY).read_text()
    adapted = adapter.transform_topology(source)
    def methods(text):
        return {node.name + '.' + method.name: ast.dump(method, include_attributes=False)
                for node in ast.parse(text).body if isinstance(node, ast.ClassDef)
                for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))}
    before, after = methods(source), methods(adapted)
    assert {key for key in before if before[key] != after[key]} == {
        'DockerLifecycle.__init__', 'DockerLifecycle.start', 'DockerLifecycle.stop'}
    assert set(after) - set(before) == {'DockerLifecycle.container_name'}
    result = adapter.prepare_host(adapter.BASELINE_PARENT, tmp_path / 'host')
    assert result['changed_runtime_files'] == [adapter.CELL, adapter.TOPOLOGY]
    assert (tmp_path / 'host' / adapter.TOPOLOGY).read_text() == adapted


def test_dynamic_lifecycle_actual_and_new_names_stay_within_task(tmp_path):
    import asyncio
    import hashlib
    import json
    source = adapter.transform_topology((adapter.BASELINE_PARENT / adapter.TOPOLOGY).read_text())
    tree = ast.parse(source)
    lifecycle = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DockerLifecycle')
    namespace = dict(asyncio=asyncio, hashlib=hashlib, json=json, Path=Path)
    exec(compile(ast.Module(body=[lifecycle], type_ignores=[]), '<frozen-topology-cpu-test>', 'exec'), namespace)
    cls = namespace['DockerLifecycle']
    template = dict(observation_container_prefix='slo90-14b-baselines-',
                    observation_container_names={'base100ar0': 'slo90-14b-baselines-base100ar0'},
                    observation_engine_pythonpath='/frozen/native/src')
    owner = cls(tmp_path / 'runtime', 'frozen-image', template)
    commands = []
    async def command(*args, **kwargs):
        commands.append(args)
        return 'cpu-only'
    owner.command = command
    for iid in ('base100ar0', 'dynamic-new-v1'):
        spec = SimpleNamespace(instance_id=iid, tp=1, port=36000, kv_port=55000,
                               gpus=(0,), role='mixed', generation=1)
        asyncio.run(owner.start(spec, {}))
        asyncio.run(owner.stop(spec))
        name = 'slo90-14b-baselines-' + iid
        assert any(args[:3] == ('run', '-d', '--name') and args[3] == name for args in commands)
        assert any(args[:2] == ('rm', '-f') and args[2] == name for args in commands)
    assert all('pdb-v2-base100ar0' not in args for args in commands)
    assert any('PYTHONPATH=/frozen/native/src' in args for args in commands)
    with pytest.raises(ValueError, match='escapes this task prefix'):
        cls(tmp_path / 'bad', 'image', dict(template, observation_container_names={'base100ar0': 'pdb-v2-base100ar0'}))
    with pytest.raises(ValueError, match='independent task container prefix'):
        cls(tmp_path / 'bad-prefix', 'image', dict(template, observation_container_prefix='pdb-v2-'))


def test_prepare_keeps_parent_immutable_and_refuses_overwrite(tmp_path):
    parent_hash = adapter.sha(adapter.PDB_PARENT / adapter.CELL)
    result = adapter.prepare_host(adapter.PDB_PARENT, tmp_path / 'host')
    assert result['changed_runtime_files'] == [adapter.CELL]
    assert adapter.sha(adapter.PDB_PARENT / adapter.CELL) == parent_hash
    assert adapter.checked_manifest(tmp_path / 'host') == result
    with pytest.raises(ValueError):
        adapter.prepare_host(adapter.PDB_PARENT, tmp_path / 'host')
    common = adapter.prepare_common(tmp_path / 'common')
    assert common['unchanged_child']
    assert adapter.sha(tmp_path / 'common/child.py') == adapter.sha(adapter.COMMON_PARENT / 'child.py')


def test_cross_host_binding_drops_foreign_stats_and_freezes_actual_inputs(tmp_path):
    adapter.prepare_host(adapter.PDB_PARENT, tmp_path / 'host')
    adapter.prepare_common(tmp_path / 'common')
    profile = tmp_path / 'profile.json'
    adapter.write(profile, {'actual_profile': True})
    source = tmp_path / 'engine.py'
    source.write_text('# actual imported source\n')
    engine_cfg = tmp_path / 'engine.json'
    adapter.write(engine_cfg, {'model': '/models/Qwen2.5-14B-Instruct'})
    instance = dict(id='nextv3a6', tp=1, gpus=[6], url='http://127.0.0.1:24306',
                    engine_config=str(engine_cfg), provenance={'source_files_at_import': {str(source): adapter.sha(source)}})
    base = dict(model='14b', system='pdblend', hostname='new-actual-host', instances=[instance],
                files={'/old/foreign/path': 'not-copied'},
                large_inputs={'/old/foreign/model': {'stat': {'inode': 123}}})
    config = tmp_path / 'config.json'
    adapter.write(config, dict(measurement_window_protocol=adapter.PROTOCOL, profiles=str(profile),
                               instances=[{k: instance[k] for k in ('id', 'tp', 'gpus', 'url')}]))
    result = adapter.make_binding(base, host_release=tmp_path / 'host', config=config,
        common_dir=tmp_path / 'common', output=tmp_path / 'results', large_input_paths=[profile])
    assert '/old/foreign/path' not in result['files']
    assert '/old/foreign/model' not in result['large_inputs']
    assert result['large_inputs'][str(profile)]['stat'] == adapter.stat_identity(profile)
    assert result['files'][str(source)] == adapter.sha(source)
    assert result['files'][str(profile)] == adapter.sha(profile)
    assert result['hostname'] == 'new-actual-host'
    assert result['deadline_s'] is None
