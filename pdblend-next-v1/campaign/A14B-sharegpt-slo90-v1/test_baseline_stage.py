import ast
import asyncio
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('slo90_baseline_stage_tested', HERE / 'baseline_stage.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def instances():
    return [dict(id='base100ar' + str(j), tp=1, gpus=[j], url='http://127.0.0.1:' + str(36000+j),
        port=36000+j, kv_port=55000+j*32, native_kind='legacy_sync_put',
        engine_config=str(stage.adapter.REPO / 'campaign/AC-baseline-deployment-prepared-v1/A-resident/engines' /
                          ('base100ar' + str(j) + '.json')),
        container=dict(name='pdb-v2-base100ar' + str(j), image=stage.BASELINE_IMAGE),
        provenance=dict(model='/models/Qwen2.5-14B-Instruct')) for j in range(8)]


def test_fresh_gate_preserves_actual_checks_and_raw_auditor(tmp_path):
    stage.adapter.prepare_common(tmp_path / 'common')
    manifest = stage.prepare_gate(tmp_path / 'gate', tmp_path / 'common')
    assert stage.sha(tmp_path / 'gate/checks.py') == stage.sha(stage.PARENT_GATE / 'checks.py')
    assert stage.sha(tmp_path / 'gate/gate_evidence.py') == stage.sha(stage.PARENT_AUDIT)
    original = functions((stage.PARENT_GATE / 'validate.py').read_text())
    adapted = functions((tmp_path / 'gate/validate.py').read_text())
    assert {k for k in original if original[k] != adapted[k]} == {'execute', 'validate_scope'}
    assert manifest['work_timeout_s'] == 390 and manifest['cleanup_timeout_s'] == 90


@pytest.mark.parametrize('system', ['mixed', 'distserve', 'ecoserve'])
def test_policy_preserves_scheduling_and_roles(system, tmp_path):
    values = instances()
    for item in values:
        item['port'] += 1000
        item['url'] = 'http://127.0.0.1:' + str(item['port'])
    result = stage.prepare_policy(system, values, tmp_path / system, host_release=tmp_path / 'runtime')
    cfg = stage.read(result['path'])
    old = stage.read(stage.read(stage.OLD_BINDINGS / system / 'binding.json')['configs']['sharegpt'])
    assert [i['role'] for i in cfg['instances']] == [i['role'] for i in old['instances']]
    assert cfg['measurement_window_protocol'] == stage.adapter.PROTOCOL
    assert cfg['instances'][0]['port'] == 37000


def test_foreign_model_and_remapped_ids_rejected():
    binding = dict(model='14b', hostname='new-actual-host', instances=instances())
    stage.validate_actual_layout(binding)
    binding['instances'][0]['provenance']['model'] = '/models/Qwen2.5-7B-Instruct'
    with pytest.raises(ValueError):
        stage.validate_actual_layout(binding)


def test_qualification_rejects_unlocked_descriptor_without_acquiring_it(tmp_path, monkeypatch):
    binding = dict(model='14b', hostname='cpu-only-host', instances=instances())
    monkeypatch.setattr(stage.os.path, 'samefile', lambda *args: True)
    with (tmp_path / 'cpu-only-lock').open('w') as lease:
        with pytest.raises(ValueError, match='does not hold an exclusive node lease'):
            asyncio.run(stage.qualify(binding, gate_code=tmp_path / 'unused',
                                      out=tmp_path / 'never-created', runtime_dir=tmp_path, lease=lease))
        assert 'FLOCK' not in Path('/proc/self/fdinfo/' + str(lease.fileno())).read_text()
    assert not (tmp_path / 'never-created').exists()


def test_dynamic_full_strategy_requires_actual_engine_template(tmp_path):
    with pytest.raises(ValueError, match='actual observation engine'):
        stage.prepare_policy('dynamollm', instances(), tmp_path / 'dynamo', host_release=tmp_path / 'runtime')


def test_full_dynamo_binds_actual_observation_entry_and_new_runtime(tmp_path):
    values = instances()
    config = tmp_path / 'actual-engine.json'
    stage.write(config, dict(model='/models/Qwen2.5-14B-Instruct', runtime_dir=str(tmp_path / 'actual-engine-runtime')))
    for item in values:
        item['engine_config'] = str(config)
        item['container']['name'] = 'slo90-14b-baselines-' + item['id']
    entry = tmp_path / 'frozen-native/engines/engine.py'
    entry.parent.mkdir(parents=True)
    (entry.parent.parent / 'src').mkdir()
    entry.write_text('# frozen 14B engine observation entry\n')
    result = stage.prepare_policy('dynamollm', values, tmp_path / 'dynamo',
        host_release=tmp_path / 'host', engine_entry=entry)
    cfg = stage.read(result['path'])
    assert cfg['strategy'] == 'dynamollm'
    assert cfg['topology']['runtime_dir'] == str(tmp_path / 'dynamo/dynamic-runtime')
    template = stage.read(cfg['topology']['engine_template'])
    assert template['observation_engine_entry'] == str(entry)
    assert template['observation_engine_sha256'] == stage.sha(entry)
    assert template['observation_engine_pythonpath'] == str(entry.parent.parent / 'src')
    assert template['observation_container_prefix'] == 'slo90-14b-baselines-'
    assert template['observation_container_names'] == {i['id']: i['container']['name'] for i in values}


def test_full_dynamo_rejects_missing_frozen_native_python_tree(tmp_path):
    entry = tmp_path / 'frozen-native/engines/engine.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('# frozen CPU fixture\n')
    with pytest.raises(ValueError, match='Python source directory missing'):
        stage.prepare_policy('dynamollm', instances(), tmp_path / 'dynamo',
                             host_release=tmp_path / 'host', engine_entry=entry)
