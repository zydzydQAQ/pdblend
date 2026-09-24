"""Restart tests protect identities, immutable inputs and fresh queue ownership."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def entrypoint(name='2026-09-25_reproduce_v3.py'):
    path = Path(__file__).resolve().parents[2] / 'scripts' / name
    spec = importlib.util.spec_from_file_location('reproduction_test_entry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign_fixture(root):
    module = entrypoint()
    source = root / 'sources' / 'snapshot'
    source.mkdir(parents=True)
    (source / 'runtime.py').write_text('pass\n')
    module.write(source / 'manifest.json', dict(source_sha256='snapshot', files={'runtime.py': module.sha(source / 'runtime.py')}))
    refs = {}
    for name, value in [('verification', {}), ('trace', {}), ('planning_trace', {}),
                        ('base', {}), ('evidence', {'historical': True}),
                        ('choice', {}), ('config', {})]:
        module.write(root / (name + '.json'), value)
        refs[name] = module.binding(root / (name + '.json'))
    module.write(root / 'compiled.json', dict(source_bindings=[refs['evidence']]))
    module.write(root / 'profile.json', dict(kind='pdblend_development_composite_profile_v1',
        base_profile=refs['base'], compiled=module.binding(root / 'compiled.json')))
    module.write(root / 'execution.json', dict(model_verification=refs['verification']))
    points = [dict(name=f'7b-pdblend-{dataset}-x{scale:g}-seed701', system='pdblend',
        seed=701, duration_s=150, inputs=dict(trace=refs['trace'], planning_trace=refs['planning_trace'],
        offline_choice=refs['choice'], system_config=refs['config'], profiles=[module.binding(root / 'profile.json')],
        source_manifest=module.binding(source / 'manifest.json')))
        for dataset in ('alpaca', 'sharegpt', 'longbench') for scale in (.25, .5, .75, 1.)]
    data = dict(execution_inputs=module.binding(root / 'execution.json'),
        groups=[dict(model_id='Qwen2.5-7B-Instruct', points=points)],
        # Stale historical state is deliberately not a runtime input.
        points=[dict(name='old-failed-attempt')], queue=str(root / 'old-queue.json'))
    module.write(root / 'campaign.json', data)
    (root / 'old-queue.json').write_text('{"leases":{"live-source-machine":{}}}')
    return module, root / 'campaign.json'


def test_runtime_inventory_preserves_profile_evidence_and_excludes_live_queue(tmp_path):
    module, path = campaign_fixture(tmp_path)
    result = module.runtime_assets(path, ('7b',))
    names = {Path(row['path']).name for row in result['files']}
    assert {'evidence.json', 'base.json', 'compiled.json', 'runtime.py'} <= names
    assert 'old-queue.json' not in names
    assert result['includes_queue_state'] is False
    assert result['formal_eligible'] is False


def test_runtime_inventory_refuses_changed_profile_component(tmp_path):
    module, path = campaign_fixture(tmp_path)
    (tmp_path / 'evidence.json').write_text('{"historical":false}')
    with pytest.raises(ValueError, match='changed runtime asset'):
        module.runtime_assets(path, ('7b',))


def test_target_uuid_remapping_preserves_tensor_parallel_placement_and_old_identity():
    module = entrypoint()
    before = dict(fleet_gpu_uuids=[f'old-{i}' for i in range(8)],
        instances=[dict(instance_id=f'pd{i}', tp=2, pp=1, gpu_uuids=[f'old-{2*i}', f'old-{2*i+1}']) for i in range(4)])
    saved = deepcopy(before)
    result = module.rebind_identity(before, [f'new-{i}' for i in range(8)])
    assert before == saved
    assert result['instances'][3]['gpu_uuids'] == ['new-6', 'new-7']
    assert not any('old-' in uuid for row in result['instances'] for uuid in row['gpu_uuids'])
    with pytest.raises(ValueError, match='distinct'):
        module.rebind_identity(before, ['new-0'] * 8)


def test_selection_uses_active_groups_without_historical_attempts(tmp_path):
    module, path = campaign_fixture(tmp_path)
    data = json.loads(path.read_text())
    result = module.selected_groups(data, ('7b',))
    assert len(result[0]['points']) == 12
    result[0]['points'][0]['name'] = 'changed-copy'
    assert data['groups'][0]['points'][0]['name'] != 'changed-copy'
    data['groups'][0]['points'].append(deepcopy(data['groups'][0]['points'][0]))
    with pytest.raises(ValueError, match='twelve standard'):
        module.selected_groups(data, ('7b',))


def test_start_refuses_reusing_existing_queue_before_any_execution(tmp_path, monkeypatch):
    module = entrypoint()
    uuids = [f'target-{i}' for i in range(8)]
    module.write(tmp_path / 'campaign.json', dict(execution=dict(gpu_uuids=uuids)))
    module.write(tmp_path / 'jobs.json', [])
    module.write(tmp_path / 'preparation.json', dict(cpu_image_preflight_passed=True,
        campaign=module.binding(tmp_path / 'campaign.json'), jobs=module.binding(tmp_path / 'jobs.json')))
    (tmp_path / 'queue.json').write_text('source server active leases')
    monkeypatch.setattr(module, 'target_gpus', lambda: uuids)
    with pytest.raises(FileExistsError, match='queue exists'):
        module.start(tmp_path)
    assert (tmp_path / 'queue.json').read_text() == 'source server active leases'


def test_baseline_groups_never_mix_independent_source_revisions(monkeypatch):
    from pdblend.bench import resident_session
    monkeypatch.setattr(resident_session, 'engine_signature', lambda value: 'same-engine')
    module = entrypoint('2026-09-25_reproduce_baselines_v3.py')
    points = [dict(name=f'7b-mixed-{dataset}-x{scale:g}-seed701', system='mixed',
        model_id='Qwen2.5-7B-Instruct', engine_identity={'fleet': 'eight'},
        source_manifest={'sha256': 'source-' + ('a' if scale < .75 else 'b')})
        for dataset in ('alpaca', 'sharegpt', 'longbench') for scale in (.25, .5, .75, 1.)]
    groups = module.groups_for(dict(points=points), ('7b',), ('mixed',))
    assert len(groups) == 2
    assert sum(len(g['points']) for g in groups) == 12
    assert all(len({p['source_manifest']['sha256'] for p in g['points']}) == 1 for g in groups)
    groups[0]['points'][0]['name'] = 'new-attempt'
    assert points[0]['name'] != 'new-attempt'
