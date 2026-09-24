"""CPU-only preparation guards; no GPU jobs or capacity certificates created."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_repaired_standard_matrix.py'
    spec = importlib.util.spec_from_file_location('standard_preparation_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent(script):
    points = []
    for name in script.CASES:
        size, _, dataset, scale, _ = name.split('-')
        points.append(dict(name=name, system='pdblend', model_id='Qwen2.5-'+size.upper()+'-Instruct',
            dataset=dataset, scale=float(scale[1:]), seed=701, duration_s=150.))
    return dict(points=points)


def test_default_matrix_is_exactly_original_36_and_subset_is_group_ordered(script):
    campaign = parent(script)
    campaign['points'].append(dict(name='7b-pdblend-sharegpt-x1.953125-seed701'))
    selected = script.select_cases(campaign)
    assert len(selected) == 36 and [row['name'] for row in selected] == list(script.CASES)
    requested = list(reversed(script.FLOOR_CASES))
    assert [row['name'] for row in script.select_cases(campaign, requested)] == list(script.FLOOR_CASES)


@pytest.mark.parametrize('fault', ['extension', 'duplicate', 'missing', 'wrong_seed', 'wrong_scale'])
def test_standard_selection_rejects_relabelled_or_missing_points(script, fault):
    campaign = parent(script); requested = list(script.CASES)
    if fault == 'extension': requested.append('7b-pdblend-sharegpt-x1.953125-seed701')
    elif fault == 'duplicate': requested.append(requested[0])
    elif fault == 'missing': campaign['points'].pop()
    elif fault == 'wrong_seed': campaign['points'][0]['seed'] = 8801
    else: campaign['points'][0]['scale'] = 1.25
    with pytest.raises(ValueError):
        script.select_cases(campaign, requested)


def frozen_source(script, tmp_path):
    name = 'pdblend/bench/comparison_runtime.py'; contents = b'# frozen CPU fixture\n'
    files = {name:hashlib.sha256(contents).hexdigest()}
    root = tmp_path/script._digest(files)
    target = root/name; target.parent.mkdir(parents=True); target.write_bytes(contents)
    path = root/'manifest.json'; path.write_text(json.dumps(dict(source_sha256=root.name, files=files)))
    return path, target


@pytest.mark.parametrize('fault', ['changed_byte', 'extra_file', 'symlink', 'wrong_directory'])
def test_source_inventory_cannot_silently_import_a_changed_algorithm(script, tmp_path, fault):
    path, target = frozen_source(script, tmp_path)
    assert script.verify_source(path)[0] == path
    if fault == 'changed_byte': target.write_text('# changed\n')
    elif fault == 'extra_file': (path.parent/'extra.py').write_text('# extra\n')
    elif fault == 'symlink': (path.parent/'alias.py').symlink_to(target)
    else:
        moved = tmp_path/'renamed'; path.parent.rename(moved); path=moved/'manifest.json'
    with pytest.raises(ValueError):
        script.verify_source(path)


def capacity_case(script):
    point = next(row for row in parent(script)['points'] if row['name'] in script.FLOOR_CASES)
    profile = dict(path='/bound-profile.json', sha256='profile')
    point.update(rate_rps=2., slo=dict(ttft_s=5., tpot_s=.15), inputs=dict(profiles=[profile]))
    family = dict(model_id=point['model_id'], dataset='sharegpt',
        corpus_manifest=dict(path='/bound-corpus.json', sha256='corpus'),
        corpus_dataset=dict(path='/bound-sharegpt.json', sha256='dataset'))
    context = dict(algorithm_source_sha256='code', workload_family_sha256='family',
                   recovery_policy_sha256='recovery', nominal_rate_rps=2.)
    capacity = SimpleNamespace(artifact=dict(path='/bound-floor.json',sha256='floor'),
        manifest=dict(profile=profile,workload_identity=family),
        floors=[SimpleNamespace(context=context,frequency_mhz=1800)])
    return point, capacity, context


def test_only_corresponding_nominal_certificate_binds_and_unknown_shapes_remain_guarded(script, monkeypatch):
    point, capacity, context = capacity_case(script)
    monkeypatch.setattr('pdblend.bench.pdblend_runtime_options.capacity_workload_context',
                        lambda *args:context)
    options = {}
    result = script.capacity_binding(point, capacity, options)
    assert options['capacity_workload_binding'] == dict(capacity.manifest['workload_identity'], nominal_rate_rps=2.)
    assert options['capacity_floor_path'] == capacity.artifact
    assert result['unknown_shape_action'] == 'restore_canonical_M4'
    point = deepcopy(point); point['name']='7b-pdblend-sharegpt-x0.5-seed701'; point['rate_rps']=4.
    options = {}
    assert script.capacity_binding(point, capacity, options) is None and options == {}
    point['name']='14b-pdblend-sharegpt-x0.5-seed701'
    assert script.capacity_binding(point, capacity, options) is None and options == {}


@pytest.mark.parametrize('fault', ['rate', 'profile', 'slo', 'context'])
def test_capacity_cannot_cross_nominal_profile_slo_or_algorithm_domain(script, monkeypatch, fault):
    point, capacity, context = capacity_case(script)
    if fault == 'rate': point['rate_rps']=1.
    elif fault == 'profile': point['inputs']['profiles']=[]
    elif fault == 'slo': point['slo']['ttft_s']=50.
    else: context={}
    monkeypatch.setattr('pdblend.bench.pdblend_runtime_options.capacity_workload_context',
                        lambda *args:context)
    with pytest.raises(ValueError):
        script.capacity_binding(point, capacity, {})


def test_stored_capacity_pass_flags_cannot_replace_raw_v2_evidence(script, tmp_path):
    from pdblend.bench.comparison_campaign import binding
    manifest = tmp_path/'tuning.json'; manifest.write_text(json.dumps(dict(passed=True,qualified=True)))
    artifact = tmp_path/'floor.json'
    artifact.write_text(json.dumps(dict(kind='pdblend_capacity_floor_v2', passed=True, qualified=True,
                                       tuning_manifest=binding(manifest), floors=[])))
    with pytest.raises(ValueError, match='independent tuning scope'):
        script.qualified_capacity(artifact, tmp_path, 2100)
