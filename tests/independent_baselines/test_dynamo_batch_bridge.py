"""Preparation must retain completed B1 cells and collect only the B16 gap."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import asset_preflight
from pdblend_baselines.dynamollm.profile_v1 import measurement_points


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_prepare_dynamo_batch_bridge.py'
SPEC = importlib.util.spec_from_file_location('dynamo_batch_bridge_test', SCRIPT)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


@pytest.fixture
def prior(tmp_path):
    model = 'Qwen2.5-32B-Instruct'
    points = [dict(frequency_mhz=f, input_tokens=n, context_tokens=n+o, batch=1)
              for f in (900,1200,1500,1800,2100,2520) for n in (16,7168) for o in (74,512)]
    (tmp_path / 'profile.json').write_text(json.dumps(dict(model_id=model, tp=2,
        independent_profile=True, measurement='hardware', points=points)))
    (tmp_path / 'completion.json').write_text(json.dumps(dict(status='passed', complete=True,
        cleanup_errors=[], points=24)))
    return tmp_path, model


def test_bridge_is_exact_missing_batch_boundary_and_binds_prior_bytes(prior):
    path, model = prior
    before = {name: (path / name).read_bytes() for name in ('profile.json', 'completion.json')}
    plan = bridge.missing_plan(path, model, 2)
    assert len(plan['points']) == 24 and {p['batch'] for p in plan['points']} == {16}
    assert {(p['frequency_mhz'], p['input_tokens'], p['output_tokens']) for p in plan['points']} == {
        (f,n,o) for f in (900,1200,1500,1800,2100,2520) for n in (16,7168) for o in (74,512)}
    assert not plan['prior_cells_recollect'] and not plan['selection_used_evaluation_outputs']
    assert not plan['formal_eligible'] and 'original stationary-weight retention' in plan['remaining_gates']
    for key, name in (('reused_profile','profile.json'), ('reused_completion','completion.json')):
        assert plan[key] == bridge.ref(path / name)
        assert (path / name).read_bytes() == before[name]
    target = path / 'plan.json'
    target.write_text(json.dumps(plan))
    assert measurement_points(SimpleNamespace(points_file=target, tp=2), model) == plan['points']


@pytest.mark.parametrize('field,value', [('status','failed'), ('complete',False),
                                       ('cleanup_errors',['leaked process']), ('points',23)])
def test_incomplete_or_unclean_prior_cannot_be_reused(prior, field, value):
    path, model = prior
    completion = json.loads((path / 'completion.json').read_text())
    completion[field] = value
    (path / 'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError):
        bridge.missing_plan(path, model, 2)


def test_cross_model_or_missing_corner_cannot_claim_boundary_coverage(prior):
    path, model = prior
    with pytest.raises(ValueError, match='same-model'):
        bridge.missing_plan(path, 'Qwen2.5-7B-Instruct', 1)
    profile = json.loads((path / 'profile.json').read_text())
    profile['points'][-1] = profile['points'][0]
    (path / 'profile.json').write_text(json.dumps(profile))
    with pytest.raises(ValueError, match='24-corner'):
        bridge.missing_plan(path, model, 2)


def test_cpu_preflight_reports_actual_batch_plan_instead_of_hardcoded_one(prior, monkeypatch):
    path, model = prior
    target = path / 'plan.json'
    target.write_text(json.dumps(bridge.missing_plan(path, model, 2)))
    monkeypatch.setattr(asset_preflight, 'model_identity', lambda _: dict(model=model))
    result = asset_preflight.check(dict(kind='profile', model_path='/cpu-fixture', model_id=model,
        tp=2, gpus=[0,1], immutable_inputs={str(target): bridge.ref(target)['sha256']},
        points_file=str(target), source_snapshot='/cpu-fixture-source'))
    assert result['ready'] and result['gpu_started'] is False
    assert result['detail']['batch_values'] == [16]
    assert result['detail']['deferred_batches_above'] == 16
    assert result['detail']['training_repeats'] == 3 and result['detail']['holdout_repeats'] == 1
    assert not result['formal_eligible']


@pytest.mark.parametrize('model', ['Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct'])
def test_target_tp_bounds_do_not_recollect_or_extrapolate_six_old_singletons(tmp_path, model):
    profile = dict(model_id=model, tp=2, independent_profile=True, measurement='hardware',
        points=[dict(frequency_mhz=f,input_tokens=512,context_tokens=576,batch=1)
                for f in (900,1200,1500,1800,2100,2520)])
    (tmp_path/'profile.json').write_text(json.dumps(profile))
    (tmp_path/'completion.json').write_text(json.dumps(dict(status='passed',complete=True,
                                                         cleanup_errors=[],points=6)))
    plan = bridge.target_plan(tmp_path, model)
    assert len(plan['points']) == 48 and {p['batch'] for p in plan['points']} == {1,16}
    assert {(p['input_tokens'],p['output_tokens']) for p in plan['points']} == {
        (n,o) for n in (16,7168) for o in (74,512)}
    assert not plan['prior_cells_recollect'] and not plan['formal_eligible']
    assert all((p['input_tokens'],p['output_tokens'],p['batch']) != (512,64,1) for p in plan['points'])
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert measurement_points(SimpleNamespace(points_file=path,tp=2),model) == plan['points']
    with pytest.raises(ValueError,match='same-model'):
        bridge.target_plan(tmp_path, 'Qwen2.5-32B-Instruct')
