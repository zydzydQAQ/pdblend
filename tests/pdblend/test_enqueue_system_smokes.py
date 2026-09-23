import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[2] / 'scripts/2026-09-22_enqueue_system_smokes.py'
    spec = importlib.util.spec_from_file_location('enqueue_system_smokes', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_only_real_runners_are_queued_with_group_scope_and_seed701(tmp_path):
    enq = module()
    profile = tmp_path / 'profile.json'
    profile.write_text('{}')
    profiles = {(f'Qwen2.5-{m}-Instruct', tp): profile for m, tp in (('7B', 1), ('14B', 1), ('32B', 2))}
    jobs = enq.build_specs(snapshot=tmp_path, source_hash='abc', image='sha256:xyz',
                           verification=tmp_path/'receipt.json', profiles=profiles)
    assert len(jobs) == 6
    assert {j['payload']['system'] for j in jobs} == {'pdblend', 'mixed'}
    for j in jobs:
        p = j['payload']
        assert p['gpu_count'] == 2*p['tp'] and not p['exclusive']
        assert p['seed'] == 701 and j['priority'] > 0
        assert not p['formal_eligible'] and not p['energy_comparable']
        assert '--seed' in p['argv']
        if p['system'] == 'mixed':
            assert '--profile' not in p['argv'] and p['profile_sha256'] is None
        else:
            assert '--profile' in p['argv'] and p['profile_sha256']


def test_missing_model_profile_never_falls_back_to_another_model(tmp_path):
    with pytest.raises(ValueError, match='missing own measured PDBlend profile'):
        module().build_specs(snapshot=tmp_path, source_hash='abc', image='sha256:xyz',
                             verification=tmp_path/'receipt.json', profiles={})
