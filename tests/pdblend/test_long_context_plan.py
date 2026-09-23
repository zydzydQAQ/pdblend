import pytest

from pdblend.profile.long_context_plan import FREQUENCIES, long_context_sampling_plan


def raw(capacity):
    return dict(system='pdblend', model_id='Qwen2.5-32B-Instruct', tp=2, pp=1,
                kv_capacity_tokens=capacity, freqs=list(FREQUENCIES))


def test_long_plan_keeps_only_memory_feasible_shapes_and_full_output_reserve():
    plan = long_context_sampling_plan(raw(44416))
    grid = {(row['batch'], row['context_tokens']) for row in plan['training']}
    assert grid == {(1, 5120), (4, 5120), (6, 5120), (1, 7168), (4, 7168)}
    assert len(plan['training']) == 30 and len(plan['holdout']) == 24
    for row in plan['training'] + plan['holdout']:
        assert row['batch'] * (row['context_tokens'] + row['max_tokens']) <= 44416 * .9
        assert row['repeats'] == 3 and row['settle_s'] == 2 and row['measure_s'] == 5
    assert not plan['fit_existing_holdout'] and not plan['formal_eligible']
    assert '7679' in plan['domain_note']


def test_long_plan_rejects_cross_system_profile():
    other = raw(44416)
    other['system'] = 'distserve'
    with pytest.raises(ValueError, match='independent PDBlend'):
        long_context_sampling_plan(other)
