import json
import math

import pytest

from pdblend_baselines.ecoserve import run_native


def test_load_trace_requires_seed701_and_preserves_real_shapes(tmp_path):
    path = tmp_path/'trace.json'
    path.write_text(json.dumps(dict(seed=701, requests=[dict(arrival_s=0, prompt=[1, 2, 3], max_tokens=2)])))
    value, rows = run_native.load_trace(path, 100)
    assert value['seed'] == 701 and rows[0][1] == [1, 2, 3] and rows[0][2] == 2


@pytest.mark.parametrize('duration', [0, -1, math.nan, math.inf])
def test_service_window_requires_finite_positive_duration(tmp_path, duration):
    with pytest.raises(ValueError, match='service duration'):
        run_native.load_trace(tmp_path/'missing.json', duration)


def test_observation_ticks_and_manual_actions_are_not_automatic_scaling():
    rows = [dict(kind='eco_scale_observation', origin='paper_supplement'),
            dict(kind='eco_scale_macro_observation', origin='paper_supplement'),
            dict(kind='eco_output_flush', origin='paper_supplement'),
            dict(kind='eco_membership_commit', origin='paper_supplement', trigger='explicit_functional')]
    assert run_native.automatic_actions(rows) == []


def test_automatic_commit_requires_native_receipt_and_matching_prepare():
    change = dict(origin='paper_supplement', operation='add', instance_id='b', trigger='mean_ttft', before=[['a']])
    prepare = dict(kind='eco_membership_prepare', **change)
    commit = dict(kind='eco_membership_commit', after=[['a','b']], **change)
    receipt = dict(kind='eco_http_receipt', instance_id='b', path='/baseline/clock', response=dict(acknowledged=True))
    assert run_native.automatic_actions([commit]) == []
    assert run_native.automatic_actions([prepare, commit]) == []
    assert len(run_native.automatic_actions([prepare, receipt, commit])) == 1
