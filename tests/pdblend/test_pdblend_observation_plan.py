from copy import deepcopy
import json

import pytest

from pdblend.bench.pdblend_observation_plan import encode_plan, decode_plan
from pdblend.planner.pool import Plan


def plan():
    return Plan({'M':8},2520,2520,2520,0,float('inf'),float('inf'),float('inf'),
                dict(M=dict(power_w=float('inf'),ttft_s=float('inf'),tpot_s=float('inf'))),tp=1,pp=1)


def test_unknown_predictions_are_lossless_strict_json_and_observation_only():
    original=plan()
    encoded=encode_plan(original)
    serialized=json.loads(json.dumps(encoded,sort_keys=True,allow_nan=False))
    assert serialized['plan']['power_w'] is None
    assert decode_plan(serialized,observation=True)==original
    with pytest.raises(ValueError,match='observation scope'):
        decode_plan(serialized)


@pytest.mark.parametrize('value',[float('nan'),float('-inf')])
def test_non_prediction_nonfinite_values_rejected(value):
    original=plan();original.detail['invalid']=value
    with pytest.raises(ValueError,match='positive-infinite'):
        encode_plan(original)


def test_unannotated_infinity_or_forged_ordinary_field_is_not_accepted():
    encoded=encode_plan(plan())
    encoded['unavailable_estimates']=[row for row in encoded['unavailable_estimates'] if row['path']!='/power_w']
    with pytest.raises(ValueError):decode_plan(encoded,observation=True)
    encoded=encode_plan(plan())
    encoded['plan']['f_M']=None
    encoded['unavailable_estimates'].append(dict(path='/f_M',kind='positive_infinity'))
    with pytest.raises(ValueError,match='path'):
        decode_plan(encoded,observation=True)


def test_finite_legacy_plan_needs_no_new_metadata():
    value=plan();value.power_w=400;value.ttft_s=.4;value.tpot_s=.03;value.detail={}
    encoded=encode_plan(value);del encoded['unavailable_estimates']
    assert decode_plan(encoded)==value
