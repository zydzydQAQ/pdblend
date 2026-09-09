import asyncio
from dataclasses import replace
from pathlib import Path
import pytest

from ecopadg.serving.baselines import DynamoLLMPolicy, ReconfigurationCost
from ecopadg.serving.reconfigure import ResidentRolePlanner, RoleCost
from ecopadg.serving.evidence import mean_ci95, common_capacity, baseline_gaps, validate_freeze, freeze_files
from test_planner import system


def test_dynamo_actual_periods_and_all_nine_shape_classes():
    policy=DynamoLLMPolicy()
    assert policy.due(0)==()
    assert policy.due(4.9)==()
    assert policy.due(5)==('ScaleFreq',)
    assert policy.due(300)==('ScaleShard','ScaleFreq')
    assert policy.due(1800)==('ScaleInst','ScaleShard','ScaleFreq')
    assert len({policy.classify(i,o) for i in (32,512,4096) for o in (32,256,1024)})==9
    assert policy.classify(255,99)=='SS'
    assert policy.classify(256,100)=='MM'
    assert policy.classify(1024,350)=='LL'
    policy.observe_arrival(20,32,256)
    assert policy.forecast(19)['SM']==0  # future observations cannot leak
    assert policy.forecast(20)['SM']==pytest.approx(1/300)


def test_reconfiguration_requires_measured_amortized_cost():
    policy=DynamoLLMPolicy([ReconfigurationCost('ScaleShard',120,10000,'proof')])
    assert not policy.amortizes('ScaleShard',100,130)
    assert policy.amortizes('ScaleShard',100,300)
    assert not policy.amortizes('ScaleInst',10000,300)


def test_resident_role_policy_preserves_source_capacity_and_dwell():
    planner,snapshot,request=system()
    roles=ResidentRolePlanner([RoleCost(1,'mixed','decode',.01,2,'proof')])
    assert roles.plan(snapshot,'decode',100,60,.2,now=10) is not None
    snapshot=replace(snapshot,instances=snapshot.instances+(replace(snapshot.instances[0],instance_id='3',gpus=(3,)),))
    plan=roles.plan(snapshot,'decode',100,60,.2,now=10)
    assert plan.roles[0].savings_lower_j>plan.roles[0].switching_upper_j
    roles.confirmed(plan,10)
    altered=replace(snapshot,instances=tuple(i for i in snapshot.instances if i.instance_id=='0'))
    assert roles.plan(altered,'decode',100,60,.2,now=11) is None


def test_evidence_requires_independent_repeats_and_unchanged_source(tmp_path):
    assert mean_ci95([.1,.1,.1])==(.1,.1)
    assert mean_ci95([.04,.10,.16])[0]<.05
    file=tmp_path/'source.py'
    file.write_text('before')
    frozen=freeze_files([file])
    assert validate_freeze(frozen)==[]
    file.write_text('after')
    assert validate_freeze(frozen)==[str(file)]
    assert set(baseline_gaps({}))=={'distserve','dynamollm','ecoserve','mixed','mixed_dvfs'}
    with pytest.raises(ValueError):
        common_capacity([])
