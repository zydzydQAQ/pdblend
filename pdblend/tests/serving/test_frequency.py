from dataclasses import replace
import json

import pytest

from ecopadg.serving.frequency import FrequencyCost,FrequencyPlanner,verify_frozen_costs
from test_planner import system


def active_system():
    estimator,snapshot,request=system()
    old=replace(request,emitted=10,first_token_s=9.5,last_token_s=10.)
    instance=replace(snapshot.instances[0],requests=(old,),running=1)
    return estimator,replace(snapshot,instances=(instance,))


def test_mixed_decode_can_downclock_after_prefill_with_measured_cost():
    estimator,snapshot=active_system()
    costs=[FrequencyCost(1,2520,f,.01,2,'measured') for f in (900,1500)]
    optimizer=FrequencyPlanner(estimator,costs)
    plan=optimizer.plan(snapshot,10)
    assert plan.frequencies[0].frequency_mhz==1500  # 900 MHz uses more energy
    costly=FrequencyPlanner(estimator,[replace(c,energy_upper_j=10000) for c in costs])
    assert not costly.plan(snapshot,10).frequencies
    assert not FrequencyPlanner(estimator,()).plan(snapshot,10).frequencies


def test_one_request_without_slack_blocks_batch_downclock():
    estimator,snapshot=active_system();instance=snapshot.instances[0]
    risky=replace(instance.requests[0],request_id='risk',first_token_s=9.04)
    snapshot=replace(snapshot,instances=(replace(instance,requests=instance.requests+(risky,)),))
    optimizer=FrequencyPlanner(estimator,[FrequencyCost(1,2520,1500,.1,2,'measured')])
    assert not optimizer.plan(snapshot,10).frequencies
    waiting=replace(instance,requests=(replace(risky,emitted=0,first_token_s=None),))
    assert not optimizer.plan(replace(snapshot,instances=(waiting,)),10).frequencies


def test_instant_serving_rejects_average_clock_cost_evidence(tmp_path):
    from ecopadg.serving.evidence import sha256
    path=tmp_path/'raw.json'
    path.write_text(json.dumps(dict(complete=True,prefix_matches_reference=True,
        frequency_samples=[[1,[2520]*8]],engine_provenance=[dict(image_id='image')],
        power_source=dict(mode='average'),power_samples=[[1,[100]*8]],
        switches=[dict(tp=1,source_mhz=2520,target_mhz=1500,started_s=1,finished_s=1.1,energy_j=5)])))
    digest=sha256(path)
    config=dict(power_mode='instant',frequency_evidence=[str(path)],
        frequency_costs=[dict(tp=1,source_mhz=2520,target_mhz=1500,
            duration_upper_s=.2,energy_upper_j=10,source_sha256=digest)])
    freeze=dict(groups=dict(profiles=[str(path)]),files={str(path):digest},identities=dict(engine_image='image'))
    with pytest.raises(ValueError,match='verified instantaneous power'):
        verify_frozen_costs(config,dict(points=[]),freeze)
