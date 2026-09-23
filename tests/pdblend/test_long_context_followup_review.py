"""Independent raw-evidence regression checks for the incremental review."""
import copy
import importlib.util
from pathlib import Path

import pytest
from pdblend.profile import long_context_followup as lf, long_context_collect as lc
from pdblend.profile.identity import sha256_value
from pdblend.profile.wave import atomic_json

_spec=importlib.util.spec_from_file_location('long_followup_review_fixture',Path(__file__).with_name('test_long_context_followup.py'))
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)


@pytest.mark.asyncio
async def test_training_rejects_reusing_one_window_as_three_repeats(tmp_path):
    package,profiler=await _fixture.fixture(tmp_path)
    manifest,_,raw=lf.load_package(package)
    row=raw['decode'][0]
    row['repeats']=[copy.deepcopy(row['repeats'][0]) for _ in range(3)]
    for index,rep in enumerate(row['repeats']):rep['repeat']=index
    raw['identity_sha256']=sha256_value({k:v for k,v in raw.items() if k!='identity_sha256'})
    plan=__import__('json').loads((profiler.out_dir/'training-plan.json').read_text())
    with pytest.raises(ValueError):
        lf.verify_training(raw,profiler.out_dir,plan['training'],manifest['inputs']['training-plan.json']['sha256'])


@pytest.mark.asyncio
async def test_window_frequency_claim_must_match_immutable_raw(tmp_path):
    package,profiler=await _fixture.fixture(tmp_path)
    _,_,raw=lf.load_package(package)
    rep=raw['decode'][0]['repeats'][0]
    path=profiler.out_dir/rep['samples_file']
    evidence=__import__('json').loads(path.read_text())
    evidence['frequency']=[[time,[800]*len(values)] for time,values in evidence['frequency']]
    atomic_json(path,evidence)
    rep['samples_sha256']=lc.digest(path)
    with pytest.raises(ValueError):
        lc.validate_repeat(profiler.out_dir,rep,expected_point=evidence['point'],expected_plan_sha256=evidence['plan_sha256'])


@pytest.mark.asyncio
async def test_endpoint_window_must_contain_target_context_not_only_exceed_it(tmp_path):
    from types import SimpleNamespace
    candidate=dict(kind=lf.KIND,exact_batches=[1],nodes={'900/1':[
        dict(context=6000,step_seconds=.025,power_w=100),dict(context=8000,step_seconds=.025,power_w=100)]})
    point=dict(freq_mhz=900,batch=1,context_tokens=7600,max_tokens=512,repeats=3,settle_s=2,measure_s=5,
        purpose='independent_holdout_repair',long_context_role='campaign_endpoint')
    plan=dict(candidate_sha256=sha256_value(candidate),points=[point],missing_points=[],expected_points=1)
    clock=_fixture.Clock()
    profiler=SimpleNamespace(out_dir=tmp_path,tp=1,parallel_layout={},
        raw=dict(kv_capacity_tokens=200000,measurement_plan_sha256='plan',profile_key={}),
        meter=SimpleNamespace(sampler=lambda g:_fixture.Sampler(clock,1)))
    row=await lc.collect_bounded_decode_point(profiler,None,[0],point,purpose='independent_holdout_repair',
        _clock=clock,_sleep=clock.sleep,_background_factory=clock.background)
    assert all(rep['observed_context_min']>7679 for rep in row['repeats'])
    receipt=tmp_path/'qualification.json';atomic_json(receipt,dict(complete=True,cross_job=True,passed=True))
    qualifier=lc.digest(receipt)
    for rep in row['repeats']:rep['qualification_sha256']=qualifier
    raw=dict(measurement_plan_sha256='plan',decode=[row],
        qualification_history={qualifier:dict(samples_file=receipt.name,samples_sha256=qualifier)})
    checked=lf.audit(candidate,raw,tmp_path,plan)
    assert not checked['passed'],'a measurement entirely after 7679 cannot claim it observed context 7679'
