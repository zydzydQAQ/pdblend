import copy
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[2]/'scripts/2026-09-23_prepare_long_context_phase1.py'
spec=importlib.util.spec_from_file_location('longctx_phase1',path)
phase1=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase1)


def fixture():
    plan=dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1,
        training_source='/immutable/raw.json', training_source_sha256='a'*64, fit_existing_holdout=False,
        training=[dict(freq_mhz=f,context_tokens=c,batch=b,repeats=3,settle_s=2,measure_s=5,
                       max_tokens=1024,purpose='training_extension')
                  for f in phase1.FREQUENCIES for c in (5120,7168) for b in (1,4,8,256)],
        holdout=[dict(freq_mhz=f,context_tokens=c,batch=b,repeats=3,settle_s=2,measure_s=5,
                      max_tokens=1024,purpose='independent_holdout_after_candidate_freeze')
                 for f in phase1.FREQUENCIES for c,b in ((6144,2),(6144,22),(7168,1),(7168,256))])
    raw=dict(system='pdblend',model_id=plan['model_id'],tp=4,pp=1,
             prefill=[dict(freq_mhz=f,input_tokens=n,seconds=s,concurrency=1)
                      for f in phase1.FREQUENCIES for n,s in ((4096,.5),(7168,.8))])
    return plan,raw


def test_low_batch_phase_preserves_original_and_excludes_unmeasured_corner():
    plan,raw=fixture();before=copy.deepcopy(plan)
    selected,report=phase1.build_phase1(plan,raw,original_path='/immutable/plan.json',original_sha256='b'*64)
    assert plan==before
    assert len(selected['training'])==36 and len(selected['deferred_training'])==12
    assert len(selected['holdout'])==12 and len(selected['deferred_holdout'])==12
    assert {x['batch'] for x in selected['training']}=={1,4,8}
    assert all(x['coverage_status']=='missing_profile' for x in selected['deferred_training'])
    assert selected['full_training_matrix_complete'] is False
    assert selected['coverage_constraints']['forbid_rectangular_domain_union_with_short_context_high_batch']
    assert len(report['rows'])==48 and report['phase1']['points']==36
    assert report['full_plan']['scheduling_proxy_seconds']==pytest.approx(
        report['phase1']['scheduling_proxy_seconds']+report['deferred']['scheduling_proxy_seconds'])
    first=report['rows'][0]
    assert first['single_request_prefill_seconds']==pytest.approx(.6)
    assert first['scheduling_proxy_seconds']==pytest.approx(22.8)
    assert not report['actual_batch_runtime_measured']


def test_costs_never_extrapolate_or_accept_wrong_identity():
    plan,raw=fixture()
    with pytest.raises(ValueError,match='extrapolate'):
        phase1.prefill_proxy(raw,900,8192)
    raw['tp']=2
    with pytest.raises(ValueError,match='identity|independent'):
        phase1.build_phase1(plan,raw,original_path='/plan',original_sha256='b'*64)


def test_reduced_plan_cannot_be_claimed_as_the_original_complete_matrix():
    plan,raw=fixture();plan['training'].pop()
    with pytest.raises(ValueError,match='48-point'):
        phase1.build_phase1(plan,raw,original_path='/plan',original_sha256='b'*64)
