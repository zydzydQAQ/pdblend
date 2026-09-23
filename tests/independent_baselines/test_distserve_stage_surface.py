import json

import numpy as np
import pytest

from pdblend_baselines.distserve.stage_surface import (StageSurface, _Hull, coverage_features,
                                                      features, fit_surface, measured_events)
from pdblend_baselines.native_profile import DIST_FREQS


def sample():
    return dict(ranks=[dict(rank=rank,samples=[dict(system='distserve',role='decode',measurement_scope='runner',
        tp=2,pp=1,rank=rank,batch=2,request_ids=['a','b'],prompt_lengths=[128,512],
        context_lengths=[133,517],scheduled_lengths=[1,1],gpu_elapsed_ms=10+rank,at_s=1.)]) for rank in range(2)])


def test_heterogeneous_measurement_uses_slowest_rank_and_exact_context_vectors():
    raw=sample();rows=measured_events(raw,tp=2)
    assert rows[0]['lengths']==[133,517] and rows[0]['latency_ms']==11
    raw['ranks'][1]['samples'][0]['context_lengths'][0]+=1
    with pytest.raises(ValueError,match='alignment'):measured_events(raw,tp=2)
    raw=sample();del raw['ranks'][0]['samples'][0]['context_lengths']
    with pytest.raises(ValueError,match='heterogeneous'):measured_events(raw,tp=2)


def fitted():
    train_shapes=[[128],[1024],[128,896],[384,640],[128]*4,[1024]*4]
    hold_shapes=[[512],[256,768]]
    def rows(shapes,purpose):
        result=[]
        for frequency in DIST_FREQS:
            for role in ('prefill','decode'):
                for i,lengths in enumerate(shapes):
                    x=[1,*features(lengths)]
                    result.append(dict(frequency_mhz=frequency,role=role,lengths=lengths,
                        latency_ms=float(np.dot(x,[1,2,3,4])),power_w=float(np.dot(x,[20,3,6,2])),
                        window_id=f'{purpose}-{frequency}-{role}-{i}'))
        return result
    train,hold=rows(train_shapes,'train'),rows(hold_shapes,'hold')
    return fit_surface(train,hold,identity=dict(system='distserve',model_id='Qwen2.5-7B-Instruct',tp=1,pp=1),
                       raw_bindings=[],measurement_qualification=dict(passed=True))


def test_independent_fit_interpolates_actual_heterogeneous_shape_but_never_extrapolates():
    artifact=fitted();assert artifact['qualified']
    provider=StageSurface(artifact)
    expected=float(np.dot([1,*features([256,768])],[1,2,3,4]))
    assert provider.stage_latency('decode',1,1,0,2,(),[255,767])==pytest.approx(expected)
    assert provider.stage_latency('prefill',1,1,0,1,[512],[512])>0
    with pytest.raises(ValueError,match='missing_profile'):provider.stage_latency('decode',1,1,0,2,(),[4095,4095])
    with pytest.raises(ValueError,match='unsupported_engine'):provider.stage_latency('prefill',1,2,0,1,[512],[512])


def test_holdout_is_not_refitted_or_allowed_to_overlap_training():
    original=fitted();hold=[dict(row,latency_ms=row['latency_ms']*2) for row in original['holdout']]
    changed=fit_surface(original['training'],hold,identity=original['identity'],raw_bindings=[],measurement_qualification=dict(passed=True))
    assert not changed['qualified']
    assert original['cells'][0]['latency_coefficients']==changed['cells'][0]['latency_coefficients']
    with pytest.raises(ValueError,match='overlap'):
        fit_surface(original['training'],original['training'],identity=original['identity'],raw_bindings=[],measurement_qualification=dict(passed=True))
    with pytest.raises(ValueError,match='calibration'):StageSurface(changed)


def test_short_and_long_disconnected_shapes_do_not_cover_unobserved_heterogeneity():
    hull=_Hull([coverage_features([128]),coverage_features([1024])])
    assert hull.contains(coverage_features([512]))
    assert not hull.contains(coverage_features([128,1024]))


def test_disk_surface_requires_raw_sample_and_measurement_receipt_bindings(tmp_path):
    path=tmp_path/'surface.json';path.write_text(json.dumps(fitted()))
    with pytest.raises(ValueError,match='raw sample'):StageSurface.load(path)
