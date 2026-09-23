import copy
import json

import pytest

from pdblend.profile.model import PerfModel, StaticState
from pdblend.profile.power_table import KIND, PowerCoverageError, validate
from pdblend.profile.calibration import evaluate_holdout
from pdblend.control.planner import PoolPlanner, PlannerConfig, SLO
from pdblend.control.forecast import Forecast


def model():
    fs=(900,2520)
    m=PerfModel(freqs=fs,prefill_time={f:(.001,.00001,0.) for f in fs},
        prefill_power={f:(100.,0.) for f in fs},decode_time={f:(.01,0.,0.,0.) for f in fs},
        decode_power={f:(999.,10.) for f in fs},static={**{f'active_idle@{f}':StaticState(50.) for f in fs},
            'active_idle_reset':StaticState(50.),'parked':StaticState(20.),'off':StaticState(20.)},kv_capacity_tokens=1000000)
    m.decode_power_overrides={f:dict(kind=KIND,batch_interpolation='linear',nodes=[
        dict(batch=b,context_min=c,context_max=c+10,power_w=(200 if b==1 else 100)+b+c/10)
        for b in (1,4,8,16) for c in (500,1000,4000)]) for f in fs}
    return m


def test_new_power_requires_real_context_and_exact_frequency_and_preserves_timing():
    m=model();before=copy.deepcopy(m.decode_time)
    for b,f,c in ((8,900,None),(8,1500,2000),(2,900,2000),(1.5,900,2000),(8,900,499),(8,900,4011)):
        with pytest.raises(PowerCoverageError,match='missing_profile'):
            m.decode_power_w(b,f,ctx=c)
        assert not m.decode_power_supported(b,c,f)
    assert m.decode_power_w(8,900,ctx=1000)==208
    assert m.decode_power_w(1,900,ctx=1000)==301
    assert m.decode_time==before
    assert m.token_energy_j(8,1000,900)==pytest.approx(.01*208/8)
    back=PerfModel.from_json(m.to_json())
    assert back.decode_power_w(8,900,ctx=1000)==208
    assert back.decode_power_overrides==m.decode_power_overrides


def test_legacy_affine_nearest_frequency_remains_compatible():
    m=model();m.decode_power_overrides={}
    assert m.decode_power_w(8,950)==1079
    assert m.decode_power_w(8,950,ctx=999999)==1079
    assert 'decode_power_overrides' not in json.loads(m.to_json())


def test_malformed_override_does_not_fallback_to_affine():
    m=model();d=json.loads(m.to_json());d['decode_power_overrides'].pop('900')
    with pytest.raises(ValueError,match='every declared frequency'):PerfModel.from_json(json.dumps(d))
    d=json.loads(m.to_json());d['decode_power_overrides']['900']['kind']='unknown'
    with pytest.raises(ValueError,match='unknown'):PerfModel.from_json(json.dumps(d))
    spec=copy.deepcopy(m.decode_power_overrides[900]);spec['nodes'].append(copy.deepcopy(spec['nodes'][0]))
    with pytest.raises(ValueError,match='overlapping'):validate(spec)


def test_planner_checks_power_domain_and_uses_available_context():
    m=model();planner=PoolPlanner(m,PlannerConfig(slots=4,slo=SLO(5,.15)))
    fc=Forecast(.1,0,950,950,100,0,(950,)*10)
    planner._decode_batch=lambda *a,**kw:8.
    point=planner._decode_pool(.1,fc,950,1,900)
    assert point['power_w']==208
    planner._decode_batch=lambda *a,**kw:2.
    assert planner._decode_pool(.1,fc,950,1,900) is None
    assert planner._mixed_pool(.1,fc,950,950,1,900) is None
    planner._decode_batch=lambda *a,**kw:8.
    assert planner._decode_pool(.1,fc,100,1,900) is None


def test_calibration_power_uses_each_repeat_actual_context_and_fails_closed(tmp_path):
    from test_calibration_incremental import holdout_fixture, HoldoutModel
    class ContextModel(HoldoutModel):
        decode_power_overrides={1500:{'present':True}}
        def decode_power_w(self,b,f,*,ctx=None):
            if ctx is None or ctx>250:raise PowerCoverageError('outside')
            return ctx
    raw=holdout_fixture(tmp_path)
    for r in raw['decode'][0]['repeats']:r['power_w']=r['effective_context_tokens']
    audit=evaluate_holdout(raw,ContextModel(),tmp_path)
    bad=[r for r in audit['failures'] if r['metric']=='decode_power_repeat']
    assert len(bad)==1 and bad[0]['repeat']==2
    assert bad[0]['status']=='outside_coverage' and bad[0]['relative_error'] is None
    assert [r['predicted'] for r in audit['power_points']]==[100,200,None]
    assert audit['timing_max']==0


def test_new_power_missingness_cannot_silently_turn_into_fixed_layout_fallback():
    m=model();p=PoolPlanner(m,PlannerConfig(slots=4,slo=SLO(5,.15)))
    p._decode_batch=lambda *a,**kw:2.
    fc=Forecast(.1,0,950,950,100,0,(950,)*10)
    with pytest.raises(PowerCoverageError,match='fallback layout'):p.fallback(fc)
    p._decode_batch=lambda *a,**kw:.5
    assert p._decode_pool(.1,fc,950,1,900)['power_w']==pytest.approx(50+.5*(301-50))
