import pytest
from boundary_plan import next_boundary,paired_jobs

def point(rate,slo,seed=701,complete=True,version='frozen'):
    return dict(rate_rps=rate,slo_attainment=slo,seed=seed,work_complete=complete,
        measurement_valid=True,implementation_id=version)

def test_expand_highest_complete_service_rate():
    d=next_boundary([point(2,.99),point(2.5,.95)])
    assert d['action']=='expand_1_25' and d['jobs']==[dict(rate='3.125',seed=701)]

def test_midpoint_without_rounding_changes():
    d=next_boundary([point(1,.98),point(1.25,.8)])
    assert d['jobs']==[dict(rate='1.125',seed=701)] and not d['certified']

def test_503_is_not_an_upper_saturation_endpoint():
    d=next_boundary([point(.5,1),point(.8,.5,complete=False)])
    assert d['upper'] is None and d['jobs']==[] and not d['certified']

def test_all_six_independent_endpoint_seed_jobs_required():
    d=next_boundary([point(1,.98),point(1.1,.8)])
    assert d['action']=='confirm_independent_seeds' and len(d['jobs'])==6
    assert {j['seed'] for j in d['jobs']}=={1701,2701,3701}
    assert all(len(j['systems'])==5 for j in paired_jobs([d]))

def test_seed_crossing_retains_uncertainty():
    rows=[point(r,q,s) for r,q in ((1,.98),(1.1,.8)) for s in (701,1701,2701,3701)]
    assert next_boundary(rows)['certified']
    rows[-1]['slo_attainment']=.95
    d=next_boundary(rows)
    assert not d['certified'] and not d['jobs']

def test_version_splicing_is_rejected():
    with pytest.raises(ValueError,match='mix'):
        next_boundary([point(1,1),point(1.1,.5,version='other')])
