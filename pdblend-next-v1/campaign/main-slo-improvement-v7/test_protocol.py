import copy
import pytest
import protocol as p

def points():
    common = dict(model='7b', dataset='alpaca', rate_rps=12., seed=701,
        trace_sha256='a'*64, content_pairing_sha256='b'*64,
        slo_ttft_s=1., slo_tpot_s=.1, n_expected=100, expected_generated_tokens=1000)
    a = dict(common, measurement_valid=True, work_complete=True, energy_j=100.,
             slo_attainment=.9, completed_work_requests=100, generated_tokens=1000)
    b = dict(common, system='mixed', cell_id='baseline', energy_j=100., slo_attainment=1.)
    return a, b

def test_exact_thresholds_pass():
    a, b = points()
    assert p.verdict(a, b)['passed']

def test_no_energy_allowance():
    a, b = points(); a['energy_j'] += 1e-10
    assert not p.verdict(a, b)['passed']

def test_saturation_uses_actual_baseline():
    a, b = points(); b['slo_attainment'] = .7; a['slo_attainment'] = .7
    assert p.verdict(a, b)['passed']
    a['slo_attainment'] = .699999
    assert not p.verdict(a, b)['passed']

def test_low_energy_from_incomplete_work_does_not_pass():
    a, b = points(); a.update(energy_j=1., completed_work_requests=99)
    assert not p.verdict(a, b)['passed']

def test_prescribed_outputs_are_required():
    a, b = points(); a['generated_tokens'] = 999
    assert not p.verdict(a, b)['passed']

@pytest.mark.parametrize('field', p.PAIR_FIELDS)
def test_each_pair_field_is_checked(field):
    a, b = points(); b[field] = 'changed'
    with pytest.raises(ValueError, match='exact workload pair'):
        p.verdict(a, b)

def test_technical_failure_never_passes():
    a, b = points(); a['measurement_valid'] = False
    assert not p.verdict(a, b)['passed']

def test_non_finite_metric_never_passes():
    a, b = points(); a['energy_j'] = float('nan')
    assert not p.verdict(a, b)['passed']
