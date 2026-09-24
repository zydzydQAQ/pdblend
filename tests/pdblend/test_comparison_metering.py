from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import pytest

from pdblend.bench.comparison_metering import ComparisonMeteringSession, summarize_comparison
from pdblend.measure.backends import GPU_UTILIZATION_SOURCE_ID, INSTANT_POWER_SOURCE_ID, PynvmlBackend
from pdblend.measure.power import PowerSampler


UUIDS = [f'GPU-test-{i}' for i in range(8)]
SOURCE = dict(mode='instant', source_id=INSTANT_POWER_SOURCE_ID, field_id=186, scope_id=0)


def evidence(times, *, watts=100., util=50.):
    return dict(gpus=list(range(8)), samples=[(t, [watts] * 8) for t in times],
                utilization_samples=[(t, [util] * 8) for t in times],
                power_metadata=[dict(gpus=list(range(8)), read_finished_s=[t] * 8,
                    **{k: [v] * 8 for k, v in dict(SOURCE, return_code=0, value_type=1).items()})
                    for t in times],
                power_source=SOURCE, error=None)


def summarize(raw, *, end=1., tail=None, **kwargs):
    return summarize_comparison(raw, gpu_uuids=UUIDS, origin_s=0., duration_s=end,
                                tail_end_s=end if tail is None else tail, **kwargs)


def test_150_second_service_and_tail_keep_parked_gpus_in_denominator():
    raw = evidence([i / 10 for i in range(-1, 1512)])
    raw['utilization_samples'] = [(t, [100.] + [0.] * 7) for t, _ in raw['samples']]
    result = summarize(raw, end=150., tail=151., gpu_uuid_binding_verified=True)
    assert result['energy_comparable'] and result['formal_eligible'] is False
    assert result['energy_service_j'] == pytest.approx(120000.)
    assert result['energy_tail_j'] == pytest.approx(800.)
    assert result['energy_service_tail_j'] == pytest.approx(120800.)
    assert result['service_mean_power_w'] == pytest.approx(800.)
    assert result['gpu_util_mean_pct'] == pytest.approx(12.5)
    assert result['gpu_busy_equivalent_s'] == pytest.approx(150.)
    assert result['service']['utilization']['per_gpu'][UUIDS[0]]['peak_pct'] == 100.
    assert result['service']['utilization']['per_gpu'][UUIDS[1]]['mean_pct'] == 0.
    assert result['service']['utilization']['per_gpu'][UUIDS[0]]['samples'] == 1500


def test_irregular_intervals_are_time_weighted_and_boundary_clipped():
    raw = evidence([-.1, 0., .2, .8, 1., 1.1])
    raw['utilization_samples'] = list(zip([-.1, 0., .2, .8, 1., 1.1],
                                        [[v] * 8 for v in (100., 0., 100., 100., 0., 100.)]))
    result = summarize(raw)
    assert result['gpu_util_mean_pct'] == pytest.approx(80.)
    assert result['energy_service_j'] == pytest.approx(800.)
    assert result['util_max_gap_s'] == pytest.approx(.6)
    assert result['energy_tail_j'] == 0.
    assert result['tail']['utilization']['mean_pct'] is None
    assert result['energy_comparable'] is False  # No physical UUID attestation.


def test_service_boundaries_use_bracketing_samples_without_extrapolation():
    raw = evidence([-.1, .4, .9, 1.4], watts=200.)
    result = summarize(raw)
    assert result['energy_service_j'] == pytest.approx(1600.)
    assert result['util_coverage_fraction'] == pytest.approx(1.)
    raw['samples'] = raw['samples'][1:]
    raw['power_metadata'] = raw['power_metadata'][1:]
    result = summarize(raw)
    assert result['energy_service_j'] is None
    assert result['service']['power']['coverage_fraction'] == pytest.approx(.6)


def test_long_gap_never_becomes_zero_energy_or_utilization():
    result = summarize(evidence([0., .5, 2., 2.5]), end=2.5)
    assert result['energy_service_j'] is None
    assert result['energy_service_tail_j'] is None
    assert result['gpu_util_mean_pct'] is None
    assert result['util_coverage_fraction'] == pytest.approx(.4)
    assert result['util_max_gap_s'] == pytest.approx(1.5)
    assert result['service']['power']['covered_energy_j'] == pytest.approx(800.)
    assert result['service']['utilization']['max_missing_gap_s'] == pytest.approx(1.5)


def test_failed_device_read_breaks_only_its_utilization_continuity():
    raw = evidence([0., .5, 1., 1.5])
    raw['utilization_readings'] = [dict(gpu=g, gpu_util_pct=None if (g == 7 and t == .5) else 50.,
        read_finished_s=t, error='NVML unavailable' if (g == 7 and t == .5) else None)
        for t in (0., .5, 1., 1.5) for g in range(8)]
    result = summarize(raw, end=1.5, gpu_uuid_binding_verified=True)
    assert result['energy_comparable']
    assert result['energy_service_j'] == pytest.approx(1200.)
    assert result['gpu_util_mean_pct'] is None
    assert result['util_coverage_fraction'] == pytest.approx((7 + 1 / 3) / 8)
    broken = result['service']['utilization']['per_gpu'][UUIDS[7]]
    assert broken['mean_pct'] is None and broken['missing_samples'] == 1
    assert result['service']['utilization']['per_gpu'][UUIDS[0]]['mean_pct'] == 50.


def test_individual_utilization_times_take_precedence_over_legacy_row_times():
    raw = evidence([0., .5, 1.], util=99.)
    raw['utilization_readings'] = [dict(gpu=g, gpu_util_pct=20., read_finished_s=t, error=None)
                                  for g in range(8) for t in (-.01 * g, .5, 1. + .01 * g)]
    result = summarize(raw)
    assert result['gpu_util_mean_pct'] == pytest.approx(20.)
    assert result['utilization_timestamp_semantics'] == 'per_device_acquisition_time'


def test_power_uses_individual_acquisition_times_and_checks_device_order():
    raw = evidence([0., .5, 1.])
    raw['power_metadata'] = [dict(gpus=list(range(8)), read_finished_s=[t + .1] * 8)
                             for t, _ in raw['samples']]
    result = summarize(raw)
    assert result['energy_service_j'] is None
    assert result['service']['power']['coverage_fraction'] == pytest.approx(.9)
    raw['power_metadata'][0]['gpus'].reverse()
    with pytest.raises(ValueError, match='order'):
        summarize(raw)


@pytest.mark.parametrize('damage', ['duplicate_uuid', 'seven_gpus', 'wrong_width', 'regressed_time', 'uuid_order'])
def test_invalid_scope_and_samples_fail_closed(damage):
    raw = evidence([0., .5, 1.])
    uuids = list(UUIDS)
    if damage == 'duplicate_uuid':
        uuids[-1] = uuids[0]
    elif damage == 'seven_gpus':
        raw['gpus'].pop()
    elif damage == 'wrong_width':
        raw['samples'][1][1].pop()
    elif damage == 'regressed_time':
        raw['samples'] = list(reversed(raw['samples']))
        raw['power_metadata'] = list(reversed(raw['power_metadata']))
    else:
        raw['gpu_uuids'] = list(reversed(UUIDS))
    with pytest.raises(ValueError):
        summarize_comparison(raw, gpu_uuids=uuids, origin_s=0., duration_s=1., tail_end_s=1.)


def test_empty_or_average_power_never_qualifies_and_does_not_mutate_old_rows():
    raw = evidence([0., .5, 1.])
    before = deepcopy(raw)
    summarize(raw)
    assert raw == before
    raw['power_source'] = dict(mode='average')
    assert not summarize(raw, gpu_uuid_binding_verified=True)['energy_comparable']
    raw['samples'] = []
    raw['power_metadata'] = []
    raw['utilization_samples'] = []
    result = summarize(raw)
    assert result['energy_service_j'] is None and result['gpu_util_mean_pct'] is None
    assert result['util_coverage_fraction'] == 0. and result['util_max_gap_s'] == 1.


def test_error_and_missing_peaks_are_explicit():
    raw = evidence([0., .5, 1.])
    raw['error'] = 'power read stopped'
    raw['utilization_samples'] = []
    result = summarize(raw, gpu_uuid_binding_verified=True)
    assert not result['energy_comparable']
    assert result['service']['utilization']['per_gpu'][UUIDS[0]]['peak_pct'] is None


def test_later_session_error_does_not_invalidate_completed_earlier_window():
    raw = evidence([0., .5, 1.])
    raw.update(error='failed after earlier window completed', error_at_s=2.)
    result = summarize(raw, gpu_uuid_binding_verified=True)
    assert result['energy_comparable'] and not result['power_error_affects_window']


class StopAfter:
    def __init__(self, loops):
        self.loops = loops

    def is_set(self):
        return self.loops <= 0

    def wait(self, interval):
        self.loops -= 1


def test_sampler_util_error_keeps_power_and_other_device_readings():
    clock = lambda: next(ticks) / 100
    ticks = count(10000)

    class Backend:
        power_calls = 0
        util_calls = 0

        def power_w(self, gpu):
            self.power_calls += 1
            return 100.

        def utilization_pct(self, gpu):
            self.util_calls += 1
            if self.util_calls == 2:
                raise RuntimeError('one bad utilization read')
            return 40. + gpu

    sampler = PowerSampler([0, 1], backend=Backend(), clock=clock)
    sampler._stop = StopAfter(3)
    sampler._loop()
    assert sampler.error is None and len(sampler.samples) == 3
    assert len(sampler.utilization_readings) == 6 and len(sampler.utilization_errors) == 1
    assert sampler.utilization_readings[1]['gpu_util_pct'] is None
    assert len(sampler.utilization_samples) == 2
    assert sampler.utilization_readings[0]['read_started_s'] > sampler.samples[0][0]


def test_sampler_power_error_still_stops_measurement():
    class Backend:
        def power_w(self, gpu):
            raise RuntimeError('power unavailable')

    sampler = PowerSampler([0], backend=Backend())
    sampler._stop = StopAfter(3)
    sampler._loop()
    assert sampler.error == 'power unavailable' and sampler.samples == []


def test_nvml_utilization_is_gpu_busy_with_its_own_times_and_not_memory():
    backend = PynvmlBackend.__new__(PynvmlBackend)
    backend._power_clock = iter([11., 11.02]).__next__
    backend._handle = lambda gpu: gpu
    backend._nvml = SimpleNamespace(nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=30, memory=80))
    row = backend.utilization_reading(0)
    assert row['gpu_util_pct'] == 30.
    assert row['read_started_s'] == 11. and row['read_finished_s'] == 11.02
    assert row['source_id'] == GPU_UTILIZATION_SOURCE_ID


def test_session_verifies_actual_uuid_order_and_snapshots_serializable_evidence():
    backend = SimpleNamespace(gpu_uuid=lambda gpu: UUIDS[gpu], power_source=SOURCE)
    session = ComparisonMeteringSession(range(8), UUIDS, backend=backend)
    raw = evidence([0., .5, 1.])
    session.sampler.samples = raw['samples']
    session.sampler.power_metadata = raw['power_metadata']
    session.sampler.utilization_samples = raw['utilization_samples']
    result = session.summarize(origin_s=0., tail_end_s=1., duration_s=1.)
    assert result['energy_comparable']
    import json
    json.dumps(session.snapshot(), allow_nan=False)
    with pytest.raises(ValueError, match='actual physical'):
        ComparisonMeteringSession(range(8), list(reversed(UUIDS)), backend=backend)


def test_unverified_or_changed_power_source_preserves_numbers_but_not_comparability():
    raw = evidence([0., .5, 1.])
    raw['power_metadata'][1]['field_id'][3] = 185
    result = summarize(raw, gpu_uuid_binding_verified=True)
    assert result['energy_service_j'] == 800.
    assert not result['power_source_verified'] and not result['energy_comparable']
    raw.pop('power_metadata')
    assert not summarize(raw, gpu_uuid_binding_verified=True)['energy_comparable']
