"""Physical clock evidence must remain separate from requested clock success."""
import threading

import pytest

from pdblend.measure.backends import FakeBackend
from pdblend.measure.power import PowerSampler
from pdblend.bench.comparison_pdblend_acceptance import _frequencies


class Clock:
    def __init__(self):
        self.t = 100.

    def __call__(self):
        self.t += .001
        return self.t


def test_frequency_timestamps_are_actual_reads_and_diagnostics_are_kept():
    backend = FakeBackend(2)
    backend.clock_diagnostics = lambda gpu: dict(throttle_reasons=4, power_limit_w=350., temperature_c=60.)
    sampler = PowerSampler([0, 1], backend=backend, clock=Clock(), sample_clocks=True)
    power_t = sampler._read()[0]
    rows = sampler.capture_frequency({0: 2520, 1: 2100}, 'before_transition')
    assert rows[0]['read_started_s'] > power_t
    assert rows[0]['read_finished_s'] < rows[1]['read_finished_s']
    assert sampler.frequency_samples[0][0] == rows[-1]['read_finished_s']
    assert rows[1]['requested_mhz'] == 2100 and rows[1]['observed_mhz'] == 2520
    assert rows[0]['throttle_reasons'] == 4


def test_optional_clock_failure_does_not_abort_energy_sampling():
    backend = FakeBackend(1)
    sampler = PowerSampler([0], backend=backend, interval=0, sample_clocks=True)
    count = 0
    def watts(gpu):
        nonlocal count
        count += 1
        if count == 3:
            sampler._stop.set()
        return 100.
    backend.power_w = watts
    def failed(gpu):
        raise RuntimeError('clock unavailable')
    backend.current_freq = failed
    sampler._loop()
    assert len(sampler.samples) == 3 and sampler.error is None
    assert len(sampler.frequency_errors) == 3 and not sampler.frequency_samples


def test_periodic_and_transition_frequency_snapshots_are_serialized():
    sampler = PowerSampler([0], backend=FakeBackend(1), clock=Clock())
    threads = [threading.Thread(target=lambda: [sampler.capture_frequency() for _ in range(10)]) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(sampler.frequency_samples) == 30
    assert all(b[0] > a[0] for a, b in zip(sampler.frequency_samples, sampler.frequency_samples[1:]))


def evidence():
    samples = [[101+i*.5, [2100]*8] for i in range(298)]
    plans = [dict(roles={'m': 'M'}, f_M=2100)]
    completed = [dict(finished_s=100., started_s=99.)]
    instances = {'m': {'gpu_uuids': ['g'+str(i) for i in range(8)]}}
    identity = {'fleet_gpu_uuids': instances['m']['gpu_uuids']}
    return samples, plans, completed, instances, identity, 100.


def test_sustained_throttling_fails_even_when_set_clock_succeeded():
    args = evidence()
    args[0][5][1][0] = 2040
    with pytest.raises(ValueError, match='actual active/parked clock'):
        _frequencies(*args)


def test_cross_transition_read_is_not_attributed_to_stable_interval():
    args = evidence()
    readings = [dict(gpu=g, read_started_s=t-.01, read_finished_s=t,
                     observed_mhz=2100, error=None) for t, _ in args[0] for g in range(8)]
    # Only crossing read for GPU0 at the beginning must not manufacture evidence.
    readings = [r for r in readings if r['gpu'] != 0 or r['read_finished_s'] > 102.]
    readings.insert(0, dict(gpu=0, read_started_s=100.9, read_finished_s=101.1,
                           observed_mhz=2100, error=None))
    with pytest.raises(ValueError, match='per-GPU frequency observations contain a gap'):
        _frequencies(*args, readings=readings)


def test_short_stable_interval_requires_real_snapshot():
    args = list(evidence())
    args[1] = args[1] * 2
    args[2] = [dict(finished_s=100., started_s=99.), dict(finished_s=101.04, started_s=101.033)]
    args[0] = [row for row in args[0] if row[0] != 101.]
    with pytest.raises(ValueError, match='observations contain a gap'):
        _frequencies(*args)
    args[0].insert(0, [101.02, [2100]*8])
    _frequencies(*args)
