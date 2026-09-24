"""Spawned CPU-only samplers; no NVML or GPU is initialized in these tests."""
import os
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from pdblend.bench.isolated_comparison_meter import IsolatedComparisonMeter


class FakeSession:
    def __init__(self, gpus, gpu_uuids, *, interval_s, max_gap_s):
        self.gpus, self.uuids = list(gpus), list(gpu_uuids)
        self.samples = []
        self.done = threading.Event()
        self.after_s = None
        self.sampler = SimpleNamespace(sample_clocks=False, frequency_samples=[], _thread=None)

    def start(self):
        def sample():
            while not self.done.is_set():
                stamp = time.time()
                self.samples.append((stamp, [1.]*8))
                if self.sampler.sample_clocks:
                    self.sampler.frequency_samples.append((stamp, [2520]*8))
                self.done.wait(.01)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.sampler._thread = self.thread
        self.thread.start()

    def snapshot(self):
        return dict(gpus=self.gpus, gpu_uuids=self.uuids, samples=list(self.samples),
                    test_child_pid=os.getpid(), after_s=self.after_s, stopped=self.done.is_set())

    def stop(self, *, after_s=None, timeout_s=2.):
        self.after_s = after_s
        self.done.set()
        self.thread.join(timeout=timeout_s)


class FailingStart(FakeSession):
    def start(self):
        raise ValueError('test startup failed')


class FailingSnapshot(FakeSession):
    def snapshot(self):
        raise ValueError('test snapshot failed')


class HangingSnapshot(FakeSession):
    def snapshot(self):
        threading.Event().wait(30.)


class ExitingSnapshot(FakeSession):
    def snapshot(self):
        os._exit(17)


def meter(factory=FakeSession, **kwargs):
    return IsolatedComparisonMeter(range(8), [f'GPU-{i}' for i in range(8)], session_factory=factory, **kwargs)


def test_spawn_isolates_sampler_and_preserves_raw_snapshot_and_stop_contract():
    value = meter().start()
    try:
        initial = value.snapshot()
        assert initial['test_child_pid'] != os.getpid()
        # The parent does CPU work while the independent sampler continues.
        value.begin_window()
        deadline = time.monotonic()+.12
        while time.monotonic() < deadline:
            sum(range(1000))
        value.end_window()
        later = value.snapshot()
        assert len(later['samples']) >= len(initial['samples'])+3
        assert later.keys() == initial.keys()
        stamp = time.time()
        value.stop(after_s=stamp, timeout_s=.5)
        final = value.snapshot()
        assert final['after_s'] == stamp and final['stopped']
        assert final['gpus'] == list(range(8))
        assert all(isinstance(row, tuple) and row[1] == [1.]*8 for row in final['samples'])
        final['samples'].clear()
        assert value.snapshot()['samples'], 'callers cannot mutate saved final evidence'
        receipt = value.method_receipt()
        assert receipt['status'] == 'stopped' and receipt['child_exitcode'] == 0
        assert receipt['child_pid'] == initial['test_child_pid']
        assert receipt['parent_pid'] == os.getpid() and receipt['test_factory']
        assert receipt['raw_snapshot_modified'] and not receipt['public_snapshot_fields_modified']
        assert not receipt['formal_eligible'] and receipt['sample_clocks'] is True
        assert receipt['additional_frequency_observation']
        assert receipt['additional_snapshot_fields'] == ['frequency_samples']
        assert receipt['wrapper']['sha256'] == hashlib.sha256(Path(receipt['wrapper']['path']).read_bytes()).hexdigest()
        assert receipt['public_sampler']['module'] == 'pdblend.bench.comparison_metering'
        assert receipt['backend']['module'] == 'pdblend.measure.backends'
        assert all(row['passed'] for row in receipt['commands'])
        assert receipt['liveness_observations'][0]['sampler_thread_alive']
        assert not receipt['liveness_observations'][-1]['sampler_thread_alive']
        assert not receipt['child_alive']
        assert receipt['local_window_guards'][0]['end_s'] is not None
        value.stop()  # Idempotent, with no second RPC.
    finally:
        if value.method_receipt()['status'] == 'running':value.stop()


def test_no_rpc_inside_service_or_tail_guard_and_no_restart():
    value = meter().start()
    try:
        value.begin_window()
        with pytest.raises(RuntimeError, match='forbidden'):value.snapshot()
        with pytest.raises(RuntimeError, match='forbidden'):value.stop()
        assert value.method_receipt()['commands'] == []
        value.end_window()
        with pytest.raises(RuntimeError, match='cannot be restarted'):value.start()
    finally:
        value.stop()
    with pytest.raises(RuntimeError, match='cannot be restarted'):value.start()


def test_child_start_failure_is_explicit_and_reaped():
    value = meter(FailingStart)
    with pytest.raises(RuntimeError, match='test startup failed'):value.start()
    assert value.method_receipt()['status'] == 'failed'
    assert not value._process.is_alive()
    with pytest.raises(RuntimeError, match='cannot be restarted'):value.start()


@pytest.mark.parametrize('factory,exception,match', [
    (FailingSnapshot, RuntimeError, 'test snapshot failed'),
    (HangingSnapshot, TimeoutError, 'timed out'),
    (ExitingSnapshot, RuntimeError, 'child'),
])
def test_child_rpc_failure_timeout_or_exit_never_restarts_or_invents_data(factory, exception, match):
    value = meter(factory, rpc_timeout_s=.15).start()
    with pytest.raises(exception, match=match):value.snapshot()
    assert not value._process.is_alive()
    receipt = value.method_receipt()
    assert receipt['status'] == 'failed' and receipt['commands'][-1]['passed'] is False
    with pytest.raises(RuntimeError, match='not running'):value.snapshot()
    with pytest.raises(RuntimeError, match='cannot be restarted'):value.start()


def test_context_manager_reaps_owned_process():
    with meter() as value:
        assert value.snapshot()['test_child_pid'] == value.method_receipt()['child_pid']
    assert not value._process.is_alive()
    assert value.method_receipt()['status'] == 'stopped'


def test_unpicklable_factory_preserves_start_error_without_joining_unstarted_child():
    value = meter(lambda *args, **kwargs: None)
    with pytest.raises((AttributeError, TypeError), match='pickle|local'):value.start()
    assert value.method_receipt()['status'] == 'failed'
    assert not value.method_receipt()['child_alive']


@pytest.mark.parametrize('guarded', [False, True])
def test_context_failure_preserves_original_error_and_reaps_child(guarded):
    value = meter(FailingSnapshot)
    with pytest.raises(ValueError, match='original failure'):
        with value:
            if guarded:value.begin_window()
            raise ValueError('original failure')
    assert not value.method_receipt()['child_alive']
    assert value.method_receipt()['status'] == 'failed'


def test_context_cannot_silently_finish_with_active_measurement_guard():
    value = meter()
    with pytest.raises(RuntimeError, match='guarded measurement'):
        with value:value.begin_window()
    assert not value.method_receipt()['child_alive']


class CpuBackend:
    """Deterministic sensor values with actual CPU acquisition timestamps."""
    power_source = dict(mode='instant', source_id='nvml:field:186:scope:0:mW', field_id=186, scope_id=0, unit='W')
    utilization_source = dict(source_id='cpu-test-utilization', sensor_period_s=None, unit='percent')

    def gpu_uuid(self, gpu):return f'GPU-{gpu}'

    def power_reading(self, gpu):
        stamp = time.time()
        return dict(watts=100.+gpu, mode='instant', source_id=self.power_source['source_id'],
            field_id=186, scope_id=0, value_type=1, return_code=0,
            nvml_timestamp_us=int(stamp*1e6), nvml_latency_us=0,
            read_started_s=stamp, read_finished_s=stamp)

    def utilization_pct(self, gpu):return 50.+gpu

    def current_freq(self, gpu):return 2520 if gpu < 2 else 900


def public_session_with_cpu_backend(gpus, gpu_uuids, **kwargs):
    from pdblend.bench.comparison_metering import ComparisonMeteringSession
    return ComparisonMeteringSession(gpus, gpu_uuids, backend=CpuBackend(), **kwargs)


def test_original_sampler_frequency_rows_survive_archive_and_do_not_change_energy_reduction(tmp_path):
    from pdblend.bench.comparison_metering import summarize_comparison
    from pdblend.results.power_archive import write_power_archive, read_power_archive
    from pdblend.bench.comparison_native_acceptance import _equal

    with meter(public_session_with_cpu_backend, interval_s=.01) as value:
        value.begin_window()
        time.sleep(.08)
        value.end_window()
    snapshot = value.snapshot()
    assert snapshot['error'] is None
    clocks = snapshot['frequency_samples']
    assert len(clocks) >= 3 and len(clocks) == len(snapshot['samples'])
    assert all(values == [2520]*2+[900]*6 for _, values in clocks)
    assert all(power_stamp <= clock_stamp for (clock_stamp, _), (power_stamp, _)
               in zip(clocks, snapshot['samples']))
    assert any(power_stamp < clock_stamp for (clock_stamp, _), (power_stamp, _)
               in zip(clocks, snapshot['samples'])), 'retain actual frequency acquisition time'
    assert value.method_receipt()['sampler'] == value.method_receipt()['public_sampler']
    path = tmp_path/'power.json'
    write_power_archive(path, snapshot)
    restored = read_power_archive(path)
    # The existing archive replaces/removes the outer schema; sensor data and
    # all public sample/metadata fields must survive without modification.
    assert _equal(restored, {key: item for key, item in snapshot.items() if key != 'schema'})
    assert restored['frequency_samples'], 'Eco acceptance reads this exact archived field'

    origin = max(snapshot['power_metadata'][0]['read_finished_s'])
    end = min(snapshot['power_metadata'][-1]['read_finished_s'])
    arguments = dict(gpu_uuids=snapshot['gpu_uuids'], origin_s=origin,
                     duration_s=(end-origin)/2, tail_end_s=end)
    summary = summarize_comparison(restored, **arguments)
    assert summary['energy_comparable'], summary
    restored.pop('frequency_samples')
    assert summary == summarize_comparison(restored, **arguments)


@pytest.fixture(scope='module')
def startup_observations():
    with meter(public_session_with_cpu_backend, interval_s=.01) as value:
        time.sleep(2.15)
        snapshot = value.snapshot()
        receipt = value.method_receipt()
    return snapshot, receipt


def test_startup_preflight_replays_original_sampler_and_keeps_window_gate_closed(startup_observations):
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    snapshot, receipt = startup_observations
    result = qualify_startup_snapshot(snapshot, receipt, snapshot['gpu_uuids'])
    assert result['passed'] and result['observed_span_s'] >= 2.
    assert result['public_metering']['energy_comparable']
    assert not result['formal_eligible'] and not result['evaluation_window_qualified']


@pytest.mark.parametrize('fault', ['before_power', 'next_loop', 'nonfinite', 'missing_middle'])
def test_startup_frequency_acquisition_must_belong_to_its_sampler_loop(startup_observations, fault):
    from copy import deepcopy
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    snapshot, receipt = deepcopy(startup_observations)
    rows = snapshot['frequency_samples']
    if fault == 'before_power':
        rows[1] = (snapshot['samples'][1][0]-.01, rows[1][1])
    elif fault == 'next_loop':
        rows[1] = (snapshot['samples'][2][0]+.001, rows[1][1])
    elif fault == 'nonfinite':
        rows[1] = (float('nan'), rows[1][1])
    else:
        rows.pop(1)
    with pytest.raises(ValueError, match='frequency'):
        qualify_startup_snapshot(snapshot, receipt, snapshot['gpu_uuids'])


def test_startup_legacy_frequency_timestamps_remain_valid(startup_observations):
    from copy import deepcopy
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    snapshot, receipt = deepcopy(startup_observations)
    snapshot['frequency_samples'] = [(power[0], clock[1]) for power, clock in
                                      zip(snapshot['samples'], snapshot['frequency_samples'])]
    assert qualify_startup_snapshot(snapshot, receipt, snapshot['gpu_uuids'])['passed']


def test_startup_uses_common_acquisition_span_without_changing_sensor_data(startup_observations):
    from copy import deepcopy
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    from pdblend.bench.comparison_metering import summarize_comparison
    snapshot, receipt = deepcopy(startup_observations)
    # Real NVML initialization can make the first utilization reads arrive
    # several milliseconds after all first power reads have completed.
    start = max(snapshot['power_metadata'][0]['read_finished_s'])
    for gpu in range(8):
        device_rows = [r for r in snapshot['utilization_readings'] if r['gpu'] == gpu]
        delay = start + .005 + gpu * .001 - device_rows[0]['read_finished_s']
        for row in device_rows:
            row['read_finished_s'] += delay
    before = deepcopy(snapshot)
    end = min(snapshot['power_metadata'][-1]['read_finished_s']) - .001
    origin = start + .001
    old = summarize_comparison(snapshot, gpu_uuids=snapshot['gpu_uuids'],
        origin_s=origin, tail_end_s=end, duration_s=end-origin)
    assert old['util_status'] == 'missing'
    result = qualify_startup_snapshot(snapshot, receipt, snapshot['gpu_uuids'])
    assert result['public_metering']['util_status'] == 'complete'
    assert result['public_metering']['util_coverage_fraction'] == 1.
    assert snapshot == before


@pytest.mark.parametrize('fault', ['missing_rank', 'internal_gap'])
def test_startup_common_span_does_not_hide_missing_utilization(startup_observations, fault):
    from copy import deepcopy
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    snapshot, receipt = deepcopy(startup_observations)
    rows = snapshot['utilization_readings']
    middle = (rows[0]['read_finished_s'] + rows[-1]['read_finished_s']) / 2
    snapshot['utilization_readings'] = [r for r in rows if r['gpu'] != 7 or
        (fault == 'internal_gap' and abs(r['read_finished_s']-middle) > .55)]
    with pytest.raises(ValueError):
        qualify_startup_snapshot(snapshot, receipt, snapshot['gpu_uuids'])


@pytest.mark.parametrize('fault', ['uuid', 'frequency', 'span', 'liveness', 'power'])
def test_startup_preflight_rejects_real_path_failures(startup_observations, fault):
    from copy import deepcopy
    from pdblend.bench.comparison_meter_preflight import qualify_startup_snapshot
    snapshot, receipt = deepcopy(startup_observations)
    expected=list(snapshot['gpu_uuids'])
    if fault == 'uuid':snapshot['gpu_uuids'].reverse()
    elif fault == 'frequency':snapshot['frequency_samples'][0][1][0] = 0
    elif fault == 'span':snapshot['frequency_samples'] = snapshot['frequency_samples'][:3]
    elif fault == 'liveness':receipt['liveness_observations'][-1]['sampler_thread_alive'] = False
    elif fault == 'power':snapshot['samples'][2][1][0] = float('nan')
    with pytest.raises(ValueError):qualify_startup_snapshot(snapshot, receipt, expected)
