"""Eight-device service/tail metering, independent of any serving policy.

``ComparisonMeteringSession`` owns one sampler across any number of resident
windows. Start before setup; record each service origin and native-drain end;
then summarize that window. This measures GPU boards, not host energy. It
never grants experiment/profile/SLO qualification.

Both power and kernel-busy utilization use time-weighted linear integration
with bracketed boundaries. No extrapolation, zero filling, or interpolation
across acquisition gaps over one second is allowed. Utilization is NVML's
recent-period kernel-busy percentage, not SM occupancy or peak FLOPS usage.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

from pdblend.measure.backends import INSTANT_POWER_SOURCE_ID, PynvmlBackend
from pdblend.measure.power import PowerSampler

SCHEMA = 'pdblend-comparison-metering-v1'


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _devices(gpus, gpu_uuids):
    gpus, gpu_uuids = tuple(gpus), tuple(gpu_uuids)
    if (len(gpus) != 8 or any(type(g) is not int or g < 0 for g in gpus)
            or len(set(gpus)) != 8 or len(gpu_uuids) != 8
            or any(not isinstance(u, str) or not u.startswith('GPU-') or len(u) <= 4
                   for u in gpu_uuids) or len(set(gpu_uuids)) != 8):
        raise ValueError('comparison requires eight distinct local devices and physical GPU UUIDs')
    return gpus, gpu_uuids


def _integrate(rows, start, end, max_gap):
    """Integrate one device; an explicit failed reading breaks continuity."""
    duration = end - start
    if any(not _number(t) or (v is not None and not _number(v)) for t, v in rows):
        raise ValueError('nonfinite sample timestamp/value')
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('sample timestamps must increase per physical GPU')
    valid = [v for t, v in rows if start <= t < end and v is not None]
    missing = sum(v is None for t, v in rows if start <= t < end)
    if duration == 0:
        return dict(status='complete', duration_s=0., covered_s=0., missing_s=0.,
                    coverage_fraction=1., integral=0., covered_integral=0., mean=None,
                    peak=None, samples=0, missing_samples=0, max_gap_s=0., max_missing_gap_s=0.)
    area = covered = 0.
    segments, peaks, gaps = [], list(valid), []
    for (a, va), (b, vb) in zip(rows, rows[1:]):
        left, right = max(a, start), min(b, end)
        if right <= left:
            continue
        gaps.append(b - a)
        if va is None or vb is None or b - a > max_gap:
            continue
        vl = va + (vb - va) * (left - a) / (b - a)
        vr = va + (vb - va) * (right - a) / (b - a)
        area += (vl + vr) * .5 * (right - left)
        covered += right - left
        segments.append((left, right))
        peaks.extend((vl, vr))
    cursor, missing_gaps = start, []
    for left, right in segments:
        if left > cursor:
            missing_gaps.append(left - cursor)
        cursor = right
    if cursor < end:
        missing_gaps.append(end - cursor)
    # Include unbracketed boundary gaps, including completely absent devices.
    if not rows:
        gaps.append(duration)
    else:
        gaps.extend((max(0., min(end, rows[0][0]) - start),
                     max(0., end - max(start, rows[-1][0]))))
    complete = math.isclose(covered, duration, rel_tol=0., abs_tol=1e-8)
    return dict(status='complete' if complete else 'missing', duration_s=duration,
                covered_s=covered, missing_s=max(0., duration - covered),
                coverage_fraction=min(1., covered / duration),
                integral=area if complete else None, covered_integral=area,
                mean=area / duration if complete else None,
                peak=max(peaks) if peaks else None, samples=len(valid), missing_samples=missing,
                max_gap_s=max(gaps, default=duration),
                max_missing_gap_s=max(missing_gaps, default=0.))


def _power_series(sampler, gpus):
    series = {g: [] for g in gpus}
    samples = list(_get(sampler, 'samples', []))
    metadata = list(_get(sampler, 'power_metadata', []))
    if metadata and len(metadata) != len(samples):
        raise ValueError('power metadata/sample count differs')
    for index, (timestamp, values) in enumerate(samples):
        if len(values) != 8 or any(not _number(v) or v < 0 for v in values):
            raise ValueError('power samples must contain eight finite nonnegative readings')
        times = [timestamp] * 8
        if metadata:
            row = metadata[index]
            if tuple(row.get('gpus', ())) != gpus:
                raise ValueError('power sample device order differs')
            times = row.get('read_finished_s', times)
            if len(times) != 8:
                raise ValueError('power per-device timestamps missing')
        for gpu, t, value in zip(gpus, times, values):
            series[gpu].append((t, float(value)))
    return series


def _util_series(sampler, gpus):
    series = {g: [] for g in gpus}
    readings = list(_get(sampler, 'utilization_readings', []))
    if readings:
        for row in readings:
            gpu = row.get('gpu')
            if gpu not in series:
                raise ValueError('utilization reading belongs to an unmetered GPU')
            value = row.get('gpu_util_pct')
            if row.get('error') or not _number(value) or not 0 <= value <= 100:
                value = None
            series[gpu].append((row.get('read_finished_s', row.get('t_s')), value))
        return series, 'per_device_acquisition_time'
    # Immutable historical numeric rows can be read without conversion. Their
    # shared timestamp is explicitly weaker evidence than new per-device times.
    for timestamp, values in _get(sampler, 'utilization_samples', []):
        if len(values) != 8:
            raise ValueError('legacy utilization sample does not cover all eight devices')
        for gpu, value in zip(gpus, values):
            if not _number(value) or not 0 <= value <= 100:
                value = None
            series[gpu].append((timestamp, value))
    return series, 'legacy_shared_row_timestamp'


def _verified_power_source(sampler, source):
    if not (source.get('mode') == 'instant' and source.get('source_id') == INSTANT_POWER_SOURCE_ID
            and source.get('field_id') == 186 and source.get('scope_id') == 0):
        return False
    metadata = _get(sampler, 'power_metadata', [])
    expected = dict(mode='instant', source_id=INSTANT_POWER_SOURCE_ID,
                    field_id=186, scope_id=0, return_code=0, value_type=1)
    return bool(metadata) and all(row.get(key) == [value] * 8
                                  for row in metadata for key, value in expected.items())


def _phase(power, utilization, gpus, uuids, start, end, max_gap):
    p = {u: _integrate(power[g], start, end, max_gap) for g, u in zip(gpus, uuids)}
    util = {u: _integrate(utilization[g], start, end, max_gap) for g, u in zip(gpus, uuids)}

    def coverage(rows):
        return dict(coverage_fraction=sum(r['coverage_fraction'] for r in rows.values()) / 8,
                    minimum_gpu_coverage_fraction=min(r['coverage_fraction'] for r in rows.values()),
                    max_gap_s=max(r['max_gap_s'] for r in rows.values()),
                    max_missing_gap_s=max(r['max_missing_gap_s'] for r in rows.values()),
                    missing_gpu_s=sum(r['missing_s'] for r in rows.values()),
                    missing_samples=sum(r['missing_samples'] for r in rows.values()))

    p_ok = all(r['status'] == 'complete' for r in p.values())
    u_ok = all(r['status'] == 'complete' for r in util.values())
    duration = end - start
    energy = sum(r['integral'] for r in p.values()) if p_ok else None
    busy_s = sum(r['integral'] for r in util.values()) / 100 if u_ok else None
    for rows, mean_name, peak_name in ((p, 'mean_power_w', 'peak_power_w'),
                                      (util, 'mean_pct', 'peak_pct')):
        for row in rows.values():
            row[mean_name] = row.pop('mean')
            row[peak_name] = row.pop('peak')
    return dict(start_s=start, end_s=end, duration_s=duration,
                interval_semantics='[start_s,end_s)',
                power=dict(status='complete' if p_ok else 'missing', energy_j=energy,
                           covered_energy_j=sum(r['covered_integral'] for r in p.values()),
                           mean_power_w=energy / duration if p_ok and duration else None,
                           per_gpu=p, **coverage(p)),
                utilization=dict(status='complete' if u_ok else 'missing',
                                 mean_pct=busy_s * 100 / (8 * duration) if u_ok and duration else None,
                                 gpu_busy_equivalent_s=busy_s, per_gpu=util, **coverage(util)))


def summarize_comparison(sampler, *, gpu_uuids, origin_s, tail_end_s,
                         duration_s=150., max_gap_s=1., gpu_uuid_binding_verified=False):
    """Extract service and drain-tail metrics from a sampler or its snapshot.

    A pure-data caller must attest UUID order separately. Missing power produces
    unknown full-window joules; missing utilization never invalidates otherwise
    complete energy. ``energy_comparable`` describes this meter only.
    """
    gpus, uuids = _devices(_get(sampler, 'gpus', ()), gpu_uuids)
    if (not all(_number(v) for v in (origin_s, tail_end_s, duration_s, max_gap_s))
            or duration_s <= 0 or not 0 < max_gap_s <= 1
            or tail_end_s < origin_s + duration_s):
        raise ValueError('invalid service/tail window or gap limit')
    stored_uuids = _get(sampler, 'gpu_uuids')
    if stored_uuids is not None and tuple(stored_uuids) != uuids:
        raise ValueError('sampler UUID order differs from requested comparison')
    power = _power_series(sampler, gpus)
    utilization, timestamp_semantics = _util_series(sampler, gpus)
    boundary = origin_s + duration_s
    service = _phase(power, utilization, gpus, uuids, origin_s, boundary, max_gap_s)
    tail = _phase(power, utilization, gpus, uuids, boundary, tail_end_s, max_gap_s)
    error = _get(sampler, 'error')
    error_at = _get(sampler, 'error_at_s')
    error_in_window = bool(error and (not _number(error_at) or error_at <= tail_end_s))
    source = dict(_get(sampler, 'power_source', {}))
    source_ok = _verified_power_source(sampler, source)
    complete = all(phase['power']['status'] == 'complete' for phase in (service, tail))
    service_j, tail_j = service['power']['energy_j'], tail['power']['energy_j']
    verified = (gpu_uuid_binding_verified is True
                or _get(sampler, 'gpu_uuid_binding_verified', False) is True)
    return dict(schema=SCHEMA, gpu_ids=list(gpus), gpu_uuids=list(uuids), gpu_count=8,
                gpu_uuid_binding_verified=verified, energy_scope='eight_gpu_boards_service_and_tail',
                energy_comparable=bool(verified and complete and source_ok and not error_in_window),
                formal_eligible=False, power_source=source, power_source_verified=source_ok,
                power_error=error, power_error_at_s=error_at,
                power_error_affects_window=error_in_window,
                utilization_source=_get(sampler, 'utilization_source', {}),
                utilization_timestamp_semantics=timestamp_semantics,
                utilization_errors=list(_get(sampler, 'utilization_errors', [])),
                polling_interval_s=_get(sampler, 'interval', _get(sampler, 'polling_interval_s')),
                maximum_interpolation_gap_s=max_gap_s,
                service_start_s=origin_s, service_end_s=boundary, tail_end_s=tail_end_s,
                energy_service_j=service_j, energy_tail_j=tail_j,
                energy_service_tail_j=service_j + tail_j if complete else None,
                service_mean_power_w=service['power']['mean_power_w'],
                gpu_util_mean_pct=service['utilization']['mean_pct'],
                gpu_busy_equivalent_s=service['utilization']['gpu_busy_equivalent_s'],
                util_coverage_fraction=service['utilization']['coverage_fraction'],
                util_max_gap_s=service['utilization']['max_gap_s'],
                util_status=service['utilization']['status'], service=service, tail=tail)


class ComparisonMeteringSession:
    """One verified eight-GPU sampler shared by successive service windows.

    ``stop(after_s=tail_end_s)`` waits briefly for a bracketing power sample.
    A timeout leaves the missing interval visible; it never extrapolates it.
    ``snapshot()`` returns JSON-serializable raw evidence for the caller to save.
    """

    def __init__(self, gpus: Sequence[int], gpu_uuids: Sequence[str], *,
                 interval_s=.1, max_gap_s=1., backend=None):
        self.gpus, self.gpu_uuids = _devices(gpus, gpu_uuids)
        if not _number(interval_s) or interval_s <= 0 or not _number(max_gap_s) or not 0 < max_gap_s <= 1:
            raise ValueError('invalid polling interval or maximum gap')
        self.backend = backend if backend is not None else PynvmlBackend(power_mode='instant')
        if tuple(self.backend.gpu_uuid(g) for g in self.gpus) != self.gpu_uuids:
            raise ValueError('actual physical GPU UUID order differs from lease')
        self.max_gap_s = max_gap_s
        self.sampler = PowerSampler(self.gpus, interval=interval_s, backend=self.backend)
        self._started = False

    def start(self):
        if self._started:
            raise RuntimeError('metering session already started; create a new session to reset evidence')
        self.sampler.start()
        self._started = True
        return self

    def stop(self, *, after_s=None, timeout_s=2.):
        if after_s is not None:
            if not _number(after_s) or not _number(timeout_s) or timeout_s < 0:
                raise ValueError('invalid bracketing sample deadline')
            deadline = time.monotonic() + timeout_s
            while self._started and not self.sampler.error and time.monotonic() < deadline:
                rows = self.sampler.power_metadata
                if rows and min(rows[-1]['read_finished_s']) >= after_s:
                    break
                time.sleep(min(self.sampler.interval, .05))
        self.sampler.stop()

    def snapshot(self):
        # Metadata is appended before the corresponding power sample. Cut it
        # to the copied sample count when a later window is still running.
        samples = list(self.sampler.samples)
        return dict(schema=SCHEMA, gpus=list(self.gpus), gpu_uuids=list(self.gpu_uuids),
                    gpu_uuid_binding_verified=True, samples=samples,
                    power_metadata=list(self.sampler.power_metadata[:len(samples)]),
                    power_source=dict(self.sampler.power_source), error=self.sampler.error,
                    error_at_s=self.sampler.error_at_s,
                    utilization_samples=list(self.sampler.utilization_samples),
                    utilization_readings=list(self.sampler.utilization_readings),
                    utilization_errors=list(self.sampler.utilization_errors),
                    utilization_source=dict(self.sampler.utilization_source),
                    polling_interval_s=self.sampler.interval)

    def summarize(self, *, origin_s, tail_end_s, duration_s=150.):
        return summarize_comparison(self.snapshot(), gpu_uuids=self.gpu_uuids,
                                    origin_s=origin_s, tail_end_s=tail_end_s,
                                    duration_s=duration_s, max_gap_s=self.max_gap_s)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
