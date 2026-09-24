"""Read-only validation of the first real isolated eight-board observations."""
from __future__ import annotations

import math

from .comparison_metering import summarize_comparison


def qualify_startup_snapshot(snapshot, method, gpu_uuids):
    """Use unchanged integration/source checks; this never qualifies a window."""
    if (method.get('status') != 'running' or method.get('child_alive') is not True
            or method.get('window_guard_active') is not False or method.get('error')
            or not method.get('liveness_observations')
            or method['liveness_observations'][-1].get('sampler_thread_alive') is not True
            or method['liveness_observations'][-1].get('sampler_error')):
        raise ValueError('isolated sampler startup/liveness is unqualified')
    if (snapshot.get('gpu_uuids') != list(gpu_uuids) or snapshot.get('gpus') != list(range(8))
            or snapshot.get('gpu_uuid_binding_verified') is not True or snapshot.get('error')):
        raise ValueError('isolated startup physical UUID or sampler error differs')
    samples, metadata, clocks = (snapshot.get(name, []) for name in
                                 ('samples', 'power_metadata', 'frequency_samples'))
    if len(samples) < 3 or len(metadata) != len(samples) or len(clocks) < 3 or len(clocks) > len(samples):
        raise ValueError('isolated startup samples/metadata/frequencies are incomplete')
    # A concurrent read may observe power just before its corresponding clock
    # append. Validate the available prefix without editing any raw sample.
    for (stamp, values), (power_stamp, _) in zip(clocks, samples):
        if (stamp != power_stamp or len(values) != 8 or any(type(v) not in (int,float)
                or not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError('isolated startup frequency order/values are invalid')
    # NVML power and utilization are sequential acquisitions, not simultaneous.
    # Choose the startup-only interval inside both streams on every device.
    # Formal service boundaries remain fixed and are never cropped this way.
    utilization = {gpu: [] for gpu in range(8)}
    for row in snapshot.get('utilization_readings', []):
        gpu, stamp = row.get('gpu'), row.get('read_finished_s', row.get('t_s'))
        if gpu not in utilization or type(stamp) not in (int, float) or not math.isfinite(stamp):
            raise ValueError('isolated startup utilization device/timestamp is invalid')
        utilization[gpu].append(stamp)
    if any(len(stamps) < 3 for stamps in utilization.values()):
        raise ValueError('isolated startup requires per-device utilization observations')
    first = max(*metadata[0]['read_finished_s'], *(min(v) for v in utilization.values()))
    last = min(*metadata[len(clocks)-1]['read_finished_s'], *(max(v) for v in utilization.values()))
    if last-first < 2.:
        raise ValueError('isolated startup requires two real seconds of eight-card observations')
    # Interior endpoints retain bracketing samples on every physical device.
    origin, end = first+.001, last-.001
    summary = summarize_comparison(snapshot, gpu_uuids=gpu_uuids, origin_s=origin,
                                   tail_end_s=end, duration_s=end-origin)
    if not summary['energy_comparable'] or summary['util_status'] != 'complete':
        raise ValueError('isolated startup power/utilization does not pass unchanged public integration')
    return dict(schema='isolated-comparison-meter-startup-preflight/v1', passed=True,
        hardware_executed=True, formal_eligible=False, evaluation_window_qualified=False,
        scope='same exclusive lease before model load', physical_gpu_uuids=list(gpu_uuids),
        observed_span_s=last-first, frequency_sample_count=len(clocks),
        public_metering=summary, method_child_pid=method['child_pid'])
