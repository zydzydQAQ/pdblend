"""Original pure power_evidence function, source SHA bound in manifest."""
import math
INSTANT_POWER_SOURCE_ID = 'nvml:field:186:scope:0:mW'

def power_evidence(power, source=None, metadata=None):
    """Verify exact field provenance for every raw eight-card power sample."""
    source = source or {}; metadata = metadata or []
    errors = []
    instant = (source.get('mode') == 'instant' and source.get('source_id') == INSTANT_POWER_SOURCE_ID
               and source.get('field_id') == 186 and source.get('scope_id') == 0)
    ages = []; spans = []; previous = [0]*8
    if not instant:
        errors.append('power source is not explicit NVML instant field 186 / GPU scope 0')
    if not power or len(metadata) != len(power):
        errors.append('power metadata does not cover every raw sample')
    if instant and len(metadata) == len(power):
        for (t, watts), row in zip(power, metadata):
            expected = dict(mode='instant', source_id=INSTANT_POWER_SOURCE_ID, field_id=186,
                            scope_id=0, value_type=1, return_code=0)
            if (len(watts) != 8 or row.get('t_s') != t or row.get('gpus') != list(range(8))
                    or any(row.get(k) != [v]*8 for k, v in expected.items())):
                errors.append('inconsistent field source, GPU identity or sample timestamp'); break
            vectors = [row.get(k, []) for k in ('nvml_timestamp_us', 'nvml_latency_us', 'read_started_s', 'read_finished_s')]
            if any(len(values) != 8 for values in vectors):
                errors.append('missing per-GPU NVML timestamp or read interval'); break
            for gpu, (stamp, latency, started, finished) in enumerate(zip(*vectors)):
                if (type(stamp) is not int or stamp <= 0 or stamp < previous[gpu]
                        or type(latency) is not int or latency < 0
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (started, finished))
                        or not started <= finished or not -.05 <= t-finished <= .25
                        or not -.05 <= finished-stamp/1e6 <= .25):
                    errors.append('invalid, stale or regressed NVML timestamp/read interval'); break
                previous[gpu] = stamp; ages.append(finished-stamp/1e6); spans.append(finished-started)
            if errors: break
    return dict(power_mode=source.get('mode', 'unspecified'), power_source_id=source.get('source_id'),
        power_field_id=source.get('field_id'), power_source_verified=instant and not errors,
        power_metadata_schema=1, power_metadata_samples=len(metadata), power_source_errors=errors,
        power_nvml_age_max_s=max(ages) if ages else None,
        power_read_duration_max_s=max(spans) if spans else None,
        power_timebase='host row-end epoch seconds; per-GPU NVML CPU timestamps retained in power_metadata.jsonl')
