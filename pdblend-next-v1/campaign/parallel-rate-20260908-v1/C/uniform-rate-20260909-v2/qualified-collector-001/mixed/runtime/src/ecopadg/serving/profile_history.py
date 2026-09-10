"""Training-only power uncertainty across histories preceding one prefill.

Operator profiles retain the target decode batch/context in their keys even
though every recorded prefill is a single request. The online single-prefill
lookup cannot observe the GPU power state left by the preceding decode batch.
Share power and duration uncertainty across measured histories at the same TP,
clock and input size. Do not share decode bounds or nominal routing estimates.
"""
from dataclasses import replace
import math

from .profiles import ProfilePoint, ProfileStore, validate_profile_observations


def apply_prefill_history_bounds(points, tables):
    """Return model points and an audit; accept only training tables.

    Legacy averaged power and device-limit fallbacks are not observations of
    this state variation. Only fully supported instantaneous single-prefill
    records can contribute; batched P kernels cannot contribute or receive a
    bound derived from single-request kernels. Existing within-bucket errors
    already cover repeated measurements and are retained without fitting.
    """
    points = tuple(points)
    ProfileStore(points)
    training_sources = {p.source_sha256 for p in points}
    eligible = set()
    evidence = {}
    for table in tables:
        source = table.get('source_sha256')
        if source not in training_sources:
            raise ValueError('prefill history evidence must belong to training points')
        if (table.get('power_source', {}).get('mode') != 'instant'
                or table.get('power_source_verified') is not True):
            continue
        phases = [row for row in table.get('phase_power_sources', ())
                  if row.get('phase') == 'prefill']
        for item in table['points']:
            role = item['role']
            if role != 'mixed' and not (role == 'prefill' and item['batch'] == 1):
                continue
            key = (source, role, item['tp'], item['frequency_mhz'],
                   item['input_tokens'], item['context_tokens'], item['batch'])
            rows = [row for row in phases
                    if (row.get('role'), row.get('tp'), row.get('frequency_mhz'),
                        row.get('input_tokens')) == key[1:5]
                    and (role == 'prefill' or row.get('batch') == item['batch'])]
            if not rows:
                continue
            supported = True
            for row in rows:
                power = row.get('power_evidence', {})
                values = [power.get(k) for k in
                          ('started_s', 'finished_s', 'integrated_power_w')]
                supported &= (row.get('source') == 'integrated_nvml_instant'
                              and power.get('sampling_supported') is True
                              and not power.get('fallback_reasons')
                              and all(isinstance(v, (int, float)) and math.isfinite(v)
                                      for v in values)
                              and values[1] > values[0]
                              and values[2] >= 0)
            if supported:
                eligible.add(key)
                original = ProfilePoint(**item)
                validate_profile_observations([original])
                evidence.setdefault(key, []).append(dict(events=len(rows),
                    original_phase_power_w=original.phase_power('prefill'),
                    original_energy_error_fraction=original.energy_error_fraction,
                    original_prefill_s=original.prefill_s,
                    original_error_fraction=original.error_fraction,
                    maximum_observed_duration_s=max(row['power_evidence']['finished_s'] -
                                                     row['power_evidence']['started_s'] for row in rows),
                    maximum_observed_power_w=max(row['power_evidence']['integrated_power_w']
                                                 for row in rows)))
    groups = {}
    for point in points:
        key = (point.source_sha256, point.role, point.tp, point.frequency_mhz,
               point.input_tokens, point.context_tokens, point.batch)
        if key not in eligible:
            continue
        group = groups.setdefault((point.tp, point.frequency_mhz, point.input_tokens), [])
        for original in evidence[key]:
            # Read repeat uncertainty from the original instantaneous table.
            # A cross-table model adjustment may contain averaged-power or
            # device-limit evidence and cannot become an instant contributor.
            original_upper = original['original_phase_power_w'] * (
                1 + original['original_energy_error_fraction'])
            if original['maximum_observed_power_w'] > original_upper * (1 + 1e-12):
                raise ValueError('training bucket fails to cover its own prefill power observations')
            duration_upper = original['original_prefill_s'] * (1 + original['original_error_fraction'])
            if original['maximum_observed_duration_s'] > duration_upper * (1 + 1e-12):
                raise ValueError('training bucket fails to cover its own prefill duration observations')
            upper = max(original['original_phase_power_w'], point.residency_w) * (
                1 + original['original_energy_error_fraction'])
            group.append(dict(role=point.role, batch=point.batch, context_tokens=point.context_tokens,
                              source_sha256=point.source_sha256, power_upper_w=upper,
                              duration_upper_s=duration_upper, **original))
    updated = []
    for point in points:
        group = groups.get((point.tp, point.frequency_mhz, point.input_tokens))
        single_prefill = point.role == 'mixed' or (point.role == 'prefill' and point.batch == 1)
        if group and single_prefill:
            point = replace(point, prefill_power_upper_w=max(point.prefill_power_upper_w,
                            *(entry['power_upper_w'] for entry in group)),
                            prefill_duration_upper_s=max(point.prefill_duration_upper_s,
                            *(entry['duration_upper_s'] for entry in group)))
        updated.append(point)
    audit = [dict(tp=key[0], frequency_mhz=key[1], input_tokens=key[2],
                  prefill_power_upper_w=max(row['power_upper_w'] for row in rows),
                  prefill_duration_upper_s=max(row['duration_upper_s'] for row in rows),
                  contributors=rows) for key, rows in sorted(groups.items())]
    ProfileStore(updated)
    return updated, dict(method='training instantaneous single-prefill history envelope',
                         nominal_estimates_changed=False, prefill_bounds_changed=True,
                         decode_bound_changed=False, groups=audit)
