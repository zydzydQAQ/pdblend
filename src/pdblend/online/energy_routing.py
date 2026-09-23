"""Exact-domain incremental routing energy with independently audited samples.

Ordinary prefill/decode power fits do not qualify this interface. A publisher
must supply independent paired common-window measurements and an independent
holdout for every admitted route key. No interpolation or automatic latest is
used; a missing key deliberately preserves the latency fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCHEMA = 'pdblend.incremental-route-energy/v1'
MIN_REPEATS = 3
MAX_HOLDOUT_RELATIVE_ERROR = .10
KEY_FIELDS = {'model_id', 'tp', 'pp', 'path', 'profile_keys', 'frequencies_mhz',
              'input_tokens', 'max_tokens', 'batch', 'context_tokens',
              'queued_prefill_tokens', 'reservation_tokens'}


def qualified_power_source(value):
    if isinstance(value, dict):
        return (value.get('mode') == 'instant' and value.get('source_id') == 'nvml:field:186:scope:0:mW'
                and value.get('field_id') == 186 and value.get('scope_id') == 0 and value.get('unit') == 'W')
    # Cumulative counters have distinct integration semantics and must be
    # explicitly labeled by a collector; arbitrary affine/average fits fail.
    return value == 'nvml_total_energy_counter'


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _canonical(key):
    if not isinstance(key, dict) or set(key) != KEY_FIELDS:
        raise ValueError('incremental energy key must explicitly bind every route dimension')
    roles = ['M'] if key['path'] == 'M' else ['P', 'D'] if key['path'] == 'PD' else []
    if (not roles or not isinstance(key['model_id'], str) or not key['model_id']
            or type(key['tp']) is not int or key['tp'] not in (1, 2, 4)
            or type(key['pp']) is not int or key['pp'] != 1
            or not isinstance(key['profile_keys'], list) or len(key['profile_keys']) != len(roles)
            or any(not isinstance(x, str) or not x for x in key['profile_keys'])
            or not isinstance(key['frequencies_mhz'], dict) or set(key['frequencies_mhz']) != set(roles)
            or any(type(v) is not int or v <= 0 for v in key['frequencies_mhz'].values())
            or not isinstance(key['reservation_tokens'], dict) or set(key['reservation_tokens']) != set(roles)
            or any(type(v) is not int or v < 0 for v in key['reservation_tokens'].values())
            or any(type(key[k]) is not int or key[k] < 1 for k in ('input_tokens', 'max_tokens', 'batch'))
            or type(key['queued_prefill_tokens']) is not int or key['queued_prefill_tokens'] < 0
            or not _number(key['context_tokens'], positive=True)):
        raise ValueError('invalid incremental route domain identity')
    # Canonicalize numeric representation without coarsening the exact domain.
    key = dict(key, context_tokens=float(key['context_tokens']))
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def route_key(router, choice, context):
    path, p, d = choice
    ids = [p] if path == 'M' else [p, d]
    roles = ['M'] if path == 'M' else ['P', 'D']
    load = router.loads[d]
    if any((router.loads[i].model_id, router.loads[i].tp, router.loads[i].pp) !=
           (load.model_id, load.tp, load.pp) for i in ids):
        raise ValueError('incremental energy requires identical P/D model and topology')
    return dict(model_id=load.model_id, tp=load.tp, pp=load.pp, path=path,
                profile_keys=[router.loads[i].profile_key for i in ids],
                frequencies_mhz={role: context['frequencies'][i] for role, i in zip(roles, ids)},
                input_tokens=context['input_tokens'], max_tokens=context['max_tokens'], batch=context['batch'],
                context_tokens=context['context_tokens'], queued_prefill_tokens=context['queued_prefill_tokens'],
                reservation_tokens={role: context['reservation_tokens'][i] for role, i in zip(roles, ids)})


def _read_component(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('incremental energy component requires a path and immutable SHA256')
    path = (root / ref['path']).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref['sha256']:
        raise ValueError('incremental energy component digest mismatch')
    data = json.loads(payload)
    if not isinstance(data, list) or not data:
        raise ValueError('incremental energy component must contain raw paired windows')
    return path, data


def _samples(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get('measurement') != 'paired_common_window'
                or not qualified_power_source(row.get('power_source'))
                or not isinstance(row.get('sample_id'), str) or not row['sample_id']
                or row['sample_id'] in seen):
            raise ValueError('incremental energy requires distinct measured sample IDs and native power evidence')
        seen.add(row['sample_id'])
        key = _canonical(row.get('key'))
        if (row.get('completion_tokens') != row['key']['max_tokens']
                or row.get('error') is not None or not isinstance(row.get('request_id'), str)
                or not row['request_id'] or not _number(row.get('ttft_s')) or not _number(row.get('tpot_s'))):
            raise ValueError('incremental energy sample must bind successful measured request output and timing')
        devices = row.get('gpu_uuids')
        if (not isinstance(devices, list) or len(devices) != row['key']['tp'] * len(row['key']['profile_keys'])
                or len(set(devices)) != len(devices) or any(not isinstance(x, str) or not x for x in devices)):
            raise ValueError('incremental energy must meter every route GPU exactly once')
        windows = [row.get('baseline'), row.get('with_request')]
        for window in windows:
            if (not isinstance(window, dict) or not _number(window.get('energy_j'))
                    or not _number(window.get('start_s')) or not _number(window.get('end_s'))
                    or window['end_s'] <= window['start_s']):
                raise ValueError('incremental energy requires finite paired window integrals')
        baseline, measured = windows
        durations = [w['end_s']-w['start_s'] for w in windows]
        if (not math.isclose(*durations, rel_tol=.01, abs_tol=1e-6)
                or max(w['start_s'] for w in windows) < min(w['end_s'] for w in windows)):
            raise ValueError('baseline and request windows must be distinct and have equal duration')
        energy = measured['energy_j'] - baseline['energy_j']
        if not _number(energy, positive=True):
            raise ValueError('incremental request energy must exceed measurement noise and be positive')
        grouped[key].append(dict(sample_id=row['sample_id'], request_id=row['request_id'], energy_j=energy,
                                 gpu_uuids=tuple(devices), windows=windows,
                                 ttft_s=row['ttft_s'], tpot_s=row['tpot_s']))
    return grouped


class MeasuredEnergyEstimator:
    def __init__(self, router, values, evidence):
        self.router, self.values, self.evidence = router, values, evidence

    def __call__(self, choice, context):
        key = route_key(self.router, choice, context)
        value = self.values.get(_canonical(key))
        if value is None:
            raise ValueError('incremental energy query outside measured coverage')
        return dict(qualified=True, incremental_energy_j=value['energy_j'], profile_keys=key['profile_keys'],
                    coverage=key, evidence=self.evidence, holdout=value['holdout'],
                    measured_ttft_s=value['ttft_s'], measured_tpot_s=value['tpot_s'])


def load_energy_estimator(path, *, router):
    """Audit immutable measurement components before creating a route scorer."""
    path = Path(path).resolve()
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA
            or manifest.get('system') != 'pdblend'):
        raise ValueError('expected an explicitly versioned incremental request-energy artifact')
    train_path, train_rows = _read_component(path.parent, manifest.get('training'))
    holdout_path, holdout_rows = _read_component(path.parent, manifest.get('holdout'))
    if train_path == holdout_path or manifest['training']['sha256'] == manifest['holdout']['sha256']:
        raise ValueError('incremental energy requires an independent holdout component')
    training, holdout = _samples(train_rows), _samples(holdout_rows)
    if set(training) != set(holdout):
        raise ValueError('every incremental route requires matching independent holdout coverage')
    all_train_ids = {r['sample_id'] for rows in training.values() for r in rows}
    all_holdout_ids = {r['sample_id'] for rows in holdout.values() for r in rows}
    if all_train_ids & all_holdout_ids:
        raise ValueError('incremental energy holdout reuses training sample IDs')
    values, ranking = {}, defaultdict(list)
    for key, rows in training.items():
        validation = holdout[key]
        if len(rows) < MIN_REPEATS or len(validation) < MIN_REPEATS:
            raise ValueError('incremental energy requires three training and three independent holdout windows')
        combined = rows + validation
        if len({r['request_id'] for r in combined}) != len(combined):
            raise ValueError('incremental energy repeats must execute distinct requests')
        # Compare observations on the same physical fleet to avoid approving
        # a device change as model validation. Deployment keys remain reusable
        # only through a matching calibrated profile key.
        if len({r['gpu_uuids'] for r in combined}) != 1:
            raise ValueError('training and holdout GPU identities differ')
        intervals = sorted((w['start_s'], w['end_s']) for row in combined for w in row['windows'])
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('incremental energy training/holdout windows are not independent')
        energy = mean(row['energy_j'] for row in rows)
        errors = [abs(energy-row['energy_j']) / row['energy_j'] for row in validation]
        if max(errors) > MAX_HOLDOUT_RELATIVE_ERROR:
            raise ValueError('incremental energy independent-window error exceeds 10 percent')
        values[key] = dict(energy_j=energy, ttft_s=max(r['ttft_s'] for r in validation),
                           tpot_s=max(r['tpot_s'] for r in validation),
                           holdout=dict(windows=len(validation), max_relative_error=max(errors)))
        domain = json.loads(key)
        workload = tuple(domain[k] for k in ('model_id', 'input_tokens', 'max_tokens', 'batch',
                                           'context_tokens', 'queued_prefill_tokens'))
        ranking[workload].append((energy, mean(row['energy_j'] for row in validation)))
    for alternatives in ranking.values():
        if len(alternatives) < 2:
            raise ValueError('incremental energy requires alternative-route ranking holdouts for every workload')
        for i, (train_a, val_a) in enumerate(alternatives):
            for train_b, val_b in alternatives[i+1:]:
                if (train_a-train_b) * (val_a-val_b) < 0:
                    raise ValueError('incremental energy holdout reverses candidate ordering')
    evidence = dict(schema=SCHEMA, manifest_sha256=hashlib.sha256(payload).hexdigest(),
                    training_sha256=manifest['training']['sha256'], holdout_sha256=manifest['holdout']['sha256'],
                    max_holdout_relative_error=MAX_HOLDOUT_RELATIVE_ERROR, minimum_repeats=MIN_REPEATS,
                    ranking_preserved=True, hardware_qualified=False, formal_eligible=False)
    return MeasuredEnergyEstimator(router, values, evidence)
