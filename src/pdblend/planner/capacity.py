"""Explicit selection and evidence checks for qualified capacity-floor artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

from .pool import QualifiedCapacityFloor
from .forecast import Forecast, InFlightWork
from .transitions import identity


def select_artifact(path, model):
    """A topology-keyed index can bind different resident profiles to different files."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('kind') == 'pdblend_optimization_artifact_set_v1':
        key = f'tp{model.tp}-pp{model.pp}'
        if key not in data.get('profiles', {}):
            raise ValueError('optimization artifact set misses ' + key)
        return (path.parent / data['profiles'][key]).resolve()
    return path


def load_capacity_floors(path, *, model):
    """Require a reproducible controlled GPU acceptance, not a stored passed flag."""
    from pdblend.bench.optimization_acceptance import evaluate_stage
    path = select_artifact(path, model)
    data = json.loads(path.read_text())
    if data.get('kind') == 'pdblend_capacity_floor_v2':
        from pdblend.bench.capacity_floor_v2 import load_v2_floors
        return load_v2_floors(path, model=model)
    if data.get('kind') != 'pdblend_capacity_floor_v1' or data.get('identity') != identity(model):
        raise ValueError('capacity-floor model/TP/profile identity mismatch')
    ref = data.get('acceptance_manifest', {})
    manifest_path = (path.parent / ref['path']).resolve()
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.get('sha256'):
        raise ValueError('capacity-floor acceptance manifest checksum mismatch')
    verdict = evaluate_stage(manifest_path)
    if not verdict['accepted'] or verdict['stage'] != data.get('stage'):
        raise ValueError('capacity floor lacks a passed controlled GPU acceptance: ' + repr(verdict['errors']))
    manifest = json.loads(raw)
    matching = [pair for pair in manifest['pairs'] if
                pair['candidate']['conditions']['model_id'] == identity(model)['model_id']
                and (pair['candidate']['conditions']['tp'], pair['candidate']['conditions']['pp']) == (model.tp, model.pp)]
    if not matching:
        raise ValueError('capacity floor has no accepted trials for this model/TP')
    floors = []
    for floor in data.get('floors', []):
        minimum = floor['min_m_instances']
        if type(minimum) is not int or minimum < 1:
            raise ValueError('capacity floor must be a positive instance count')
        for axis in ('rate_range', 'input_range', 'output_range'):
            values = floor[axis]
            if (len(values) != 2 or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values)
                    or values[0] > values[1]):
                raise ValueError('invalid capacity-floor workload domain')
        slo = floor['slo']
        if set(slo) != {'ttft_s', 'tpot_s'} or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in slo.values()):
            raise ValueError('capacity floor requires an explicit positive SLO')
        trials = [pair for pair in matching if pair['candidate']['conditions']['slo'] == slo and floor['rate_range'][0] <= pair['candidate']['conditions']['rate_rps'] <= floor['rate_range'][1]]
        rates = {pair['candidate']['conditions']['rate_rps'] for pair in trials}
        if not trials or min(rates) > floor['rate_range'][0] or max(rates) < floor['rate_range'][1]:
            raise ValueError('capacity-floor rate bounds were not both tested')
        for pair in trials:
            summary = json.loads((manifest_path.parent / pair['candidate']['summary']['path']).read_text())
            counts = (summary.get('fixed_plan') or {}).get('counts', {})
            if counts.get('M') != minimum or counts.get('P', 0) or counts.get('D', 0):
                raise ValueError('capacity-floor evidence must actually run the claimed fixed M capacity')
            if summary.get('profile_key') != model.profile_key:
                raise ValueError('capacity-floor candidate did not use this exact bound profile')
            shape = summary.get('trace', {})
            for axis, field in (('input_range', 'input'), ('output_range', 'output')):
                if (shape.get(field + '_min') is None or shape.get(field + '_max') is None
                        or shape[field + '_min'] > floor[axis][0] or shape[field + '_max'] < floor[axis][1]):
                    raise ValueError('capacity-floor length bounds lack measured trace coverage')
        floors.append(QualifiedCapacityFloor(identity(model)['model_id'], model.tp, model.pp, minimum,
                      tuple(floor['rate_range']), tuple(floor['input_range']), tuple(floor['output_range']),
                      str(manifest_path) + '#sha256=' + ref['sha256'], qualified=True,
                      profile_key=json.dumps(model.profile_key, sort_keys=True, separators=(',', ':')),
                      accepted_slo=(slo['ttft_s'], slo['tpot_s'])))
    if not floors:
        raise ValueError('empty capacity-floor artifact')
    return tuple(floors)


def floor_identity(floor):
    """Content identity includes the original acceptance-manifest digest."""
    payload = asdict(floor)
    if floor.version == 1:
        for key in ('version', 'frequency_mhz', 'context'):
            payload.pop(key)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def forecast_snapshot(fc):
    return json.loads(json.dumps(asdict(fc), allow_nan=False))


def forecast_from_snapshot(snapshot):
    """Require a full finite forecast, including paired lengths and bound work."""
    if not isinstance(snapshot, dict) or set(snapshot) != {f.name for f in fields(Forecast)}:
        raise ValueError('capacity-floor forecast snapshot is incomplete')
    values = dict(snapshot)
    for key in ('rate_rps', 'trend_rps', 'input_mean', 'input_p95', 'output_mean',
                'peak_rps', 'recent_rate_rps'):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or (key != 'trend_rps' and value < 0):
            raise ValueError('capacity-floor forecast has invalid ' + key)
    for key in ('inflight', 'completed_bins'):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError('capacity-floor forecast has invalid ' + key)
    for key in ('inputs', 'outputs'):
        if not isinstance(values[key], (list, tuple)) or any(type(v) not in (int, float)
                or not math.isfinite(v) or v < 0 for v in values[key]):
            raise ValueError('capacity-floor forecast has invalid ' + key)
        values[key] = tuple(values[key])
    pairs = values['length_pairs']
    if not isinstance(pairs, (list, tuple)) or any(not isinstance(p, (list, tuple)) or len(p) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in p) for p in pairs):
        raise ValueError('capacity-floor forecast has invalid paired lengths')
    values['length_pairs'] = tuple(tuple(p) for p in pairs)
    if not isinstance(values['backlog'], (list, tuple)):
        raise ValueError('capacity-floor forecast has invalid backlog')
    works = []
    for work in values['backlog']:
        if not isinstance(work, dict) or set(work) != {f.name for f in fields(InFlightWork)}:
            raise ValueError('capacity-floor forecast backlog is incomplete')
        if any(type(work[k]) is not int or work[k] < 0 for k in
               ('input_tokens', 'remaining_output_tokens', 'waiting_prefill_tokens', 'kv_tokens')):
            raise ValueError('capacity-floor forecast backlog has invalid work counts')
        if any(not isinstance(work[k], str) for k in ('request_id', 'branch', 'pool_id')):
            raise ValueError('capacity-floor forecast backlog has invalid ownership')
        works.append(InFlightWork(**work))
    values['backlog'] = tuple(works)
    return Forecast(**values)


def capacity_floor_decision(floors, model, fc, slo, *, canonical_floor, context=None,
                            n_m=None, frequency_mhz=None):
    """Pure decision shared by planning and independent acceptance replay."""
    snapshot = forecast_snapshot(fc)
    forecast_from_snapshot(snapshot)
    rejected = {floor_identity(q): q.rejection_reason(model, fc, slo, context=context,
        n_m=n_m, frequency_mhz=frequency_mhz) for q in floors}
    matched = [q for q in floors if rejected[floor_identity(q)] is None]
    effective = min((q.min_m_instances for q in matched), default=canonical_floor)
    result = dict(schema='pdblend-capacity-floor-decision/v1', forecast=snapshot,
        profile_key=json.loads(json.dumps(model.profile_key, allow_nan=False)),
        model_id=model.profile_key.get('model_id', model.model), tp=model.tp, pp=model.pp,
        slo=dict(ttft_s=slo.ttft_s, tpot_s=slo.tpot_s), canonical_floor=canonical_floor,
        effective_floor=effective, configured_floor_ids=sorted(rejected),
        matched_floor_ids=sorted(floor_identity(q) for q in matched),
        selected_floor_ids=sorted(floor_identity(q) for q in matched if q.min_m_instances == effective),
        rejected_floors={key: reason for key, reason in rejected.items() if reason is not None},
        fallback_reason=None if matched else 'no_matching_qualified_floor')
    if any(q.version == 2 for q in floors):
        result.update(schema='pdblend-capacity-floor-decision/v2', context=context or {},
                      n_m=n_m, frequency_mhz=frequency_mhz)
    return result
