"""Explicit selection and evidence checks for qualified capacity-floor artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .pool import QualifiedCapacityFloor
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
