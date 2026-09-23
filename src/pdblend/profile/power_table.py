"""Measured decode-power tables with explicit, nonrectangular coverage.

The table describes means of complete power windows at their actual effective
token contexts. It never gives nominal prompts, missing batches or frequencies
an implicit affine fallback.
"""
from __future__ import annotations

import math

KIND = 'bounded_context_power_v1'


class PowerCoverageError(ValueError):
    """The requested power prediction has no measured coverage."""


def validate(spec):
    if spec.get('kind') != KIND or spec.get('batch_interpolation') != 'linear':
        raise ValueError('unknown decode power override')
    nodes = spec.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('empty decode power override')
    groups = {}
    for node in nodes:
        b = node.get('batch')
        if type(b) is not int or b < 1 or b in (2, 3):
            raise ValueError('invalid measured power batch')
        for key in ('context_min', 'context_max', 'power_w'):
            value = node.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid measured power node')
        if node['context_min'] > node['context_max']:
            raise ValueError('inverted measured power context band')
        groups.setdefault(b, []).append(node)
    for points in groups.values():
        points.sort(key=lambda p: p['context_min'])
        if any(a['context_max'] >= b['context_min'] for a, b in zip(points, points[1:])):
            raise ValueError('overlapping measured power context bands')


def context_bounds(spec, batch):
    """Exact batch bounds or intersection of the two bracketing batch bounds."""
    if not isinstance(batch, (int, float)) or not math.isfinite(batch) or batch < 1:
        raise PowerCoverageError('missing_profile: invalid decode power batch')
    groups = {}
    for node in spec['nodes']:
        if (node['batch'] == 1) == (batch == 1):
            groups.setdefault(node['batch'], []).append(node)
    if batch in groups:
        chosen = [batch]
    else:
        lower, upper = [b for b in groups if b < batch], [b for b in groups if b > batch]
        if not lower or not upper:
            raise PowerCoverageError('missing_profile: decode power batch outside measured coverage')
        chosen = [max(lower), min(upper)]
    low = max(min(n['context_min'] for n in groups[b]) for b in chosen)
    high = min(max(n['context_max'] for n in groups[b]) for b in chosen)
    if low > high:
        raise PowerCoverageError('missing_profile: no common power context between batches')
    return low, high


def predict(spec, batch, context):
    if not isinstance(context, (int, float)) or not math.isfinite(context):
        raise PowerCoverageError('missing_profile: actual decode power context is required')
    low, high = context_bounds(spec, batch)
    if not low <= context <= high:
        raise PowerCoverageError('missing_profile: decode power context outside measured coverage')
    batches = sorted({n['batch'] for n in spec['nodes'] if (n['batch'] == 1) == (batch == 1)})

    def at(b):
        points = sorted((n for n in spec['nodes'] if n['batch'] == b), key=lambda n: n['context_min'])
        for p in points:
            if p['context_min'] <= context <= p['context_max']:
                return p['power_w']
        for lo, hi in zip(points, points[1:]):
            if lo['context_max'] < context < hi['context_min']:
                weight = (context-lo['context_max'])/(hi['context_min']-lo['context_max'])
                return lo['power_w'] + weight*(hi['power_w']-lo['power_w'])
        raise PowerCoverageError('missing_profile: incomplete decode power context coverage')

    if batch in batches:
        return at(batch)
    lo, hi = max(b for b in batches if b < batch), min(b for b in batches if b > batch)
    return at(lo) + (batch-lo)/(hi-lo)*(at(hi)-at(lo))
