"""Measured decode-power tables with explicit, nonrectangular coverage.

The table describes means of complete power windows at their actual effective
token contexts. It never gives nominal prompts, missing batches or frequencies
an implicit affine fallback.
"""
from __future__ import annotations

import math

KIND = 'bounded_context_power_v1'
LOW_BATCH_KIND = 'bounded_context_power_exact_low_batch_v2'


class PowerCoverageError(ValueError):
    """The requested power prediction has no measured coverage."""


def validate(spec):
    if spec.get('kind') not in (KIND, LOW_BATCH_KIND) or spec.get('batch_interpolation') != 'linear':
        raise ValueError('unknown decode power override')
    if spec['kind'] == LOW_BATCH_KIND:
        qualification = spec.get('low_batch_qualification', {})
        if (qualification.get('independent_holdout_passed') is not True or
                qualification.get('exact_batches') != [2, 3] or
                not qualification.get('evidence_sha256') or
                qualification.get('batch_interpolation_qualified') is not False):
            raise ValueError('new low batch schema requires independent exact-batch qualification')
    nodes = spec.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('empty decode power override')
    groups = {}
    for node in nodes:
        b = node.get('batch')
        if type(b) is not int or b < 1 or (b in (2, 3) and spec['kind'] != LOW_BATCH_KIND):
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
    if spec['kind'] == LOW_BATCH_KIND and not {2, 3} <= set(groups):
        raise ValueError('new low batch schema requires measured B2 and B3')


def _batch_family(spec, measured, query):
    if spec.get('kind') == LOW_BATCH_KIND and (query < 4 or measured < 4):
        return measured == query
    return (measured == 1) == (query == 1)


def context_bounds(spec, batch):
    """Exact batch bounds or intersection of the two bracketing batch bounds."""
    if not isinstance(batch, (int, float)) or not math.isfinite(batch) or batch < 1:
        raise PowerCoverageError('missing_profile: invalid decode power batch')
    groups = {}
    for node in spec['nodes']:
        if _batch_family(spec, node['batch'], batch):
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
    batches = sorted({n['batch'] for n in spec['nodes'] if _batch_family(spec, n['batch'], batch)})

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


class CompiledPowerTable:
    """Compiled scalar decoder; diagnostic B1 stays separate from B>=4."""
    def __init__(self, spec):
        from .index import AxisIndex, CurveIndex, IndexQualificationError, MAX_INDEX_BYTES
        validate(spec)
        groups = {}
        for node in spec['nodes']:
            groups.setdefault(node['batch'], []).append(node)
        curves = {}
        for batch, nodes in groups.items():
            coordinates, values = [], []
            for node in sorted(nodes, key=lambda n: n['context_min']):
                for value in (node['context_min'], node['context_max']):
                    if coordinates and coordinates[-1] == value:
                        continue
                    coordinates.append(value); values.append(node['power_w'])
            curves[batch] = CurveIndex(coordinates, values)
        self.single = curves.get(1)
        self.low = {b:curves[b] for b in (2, 3) if b in curves}
        batches = sorted(b for b in curves if b >= 4)
        self.batches = AxisIndex(batches) if batches else None
        self.curves = tuple(curves[b] for b in batches)
        self.bytes = sum(curve.bytes for curve in curves.values()) + (self.batches.bytes if self.batches else 0)
        if self.bytes > MAX_INDEX_BYTES:
            raise IndexQualificationError('O(1) power table memory limit exceeded')

    def _pair(self, batch):
        from .index import finite
        if not finite(batch) or batch < 1:
            raise PowerCoverageError('missing_profile: invalid decode power batch')
        if batch == 1 and self.single is not None:
            return self.single, self.single, 1, 1
        if batch in self.low:
            return self.low[batch], self.low[batch], batch, batch
        if batch == 1 or self.batches is None:
            raise PowerCoverageError('missing_profile: decode power batch outside measured coverage')
        try:
            left, right = self.batches.bracket(batch)
        except ValueError as error:
            raise PowerCoverageError('missing_profile: decode power batch outside measured coverage') from error
        return self.curves[left], self.curves[right], self.batches.knots[left], self.batches.knots[right]

    def context_bounds(self, batch):
        left, right, _, _ = self._pair(batch)
        low, high = max(left.axis.low, right.axis.low), min(left.axis.high, right.axis.high)
        if low > high:
            raise PowerCoverageError('missing_profile: no common power context between batches')
        return low, high

    def predict(self, batch, context):
        left, right, lo, hi = self._pair(batch)
        try:
            value = left.predict(context)
            if lo == hi:
                return value
            return value + (batch - lo) / (hi - lo) * (right.predict(context) - value)
        except (TypeError, ValueError) as error:
            raise PowerCoverageError('missing_profile: decode power context outside measured coverage') from error


def _observe(value, callback):
    if isinstance(value, dict):
        return _ObservedDict(value, callback)
    if isinstance(value, list):
        return _ObservedList(value, callback)
    return value


class _ObservedDict(dict):
    def __init__(self, values=(), callback=None):
        self.callback = callback
        super().__init__((key, _observe(value, callback)) for key, value in dict(values).items())

    def __setitem__(self, key, value):
        super().__setitem__(key, _observe(value, self.callback))
        if self.callback: self.callback()

    def __delitem__(self, key):
        super().__delitem__(key)
        if self.callback: self.callback()

    def update(self, values=(), **kwargs):
        for key, value in dict(values, **kwargs).items():
            super().__setitem__(key, _observe(value, self.callback))
        if self.callback: self.callback()

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return deepcopy(dict(self), memo)

    def clear(self):
        super().clear()
        if self.callback: self.callback()

    def pop(self, key, *default):
        value = super().pop(key, *default)
        if self.callback: self.callback()
        return value

    def popitem(self):
        value = super().popitem()
        if self.callback: self.callback()
        return value

    def setdefault(self, key, default=None):
        if key not in self: self[key] = default
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self


class _ObservedList(list):
    def __init__(self, values=(), callback=None):
        self.callback = callback
        super().__init__(_observe(value, callback) for value in values)

    def __setitem__(self, key, value):
        value = [_observe(v, self.callback) for v in value] if isinstance(key, slice) else _observe(value, self.callback)
        super().__setitem__(key, value)
        if self.callback: self.callback()

    def __delitem__(self, key):
        super().__delitem__(key)
        if self.callback: self.callback()

    def append(self, value):
        super().append(_observe(value, self.callback))
        if self.callback: self.callback()

    def extend(self, values):
        super().extend(_observe(value, self.callback) for value in values)
        if self.callback: self.callback()

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return deepcopy(list(self), memo)

    def insert(self, index, value):
        super().insert(index, _observe(value, self.callback))
        if self.callback: self.callback()

    def pop(self, index=-1):
        value = super().pop(index)
        if self.callback: self.callback()
        return value

    def remove(self, value):
        super().remove(value)
        if self.callback: self.callback()

    def clear(self):
        super().clear()
        if self.callback: self.callback()

    def reverse(self):
        super().reverse()
        if self.callback: self.callback()

    def sort(self, *args, **kwargs):
        super().sort(*args, **kwargs)
        if self.callback: self.callback()

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __imul__(self, count):
        super().__imul__(count)
        if self.callback: self.callback()
        return self


class PowerTableMap(dict):
    """Construction and calibration mutations compile; reads never do."""
    def __init__(self, values=()):
        self.index = []
        super().__init__()
        self.update(values)

    def __setitem__(self, frequency, spec):
        from .index import IndexQualificationError, MAX_INDEX_BYTES
        if type(frequency) is not int or not 0 < frequency < MAX_INDEX_BYTES // 8:
            raise IndexQualificationError('invalid direct-address power frequency')
        if frequency >= len(self.index):
            self.index.extend([None] * (frequency + 1 - len(self.index)))
        super().__setitem__(frequency, _observe(spec, lambda: self._compile(frequency)))
        self._compile(frequency)

    def _compile(self, frequency):
        from .index import IndexQualificationError, MAX_INDEX_BYTES
        # Invalidate before validating a changed spec: a rejected mutation
        # cannot leave the old table silently queryable.
        self.index[frequency] = None
        compiled = CompiledPowerTable(self[frequency])
        size = compiled.bytes + len(self.index) * 8 + sum(table.bytes for table in self.index if table is not None)
        if size > MAX_INDEX_BYTES:
            raise IndexQualificationError('O(1) profile index memory limit exceeded')
        self.index[frequency] = compiled

    def update(self, values=(), **kwargs):
        for frequency, spec in dict(values, **kwargs).items():
            self[frequency] = spec

    def __delitem__(self, frequency):
        super().__delitem__(frequency)
        self.index[frequency] = None

    def clear(self):
        super().clear()
        self.index = []

    def pop(self, key, *default):
        if key not in self:
            return super().pop(key, *default)
        value = self[key]
        del self[key]
        return value

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(dict(self), memo))

    def __reduce__(self):
        from copy import deepcopy
        return type(self), (deepcopy(dict(self)),)

    def popitem(self):
        frequency, value = super().popitem()
        self.index[frequency] = None
        return frequency, value

    def setdefault(self, key, default=None):
        if key not in self: self[key] = default
        return self[key]

    def __ior__(self, values):
        self.update(values)
        return self

    def at(self, frequency):
        if type(frequency) is not int:
            if not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or int(frequency) != frequency:
                raise PowerCoverageError('missing_profile: exact decode power frequency is required')
            frequency = int(frequency)
        if frequency < 0 or frequency >= len(self.index) or self.index[frequency] is None:
            raise PowerCoverageError('missing_profile: exact decode power frequency is required')
        return self.index[frequency]
