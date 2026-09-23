"""Bounded direct-address indices. Compilation may scan; scalar queries never do.

The directory selects a segment, not a rounded query value. Up to four exact
breakpoints can lie in one integer bucket; denser or oversized data is rejected
instead of silently falling back to a search. All interpolation uses the
original floating point input and the original knot coordinates.
"""
from array import array
from bisect import bisect_right
import math

MAX_BREAKPOINTS = 4
MAX_INDEX_BYTES = 64 * 1024 * 1024


class IndexQualificationError(ValueError):
    pass


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


class AxisIndex:
    """At most five fixed comparisons and one direct directory access."""
    def __init__(self, coordinates, *, max_bytes=MAX_INDEX_BYTES):
        self.knots = tuple(coordinates)
        if not self.knots or any(not finite(x) for x in self.knots):
            raise IndexQualificationError('O(1) index requires finite nonempty coordinates')
        if any(a >= b for a, b in zip(self.knots, self.knots[1:])):
            raise IndexQualificationError('O(1) index coordinates must increase strictly')
        self.low, self.high = self.knots[0], self.knots[-1]
        self.origin, end = math.floor(self.low), math.floor(self.high)
        size = end - self.origin + 1
        if size * 4 + len(self.knots) * 8 > max_bytes:
            raise IndexQualificationError('O(1) index memory limit exceeded')
        occupancy = {}
        for point in self.knots:
            bucket = math.floor(point)
            occupancy[bucket] = occupancy.get(bucket, 0) + 1
            if occupancy[bucket] > MAX_BREAKPOINTS:
                raise IndexQualificationError('O(1) index exceeds four breakpoints per context bucket')
        self.directory = array('i', (bisect_right(self.knots, bucket) - 1
                                    for bucket in range(self.origin, end + 1)))
        self.bytes = len(self.directory) * self.directory.itemsize + len(self.knots) * 8

    def lower(self, value):
        if not finite(value) or value < self.low or value > self.high:
            raise ValueError('missing_profile: outside indexed coverage')
        index = self.directory[math.floor(value) - self.origin]
        knots, count = self.knots, len(self.knots)
        # Unrolled constant upper bound, independent of table size. A bucket
        # can contain at most four knots, including its integer boundary.
        if index + 1 < count and knots[index + 1] <= value: index += 1
        if index + 1 < count and knots[index + 1] <= value: index += 1
        if index + 1 < count and knots[index + 1] <= value: index += 1
        if index + 1 < count and knots[index + 1] <= value: index += 1
        return index

    def bracket(self, value):
        index = self.lower(value)
        return (index, index) if self.knots[index] == value else (index, index + 1)


class CurveIndex:
    def __init__(self, coordinates, values, *, expression='difference'):
        self.axis = AxisIndex(coordinates)
        self.values = tuple(values)
        self.expression = expression
        if len(self.values) != len(self.axis.knots) or any(not finite(x) for x in self.values):
            raise IndexQualificationError('invalid indexed curve values')
        self.bytes = self.axis.bytes + len(self.values) * 8

    def predict(self, value):
        index = self.axis.lower(value)
        knots, values = self.axis.knots, self.values
        if index == len(knots) - 1:
            return values[index]
        left, right = values[index], values[index + 1]
        if self.expression == 'difference' and left == right:
            return left
        weight = (value - knots[index]) / (knots[index + 1] - knots[index])
        if self.expression == 'weighted':
            return left * (1 - weight) + right * weight
        return left + weight * (right - left)


class FrequencyIndex:
    def __init__(self, frequencies):
        self.original = tuple(frequencies)
        if any(type(f) is not int or f <= 0 for f in self.original):
            raise IndexQualificationError('frequency tiers must be positive integers')
        self.axis = AxisIndex(sorted(set(self.original)))
        self.order = tuple(self.original.index(f) for f in self.axis.knots)
        self.present = bytearray(self.axis.high + 1)
        for frequency in self.original: self.present[frequency] = 1
        self.bytes = self.axis.bytes + len(self.present)

    def contains(self, frequency):
        if type(frequency) is int:
            return 0 <= frequency < len(self.present) and bool(self.present[frequency])
        if not finite(frequency) or frequency != int(frequency) or frequency < 0 or frequency >= len(self.present):
            return False
        return bool(self.present[int(frequency)])

    def nearest(self, frequency):
        if type(frequency) is int and 0 <= frequency < len(self.present) and self.present[frequency]:
            return frequency
        if not finite(frequency):
            raise ValueError('missing_profile: non-finite frequency')
        if frequency <= self.axis.low: return self.axis.low
        if frequency >= self.axis.high: return self.axis.high
        left, right = self.axis.bracket(frequency)
        a, b = self.axis.knots[left], self.axis.knots[right]
        da, db = frequency - a, b - frequency
        return a if da < db or (da == db and self.order[left] <= self.order[right]) else b


class FrequencyMap(dict):
    """JSON-compatible coefficient mapping with direct scalar retrieval."""
    def __init__(self, values=()):
        self.slots = []
        super().__init__()
        self.update(values)

    def __setitem__(self, key, value):
        if type(key) is not int or not 0 < key < MAX_INDEX_BYTES // 8:
            raise IndexQualificationError('invalid direct-address frequency')
        if key >= len(self.slots): self.slots.extend([None] * (key + 1 - len(self.slots)))
        self.slots[key] = value
        super().__setitem__(key, value)

    def get(self, key, default=None):
        if type(key) is int:
            if key < 0 or key >= len(self.slots): return default
        else:
            if not finite(key) or key != int(key) or key < 0 or key >= len(self.slots): return default
            key = int(key)
        value = self.slots[key]
        return default if value is None else value

    def __getitem__(self, key):
        value = self.get(key)
        if value is None: raise KeyError(key)
        return value

    def __delitem__(self, key):
        super().__delitem__(key)
        self.slots[key] = None

    def update(self, values=(), **kwargs):
        for key, value in dict(values, **kwargs).items(): self[key] = value

    def clear(self):
        super().clear()
        self.slots = []

    def pop(self, key, *default):
        if key not in self: return super().pop(key, *default)
        value = self[key]
        del self[key]
        return value

    def popitem(self):
        key, value = super().popitem()
        self.slots[key] = None
        return key, value

    def setdefault(self, key, default=None):
        if key not in self: self[key] = default
        return self[key]

    def __ior__(self, values):
        self.update(values)
        return self

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(dict(self), memo))

    def __reduce__(self):
        return type(self), (dict(self),)
