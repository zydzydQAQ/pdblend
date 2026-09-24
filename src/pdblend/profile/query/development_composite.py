"""Explicit development composition of new observations and inherited profiles.

Interpolation is bounded by the training shape envelope. Missing domains retain
the checksum-bound legacy model and are reported as inheritance, never as newly
measured or formally qualified evidence. No fitting or evidence I/O occurs in a
serving query.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

KIND = 'pdblend_development_composite_profile_v1'


def _bound(ref):
    path = Path(ref['path']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('development profile evidence checksum mismatch: ' + str(path))
    return path


def _between(nodes, value):
    """Return exact node or adjacent nodes; never extrapolate."""
    nodes = sorted(nodes)
    if value in nodes:
        return value, value
    if not nodes or not nodes[0] <= value <= nodes[-1]:
        return None
    return next((a, b) for a, b in zip(nodes, nodes[1:]) if a < value < b)


def _lerp(x, a, b, va, vb):
    return va if a == b else va + (vb - va) * (x - a) / (b - a)


class DevelopmentCompositeModel:
    pd_first_gap_source = 'development_endpoint_first_gap_including_first_decode'
    pd_first_gap_extended_domain = True

    def __init__(self, base, compiled, profile_key):
        self.base = base
        self.profile_key = deepcopy(profile_key)
        self.compiled = deepcopy(compiled)
        self.freqs = tuple(base.freqs)
        self._counts = Counter()
        self._query_log = Counter()
        self._timing = {}
        for row in compiled['timing_models']:
            spec = deepcopy(row)
            vertices = np.asarray(row['coverage_vertices'])
            spec['_domain'] = (ConvexHull(vertices).equations if row['role'] == 'decode'
                               else (float(vertices.min()), float(vertices.max())))
            self._timing.setdefault(row['role'], {})[row['frequency_mhz']] = spec
        self._power = {}
        for key, nodes in compiled['power_nodes'].items():
            family, frequency, batch, chunk, rate = key.split('/')
            coords = {}
            for node in nodes:
                coords[node['context_min']] = node['power_w']
                coords[node['context_max']] = node['power_w']
            self._power[(family, float(frequency), float(batch), float(chunk), float(rate))] = coords
        self._capacity = compiled.get('capacity', {})

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _record(self, component, source, query=()):
        self._counts[component + ':' + source] += 1
        self._query_log[(component, source, tuple(query))] += 1

    def query_provenance_log(self):
        """Lossless aggregation of identical queries; never log once per call."""
        return [dict(component=c, source=s, query=list(q), count=n)
                for (c, s, q), n in self._query_log.items()]

    def query_provenance_summary(self):
        return dict(profile_key=deepcopy(self.profile_key), query_counts=dict(self._counts),
                    components=deepcopy(self.compiled['summary']),
                    effective_kv_capacity_tokens=self.kv_capacity_tokens,
                    capacity_composition='minimum of inherited and new observed physical capacity',
                    query_arguments=dict(prefill_timing=['frequency_mhz', 'input_tokens'],
                        prefill_marginal=['frequency_mhz', 'input_tokens'],
                        decode_timing=['frequency_mhz', 'batch', 'context_tokens'],
                        decode=['frequency_mhz', 'batch', 'context_tokens'],
                        decode_power=['frequency_mhz', 'batch', 'context_tokens',
                                      'chunk_tokens_if_native', 'prefill_rate_if_native'],
                        mixed_power=['frequency_mhz', 'batch', 'context_tokens', 'chunk_tokens', 'prefill_rate_rps'],
                        first_gap=['input_tokens', 'output_tokens', 'f_P_mhz', 'f_D_mhz',
                                   'batch', 'context_tokens', 'prefill_instance', 'decode_instance']),
                    development_only=True, formal_eligible=False)

    @property
    def query_qualification(self):
        return dict(self.base.query_qualification, development_composite=True,
                    interpolation_qualified=False, formal_eligible=False,
                    complexity='bounded component tables, compiled at load',
                    components=deepcopy(self.compiled['summary']))

    @property
    def kv_capacity_tokens(self):
        observed = self._capacity.get('actual_total_kv_tokens')
        return min(self.base.kv_capacity_tokens, observed) if observed else self.base.kv_capacity_tokens

    def _physical_supported(self, batch, ctx):
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (batch, ctx)):
            return False
        block = self._capacity.get('block_size', 16)
        return (batch <= self._capacity.get('max_num_seqs', 32) and ctx <= 8192
                and batch * math.ceil(ctx / block) * block <= .9 * self.kv_capacity_tokens)

    @staticmethod
    def _covered(spec, coords):
        if spec['role'] == 'prefill':
            lo, hi = spec['_domain']
            return lo - 1e-9 <= coords[0] <= hi + 1e-9
        eq = spec['_domain']
        return bool(np.all(eq[:, :-1] @ coords + eq[:, -1] <= 1e-9))

    def _timing_value(self, role, frequency, coords, features):
        models = self._timing.get(role, {})
        bracket = _between(models, frequency)
        if not bracket:
            return None
        low, high = bracket
        if not all(self._covered(models[f], coords) for f in set(bracket)):
            return None
        values = [float(np.dot(models[f]['coefficients'], features)) / 1000 for f in bracket]
        source = 'native_fit' if low == high else 'frequency_interpolation'
        component = 'prefill_marginal' if role == 'prefill' and features[0] == 0 else role + '_timing'
        query = [frequency, coords[0]*8192] if role == 'prefill' else [frequency, coords[0], coords[1]*8192]
        self._record(component, source, query)
        return max(_lerp(frequency, low, high, *values), 0.)

    def prefill_seconds(self, n, f):
        x = n / 8192
        value = self._timing_value('prefill', f, [x], [1., x, x*x])
        if value is not None:
            return max(value, 1e-4)
        self._record('prefill_timing', 'inherited', [f, n])
        return self.base.prefill_seconds(n, f)

    def prefill_marginal_seconds(self, n, f):
        x = n / 8192
        value = self._timing_value('prefill', f, [x], [0., x, x*x])
        if value is not None:
            return value
        self._record('prefill_marginal', 'inherited', [f, n])
        return self.base.prefill_marginal_seconds(n, f)

    def decode_supported(self, batch, ctx, f):
        if not self._physical_supported(batch, ctx):
            self._record('decode', 'unsupported_capacity', [f, batch, ctx])
            return False
        if f not in self.freqs:
            return False
        # Missing new shape coverage explicitly inherits the original domain.
        models = self._timing.get('decode', {})
        bracket = _between(models, f)
        if bracket and all(self._covered(models[v], [batch, ctx/8192]) for v in set(bracket)):
            return True
        return self.base.decode_supported(batch, ctx, f)

    def step_seconds(self, batch, ctx, f):
        if not self._physical_supported(batch, ctx):
            self._record('decode', 'unsupported_capacity', [f, batch, ctx])
            raise ValueError('missing_profile: observed physical KV/sequence capacity exceeded')
        value = self._timing_value('decode', f, [batch, ctx/8192], [1., batch, batch*ctx/8192])
        if value is not None:
            return max(value, 1e-4)
        self._record('decode_timing', 'inherited', [f, batch, ctx])
        return self.base.step_seconds(batch, ctx, f)

    def _power_value(self, family, batch, ctx, f, chunk=0., rate=0.):
        if ctx is None or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                  for v in (batch, ctx, f, chunk, rate)):
            return None
        frequencies = {key[1] for key in self._power if key[0] == family and key[3:] == (chunk, rate)}
        frequencies = _between(frequencies, f)
        if not frequencies:
            return None
        frequency_values = []
        interpolated = frequencies[0] != frequencies[1]
        for freq in frequencies:
            batches = _between({key[2] for key in self._power
                if key[0:2] == (family, freq) and key[3:] == (chunk, rate)}, batch)
            if not batches:
                return None
            values = []
            interpolated |= batches[0] != batches[1]
            for b in batches:
                curve = self._power[(family, freq, b, chunk, rate)]
                contexts = _between(curve, ctx)
                if not contexts:
                    return None
                a, z = contexts
                values.append(_lerp(ctx, a, z, curve[a], curve[z]))
                interpolated |= a != z
            frequency_values.append(_lerp(batch, *batches, *values))
        self._record(family + '_power', 'interpolation' if interpolated else 'native_node',
                     [f, batch, ctx, chunk, rate])
        return _lerp(f, *frequencies, *frequency_values)

    def decode_power_supported(self, batch, ctx, f):
        if not self._physical_supported(batch, ctx):
            return False
        return (self._power_value('decode', batch, ctx, f) is not None
                or self.base.decode_power_supported(batch, ctx, f))

    def decode_power_w(self, batch, f, *, ctx=None):
        value = self._power_value('decode', batch, ctx, f)
        if value is not None:
            return value
        self._record('decode_power', 'inherited', [f, batch, ctx])
        return self.base.decode_power_w(batch, f, ctx=ctx)

    def mixed_power_supported(self, batch, ctx, f, *, chunk_tokens, prefill_rate_rps):
        value = self._power_value('mixed', batch, ctx, f, chunk_tokens, prefill_rate_rps)
        if value is None:
            self._record('mixed_power', 'inherited_composed_model', [f, batch, ctx, chunk_tokens, prefill_rate_rps])
        return value is not None

    def mixed_power_w(self, batch, ctx, f, *, chunk_tokens, prefill_rate_rps):
        value = self._power_value('mixed', batch, ctx, f, chunk_tokens, prefill_rate_rps)
        if value is None:
            raise ValueError('missing_profile: mixed power component outside observed domain')
        return value

    def prefill_power_w(self, n, f):
        self._record('prefill_power', 'inherited', [f, n])
        return self.base.prefill_power_w(n, f)

    def prefill_energy_j(self, n, f):
        return self.prefill_seconds(n, f) * self.prefill_power_w(n, f)

    def token_energy_j(self, batch, ctx, f):
        return self.step_seconds(batch, ctx, f) * self.decode_power_w(batch, f, ctx=ctx) / max(batch, 1e-6)

    def static_power_w(self, state, f=None):
        self._record('static_power', 'inherited', [state, f])
        return self.base.static_power_w(state, f)

    def wake_seconds(self, state):
        self._record('wake', 'inherited', [state])
        return self.base.wake_seconds(state)

    def transfer_seconds(self, tokens):
        self._record('transfer', 'inherited', [tokens])
        return self.base.transfer_seconds(tokens)

    def pd_first_gap_seconds(self, *, input_tokens, output_tokens, f_P_mhz, f_D_mhz,
                             batch, context_tokens, prefill_instance, decode_instance):
        nodes = self.compiled.get('handoff_nodes', [])
        applicable = [n for n in nodes if n['output_tokens'] == output_tokens
            and n['requested_f_P_mhz'] == f_P_mhz and n['requested_f_D_mhz'] == f_D_mhz]
        # An output-16 endpoint observation does not prove a two-token risk
        # estimate, another requested clock, or a loaded decode batch.
        by_input = {n['input_tokens']: n for n in applicable}
        bracket = _between(by_input, input_tokens)
        if (batch != 1 or context_tokens != input_tokens + output_tokens or not bracket
                or prefill_instance == decode_instance):
            self._record('first_gap', 'unsupported_domain',
                [input_tokens, output_tokens, f_P_mhz, f_D_mhz, batch, context_tokens, prefill_instance, decode_instance])
            raise ValueError('missing_profile: endpoint first-gap output/clock/batch/input domain unavailable')
        values = [by_input[v]['observed_training_max_first_gap_s'] for v in bracket]
        self._record('first_gap', 'development_endpoint_interpolation',
            [input_tokens, output_tokens, f_P_mhz, f_D_mhz, batch, context_tokens, prefill_instance, decode_instance])
        return _lerp(input_tokens, *bracket, *values)


def load_development_profile(path, *, system, model_id, tp, pp, usage):
    from .versions import LoadedVersion, VersionError, load_profile
    if usage != 'development':
        raise VersionError('development composite is not a formally qualified profile')
    path = Path(path).resolve()
    descriptor = json.loads(path.read_text())
    expected = dict(system=system, model_id=model_id, tp=tp, pp=pp)
    if descriptor.get('kind') != KIND or any(descriptor.get(k) != v for k, v in expected.items()):
        raise VersionError('development profile system/model/TP/PP mismatch')
    base_path, compiled_path = _bound(descriptor['base_profile']), _bound(descriptor['compiled'])
    compiled = json.loads(compiled_path.read_text())
    if any(compiled['identity'].get(k) != v for k, v in expected.items()):
        raise VersionError('development component identity mismatch')
    for ref in compiled['source_bindings']:
        _bound(ref)
    base = load_profile(base_path, **expected, usage=usage)
    key = dict(base.profile_key, source='explicit_development_composite',
        profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        base_profile_sha256=descriptor['base_profile']['sha256'],
        compiled_sha256=descriptor['compiled']['sha256'])
    identity = dict(compiled['identity'], source=key['source'], profile_sha256=key['profile_sha256'])
    model = DevelopmentCompositeModel(base.model, compiled, key)
    coverage = dict(base.coverage, development_composite=deepcopy(compiled['summary']))
    qualification = dict(usage=usage, formal_eligible=False, full_profile_qualified=False,
        development_composite=True, consumer_loader_available=True, planner_automatically_wired=True,
        holdout_used_for_fitting=False, evaluation_used_for_selection=False,
        transition_costs_qualified=False, inherited_components_explicit=True,
        components=deepcopy(compiled['summary']), query_index=model.query_qualification)
    model.calibration_identity, model.calibration_coverage = deepcopy(identity), deepcopy(coverage)
    model.calibration_qualification = deepcopy(qualification)
    return LoadedVersion(model, identity, coverage, qualification, key)
