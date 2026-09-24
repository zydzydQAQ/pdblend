"""Independent DistServe stage fits with measured feature-hull coverage.

The timing model is a nonnegative affine fit of request count, token work and
attention work. Each frequency and role has its own fit. Independent windows
validate timing and group power; unqualified fits cannot drive deployment.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.spatial import ConvexHull

from ..native_profile import DIST_FREQS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def features(lengths):
    if not lengths or any(type(n) is not int or not 1 <= n <= 8192 for n in lengths):
        raise ValueError('positive actual per-request lengths within 8192 required')
    return [float(len(lengths)), sum(lengths)/8192., sum(n*n for n in lengths)/(8192.*8192.)]


def coverage_features(lengths):
    features(lengths)  # Apply the same strict length validation.
    # Coverage describes actual shape coordinates, not polynomial basis terms.
    # Using length squared as a hull coordinate would reject every interior
    # batch-one length on its curved quadratic feature manifold.
    return [float(len(lengths)), sum(lengths)/8192., min(lengths)/8192., max(lengths)/8192.]


def measured_events(sample, *, tp):
    """Match all ranks by ordered requests and exact observed token vectors."""
    ranks = sample.get('ranks', [])
    if len(ranks) != tp or {r.get('rank') for r in ranks} != set(range(tp)):
        raise ValueError('DistServe all-rank shape evidence missing')
    ranks = sorted(ranks, key=lambda row: row['rank'])
    selected = []
    for rank in ranks:
        rows = []
        for event in rank.get('samples', []):
            role = event.get('role')
            if role not in ('prefill', 'decode'):
                continue
            if (event.get('system') != 'distserve' or event.get('measurement_scope') != 'runner'
                    or event.get('tp') != tp or event.get('pp') != 1 or event.get('rank') != rank['rank']
                    or event.get('failed') or not math.isfinite(event.get('gpu_elapsed_ms', float('nan')))
                    or event['gpu_elapsed_ms'] <= 0):
                raise ValueError('DistServe event identity or CUDA timing differs')
            prompts, contexts, scheduled = [event.get(key) for key in ('prompt_lengths','context_lengths','scheduled_lengths')]
            if (not all(isinstance(v, list) and len(v)==event['batch'] for v in (prompts, contexts, scheduled))
                    or len(event.get('request_ids', [])) != event['batch']):
                raise ValueError('per-request heterogeneous shape evidence missing')
            features(prompts); features(contexts)
            if role == 'prefill' and (prompts != contexts or scheduled != prompts):
                continue  # A chunk fragment is not a complete prefill service.
            if role == 'decode' and scheduled != [1]*event['batch']:
                raise ValueError('decode is not one real token per scheduled request')
            rows.append(event)
        selected.append(rows)
    def identity(row):
        return tuple((key, tuple(row[key]) if isinstance(row[key], list) else row[key])
                     for key in ('request_ids','role','prompt_lengths','context_lengths','scheduled_lengths'))
    if any(list(map(identity, rows)) != list(map(identity, selected[0])) for rows in selected[1:]):
        raise ValueError('DistServe rank request/shape alignment differs')
    return [dict(role=row['role'], lengths=row['prompt_lengths'] if row['role']=='prefill' else row['context_lengths'],
                 latency_ms=max(rows[i]['gpu_elapsed_ms'] for rows in selected), at_s=row['at_s'],
                 request_ids=row['request_ids']) for i, row in enumerate(selected[0])]


def in_hull(vertices, target):
    vertices = np.asarray(vertices, dtype=float)
    target = np.asarray(target, dtype=float)
    if not len(vertices): return False
    # Exact hits avoid an LP at the most frequent simulator cells.
    if np.any(np.all(np.isclose(vertices, target, rtol=0, atol=1e-9), axis=1)): return True
    result = linprog(np.zeros(len(vertices)), A_eq=np.vstack((vertices.T, np.ones(len(vertices)))),
                     b_eq=np.r_[target, 1.], bounds=(0, None), method='highs')
    return bool(result.success and np.max(np.abs(vertices.T @ result.x-target)) <= 1e-7)


class _Hull:
    """Cache affine-hull halfspaces so every simulated decode avoids an LP."""
    def __init__(self, vertices):
        points=np.asarray(vertices,dtype=float);self.origin=points[0]
        delta=points-self.origin
        _,singular,v=np.linalg.svd(delta,full_matrices=False)
        self.basis=v[singular>1e-10].T
        projected=delta @ self.basis
        self.rank=self.basis.shape[1]
        self.bounds=(projected.min(axis=0),projected.max(axis=0))
        self.equations=ConvexHull(projected).equations if self.rank>1 else None
    def contains(self,target):
        delta=np.asarray(target)-self.origin;projected=delta @ self.basis
        if np.max(np.abs(delta-self.basis @ projected))>1e-7:return False
        if self.equations is not None:
            return bool(np.all(self.equations[:,:-1] @ projected+self.equations[:,-1]<=1e-7))
        return bool(np.all(projected>=self.bounds[0]-1e-7) and np.all(projected<=self.bounds[1]+1e-7))


def _fit(rows, target):
    x = np.asarray([[1., *features(row['lengths'])] for row in rows])
    y = np.asarray([row[target] for row in rows])
    coefficients, _ = nnls(x, y)
    errors = np.abs((x @ coefficients-y)/y)
    return coefficients.tolist(), errors.tolist()


def _predict(coefficients, lengths):
    return float(np.dot([1., *features(lengths)], coefficients))


def fit_surface(training, holdout, *, identity, raw_bindings, measurement_qualification):
    """Never fit on holdout; retain failed or uncovered cells explicitly."""
    if identity.get('system') != 'distserve' or identity.get('pp') != 1:
        raise ValueError('independent DistServe PP1 identity required')
    if set(row['window_id'] for row in training) & set(row['window_id'] for row in holdout):
        raise ValueError('training and independent holdout overlap')
    cells = []
    for frequency in DIST_FREQS:
        for role in ('prefill', 'decode'):
            train = [r for r in training if (r['frequency_mhz'],r['role'])==(frequency,role)]
            test = [r for r in holdout if (r['frequency_mhz'],r['role'])==(frequency,role)]
            cell = dict(frequency_mhz=frequency,role=role,status='missing_profile')
            cells.append(cell)
            if not train or not test: continue
            vertices = [list(vertex) for vertex in sorted({tuple(coverage_features(row['lengths'])) for row in train})]
            hull = _Hull(vertices)
            latency, errors = _fit(train, 'latency_ms')
            power_train = [r for r in train if r.get('power_w', 0)>0]
            if not power_train: continue
            power, power_errors = _fit(power_train, 'power_w')
            observed = []
            for row in test:
                covered = hull.contains(coverage_features(row['lengths']))
                observed.append(dict(window_id=row['window_id'],covered=covered,
                    timing_relative_error=abs(_predict(latency,row['lengths'])/row['latency_ms']-1),
                    power_relative_error=abs(_predict(power,row['lengths'])/row['power_w']-1)))
            hetero = lambda rows:any(len(set(row['lengths']))>1 for row in rows)
            gates = dict(training_timing=max(errors)<=.10, training_power_mape=float(np.mean(power_errors))<=.10,
                training_power_max=max(power_errors)<=.15, holdout_covered=all(r['covered'] for r in observed),
                holdout_timing=max(r['timing_relative_error'] for r in observed)<=.10,
                holdout_power=max(r['power_relative_error'] for r in observed)<=.10,
                heterogeneous_training=hetero(train),heterogeneous_holdout=hetero(test),
                measurement_qualified=measurement_qualification.get('passed') is True)
            cell.update(status='qualified' if all(gates.values()) else 'calibration_failed',gates=gates,
                latency_coefficients=latency,power_coefficients=power,vertices=vertices,
                length_bounds=[min(min(r['lengths']) for r in train),max(max(r['lengths']) for r in train)],
                training_windows=sorted({r['window_id'] for r in train}),
                holdout_windows=sorted({r['window_id'] for r in test}),holdout=observed,
                timing_max_error=max(errors),power_mape=float(np.mean(power_errors)),power_max_error=max(power_errors))
    return dict(schema='distserve-independent-stage-surface-v1',identity=identity,cells=cells,
        qualified=all(c['status']=='qualified' for c in cells),raw_bindings=raw_bindings,
        measurement_qualification=measurement_qualification,training_sha256=digest(training),
        holdout_sha256=digest(holdout),training=training,holdout=holdout,formal_eligible=False,
        coverage='convex hull of measured request-count/token-sum/minimum/maximum lengths, plus per-request bounds',
        limitations=['PP1 runner CUDA timing; original simulator host overhead retained.',
                     'No cross-system data, frequency interpolation, PP extrapolation, or uncovered batch fallback.'])


class StageSurface:
    def __init__(self, artifact, *, frequency=2520, require_qualified=True):
        self.artifact, self.identity, self.frequency = artifact,artifact['identity'],frequency
        if artifact.get('schema')!='distserve-independent-stage-surface-v1' or self.identity.get('system')!='distserve':
            raise ValueError('independent DistServe surface required')
        if require_qualified and not artifact.get('qualified'):
            raise ValueError('missing_profile: DistServe stage calibration has not passed')
        self.cells={c['role']:c for c in artifact['cells'] if c['frequency_mhz']==frequency}
        self.hulls={role:_Hull(cell['vertices']) for role,cell in self.cells.items() if cell.get('vertices')}
        self.cache={}

    def stage_latency(self, role, tp, pp, stage, batch, inputs, contexts):
        if pp!=1 or stage!=0 or tp!=self.identity['tp']:
            raise ValueError('unsupported_engine: stage surface topology differs')
        lengths=list(inputs) if role=='prefill' else [n+1 for n in contexts]
        if len(lengths)!=batch:raise ValueError('simulator batch/vector length differs')
        key=(role,tuple(sorted(lengths)))
        if key in self.cache:return self.cache[key]
        cell=self.cells.get(role,{})
        if (cell.get('status')!='qualified' or not cell['length_bounds'][0]<=min(lengths)<=max(lengths)<=cell['length_bounds'][1]
                or not self.hulls[role].contains(coverage_features(lengths))):
            raise ValueError(f'missing_profile: DistServe {role} TP{tp} B{batch} lengths={lengths}')
        value=_predict(cell['latency_coefficients'],lengths)
        if not math.isfinite(value) or value<=0:raise ValueError('invalid independent stage prediction')
        self.cache[key]=value
        return value

    @classmethod
    def load(cls, path, **kwargs):
        artifact=json.loads(Path(path).read_text())
        audit_surface(artifact,base_dir=Path(path).parent)
        return cls(artifact, **kwargs)


def audit_surface(artifact, *, base_dir=None):
    if not artifact.get('raw_bindings'):
        raise ValueError('independent raw sample bindings are absent')
    from .stage_collect import rows_from_window
    def resolve(path):
        value=Path(path)
        if value.is_absolute():return value
        if base_dir is None:raise ValueError('relative profile artifact requires its owning directory')
        return Path(base_dir)/value
    training,holdout=[],[]
    windows=set()
    for binding in artifact['raw_bindings']:
        path=resolve(binding['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:
            raise ValueError('independent raw sample checksum differs')
        raw=json.loads(path.read_text())
        if raw.get('status')=='unsupported_engine':continue
        if raw.get('window_id') in windows:
            raise ValueError('independent raw measurement window was reused')
        windows.add(raw.get('window_id'))
        cap=raw.get('capability',{})
        if any(cap.get(key)!=artifact['identity'].get(key) for key in
               ('model_id','model_hash','tokenizer_hash','engine_revision','image_digest','source_revision','tp','pp')):
            raise ValueError('raw sample and fitted execution identity differ')
        for row in rows_from_window(raw):
            (training if row['purpose']=='training' else holdout).append(row)
    if training!=artifact['training'] or holdout!=artifact['holdout']:
        raise ValueError('fitted training/holdout rows differ from bound raw measurements')
    qualification=artifact['measurement_qualification']
    path=qualification.get('receipt_path')
    if not path or hashlib.sha256(resolve(path).read_bytes()).hexdigest()!=qualification.get('receipt_sha256'):
        raise ValueError('measurement qualification receipt binding differs')
    receipt=json.loads(resolve(path).read_text())
    if receipt.get('passed') is not True or qualification.get('passed') is not True:
        raise ValueError('independent measurement qualification has not passed')
    external=resolve(receipt['external_interference_path'])
    if hashlib.sha256(external.read_bytes()).hexdigest()!=receipt['external_interference_sha256']:
        raise ValueError('external interference evidence checksum differs')
    rebuilt=fit_surface(artifact['training'],artifact['holdout'],identity=artifact['identity'],
        raw_bindings=artifact['raw_bindings'],measurement_qualification=qualification)
    if rebuilt!=artifact:raise ValueError('stage surface differs from independent training/holdout reconstruction')
    return True
