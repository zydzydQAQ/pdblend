#!/usr/bin/env python3
"""Compose existing calibration only; no GPU, queue, evaluation fitting or retries."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.spatial import ConvexHull

from pdblend.profile.collection.native_timing_audit import measured_events
from pdblend.profile.query.development_composite import KIND, _bound
from pdblend.profile.query.optimization import attach_component
from pdblend.profile.query.versions import load_profile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/2026-09-24'
ATTEMPTS = ROOT / 'results/2026-09-22/three-model/queue-attempts'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return binding(path)


def match_identity(actual, expected):
    for key in ('system', 'model_id', 'tp', 'pp', 'model_hash', 'tokenizer_hash'):
        if actual.get(key) != expected[key]:
            raise ValueError('calibration identity differs: ' + key)


def fit_partial(rows):
    """Training-only nonnegative fits. Two prefill nodes fit affine, not quadratic."""
    models = []
    for frequency, role in sorted({(r['frequency_mhz'], r['role']) for r in rows if r['purpose'] == 'training'}):
        train = [r for r in rows if (r['frequency_mhz'], r['role'], r['purpose']) == (frequency, role, 'training')]
        hold = [r for r in rows if (r['frequency_mhz'], r['role'], r['purpose']) == (frequency, role, 'holdout')]
        def coords(r):
            return [r['batch'], r['context_tokens']/8192] if role == 'decode' else [r['prompt_tokens']/8192]
        def feature(r):
            if role == 'decode':
                b, c = coords(r)
                return [1., b, b*c]
            x, = coords(r)
            return [1., x, x*x]
        vertices = np.unique(np.asarray([coords(r) for r in train]), axis=0)
        features = np.asarray([feature(r) for r in train])
        rank = np.linalg.matrix_rank(features)
        if role == 'decode' and (rank < 3 or len(vertices) < 3):
            continue
        if role == 'prefill' and len(vertices) < 2:
            continue
        target = np.asarray([r['latency_ms'] for r in train])
        coefficients, _ = nnls(features[:, :rank], target)
        coefficients = np.pad(coefficients, (0, 3-rank))
        hull = ConvexHull(vertices) if role == 'decode' else None
        errors = [abs(float(np.dot(feature(r), coefficients))-r['latency_ms'])/r['latency_ms'] for r in hold]
        models.append(dict(frequency_mhz=frequency, role=role, coefficients=coefficients.tolist(),
            coverage_vertices=(vertices[hull.vertices] if hull is not None else vertices).tolist(),
            estimator='nonnegative_training_only_' + ('affine' if rank == 2 else 'three_parameter'),
            training_events=len(train), training_windows=len({r['window_id'] for r in train}),
            holdout_events=len(hold), holdout_windows=len({r['window_id'] for r in hold}),
            holdout_diagnostic_max_relative_error=max(errors) if errors else None,
            holdout_used_for_fitting=False, qualified=False))
    return models


def partial_timing(size, identity):
    inventory_path = RESULTS / f'pdblend-single-pass-partial-v1/{size}/inventory.json'
    inventory = read(inventory_path)
    rows, raw_bindings, observed_capacity, clocks = [], [], [], []
    accepted_frequency_variations = []
    for record in inventory['points']:
        frequency_variation = (record['status'] == 'invalid'
            and record.get('error') == 'ValueError: observed frequency coverage differs')
        if record['status'] != 'measured' and not frequency_variation:
            continue
        raw_path = _bound(record['raw'])
        raw = read(raw_path)
        match_identity(dict(raw['capability'], system=raw['system']), identity)
        point = record['point']
        if (point['purpose'] not in ('training', 'holdout')
                or any(raw['point'].get(k) != v for k, v in point.items())):
            raise ValueError('unknown split or changed native timing point')
        clients = raw['client_requests']
        if (not raw['drain'].get('drained') or raw.get('cleanup_errors')
                or not clients or any(not r.get('terminal') or r.get('error')
                    or r['completion_tokens'] != point['output_tokens'] for r in clients)):
            raise ValueError('incomplete request/drain timing evidence')
        events = [r for r in measured_events(raw['sample'], tp=identity['tp'])
            if raw['start_s'] <= r['at_s'] < raw['end_s'] and r['role'] == point['role']
            and r['batch'] == point['batch'] and r['prompt_tokens'] == point['prompt_tokens']]
        if not events:
            raise ValueError('no actual CUDA training shape')
        rows.extend(dict(r, frequency_mhz=point['frequency_mhz'], purpose=point['purpose'],
                         window_id=raw['window_id']) for r in events)
        raw_bindings.append(record['raw'])
        if frequency_variation:
            accepted_frequency_variations.append(record['raw'])
        observed_capacity.append(raw['capability']['state'])
        observed = [f for t, freqs in raw['frequency_samples'] if raw['start_s'] <= t <= raw['end_s'] for f in freqs]
        clocks.append(dict(raw=record['raw'], requested_frequency_mhz=point['frequency_mhz'],
                           observed_min_mhz=min(observed), observed_max_mhz=max(observed)))
    capacity = dict(actual_total_kv_tokens=min(s['total_kv_tokens'] for s in observed_capacity),
        block_size=max(s['block_size'] for s in observed_capacity),
        max_num_seqs=min(s['max_num_seqs'] for s in observed_capacity),
        policy='preserve measured 90 percent block reservation capacity guard')
    return fit_partial(rows), [binding(inventory_path)], dict(raw_bindings=raw_bindings,
        clock_observations=clocks, accepted_windows=len(raw_bindings),
        accepted_frequency_variation_windows=accepted_frequency_variations,
        insufficient_rank_domains_inherit_base=True,
        splits=dict(Counter(r['purpose'] for r in rows)),
        excluded_inventory_statuses=dict(Counter(r['status'] for r in inventory['points']
            if r['status'] not in ('measured', 'invalid')))), capacity


def build_model(size, point, output):
    identity = dict(system='pdblend', model_id=point['model_id'], tp=2 if size == '32b' else 1, pp=1,
        **{k: point['engine_identity'][k] for k in ('model_hash', 'tokenizer_hash', 'image_digest',
              'runtime_source_sha256', 'measurement_source_sha256')})
    base_ref = point['inputs']['profiles'][0]
    _bound(base_ref)
    source_refs = []
    if size == '32b':
        paths = list(ATTEMPTS.glob('pdblend-native-timing-32b-77286f41f7bf8c04/attempt-*/native-timing/timing-component.json'))
        if len(paths) != 1:
            raise ValueError('32B timing selection ambiguous')
        component_path = paths[0]
        component = read(component_path)['component']
        match_identity(component['identity'], identity)
        timing_models = deepcopy(component['models'])
        if not component['component_qualified']:
            raise ValueError('32B completed timing component changed')
        first_raw = read(_bound(component['raw_bindings'][0]))
        state = first_raw['capability']['state']
        capacity = dict(actual_total_kv_tokens=state['total_kv_tokens'], block_size=state['block_size'],
            max_num_seqs=state['max_num_seqs'], policy='preserve measured 90 percent block reservation capacity guard')
        timing_evidence = dict(component=binding(component_path), raw_bindings=component['raw_bindings'],
            original_holdout_passed=True, holdout_used_for_fitting=False)
        source_refs.append(binding(component_path))
    else:
        timing_models, refs, timing_evidence, capacity = partial_timing(size, identity)
        source_refs.extend(refs)
    power_nodes, panels = {}, []
    for frequency in (1500, 2520):
        paths = list(ATTEMPTS.glob(f'pdblend-energy-gaps-{size}-{frequency}-7bc52ec01720186b-after-quick-d4088aa3-after-quick-061f0a30/attempt-*/components/completion.json'))
        if len(paths) != 1:
            raise ValueError('optimization power panel selection ambiguous')
        root = paths[0].parent
        package = read(root/'package-manifest.json')
        panel_base = package['inputs']['base_candidate']
        panel_profile = load_profile(_bound(panel_base), system='pdblend', model_id=identity['model_id'],
            tp=identity['tp'], pp=1, usage='development')
        verified = attach_component(panel_profile, root)
        candidate = verified.model.candidate
        match_identity(candidate, identity)
        if power_nodes.keys() & candidate['nodes'].keys():
            raise ValueError('overlapping power panel domains')
        power_nodes.update(deepcopy(candidate['nodes']))
        panel = dict(candidate=binding(root/'candidate.json'), completion=binding(root/'completion.json'),
            package=binding(root/'package-manifest.json'), base_candidate=panel_base,
            base_binding_verified=True, raw_audit_reproduced=True,
            independent_holdout=deepcopy(verified.model.power_validation))
        panels.append(panel)
        source_refs.extend(panel[k] for k in ('candidate', 'completion', 'package', 'base_candidate'))
    handoff_manifest = read(RESULTS/'pdblend-handoff-recovery-v2/manifest.json')
    extractions = [r['extraction'] for r in handoff_manifest['raw_replays']
                   if r.get('extracted') and f'timing-{size}-' in r['extraction']['path']]
    if len(extractions) != 1:
        raise ValueError('handoff extraction selection ambiguous')
    extraction = read(_bound(extractions[0]))
    match_identity(extraction['candidate']['identity'], identity)
    if extraction.get('evaluation_used_for_selection') or extraction['candidate'].get('evaluation_used_for_selection'):
        raise ValueError('evaluation contamination in endpoint calibration')
    source_refs.append(extractions[0])
    summary = dict(native_timing_domains=[{k: v for k, v in m.items() if k != 'coefficients'} for m in timing_models],
        power_frequencies_mhz=[1500, 2520], power_families=sorted(power_nodes),
        power_base_bindings=[p['base_candidate'] for p in panels],
        handoff_domain=dict(input_tokens=[512, 7168], output_tokens=[16], batch=[1],
            requested_f_P_mhz=[2520], requested_f_D_mhz=[2520],
            estimator='interpolated training maximum endpoint first gap including first decode',
            actual_frequency_qualified=False, p99_risk_qualified=False,
            physical_copy_time=False, short_output_extrapolation=False),
        inherited_components=['timing outside new shape/frequency domains',
            'power outside new shape/frequency domains', 'prefill_power', 'static', 'wake', 'transfer', 'clock_transition'],
        capacity=capacity, holdout_used_for_fitting=False, evaluation_used_for_selection=False,
        formal_eligible=False)
    compiled = dict(schema='pdblend-development-compiled-components/v1', identity=identity,
        timing_models=timing_models, timing_evidence=timing_evidence, capacity=capacity,
        power_nodes=power_nodes, power_panels=panels, handoff_nodes=extraction['candidate']['nodes'],
        handoff_evidence=extractions[0], source_bindings=source_refs, summary=summary)
    compiled_ref = write(output/f'{size}-components.json', compiled)
    profile_ref = write(output/f'{size}-profile.json', dict(kind=KIND,
        system='pdblend', model_id=identity['model_id'], tp=identity['tp'], pp=1,
        base_profile=base_ref, compiled=compiled_ref, development_only=True))
    loaded = load_profile(profile_ref['path'], system='pdblend', model_id=identity['model_id'],
                          tp=identity['tp'], pp=1, usage='development')
    return dict(profile=profile_ref, compiled=compiled_ref, identity=identity,
                frequencies_mhz=list(loaded.model.freqs), profile_key=loaded.profile_key, summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    campaign_path = RESULTS/'resident-comparison-pd-observation-v5/campaign.json'
    campaign = read(campaign_path)
    manifest = dict(schema='pdblend-development-profile-composition/v1', models={},
        builder=binding(__file__), base_campaign=binding(campaign_path),
        holdout_used_for_fitting=False, evaluation_used_for_selection=False,
        new_gpu_measurements=False, formal_eligible=False)
    for size in ('7b', '14b', '32b'):
        point = next(p for p in campaign['points'] if p['system']=='pdblend'
                     and p['model_id'] == f'Qwen2.5-{size.upper()}-Instruct')
        manifest['models'][size] = build_model(size, point, output)
        print(size + ': development profile ready', flush=True)
    write(output/'manifest.json', manifest)
    print(str(output/'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
