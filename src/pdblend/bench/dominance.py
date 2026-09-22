"""Strict identities and paired gates for isolated calibration experiments."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

BASELINES = ('mixed', 'distserve_static', 'dynamollm', 'ecoserve')
SINGLE_SEED = 701
SEEDS = (SINGLE_SEED,)
SEED_POLICY = "single_seed_701"
PAIR_FIELDS = ('dataset', 'rate', 'seed', 'duration', 'model', 'tp', 'gpus',
               'trace_sha256', 'corpus_sha256', 'image', 'hardware', 'clock_protocol', 'energy_protocol')
METRICS = ('j_per_token', 'mean_power_w', 'window_mean_power_w', 'joint_slo_rate',
           'ttft_p90', 'ttft_p99', 'tpot_p99', 'plans', 'wakes', 'parks')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def trace_digest(trace):
    h = hashlib.sha256()
    for r in trace:
        h.update(json.dumps([r.idx, r.arrival_s, r.prompt, r.max_tokens], separators=(',', ':')).encode())
        h.update(b'\n')
    return h.hexdigest()


def metrics(summary):
    events = summary.get('controller', {}).get('events', {})
    return {**{k: summary.get(k) for k in METRICS[:3]},
            **{k: summary.get('slo', {}).get(k) for k in METRICS[3:7]},
            'plans': events.get('plan', 0), 'wakes': events.get('wake', 0), 'parks': events.get('park', 0)}


def load_evidence(directory, expected=None):
    """A directory/summary alone never proves completion or source identity."""
    directory = Path(directory)
    try:
        evidence = json.loads((directory / 'evidence.json').read_text())
        ident = evidence['identity']
        if evidence.get('status') != 'complete' or evidence.get('returncode') != 0:
            return None
        if not evidence.get('inputs_unchanged') or evidence.get('identity_sha256') != digest(ident):
            return None
        if expected is not None and any(ident.get(k) != v for k, v in expected.items()):
            return None
        required_groups = (('summary.json',), ('outcomes.jsonl', 'outcomes.jsonl.gz'),
                           ('power.jsonl', 'power.jsonl.gz'),
                           ('controller.jsonl', 'controller.jsonl.gz'),
                           ('freq.jsonl', 'frequency.jsonl'))
        if any(not any(name in evidence['artifacts'] for name in group) for group in required_groups):
            return None
        if any(sha(directory / name) != value for name, value in evidence['artifacts'].items()):
            return None
        summary = json.loads((directory / 'summary.json').read_text())
        if summary['trace_meta']['seed'] != ident['seed'] or summary['policy']['name'] != ident['policy']:
            return None
        if summary['requests'] != ident['requests'] or summary['slo']['offered'] != ident['requests']:
            return None
        # The frozen protocol meters through the last scheduled arrival and tail.
        # Check that exact window, not an arbitrary 295-second directory heuristic.
        if abs(summary['window_s'] - ident['trace_window_s']) > .1 or ident['duration'] != 300:
            return None
        return dict(identity=ident, metrics=metrics(summary), summary=summary, directory=str(directory))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def compare_seed(candidate, baselines, previous=None):
    reasons = []
    if candidate is None:
        return dict(status='inconclusive', reasons=['candidate_missing_or_stale'])
    c, identity = candidate['metrics'], candidate['identity']
    if any(c.get(k) is None or not math.isfinite(c[k]) for k in METRICS):
        return dict(status='inconclusive', reasons=['invalid_metrics'])
    if c['joint_slo_rate'] < .9:
        reasons.append('candidate_slo')
    pairs = {}
    missing = False
    for policy in BASELINES:
        b = baselines.get(policy)
        if b is None:
            missing = True
            reasons.append(policy + ':missing_or_stale')
            continue
        mismatches = [k for k in PAIR_FIELDS if identity.get(k) is None or identity.get(k) != b['identity'].get(k)]
        if mismatches:
            missing = True
            reasons.append(policy + ':mismatch:' + ','.join(mismatches))
            continue
        bm = b['metrics']
        if bm.get('j_per_token') is None or not math.isfinite(bm['j_per_token']):
            missing = True
            reasons.append(policy + ':invalid_energy')
            continue
        pairs[policy] = dict(j_per_token=bm['j_per_token'], joint_slo_rate=bm['joint_slo_rate'],
                             energy_win=c['j_per_token'] < bm['j_per_token'])
        if not pairs[policy]['energy_win']:
            reasons.append(policy + ':energy_loss')
    # A measured baseline that fails SLO is still shown (and its energy must be
    # beaten for the user's stricter all-baseline claim); it is never called SLO-valid.
    if previous is None:
        missing = True
        reasons.append('previous_pdblend:missing_tail_reference')
    elif any(identity.get(k) != previous['identity'].get(k) for k in PAIR_FIELDS):
        missing = True
        reasons.append('previous_pdblend:mismatched_tail_reference')
    else:
        for key in ('ttft_p90', 'ttft_p99', 'tpot_p99'):
            if c[key] > previous['metrics'][key]:
                reasons.append('tail_regression:' + key)
    return dict(status='inconclusive' if missing else ('failed' if reasons else 'screen_pass'),
                reasons=reasons, pairs=pairs, metrics=c, seed=identity['seed'])


def aggregate(rows):
    seeds = [r.get('seed') for r in rows]
    complete = len(seeds) == len(SEEDS) and set(seeds) == set(SEEDS)
    stats = {}
    for key in METRICS:
        values = [r.get('metrics', {}).get(key) for r in rows]
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        stats[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                          std=statistics.stdev(values) if len(values) > 1 else None,
                          worst=(min(values) if key == 'joint_slo_rate' else max(values)) if values else None)
    return dict(status='win' if complete and all(r['status'] == 'screen_pass' for r in rows) else 'unproven',
                complete=complete, seeds=seeds, single_seed=True,
                seed_policy=SEED_POLICY, metrics=stats)
