"""Strict evidence checks for independent decode accuracy and benchmark reuse."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from pdblend.profile.calibration.merge import sha256
from pdblend.profile.calibration.decode_fit import errors


def audit_independent(candidate: Path, directory: Path, model):
    candidate, directory = Path(candidate).resolve(), Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    failures, rows, skipped, evidence = [], [], [], {}
    digest = sha256(candidate)
    if manifest.get('candidate_sha256') != digest or not manifest.get('candidate_unchanged'):
        failures.append('candidate hash changed or unconfirmed')
    if manifest.get('returncodes') != [0] * 8 or not manifest.get('finished_s'):
        failures.append('all eight workers must finish successfully')
    frozen = json.loads((candidate.parent / 'candidate-frozen.json').read_text())
    if frozen.get('profile_sha256') != digest or frozen.get('frozen_at_s', float('inf')) > manifest.get('started_s', 0):
        failures.append('candidate was not frozen before holdout collection')
    spec = model.decode_overrides.get(900, {})
    source = spec.get('source', {})
    if source.get('development_report') and Path(source['development_report']).resolve() == (candidate.parent / 'independent-report.json').resolve():
        failures.append('validation data reused for fitting')
    worker_paths = sorted(directory.glob('gpu-*/measurement/raw.json'))
    if len(worker_paths) != 8:
        failures.append('missing worker raw files')
    for path in worker_paths:
        worker = path.parent.parent.name
        raw = json.loads(path.read_text())
        plan = json.loads((path.parent.parent / 'plan.json').read_text())
        env, cfg = raw.get('environment', {}), raw.get('config', {})
        if (plan.get('candidate_sha256') != digest or
            not raw.get('validation', {}).get('complete') or
            env.get('source_hash') != manifest.get('source_hash') or
            env.get('image_digest') != manifest.get('image') or not env.get('gpu_uuids')):
            failures.append(f'{worker}: provenance/incomplete measurement')
        if any(cfg.get(k, 0) < v for k, v in [('decode_repeats', 3), ('decode_settle_s', 2), ('decode_measure_s', 5)]):
            failures.append(f'{worker}: insufficient measurement settings')
        expected = Counter((p['batch'], p['context_tokens']) for p in plan['points'])
        actual = Counter((p['batch'], p['context_tokens']) for p in raw.get('decode', []) + raw.get('skipped', []))
        if actual != expected:
            failures.append(f'{worker}: planned point coverage mismatch')
        evidence[str(path)] = sha256(path)
        for point in raw.get('skipped', []):
            if (point.get('reason') != 'full_generation_KV_capacity' or
                point['batch'] * (point['context_tokens'] + 512) <= .9 * raw['kv_capacity_tokens']):
                failures.append(f'{worker}: invalid capacity skip')
            skipped.append(dict(worker=worker, **point))
        for d in raw.get('decode', []):
            reps = d.get('repeats', [])
            if d.get('freq_mhz') != 900 or len(reps) < 3:
                failures.append(f'{worker}: wrong frequency/insufficient repeats')
            for rep in reps:
                name = rep.get('samples_file', '')
                sample_path = path.parent / name
                if not name or not sample_path.is_file():
                    failures.append(f'{worker}: missing sample {name}')
                    continue
                data = json.loads(sample_path.read_text())
                evidence[str(sample_path)] = sha256(sample_path)
                if (rep.get('steady_window_s', 0) < 5 or rep.get('end_s', 0)-rep.get('start_s', 0) < 5 or
                    rep.get('min_steps', 0) < 8 or rep.get('power_samples', 0) < 2 or
                    rep.get('frequency_samples', 0) < 1 or
                    len(data.get('power', [])) != rep.get('power_samples') or
                    len(data.get('frequency', [])) != rep.get('frequency_samples')):
                    failures.append(f'{worker}: incomplete stable evidence {name}')
            ctx = d['effective_context_tokens']
            observed = d['step_seconds']
            pred = model.step_seconds(d['batch'], ctx, 900)
            rows.append(dict(worker=worker, batch=d['batch'], context_tokens=d['context_tokens'],
                             effective_context_tokens=ctx, observed=observed, predicted=pred,
                             relative_error=abs(pred / observed - 1), underestimate=max(1-pred/observed, 0),
                             repeat_count=len(reps), repeats=d.get('step_repeats', []),
                             steady_window_s=d.get('steady_window_s'),
                             extrapolated=not model.decode_supported(d['batch'],ctx,900),
                             gpu_uuids=env['gpu_uuids'], source=str(path)))
    quality = errors([r['observed'] for r in rows], [r['predicted'] for r in rows])
    if not rows or quality['max'] > .10:
        failures.append('independent decode-time maximum error exceeds 10%')
    if any(r['extrapolated'] for r in rows):
        failures.append('independent points require extrapolation')
    regions = {}
    for axis in ('batch', 'context_tokens'):
        for value in sorted({r[axis] for r in rows}):
            subset = [r for r in rows if r[axis] == value]
            regions[f'{axis}={value}'] = errors([r['observed'] for r in subset], [r['predicted'] for r in subset])
    return dict(candidate=str(candidate), candidate_sha256=digest, manifest=manifest, samples=len(rows),
                skipped=skipped, errors=quality, rows=rows, evidence_sha256=evidence, regions=regions,
                failures=failures, gate=dict(status='FAIL' if failures else 'PASS', complete=not failures,
                max_error_le_10=bool(rows) and quality['max'] <= .10,
                has_unvalidated_capacity_points=bool(skipped)))


def verify_audit_binding(profile: Path):
    """Fail closed on stale or incomplete training/holdout evidence."""
    profile = Path(profile).resolve()
    digest = sha256(profile)
    audit = json.loads((profile.parent / 'audit.json').read_text())
    independent = json.loads((profile.parent / 'independent-report.json').read_text())
    if (audit.get('profile_sha256') != digest or not audit.get('basic_passed') or
        independent.get('candidate_sha256') != digest or independent.get('gate', {}).get('status') != 'PASS'):
        raise ValueError('stale or failed profile audit')
    for name, expected in independent.get('evidence_sha256', {}).items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'changed independent evidence: {name}')
    if not independent.get('evidence_sha256'):
        raise ValueError('independent sample checksums absent')
    return digest


def benchmark_evidence(folder, profile, seed, policy, layout, clocks, expected_source=None):
    """Validate a completed 300 s run; reconstruct that seed's exact Poisson trace."""
    import hashlib
    from dataclasses import asdict
    from pdblend.bench.client import load_split, poisson_trace
    from pdblend.bench.run import offline_forecast
    folder, profile = Path(folder), Path(profile).resolve()
    execution = json.loads((folder / 'execution.json').read_text())
    summary_path = folder / 'measurement/summary.json'
    summary = json.loads(summary_path.read_text())
    command = execution['command']
    def arg(name):
        return command[command.index(name)+1]
    required = {'--seed': str(seed), '--policy': policy, '--layout': layout,
                '--clocks': clocks, '--duration': '300', '--profile': str(profile)}
    if any(arg(k) != v for k, v in required.items()):
        raise ValueError('benchmark parameters do not match')
    if execution.get('returncode') != 0 or not execution.get('finished_s') or not execution.get('inputs_unchanged'):
        raise ValueError('benchmark execution incomplete')
    if execution['inputs'].get(str(profile)) != sha256(profile):
        raise ValueError('benchmark profile mismatch')
    for name, checksum in execution['inputs'].items():
        if sha256(Path(name)) != checksum:
            raise ValueError(f'benchmark input changed: {name}')
    snapshot = folder / 'source'
    actual_source = {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob('*.py')}
    if actual_source != execution['source'] or (expected_source is not None and actual_source != expected_source):
        raise ValueError('benchmark source mismatch')
    if summary['trace_meta']['seed'] != seed or summary['profile'] != str(profile) or summary['window_s'] < 298:
        raise ValueError('summary parameters/window mismatch')
    trace = poisson_trace(load_split(Path(arg('--corpus')), arg('--dataset'), arg('--split')),
                          float(arg('--rate')), 300, seed)
    outcomes = [json.loads(line) for line in (folder/'measurement/outcomes.jsonl').read_text().splitlines()]
    if len(outcomes) != len(trace) or len({r['idx'] for r in outcomes}) != len(trace):
        raise ValueError('incomplete request trace')
    indexed = {r['idx']: r for r in outcomes}
    for r in trace:
        o = indexed[r.idx]
        if o['arrival_s'] != r.arrival_s or o['input_tokens'] != r.input_tokens or o['max_tokens'] != r.max_tokens:
            raise ValueError('observed request trace mismatch')
    trace_digest = hashlib.sha256(json.dumps([asdict(r) for r in trace], sort_keys=True).encode()).hexdigest()
    return summary, offline_forecast(trace), dict(
        profile_sha256=sha256(profile), trace_sha256=trace_digest,
        source_sha256=hashlib.sha256(json.dumps(actual_source, sort_keys=True).encode()).hexdigest(),
        summary_sha256=sha256(summary_path), image=execution['image'], hardware=execution['hardware'])
