"""Profile evidence validation and experiment gates; no changes to planner models."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from pdblend.profile.calibration.merge import sha256
from pdblend.profile.collection.profiler import DECODE_BATCHES, DECODE_CONTEXTS, DECODE_STEPS, PREFILL_INPUTS, MIXED_PROBES
from pdblend.profile.identity import require_profile_provenance
from pdblend.seed_config import SEEDS, SEED_POLICY, seed_metadata

FREQS = (900, 1200, 1500, 1800, 2100, 2520)
def validate_parallel_interference(raw, directory: Path | None = None) -> dict:
    """Validate the representative isolated/concurrent interference receipt.

    Profiles without parallel mode metadata remain diagnostic-compatible.  A
    profile claiming parallel qualification must carry a completed receipt,
    matching representative points, bounded timing and power deltas, and (when
    a sample path is supplied) an immutable checksum.
    """
    external = raw.get('external_interference')
    local = raw.get('parallel_interference')
    receipt = external or local
    receipt_kind = 'external' if external is not None else 'local'
    mode = raw.get('measured_mode') or (receipt or {}).get('measured_mode')
    if mode is None and receipt is None:
        return dict(passed=True, formal_eligible=True, skipped=True, failures=[])
    failures = []
    formal_eligible = False
    if not isinstance(receipt, dict):
        failures.append('parallel interference receipt is missing')
        return dict(passed=False, formal_eligible=False, skipped=False, failures=failures)
    # ProfileWave stores the full cohort comparison in its checksummed sample
    # and leaves only a compact pointer in raw.json.  Validate the canonical
    # file contents as the receipt payload while retaining the pointer fields.
    if receipt_kind == 'external' and directory is not None and receipt.get('samples_file'):
        evidence_path = directory / receipt['samples_file']
        if evidence_path.is_file() and receipt.get('samples_sha256') == sha256(evidence_path):
            try:
                evidence_payload = json.loads(evidence_path.read_text())
                if isinstance(evidence_payload, dict):
                    receipt = {**receipt, **evidence_payload}
            except (OSError, json.JSONDecodeError):
                pass
    if mode == 'serial_cohort':
        if not receipt.get('complete'):
            failures.append('serial_cohort receipt is incomplete')
        return dict(passed=not failures, formal_eligible=False, skipped=False, failures=failures)
    if mode == 'serial_fallback':
        if not receipt.get('error') and receipt.get('complete') is not False:
            failures.append('serial fallback lacks failure reason')
        return dict(passed=not failures, formal_eligible=False, skipped=False, failures=failures)
    if mode != 'parallel':
        failures.append(f'unknown measured_mode: {mode!r}')
    if receipt.get('complete') is not True:
        failures.append('parallel interference receipt is incomplete')
    point = receipt.get('point', {})
    if not point and isinstance(receipt.get('parallel'), list) and receipt['parallel']:
        first_member = receipt['parallel'][0]
        if isinstance(first_member, dict):
            point = first_member.get('point', {})
    frequency = point.get('freq_mhz', point.get('frequency'))
    context = point.get('context_tokens', point.get('context'))
    if frequency != 2100 or point.get('batch') != 8 or context != 1024:
        failures.append('parallel interference point must be 2100/B8/context1024')
    isolated, parallel = receipt.get('isolated'), receipt.get('parallel')
    def flatten(samples):
        # ProfileWave records one member object per job, each with an
        # ``instances`` list. Local probes already provide flat rows.
        if (isinstance(samples, list) and samples and
                all(isinstance(item, dict) and isinstance(item.get('instances'), list)
                    for item in samples)):
            return [instance for member in samples for instance in member['instances']]
        return samples
    isolated, parallel = flatten(isolated), flatten(parallel)
    if not isinstance(isolated, list) or not isinstance(parallel, list) or not isolated or len(isolated) != len(parallel):
        failures.append('parallel interference isolated/concurrent samples do not match')
    else:
        for index, (base, together) in enumerate(zip(isolated, parallel)):
            try:
                timing = relative_error(together['step_seconds'], base['step_seconds'])
                power = relative_error(together['power_w'], base['power_w'])
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f'interference sample {index} invalid: {exc}')
                continue
            if timing > .05 or power > .05:
                failures.append(f'interference sample {index} exceeds 5%: timing={timing:g}, power={power:g}')
        validation = receipt.get('validation', {})
        comparisons_ok = all(item.get('passed') is True for item in receipt.get('comparisons', ()))
        if (validation.get('passed') is not True and receipt.get('passed') is not True
                and not comparisons_ok):
            failures.append('parallel interference validation did not pass')
    sample = receipt.get('samples_file')
    digest = receipt.get('samples_sha256')
    evidence = None
    if sample or digest:
        if directory is None or not sample or not digest or not (directory / sample).is_file():
            failures.append('parallel interference evidence file is missing')
        elif sha256(directory / sample) != digest:
            failures.append('parallel interference evidence checksum mismatch')
        else:
            try:
                evidence = json.loads((directory / sample).read_text())
            except (OSError, json.JSONDecodeError):
                failures.append('parallel interference evidence is not valid JSON')
    full_host_failures = []
    full_host = raw.get('concurrency_environment')
    if receipt_kind == 'local' and mode == 'parallel' and isinstance(full_host, dict):
        inventory = full_host.get('physical_gpu_uuids') or full_host.get('all_gpu_uuids')
        if inventory is None and isinstance(full_host.get('inventory'), list):
            inventory = [item.get('uuid') if isinstance(item, dict) else item
                         for item in full_host['inventory']]
        allocated = full_host.get('allocated_gpu_uuids')
        peer_snapshots = full_host.get('peer_snapshots')
        peers = full_host.get('peer_jobs')
        if peers is None and isinstance(peer_snapshots, list):
            peers = [peer for snapshot in peer_snapshots if isinstance(snapshot, dict)
                     for peer in snapshot.get('peers', [])]
        if not isinstance(inventory, list) or len(inventory) != 8 or len(set(inventory)) != 8:
            full_host_failures.append('full-host receipt requires eight unique physical GPU UUIDs')
        if not isinstance(allocated, list) or set(allocated or ()) != set(inventory or ()):
            full_host_failures.append('allocated UUIDs must equal the complete physical inventory')
        if peers != []:
            full_host_failures.append('full-host receipt must explicitly contain no peer jobs')
        manifest = full_host.get('lease_manifest_file') or full_host.get('samples_file')
        manifest_sha = full_host.get('lease_manifest_sha256') or full_host.get('samples_sha256')
        if (directory is None or not manifest or not manifest_sha or
                not (directory / manifest).is_file() or
                sha256(directory / manifest) != manifest_sha):
            full_host_failures.append('full-host lease manifest checksum binding is missing or invalid')
        if not full_host_failures:
            formal_eligible = True
    else:
        formal_eligible = receipt_kind == 'external' and mode == 'parallel'
    failures.extend(full_host_failures)
    if formal_eligible and receipt_kind == 'external':
        if not isinstance(evidence, dict) or evidence.get('cross_job') is not True:
            failures.append('external interference evidence lacks cross-job coordinator marker')
        if not evidence or not evidence.get('cohort_id') or len(evidence.get('members', ())) < 2:
            failures.append('external interference evidence lacks a multi-member cohort')
        if not evidence or evidence.get('overlapping_windows') is not True:
            failures.append('external interference evidence lacks overlapping windows')
        if isinstance(evidence, dict):
            from pdblend.profile.collection.parallel import common_window_overlap
            measured = flatten(evidence.get('parallel'))
            if not isinstance(measured, list) or not common_window_overlap(measured)['passed']:
                failures.append('external interference lacks common windows across all instances')
    return dict(passed=not failures, formal_eligible=formal_eligible and not failures,
                skipped=False, failures=failures)


def validate_parallel_layout(raw) -> dict:
    """Validate physical ownership and concurrency annotations in profile raw data.

    Older synthetic fixtures do not have these fields and are accepted by
    returning a skipped result.  Profiles emitted by :class:`Profiler` carry
    the manifest and are rejected when a row could have been collected under a
    different layout or concurrency level.
    """
    layout = raw.get('parallel_layout')
    concurrency = raw.get('concurrency')
    if layout is None and concurrency is None:
        return dict(passed=True, skipped=True, failures=[])
    failures = []
    if not isinstance(layout, dict):
        failures.append('parallel_layout must be an object')
    else:
        instances = layout.get('instances')
        gpus = layout.get('gpus')
        if not isinstance(instances, list) or not instances:
            failures.append('parallel_layout.instances is empty')
        if not isinstance(gpus, list) or not gpus:
            failures.append('parallel_layout.gpus is empty')
        elif len(gpus) != len(set(gpus)):
            failures.append('parallel_layout reuses a physical GPU')
        if isinstance(instances, list):
            flattened = []
            for instance in instances:
                if not isinstance(instance, dict):
                    failures.append('parallel_layout instance is not an object')
                    continue
                igpus = instance.get('gpus', [])
                tp, pp = instance.get('tp'), instance.get('pp', 1)
                valid_shape = (isinstance(igpus, list) and isinstance(tp, int) and isinstance(pp, int)
                               and len(igpus) == tp * pp)
                if not valid_shape:
                    failures.append('parallel_layout instance GPU count does not equal TP*PP')
                flattened.extend(igpus if isinstance(igpus, list) else [])
            if isinstance(gpus, list) and flattened != gpus:
                failures.append('parallel_layout.gpus does not match instance ownership')
    if not isinstance(concurrency, dict):
        failures.append('concurrency must be an object')
    else:
        for key in ('decode_max_batch', 'mixed_background_max_batch', 'prefill_inflight', 'transfer_inflight'):
            value = concurrency.get(key)
            if not isinstance(value, int) or value < 1:
                failures.append(f'concurrency.{key} must be a positive integer')

    def check_rows(section, expected):
        for index, row in enumerate(raw.get(section, [])):
            if 'concurrency' not in row or row.get('concurrency') != expected(row):
                failures.append(f'{section}[{index}] concurrency does not match workload')
            if layout is not None and row.get('parallel_layout') != layout:
                failures.append(f'{section}[{index}] parallel_layout mismatch')

    check_rows('prefill', lambda row: 1)
    check_rows('decode', lambda row: row.get('batch'))
    check_rows('mixed', lambda row: row.get('batch'))
    check_rows('transfer', lambda row: 1)
    return dict(passed=not failures, skipped=False, failures=failures)


def relative_error(predicted, observed):
    if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in (predicted, observed)) or observed <= 0:
        raise ValueError('relative error requires finite values and positive observed value')
    return abs(predicted - observed) / observed


def quality_audit(raw, model, directory: Path):
    failures, points = [], []
    def fail(metric, **details):
        failures.append(dict(metric=metric, **details))
    env = raw.get('environment', {})
    for key in ('image_digest', 'source_hash', 'vllm', 'torch', 'cuda', 'timestamp_utc', 'gpu_uuids'):
        if not env.get(key):
            fail('environment', missing=key)
    if not isinstance(env.get('gpu_uuids'), list) or any(not x.startswith('GPU-') for x in env.get('gpu_uuids', [])):
        fail('environment', reason='invalid_gpu_uuids')
    if raw.get('schema') != 2:
        fail('schema', observed=raw.get('schema'))
    # Legacy synthetic/raw profiles remain loadable for diagnostics, but every
    # new profile carrying an identity must pass the independent provenance and
    # sample/holdout binding gate before it can be accepted for a campaign.
    if raw.get('profile_key') is not None:
        try:
            require_profile_provenance(raw)
        except (TypeError, ValueError) as exc:
            fail('profile_provenance', reason=str(exc))
    layout_check = validate_parallel_layout(raw)
    if not layout_check['passed']:
        fail('parallel_layout', failures=layout_check['failures'])
    interference_check = validate_parallel_interference(raw, directory)
    if not interference_check['passed']:
        fail('parallel_interference', failures=interference_check['failures'])
    elif (raw.get('parallel_interference') is not None or raw.get('external_interference') is not None) \
            and not interference_check.get('formal_eligible', False):
        # Local-fleet evidence remains useful for diagnostics, but quality
        # audit must never turn it into formal profile eligibility.
        fail('parallel_interference_formal', reason='cross-job coordinator evidence required')
    cfg = raw.get('config', {})
    for key, minimum in (('decode_repeats', 3), ('decode_settle_s', 2), ('decode_measure_s', 5)):
        if cfg.get(key, 0) < minimum:
            fail('config', field=key, minimum=minimum)
    if set(raw.get('freqs', [])) != set(FREQS) or set(model.freqs) != set(FREQS):
        fail('frequency_coverage', required=list(FREQS))
    cap = raw.get('kv_capacity_tokens', 0)
    if cap <= 0 or raw.get('kv_bytes_per_token', 0) <= 0:
        fail('kv', reason='missing_positive_capacity')
    expected = {
        'prefill': {(f, n) for f in FREQS for n in PREFILL_INPUTS},
        'decode': {(f, c, b) for f in FREQS for c in DECODE_CONTEXTS for b in DECODE_BATCHES
                   if not cap or b * (c + DECODE_STEPS) <= .9 * cap},
        'mixed': {(f, b, c) for f in (1500, 2100, 2520) for b, c in MIXED_PROBES},
        'transfer': {(n,) for n in (512, 2048, 7168)},
    }
    fields = {'prefill': ('freq_mhz', 'input_tokens'), 'decode': ('freq_mhz', 'context_tokens', 'batch'),
              'mixed': ('freq_mhz', 'batch', 'chunk_tokens'), 'transfer': ('input_tokens',)}
    for section, required in expected.items():
        keys = [tuple(x.get(k) for k in fields[section]) for x in raw.get(section, [])]
        missing = required - set(keys)
        if missing or len(keys) != len(set(keys)):
            fail(section + '_coverage', missing=sorted(missing), duplicate_count=len(keys) - len(set(keys)))
    required_static = {f'active_idle@{f}' for f in FREQS} | {'parked', 'off', 'active_idle_reset'}
    if not required_static <= set(raw.get('static', {})):
        fail('static_coverage', missing=sorted(required_static - set(raw.get('static', {}))))

    def check_file(row, point):
        name = row.get('samples_file')
        if not name or not (directory / name).is_file():
            fail('sample_evidence', point=point, file=name)
            return None
        path = directory / name
        if not row.get('samples_sha256') or sha256(path) != row['samples_sha256']:
            fail('sample_checksum', point=point, file=name)
        return json.loads(path.read_text())

    for shard in raw.get('shards', []):
        p = directory / shard['path']
        if not p.is_file() or sha256(p) != shard['sha256']:
            fail('shard_checksum', path=str(p))
    for d in raw.get('decode', []):
        point = {k: d[k] for k in fields['decode']}
        repeats = d.get('repeats', [])
        if len(repeats) < max(3, cfg.get('decode_repeats', 3)):
            fail('decode_repeats', point=point, actual=len(repeats))
        for rep in repeats:
            data = check_file(rep, point)
            if rep.get('steady_window_s', 0) < 5 or rep.get('end_s', 0) - rep.get('start_s', 0) < 5:
                fail('steady_window', point=point)
            if rep.get('min_steps', 0) < 8 or rep.get('power_samples', 0) < 2 or rep.get('frequency_samples', 0) < 1:
                fail('decode_evidence', point=point)
            if data is not None:
                if len(data.get('power', [])) != rep.get('power_samples') or len(data.get('frequency', [])) != rep.get('frequency_samples'):
                    fail('sample_counts', point=point)
        powers = d.get('power_repeats', [])
        if len(powers) != len(repeats) or len(d.get('step_repeats', [])) != len(repeats):
            fail('repeat_summary', point=point)

    # Recompute residuals from model predictions and measured points, not stored quality fields.
    metrics = {}
    for section, timing in (('prefill', 'prefill_time'), ('decode', 'decode_time')):
        for r in raw.get(section, []):
            f = r['freq_mhz']
            observed = r['seconds'] if section == 'prefill' else r['step_seconds']
            pred = model.prefill_seconds(r['input_tokens'], f) if section == 'prefill' else model.step_seconds(
                r['batch'], r.get('effective_context_tokens', r['context_tokens']), f)
            err = relative_error(pred, observed)
            point = dict(metric=f'{timing}@{f}', **{k: r[k] for k in fields[section]},
                         predicted=pred, observed=observed, relative_error=err)
            points.append(point)
            metrics.setdefault(point['metric'], []).append(err)
            if err > .10:
                fail(point['metric'], reason='max_relative_error', point=point, limit=.10)
    for f in FREQS:
        stable = []
        for d in raw.get('decode', []):
            if d['freq_mhz'] != f or (d['batch'] < 4 and not model.decode_power_overrides):
                continue
            values = d.get('power_repeats', [])
            cv = statistics.stdev(values) / statistics.mean(values) if len(values) >= 3 and min(values) > 0 else float('inf')
            if cv <= .10:
                stable.append(d)
                observations = d['repeats'] if model.decode_power_overrides else [d]
                for index, obs in enumerate(observations):
                    context = obs.get('effective_context_tokens', d.get('effective_context_tokens'))
                    point = dict(metric=f'decode_power@{f}', freq_mhz=f, batch=d['batch'],
                                 context_tokens=context, repeat=index, observed=obs['power_w'])
                    try:
                        prediction = model.decode_power_w(d['batch'], f, ctx=context)
                    except ValueError as exc:
                        fail(f'decode_power@{f}', reason='outside_coverage', point=point, error=str(exc))
                        continue
                    err = relative_error(prediction, obs['power_w'])
                    metrics.setdefault(f'decode_power@{f}', []).append(err)
                    points.append(dict(point, relative_error=err, predicted=prediction))
            elif model.decode_power_overrides:
                fail(f'decode_power@{f}', reason='unstable_power_repeats', batch=d['batch'], cv=cv)
        if len({d['batch'] for d in stable}) < 2:
            fail(f'decode_power@{f}', reason='insufficient_stable_batches')
    quality = {k: dict(mape=statistics.mean(v), max=max(v), samples=len(v)) for k, v in metrics.items()}
    for k, q in quality.items():
        if k.startswith('decode_power@') and (q['mape'] > .10 or q['max'] > .15):
            fail(k, reason='planner_power_error', value=q,
                 points=[p for p in points if p['metric'] == k and p['relative_error'] > .15])
    bases = {(r['freq_mhz'], r['batch'], r['context_tokens']): r for r in raw.get('decode', [])}
    prefills = {(r['freq_mhz'], r['input_tokens']): r for r in raw.get('prefill', [])}
    mixed_errors = []
    mixed_rows = []
    for row in raw.get('mixed', []):
        base = bases.get((row['freq_mhz'], row['batch'], row['context_tokens']))
        alone = prefills.get((row['freq_mhz'], row['chunk_tokens']))
        if not row.get('valid') or not base or not alone or not row.get('base_step_s') or row['base_step_s'] <= 0:
            fail('mixed_base_or_validity', point=row)
            continue
        if not math.isclose(row['base_step_s'], base['step_seconds'], rel_tol=1e-9):
            fail('mixed_base_mismatch', point=row)
        data = check_file(row, {k: row[k] for k in fields['mixed']})
        probes = row.get('probe_ttft_samples', [])
        if len(probes) < 3 or row.get('stable_tokens', 0) < 16:
            fail('mixed_stability', point=row)
        if data:
            start = row.get('stable_start_s', 0)
            backgrounds = data.get('background', [])
            if len(backgrounds) != row['batch'] or any(sum(t <= start for t in stream) < 17 for stream in backgrounds):
                fail('mixed_barrier_evidence', point={k: row[k] for k in fields['mixed']})
            if any(p['start_s'] < start for p in data.get('probes', [])):
                fail('mixed_probe_before_barrier', point=row)
        pred = base['step_seconds'] + alone['seconds']
        error = relative_error(pred, row['probe_ttft_s'])
        mixed_errors.append(error)
        mixed_rows.append(dict(freq_mhz=row['freq_mhz'], batch=row['batch'], chunk_tokens=row['chunk_tokens'],
                               predicted=pred, observed=row['probe_ttft_s'], relative_error=error,
                               probe_cv=statistics.stdev(probes) / statistics.mean(probes) if len(probes) >= 2 else None))
    mixed = dict(valid=len(mixed_errors), invalid=len(raw.get('mixed', [])) - len(mixed_errors), rows=mixed_rows,
                 missing_base=sum(not x.get('base_step_s') for x in raw.get('mixed', [])))
    if mixed_errors:
        mixed.update(median=statistics.median(mixed_errors), mape=statistics.mean(mixed_errors), max=max(mixed_errors))
        if mixed['median'] > .15:
            fail('mixed_additive', value=mixed['median'], limit=.15)
    else:
        fail('mixed_additive', reason='no_valid_points')
    return dict(passed=not failures, failures=failures, quality=quality, mixed=mixed, points=points,
                error_definition='abs(predicted-observed)/observed')


def m2_gate(rows):
    """Rows must be completed, provenance-checked results for active seeds."""
    reasons = []
    if len(rows) != len(SEEDS) or {r.get('seed') for r in rows} != set(SEEDS):
        reasons.append(f'requires exactly seeds {list(SEEDS)} under {SEED_POLICY}')
    for r in rows:
        checks = (r.get('complete') is True, r.get('joint_slo_rate', -1) >= .9,
                  r.get('ttft_p99', float('inf')) <= 5, r.get('tpot_p99', float('inf')) <= .15,
                  r.get('power_error') is not None and abs(r['power_error']) <= .05)
        if not all(checks):
            reasons.append(f'seed {r.get("seed")} failed completion/SLO/tail/power checks')
    return dict(passed=not reasons, reasons=reasons, min_m_instances=2 if not reasons else 4,
                **seed_metadata())
