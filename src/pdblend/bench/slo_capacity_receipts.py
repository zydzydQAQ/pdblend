"""Read-only bridge from immutable native windows to SLO capacity evidence.

The ledger binds each expected point and receipt by path and SHA256. It selects
only calibration/tuning traces. Acceptance is consumed as recorded; this reader
does not rerun hardware audits or promote profile qualification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .capacity_workloads import validate_trace
from .resident_session import digest, engine_signature, file_sha
from .slo_capacity import CapacityConfig, CapacityTrial, capacity_state, freeze_evaluation_grid, trial_verdict


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _verified_path(ref):
    _need(isinstance(ref, dict) and isinstance(ref.get('path'), str)
          and isinstance(ref.get('sha256'), str), 'explicit path/sha256 binding required')
    path = Path(ref['path'])
    _need(path.is_absolute(), 'capacity evidence paths must be absolute')
    _need(file_sha(path) == ref['sha256'], 'capacity evidence checksum differs: ' + str(path))
    return path.resolve()


def _bound(ref):
    return json.loads(_verified_path(ref).read_text())


def _input_bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            _verified_path(value)
        else:
            for child in value.values():
                _input_bindings(child)
    elif isinstance(value, list):
        for child in value:
            _input_bindings(child)


def trial_from_receipt(receipt_ref, point_ref, *, config, series, repeat_id, raw_refs=None):
    """Load one hash-bound window; malformed bindings raise, failed gates abstain.

``raw_refs`` is only needed for older audits which recorded the digest of their
raw-reference mapping without embedding that mapping. Its digest must still
match the original acceptance. A changed repeat label cannot duplicate raw data.
    """
    receipt, point = _bound(receipt_ref), _bound(point_ref)
    _need(isinstance(receipt, dict) and isinstance(point, dict), 'capacity receipt and point must be objects')
    _need(isinstance(series, dict) and set(series) == {'model_id', 'system', 'dataset'}
          and all(isinstance(v, str) and v for v in series.values()),
          'capacity series needs explicit model_id, system and dataset')
    _need(all(point.get(k) == v for k, v in series.items()), 'capacity point series differs')
    _need(receipt.get('point_sha256') == digest(point) and receipt.get('point') == point.get('name'),
          'capacity receipt does not bind the expected point')
    window = Path(receipt_ref['path']).resolve().parent
    artifacts = receipt.get('artifacts')
    _need(isinstance(artifacts, dict) and {'point.json', 'result.json', 'reset.json', 'drain.json'}
          <= artifacts.keys(), 'capacity receipt lacks required window artifacts')
    for name, sha in artifacts.items():
        path = (window / name).resolve()
        _need(path.is_relative_to(window), 'capacity artifact escapes its window')
        _verified_path(dict(path=str(path), sha256=sha))
    _need(json.loads((window/'point.json').read_text()) == point,
          'capacity window point differs from the expected point')
    result = json.loads((window/'result.json').read_text())
    _need(receipt.get('result') == result, 'capacity result differs from its receipt')
    inputs = point.get('inputs', {})
    _input_bindings(inputs)
    _need(inputs.get('trace') == point.get('trace'), 'capacity dispatcher trace differs')
    split = _bound(point['trace']).get('selection_split')
    _need(split in {'calibration', 'tuning'}, 'capacity cannot select from evaluation traces')
    # Replay the immutable generator against its actual corpus and independent
    # anchor. A split label and a plausible rate alone do not identify workload.
    workload = validate_trace(point['trace'])
    trace, family_ref = workload['trace'], workload['family_ref']
    family = workload['family']
    family_identity = family['identity']
    _need(point.get('capacity_workload_family') == family_ref
          and inputs.get('capacity_workload_family') == family_ref,
          'capacity point/dispatcher workload family binding differs')
    _need(config.required_repeats == len(family_identity['seeds'])
          and config.min_requests_per_trial == family_identity['minimum_requests'],
          'capacity bracket protocol differs from workload family')
    _need(trace.get('selection_split') == split, 'capacity trace split changed during replay')
    _need(point.get('selection_split', split) == split, 'capacity point/trace split differs')
    _need(point.get('system') in family_identity['systems'], 'capacity system is outside workload family')
    _need(all(trace.get(k) == point.get(k) for k in ('model_id', 'dataset', 'seed', 'duration_s',
          'rate_rps', 'scale', 'family_id', 'repeat_id', 'measurement_protocol_version', 'output_workload')),
          'capacity trace identity differs from its point')
    _need(repeat_id == trace['repeat_id'], 'capacity repetition must use the frozen seed identity')
    slo = dict(ttft_s=config.slo_ttft_s, tpot_s=config.slo_tpot_s)
    _need(point.get('slo') == slo and trace.get('slo') == slo, 'capacity point/trace frozen SLO differs')
    _need(isinstance(trace.get('requests'), list) and trace['requests'], 'capacity trace has no request cohort')
    scale, rate = point.get('scale'), point.get('rate_rps')
    _need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (scale, rate)),
          'capacity rate scale and offered rate must be finite and positive')
    _need(type(point.get('duration_s')) in (float, int) and math.isfinite(point['duration_s'])
          and point['duration_s'] > 0, 'capacity service duration must be finite and positive')
    signature = engine_signature(point.get('engine_identity'))
    _need(receipt.get('engine_signature') == signature, 'capacity engine identity differs')
    devices = [u for row in point['engine_identity']['instances'] for u in row['gpu_uuids']]
    _need(len(devices) == 8 and all(isinstance(u, str) and u for u in devices)
          and len(set(devices)) == 8, 'capacity trial must bind all eight physical GPUs')
    _need(all(point['engine_identity'].get(k) == family_identity[k] for k in ('model_hash', 'tokenizer_hash')),
          'capacity engine model/tokenizer differs from workload family')
    source = point.get('source_manifest', inputs.get('source_manifest'))
    source_manifest = _bound(source)
    _need(isinstance(point.get('revision'), str) and point['revision'], 'capacity source revision is missing')
    _need(source_manifest.get('source_sha256') == point['revision'], 'capacity source manifest revision differs')
    _need(inputs.get('source_manifest', source) == source, 'capacity dispatcher source manifest differs')
    if 'identity' in result:
        _need(result['identity'].get('source_sha256') == point['revision'],
              'capacity measured source differs from point revision')
    system_config = _bound(inputs.get('system_config'))
    _need(all(system_config.get(k) == point[k] for k in ('model_id', 'system')),
          'capacity system configuration identity differs')
    profiles = inputs.get('profiles', [])
    _need(isinstance(profiles, list) and (point['system'] == 'mixed' or profiles),
          'capacity system profile bindings are missing')
    audit, metrics = result.get('acceptance', {}), dict(result.get('metrics', {}))
    _need(isinstance(audit, dict), 'capacity measurement acceptance is missing')
    refs = audit.get('raw_refs', result.get('raw_refs', raw_refs))
    _need(isinstance(refs, dict) and {'trace', 'outcomes', 'canonical_requests', 'native_result',
          'power', 'metering', 'reset', 'drain'} <= refs.keys(), 'capacity raw-reference mapping is incomplete')
    _need(audit.get('evidence_sha256') == digest(refs), 'capacity raw references differ from recorded acceptance')
    _need(refs['trace'] == point['trace'], 'capacity raw trace differs')
    for name, ref in refs.items():
        path = _verified_path(ref)
        if path.is_relative_to(window):
            _need(artifacts.get(str(path.relative_to(window))) == ref['sha256'],
                  'capacity raw reference is not bound by window artifacts: ' + name)
    if 'point_sha256' in audit:
        _need(audit['point_sha256'] == digest(point), 'capacity acceptance point differs')
    if 'metrics_sha256' in audit:
        _need(audit['metrics_sha256'] == digest(metrics), 'capacity acceptance metrics differ')
    _need(metrics.get('offered_requests') == len(trace['requests']), 'capacity request denominator differs from trace')
    _need(metrics.get('duration_s') == point['duration_s'], 'capacity metric service duration differs')
    _need(metrics.get('measurement_protocol_version') == point.get('measurement_protocol_version')
          and bool(point.get('measurement_protocol_version')), 'capacity measurement protocol differs')
    checked = set(audit.get('checked_gates', []))
    canonical = {'pdblend.canonical_metrics', 'metrics.client_canonical', 'eco.raw_protocol_and_canonical_metrics'}
    reset, drain = (json.loads((window/(name+'.json')).read_text()) for name in ('reset', 'drain'))
    metrics['measurement_usable'] = bool(
        receipt.get('cleanup_passed') is True and reset.get('passed') is True and drain.get('passed') is True
        and result.get('measurement_evidence_valid', result.get('evidence_valid')) is True
        and audit.get('measurement_evidence_valid', audit.get('evidence_valid')) is True
        and not audit.get('missing_gates') and not audit.get('gate_failures') and not audit.get('blocked_gates')
        and 'metering.raw_eight_gpu_window' in checked and canonical & checked)
    identity = dict(series=series, revision=point['revision'], source_manifest_sha256=source['sha256'],
        system_config_sha256=inputs['system_config']['sha256'], profiles=sorted(r['sha256'] for r in profiles),
        engine_signature=signature, measurement_protocol_version=point['measurement_protocol_version'],
        duration_s=point['duration_s'], base_rate_rps=rate/scale,
        family_sha256=family_ref['sha256'], family_id=family['family_id'],
        gpu_uuids=sorted(devices), output_workload=family_identity['output_workload'],
        model_hash=family_identity['model_hash'], tokenizer_hash=family_identity['tokenizer_hash'])
    _need(math.isfinite(identity['base_rate_rps']) and identity['base_rate_rps'] > 0,
          'capacity base rate is outside numeric range')
    metrics['_capacity_evidence'] = dict(receipt=receipt_ref, point=point_ref, trace=point['trace'],
        family=family_ref, seed=trace['seed'], repeat_id=trace['repeat_id'],
        identity=identity, measurement_usable=metrics['measurement_usable'])
    # Paths, point names and user-supplied repetition labels cannot turn the
    # same measured window into independent evidence.
    evidence = digest({name: refs[name]['sha256'] for name in ('outcomes', 'native_result', 'power', 'canonical_requests')})
    trial = CapacityTrial(config.series_id, split, scale, repeat_id, 'raw-window:'+evidence, metrics)
    trial_verdict(trial, config)
    return trial


def read_capacity_ledger(path):
    """Return state/next rate and, only after convergence, a frozen future grid.

    Schema: ``{schema, config, series, trials:[{receipt, point, repeat_id}],
    multipliers?}``. All evidence references are absolute path/SHA256 objects.
    The function reads files and returns JSON-compatible data; it writes nothing.
    """
    path = Path(path).resolve()
    ref = dict(path=str(path), sha256=file_sha(path))
    ledger = _bound(ref)
    _need(isinstance(ledger, dict), 'capacity ledger must be an object')
    _need(ledger.get('schema') == 'pdblend-slo-capacity-ledger/v1', 'unsupported capacity ledger schema')
    config = CapacityConfig(**ledger['config'])
    series = ledger['series']
    _need(isinstance(series, dict) and set(series) == {'model_id', 'system', 'dataset'}
          and all(isinstance(v, str) and v for v in series.values()), 'invalid capacity series identity')
    _need(isinstance(ledger.get('trials'), list), 'capacity ledger requires a trials list')
    trials = [trial_from_receipt(row['receipt'], row['point'], config=config, series=series,
              repeat_id=row['repeat_id'], raw_refs=row.get('raw_refs')) for row in ledger['trials']]
    evidence = [trial.metrics['_capacity_evidence'] for trial in trials]
    if evidence:
        expected = evidence[0]['identity']
        for row in evidence[1:]:
            actual = row['identity']
            _need(all(actual[k] == v for k, v in expected.items() if k != 'base_rate_rps')
                  and math.isclose(actual['base_rate_rps'], expected['base_rate_rps'], rel_tol=1e-12, abs_tol=0),
                  'capacity series changed source, profile, configuration, hardware or workload family/rate anchor')
    state = capacity_state(trials, config)
    grid = (freeze_evaluation_grid(trials, config, multipliers=ledger.get('multipliers', (.25, .5, .75, 1., 1.1)))
            if state['converged'] else None)
    return dict(schema='pdblend-slo-capacity-report/v1', ledger=ref, series=series,
        state=state, frozen_evaluation_grid=grid, evidence=evidence,
        formal_eligible=False, profile_qualification_promoted=False,
        hardware_executed=False, jobs_enqueued=False)
