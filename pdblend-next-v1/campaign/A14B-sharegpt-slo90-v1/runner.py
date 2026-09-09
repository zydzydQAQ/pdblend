"""Durable sequential SLO queue; one process owns the host lease throughout.

No retry, result reuse, baseline-local cutoff, or post-endpoint refinement is
implicit. An interrupted point is reconciled from its sealed receipt before
any new point is dispatched. Incomplete/invalid evidence requires diagnosis.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import protocol as p
import trace_source as traces

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def new_state(manifest):
    return dict(schema=1, protocol_id=p.PROTOCOL, created_s=time.time(),
                manifest_sha256=traces.file_sha(manifest), scan=p.new_state(),
                records=[], phase='pdblend', status='ready', next_attempt=1,
                executing_host=socket.gethostname(), stages={}, events=[])


def point_key(system, scale, rate):
    return system, p.scale_key(scale), p.number(rate)


def record_key(record):
    return point_key(record['system'], record['slo_scale'], record['rate_rps_decimal'])


def next_point(state):
    """Only return an authorized new measurement; blocked work never advances."""
    if state['status'] in ('blocked', 'running', 'complete'):
        return None
    scan = state['scan']
    for scale in p.SCALES:
        track = scan['scales'][p.scale_key(scale)]
        if track['status'] == 'blocked':
            return None
        rate = p.next_rate(scan, scale)
        if rate is not None:
            return point_key('pdblend', scale, rate)
    p.require(p.pdb_complete(scan), 'PDB scanning has not established both endpoints')
    valid = {record_key(r) for r in state['records'] if r.get('status') == 'valid'}
    for system in p.BASELINES:
        for scale in p.SCALES:
            for rate in p.required_baseline_rates(scan, scale):
                key = point_key(system, scale, rate)
                if key not in valid:
                    return key
    return None


def apply_result(state, index, meta):
    result = copy.deepcopy(state)
    row = result['records'][index]
    p.require(row['status'] == 'running', 'a sealed attempt cannot be applied twice')
    verdict = meta.get('verdict', {})
    errors = result_errors(row, meta)
    valid = verdict.get('measurement_valid') is True and not errors
    row.update(status='valid' if valid else 'invalid', finished_s=time.time(),
               verdict=verdict, technical_error='; '.join(errors) or None)
    for field in ('receipt_path', 'summary_path', 'receipt_sha256', 'summary_sha256'):
        if meta.get(field):
            row[field] = meta[field]
    if row['system'] == 'pdblend':
        # Counts are independently re-derived from every offered raw row. The
        # report auditor distinguishes legitimate admission refusals from 503
        # engineering failures; raw receipt remains unchanged and preserved.
        summary = dict(measurement_valid=valid, fixed_window_valid=valid,
            post_measurement_cleanup={'cleanup_complete': valid},
            offered_requests=verdict.get('offered_requests'),
            good_requests=verdict.get('good_requests'),
            slo_attainment=verdict.get('slo_attainment'),
            technical_error=row['technical_error'])
        result['scan'] = p.record_pdb(result['scan'], row['slo_scale'], row['rate_rps_decimal'],
            summary, technical_valid=valid,
            evidence={'point_result': row['point_result_path'], 'attempt': row['attempt']})
    result['status'] = 'ready' if valid else 'blocked'
    result['phase'] = 'baselines' if p.pdb_complete(result['scan']) else 'pdblend'
    result['updated_s'] = time.time()
    if valid and next_point(result) is None:
        result.update(status='complete', finished_s=time.time(),
                      stop_reason='both_first_valid_pdb_below_90_endpoints_and_all_paired_baselines_complete')
    elif not valid:
        result['stop_reason'] = 'technical_invalid_point_requires_diagnosis_not_slo_endpoint'
    return result


def result_errors(row, meta):
    """Bind a sealed worker result to this exact pending attempt on recovery."""
    errors = [str(value) for value in (meta.get('error'), meta.get('verdict', {}).get('technical_error')) if value]
    expected = dict(schema='a14b-sharegpt-slo90-point-result-v1', protocol_id=p.PROTOCOL,
                    complete=True, cell_id=row.get('cell_id'), executing_host=row.get('executing_host'),
                    source_version=row.get('source_version'))
    for key, value in expected.items():
        if value is None or meta.get(key) != value:
            errors.append('worker result identity differs: ' + key)
    verdict = meta.get('verdict', {})
    if verdict.get('measurement_valid') is True:
        offered, good = verdict.get('offered_requests'), verdict.get('good_requests')
        if type(offered) is not int or offered < 1 or type(good) is not int or not 0 <= good <= offered:
            errors.append('worker lacks complete integer request denominators')
        elif not isinstance(verdict.get('slo_attainment'), (int, float)) or not math.isfinite(verdict['slo_attainment']) or abs(verdict['slo_attainment'] - good / offered) > 1e-12:
            errors.append('worker joint SLO differs from offered denominator')
        if meta.get('measurement_valid') is not True or meta.get('exit_status') != 0 or meta.get('process_exit_status', 0) != 0:
            errors.append('worker completion/exit status differs')
        if verdict.get('stop_eligible') is not True:
            errors.append('valid worker result lacks audited stop eligibility')
        for key in ('cell_id', 'system', 'source_version', 'executing_host'):
            if verdict.get(key) != row.get(key):
                errors.append('worker verdict identity differs: ' + key)
        for key in ('slo_scale', 'rate_rps'):
            try:
                if p.number(verdict.get(key)) != p.number(row.get(key)):
                    errors.append('worker verdict workload differs: ' + key)
            except ValueError:
                errors.append('invalid worker verdict workload: ' + key)
        if verdict.get('trace_sha256') != row.get('trace', {}).get('sha256'):
            errors.append('worker verdict trace differs')
        for key in ('receipt_path', 'summary_path'):
            digest_key = key.replace('_path', '_sha256')
            try:
                if Path(meta.get(key, '')).resolve() != Path(row[key]).resolve():
                    errors.append('worker artifact belongs to another attempt: ' + key)
                if not meta.get(digest_key) or traces.file_sha(row[key]) != meta[digest_key]:
                    errors.append('worker artifact digest differs: ' + key)
            except (OSError, KeyError, TypeError):
                errors.append('worker artifact unavailable: ' + key)
        if row.get('row_path') and meta.get('sources', {}).get(row['row_path']) != row.get('row_sha256'):
            errors.append('worker executed another declaration')
    return errors


def reconcile(state_path):
    state = read(state_path)
    pending = [i for i, r in enumerate(state['records']) if r['status'] == 'running']
    p.require(len(pending) <= 1, 'concurrent point records violate single-host sequence')
    if pending:
        index = pending[0]
        meta = Path(state['records'][index]['point_result_path'])
        if meta.exists():
            state = apply_result(state, index, read(meta))
        else:
            state.update(status='blocked', stop_reason='interrupted_attempt_requires_evidence_reconciliation',
                         updated_s=time.time())
        save(state_path, state)
    return state


def load_trace(rate, root):
    directory = Path(root) / 'traces' / ('rate-' + p.number(rate))
    path = directory / 'manifest.json'
    if path.exists():
        materialized = read(path)
        for name, digest in materialized['files'].items():
            p.require(traces.file_sha(name) == digest, 'materialized trace input changed: ' + name)
        return materialized
    p.require(not directory.exists(), 'incomplete trace materialization requires diagnosis')
    return traces.materialize(rate, directory)


def prepare_attempt(state, manifest, root, binding_path):
    key = next_point(state)
    p.require(key is not None, 'no authorized pending point')
    system, scale, rate = key
    trace = load_trace(rate, root)['trace']
    row = traces.execution_row(trace, system, scale)
    binding = read(binding_path)
    p.require(binding['system'] == system, 'wrong system binding')
    p.require(binding['hostname'] == state['executing_host'], 'wrong actual execution host')
    config = read(binding['configs']['sharegpt'])
    row.update(source_version=manifest['versions'][system], executing_host=state['executing_host'],
               strategy=config['strategy'], controller_config=binding['configs']['sharegpt'],
               binding_path=str(binding_path), binding_sha256=traces.file_sha(binding_path),
               dispatch_delay_max_limit_s=manifest['dispatch_delay_max_limit_s'],
               dispatch_delay_p99_limit_s=manifest['dispatch_delay_p99_limit_s'])
    attempt = state['next_attempt']
    attempt_dir = Path(root) / 'attempts' / ('%04d-' % attempt + row['cell_id'])
    p.require(not attempt_dir.exists(), 'an existing attempt cannot be dispatched again')
    attempt_dir.mkdir(parents=True)
    job = attempt_dir / 'row.json'
    save(job, row)
    out = attempt_dir / 'results'
    record = dict(row, trace=trace, attempt=attempt, status='running', started_s=time.time(),
        row_path=str(job), row_sha256=traces.file_sha(job), point_result_path=str(attempt_dir / 'point-result.json'),
        receipt_path=str(out / 'operations' / row['cell_id'] / 'receipt.json'),
        summary_path=str(out / 'cells' / row['cell_id'] / 'summary.json'))
    result = copy.deepcopy(state)
    result['records'].append(record)
    result.update(status='running', next_attempt=attempt + 1, updated_s=time.time())
    return result, job, out


def execute_one(manifest_path, state_path, root, binding_path, lease_fd):
    manifest, state = read(manifest_path), read(state_path)
    p.require(traces.file_sha(manifest_path) == state['manifest_sha256'], 'declaration changed during scan')
    state, job, out = prepare_attempt(state, manifest, root, binding_path)
    save(state_path, state)  # Commit before spawn, preventing crash duplicate dispatch.
    binding = read(binding_path)
    command = [sys.executable, '-u', str(HERE / 'worker.py'), 'run-point',
               '--binding', str(binding_path), '--host-release', binding['host_release'],
               '--common', str(Path(binding['executor']).parent), '--row', str(job), '--out', str(out)]
    env = dict(os.environ, PDBLEND_NODE_LOCK_FD=str(lease_fd), PYTHONDONTWRITEBYTECODE='1')
    with (out.parent / 'worker.log').open('x') as log:
        process = subprocess.run(command, env=env, pass_fds=(lease_fd,), stdout=log, stderr=subprocess.STDOUT)
    meta_path = out.parent / 'point-result.json'
    meta = read(meta_path) if meta_path.exists() else dict(error='worker exited without sealed result: ' + str(process.returncode))
    meta['process_exit_status'] = process.returncode
    state = apply_result(read(state_path), len(state['records']) - 1, meta)
    save(state_path, state)
    return state


def make_report(manifest_path, state_path, root, figures=False):
    import report
    root = Path(root)
    report_id = str(time.time_ns())
    directory = root / 'reports' / report_id
    result = report.build_report(manifest_path, state_path, directory, figures=figures)
    save(root / 'reports/latest.json', dict(path=str(directory), totals=result['totals'], created_s=time.time()))
    return result
