"""Lease-scoped engine reuse with immutable, independently reset windows.

The adapter owns hardware operations. This coordinator never changes a system's
policy and never resumes a window whose evidence cannot be verified.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError('immutable evidence differs: ' + str(path))
        return
    # Exclusive creation prevents two resume workers from replacing evidence.
    with path.open('x') as stream:
        stream.write(payload)


def engine_signature(identity):
    required = {'model_hash', 'tokenizer_hash', 'image_digest', 'runtime_source_sha256',
                'entrypoint', 'worker_extension', 'dtype', 'instances', 'environment'}
    if not isinstance(identity, dict) or not required <= identity.keys():
        raise ValueError('incomplete engine compatibility identity')
    for key in required - {'instances', 'environment', 'worker_extension'}:
        if not identity[key] or str(identity[key]).lower() in ('unknown', 'none'):
            raise ValueError('unbound engine identity: ' + key)
    instances = identity['instances']
    if not instances or not isinstance(identity['environment'], dict):
        raise ValueError('missing complete engine inventory/environment')
    fields = {'instance_id', 'tp', 'pp', 'gpu_uuids', 'launch_options'}
    ids, devices = set(), set()
    for row in instances:
        if not fields <= row.keys() or not isinstance(row['launch_options'], dict):
            raise ValueError('incomplete engine launch identity')
        if row['instance_id'] in ids or len(row['gpu_uuids']) != row['tp'] * row['pp']:
            raise ValueError('duplicate instance or invalid TP/PP mapping')
        if len(set(row['gpu_uuids'])) != len(row['gpu_uuids']) or devices & set(row['gpu_uuids']):
            raise ValueError('overlapping physical engine devices')
        ids.add(row['instance_id']); devices.update(row['gpu_uuids'])
    # Hash the ENTIRE supplied identity, including additional launch switches.
    return digest(identity)


def frequency_rejection_can_continue(point, result):
    """A measured clock rejection is not permission to relax its validity gate.

    This explicit policy only permits the next independently reset window
    after every other native/control/metric/meter gate has passed. The caller
    separately requires the current window's final all-rank drain to pass.
    """
    gate='eco.observed_active_frequency'
    audit=result.get('acceptance',{})
    required={'eco.observation_boundary','eco.fixed_fleet','eco.startup','eco.reset',
              'eco.all_rank_drain','metering.raw_eight_gpu_window',
              'eco.raw_protocol_and_canonical_metrics'}
    required|={'raw.'+name for name in ('trace','native_result','events','power','canonical_requests',
                                       'metering','startup_qualification','reset','drain')}
    required|={'binding.'+name for name in ('native_result','startup_qualification','reset','metering','drain','trace')}
    return (point.get('observation_failure_policy')=='continue_after_verified_frequency_rejection'
        and point.get('system')=='ecoserve'
        and point.get('qualification_mode')=='ecoserve_native_bootstrap'
        and result.get('evidence_valid') is False and result.get('formal_eligible') is False
        and result.get('missing_gates')==[gate] and audit.get('missing_gates')==[gate]
        and set(audit.get('gate_failures',{}))=={gate}
        and audit.get('evidence_valid') is False and audit.get('formal_eligible') is False
        and audit.get('preflight',{}).get('preflight_ready') is True
        and audit.get('preflight',{}).get('missing_gates')==[]
        and audit.get('preflight',{}).get('gate_failures')=={}
        and required<=set(audit.get('checked_gates',[]))
        and (point.get('metering_execution')!='isolated_process'
             or 'metering.isolated_process_method' in audit.get('checked_gates',[])))


RECORDED_RESULT_POLICY = 'all_recorded_windows/v1'


def recorded_window_can_continue(point, result):
    """User-selected analysis scope; strict audits remain unchanged diagnostics.

    A completed adapter result and the caller's successful native drain permit
    the next reset. Missing request timing or measured clock mismatches do not
    erase the actual observation, and neither SLO nor energy values are edited.
    """
    if point.get('result_policy') != RECORDED_RESULT_POLICY:
        return False
    metrics = result.get('metrics')
    if not isinstance(metrics, dict):
        return False
    start, end = metrics.get('service_started_s'), metrics.get('service_finished_s')
    return (point.get('duration_s') == metrics.get('duration_s') == 150
        and type(start) in (int, float) and math.isfinite(start)
        and type(end) in (int, float) and math.isfinite(end)
        and math.isclose(end-start, 150., abs_tol=1e-6, rel_tol=0.)
        and type(metrics.get('offered_requests')) is int and metrics['offered_requests'] > 0)


def observation_artifacts_match(point, result, window, artifacts=None):
    from .comparison_pdblend_observation import valid_observation_result
    if not valid_observation_result(point, result):
        return False
    relative = 'run/observation-acceptance.json'
    path = Path(window) / relative
    if not path.is_file():
        raise ValueError('observation acceptance artifact is missing')
    if artifacts is not None and artifacts.get(relative) != file_sha(path):
        raise ValueError('observation acceptance artifact binding differs')
    if json.loads(path.read_text()) != result['observation_acceptance']:
        raise ValueError('observation acceptance artifact differs from result')
    for name, ref in result['observation_acceptance']['raw_refs'].items():
        if file_sha(ref['path']) != ref.get('sha256'):
            raise ValueError('observation raw evidence changed: ' + name)
    return True


class ResidentGroupSession:
    """One adapter, one lease, multiple windows; reset failure quarantines it.

    Adapter methods are async: start(group), reset(point), execute(point, out),
    drain(point), close(). reset/drain return receipts with passed=True;
    execute returns a result whose evidence_valid distinguishes measurement
    integrity from SLO success. A measured SLO failure is still frozen.
    """
    def __init__(self, group, adapter, out, *, previous=()):
        self.group, self.adapter, self.out = group, adapter, Path(out)
        self.signature = engine_signature(group['engine_identity'])
        if group.get('engine_signature') != self.signature:
            raise ValueError('group signature differs')
        self.previous = tuple(Path(p) for p in previous)
        self.quarantined = False

    def _completed(self, point):
        for root in self.previous:
            path = root / 'windows' / point['name'] / 'receipt.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            if row.get('point_sha256') != digest(point) or row.get('engine_signature') != self.signature:
                continue
            if row.get('cleanup_passed') is not True:
                continue
            result = row.get('result', {})
            recorded = (recorded_window_can_continue(point, result)
                and row.get('recorded_window_complete') is True
                and row.get('result_policy') == RECORDED_RESULT_POLICY)
            observation = (False if recorded else
                observation_artifacts_match(point, result, path.parent, row.get('artifacts', {})))
            if not row.get('evidence_valid') and not observation and not recorded:
                continue
            if recorded:
                if json.loads((path.parent/'result.json').read_text()) != result:
                    raise ValueError('recorded result artifact differs')
                if json.loads((path.parent/'drain.json').read_text()).get('passed') is not True:
                    raise ValueError('recorded final drain is not passed')
            if observation:
                if row.get('baseline_frozen') is not False or row.get('measurement_evidence_valid') is not True:
                    raise ValueError('observation completion flags differ')
                if json.loads((path.parent/'result.json').read_text()) != row['result']:
                    raise ValueError('observation result artifact differs')
                if json.loads((path.parent/'drain.json').read_text()).get('passed') is not True:
                    raise ValueError('observation final drain is not passed')
            for name, expected in row.get('artifacts', {}).items():
                target = path.parent / name
                if not target.resolve().is_relative_to(path.parent.resolve()) or file_sha(target) != expected:
                    raise ValueError('completed window evidence changed: ' + str(target))
            if not row.get('artifacts'):
                raise ValueError('completed window lacks bound artifacts')
            return {'path': str(path.resolve()), 'sha256': file_sha(path)}
        return None

    async def run(self):
        self.out.mkdir(parents=True, exist_ok=False)
        report = dict(schema='resident-group-session/v1', session_id=self.group['session_id'],
                      engine_signature=self.signature, started_s=time.time(), status='failed',
                      group_sha256=digest(self.group),
                      planned_points={p['name']:digest(p) for p in self.group['points']},
                      complete=False, windows=[], skipped=[], cleanup_errors=[])
        pending = []
        for point in self.group['points']:
            old = self._completed(point)
            if old:
                report['skipped'].append(dict(point=point['name'], frozen_receipt=old))
            else:
                pending.append(point)
        started = False
        try:
            if pending:
                started = True
                # Frozen observations may bind an older controller/auditor
                # source. They are verified above, never requalified as though
                # executed by this new attempt's source bundle.
                report['startup'] = await self.adapter.start(dict(self.group,points=pending))
            for index, point in enumerate(pending):
                window = self.out / 'windows' / point['name']
                window.mkdir(parents=True, exist_ok=False)
                write_new(window / 'point.json', point)
                receipt = dict(point=point['name'], point_sha256=digest(point),
                               engine_signature=self.signature, window_index=index,
                               session_id=self.group['session_id'], resident_reused=index > 0,
                               evidence_valid=False, baseline_frozen=False, cleanup_passed=False)
                try:
                    reset = await self.adapter.reset(point)
                    write_new(window / 'reset.json', reset)
                    if reset.get('passed') is not True:
                        raise RuntimeError('window reset unqualified')
                    result = await self.adapter.execute(point, window / 'run')
                    write_new(window / 'result.json', result)
                    drain = await self.adapter.drain(point)
                    write_new(window / 'drain.json', drain)
                    if drain.get('passed') is not True:
                        raise RuntimeError('window native drain unqualified')
                    receipt.update(result=result, cleanup_passed=True,
                                   evidence_valid=result.get('evidence_valid') is True)
                    recorded = recorded_window_can_continue(point, result)
                    observation = False if recorded else observation_artifacts_match(point, result, window)
                    if recorded:
                        receipt.update(recorded_window_complete=True,
                            result_policy=RECORDED_RESULT_POLICY,
                            measurement_evidence_valid=result.get('measurement_evidence_valid', result.get('evidence_valid')) is True,
                            continuation='next_window_requires_fresh_reset_after_recorded_observation',
                            diagnostics=dict(strict_audit_missing_gates=result.get('missing_gates', []),
                                strict_audit_gate_failures=result.get('acceptance', {}).get('gate_failures', {})))
                    if observation:
                        receipt.update(measurement_evidence_valid=True, observation_complete=True,
                                       observation_scope=result['observation_scope'])
                    if not receipt['evidence_valid'] and not observation and not recorded:
                        if not frequency_rejection_can_continue(point,result):
                            raise RuntimeError('window measurement evidence invalid')
                        receipt.update(error='window rejected by observed active frequency gate',
                            continuation='next_window_requires_fresh_reset_after_verified_frequency_rejection')
                    else:receipt['baseline_frozen'] = point['system'] != 'pdblend'
                except Exception as exc:
                    receipt['error'] = f'{type(exc).__name__}: {exc}'
                    self.quarantined = True
                receipt['artifacts'] = {str(p.relative_to(window)): file_sha(p)
                                        for p in sorted(window.rglob('*')) if p.is_file()}
                write_new(window / 'receipt.json', receipt)
                report['windows'].append(dict(point=point['name'], path=str(window / 'receipt.json'),
                                               sha256=file_sha(window / 'receipt.json'),
                                               evidence_valid=receipt['evidence_valid'],
                                               recorded_window_complete=receipt.get('recorded_window_complete', False),
                                               measurement_evidence_valid=receipt.get('measurement_evidence_valid', receipt['evidence_valid'])))
                if self.quarantined:
                    raise RuntimeError('session quarantined after ' + point['name'])
            report.update(status='passed', complete=True,
                recorded_windows=sum(w['recorded_window_complete'] for w in report['windows']),
                all_observations_valid=all(w['measurement_evidence_valid'] for w in report['windows']),
                invalid_observations=sum(not w['measurement_evidence_valid'] for w in report['windows']))
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if started:
                cleanup_started = time.time()
                try:
                    report['cleanup'] = await self.adapter.close()
                    if report['cleanup'].get('passed') is not True:
                        raise RuntimeError('session process/clock cleanup incomplete')
                except Exception as exc:
                    report['cleanup_errors'].append(str(exc))
                    report.update(status='failed', complete=False)
                finally:
                    report['cleanup_s'] = time.time() - cleanup_started
            report.update(finished_s=time.time(), quarantined=self.quarantined)
            write_new(self.out / 'completion.json', report)
        return report
