#!/usr/bin/env python3
"""Explicitly owned, auxiliary-only meter. Never imports the experiment driver.

Run during loading; prepare-stop only after session completion and while the
root scheduler holds the next service window until final.json exists. No queue,
GPU control, canonical mutation, automatic restart, or historical backfill.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
SCHEMA = 'auxiliary-resident-meter/v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def read_json(path, *, limit=4 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError('metadata exceeds explicit size limit: ' + str(path))
    return json.loads(path.read_text())


def read_ref(ref):
    if file_ref(ref['path']) != ref:
        raise ValueError('bound file changed: ' + str(ref['path']))
    return read_json(ref['path'])


def write_new(path, value):
    """Atomic, no-clobber publication; never truncates an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return file_ref(path)


class OwnerLock:
    """The same eight physical cards cannot have two auxiliary owners."""
    def __init__(self, directory, uuids):
        self.directory, self.uuids, self.files = Path(directory), uuids, []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            for uuid in sorted(self.uuids):
                stream = (self.directory / (hashlib.sha256(uuid.encode()).hexdigest() + '.lock')).open('a+')
                self.files.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.__exit__(None, None, None)
            raise RuntimeError('another auxiliary observer owns an overlapping GPU set')
        return self

    def __exit__(self, *_):
        for stream in reversed(self.files):
            stream.close()
        self.files.clear()


def verify_source(source_dir):
    source_dir = Path(source_dir).resolve()
    manifest = read_json(source_dir / 'manifest.json')
    if manifest.get('source_sha256') != digest(manifest['files']):
        raise ValueError('source inventory digest mismatch')
    for name, expected in manifest['files'].items():
        path = (source_dir / name).resolve()
        if not path.is_relative_to(source_dir) or file_ref(path)['sha256'] != expected:
            raise ValueError('source file mismatch: ' + name)
    files = manifest['files']
    meter = {k: v for k, v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')}
    return dict(manifest=file_ref(source_dir / 'manifest.json'), source_sha256=manifest['source_sha256'],
                measurement_source_sha256=digest(meter))


def import_meter(source_dir):
    source_dir = Path(source_dir).resolve()
    sys.path.insert(0, str(source_dir))
    module = importlib.import_module('pdblend.bench.isolated_comparison_meter')
    if not Path(module.__file__).resolve().is_relative_to(source_dir):
        raise ValueError('loaded meter is not from the selected frozen source')
    return module.IsolatedComparisonMeter


def prepare_config(args):
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('observer output directory must be new or empty')
    group_ref = file_ref(args.group)
    group = read_ref(group_ref)
    uuids = group['engine_identity']['fleet_gpu_uuids']
    if len(uuids) != 8 or len(set(uuids)) != 8 or any(not str(u).startswith('GPU-') for u in uuids):
        raise ValueError('exactly eight distinct ordered physical UUIDs required')
    source = verify_source(args.source_dir)
    execution_sources = sorted({read_ref(point['source_manifest'])['source_sha256'] for point in group['points']})
    return dict(schema=SCHEMA, created_s=time.time(), auxiliary_only=True,
        owner_pid=os.getpid(), owner_parent_pid=os.getppid(), job_id=args.job_id,
        session_dir=str(Path(args.session_dir).resolve()), session_id=group['session_id'],
        group=group_ref, group_sha256=digest(group), engine_signature=group['engine_signature'],
        execution_source_sha256=execution_sources, observer_source=source,
        script=file_ref(__file__), source_dir=str(Path(args.source_dir).resolve()),
        gpu_ids=list(range(8)), gpu_uuids=uuids, interval_s=.1, max_gap_s=1.,
        lock_dir=str(Path(args.lock_dir).resolve()), out=str(out),
        production_factory=True, selection_rule='auxiliary-only; canonical unchanged',
        stop_protocol='root binds terminal session and prevents any next service until final.json',
        failure_protocol='observer exits; no experiment signal, restart, or fragment joining')


def bound_artifact(receipt, directory, name):
    expected = receipt.get('artifacts', {}).get(name)
    if not expected:
        raise ValueError('receipt lacks ' + name)
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('artifact escapes window')
    ref = file_ref(path)
    if ref['sha256'] != expected:
        raise ValueError('receipt artifact mismatch: ' + name)
    return read_json(path), ref


def window_binding(ref, config):
    path = Path(ref['path']).resolve()
    if not path.is_relative_to(Path(config['session_dir']) / 'windows'):
        raise ValueError('window is outside this exact session; no historical backfill')
    receipt = read_ref({'path': str(path), 'sha256': ref['sha256']})
    item = dict(receipt=file_ref(path), point=receipt.get('point'), available=False)
    if not receipt.get('result'):
        return dict(item, unavailable_reason='no_recorded_service_result')
    result, result_ref = bound_artifact(receipt, path.parent, 'result.json')
    if result != receipt['result']:
        raise ValueError('embedded and bound results differ')
    identity = result['identity']
    if identity['gpu_uuids'] != config['gpu_uuids']:
        raise ValueError('window UUID order differs from observer')
    if identity['source_sha256'] not in config['execution_source_sha256']:
        raise ValueError('window execution source differs from group')
    if receipt.get('engine_signature') != config['engine_signature']:
        raise ValueError('window engine differs from observer group')
    point, point_ref = bound_artifact(receipt, path.parent, 'point.json')
    if point['name'] != receipt['point']:
        raise ValueError('point identity mismatch')
    drain, drain_ref = bound_artifact(receipt, path.parent, 'drain.json')
    metrics = result['metrics']
    start, end, tail = (metrics[k] for k in ('service_start_s', 'service_end_s', 'tail_end_s'))
    duration = point['duration_s']
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end, tail, duration)):
        raise ValueError('nonfinite window boundaries')
    if duration <= 0 or abs(end - start - duration) > 1e-5 or tail < end:
        raise ValueError('window boundary ordering mismatch')
    if drain.get('passed') is not True or abs(drain.get('tail_end_s', -1) - tail) > 1e-5:
        return dict(item, unavailable_reason='terminal_native_drain_not_verified', result=result_ref, drain=drain_ref)
    return dict(item, available=True, result=result_ref, point_artifact=point_ref, drain=drain_ref,
                execution_identity=identity, service_start_s=start, service_end_s=end,
                duration_s=duration, tail_end_s=tail)


def build_stop_request(config_ref, completion_path, *, no_next_service_until_finalized):
    validation_started_s = time.time()
    validation_clock = time.perf_counter()
    if no_next_service_until_finalized is not True:
        raise ValueError('root must hold the next service until observer export finishes')
    config = read_ref(config_ref)
    completion_path = Path(completion_path).resolve()
    if completion_path != Path(config['session_dir']) / 'completion.json':
        raise ValueError('completion belongs to a different session')
    completion_ref = file_ref(completion_path)
    completion = read_ref(completion_ref)
    if completion.get('session_id') != config['session_id'] or completion.get('engine_signature') != config['engine_signature']:
        raise ValueError('session completion identity mismatch')
    if completion.get('group_sha256') != config['group_sha256']:
        raise ValueError('session completion group digest mismatch')
    if (type(completion.get('finished_s')) not in (int, float)
            or not math.isfinite(completion['finished_s']) or completion['finished_s'] > time.time()):
        raise ValueError('session has not reached a recorded terminal time')
    paths = [str(Path(r['path']).resolve()) for r in completion.get('windows', [])]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate window receipt in session completion')
    windows = [window_binding(ref, config) for ref in completion.get('windows', [])]
    if any(w['available'] and w['tail_end_s'] > completion['finished_s'] + 1e-5 for w in windows):
        raise ValueError('a native drain tail ends after recorded session completion')
    # A terminated session may contain only a reset failure; record that honestly.
    tail = max([w['tail_end_s'] for w in windows if w['available']] + [completion['finished_s']])
    if tail > time.time():
        raise ValueError('a recorded drain tail has not ended')
    return dict(schema=SCHEMA + '/stop', created_s=time.time(), requester_pid=os.getpid(),
                config=config_ref, completion=completion_ref, windows=windows,
                excluded_historical_skips=completion.get('skipped', []), after_s=tail,
                metadata_validation_started_s=validation_started_s,
                metadata_validation_elapsed_s=time.perf_counter()-validation_clock,
                no_next_service_until_finalized=True)


def validate_stop(request, config_ref):
    if request.get('config') != config_ref or request.get('no_next_service_until_finalized') is not True:
        raise ValueError('stop request identity or scheduler hold missing')
    fresh = build_stop_request(config_ref, request['completion']['path'], no_next_service_until_finalized=True)
    for key in ('completion', 'windows', 'after_s', 'excluded_historical_skips'):
        if fresh[key] != request[key]:
            raise ValueError('stop request evidence changed: ' + key)
    return fresh


def project_windows(snapshot, method, request, config, summarize):
    result = []
    for bound in request['windows']:
        row = dict(bound, auxiliary_only=True, used_for_ranking=False, canonical_modified=False)
        if not bound['available']:
            result.append(row)
            continue
        if method['started_s'] > bound['service_start_s']:
            row.update(available=False, unavailable_reason='observer_started_after_service_start')
        elif method.get('test_factory') is True:
            row.update(available=False, unavailable_reason='test_factory_never_measurement_evidence')
        else:
            summary = summarize(snapshot, gpu_uuids=config['gpu_uuids'],
                origin_s=bound['service_start_s'], duration_s=bound['duration_s'],
                tail_end_s=bound['tail_end_s'], max_gap_s=1., gpu_uuid_binding_verified=True)
            row.update(summary=summary, available=summary['energy_comparable'])
            if not row['available']:
                row['unavailable_reason'] = 'original_source_or_coverage_checks_not_complete'
        result.append(row)
    return result


def run_observer(config, *, meter_factory=None, poll_s=.5):
    """Factory injection is CPU-test-only; the CLI never exposes it."""
    out = Path(config['out'])
    with OwnerLock(config['lock_dir'], config['gpu_uuids']):
        config_ref = write_new(out / 'config.json', config)
        meter = None
        try:
            factory = meter_factory or import_meter(config['source_dir'])
            meter = factory(config['gpu_ids'], config['gpu_uuids'], interval_s=.1, max_gap_s=1.)
            meter.start()
            meter.begin_window()  # Guard the entire externally owned active interval.
            method = meter.method_receipt()
            write_new(out / 'ready.json', dict(schema=SCHEMA, config=config_ref, ready_s=time.time(),
                      owner_pid=os.getpid(), method=method, auxiliary_only=True))
            while not (out / 'stop-request.json').exists():
                # Read only local process state. No child RPC or raw serialization.
                if not meter.method_receipt()['child_alive']:
                    raise RuntimeError('auxiliary child exited before root stop request')
                time.sleep(poll_s)
            export_started_s, export_clock = time.time(), time.perf_counter()
            request = read_json(out / 'stop-request.json')
            request = validate_stop(request, config_ref)
            meter.end_window()
            stop_clock = time.perf_counter()
            meter.stop(after_s=request['after_s'], timeout_s=2.)
            snapshot = meter.snapshot()  # Already stopped; local final copy.
            stop_and_copy_s = time.perf_counter()-stop_clock
            method = meter.method_receipt()
            if method.get('child_exitcode') != 0 or method.get('child_alive'):
                raise RuntimeError('auxiliary child did not exit cleanly')
            save_clock = time.perf_counter()
            snapshot_ref = write_new(out / 'raw-snapshot.json', snapshot)
            method_ref = write_new(out / 'method.json', method)
            raw_save_and_hash_s = time.perf_counter()-save_clock
            summarize = importlib.import_module('pdblend.bench.comparison_metering').summarize_comparison
            project_clock = time.perf_counter()
            windows = project_windows(snapshot, method, request, config, summarize)
            projection_s = time.perf_counter()-project_clock
            return write_new(out / 'final.json', dict(schema=SCHEMA, status='completed',
                finished_s=time.time(), config=config_ref, stop_request=file_ref(out / 'stop-request.json'),
                raw_snapshot=snapshot_ref, method=method_ref, windows=windows,
                auxiliary_only=True, canonical_modified=False, queue_modified=False,
                nonmeasurement_cost=dict(export_started_s=export_started_s,
                    export_elapsed_before_final_write_s=time.perf_counter()-export_clock,
                    stop_rpc_and_local_copy_s=stop_and_copy_s,
                    raw_save_and_hash_s=raw_save_and_hash_s, window_projection_s=projection_s,
                    snapshot_sample_rows=len(snapshot.get('samples', [])),
                    raw_snapshot_bytes=(out / 'raw-snapshot.json').stat().st_size,
                    window_count=len(windows), meter_rpc_timeout_s=meter.rpc_timeout_s,
                    sampler_final_bracketing_timeout_s=2.),
                fragment_joining=False, restart_count=0))
        except BaseException as exc:
            if meter is not None and meter.method_receipt().get('status') not in ('failed', 'stopped'):
                # Reap only this object's owned sampler child; never the experiment.
                meter._abort('external observer failed: ' + type(exc).__name__)
            write_new(out / 'failure.json', dict(schema=SCHEMA, failed_s=time.time(), config=config_ref,
                error=type(exc).__name__ + ': ' + str(exc), auxiliary_only=True,
                method=meter.method_receipt() if meter is not None else None,
                raw_evidence_complete=False, matrix_control_action=None, restart_count=0))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    for field in ('source-dir', 'group', 'session-dir', 'job-id', 'out'):
        run.add_argument('--' + field, required=True)
    run.add_argument('--lock-dir', default='/home/pdblend4/results/auxiliary-meter-owner-locks')
    stop = sub.add_parser('prepare-stop')
    stop.add_argument('--out', required=True)
    stop.add_argument('--hold-next-service-until-finalized', action='store_true', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run_observer(prepare_config(args))
    else:
        out = Path(args.out).resolve()
        config_ref = file_ref(out / 'config.json')
        config = read_ref(config_ref)
        request = build_stop_request(config_ref, Path(config['session_dir']) / 'completion.json',
            no_next_service_until_finalized=args.hold_next_service_until_finalized)
        result = write_new(out / 'stop-request.json', request)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
