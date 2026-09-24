"""Project existing session power onto a failed window's verified service span.

Independent CPU analysis only. No tail drain completion, canonical amendment,
baseline selection, or ranking is inferred from this service-only projection.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import sys
import time

from importlib import import_module

native_review = import_module('2026-09-25_extract_failed_distserve_native')

OUT = native_review.OUT
FROZEN = native_review.ROOT/'results/2026-09-24/profile-saturation-round-v3/baseline-sources/dfbb7d667198cc86e8826099f08eef4b12582062503ec798cbeb5048b557c92c'
INTEGRATOR_SHA = '14eefd4c284a37c5211eb55a143f61ce1b5c7aee277d4b2f88c4af5b1e5f02f8'
ARRAYS = ('samples', 'frequency_samples', 'utilization_samples', 'power_metadata')


def decode_snapshot(manifest, raw):
    if manifest.get('schema') != 'pdblend-power-v1':
        raise ValueError('expected original compact power archive')
    if hashlib.sha256(raw).hexdigest() != manifest['raw_sha256']:
        raise ValueError('power journal binding differs')
    snap = {k: v for k, v in manifest.items() if k not in
            ('schema', 'raw_path', 'raw_sha256', 'counts', 'power_source_epochs')}
    if any(k not in ARRAYS for k in manifest['counts']):
        raise ValueError('unexpected power series name')
    snap.update({k: [] for k in manifest['counts']})
    with gzip.open(io.BytesIO(raw), 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('_journal_schema') != 'pdblend-journal-v1':
                raise ValueError('unexpected journal schema')
            kind = row['kind']
            if kind not in manifest['counts']:
                raise ValueError('unregistered power series')
            value = row['values']
            if kind == 'power_metadata':
                epoch = row['source_epoch']
                if type(epoch) is not int or not 0 <= epoch < len(manifest['power_source_epochs']):
                    raise ValueError('invalid source epoch')
                value = {**manifest['power_source_epochs'][epoch], **value}
            snap[kind].append(value)
    if any(len(snap[k]) != n for k, n in manifest['counts'].items()):
        raise ValueError('power journal count mismatch')
    return snap


def stable_read(path, limit):
    path = Path(path); before = path.stat()
    if before.st_size > limit:
        raise ValueError('predeclared file size bound exceeded')
    data = path.read_bytes(); after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('source changed during read')
    return data, dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                      size_bytes=len(data), mtime_ns=before.st_mtime_ns)


def recover(job_id):
    destination = OUT/'distserve-sharegpt-lower-recovered-service-power.json'
    if destination.exists():
        return dict(status='already_cached', path=str(destination))
    native_path = OUT/'distserve-sharegpt-lower-failed-native-review.json'
    if not native_path.exists():
        return dict(status='deferred', reason='native_scalar_cache_not_ready')
    phase = native_review.phase(native_review.QUEUE, job_id)
    if not phase['ready']:
        return dict(status='deferred', reason='not_early_loading', phase=phase)
    native, native_ref, _ = native_review.read_bound(native_path, limit=1024*1024)
    if native['bindings']['receipt']['sha256'] != native_review.RECEIPT_SHA:
        raise ValueError('wrong failed-window binding')
    receipt, point, bindings = native_review.context()
    failure_method = native_review.small_ref(dict(
        path=str(native_review.WINDOW/'run/metering-method-failure.json'),
        sha256=receipt['artifacts']['run/metering-method-failure.json']))
    method_path = FROZEN/'pdblend/bench/comparison_metering.py'
    if hashlib.sha256(method_path.read_bytes()).hexdigest() != INTEGRATOR_SHA or failure_method['factory']['sha256'] != INTEGRATOR_SHA:
        raise ValueError('original integration method identity differs')
    phase = native_review.phase(native_review.QUEUE, job_id)
    if not phase['ready']:
        return dict(status='deferred', reason='loading_slot_ended_before_read', phase=phase)
    began = time.time(); clock = time.perf_counter()
    session = native_review.WINDOW.parent.parent
    data, manifest_ref = stable_read(session/'session-power.json', 24*1024*1024)
    manifest = json.loads(data); del data
    raw_path = session/manifest['raw_path']
    if raw_path.resolve().parent != session.resolve():
        raise ValueError('raw snapshot escapes bound session')
    raw, raw_ref = stable_read(raw_path, 3*1024*1024)
    read_ended = time.time()
    snap = decode_snapshot(manifest, raw); del raw
    uuids = point['engine_identity']['fleet_gpu_uuids']
    if (snap.get('gpu_uuids') != uuids or failure_method.get('gpu_uuids') != uuids
            or snap.get('gpu_uuid_binding_verified') is not True):
        raise ValueError('physical GPU order not verified')
    sys.path.insert(0, str(FROZEN))
    from pdblend.bench.comparison_metering import summarize_comparison
    service_start = native['service_started_s']; duration = point['duration_s']
    value = summarize_comparison(snap, gpu_uuids=uuids, origin_s=service_start,
                                 tail_end_s=service_start+duration, duration_s=duration,
                                 max_gap_s=1., gpu_uuid_binding_verified=True)
    after = native_review.phase(native_review.QUEUE, job_id)
    bound = value['energy_comparable']
    summary = dict(schema='failed-native-recovered-service-power/v1',
        bindings=bindings, native_scalar_review=native_ref,
        session_power_manifest=manifest_ref, session_power_raw=raw_ref,
        raw_binding_scope='raw SHA stored in original session-power manifest; manifest SHA first captured by this independent review, not included in the failed window receipt',
        source_unchanged_during_read=True,
        original_method=dict(path=str(method_path), sha256=INTEGRATOR_SHA),
        analysis_method=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        service_start_s=service_start, service_end_s=service_start+duration,
        recovered_service_available=bound,
        recovered_energy_service_j=value['energy_service_j'] if bound else None,
        recovered_service_mean_power_w=value['service_mean_power_w'] if bound else None,
        recovered_gpu_util_mean_pct=value['gpu_util_mean_pct'],
        service=value['service'], gpu_uuids=uuids,
        power_source_verified=value['power_source_verified'],
        power_error=value['power_error'], power_error_at_s=value['power_error_at_s'],
        power_error_affects_service=value['power_error_affects_window'],
        canonical_energy_service_j=None, canonical_energy_tail_j=None,
        recovered_energy_tail_j=None, recovered_tail_available=False,
        recovered_tail_reason='outer drain failed; no accepted drain-end boundary',
        outer_drain_complete=False, canonical_modified=False, used_for_ranking=False,
        extraction=dict(started_s=began, raw_read_finished_s=read_ended,
                        raw_bytes_read=manifest_ref['size_bytes']+raw_ref['size_bytes'],
                        wall_elapsed_s=time.perf_counter()-clock,
                        phase_before=phase, phase_after=after,
                        same_early_loading_phase_before_after=after['ready'] and phase.get('attempt_dir')==after.get('attempt_dir')))
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return dict(status='recovered', path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                available=bound, energy_j=summary['recovered_energy_service_j'],
                coverage=summary['service']['power']['coverage_fraction'],
                max_gap_s=summary['service']['power']['max_gap_s'],
                phase=after, elapsed_s=summary['extraction']['wall_elapsed_s'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-job', default=native_review.TARGET_JOB)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args(); deadline = time.monotonic()+args.wait_seconds
    while True:
        result = recover(args.wait_for_job)
        if result['status'] != 'deferred' or time.monotonic() >= deadline:
            print(json.dumps(result, sort_keys=True)); return
        time.sleep(min(20, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
