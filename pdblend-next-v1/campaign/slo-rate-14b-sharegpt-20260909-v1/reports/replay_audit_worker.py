"""One isolated, CPU-only replay of an immutable campaign checkpoint.

Calls only audit.audit. Never imports run_cell or invokes prepare/execute.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

csv.field_size_limit(16 * 1024**2)
CORE_FIELDS = ('n_expected', 'completed_work_requests', 'good_requests', 'generated_tokens',
    'completion_fraction', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s', 'measurement_duration_s',
    'energy_j', 'energy_per_gpu_j', 'gpu_util', 'gpu_util_per_gpu', 'request_throughput_rps',
    'token_throughput_tps', 'goodput_measurement_rps', 'energy_per_good_request_j',
    'failed_requests', 'request_timeouts', 'admission_rejections', 'unknown_error_count',
    'actual_output_tokens', 'observed_output_tokens_lower_bound',
    'completed_work_throughput_rps', 'generated_token_throughput_tps', 'producer_slo_attainment')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    if sha(reference['path']) != reference['sha256']:
        raise ValueError('immutable input hash changed: ' + reference['path'])
    return read(reference['path'])


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def differences(expected, actual, path='$'):
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                found.append(dict(path=path + '.' + key, reason='key_presence_differs'))
            else:
                found.extend(differences(expected[key], actual[key], path + '.' + key))
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [dict(path=path, reason='list_length_differs')]
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in differences(a, b, path + '[' + str(i) + ']')]
    if type(expected) in (int, float) and type(actual) in (int, float):
        matches = math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8)
    else:
        matches = type(expected) is type(actual) and expected == actual
    return [] if matches else [dict(path=path, expected=expected, actual=actual, reason='value_differs')]


def required_inputs(checkpoint_ref):
    cp = checked(checkpoint_ref)
    required = [checkpoint_ref]
    required.extend(dict(path=path, sha256=digest) for path, digest in cp['artifacts'].items())
    required.extend(cp[key] for key in ('release', 'binding', 'receipt'))
    for key in ('release', 'binding'):
        reference = cp[key]
        if not Path(reference['path']).is_file():
            continue
        value = checked(reference)
        if key == 'release':
            required.extend(value[field] for field in ('raw_auditor', 'measurement_auditor'))
        else:
            path = value['configs'][cp['row']['dataset']]
            required.append(dict(path=path, sha256=value.get('files', {}).get(path)))
    required.append(dict(path=cp['row']['trace'], sha256=cp['row']['trace_sha256']))
    distinct = {(item['path'], item.get('sha256')): item for item in required}
    return [distinct[key] for key in sorted(distinct, key=lambda x: (x[0], str(x[1])))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('fresh replay evidence directory required')
    args.out.mkdir(parents=True)
    started = time.time()
    result = dict(schema='slo-rate-isolated-full-raw-replay-v1', cell_id=args.cell_id,
        started_s=started, pid=__import__('os').getpid(), CPU_only=True, GPU_operations=False,
        called_entry='audit.audit(checkpoint_reference)', run_cell_imported=False,
        complete=False, passed=False, missing_files=[], hash_mismatches=[])
    try:
        manifest_ref = dict(path=str(args.manifest), sha256=args.manifest_sha256)
        manifest = checked(manifest_ref)
        item = next(item for item in manifest['cells'] if item['cell_id'] == args.cell_id)
        result.update(manifest=manifest_ref, checkpoint=item['checkpoint'],
                      original_audited=item['original_audited'], auditor=manifest['auditor'], worker=ref(__file__))
        original = checked(item['original_audited'])
        required = required_inputs(item['checkpoint']) + [manifest['auditor'], item['original_audited']]
        indexed = []
        for reference in required:
            path = Path(reference['path'])
            digest = sha(path) if path.is_file() else None
            record = dict(reference, actual_sha256=digest,
                status='missing' if digest is None else 'verified' if reference.get('sha256') in (None, digest) else 'mismatch')
            indexed.append(record)
            if digest is None:
                result['missing_files'].append(str(path))
            elif reference.get('sha256') and reference['sha256'] != digest:
                result['hash_mismatches'].append(record)
        save(args.out / 'required-inputs.json', indexed)
        if result['missing_files'] or result['hash_mismatches']:
            raise ValueError('required replay inputs are missing or changed')
        auditor_path = Path(manifest['auditor']['path'])
        sys.path.insert(0, str(auditor_path.parent))
        spec = importlib.util.spec_from_file_location('isolated_campaign_raw_auditor', auditor_path)
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        replayed = auditor.audit(item['checkpoint'])
        save(args.out / 'replayed.json', replayed)
        diff = differences(original, replayed)
        core = []
        for field in CORE_FIELDS:
            if field not in original or field not in replayed:
                core.append(dict(field=field, status='missing', original_present=field in original,
                    replay_present=field in replayed))
            else:
                field_diff = differences(original[field], replayed[field], field)
                core.append(dict(field=field, original=original[field], replayed=replayed[field],
                    exact_equal=original[field] == replayed[field], passed=not field_diff))
        result.update(complete=True, passed=not diff and all(entry.get('passed') is True for entry in core),
            core_numeric_comparison=core, full_result_differences=diff,
            full_result_exact_equal=original == replayed, replayed=ref(args.out / 'replayed.json'),
            numeric_tolerance=dict(relative=1e-8, absolute=1e-8),
            full_qualification_replayed=False,
            qualification_note='Replays the complete measurement audit; does not rerun native qualification or weight reconstruction')
    except BaseException as exc:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            result['missing_files'].append(exc.filename)
        result.update(error=repr(exc), traceback=traceback.format_exc())
    result.update(finished_s=time.time(), elapsed_s=time.time() - started,
        run_cell_imported=any(name == 'run_cell' or name.endswith('.run_cell') for name in sys.modules))
    if result['run_cell_imported']:
        result['passed'] = False
        result['error'] = 'unexpected run_cell import'
    save(args.out / 'result.json', result)
    print(json.dumps({key: result.get(key) for key in ('cell_id', 'complete', 'passed', 'elapsed_s', 'missing_files', 'error')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
