"""Independent diagnostic phase arithmetic. These are not formal rate observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise ValueError(message)


def close(actual, expected, message):
    need(abs(actual - expected) <= max(1e-6, abs(expected) * 1e-8), message)


def checked(ref, sources):
    path = ref['path']
    need(digest(path) == ref['sha256'], 'changed evidence: ' + path)
    sources[path] = ref['sha256']
    return read(path)


def integrate(path, start, end):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(r['t_s']) for r in rows])
    need(len(t) >= 2 and np.all(np.diff(t) > 0), 'power timestamps not strictly increasing')
    need(t[0] <= start < end <= t[-1], 'power does not bracket full phase')
    x = np.r_[start, t[(t > start) & (t < end)], end]
    energy, util = [], []
    for gpu in range(8):
        for suffix, target in [('w', energy), ('util_pct', util)]:
            y = np.array([float(r[f'gpu{gpu}_{suffix}']) for r in rows])
            need(np.all(np.isfinite(y)) and np.all(y >= 0), 'invalid power/util samples')
            z = np.interp(x, t, y)
            area = float(np.sum(np.diff(x) * (z[1:] + z[:-1]) / 2))
            target.append(area if suffix == 'w' else area / (end - start))
    return energy, util


def inspect(path, sources):
    result = checked(dict(path=str(path), sha256=digest(path)), sources)
    need(result['schema'] == 'capacity-load-measurement-v1', 'unknown phase schema')
    source = {k: checked(v, sources) for k, v in result['source'].items()}
    for owner in [result, source.get('capacity_binding', {}), source.get('original_binding', {})]:
        for f, sha in owner.get('files', {}).items():
            need(digest(f) == sha, 'changed frozen phase input: ' + f)
            sources[f] = sha
    for f, sha in result['artifacts'].items():
        need(digest(f) == sha, 'changed phase raw: ' + f)
        sources[f] = sha
    raw = checked(result['raw_measurement'], sources)
    for f, sha in raw['artifacts'].items():
        need(digest(f) == sha, 'changed measurement raw: ' + f)
        sources[f] = sha
    need(raw['measurement_valid'] and raw['gpu_indices'] == list(range(8))
         and raw['power_evidence']['power_source_verified'], 'unverified eight-GPU measurement')
    start, end = raw['measurement_start_s'], raw['measurement_end_s']
    energy, util = integrate(Path(result['raw_measurement']['path']).parent / 'power.csv', start, end)
    close(sum(energy), raw['energy_j'], 'independent energy differs from measurement')
    close(sum(energy), result['energy_j'], 'independent energy differs from phase')
    close(end - start, raw['duration_s'], 'duration differs')
    rows = read(path.parent / 'requests.json')
    idle = result.get('phase_kind') == 'idle'
    counts = dict(expected_requests=0, complete_requests=0, good_requests=0,
                  failed_requests=0, request_timeouts=0, http503=0,
                  expected_generated_tokens=0, generated_tokens=0)
    trace_sha, seed = None, None
    unverified_output_requests = 0
    if idle:
        need(not rows and result['n_expected'] == result['n_rows'] == result['n_good'] == 0,
             'idle contains business requests')
        need(result.get('zero_work_is_not_capacity_evidence') is True, 'idle capacity distinction missing')
    else:
        trace = checked(result['trace'], sources)
        trace_sha, seed = result['trace']['sha256'], trace['seed']
        wanted = trace['requests']
        need(len(rows) == len(wanted) == result['n_expected'] == result['n_rows'], 'phase request count differs')
        need(sorted(int(r['idx']) for r in rows) == list(range(len(wanted))), 'duplicate/missing request IDs')
        counts['expected_requests'] = len(wanted)
        counts['expected_generated_tokens'] = sum(r['output_len'] for r in wanted)
        for row in rows:
            request = wanted[int(row['idx'])]
            need(row['prompt_len'] == request['prompt_len'] and row['output_len'] == request['output_len'],
                 'prescribed request shape differs')
            success = bool(row['success'])
            complete = success and row['generated_tokens'] == request['output_len']
            good = (complete and row['ttft_s'] is not None and row['tpot_s'] is not None
                    and row['ttft_s'] <= source['config']['slo_ttft_s']
                    and row['tpot_s'] <= source['config']['slo_tpot_s'])
            need(bool(row['slo_ok']) == good, 'per-request SLO arithmetic differs')
            counts['complete_requests'] += int(complete)
            counts['good_requests'] += int(good)
            counts['failed_requests'] += int(not success)
            counts['request_timeouts'] += int(bool(row.get('request_timeout')))
            counts['http503'] += int(row.get('http_status') == 503)
            counts['generated_tokens'] += row['generated_tokens']
            unverified_output_requests += int(not bool(row.get('token_ids_verified')))
        need(counts['good_requests'] == result['n_good'], 'good request count differs')
        close(counts['good_requests'] / len(wanted), result['slo_attainment'], 'phase SLO differs')
        for field in ('failed_requests', 'request_timeouts'):
            if field in result:
                need(counts[field] == result[field], field + ' differs')
    complete = counts['complete_requests'] == counts['expected_requests']
    need(complete == result['work_complete'], 'phase work completeness differs')
    return dict(campaign=path.parents[1].name, phase=result['phase'], phase_kind='idle' if idle else 'business',
        formal_rate_observation=False, numerical_audit_verified=True,
        qualification_certificate_issued_by_this_report=False,
        result_path=str(path), result_sha256=sources[str(path)],
        controller_manifest_sha256=result['source']['host_manifest']['sha256'],
        capacity_binding_sha256=result['source']['capacity_binding']['sha256'],
        trace_sha256=trace_sha, seed=seed,
        nominal_phase=result['phase'], realized_arrival_rate_rps=result.get('offered_rate_rps'),
        arrival_duration_s=result.get('measured_arrival_duration_s'),
        measurement_start_s=start, measurement_end_s=end, measurement_duration_s=end-start,
        energy_j=sum(energy), per_gpu_energy_j=energy, per_gpu_util_pct=util,
        whole_node_mean_power_w=sum(energy)/(end-start),
        slo_attainment=result['slo_attainment'], work_complete=complete, **counts,
        generated_tokens_semantics='producer-recorded token count; partial failed requests may lack terminal usage',
        unverified_output_requests=unverified_output_requests,
        resident_gpus_before=sorted({g for x in result['resident_before'] for g in x['gpus']}),
        resident_gpus_after=sorted({g for x in result['resident_after'] for g in x['gpus']}),
        interpretation='Idle power only; no service-capacity evidence' if idle else '60-second calibration; not a 100-second formal comparison',
        energy_accounting='Each phase window independently; overlapping operation windows are not added')


def main(out, directories):
    need(not out.exists(), 'immutable output already exists')
    sources = {str(Path(__file__)): digest(__file__)}
    rows, invalid = [], []
    paths = sorted(p for d in directories for p in d.rglob('result.json'))
    for path in paths:
        try:
            rows.append(inspect(path, sources))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            invalid.append(dict(path=str(path), error=str(exc)))
    for path, sha in sources.items():
        need(digest(path) == sha, 'evidence changed during reading: ' + path)
    out.mkdir(parents=True)
    result = dict(created_s=time.time(), diagnostic_only=True, valid_phases=len(rows),
                  invalid_phases=invalid, rows=rows)
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'phases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    (out/'README.md').write_text('扩容校准诊断表：逐项保留旧版失败、新版冷启动的 SLO 负结果、空闲功耗和实际八卡能耗。\n\n这些 60 秒校准项不计入正式 100 秒 rate 比较。该表独立核算工作量、SLO 和八卡功率积分，不签发布局资格证书，不叠加相互重叠的能耗窗口。\n')
    (out/'manifest.json').write_text(json.dumps(dict(sources=sources,
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(json.dumps(dict(valid_phases=len(rows), invalid_phases=invalid)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('directories', type=Path, nargs='+'); args=parser.parse_args()
    main(args.out.resolve(), [d.resolve() for d in args.directories])
