"""Extend immutable preclose evidence with final five C observations; CPU/read-only sources."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'reports/trace-pairing-final-001'
SNAPSHOTS = ROOT / 'reports/trace-pairing-final-001-sources'
PRECLOSE = ROOT / 'reports/trace-pairing-preclose-001.json'
PRECLOSE_SHA = '06b829e205bb116c81a2694ccc8680e8b58ab52b407071d9db630f9664f22f6e'
SYSTEMS = {'pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve'}
csv.field_size_limit(16 * 1024**2)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'frozen source changed: ' + reference['path'])
    return read(reference['path'])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    need(not PREFIX.with_suffix('.json').exists() and not SNAPSHOTS.exists(), 'fresh final version required')
    preclose_ref = dict(path=str(PRECLOSE), sha256=PRECLOSE_SHA)
    prior = checked(preclose_ref)
    need(prior['passed'] and prior['observed_cell_count'] == 88 and prior['pending_new_measurement_count'] == 5,
         'unexpected preclose scope')
    checked(prior['sources']['snapshots'])
    need(sha(prior['verification_code']['path']) == prior['verification_code']['sha256'], 'preclose verifier changed')
    old_nodes = {}
    for source in prior['sources']['source_records']:
        value = checked(dict(path=source['snapshot_path'], sha256=source['sha256']))
        if source.get('node'):
            old_nodes[source['node']] = value
    for key in ('frozen_inputs', 'frozen_group', 'B_reference', 'B_immutable_source'):
        checked(prior['sources'][key])
    for key in ('frozen_generator100', 'frozen_generator300'):
        item = prior['sources'][key]
        need(sha(item['path']) == item['sha256'], 'frozen generator changed')
    nodes = {host: read(ROOT / host / 'observations.json') for host in ('A', 'C')}
    need(nodes['A'] == old_nodes['A'] and nodes['C'][:36] == old_nodes['C'], 'preclose observations modified')
    need(len(nodes['A']) == 21 and len(nodes['C']) == 41, 'unexpected final A/C counts')
    appended = nodes['C'][36:]
    need({row['cell_id'] for row in appended} == {row['cell_id'] for row in prior['pending_input_checks']},
         'final five cells differ from pending declarations')
    b_reference = checked(prior['sources']['B_reference'])
    old = checked(b_reference['source'])
    selected = set(b_reference.get('cell_ids', []))
    nodes['B'] = [row for row in old['observations'] if
        (row.get('measurement_host'), row.get('model'), row.get('dataset'), row.get('slo_scale')) ==
        ('B', '14b', 'sharegpt', 1.) and (not selected or row['cell_id'] in selected)]
    need(len(nodes['B']) == 31, 'unexpected B reference count')
    ledger = read(ROOT / 'reports/crosscheck-ledger.json')
    need(sha(ledger['checker']['path']) == ledger['checker']['sha256'], 'automatic checker changed')
    automatic = {(entry['cell_id'], entry.get('engineering_attempt', 1)): entry for entry in ledger['entries']}
    for host in ('A', 'C'):
        for row in nodes[host]:
            auto = automatic.get((row['cell_id'], row.get('engineering_attempt', 1)))
            need(auto and auto['status'] == 'passed' and auto['audit_reference'] == row['audit_reference'],
                 'final automatic core check absent or stale: ' + row['cell_id'])
    # The already reproduced eight frozen traces and their payload manifests are unchanged.
    # Recheck identity here, without repeating the original pool/generator reconstruction.
    workloads, traces = {}, {}
    for rate in prior['rates']:
        workload = checked(rate['workload'])
        checked(rate['manifest'])
        trace = checked(rate['trace'])
        need(workload['materialization_manifest'] == rate['manifest'] and
             workload['trace_reference'] == rate['trace'], 'frozen workload references changed')
        workloads[rate['rate_rps']] = workload
        traces[rate['rate_rps']] = trace
    # Inherited 88 checks remain attached to unchanged source bytes, not just cached booleans.
    inherited_refs = {}
    for check in prior['observed_cell_checks']:
        for key in ('checkpoint', 'trace', 'raw_requests', 'summary', 'receipt', 'runtime_config', 'historical_B_release'):
            reference = check.get(key)
            if reference:
                inherited_refs[(reference['path'], reference['sha256'])] = reference
    for reference in inherited_refs.values():
        need(sha(reference['path']) == reference['sha256'], 'preclose cell input changed: ' + reference['path'])
    new_checks = []
    for row in appended:
        need(row['measurement_valid'] is True and row['strict_slo_recomputed'] is True, 'invalid final cell')
        workload, trace = workloads[row['rate_rps']], traces[row['rate_rps']]
        audited = checked(row['audit_reference'])
        need(all(row.get(key) == value for key, value in audited.items()), 'published observation differs from audit')
        need(row['materialization_manifest'] == workload['materialization_manifest'], 'actual manifest differs')
        actual_trace = checked(row['trace_reference'])
        need(row['trace_sha256'] == workload['trace_sha256'] and actual_trace == trace, 'executed trace differs')
        need(row['content_pairing_sha256'] == workload['content_pairing_sha256'] and
             row['source_indices_sha256'] == workload['source_indices_sha256'] and
             row['sampling_seed'] == 20260907 and row['seed'] == 701 and row['arrival_window_s'] == 100.,
             'actual content/arrival identity differs')
        cp = checked(row['checkpoint'])
        for key in ('cell_id', 'system', 'repeat', 'trace_sha256', 'content_pairing_sha256',
                    'source_indices_sha256', 'seed', 'sampling_seed', 'arrival_window_s'):
            need(cp['row'][key] == row[key], 'executed checkpoint row differs: ' + key)
        summary, receipt = checked(row['summary']), checked(row['receipt'])
        need(summary['trace_sha256'] == receipt['trace_sha256'] == row['trace_sha256'], 'receipt trace differs')
        request_path = Path(row['raw_requests']['path'])
        need(sha(request_path) == row['raw_requests']['sha256'], 'raw bench changed')
        config_path = request_path.parent / 'runtime_config.json'
        config_ref = dict(path=str(config_path), sha256=cp['artifacts'][str(config_path)])
        config = checked(config_ref)
        need((config['slo_scale'], config['slo_ttft_s'], config['slo_tpot_s'], config['arrival_window_s']) ==
             (2., 10., .3, 100.), 'actual C controller SLO differs')
        need(summary['fixed_window']['effective_slo_s'] == dict(ttft=10., tpot=.3), 'effective C SLO differs')
        with request_path.open() as stream:
            requests = list(csv.DictReader(stream))
        actual = {request['request_id']: request for request in requests}
        need(len(actual) == len(requests) == trace['n_requests'] and
             set(actual) == {str(i) for i in range(trace['n_requests'])}, 'actual request denominator differs')
        offsets, deadlines = [], []
        epoch = summary['fixed_window']['arrival_epoch_s']
        for i, expected in enumerate(trace['requests']):
            request = actual[str(i)]
            need(int(request['prompt_len']) == expected['prompt_len'] and int(request['output_len']) == expected['output_len'],
                 'actual requested content lengths differ')
            offsets.append(abs(float(request['planned_arrival_s']) - epoch - expected['arrival_s']))
            deadlines.append(abs(float(request['request_deadline_s']) - float(request['planned_arrival_s']) - 120.))
        need(max(offsets) < 1e-5 and max(deadlines) < 1e-5, 'actual offset/deadline differs')
        new_checks.append(dict(cell_id=row['cell_id'], measurement_host='C', reference_only=False,
            system=row['system'], rate_rps=row['rate_rps'], repeat=row['repeat'], n_requests=trace['n_requests'],
            passed=True, checkpoint=row['checkpoint'], trace=row['trace_reference'], raw_requests=row['raw_requests'],
            summary=row['summary'], receipt=row['receipt'], runtime_config=config_ref,
            actual_slo_scale=2., actual_ttft_s=10., actual_tpot_s=.3, actual_arrival_window_s=100.,
            max_frozen_arrival_offset_error_s=max(offsets), max_request_deadline_error_s=max(deadlines),
            existing_automatic_core_crosscheck_passed=True, historical_B_release=None,
            historical_B_release_declaration_verified=None))
    # Exact declared grids include one PDB repeat2 at each host's confirmed boundary.
    for host, last_step in (('A', 4), ('B', 6), ('C', 8)):
        expected = {(step / 4, system, 1) for step in range(1, last_step + 1) for system in SYSTEMS}
        expected.add((last_step / 4, 'pdblend', 2))
        actual = {(row['rate_rps'], row['system'], row['repeat']) for row in nodes[host]}
        need(len(nodes[host]) == len(actual) and actual == expected, 'duplicate or incomplete declared grid: ' + host)
    SNAPSHOTS.mkdir()
    metadata = []
    for host in ('A', 'C'):
        source, destination = ROOT / host / 'observations.json', SNAPSHOTS / (host + '-observations.json')
        destination.write_bytes(source.read_bytes())
        need(read(destination) == nodes[host], 'observations changed during snapshot')
        metadata.append(dict(node=host, source_path=str(source), snapshot_path=str(destination),
                             sha256=sha(destination), count=len(nodes[host]), captured_s=time.time()))
    source, destination = ROOT / 'reports/crosscheck-ledger.json', SNAPSHOTS / 'crosscheck-ledger.json'
    destination.write_bytes(source.read_bytes())
    need(read(destination) == ledger, 'crosscheck ledger changed during snapshot')
    metadata.append(dict(kind='existing_automated_core_crosscheck', source_path=str(source), snapshot_path=str(destination),
                         sha256=sha(destination), captured_s=time.time()))
    save(SNAPSHOTS / 'snapshots.json', metadata)
    result = copy.deepcopy(prior)
    result.update(schema='slo-rate-trace-pairing-final-v1', created_s=time.time(), verification_code=ref(__file__),
        inherited_preclose=preclose_ref, inherited_preclose_verification_code=prior['verification_code'],
        inherited_observed_checks=88, appended_observed_checks=len(new_checks),
        inherited_input_reference_hashes_reverified=len(inherited_refs),
        generator_reconstruction_reused_from_preclose=True, generator_reconstruction_reexecuted=False,
        preclose_A_observations_unchanged=True, preclose_C36_observations_unchanged=True,
        final_C_observations=ref(ROOT / 'C/observations.json'),
        observed_counts={host: len(nodes[host]) for host in ('A', 'B', 'C')}, observed_cell_count=93,
        pending_new_measurement_count=0, complete_new_measurement_grid=True,
        exact_declared_grids_verified={'A': 21, 'B_reference': 31, 'C': 41},
        observed_cell_checks=prior['observed_cell_checks'] + new_checks, pending_input_checks=[],
        limitations=[text for text in prior['limitations'] if not text.startswith('Pending inputs')])
    result['sources'].update(snapshots=ref(SNAPSHOTS / 'snapshots.json'), source_records=metadata)
    for rate in result['rates']:
        rate['observed_cells'] = [row['cell_id'] for host in ('A', 'B', 'C') for row in nodes[host] if row['rate_rps'] == rate['rate_rps']]
        rate['pending_cells'] = []
        local_source = workloads[rate['rate_rps']]['source_300s_local']
        checked(local_source)
        rate.update(source300_local_file_present=True, source300_local_reference=local_source,
                    source300_local_hash_verified=True)
        for host in ('A', 'B', 'C'):
            rows = [row for row in nodes[host] if row['rate_rps'] == rate['rate_rps']]
            rate['observed_by_host'][host] = dict(cell_count=len(rows), systems=sorted({row['system'] for row in rows}),
                                               repeat2_cells=[row['cell_id'] for row in rows if row['repeat'] == 2])
            rate['observed_five_system_complete_by_host'][host] = {row['system'] for row in rows} == SYSTEMS
    result['maximum_observed_arrival_offset_error_s'] = max(row['max_frozen_arrival_offset_error_s'] for row in result['observed_cell_checks'])
    save(PREFIX.with_suffix('.json'), result)
    lines = ['最终冻结截面共 93 条：A 21/21、B 31 条固定历史参考、C 41/41。正式网格逐项一致且没有重复；全部实际 trace、内容采样、到达时序与生效 SLO 配置核对通过，没有待测点。',
        '', '沿用原预收尾版本已完成的 8 rate 原始 pool / 冻结生成器逐字节重建证据，并重新验证这些 workload、manifest 和 trace 哈希未变。原 88 条数据及其原始引用均未变；本次补验 C 的 EcoServe 在 1、1.25、1.5、1.75、2 req/s 的最后 5 条实际观测。预收尾文件保留原样。',
        '', '| rate (req/s) | 请求数 | A 观测数 | B 历史观测数 | C 观测数 |', '|---:|---:|---:|---:|---:|']
    for rate in result['rates']:
        counts = [str(rate['observed_by_host'][host]['cell_count']) if rate['observed_by_host'][host]['cell_count'] else '范围外' for host in ('A', 'B', 'C')]
        lines.append('| ' + str(rate['rate_rps']) + ' | ' + str(rate['n_requests']) + ' | ' + ' | '.join(counts) + ' |')
    lines += ['', '每个主机各自范围内的所有 rate 均含五系统；6 条表示五系统首测外增加 PDB 边界 repeat2。同 rate 的所有系统、跨机重叠 rate 及 repeat2 均使用相同 trace 字节、prompt、source indices、请求长度和到达偏移。arrival seed=701，内容采样 seed=20260907，窗口 100s；不同 rate 的内容序列为同一冻结采样的前缀。',
        '', '实际阈值逐条读取 checkpoint 哈希固定的 runtime_config.json，并核对实际 summary.fixed_window：A TTFT=2.5s / TPOT=0.075s；B=5s / 0.15s；C=10s / 0.3s。该结论来自实际运行配置，不是对 B 的结果事后重新评分。到达偏移最大误差 ' + format(result['maximum_observed_arrival_offset_error_s'], '.3g') + ' 秒，低于 1e-5 秒容差。',
        '', '最终 C observations SHA256：`' + result['final_C_observations']['sha256'] + '`。新 A/C 共 62 条自动原始核心指标核对全部通过，并与最终观测中的 audited 引用一致。',
        '', '相同 seed/content 不消除机器差异，不能据此推断纯 SLO 倍率因果效果，也不形成独立 seed 置信区间。bench 核查请求身份、长度与时序；prompt 通过实际执行且哈希固定的 trace 绑定。全程仅 CPU 读取与新证据写入，未改 runtime、旧数据、现场 audit 或 reports/current。',
        '', '[逐条最终证据](' + str(PREFIX.with_suffix('.json')) + ')；[原预收尾证据](' + str(PRECLOSE) + ')。']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(passed=True, observed_counts=result['observed_counts'], observed_cell_count=93,
                         appended_observed_checks=5, pending=0, final_evidence=ref(PREFIX.with_suffix('.json'))), indent=2))


if __name__ == '__main__':
    main()
