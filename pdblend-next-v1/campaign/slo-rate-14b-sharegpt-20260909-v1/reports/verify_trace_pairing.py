"""Read-only paired workload/config review; invokes no measurement executor."""
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'reports/trace-pairing-preclose-001'
SNAPSHOTS = ROOT / 'reports/trace-pairing-preclose-001-sources'
csv.field_size_limit(16 * 1024**2)


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


def need(condition, message):
    if not condition:
        raise ValueError(message)


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed frozen source: ' + reference['path'])
    return read(reference['path'])


def module(reference, name):
    need(sha(reference['path']) == reference['sha256'], 'changed source code')
    spec = importlib.util.spec_from_file_location(name, reference['path'])
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    need(not PREFIX.with_suffix('.json').exists(), 'new pairing evidence version required')
    metadata = read(SNAPSHOTS / 'snapshots.json')
    for item in metadata:
        need(sha(item['snapshot_path']) == item['sha256'], 'snapshot bytes changed')
    nodes = {node: read(SNAPSHOTS / (node + '-observations.json')) for node in ('A', 'C')}
    b_reference = checked(ref(ROOT / 'B/reference.json'))
    old = checked(b_reference['source'])
    selected = set(b_reference.get('cell_ids', []))
    nodes['B'] = [row for row in old['observations'] if
        (row.get('measurement_host'), row.get('model'), row.get('dataset'), row.get('slo_scale')) ==
        ('B', '14b', 'sharegpt', 1.) and (not selected or row['cell_id'] in selected)]
    need(len(nodes['A']) == 21 and len(nodes['B']) == 31, 'A or B snapshot scope differs')
    inputs_ref = ref(ROOT / 'inputs/manifest.json')
    inputs = checked(inputs_ref)
    generator = module(inputs['dependencies']['generator100'], 'paired_frozen_generator100')
    parent = module(inputs['dependencies']['generator300'], 'paired_frozen_generator300')
    group = checked(inputs['dependencies']['group'])
    spec = checked(inputs['dependencies']['spec'])
    original_groups, sampling_seed = parent.load_sources(spec)
    need(sampling_seed == inputs['sampling_seed'] == 20260907, 'content sampling seed differs')
    source_group = original_groups[('14b', 'sharegpt')]
    need(all(source_group[key] == value for key, value in group.items()), 'frozen source-group payload differs')
    ledger = read(SNAPSHOTS / 'crosscheck-ledger.json')
    need(sha(ledger['checker']['path']) == ledger['checker']['sha256'], 'automatic checker code drift')
    automatic = {(entry['cell_id'], entry.get('engineering_attempt', 1)): entry for entry in ledger['entries']}
    workloads, traces, rate_rows, cell_checks, pending = {}, {}, {}, [], []
    for step in range(1, 9):
        rate = generator.number(step / 4)
        directory = ROOT / 'workloads' / ('r' + rate)
        workload_ref = ref(directory / 'workload.json')
        workload = checked(workload_ref)
        manifest_ref = workload['materialization_manifest']
        manifest = checked(manifest_ref)
        payload = dict(workload)
        payload.pop('materialization_manifest')
        need(generator.digest(payload) == manifest['workload_payload_sha256'], 'workload payload manifest differs')
        trace = checked(workload['trace_reference'])
        need(workload['trace_sha256'] == workload['trace_reference']['sha256'] == manifest['trace']['sha256'],
             'workload/manifest trace hash differs')
        need((trace['rate'], trace['seed'], trace['arrival_seed'], trace['sampling_seed'], trace['arrival_window_s']) ==
             (float(rate), 701, 701, 20260907, 100.), 'trace rate/seed/window differs')
        need(generator.workload_hash(trace) == trace['content_pairing_sha256'] == workload['content_pairing_sha256'],
             'full prompt/content digest differs')
        need(generator.digest(trace['source_pool_indices']) == workload['source_indices_sha256'], 'source-index hash differs')
        original_trace = parent.build_trace(spec, '14b', 'sharegpt', Decimal(rate), 701, group, sampling_seed)[1]
        need(generator.digest(original_trace) == trace['source_300s_trace']['sha256'], 'frozen 300s regeneration differs')
        expected = generator.prefix_trace(original_trace, trace['source_300s_trace'])
        need(generator.encode(expected) == Path(workload['trace_reference']['path']).read_bytes(),
             'executed trace bytes differ from the original frozen generator prefix')
        n = trace['n_requests']
        need(n == len(trace['requests']) == len(trace['prompts']) == len(trace['source_pool_indices']), 'unaligned trace')
        expected_indices = (group['order'] * math.ceil(n / len(group['order'])))[:n]
        need(trace['source_pool_indices'] == expected_indices, 'content sampling order differs')
        for index, (request, prompt, selected_index) in enumerate(zip(trace['requests'], trace['prompts'], expected_indices)):
            record = group['records'][selected_index]
            need(request['idx'] == index and prompt == record['prompt'] and
                 request['prompt_len'] == record['input_tokens'] and request['output_len'] == record['output_tokens'],
                 'a prompt or requested output was modified')
        workloads[rate], traces[rate] = workload, trace
        rate_rows[rate] = dict(rate_rps=float(rate), n_requests=n, workload=workload_ref, manifest=manifest_ref,
            trace=workload['trace_reference'], content_pairing_sha256=trace['content_pairing_sha256'],
            source_indices_sha256=workload['source_indices_sha256'], request_array_sha256=generator.digest(trace['requests']),
            prompt_array_sha256=generator.digest(trace['prompts']), arrival_seed=701, sampling_seed=20260907,
            frozen_generator_full_byte_reproduction_passed=True, unmodified_pool_content_verified=True,
            source300_reference=trace['source_300s_trace'],
            source300_local_file_present=Path(workload['source_300s_local']['path']).is_file(),
            source300_hash_verified_by_in_memory_frozen_regeneration=True, observed_cells=[],
            observed_by_host={}, observed_five_system_complete_by_host={}, pending_cells=[])
    for host in ('A', 'B', 'C'):
        for row in nodes[host]:
            need(row.get('measurement_valid') is True and row.get('strict_slo_recomputed') is True, 'invalid observed cell')
            rate = generator.number(row['rate_rps'])
            trace, workload = traces[rate], workloads[rate]
            if host != 'B':
                raw = checked(row['audit_reference'])
                need(all(row.get(key) == value for key, value in raw.items()), 'observation changed after audited publication')
                need(row['materialization_manifest'] == workload['materialization_manifest'], 'actual workload manifest differs')
            actual_trace = checked(row['trace_reference'])
            need(row['trace_sha256'] == workload['trace_sha256'] and actual_trace == trace,
                 'cross-host/system trace pairing differs')
            need(row['content_pairing_sha256'] == workload['content_pairing_sha256'] and
                 row['source_indices_sha256'] == workload['source_indices_sha256'] and
                 row['sampling_seed'] == 20260907 and row['seed'] == 701 and row['arrival_window_s'] == 100.,
                 'row content/arrival seed identity differs')
            cp = checked(row['checkpoint'])
            for field in ('cell_id', 'system', 'repeat', 'trace_sha256', 'content_pairing_sha256',
                          'source_indices_sha256', 'seed', 'sampling_seed', 'arrival_window_s'):
                need(cp['row'][field] == row[field], 'executed checkpoint row differs: ' + field)
            release_ref = cp['release']
            release_verified = None
            if host == 'B':
                release = checked(release_ref)
                checked(cp['declaration'])
                need(cp['row'] in release['rows'] and cp['declaration'] == release['declaration'],
                     'historical B declaration/actual release differs')
                release_verified = True
            summary = checked(row['summary'])
            receipt = checked(row['receipt'])
            need(receipt['trace_sha256'] == summary['trace_sha256'] == row['trace_sha256'], 'actual receipt trace differs')
            request_path = Path(row['raw_requests']['path'])
            need(sha(request_path) == row['raw_requests']['sha256'], 'raw request hash changed')
            config_path = request_path.parent / 'runtime_config.json'
            config_ref = dict(path=str(config_path), sha256=cp['artifacts'][str(config_path)])
            config = checked(config_ref)
            scale = {'A': .5, 'B': 1., 'C': 2.}[host]
            ttft, tpot = 5 * scale, .15 * scale
            need((config['slo_scale'], config['slo_ttft_s'], config['slo_tpot_s'], config['arrival_window_s']) ==
                 (scale, ttft, tpot, 100.), 'executed controller SLO differs')
            need(summary['fixed_window']['effective_slo_s'] == dict(ttft=ttft, tpot=tpot), 'actual effective SLO differs')
            with request_path.open() as stream:
                requests = list(csv.DictReader(stream))
            actual = {request['request_id']: request for request in requests}
            need(len(actual) == len(requests) == trace['n_requests'] and
                 set(actual) == {str(i) for i in range(trace['n_requests'])}, 'raw arrival denominator differs')
            epoch = summary['fixed_window']['arrival_epoch_s']
            offsets, deadline_errors = [], []
            for index, expected in enumerate(trace['requests']):
                request = actual[str(index)]
                need(int(request['prompt_len']) == expected['prompt_len'] and int(request['output_len']) == expected['output_len'],
                     'actual requested lengths differ')
                offsets.append(abs(float(request['planned_arrival_s']) - epoch - expected['arrival_s']))
                deadline_errors.append(abs(float(request['request_deadline_s']) - float(request['planned_arrival_s']) - 120.))
            need(max(offsets) < 1e-5 and max(deadline_errors) < 1e-5, 'actual arrival offset/deadline differs')
            auto = automatic.get((row['cell_id'], row.get('engineering_attempt', 1)))
            auto_verified = bool(auto and auto['status'] == 'passed' and auto['audit_reference'] == row.get('audit_reference'))
            result = dict(cell_id=row['cell_id'], measurement_host=host, reference_only=host == 'B',
                system=row['system'], rate_rps=row['rate_rps'], repeat=row['repeat'], n_requests=trace['n_requests'],
                passed=True, checkpoint=row['checkpoint'], trace=row['trace_reference'], raw_requests=row['raw_requests'],
                summary=row['summary'], receipt=row['receipt'], runtime_config=config_ref,
                actual_slo_scale=scale, actual_ttft_s=ttft, actual_tpot_s=tpot, actual_arrival_window_s=100.,
                max_frozen_arrival_offset_error_s=max(offsets), max_request_deadline_error_s=max(deadline_errors),
                existing_automatic_core_crosscheck_passed=auto_verified if host != 'B' else None,
                historical_B_release=release_ref if host == 'B' else None,
                historical_B_release_declaration_verified=release_verified)
            cell_checks.append(result)
            rate_rows[rate]['observed_cells'].append(row['cell_id'])
    systems = set(generator.SYSTEMS)
    for host, cap in (('A', 1.), ('C', 2.)):
        actual_ids = {row['cell_id'] for row in nodes[host]}
        for rate, workload in workloads.items():
            if float(rate) > cap:
                continue
            for system in generator.SYSTEMS:
                repeats = (1, 2) if system == 'pdblend' and float(rate) == cap else (1,)
                for repeat in repeats:
                    scale_label = '0.5' if host == 'A' else '2'
                    cid = f'slo-rate-14b-sharegpt-20260909-v1-{host}-14b-sharegpt-r{rate}-s701-w100-{system}-slo{scale_label}-repeat{repeat}'
                    if cid in actual_ids:
                        continue
                    point = dict(cell_id=cid, node=host, system=system, rate_rps=float(rate), repeat=repeat,
                        status='not_in_observed_snapshot', input_pairing_verified=True,
                        trace=workload['trace_reference'], materialization_manifest=workload['materialization_manifest'],
                        actual_runtime_config_verified=False, actual_measurement_verified=False)
                    pending.append(point)
                    rate_rows[rate]['pending_cells'].append(cid)
    for rate, entry in rate_rows.items():
        for host in ('A', 'B', 'C'):
            values = [row for row in nodes[host] if generator.number(row['rate_rps']) == rate]
            entry['observed_by_host'][host] = dict(cell_count=len(values),
                systems=sorted({row['system'] for row in values}),
                repeat2_cells=[row['cell_id'] for row in values if row['repeat'] == 2])
            entry['observed_five_system_complete_by_host'][host] = {row['system'] for row in values} == systems
        entry['all_observed_exact_trace_and_content_pairing_passed'] = True
    longest = traces['2']
    for rate, trace in traces.items():
        n = trace['n_requests']
        need(all(trace[key] == longest[key][:n] for key in ('prompts', 'source_shapes', 'source_pool_indices')),
             'across-rate content prefix differs')
    result = dict(schema='slo-rate-trace-pairing-preclose-v1', created_s=time.time(), passed=True,
        verification_code=ref(__file__), sources=dict(snapshots=ref(SNAPSHOTS / 'snapshots.json'),
            source_records=metadata, frozen_inputs=inputs_ref, frozen_group=inputs['dependencies']['group'],
            frozen_generator100=inputs['dependencies']['generator100'], frozen_generator300=inputs['dependencies']['generator300'],
            B_reference=ref(ROOT / 'B/reference.json'), B_immutable_source=b_reference['source']),
        observed_counts={host:len(values) for host,values in nodes.items()}, observed_cell_count=len(cell_checks),
        declared_new_measurement_counts={'A':21,'C':41}, pending_new_measurement_count=len(pending),
        complete_new_measurement_grid=not pending, rates=list(rate_rows.values()), observed_cell_checks=cell_checks,
        pending_input_checks=pending, source_pool_revalidated_from_frozen_loader=True,
        frozen_group_payload_matches_original_loader=True, arrival_seed=701, content_sampling_seed=20260907,
        one_arrival_realization_no_independent_seed_CI=True, across_rate_content_prefix_verified=True,
        maximum_observed_arrival_offset_error_s=max(item['max_frozen_arrival_offset_error_s'] for item in cell_checks),
        actual_SLO_configuration_by_host={'A':{'ttft_s':2.5,'tpot_s':.075},
            'B':{'ttft_s':5.,'tpot_s':.15},'C':{'ttft_s':10.,'tpot_s':.3}},
        actual_SLO_evidence='Checkpoint-hash-verified runtime_config.json plus summary fixed_window effective thresholds; not counterfactual rescoring',
        limitations=['Pending inputs prove declared pairing only; execution/config checks await published measurement',
            'Raw bench verifies request identities, requested lengths, planned arrival offsets and deadlines; prompt bodies are bound through the frozen executed trace',
            'Same content and seeds do not remove physical-host confounding; B is an immutable historical reference',
            'No qualification, runtime, historical data, or reports/current writes; frozen generation was replayed in memory only'],
        CPU_only=True, GPU_operations=False)
    PREFIX.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['本次冻结截面：A 21/21、B 31 条历史参考、C '+str(len(nodes['C']))+'/41；'+str(len(pending))+' 个新测量点尚未包含在截面中。已观察 '+str(len(cell_checks))+' 条全部通过 trace/content/到达时序/实际 SLO 配置核对。',
        '', '8 个 rate（0.25 至 2.0，步长 0.25）的真实 workload.json 与固定 manifest payload 哈希、trace SHA 均验证通过。使用原冻结 pool loader 与生成器，只在内存中重建 300s 原始 trace 并取原 100s 前缀，逐字节复现现有 trace；prompt、请求输入/输出长度、source indices 和内容哈希全部一致。arrival seed=701，内容采样 seed=20260907。',
        '', '| rate (req/s) | N | A 已观测/五系统齐全 | B 已观测/五系统齐全 | C 已观测/五系统齐全 |', '|---:|---:|---|---|---|']
    for entry in rate_rows.values():
        cells=[]
        for host in ('A','B','C'):
            count=entry['observed_by_host'][host]['cell_count']
            cells.append(str(count)+' / '+('是' if entry['observed_five_system_complete_by_host'][host] else '否') if count else '不在该机已观测范围')
        lines.append('| '+str(entry['rate_rps'])+' | '+str(entry['n_requests'])+' | '+' | '.join(cells)+' |')
    lines.extend(['', '同 rate 的所有已观察五系统及 PDB repeat2 使用完全相同 trace 字节与内容采样；各 rate 的内容序列也是同一冻结采样序列的前缀，N 随到达率变化。所有已观察请求的 planned arrival offset 相对冻结 trace 最大误差 '+format(result['maximum_observed_arrival_offset_error_s'],'.3g')+' 秒，满足 1e-5 秒容差。',
        '', '实际生效阈值从每条已测 runtime_config.json 读取，并核对 checkpoint SHA 与 summary fixed_window：A=TTFT 2.5s / TPOT 0.075s；B=5s / 0.15s；C=10s / 0.3s；arrival window 均为 100s。这是实际运行配置验证，不是将 1×结果事后重新评分。', '', '尚未包含的测量点：'])
    lines.extend('- '+point['node']+' / '+point['system']+' / '+str(point['rate_rps'])+' req/s：冻结 trace 与 manifest 输入配对已确认；实际测量与实际运行配置尚未在本截面观察。' for point in pending)
    lines.extend(['', '这份报告使用固定快照，不随后台后续观测改写。相同 seed/content 不消除机器差异，不能从 A/B/C 的差异推出纯 SLO 倍率因果效果，也不形成独立 seed 置信区间。原始 bench 检查身份、长度与时序；prompt 内容通过已执行且哈希固定的 trace 绑定。'])
    PREFIX.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({key:result[key] for key in ('passed','observed_counts','observed_cell_count','pending_new_measurement_count','maximum_observed_arrival_offset_error_s')}))


if __name__ == '__main__':
    main()
