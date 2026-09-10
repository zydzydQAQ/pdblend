"""Assemble immutable completed-group bundles; full scope is a hard gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEALED = HERE / 'reports/completed/sealed-groups-v2'
CURRENT = HERE / 'reports/current'
SYSTEMS = {'pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve'}
DATASETS = ('alpaca', 'sharegpt', 'longbench')
METRICS = ('energy_j', 'slo_attainment', 'ttft_avg_s', 'tpot_avg_s',
    'completed_work_throughput_rps', 'generated_token_throughput_tps', 'gpu_util')
EXPECTED_GROUPS = {(model, dataset, 'C' if model == '7b' else
    'B' if model == '32b' or dataset == 'sharegpt' else 'Anew20260909')
    for model in ('7b', '14b', '32b') for dataset in DATASETS}
NODE_ENVIRONMENTS = {
    'C': ('47.106.163.29', 'iZwz9gfq11hx1sbob59yrgZ'),
    'B': (None, 'iZwz9i5bte3xkpmcoes3t2Z'),
    'Anew20260909': ('120.79.123.62', 'iZwz9274emxme9019d2sjgZ'),
}
NODE_LEASE = '/root/workspace/pdblend/new-results/campaigns/node-experiment.lock'
ANALYSIS_SOURCE_SHA256 = '676d6abd308d9efaf5c1a16b73c9805535fa77ebb727f704d002b2330ec0b819'


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def snapshot(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = identity(os.fstat(stream.fileno()))
        data = stream.read()
        need(before == identity(os.fstat(stream.fileno())) == identity(path.stat()), 'input changed during snapshot: ' + str(path))
    return data, dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest())


def sha(path):
    return snapshot(path)[1]['sha256']


def read(path):
    data, reference = snapshot(path)
    return json.loads(data), reference


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_member(name):
    name = Path(name)
    need(not name.is_absolute() and '..' not in name.parts and str(name) not in ('', '.'), 'unsafe archive member')
    return name


def validate_science(result, rows):
    need(result['complete'] is True and not result['metric_audit_errors'], 'sealed scientific scope is incomplete')
    checks = []
    for group in result['groups']:
        need(group['complete'] and group['pdb_boundary_complete'], 'uncompleted scientific group')
        model, dataset, node = group['model'], group['dataset'], group['node']
        need((model, dataset, node) in EXPECTED_GROUPS, 'group is assigned to the wrong physical host')
        step, cap = Decimal(str(group['rate_step_rps'])), Decimal(str(group['cap_rate_rps']))
        need(cap > 0 and cap % step == 0 and Decimal(str(group['rate_start_rps'])) == step, 'rate axis is not a fixed grid from one step')
        rates = {step * n for n in range(1, int(cap / step) + 1)}
        selected = [r for r in rows if (r['model'], r['dataset']) == (model, dataset)]
        need({(r['system'], Decimal(r['rate_rps'])) for r in selected}
            == {(system, rate) for system in SYSTEMS for rate in rates}
            and len(selected) == len(rates) * 5, 'missing or duplicate system/rate coordinate')
        for row in selected:
            need(row['measurement_host'] == node, 'cross-host aggregate')
            for metric in METRICS:
                low, mean, high = (float(row[metric + suffix]) for suffix in ('_min', '_mean', '_max'))
                need(all(math.isfinite(x) for x in (low, mean, high)) and low <= mean <= high
                    and int(row[metric + '_n']) >= 1, 'missing or invalid metric: ' + metric)
        group_observations = [o for o in result['observations'] if
            (o['model'], o['dataset']) == (model, dataset)]
        need(all(o['measurement_host'] == node for o in group_observations), 'cross-host raw observation')
        observations = [o for o in group_observations if
            Decimal(str(o['rate_rps'])) in rates and o.get('measurement_purpose') != 'metric_supplement']
        for rate in rates:
            paired = [o for o in observations if Decimal(str(o['rate_rps'])) == rate]
            need({o['system'] for o in paired} == SYSTEMS, 'missing same-host comparison system')
            need(len({o['trace_sha256'] for o in paired}) == 1, 'comparison trace bytes differ')
            need(len({(o['seed'], o['n_expected'], o['slo_ttft_s'], o['slo_tpot_s']) for o in paired}) == 1,
                'comparison seed, offered work, or fixed SLO differs')
            pdb = [o for o in paired if o['system'] == 'pdblend']
            need(all(o['measurement_valid'] and o['work_complete'] for o in pdb), 'PDB boundary uses incomplete work')
            if rate < cap:
                need(all(o['slo_attainment'] >= .9 for o in pdb), 'an earlier complete PDB crossing was skipped')
            else:
                need(len(pdb) >= 2 and any(o['slo_attainment'] < .9 for o in pdb), 'first crossing lacks its confirmation')
        for observed in observations:
            need(observed['measurement_valid'], 'a normal observation has an invalid physical measurement')
            need(len(observed['energy_per_gpu_j']) == len(observed['gpu_util_per_gpu']) == 8,
                'measurement omits a GPU')
            need(math.isclose(sum(observed['energy_per_gpu_j']), observed['energy_j'], rel_tol=1e-10, abs_tol=1e-6)
                and math.isclose(sum(observed['gpu_util_per_gpu']) / 8, observed['gpu_util'], rel_tol=1e-10, abs_tol=1e-10),
                'eight-GPU totals differ')
            if observed.get('independent_reclassification_only'):
                classification = observed['baseline_service_failure']
                need((model, node, observed['system']) == ('7b', 'C', 'ecoserve')
                    and classification['classification'] == 'baseline_explicit_native_admission_queue_full'
                    and classification['native_rejections_are_not_timeouts']
                    and classification['actual_request_timeouts'] == observed['request_timeouts']
                    and classification['native_rejections'] + classification['actual_request_timeouts'] == observed['failed_requests']
                    and observed['failure_class'] == 'independently_diagnosed_capacity_rejection'
                    and classification['full_request_denominator_preserved'], 'native refusal label is inaccurate')
        checks.append(dict(model=model, dataset=dataset, node=node, step_rps=str(step), cap_rps=str(cap),
            coordinates=len(selected), observed_normal_repeats=len(observations), numeric_metrics=len(METRICS)))
    return checks


def verify_group(directory):
    directory = Path(directory)
    manifest, manifest_ref = read(directory / 'manifest.json')
    need(manifest['schema'] == 'completed-rate-scale-bundle-v1', 'unknown sealed-group manifest')
    files = dict(manifest['files'])
    files['manifest.json'] = manifest_ref['sha256']
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    need(actual == set(files), 'sealed group member set changed')
    for name, digest in files.items():
        path = directory / safe_member(name)
        need(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve())
            and sha(path) == digest, 'sealed group file changed: ' + name)
    result, result_ref = read(directory / 'results.json')
    need(result_ref['sha256'] == files['results.json'], 'sealed result changed during verification')
    data, summary_ref = snapshot(directory / 'summary.csv')
    need(summary_ref['sha256'] == files['summary.csv'], 'sealed summary changed during verification')
    checks = validate_science(result, list(csv.DictReader(io.StringIO(data.decode()))))
    need(len(checks) == 1 and directory.name == checks[0]['model'] + '-' + checks[0]['dataset'],
        'sealed directory does not contain its named group')
    for group in result['groups']:
        for suffix in ('png', 'pdf'):
            need((directory / (group['model'] + '-' + group['dataset'] + '.' + suffix)).stat().st_size > 1000,
                'scientific figure absent')
    archive = directory.with_suffix('.zip')
    archive_before = sha(archive)
    with zipfile.ZipFile(archive) as package:
        expected = {str(Path(directory.name) / name) for name in files}
        need(len(package.namelist()) == len(expected) and set(package.namelist()) == expected, 'source ZIP member set differs')
        for name, digest in files.items():
            need(hashlib.sha256(package.read(str(Path(directory.name) / name))).hexdigest() == digest,
                'source ZIP member differs: ' + name)
    need(sha(archive) == archive_before, 'source ZIP changed during verification')
    return dict(directory=str(directory), manifest=manifest_ref, result=result_ref, files=files, checks=checks,
        archive=dict(path=str(archive), sha256=archive_before))


def clean_terminal(state):
    return bool(state.get('complete') is True and state.get('finished_s')
        and state.get('node_lease_held') is False and not state.get('error'))


def aggregate_signature(rows):
    signature = {}
    for row in rows:
        key = row['model'], row['dataset'], row['system'], Decimal(row['rate_rps'])
        need(key not in signature, 'duplicate aggregate coordinate')
        signature[key] = (row['measurement_host'], tuple(float(row[metric + suffix])
            for metric in METRICS for suffix in ('_min', '_mean', '_max', '_n')))
    return signature


def full_scope_gate(monitor, result, remaining, ledger):
    need(monitor.get('scope_complete') is True and monitor.get('finished_s')
        and monitor.get('completion_scope') == 'five_systems' and monitor.get('groups_complete') == 9
        and monitor.get('metric_audit_errors') == 0 and not monitor.get('hydration_active'),
        'monitor has not reached a clean full-scope terminal')
    need(result.get('scope_complete') is True and result.get('complete') is True
        and len(result['groups']) == 9 and all(g['complete'] and g['pdb_boundary_complete'] for g in result['groups'])
        and not result['metric_audit_errors'], 'nine scientific groups have not completed raw audit')
    need(remaining['total']['constant'] == 0 and not remaining['total']['variables']
        and not remaining['awaiting_evidence'] and remaining['raw_metric_audit_errors'] == 0, 'remaining measurements or raw evidence exist')
    need(ledger['schema'] == 'uniform-setup-energy-ledger-v1' and not ledger['pending_or_invalid_evidence'],
        'setup energy still has pending or invalid evidence')
    expected = {(g['model'], g['dataset'], g['node']) for g in result['groups']}
    need(expected == EXPECTED_GROUPS, 'nine-group physical assignment differs from the declared scope')
    need(all(monitor['completion_checks'].get(field) is True for field in
        ('raw_scope_complete', 'scope_complete', 'five_system_complete')),
        'monitor completion gates are inconsistent')
    checks = monitor['completion_checks']['supervisor_checks']
    need(len(checks) == 9 and {(x['model'], x['dataset'], x['node']) for x in checks} == expected
        and all(x['pipeline_terminal'] is True for x in checks), 'one or more supervisors are not terminal')
    return checks


def terminal_probe_script(request):
    return 'REQUEST = ' + repr(request) + '\n' + r'''
import fcntl, hashlib, json, os, socket, time
from pathlib import Path
def require(ok, message):
    if not ok: raise ValueError(message)
def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
def checked(ref):
    path = Path(ref['path'])
    with path.open('rb') as stream:
        before = signature(os.fstat(stream.fileno())); data = stream.read()
        require(before == signature(os.fstat(stream.fileno())) == signature(path.stat()), 'actual terminal changed during read')
    require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'actual terminal differs from mirrored reference')
    return json.loads(data)
def owner_exited(state):
    require(isinstance(state.get('pid'), int) and state['pid'] > 0 and state.get('startticks'), 'terminal owner identity absent')
    try: fields = Path('/proc/' + str(state['pid']) + '/stat').read_text().rpartition(')')[2].split()
    except FileNotFoundError: return True
    return fields[0] == 'Z' or fields[19] != str(state['startticks'])
require(socket.gethostname() == REQUEST['expected_hostname'], 'actual physical hostname differs')
states = []
for ref in REQUEST['terminals']:
    state = checked(ref)
    require(state.get('complete') is True and state.get('finished_s') and state.get('node_lease_held') is False and not state.get('error'), 'actual supervisor/cell is not clean and terminal')
    require(owner_exited(state), 'actual terminal owner is still running')
    states.append(dict(reference=ref, pid=state['pid'], startticks=state['startticks'], owner_exited=True))
locks = []
for path in REQUEST['locks']:
    with Path(path).open('r+') as stream:
        before = signature(os.fstat(stream.fileno()))
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: require(before == signature(Path(path).stat()), 'lease path replaced during probe')
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    locks.append(dict(path=path, free=True, probe_released=True))
for ref in REQUEST['terminals']: checked(ref)
print(json.dumps(dict(schema='readonly-final-host-verification-v1', passed=True, node=REQUEST['node'],
    hostname=socket.gethostname(), checked_s=time.time(), terminals=states, locks=locks, gpu_actions=False)))
'''


def probe_actual_hosts(requests):
    need(set(requests) == set(NODE_ENVIRONMENTS), 'all three actual hosts must be checked')
    def run(item):
        node, request = item
        address, expected_hostname = NODE_ENVIRONMENTS[node]
        need(request['node'] == node and request['expected_hostname'] == expected_hostname
            and NODE_LEASE in request['locks'] and request['terminals'], 'actual-host probe request is incomplete')
        argv = [sys.executable, '-B', '-'] if address is None else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'root@' + address, 'python3', '-B', '-']
        output = subprocess.run(argv, input=terminal_probe_script(request), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35)
        need(output.returncode == 0, 'actual host terminal/lease verification failed on ' + node + ': ' + output.stderr[-1500:])
        proof = json.loads(output.stdout)
        need(proof['passed'] is True and proof['node'] == node, 'actual host probe returned an invalid result')
        return node, proof
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(run, requests.items()))


def assemble(*, scope, destination, sealed=SEALED, current=CURRENT,
             monitor_path=HERE / 'monitor-status.json', terminal_path=None, terminal_proof_path=None):
    destination, sealed, current = Path(destination), Path(sealed), Path(current)
    archive = destination.with_suffix('.zip')
    need(scope in ('7b', 'all9') and not destination.exists() and not archive.exists(), 'fresh supported bundle output required')
    inputs, terminal_documents = {}, {}
    def remember(path):
        data, reference = snapshot(path)
        need(reference['path'] not in inputs or inputs[reference['path']] == reference['sha256'],
            'input changed between reads: ' + reference['path'])
        inputs[reference['path']] = reference['sha256']
        return data, reference
    models = ('7b',) if scope == '7b' else ('7b', '14b', '32b')
    if scope == '7b':
        need(terminal_path is not None, 'explicit final C pipeline status required for the 7B bundle')
        data, reference = remember(terminal_path); state = json.loads(data)
        need(clean_terminal(state) and state['scope'] == 'five_systems' and state['node'] == 'C' and state['model'] == '7b'
            and set(state['datasets']) == set(DATASETS) and len(state['observations']) == 64
            and all(state['group_decisions'][d]['phase'] == 'complete' for d in DATASETS), '7B pipeline not fully complete and clean')
        terminal_documents['C-7b-terminal.json'] = data
        last_data, last_ref = remember(state['last_cell_status']['path'])
        need(last_ref == state['last_cell_status'] and clean_terminal(json.loads(last_data)),
            'last C cell is not independently bound and clean')
        terminal_documents['C-7b-last-cell.json'] = last_data
        if terminal_proof_path is not None:
            data, _ = remember(terminal_proof_path); proof = json.loads(data)
            need(proof['status'] == reference and proof['complete'] is True
                and proof['pipeline_owner_exited'] is True and proof['last_cell_owner_exited'] is True
                and proof['node_lease_free'] is True and proof['node_lease_probe_released'] is True
                and proof['last_cell_status'] == last_ref and proof['observations'] == 64
                and len(proof['native_idle']) == len({x['id'] for x in proof['native_idle']}) == 8
                and all(x[field] == 0 for x in proof['native_idle'] for field in ('active', 'running', 'waiting')),
                'C terminal physical verification differs')
            terminal_documents['C-7b-physical-verification.json'] = data
    else:
        monitor_data, monitor_ref = remember(monitor_path); monitor = json.loads(monitor_data)
        result_data, _ = remember(current / 'results.json'); result = json.loads(result_data)
        remaining_data, _ = remember(current / 'remaining-work.json')
        ledger_data, _ = remember(current / 'setup-energy-ledger.json')
        checks = full_scope_gate(monitor, result, json.loads(remaining_data), json.loads(ledger_data))
        summary_data, _ = remember(current / 'summary.csv')
        global_rows = list(csv.DictReader(io.StringIO(summary_data.decode())))
        validate_science(result, global_rows)
        terminal_documents['monitor-terminal.json'] = monitor_data
        host_requests = {node: dict(node=node, expected_hostname=value[1], terminals=[], locks=[NODE_LEASE])
            for node, value in NODE_ENVIRONMENTS.items()}
        for index, check in enumerate(checks):
            if check['path']:
                data, state_ref = remember(check['path']); state = json.loads(data)
                plan_data, plan_ref = remember(state['plan']['path']); plan = json.loads(plan_data)
                need(plan_ref == state['plan'], 'supervisor plan does not match its frozen identity')
                need(clean_terminal(state) and state.get('scope') in (None, 'five_systems')
                    and state.get('model') == check['model'] and state.get('node') == check['node']
                    and check['dataset'] in state.get('datasets', plan.get('dataset_order', []))
                    and state['group_decisions'][check['dataset']]['phase'] == 'complete',
                    'supervisor state changed after final monitor gate')
                group = next(g for g in result['groups'] if
                    (g['model'], g['dataset'], g['node']) == (check['model'], check['dataset'], check['node']))
                need(state['declaration'] == group['declaration'], 'supervisor declaration differs from the completed group')
                terminal_documents[f'supervisor-{index + 1:02d}.json'] = data
                terminal_documents[f'supervisor-{index + 1:02d}-plan.json'] = plan_data
                last_data, last_ref = remember(state['last_cell_status']['path'])
                need(last_ref == state['last_cell_status'] and clean_terminal(json.loads(last_data)),
                    'final supervisor has no bound clean last cell')
                terminal_documents[f'supervisor-{index + 1:02d}-last-cell.json'] = last_data
                request = host_requests[check['node']]
                for reference in (state_ref, last_ref):
                    if reference not in request['terminals']: request['terminals'].append(reference)
                for key, value in plan.items():
                    if key.endswith('supervisor_lock') and isinstance(value, str) and value not in request['locks']:
                        request['locks'].append(value)
            else:
                need(check['reuse_only'] is True, 'missing non-reuse supervisor proof')
    groups = [verify_group(sealed / (model + '-' + dataset)) for model in models for dataset in DATASETS]
    for group in groups:
        inputs[group['archive']['path']] = group['archive']['sha256']
        for name, digest in group['files'].items():
            inputs[str(Path(group['directory']) / name)] = digest
    checks = [check for group in groups for check in group['checks']]
    need(len(checks) == (3 if scope == '7b' else 9), 'bundle group coverage differs')
    if scope == '7b':
        need(sum(x['coordinates'] for x in checks) == 145 and all(x['node'] == 'C' for x in checks), '7B paired coordinate total differs')
    else:
        sealed_rows = []
        for group in groups:
            data, _ = remember(Path(group['directory']) / 'summary.csv')
            sealed_rows.extend(csv.DictReader(io.StringIO(data.decode())))
        need(aggregate_signature(global_rows) == aggregate_signature(sealed_rows),
            'global aggregates disagree with the completed sealed groups')
        for node, proof in probe_actual_hosts(host_requests).items():
            terminal_documents[node + '-actual-host-verification.json'] = (json.dumps(proof, indent=2) + '\n').encode()
        analysis_path = HERE / 'analyze_completed_v1.py'
        analysis_data, analysis_ref = remember(analysis_path)
        need(analysis_ref['sha256'] == ANALYSIS_SOURCE_SHA256, 'final interpretation source changed')
        namespace = dict(__name__='completed_bundle_interpretation', __file__=str(analysis_path))
        exec(compile(analysis_data, str(analysis_path), 'exec'), namespace)
        analysis_markdown, analysis_csv = namespace['build_analysis'](result, global_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
    for group in groups:
        source = Path(group['directory'])
        shutil.copytree(source, staging / source.name)
        for name, digest in group['files'].items():
            need(sha(staging / source.name / name) == digest, 'copied sealed member differs: ' + name)
    (staging / 'terminal-evidence').mkdir()
    for name, data in terminal_documents.items():
        (staging / 'terminal-evidence' / name).write_bytes(data)
    if scope == 'all9':
        (staging / 'CONCLUSIONS.md').write_text(analysis_markdown)
        (staging / 'comparisons.csv').write_text(analysis_csv)
        global_files = ('measurements.csv', 'summary.csv', 'gpu-details.csv', 'raw-evidence-index.csv',
            'history.csv', 'results.json', 'remaining-work.json')
        (staging / 'global').mkdir(); (staging / 'curves').mkdir(); (staging / 'operations').mkdir()
        for name in global_files:
            data, _ = remember(current / name); (staging / 'global' / name).write_bytes(data)
        for model in models:
            for dataset in DATASETS:
                for suffix in ('png', 'pdf'):
                    name = model + '-' + dataset + '.' + suffix
                    data, _ = remember(current / name); (staging / 'curves' / name).write_bytes(data)
        for base in ('setup-energy-ledger', 'engineering-failures'):
            for suffix in ('csv', 'json'):
                name = base + '.' + suffix
                data, _ = remember(current / name); (staging / 'operations' / name).write_bytes(data)
    body = ['# 已完成的固定步长 Rate Scale 结果', '',
        '各组目录和原始 manifest 保持封存时的原样，组内相对链接可以直接使用。', '',
        '| 模型 | 数据集 | 主机 | 步长 rps | 终点 rps | 系统 × rate 坐标 |',
        '|---|---|---|---:|---:|---:|']
    for row in checks:
        name = row['model'] + '-' + row['dataset']
        body.append(f"| {row['model']} | [{row['dataset']}]({name}/README.md) | {row['node']} | {row['step_rps']} | {row['cap_rps']} | {row['coordinates']} |")
    body += ['', '能耗、SLO attainment、平均 TTFT、平均 TPOT、吞吐（request/s 与 token/s）和利用率均保留原值。拒绝和超时计入完整请求分母；原生队列拒绝单独记录，没有改记为超时。', '',
        '正常新点一次，已有重复保留；曲线阴影是重复实测范围，不是置信区间。首次有效完整单次 PDBlend SLO <90% 的坐标为终点，其确认复测不会撤销首次越界。', '',
        '原始证据索引记录工作区内原始文件位置与 SHA-256；本包包含结果和证据索引，不复制全部模型或 GPU 原始日志。']
    if scope == 'all9':
        body += ['', '[实测结论](CONCLUSIONS.md) 与 [对应数值](comparisons.csv) 比较越界前的最后一个网格点。全模型 CSV 位于 [global](global/summary.csv)，九组曲线位于 curves；准备能耗和工程异常单列于 operations，未加到 serving Energy 曲线。']
    (staging / 'README.md').write_text('\n'.join(body) + '\n')
    save(staging / 'assembly-validation.json', dict(passed=True, scope=scope, checks=checks,
        source_files=inputs, source=remember(__file__)[1], sealed_contents_preserved=True,
        scientific_coordinates=sum(x['coordinates'] for x in checks), gpu_actions=False))
    for path, digest in inputs.items():
        need(sha(path) == digest, 'input changed while assembling: ' + path)
    files = {str(path.relative_to(staging)): sha(path) for path in sorted(staging.rglob('*')) if path.is_file()}
    save(staging / 'manifest.json', dict(schema='combined-fixed-rate-results-v1', scope=scope,
        created_s=time.time(), files=files, source_sealed_groups=[g['manifest'] for g in groups]))
    expected = dict(files, **{'manifest.json': sha(staging / 'manifest.json')})
    temporary_zip = archive.with_name(archive.name + '.pending')
    need(not temporary_zip.exists(), 'previous pending archive must be preserved')
    with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as package:
        for name in sorted(expected):
            package.write(staging / name, str(Path(destination.name) / name))
    with zipfile.ZipFile(temporary_zip) as package:
        need(len(package.namelist()) == len(expected), 'combined archive has duplicate or missing members')
        for name, digest in expected.items():
            need(hashlib.sha256(package.read(str(Path(destination.name) / name))).hexdigest() == digest, 'combined ZIP member mismatch')
    os.rename(staging, destination); os.rename(temporary_zip, archive)
    return dict(scope=scope, path=str(destination), archive=str(archive), archive_sha256=sha(archive),
        groups=len(checks), scientific_coordinates=sum(x['coordinates'] for x in checks), files=len(expected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('7b', 'all9'), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--sealed', type=Path, default=SEALED)
    parser.add_argument('--current', type=Path, default=CURRENT)
    parser.add_argument('--monitor', type=Path, default=HERE / 'monitor-status.json')
    parser.add_argument('--terminal', type=Path)
    parser.add_argument('--terminal-proof', type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(scope=args.scope, destination=args.destination, sealed=args.sealed,
        current=args.current, monitor_path=args.monitor, terminal_path=args.terminal,
        terminal_proof_path=args.terminal_proof)))
