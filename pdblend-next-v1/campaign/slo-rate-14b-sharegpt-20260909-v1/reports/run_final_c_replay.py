"""One-shot CPU evidence collection; each immutable C checkpoint gets a fresh process."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-raw-replay-C-001'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    mirror = ROOT / 'staging/full-mirror-C-final-001/status.json'
    state = read(mirror)
    assert state['complete'] and state['scope_complete'] and not state['errors'] and not state['unavailable']
    terminal = ROOT / 'C/run-002/status.json'
    state = read(terminal)
    assert state['complete'] and state['phase'] == 'complete' and not state['node_lease_held']
    source = ROOT / 'C/observations.json'
    rows = read(source)
    assert len(rows) == len({row['cell_id'] for row in rows}) == 41
    assert all(row['measurement_valid'] and row['strict_slo_recomputed'] for row in rows)
    assert not OUT.exists(), 'fresh evidence directory required'
    OUT.mkdir()
    worker = ROOT / 'reports/replay_audit_worker.py'
    manifest = dict(schema='slo-rate-full-raw-replay-inputs-v1', node='C', created_s=time.time(),
        auditor=ref(ROOT / 'audit.py'), support=ref(ROOT / 'slo_support.py'), worker=ref(worker),
        orchestrator=ref(__file__), source_observations=ref(source), terminal=ref(terminal),
        full_nonweight_mirror=ref(mirror),
        cells=[dict(cell_id=row['cell_id'], system=row['system'], rate_rps=row['rate_rps'],
                    repeat=row['repeat'], checkpoint=row['checkpoint'], original_audited=row['audit_reference'])
               for row in rows],
        scope='Complete measurement audit replay only; native qualification and weight reconstruction remain distinct',
        GPU_operations=False)
    save(OUT / 'inputs.json', manifest)
    manifest_ref = ref(OUT / 'inputs.json')
    pinned = [v for v in manifest.values() if isinstance(v, dict) and 'path' in v] + [manifest_ref]

    def unchanged():
        return all(sha(item['path']) == item['sha256'] for item in pinned)

    assert unchanged()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')

    def launch(item):
        directory = OUT / 'cells' / item['cell_id']
        argv = ['nice', '-n', '15', 'python3', '-B', str(worker), '--manifest', manifest_ref['path'],
                '--manifest-sha256', manifest_ref['sha256'], '--cell-id', item['cell_id'], '--out', str(directory)]
        started = time.time()
        process = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=180)
        save(directory / 'launcher.json', dict(argv=argv, exitcode=process.returncode, started_s=started,
             finished_s=time.time(), stdout=process.stdout, stderr=process.stderr))
        result = read(directory / 'result.json')
        return item, result, ref(directory / 'result.json')

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(launch, manifest['cells']))
    sources_unchanged = unchanged()
    comparisons, cells = [], []
    for item, result, result_ref in results:
        comparisons.extend(dict(cell_id=item['cell_id'], **value) for value in result.get('core_numeric_comparison', []))
        cells.append(dict(cell_id=item['cell_id'], system=item['system'], rate_rps=item['rate_rps'], repeat=item['repeat'],
            **{key: result.get(key) for key in ('complete', 'passed', 'full_result_exact_equal', 'elapsed_s',
                                               'missing_files', 'hash_mismatches', 'error')}, result=result_ref))
    passed = sum(item['passed'] is True for item in cells)
    exact = sum(item['full_result_exact_equal'] is True for item in cells)
    summary = dict(schema='slo-rate-complete-measurement-audit-replay-summary-v1', node='C', finished_s=time.time(),
        source_manifest=manifest_ref, cell_count=len(cells),
        complete=all(item['complete'] is True for item in cells),
        passed=passed == exact == 41 and sources_unchanged and len(comparisons) == 1066,
        passed_cells=passed, full_result_exact_equal_cells=exact,
        unique_worker_pids=len({result['pid'] for _, result, _ in results}),
        core_numeric_comparisons=len(comparisons),
        exact_numeric_comparisons=sum(item.get('exact_equal') is True for item in comparisons),
        numeric_mismatches=[item for item in comparisons if item.get('passed') is not True],
        missing_files=sorted({path for item in cells for path in item['missing_files']}),
        hash_mismatches=[m for item in cells for m in item['hash_mismatches']],
        source_hashes_unchanged_before_and_after=sources_unchanged, CPU_only=True, GPU_operations=False,
        full_qualification_replayed=False, weight_reconstruction_performed=False,
        scope='Complete audit.audit re-execution in one fresh process per measurement, including CP artifact hashes, '
              'runtime/effective SLO, identities, cleanup, journal-based service classifications, exact work denominator, '
              'all8 raw power and original measurement auditor; separate from qualification/weight reconstruction',
        cells=cells)
    save(OUT / 'core-numeric-comparisons.json', comparisons)
    with (OUT / 'core-numeric-comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cell_id', 'field', 'original', 'replayed', 'exact_equal', 'passed'], extrasaction='ignore')
        writer.writeheader()
        for item in comparisons:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in item.items()})
    save(OUT / 'summary.json', summary)
    (OUT / 'README.md').write_text('C 的 41 条正式 checkpoint 已在 B 上逐条调用本轮 audit.audit；每条使用独立进程，仅 CPU。\n\n'
        f'通过 {passed}/41；完整输出对象逐项精确相等 {exact}/41；核心数字及数组精确相等 '
        f"{summary['exact_numeric_comparisons']}/{len(comparisons)}。缺失原始文件 {len(summary['missing_files'])} 项。\n\n"
        '重放包含完整测量审计所需的 checkpoint 哈希、实际生效 SLO、请求与 journal 分类、清理证明以及全 8 卡原始功耗积分。'
        '冻结的源观测、审计代码、worker、终态和镜像回执在重放前后哈希不变；每条结果与必需原始文件索引保存在 cells/。\n\n'
        '这份证据不声称重新执行了 GPU 资格检查或权重重建。现场原始数据、原 audited JSON、runtime 和 reports/current 均未改写。\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'cells'}, indent=2))
    print(json.dumps({'summary': ref(OUT / 'summary.json')}))
    assert summary['passed'] and summary['exact_numeric_comparisons'] == 1066


if __name__ == '__main__':
    main()
