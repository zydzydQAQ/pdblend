"""Seal completed scientific groups from one audited report snapshot."""
import argparse
import copy
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

import report_with_native_overlay_v1 as report

HERE = Path(__file__).resolve().parent
DEST = HERE / 'reports/completed/sealed-groups-v1'
SYSTEMS = set(report.c.SYSTEMS)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(result, rows):
    """Check the delivered grid independently of the queue's completion flag."""
    checks = []
    for group in result['groups']:
        assert group['complete'] and group['pdb_boundary_complete']
        model, dataset, host = group['model'], group['dataset'], group['node']
        step = Decimal(str(group['rate_step_rps']))
        cap = Decimal(str(group['cap_rate_rps']))
        assert cap > 0 and cap % step == 0
        assert Decimal(str(group['rate_start_rps'])) == step
        grid = {step * n for n in range(1, int(cap / step) + 1)}
        values = [r for r in rows if (r['model'], r['dataset']) == (model, dataset)]
        actual = {(r['system'], Decimal(str(r['rate_rps']))) for r in values}
        expected = {(system, rate) for system in SYSTEMS for rate in grid}
        assert actual == expected and len(values) == len(expected)
        for row in values:
            assert row['measurement_host'] == host and row['repeats'] >= 1
            for metric in report.METRICS:
                assert row[metric + '_n'] >= 1
                assert math.isfinite(row[metric + '_mean'])
                assert row[metric + '_min'] <= row[metric + '_mean'] <= row[metric + '_max']
        observations = [o for o in result['observations'] if
            (o['model'], o['dataset'], o['measurement_host']) == (model, dataset, host)
            and Decimal(str(o['rate_rps'])) in grid
            and o.get('measurement_purpose') != 'metric_supplement']
        for rate in grid:
            paired = [o for o in observations if Decimal(str(o['rate_rps'])) == rate]
            assert {o['system'] for o in paired} == SYSTEMS, (model, dataset, rate, 'missing paired system')
            fingerprints = {o.get('content_pairing_sha256') or o['trace_sha256'] for o in paired}
            assert len(fingerprints) == 1, (model, dataset, rate, 'trace mismatch')
            pdb = [o for o in paired if o['system'] == 'pdblend']
            assert all(o['measurement_valid'] and o['work_complete'] for o in pdb)
            if rate < cap:
                assert all(o['slo_attainment'] >= .9 for o in pdb)
            else:
                assert len(pdb) >= 2 and any(o['slo_attainment'] < .9 for o in pdb)
        for observation in observations:
            energy, util = observation['energy_per_gpu_j'], observation['gpu_util_per_gpu']
            assert len(energy) == len(util) == 8
            assert math.isclose(sum(energy), observation['energy_j'], rel_tol=1e-10, abs_tol=1e-6)
            assert math.isclose(sum(util) / 8, observation['gpu_util'], rel_tol=1e-10, abs_tol=1e-10)
        trigger = report.stop_trigger(observations, model, dataset, float(cap))
        assert trigger is not None
        checks.append(dict(model=model, dataset=dataset, host=host, step_rps=str(step),
            cap_rps=str(cap), coordinates=len(expected), normal_observations=len(observations),
            trigger_cell_id=trigger['cell_id'], trigger_slo_attainment=trigger['slo_attainment']))
    return checks


def publish(snapshot, keys, name, snapshot_sha):
    destination = DEST / name
    archive = destination.with_suffix('.zip')
    if destination.exists():
        manifest = json.loads((destination / 'manifest.json').read_text())
        for path, digest in manifest['files'].items():
            assert sha(destination / path) == digest, 'sealed result changed'
    else:
        selected = copy.deepcopy(snapshot)
        for field in ('groups', 'observations', 'historical_observations', 'metric_audit_errors'):
            selected[field] = [v for v in snapshot.get(field, []) if (v['model'], v['dataset']) in keys]
        assert not selected['metric_audit_errors']
        selected.update(complete=True, bundle_scope=sorted(keys), scope_complete=True)
        temporary = Path(tempfile.mkdtemp(prefix='.' + name + '-', dir=DEST))
        try:
            rows = report.export(selected, temporary)
            checks = validate(selected, rows)
            report.plot(selected, rows, temporary)
            # Make group navigation portable within the downloaded archive.
            for readme in temporary.glob('groups/*/README.md'):
                text = readme.read_text().replace(str(temporary) + '/groups/' + readme.parent.name + '/', '')
                readme.write_text(text.replace(str(temporary) + '/', '../../'))
            body = ['# 已完成的固定步长 Rate Scale 实验', '',
                '本结果包仅包含下表中已完成五系统配对与六指标核验的组。', '',
                '| 模型 | 数据集 | 主机 | 步长 rps | 终点 rps | 系统 × rate 坐标数 |',
                '|---|---|---|---:|---:|---:|']
            body.extend(f"| {r['model']} | {r['dataset']} | {r['host']} | {r['step_rps']} | {r['cap_rps']} | {r['coordinates']} |" for r in checks)
            body += ['', '每组提供六指标 PNG/PDF 曲线；summary.csv 汇总均值与实测范围，measurements.csv 保留逐次结果，gpu-details.csv 保留八卡明细，raw-evidence-index.csv 索引工作区内的原始证据及 SHA-256。', '',
                '终点由首次有效完整单次 PDBlend SLO < 90% 决定；恰好 90% 继续，确认复测恢复到 90% 以上也不改变首次越界。正常点一次，已有重复全部保留。阴影表示实测范围，不是置信区间。', '',
                'SLO 为完整且同时严格满足 TTFT、TPOT 的请求数 / trace 全部请求数。失败、拒绝、超时保留在分母。三个模型共用 Alpaca 1 s / 0.10 s、ShareGPT 5 s / 0.15 s、LongBench 15 s / 0.20 s；100 秒到达窗口、120 秒超时、arrival seed 701。', '',
                'Energy 包含八卡完整测量窗口及 drain、转换尾部；吞吐报告完成请求 / 秒与精确输出 token / 秒。延迟缺失不填零。利用率为逐卡时间加权后取八卡平均。指标补采只填缺失指标，不改变历史 SLO 或其他完整指标；各指标的 n、source、cell_ids 列注明来源。', '',
                '准备及工程异常能耗单列于持续报告的 setup-energy-ledger.csv，未加入 serving Energy。历史及网格外结果保留在 history.csv。']
            (temporary / 'README.md').write_text('\n'.join(body) + '\n')
            report.save(temporary / 'validation.json', dict(passed=True, checks=checks,
                source_report_sha256=snapshot_sha, publisher_sha256=sha(__file__), report_sha256=sha(report.__file__)))
            files = {str(p.relative_to(temporary)): sha(p) for p in sorted(temporary.rglob('*')) if p.is_file()}
            report.save(temporary / 'manifest.json', dict(schema='completed-rate-scale-bundle-v1',
                created_s=time.time(), scope=sorted(keys), source_report_sha256=snapshot_sha, files=files))
            os.rename(temporary, destination)
        except BaseException:
            # Retain failed staging output for diagnosis; never publish it as complete.
            raise
    if not archive.exists():
        temporary_zip = archive.with_suffix('.zip.pending')
        with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED) as package:
            for path in sorted(destination.rglob('*')):
                if path.is_file():
                    package.write(path, str(Path(destination.name) / path.relative_to(destination)))
        os.rename(temporary_zip, archive)
    # Verify the actual downloadable ZIP, including previously sealed bundles.
    manifest = json.loads((destination / 'manifest.json').read_text())
    with zipfile.ZipFile(archive) as package:
        expected = {str(Path(destination.name) / name) for name in manifest['files']}
        expected.add(str(Path(destination.name) / 'manifest.json'))
        assert set(package.namelist()) == expected, 'archive member set differs from manifest'
        for name, digest in manifest['files'].items():
            data = package.read(str(Path(destination.name) / name))
            assert hashlib.sha256(data).hexdigest() == digest, 'archive member digest changed'
        assert package.read(str(Path(destination.name) / 'manifest.json')) == (destination / 'manifest.json').read_bytes()
    return dict(path=str(destination), archive=str(archive), archive_sha256=sha(archive))


def run(once):
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(15)
        while True:
            raw = (HERE / 'reports/current/results.json').read_bytes()
            snapshot = report.prepare_snapshot(json.loads(raw))
            digest = hashlib.sha256(raw).hexdigest()
            completed, errors = [], []
            for group in snapshot['groups']:
                if not group['complete']:
                    continue
                key = (group['model'], group['dataset'])
                try:
                    completed.append(publish(snapshot, {key}, '-'.join(key), digest))
                except Exception as exc:
                    errors.append(dict(group=key, error=repr(exc)))
            state = dict(pid=os.getpid(), updated_s=time.time(), complete=len(completed) == 9,
                groups_completed=len(completed), results=completed, errors=errors)
            report.save(DEST / 'status.json', state)
            if state['complete'] or once:
                print(json.dumps(state), flush=True)
                return
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    run(parser.parse_args().once)
