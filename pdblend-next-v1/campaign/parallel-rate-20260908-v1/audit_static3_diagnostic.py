"""Independently verify the two static-three observations, separately from final PDB."""
import json
from pathlib import Path
import time
from collect_v10 import load_audit, read, sha, write_csv
import source_identity

ROOT = Path(__file__).resolve().parent


def run(out):
    if out.exists():
        raise FileExistsError(out)
    audit = load_audit()
    p = audit.p
    source = ROOT / 'A/static3-diagnostic-002'
    terminal_path = source / 'independent-terminal-audit.json'
    terminal = read(terminal_path)
    launch_path = ROOT / 'A/static3-diagnostic-launch-002.json'
    launch = read(launch_path)
    spec = p.checked(launch['spec'])
    p.checked(spec['profiles'])
    p.need(launch['out'] == str(source), 'diagnostic launch output differs')
    p.need(terminal['passed'] and terminal['complete_service_cleanup']
           and terminal['nodelease_free'] and terminal['pid_exited'], 'diagnostic did not finish physical cleanup')
    sources = {str(path): sha(path) for path in (Path(__file__), terminal_path, launch_path,
        Path(launch['spec']['path']), Path(spec['profiles']['path']),
        ROOT / 'source_identity.py', ROOT / 'collect_v10.py')}
    rows = []
    for cp_path in sorted((source / 'results/checkpoints').glob('*.json')):
        cp = read(cp_path)
        p.need(cp['diagnostic_only'] and not cp['dynamic_qualified'], 'unexpected diagnostic scope')
        row = cp['row']
        trace_ref = dict(path=row['trace_path'], sha256=row['trace_sha256'])
        trace = p.checked(trace_ref)
        receipt_ref = dict(path=cp['receipt'], sha256=cp['receipt_sha256'])
        binding_ref = dict(path=cp['binding'], sha256=cp['binding_sha256'])
        receipt, binding = p.checked(receipt_ref), p.checked(binding_ref)
        p.need(receipt['measurement_valid'] and receipt['clock_restore_complete']
               and receipt['child_stopped'] and not receipt['outer_cleanup_errors'], 'invalid measurement cleanup')
        p.need(all(v['complete'] for v in receipt['restoration'].values()), 'native cleanup incomplete')
        for path, expected in cp['artifacts'].items():
            p.need(sha(path) == expected, 'diagnostic raw artifact changed: ' + path)
        summary = receipt['summary']
        directory = Path(receipt_ref['path']).parents[2] / 'cells' / row['cell_id']
        p.need(read(directory / 'summary.json') == summary, 'summary and receipt differ')
        p.need(summary['gpu_count'] == 8 and summary['measurement_valid']
               and summary['fixed_window_valid'] and summary['power_source_verified'], 'eight-GPU energy invalid')
        declared = dict(row, n_expected=len(trace['requests']),
            expected_generated_tokens=sum(r['output_len'] for r in trace['requests']))
        proof = audit.audit_raw(summary, directory, dict(trace=trace_ref, original_point=declared))
        additional = audit.raw_metrics.audit_additional_metrics(summary, directory)
        # This diagnostic froze its active profile in the launch spec, although
        # the generated benchmark binding omitted it from files. Verify that
        # independent evidence and expose the normalized dependency view only
        # to the fingerprint helper; never rewrite the original binding.
        config = read(binding['configs'][row['dataset']])
        p.need(config['profiles'] == spec['profiles']['path'], 'diagnostic profile and launch differ')
        dependency_view = dict(binding, files=dict(binding['files'],
            **{spec['profiles']['path']: spec['profiles']['sha256']}))
        identity = source_identity.identity(dependency_view, row['dataset'])
        p.need(identity['configured_gpu_count'] == 3, 'static three-GPU layout differs')
        point = {k: declared[k] for k in p.PAIR_FIELDS}
        point.update({k: summary.get(k) for k in audit.METRICS})
        point.update(additional['normalized_metrics'])
        point.update(identity)
        point.update(cell_id=row['cell_id'], system='pdblend', repeat=1,
            diagnostic_only=True, final_comparison_eligible=False, dynamic_qualified=False,
            profile_freeze_evidence=launch['spec'], profile_missing_from_benchmark_binding_files=True,
            energy_includes_premeasurement_cold_start=False, energy_measured_gpu_count=8,
            measurement_valid=True, work_complete=summary['work_complete'],
            completed_work_requests=summary['completed_work_requests'], generated_tokens=summary['generated_tokens'],
            failed_requests=summary['failed_requests'], request_timeouts=summary['request_timeouts'],
            completion_fraction=summary['completed_work_requests']/summary['n_expected'],
            checkpoint=p.ref(cp_path), receipt=receipt_ref, binding=binding_ref,
            verification=proof, additional_verification=additional)
        rows.append(point)
        sources[str(cp_path)] = sha(cp_path)
    p.need(len(rows) == 2, 'two declared static diagnostic observations required')
    out.mkdir(parents=True)
    write_csv(out / 'static3-diagnostic-points.csv', rows)
    (out / 'results.json').write_text(json.dumps(dict(created_s=time.time(), points=rows,
        all_prescribed_work_complete=all(x['work_complete'] for x in rows),
        static_three_only=True, no_final_pdblend_win_counts=True), indent=2) + '\n')
    (out / 'README.md').write_text('这两次观测仅验证预先启动三张工作卡的静态布局，全部八卡能耗独立复算。\n\n'
        '冷启动发生在请求窗口前，成本另存容量证据；这些观测不计入正式动态 PDBlend 胜率，也不替代运行中扩容资格。\n')
    for path, expected in sources.items():
        p.need(sha(path) == expected, 'frozen diagnostic source changed')
    (out / 'manifest.json').write_text(json.dumps(dict(sources=sources,
        files={path.name: sha(path) for path in out.iterdir() if path.is_file()}), indent=2) + '\n')
    print(json.dumps([dict(rate=x['rate_rps'], complete=x['work_complete'], slo=x['slo_attainment'],
                          energy_j=x['energy_j']) for x in rows]))


if __name__ == '__main__':
    run(ROOT / 'reports/static3-diagnostic-verified-001')
