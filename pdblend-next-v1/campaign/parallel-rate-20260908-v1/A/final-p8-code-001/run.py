"""Declared P8 rate cells with fresh initial2, one lease and measured dynamic cleanup."""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'changed reference ' + reference['path'])
    return read(reference['path'])


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def full_work(summary):
    return (summary.get('work_complete') is True and summary.get('failed_requests') == 0
            and summary.get('request_timeouts') == 0)


def checked_artifacts(value):
    need(value.get('artifacts') and all(sha(p) == h for p, h in value['artifacts'].items()),
         'actual qualification request/raw artifacts changed')
    return dict(value['artifacts'])


def validate_candidate_qualification(q):
    from qualification_contract import build_qualification
    actual = build_qualification()
    need(q == actual, 'actual P8 autonomous gate/900 qualification differs on independent re-audit')
    return q['arms']['fixed2']


def validate_release(path):
    release = read(path)
    need(release['schema'] == 'parallel-rate-p8-qualified-dynamic-release-v1'
         and release['approved'] and release['dynamic_qualified'], 'actual qualified P8 release required')
    need(release['deadline_s'] is None and release['campaign_lifecycle'] == 'until_declared_complete_v1',
         'no total campaign deadline')
    need(release['model'] == '14b' and release['hostname'] == socket.gethostname(), 'wrong physical host')
    for file, digest in release['files'].items():
        need(sha(file) == digest, 'frozen release file changed ' + file)
    q = checked(release['qualification900'])
    validate_candidate_qualification(q)
    need(release['qualification_auditor'] == q['qualification_auditor'] and release['stage_auditor'] == q['stage_auditor'], 'actual qualification auditor differs')
    need(release['controller_calibration_compatibility'] == q['controller_calibration_compatibility'], 'declared P8/P6 provenance differs')
    need(q['passed'] and q['actual_growth_and_return'] and q['formal_100s_performance_not_inferred'],
         'actual fixed2/dynamic900 validation missing')
    need(q['source'] == release['host_manifest'] and q['profile'] == release['profile'],
         'qualification source/profile differs')
    for arm in ('dynamic',):
        result = checked(q['arms'][arm]['result'])
        checked_artifacts(result)
        status = checked(q['arms'][arm]['status'])
        need(full_work(result) and status['complete'] and status['cleanup_complete']
             and not status.get('error') and not status.get('cleanup_errors'), '900 work/cleanup failed')
    cap = checked(release['capacity_binding'])
    need(cap['calibration'] == q['certificate'] and cap['calibration_only'] is False, 'wrong capacity certificate')
    qualified_cap = checked(q['capacity_binding'])
    need(cap.get('physical_operation_timeout_s') == 120, 'explicit bounded P8 physical operation required')
    from capacity_calibration_compatibility_v3 import verify
    verify(release['controller_calibration_compatibility'], release['host_manifest'], cap)
    proof = checked(release['controller_calibration_compatibility'])
    need(all(release['files'].get(p) == h for p, h in proof['files'].items()) and release['files'].get(release['controller_calibration_compatibility']['path']) == release['controller_calibration_compatibility']['sha256'], 'formal binding must freeze the narrow dual-source proof')
    location_fields = {'owner_id', 'http_port_base', 'kv_port_base', 'runtime_dir', 'files'}
    need({k: v for k, v in cap.items() if k not in location_fields}
         == {k: v for k, v in qualified_cap.items() if k not in location_fields},
         'capacity policy/domain/source must match actual900 qualification')
    from capacity_runtime import calibration_model, load_planner
    planner = calibration_model(load_planner(cap['planner_source']), cap)
    need({tuple(t.gpus) for t in planner.transitions} == {(5,)}, 'only GPU5 transitions measured')
    need({tuple(tuple(g) for g in b.resident_groups) for b in planner.layouts} == {((6,), (7,)), ((5,), (6,), (7,))},
         'only actual initial2 and layout3 are eligible')
    base = checked(release['binding'])
    need({i['id'] for i in base['instances']} == {'nextv3a6', 'nextv3a7'}, 'each cell must start with original2')
    configs = [checked(r) for r in release['configs'].values()]
    need(len(configs) == 3 and all(c == configs[0] for c in configs), 'one model-wide strategy and profile required')
    cfg = configs[0]
    need(cfg['instances'] == base['instances'], 'dynamic inventory requires the exact frozen original native identities')
    need(cfg['capacity_integration_v1'] is True and cfg['prepare_peers'] is False
         and cfg['allow_pd'] is False and cfg['transfers'] == [] and cfg['profiles'] == release['profile']['path'],
         'independent mixed measured policy required')
    need(cfg['request_timeout_s'] == 120 and cfg['arrival_window_s'] == 100
         and cfg['measurement_window_protocol'] == base['protocol_id'], 'original100/120 protocol required')
    declaration = checked(release['declaration'])
    return release, base, declaration


def point_binding(common, base, release, release_path, cell, output, out):
    import dynamic_ownership as ownership
    config = checked(release['configs'][cell['dataset']])
    inventory = out / 'inventories' / (cell['cell_id'] + '.json')
    job = output / 'operations' / cell['cell_id'] / 'job.json'
    lease = ownership.inherited_lease()
    authority = dict(schema='capacity-parent-lease-authority-v1', fd=lease['fd'],
        holder_pid=lease['holder_pid'], holder_start_ticks=lease['holder_start_ticks'],
        lock_path=str(ownership.LOCK), lock_device=lease['device'], lock_inode=lease['inode'],
        capacity_inventory_path=str(inventory), expected_job_path=str(job), invocation=ref(release_path))
    authority_path = out / 'authorities' / (cell['cell_id'] + '.json')
    common.write(authority_path, authority)
    config.update(capacity_inventory_path=str(inventory), capacity_job_path=str(job),
                  capacity_lease_authority=ref(authority_path))
    config_path = out / 'configs' / (cell['cell_id'] + '.json')
    common.write(config_path, config)
    binding = copy.deepcopy(base)
    binding.update(output=str(output), host_release=release['host_release'], deadline_s=None,
        campaign_lifecycle='until_declared_complete_v1', unchanged_pdb_policy=False, formal_eligible=False,
        improvement=dict(arm='dynamic', repeat=cell['repeat'], original_cell_id=cell['original_cell_id'],
                         implementation=release['implementation_id']))
    binding['configs'] = {cell['dataset']: str(config_path)}
    binding['files'].update(release['files'])
    for reference in (ref(config_path), ref(authority_path), cell['trace'], release['declaration']):
        binding['files'][reference['path']] = reference['sha256']
    return binding


def checkpoint(common, binding_path, cell, row, output, receipt):
    rp = output / 'operations' / cell['cell_id'] / 'receipt.json'
    summary = receipt.get('summary', {})
    common.write(rp.parent / 'engineering-gate.json', dict(
        passed=receipt.get('measurement_valid') is True and full_work(summary),
        work_complete=summary.get('work_complete'), failed_requests=summary.get('failed_requests'),
        request_timeouts=summary.get('request_timeouts'),
        errors=[receipt['error']] if receipt.get('error') else [], faults=[],
        stop_on_any_request_failure=True, all_eight_energy_retained=True,
        low_slo_does_not_trigger_retry=True, actual_initial2_and_dynamic_cleanup_required=True))
    artifacts = {str(f): sha(f) for directory in (rp.parent, output / 'cells' / cell['cell_id'])
                 for f in directory.rglob('*') if f.is_file()}
    # Capacity transitions live outside the serving cell; preserve their raw evidence too.
    for path, digest in receipt.get('dynamic_artifacts', {}).items():
        need(sha(path) == digest, 'external physical transition artifact changed')
        artifacts[path] = digest
    cp = dict(row=row, declaration=cell, binding=str(binding_path), binding_sha256=sha(binding_path),
        receipt=str(rp), receipt_sha256=sha(rp), artifacts=artifacts,
        measurement_valid=receipt.get('measurement_valid') is True, work_complete=summary.get('work_complete'),
        completed_s=time.time(), poor_slo_does_not_trigger_retry=True)
    common.write(output / 'checkpoints' / (cell['cell_id'] + '.json'), cp)
    return cp


async def execute(args, state):
    release, base, declaration = validate_release(args.release)
    executor = load(release['dynamic_executor']['path'], 'final_p6_dynamic_executor')
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    need(not args.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh output and fresh owner required')
    args.out.mkdir(parents=True)
    output = args.out / 'results'
    cells = declaration['cells']
    executor.write(args.out / 'release-reference.json', ref(args.release))
    executor.write(args.out / 'declaration-order.json', cells)
    state.update(model='14b', stage='screen_dynamic', declared=len(cells), saturated_at={}, skipped_saturated=[],
                 remaining=[c['cell_id'] for c in cells], all_failure_stop=True)

    def update(**value):
        state.update(value, updated_s=time.time())
        executor.write(args.out / 'status.json', state)

    with node_lease():
        update(node_lease_held=True, phase='identity_and_ordinary')
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            executor.validate_binding(base)
            executor.write(args.out / 'identity.before.json', await executor.identity(session, base))
            helperdir = str(Path(release['ordinary_helper']['path']).parent)
            sys.path.insert(0, helperdir)
            helper = load(release['ordinary_helper']['path'], 'final_p6_measured_ordinary')
            state['ordinary'] = await helper.measured_ordinary(executor, session, base, hardware, args.out / 'setup')
            for cell in cells:
                rate = cell['source_row']['rate_rps']
                if rate > state['saturated_at'].get(cell['dataset'], float('inf')):
                    state['skipped_saturated'].append(cell['cell_id'])
                    update(remaining=[c['cell_id'] for c in cells if c['cell_id'] not in state['attempted'] + state['skipped_saturated']])
                    continue
                if args.stop_requested or Path(release['stop_path']).exists():
                    update(phase='stopped_at_boundary')
                    break
                validate_release(args.release)
                binding = point_binding(executor, base, release, args.release, cell, output, args.out)
                binding_path = args.out / 'bindings' / (cell['cell_id'] + '.json')
                executor.write(binding_path, binding)
                executor.validate_binding(binding)
                row = copy.deepcopy(cell['source_row'])
                row.update(cell_id=cell['cell_id'], original_cell_id=cell['original_cell_id'],
                           improvement_arm='dynamic', improvement_repeat=cell['repeat'],
                           arm='dynamic', repeat=cell['repeat'], implementation_series='parallel-rate-p8')
                state['attempted'].append(cell['cell_id'])
                update(phase='running', current_cell=cell['cell_id'])
                try:
                    receipt = await executor.run_one(session, binding, row, output, hardware)
                except BaseException:
                    rp = output / 'operations' / cell['cell_id'] / 'receipt.json'
                    if rp.exists():
                        checkpoint(executor, binding_path, cell, row, output, read(rp))
                    raise
                checkpoint(executor, binding_path, cell, row, output, receipt)
                state['completed'].append(cell['cell_id'])
                update(remaining=[c['cell_id'] for c in cells if c['cell_id'] not in state['attempted'] + state['skipped_saturated']])
                need(full_work(receipt['summary']), 'any failed/timeout/incomplete request stops expansion after CP and cleanup')
                if receipt['summary']['slo_attainment'] < .9:
                    state['saturated_at'][cell['dataset']] = min(rate, state['saturated_at'].get(cell['dataset'], rate))
                update()
            executor.write(args.out / 'identity.after.json', await executor.identity(session, base))
            complete = len(state['completed']) + len(state['skipped_saturated']) == len(cells)
            update(complete=complete, all_declared_measured=len(state['completed']) == len(cells),
                   phase='complete' if complete else state['phase'])
    update(node_lease_held=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--run', action='store_true')
    args = p.parse_args(); args.stop_requested = False
    release = read(args.release); host = Path(release['host_release'])
    paths = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps',
             str(Path(release['dynamic_executor']['path']).parent)]
    sys.path[:0] = paths; os.environ['PYTHONPATH'] = ':'.join(paths); os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    validate_release(args.release)
    if not args.run:
        print(json.dumps(dict(cpu_only=True, points=len(checked(release['declaration'])['cells']), no_gpu=True)))
        return
    state = dict(schema=1, pid=os.getpid(), started_s=time.time(), phase='starting', complete=False,
                 attempted=[], completed=[], failed=[], automatic_retries=False, node_lease_held=False)
    def stop(_sig, _frame):
        args.stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        asyncio.run(execute(args, state))
    except BaseException as exc:
        state['failed'].append(repr(exc)); state.update(phase='needs_attention', error=repr(exc), complete=False,
                                                      engineering_gate_failed=True)
        raise
    finally:
        if args.out.exists():
            state.update(finished_s=time.time(), node_lease_held=False)
            state.pop('current_cell', None)
            temporary = args.out / 'status.json.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n'); os.replace(temporary, args.out / 'status.json')


if __name__ == '__main__':
    main()
