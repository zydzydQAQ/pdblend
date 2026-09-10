"""Run frozen same-rate cells using the original eight-GPU 100+120+90 meter."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import time

import support as p


def validate_rows(release, binding, contract):
    rows = release['rows']
    p.need(rows and len({r['cell_id'] for r in rows}) == len(rows), 'empty or duplicate rows')
    p.need(len({(r['model'], r['dataset'], r['rate_rps'], r['system']) for r in rows}) == 1,
           'one numeric rate and one system per release')
    p.need([r['repeat'] for r in rows] in ([1], [2], [1, 2]), 'declared repeats must run in order')
    for row in rows:
        expected = contract.lookup_cell(release['declaration'], row['cell_id'])
        p.need(row == expected, 'row changed from declared cell')
        p.need(row['node'] == release['node'] and row['model'] == binding['model']
               and row['system'] == binding['system'], 'node/model/system mismatch')
        p.need(row['arrival_window_s'] == 100 and row['seed'] == 701 and row['slo_scale'] == 1,
               'window/seed/SLO changed')
        p.need(p.sha(row['trace']) == row['trace_sha256'], 'trace changed')
        p.need(row['dataset'] in binding['configs'], 'dataset configuration absent')
    group = contract.resolve_group(release['declaration'], rows[0]['model'], rows[0]['dataset'],
                                   actual_host=release['node'])
    dynamic_reuse = [p.checked(r) for r in release.get('dynamic_reuse_observations', [])]
    if dynamic_reuse:
        group = contract.apply_audited_reuse(group, dynamic_reuse)
    observations = [p.checked(r) for r in release['scheduling_observations']]
    decision = contract.select_group(group, observations)
    if binding['system'] == 'pdblend':
        p.need(decision['phase'] == 'pdblend', 'PDB dispatch is not the current group phase')
        available = [t['cell_id'] for t in decision['next_tasks']]
        p.need([r['cell_id'] for r in rows] == available[:len(rows)], 'PDB row skips current rate/repeat')
    else:
        p.need(decision['phase'] == 'baselines', 'baseline dispatch precedes PDB boundary')
        allowed = {t['cell_id'] for t in decision['baseline_tasks'] if t['action'] == 'execute'}
        p.need({r['cell_id'] for r in rows} <= allowed, 'baseline row lies above cap')


def load_release(reference):
    release = p.checked(reference)
    p.need(release['schema'] == 'uniform-rate-cells-release-v2', 'unknown release')
    for path, digest in release['files'].items():
        p.need(p.sha(path) == digest, 'frozen input changed: ' + path)
    binding = p.checked(release['binding'])
    contract = p.load(release['declaration_contract'], 'uniform_execution_contract')
    validate_rows(release, binding, contract)
    verifier = p.load(release['qualification_validator'], 'uniform_saved_qualification')
    qualified = verifier.verify(release['qualification'])
    p.need(qualified['passed'] and qualified['independently_recomputed'], 'qualification did not independently pass')
    qb = p.checked(qualified['binding'])
    p.need(binding['instances'] == qb['instances'] and binding['hostname'] == qb['hostname']
           and binding['host_release'] == qb['host_release'], 'qualification does not cover actual binding')
    p.need(binding['system'] == qb['system'] and binding['configs'] == qb['configs'],
           'qualified implementation/configuration changed')
    p.need(release['expected_hostname'] == binding['hostname'], 'physical hostname differs')
    p.need(binding.get('deadline_s') is None and binding.get('campaign_lifecycle') == 'until_declared_complete_v1',
           'measurement lifecycle differs')
    return release, binding, contract


async def execute(reference, out, state):
    release, binding, _ = load_release(reference)
    p.need(socket.gethostname() == binding['hostname'] and 'PDBLEND_NODE_LOCK_FD' not in os.environ,
           'must execute from actual host without inherited hardware lease')
    for previous in release['predecessors']:
        old = p.checked(previous)
        p.need(old.get('finished_s') and not old.get('node_lease_held') and not p.active_owner(old),
               'predecessor owner has not finished')
    helper = p.load(release['original_executor'], 'uniform_original_executor')
    common = helper.load_common(binding['host_release'])
    measurement = (p.load(release['measurement_executor'], 'uniform_qualified_measurement')
                   if release.get('measurement_executor') else common)
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    import audit_cell
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in release['rows']:
                if any(Path(path).exists() for path in release['stop_paths']):
                    state['stopped_at_boundary'] = True
                    break
                common.validate_binding(binding)
                cp_path = out / 'results/checkpoints' / (row['cell_id'] + '.json')
                p.need(not cp_path.exists(), 'no automatic replay of observed cell')
                state['current_cell'] = row['cell_id']
                state['attempted'].append(row['cell_id'])
                p.save(out / 'status.json', state)
                error = None
                try:
                    await measurement.run_one(session, binding, row, out / 'results', hardware)
                except BaseException as exc:
                    error = exc
                receipt_path = out / 'results/operations' / row['cell_id'] / 'receipt.json'
                if receipt_path.exists():
                    receipt = p.read(receipt_path)
                    artifacts = {str(f): p.sha(f) for base in (receipt_path.parent, out / 'results/cells' / row['cell_id'])
                                 for f in base.rglob('*') if f.is_file()}
                    cp = dict(schema='uniform-rate-checkpoint-v2', row=row, repeat=row['repeat'], node=release['node'],
                        release=reference, declaration=release['declaration'], binding=release['binding'],
                        qualification=release['qualification'], qualification_validator=release['qualification_validator'],
                        host_manifest=release['host_manifest'], receipt=p.ref(receipt_path), artifacts=artifacts,
                        finished_s=time.time(), execution_error=repr(error) if error else None,
                        measurement_valid=receipt.get('measurement_valid', False),
                        work_complete=receipt.get('summary', {}).get('work_complete', False))
                    p.save(cp_path, cp)
                    state['observed_checkpoints'].append(p.ref(cp_path))
                    if error is None:
                        try:
                            observation = audit_cell.audit(p.ref(cp_path))
                            full = observation['work_complete'] and observation['completed_work_requests'] == row['n_requests']
                            service_failure = (row['system'] != 'pdblend'
                                and observation.get('baseline_service_failure', {}).get('independently_diagnosed') is True)
                            p.need(full or service_failure, 'incomplete work requires independent diagnosis')
                            if service_failure:
                                diagnosis_path = out / 'diagnoses' / (row['cell_id'] + '.json')
                                p.save(diagnosis_path, dict(schema='uniform-baseline-deadline-diagnosis-v1',
                                    passed=True, independently_recomputed=True, checkpoint=p.ref(cp_path),
                                    classification=observation['baseline_service_failure'],
                                    verification=observation['verification'],
                                    no_unknown_errors=True, no_PDB_complete_boundary_claim=True))
                                observation.update(failure_class='independently_diagnosed_capacity_deadline',
                                                   diagnosis_reference=p.ref(diagnosis_path))
                            audit_path = out / 'audits' / (row['cell_id'] + '.json')
                            p.save(audit_path, observation)
                            state['observations'].append(p.ref(audit_path))
                            state['completed'].append(row['cell_id'])
                        except BaseException as exc:
                            error = exc
                    if error is not None:
                        state['failed'].append(dict(cell_id=row['cell_id'], error=repr(error)))
                    p.save(out / 'status.json', state)
                if error is not None:
                    raise error
                p.need(row['cell_id'] in state['completed'], 'missing valid checkpoint')
            p.save(out / 'identity.after.json', await measurement.identity(session, binding))
            state['complete'] = len(state['completed']) == len(release['rows']) and not state['failed']
        state['node_lease_held'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    reference = p.ref(args.release)
    release, _, _ = load_release(reference)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True)))
        return
    p.need(args.out and not args.out.exists(), 'fresh output directory required; automatic retry forbidden')
    out = args.out.resolve()
    out.mkdir(parents=True)
    state = dict(schema='uniform-rate-cell-status-v2', pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(), release=reference,
        node=release['node'], model=release['model'], dataset=release['dataset'],
        declaration=release['declaration'], completion_scope='five_systems',
        node_lease_held=False, complete=False, attempted=[], completed=[], failed=[], observed_checkpoints=[], observations=[])
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(reference, out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(out / 'status.json', state)


if __name__ == '__main__':
    main()
