"""Freeze and execute one scale cell using the qualified unchanged measurement core."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import slo_support as p

HOSTS = {'A': 'iZwz9274emxme9019d2sjgZ', 'C': 'iZwz9gfq11hx1sbob59yrgZ'}
EXECUTOR = p.PRIOR / 'common/execution-until-complete-v1/run.py'
BOOTSTRAP = p.PRIOR / 'B/baseline-return-after-external-source-v1/execution.py'


def prepare(*, node, rate, system, repeat, qualification, validator, out, observations=(), measurement_executor=None):
    import contract
    destination = Path(out).resolve()
    p.need(not destination.exists(), 'fresh immutable release directory required')
    row = contract.make_row(node, rate, system, repeat=repeat)
    chosen = contract.evaluate_group(node, list(observations))
    available = chosen.get('next_tasks', [])
    if chosen['status'] == 'cap_confirmed':
        available = chosen.get('baseline_tasks', [])
    # contract is the sole scheduling authority; normalize only its task wrapper.
    allowed = [x.get('row', x) for x in available]
    p.need(any(x.get('cell_id') == row['cell_id'] for x in allowed), 'cell is not the next declared task')
    verify = p.load(validator, 'slo_prepare_qualification')
    q = verify.verify(qualification)
    p.need(q.get('passed') and q.get('independently_recomputed'), 'qualification is not independently verified')
    binding_ref = q['binding']; binding = p.checked(binding_ref)
    p.need(binding['hostname'] == HOSTS[node] and binding['model'] == '14b'
           and binding['system'] == system, 'wrong host/model/system qualification')
    host = Path(binding['host_release'])
    manifest = p.ref(host / 'manifest.json')
    files = dict(binding['files'])
    files.update({str(host / name): digest for name, digest in p.checked(manifest)['files'].items()})
    for field in ('files', 'source_files'):
        files.update(p.checked(qualification).get(field, {}))
    files.update(q.get('files', {}))
    selected_executor = measurement_executor or p.ref(EXECUTOR)
    p.need(selected_executor['sha256'] == p.sha(EXECUTOR), 'unregistered measurement core change')
    references = dict(binding=binding_ref, qualification=qualification, qualification_validator=validator,
        host_manifest=manifest, protocol=p.ref(p.HERE / 'protocol.json'),
        measurement_executor=selected_executor, executor_bootstrap=p.ref(BOOTSTRAP),
        raw_auditor=p.ref(p.HERE.parent / 'main-slo-improvement-v7/report.py'),
        measurement_auditor=p.ref(p.PRIOR / 'common/uniform-rate-20260909-v2/metrics.py'))
    dependencies = [p.HERE / name for name in ('slo_support.py', 'contract.py', 'generate.py', 'audit.py', 'run_cell.py')]
    dependencies += [Path(selected_executor['path']).with_name('child.py'), p.HERE.parent / 'main-slo-improvement-v7/protocol.py',
                     p.HERE.parent / 'main-slo-improvement-v7/raw_metrics.py',
                     p.PRIOR / 'audit_cooperative_arrivals_v1.py']
    for reference in references.values():
        files[reference['path']] = reference['sha256']
    for path in dependencies:
        files[str(path)] = p.sha(path)
    for config in binding['configs'].values():
        files[config] = p.sha(config)
    files[row['trace']] = row['trace_sha256']
    destination.mkdir(parents=True)
    declaration = dict(schema='slo-rate-single-cell-declaration-v1', protocol=references['protocol'],
                       created_s=time.time(), row=row, scheduling_decision=chosen,
                       scheduling_observations=list(observations))
    p.save(destination / 'declaration.json', declaration)
    release = dict(schema='slo-rate-cell-release-v1', node=node, expected_hostname=HOSTS[node],
        rows=[row], declaration=p.ref(destination / 'declaration.json'), files=files, **references)
    p.save(destination / 'release.json', release)
    load_release(p.ref(destination / 'release.json'))
    return p.ref(destination / 'release.json')


def load_release(reference):
    import contract
    release = p.checked(reference)
    p.need(release['schema'] == 'slo-rate-cell-release-v1' and len(release['rows']) == 1, 'invalid cell release')
    for path, digest in release['files'].items():
        p.need(p.sha(path) == digest, 'frozen release input changed: ' + path)
    row = release['rows'][0]; declaration = p.checked(release['declaration'])
    p.need(row == declaration['row'], 'declared row changed')
    p.need(row == contract.make_row(release['node'], row['rate_rps'], row['system'], repeat=row['repeat']),
           'row differs from experiment contract')
    binding = p.checked(release['binding'])
    verifier = p.load(release['qualification_validator'], 'slo_saved_qualification')
    qualification = verifier.verify(release['qualification'])
    p.need(qualification.get('passed') and qualification.get('independently_recomputed'), 'qualification failed')
    qb = p.checked(qualification['binding'])
    p.need(binding == qb, 'qualification does not cover exact execution binding')
    p.need(binding['hostname'] == release['expected_hostname'] == HOSTS[release['node']], 'physical host differs')
    p.need(binding['model'] == row['model'] == '14b' and binding['system'] == row['system'], 'model/system differs')
    p.need(binding.get('deadline_s') is None and binding.get('campaign_lifecycle') == 'until_declared_complete_v1',
           'wrong experiment lifecycle')
    return release, binding


async def execute(reference, out, state):
    release, binding = load_release(reference)
    p.need(socket.gethostname() == binding['hostname'] and 'PDBLEND_NODE_LOCK_FD' not in os.environ,
           'wrong host or inherited hardware lease')
    bootstrap = p.load(release['executor_bootstrap'], 'slo_original_execution_bootstrap')
    bootstrap.load_common(binding['host_release'])
    meter = p.load(release['measurement_executor'], 'slo_qualified_measurement')
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    auditor = p.load(p.ref(p.HERE / 'audit.py'), 'slo_actual_auditor')
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        meter.validate_binding(binding)
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            row = release['rows'][0]
            state['current_cell'] = row['cell_id']
            p.save(out / 'status.json', state)
            error = None
            try:
                await meter.run_one(session, binding, row, out / 'results', hardware)
            except BaseException as exc:
                error = exc
            receipt_path = out / 'results/operations' / row['cell_id'] / 'receipt.json'
            if receipt_path.exists():
                receipt = p.read(receipt_path)
                artifacts = {str(f): p.sha(f) for root in (receipt_path.parent, out / 'results/cells' / row['cell_id'])
                             if root.exists() for f in root.rglob('*') if f.is_file()}
                cp = dict(schema='slo-rate-checkpoint-v1', node=release['node'], row=row,
                    release=reference, declaration=release['declaration'], binding=release['binding'],
                    qualification=release['qualification'], qualification_validator=release['qualification_validator'],
                    host_manifest=release['host_manifest'], receipt=p.ref(receipt_path), artifacts=artifacts,
                    finished_s=time.time(), execution_error=repr(error) if error else None,
                    measurement_valid=receipt.get('measurement_valid', False),
                    work_complete=receipt.get('summary', {}).get('work_complete', False))
                cp_path = out / 'results/checkpoints' / (row['cell_id'] + '.json')
                p.save(cp_path, cp)
                state['checkpoint'] = p.ref(cp_path)
                if error is None:
                    try:
                        observation = auditor.audit(p.ref(cp_path))
                        p.save(out / 'audited.json', observation)
                        state['audit_reference'] = p.ref(out / 'audited.json')
                        state['complete'] = True
                    except BaseException as exc:
                        error = exc
                state['cleanup_complete'] = bool(receipt.get('child_stopped') and receipt.get('clock_restore_complete')
                    and receipt.get('restoration') and all(x.get('complete') for x in receipt['restoration'].values())
                    and not receipt.get('outer_cleanup_errors'))
            p.save(out / 'status.json', state)
            if error:
                raise error
            p.need(state['complete'], 'missing valid audited measurement')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('node', 'rate', 'system', 'qualification', 'validator', 'out'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--repeat', type=int, default=1)
    prep.add_argument('--observations', type=Path)
    prep.add_argument('--measurement-executor', type=Path)
    run = sub.add_parser('execute')
    run.add_argument('--release', type=Path, required=True)
    run.add_argument('--out', type=Path)
    run.add_argument('--run', action='store_true')
    args = ap.parse_args()
    if args.command == 'prepare':
        observations = p.read(args.observations) if args.observations else []
        result = prepare(node=args.node, rate=args.rate, system=args.system, repeat=args.repeat,
            qualification=p.ref(args.qualification), validator=p.ref(args.validator), out=args.out,
            observations=observations,
            measurement_executor=p.ref(args.measurement_executor) if args.measurement_executor else None)
        print(json.dumps(result)); return
    reference = p.ref(args.release)
    load_release(reference)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True))); return
    p.need(args.out and not args.out.exists(), 'fresh attempt required; no automatic replay')
    out = args.out.resolve(); out.mkdir(parents=True)
    state = dict(schema='slo-rate-cell-status-v1', pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(),
        release=reference, complete=False, node_lease_held=False)
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
