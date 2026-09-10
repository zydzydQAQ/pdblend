"""Continue only the five unattempted old baseline cells after diagnosed timeouts."""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESUME = HERE.parent / 'ascending-resume-20260909-v2'
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v1'))
import metrics
sys.path.insert(0, str(ROOT / 'C/uniform-rate-20260909-v1'))
import support as p


def diagnose(checkpoint_path):
    audited = metrics.audit_checkpoint(checkpoint_path)
    cp = p.read(checkpoint_path)
    receipt = p.checked(cp['receipt'])
    binding = p.checked(cp['binding'])
    assert binding['hostname'] == 'iZwz9i5bte3xkpmcoes3t2Z' and binding['model'] == '32b'
    assert cp['row']['system'] != 'pdblend'
    directory = Path(cp['receipt']['path']).parents[2] / 'cells' / cp['row']['cell_id']
    requests = list(csv.DictReader((directory / 'bench.csv').open()))
    failed = [r for r in requests if not metrics.truth(r.get('success')) or r.get('error') or metrics.truth(r.get('request_timeout'))]
    assert len(failed) == audited['failed_requests']
    for request in failed:
        assert request['error'] == 'request_hard_timeout' and metrics.truth(request['request_timeout'])
        assert request.get('admission_rejection') in ('', None, '0', 'False', 'false')
        assert request.get('http_status') in ('', None, '200'), 'unknown server fault is not a capacity diagnosis'
    for side in ('before', 'after'):
        path = Path(cp['receipt']['path']).parent / ('identity.' + side + '.json')
        rows = p.read(path)
        assert len(rows) == len(binding['instances'])
        for actual, expected in zip(rows, binding['instances']):
            container = actual['container']
            assert container['Id'] == expected['container']['id'] and container['Image'] == expected['container']['image']
            assert container['State']['StartedAt'] == expected['container']['StartedAt']
            assert not container['State'].get('OOMKilled') and container['State']['Running']
            raw = actual['runtime']
            assert not raw.get('error') and not raw.get('runtime_error')
            assert all(not raw.get(key) for key in ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations'))
    audited.update(independently_recomputed=True, baseline_service_failure=bool(failed),
        failure_classification='declared_request_hard_timeout' if failed else None,
        service_failure_diagnosed=bool(failed), explicit_timeout_request_ids=[r['request_id'] for r in failed],
        unknown_http_faults_accepted=False, original_producer_preserved=True,
        slo_threshold_comparison='strict_lt')
    return audited


def initial_diagnosis():
    state = p.read(RESUME / 'pipeline-001/status.json')
    assert state.get('finished_s') and not p.active_owner(state) and state['phase'] == 'stopped_for_diagnosis'
    failed = p.read(RESUME / 'baseline-distserve-performance-001/status.json')
    assert len(failed['observed_checkpoints']) == 1 and not failed['completed'] and not p.active_owner(failed)
    reference = failed['observed_checkpoints'][0]
    assert p.sha(reference['path']) == reference['sha256']
    diagnosed = diagnose(reference['path'])
    assert diagnosed['failed_requests'] == diagnosed['request_timeouts'] == 2
    assert diagnosed['n_expected'] == 410 and diagnosed['completed_work_requests'] == 408
    path = HERE / 'predecessor-timeout-diagnosis.json'
    value = dict(schema='uniform-baseline-timeout-diagnosis-v1', independently_recomputed=True,
        slo_threshold_comparison='strict_lt', observations=[diagnosed],
        original_status=p.ref(RESUME / 'pipeline-001/status.json'), automatic_retry=False)
    if path.exists():
        assert p.read(path) == value
    else:
        p.save(path, value)
    return path


async def execute(system, out, state):
    diagnosis = initial_diagnosis()
    original_ref = p.ref(RESUME / ('baseline-' + system + '-release-001/release.json'))
    release = p.checked(original_ref)
    verifier = p.load(release['qualification_validator'], 'uniform_B_old_tail_qualification')
    qualified = verifier.verify(release['qualification'])
    assert qualified['passed'] and qualified['independently_recomputed']
    binding = p.checked(release['binding'])
    assert binding == p.checked(qualified['binding'])
    rows = release['rows'][1:] if system == 'distserve' else release['rows']
    assert [r['repeat'] for r in rows] == ([2] if system == 'distserve' else [1, 2])
    for row in rows:
        old_output = RESUME / ('baseline-' + system + '-performance-001') / 'results/operations' / row['cell_id']
        assert not old_output.exists(), 'predecessor already attempted this row'
    helper = p.load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'uniform_B_tail_meter')
    common = helper.load_common(binding['host_release'])
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    assert 'PDBLEND_NODE_LOCK_FD' not in os.environ
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in rows:
                assert not (HERE / 'STOP').exists()
                common.validate_binding(binding)
                state['current_cell'] = row['cell_id']
                state['attempted'].append(row['cell_id'])
                p.save(out / 'status.json', state)
                error = None
                try:
                    await common.run_one(session, binding, row, out / 'results', hardware)
                except BaseException as exc:
                    error = exc
                rp = out / 'results/operations' / row['cell_id'] / 'receipt.json'
                if rp.exists():
                    receipt = p.read(rp)
                    artifacts = {str(f): p.sha(f) for base in (rp.parent, out / 'results/cells' / row['cell_id'])
                        for f in base.rglob('*') if f.is_file()}
                    cp = dict(schema='uniform-original-baseline-continuation-checkpoint-v1', row=row,
                        release=original_ref, binding=release['binding'], qualification=release['qualification'],
                        qualification_validator=release['qualification_validator'], receipt=p.ref(rp), artifacts=artifacts,
                        predecessor_diagnosis=p.ref(diagnosis), measurement_valid=receipt.get('measurement_valid', False),
                        work_complete=receipt.get('summary', {}).get('work_complete', False),
                        execution_error=repr(error) if error else None, finished_s=time.time())
                    cp_path = out / 'results/checkpoints' / (row['cell_id'] + '.json')
                    p.save(cp_path, cp)
                    state['observed_checkpoints'].append(p.ref(cp_path))
                    if error is None:
                        try:
                            observation = diagnose(cp_path)
                            ap = out / 'audits' / (row['cell_id'] + '.json')
                            p.save(ap, observation)
                            state['observations'].append(p.ref(ap))
                            state['completed'].append(row['cell_id'])
                        except BaseException as exc:
                            error = exc
                if error is not None:
                    state['failed'].append(dict(cell_id=row['cell_id'], error=repr(error)))
                    p.save(out / 'status.json', state)
                    raise error
                assert row['cell_id'] in state['completed']
                p.save(out / 'status.json', state)
            state['complete'] = len(state['completed']) == len(rows)
        state['node_lease_held'] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=('distserve', 'dynamollm', 'ecoserve'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    diagnosis = initial_diagnosis()
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True, diagnosis=p.ref(diagnosis))))
        return
    assert args.system
    out = HERE / ('predecessor-' + args.system + '-001')
    out.mkdir(exist_ok=False)
    state = dict(schema='uniform-original-baseline-tail-status-v1', pid=os.getpid(),
        startticks=p.process_identity(os.getpid())['startticks'], started_s=time.time(), system=args.system,
        complete=False, node_lease_held=False, attempted=[], completed=[], failed=[], observed_checkpoints=[], observations=[])
    async def controlled():
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
        await execute(args.system, out, state)
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
