"""B32B ascending cells, wrapping the unchanged original 100+120+90 measurement."""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualify_pdb_v1 as g
p = g.p
R = g.R


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def full(summary, row):
    return bool(summary.get('measurement_valid') is True and summary.get('work_complete') is True
                and summary.get('completed_work_requests') == row['n_requests']
                and summary.get('failed_requests', 0) == 0 and summary.get('request_timeouts', 0) == 0)


def load_release(reference):
    r = p.checked(reference)
    assert r['schema'] == 'B32B-ascending-cells-release-v1' and r['node'] == 'B'
    assert r['system'] in ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve')
    for path, digest in r['files'].items():
        assert p.sha(path) == digest, path
    b = p.checked(r['binding'])
    assert b['system'] == r['system'] and b['model'] == '32b'
    assert b['hostname'] == 'iZwz9i5bte3xkpmcoes3t2Z'
    assert all(i['tp'] == 2 and i.get('service_budget_tokens', 8192) == 8192 for i in b['instances'])
    assert r['arrival_limits'] == dict(max_s=1., p99_s=.1, method='linear at(n-1)*0.99')
    contract = g.load(Path(r['declaration_contract']['path']), 'ascending_B_declaration')
    rows = [contract.lookup(r['declaration'], '32b', 'alpaca', 4.5, r['system'], rep) for rep in (1, 2)]
    assert rows == r['rows'] and len(rows) == 2
    assert len({row['trace_sha256'] for row in rows}) == 1
    q = g.load(Path(r['qualification_validator']['path']), 'ascending_B_release_qualification')
    qualified = q.verify(r['qualification'])
    assert qualified['passed'] and qualified['independently_recomputed'] and qualified['node'] == 'B'
    if r['system'] == 'pdblend':
        original = p.checked(r['original_performance_release'])
        expected_cfg = original['configs']['fixed2']['alpaca']
        assert b['configs'] == {'alpaca': expected_cfg['path']}
        assert p.sha(expected_cfg['path']) == expected_cfg['sha256']
        assert b['instances'] == p.checked(qualified['binding'])['instances']
        assert r['host_manifest'] == qualified['host_manifest']
        assert r['profile'] == qualified['profile'] == original['profile_refs']['alpaca']
        cfg = p.read(expected_cfg['path'])
        assert cfg.get('max_service_frequency_mhz', 2520) == 2520
        for flag in ('capacity_integration_v1', 'idle_domain_reacquire_v1', 'clock_failure_fresh_confirmation_v1'):
            assert cfg.get(flag, False) is False
        assert 'idle_domain_reacquire_timeout_s' not in cfg
    else:
        assert b == p.checked(qualified['binding']), 'baseline exact saved-qualified binding required'
        assert qualified['system'] == r['system']
        boundary = p.checked(r['boundary'])
        assert boundary['node'] == 'B' and boundary['model'] == '32b' and boundary['dataset'] == 'alpaca'
        assert boundary['declaration'] == r['declaration'] and boundary['cap_rate_rps'] >= 4.5
        assert {row['cell_id'] for row in rows} <= set(boundary['required_new_baseline_cell_ids'])
    assert b['host_release'] == str(Path(r['host_manifest']['path']).parent)
    return r, b, contract


async def execute(reference, out, state):
    release, binding, contract = load_release(reference)
    assert socket.gethostname() == binding['hostname'] and 'PDBLEND_NODE_LOCK_FD' not in os.environ
    for previous in release['predecessors']:
        s = p.checked(previous)
        assert s.get('finished_s') and not s.get('node_lease_held') and not alive(s['pid'])
    helper = g.load(R / 'B/baseline-return-after-external-source-v1/execution.py', 'ascending_B_cell_original')
    common = helper.load_common(binding['host_release'])
    arrival = g.load(R / 'audit_cooperative_arrivals_v1.py', 'ascending_B_arrival_raw')
    from ecopadg.measure.backends import PynvmlBackend
    from ecopadg.serving.campaign import node_lease
    import aiohttp
    with node_lease():
        state['node_lease_held'] = True
        p.save(out / 'status.json', state)
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in release['rows']:
                if (out / 'STOP').exists() or (HERE / 'STOP').exists():
                    state['stopped_at_boundary'] = True
                    break
                common.validate_binding(binding)
                assert not (out / 'results/checkpoints' / (row['cell_id'] + '.json')).exists(), 'no automatic repeat of observed cells'
                state['current_cell'] = row['cell_id']
                state['attempted'].append(row['cell_id'])
                p.save(out / 'status.json', state)
                error = None
                receipt = None
                try:
                    receipt = await common.run_one(session, binding, row, out / 'results', hardware)
                except BaseException as exc:
                    error = exc
                rp = out / 'results/operations' / row['cell_id'] / 'receipt.json'
                if rp.exists():
                    receipt = p.read(rp)
                    artifacts = {str(f): p.sha(f) for base in (rp.parent, out / 'results/cells' / row['cell_id'])
                                 for f in base.rglob('*') if f.is_file()}
                    cp = dict(schema='B32B-ascending-checkpoint-v1', row=row, repeat=row['repeat'], node='B', arm='fixed2' if row['system'] == 'pdblend' else row['system'],
                        declaration=release['declaration'], release=reference, binding=release['binding'],
                        qualification=release['qualification'], qualification_validator=release['qualification_validator'],
                        host_manifest=release['host_manifest'], receipt=p.ref(rp), artifacts=artifacts,
                        measurement_valid=receipt.get('measurement_valid', False), work_complete=receipt.get('summary', {}).get('work_complete', False),
                        finished_s=time.time(), execution_error=repr(error) if error else None)
                    bench = out / 'results/cells' / row['cell_id'] / 'bench.csv'
                    try:
                        with bench.open() as f:
                            actual_arrival = arrival.recompute(list(csv.DictReader(f)), row['n_requests'])
                        cp['arrival_qualification'] = dict(passed=True, limits=release['arrival_limits'], raw=p.ref(bench), **actual_arrival)
                    except BaseException as exc:
                        cp['arrival_qualification'] = dict(passed=False, error=repr(exc))
                    path = out / 'results/checkpoints' / (row['cell_id'] + '.json')
                    p.save(path, cp)
                    state['observed_checkpoints'].append(p.ref(path))
                    if error is None and receipt.get('measurement_valid') and cp['arrival_qualification']['passed'] and full(receipt['summary'], row):
                        state['completed'].append(row['cell_id'])
                    else:
                        state['failed'].append(row['cell_id'])
                    p.save(out / 'status.json', state)
                if error is not None:
                    raise error
                assert row['cell_id'] in state['completed'], 'incomplete request/arrival/hardware: preserve CP, stop before any successor'
            p.save(out / 'identity.after.json', await common.identity(session, binding))
            state['complete'] = len(state['completed']) == len(release['rows']) and not state['failed']
        state['node_lease_held'] = False
    state['node_lease_held'] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    reference = p.ref(args.release)
    load_release(reference)
    if not args.run:
        print(json.dumps(dict(passed=True, cpu_only=True)))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(schema='B32B-ascending-cell-status-v1', pid=os.getpid(), started_s=time.time(), release=reference,
                 complete=False, node_lease_held=False, attempted=[], completed=[], failed=[], observed_checkpoints=[])
    async def controlled():
        task = asyncio.current_task()
        stopped = False
        def cancel():
            nonlocal stopped
            if not stopped:
                stopped = True
                task.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, cancel)
        await execute(reference, args.out, state)
    try:
        asyncio.run(controlled())
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state.update(finished_s=time.time(), node_lease_held=False)
        p.save(args.out / 'status.json', state)


if __name__ == '__main__':
    main()
