"""Exclusive-node paired experiment queue; no automatic retries or result filtering."""
import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import sys
import time
import reference_map as main_references

PROTOCOL = 'per-dataset-slo-five-system-fixed-window-v1'
GLOBAL_DEADLINE = 1788872770.0400891


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


async def command(*args):
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(p.communicate(), 30)
    require(p.returncode == 0, err.decode(errors='replace')[-2000:])
    return out.decode()


async def http(session, instance, path, body=None, timeout=10):
    import aiohttp
    async with session.request('GET' if body is None else 'POST', instance['url'] + path,
            json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        text = await response.text()
        require(response.status == 200, instance['id'] + path + ': ' + text[:1000])
        return json.loads(text)


def idle(raw, instance):
    from ecopadg.serving.completion_policy import engine_residual
    require(raw.get('id') == instance['id'], 'wrong engine identity')
    residual = engine_residual(raw, time.time())
    require(not residual, 'engine residue: ' + json.dumps(residual))
    require(type(raw.get('generation')) is int, 'generation unobserved')
    if instance['native_kind'] == 'v3':
        counts = [raw.get(k) for k in ('transfer_send_started', 'transfer_send_completed', 'transfer_send_failed')]
        require(raw.get('transfer_send_counters_observed') is True and
            raw.get('transfer_inflight_sends_observed') is True and raw.get('transfer_send_healthy') is True
            and all(type(v) is int and v >= 0 for v in counts) and counts[0] == counts[1]
            and counts[2] == 0, 'unsettled native sender')
        require(raw.get('scheduler_budget_pending') is None, 'budget still pending')
    if instance.get('scheduler_cache_observed'):
        caches = [r.get('controls', {}).get('runtime') for r in raw.get('scheduler_io', [])]
        require(len(caches) == instance.get('scheduler_cache_count', 1) and all(c and c.get('generation') == raw['generation']
            and c.get('error') is None for c in caches), 'scheduler cache ACK missing')


async def wait_idle(session, instance, seconds=15):
    until = time.monotonic() + seconds
    while True:
        raw = await http(session, instance, '/runtime')
        try:
            idle(raw, instance)
            return raw
        except RuntimeError:
            if time.monotonic() >= until:
                raise
            await asyncio.sleep(.05)


async def identity(session, binding):
    names = (await command('docker', 'ps', '--format', '{{.Names}}')).split()
    expected = {i['container']['name'] for i in binding['instances']}
    require(set(names) == expected, 'unexpected running containers on energy-measured node')
    rows = json.loads(await command('docker', 'inspect', *names))
    by_name = {r['Name'].lstrip('/'): r for r in rows}
    evidence = []
    for i in binding['instances']:
        c = i['container']; actual = by_name[c['name']]
        require(actual['Id'] == c['id'] and actual['Image'] == c['image']
            and actual['State']['StartedAt'] == c['StartedAt'] and actual['State']['Running'],
            'engine process identity changed')
        provenance = await http(session, i, '/provenance')
        for key, expected_value in i['provenance'].items():
            require(provenance.get(key) == expected_value, 'engine provenance changed: ' + key)
        raw = await wait_idle(session, i)
        evidence.append(dict(container=actual, provenance=provenance, runtime=raw))
    return evidence


async def resume(session, instance, tokens=None):
    raw = await wait_idle(session, instance)
    payload = dict(generation=raw['generation'] + 1, role=instance.get('role', 'mixed'),
        mode='continuous', admit_prefill=True, admit_decode=True)
    if tokens is not None:
        payload['scheduler_budget'] = dict(schema_version=1, max_num_batched_tokens=tokens, max_num_seqs=32)
    result = await http(session, instance, '/control', payload)
    after = await wait_idle(session, instance)
    require(result.get('generation') == payload['generation'] and
        after.get('generation') == payload['generation'] and after.get('accepting') is True,
        'resume was not acknowledged')
    if tokens is not None:
        actual = after.get('scheduler_budget_effective', {})
        require(actual.get('max_num_batched_tokens') == tokens and actual.get('max_num_seqs') == 32,
            'actual scheduling budget differs')
    return dict(before=raw, control=payload, after=after)


def barrier(before, proof, instance):
    require(proof.get('drained') is True and proof.get('accepting') is False
        and proof.get('generation') == before['generation'] + 1
        and proof.get('drain_proof_type') == 'synchronous_put_owner_barrier', 'drain barrier missing')
    ranks = proof.get('transfers')
    require(isinstance(ranks, list) and len(ranks) == instance['tp'], 'all TP ranks not observed')
    for rank in ranks:
        require(rank.get('buffered_tensors') == 0 and rank.get('inflight_receives') == 0
            and rank.get('listener_alive') is True and not rank.get('allocations')
            and not rank.get('buffered_gpu_bytes'), 'rank transfer residue')
        if instance['native_kind'] == 'v3':
            counts = [rank.get(k) for k in ('send_started', 'send_completed', 'send_failed')]
            require(rank.get('send_counters_observed') is True and rank.get('send_healthy') is True
                and type(rank.get('inflight_sends')) is int and rank['inflight_sends'] == 0
                and all(type(v) is int and v >= 0 for v in counts) and counts[0] == counts[1] and counts[2] == 0,
                'rank sends not settled')


async def restore(session, instance):
    result = dict(complete=False, errors=[])
    try:
        before = await wait_idle(session, instance, seconds=20)
        result['before'] = before
        proof = await http(session, instance, '/drain', dict(expected_generation=before['generation']))
        result['proof'] = proof
        barrier(before, proof, instance)
    except BaseException as exc:
        result['errors'].append('native proof: ' + repr(exc))
    finally:
        try:
            result['resumed'] = await resume(session, instance, instance.get('restore_budget_tokens'))
        except BaseException as exc:
            result['errors'].append('resume: ' + repr(exc))
    result['complete'] = not result['errors']
    return result


def validate_binding(binding):
    require(binding['protocol_id'] == PROTOCOL and binding['hostname'] == socket.gethostname(), 'wrong host/protocol')
    require(binding['deadline_s'] == GLOBAL_DEADLINE, 'global deadline changed')
    for path, digest in binding['files'].items():
        require(sha(path) == digest, 'frozen input changed: ' + path)
    for path, reference in binding.get('large_inputs', {}).items():
        require(stat_identity(path) == reference['stat'], 'frozen large input changed: ' + path)
    require(len({i['id'] for i in binding['instances']}) == len(binding['instances']), 'duplicate engine')
    require(binding['system'] in ('pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve'), 'unknown system')
    for config in binding['configs'].values():
        strategy = read(config)['strategy']
        canonical = 'pdblend' if strategy.startswith('pdblend') else 'dynamollm' if strategy == 'dynamollm-resident' else strategy
        require(canonical == binding['system'], 'configuration does not implement the bound system')


async def run_one(session, binding, row, output, hardware):
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
    from ecopadg.serving.measurement import save_raw, power_evidence
    from ecopadg.metrics import clip_power_window
    config = binding['configs'][row['dataset']]
    require(sha(row['trace']) == row['trace_sha256'], 'trace changed')
    operation = output / 'operations' / row['cell_id']
    operation.mkdir(parents=True, exist_ok=False)
    before = await identity(session, binding)
    write(operation / 'identity.before.json', before)
    now = time.time()
    latest = min(now + 90, GLOBAL_DEADLINE - 100 - 120 - 90)
    require(now < latest, 'insufficient time for full window and cleanup')
    job = dict(row=row, config=config, out=str(output / 'cells' / row['cell_id']),
        latest_arrival_epoch_s=latest, execution_deadline_s=latest + 220,
        engine_ports=[i['port'] for i in binding['instances']])
    write(operation / 'job.json', job)
    receipt = dict(cell_id=row['cell_id'], system=row['system'], trace_sha256=row['trace_sha256'],
        n_expected=row['n_requests'], started_s=now, measurement_valid=False,
        energy_boundary='all-eight primary cell energy retained; full-operation energy overlaps and must not be added')
    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
    child = None
    started = None
    failure = None
    sampler.start()
    try:
        ready_until = min(time.monotonic() + 5, time.monotonic() + max(0, latest - time.time()))
        while True:
            require(not sampler.error, 'power sampler failed: ' + str(sampler.error))
            if len(sampler.samples) >= 2:
                break
            require(time.monotonic() < ready_until, 'power preflight timed out awaiting two real samples')
            await asyncio.sleep(.01)
        started = time.time()
        controls = await asyncio.gather(*(resume(session, i, i.get('service_budget_tokens')) for i in binding['instances']), return_exceptions=True)
        write(operation / 'controls.before.json', [dict(error=repr(c)) if isinstance(c, BaseException) else c for c in controls])
        require(not any(isinstance(c, BaseException) for c in controls), 'one or more initial controls failed')
        with (operation / 'child.log').open('xb') as log:
            child = await asyncio.create_subprocess_exec(sys.executable, '-u',
                str(Path(__file__).with_name('child.py')), str(operation / 'job.json'),
                stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            receipt['child_pid'] = child.pid
            write(operation / 'receipt.json', receipt)
            await asyncio.wait_for(child.wait(), max(.001, job['execution_deadline_s'] - time.time()))
        receipt['child_exitcode'] = child.returncode
        summary = read(Path(job['out']) / 'summary.json')
        receipt['summary'] = summary
        actual = read(Path(job['out']) / 'runtime_config.json')
        expected = read(config)
        expected.update(journal=str(Path(job['out']) / 'control.jsonl'), slo_scale=row['slo_scale'],
            slo_protocol='per-dataset-slo-v1', slo_attainment_target=.9,
            slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'], comparison_system=row['system'])
        require(actual == expected, 'actual controller configuration differs')
        require(summary.get('trace_sha256') == row['trace_sha256'] and
            summary.get('measurement_window_protocol') == PROTOCOL and
            summary.get('fixed_window_valid') is True and summary.get('measurement_valid') is True
            and summary.get('post_measurement_cleanup', {}).get('cleanup_complete') is True
            and child.returncode == 0, 'invalid cell retained')
        require(summary['measurement_end_s'] <= job['execution_deadline_s'], 'measurement exceeded reserved tail')
    except BaseException as exc:
        failure = exc
        receipt['error'] = repr(exc)
    finally:
        cleanup_end = min(GLOBAL_DEADLINE, time.time() + 90, job['execution_deadline_s'] + 90)
        async def bounded(coro, seconds):
            return await asyncio.wait_for(coro, max(.001, min(seconds, cleanup_end - time.time())))
        errors = receipt['outer_cleanup_errors'] = []
        try:
            if child is not None and child.returncode is None:
                child.terminate()
                try:
                    await bounded(child.wait(), 5)
                except asyncio.TimeoutError:
                    child.kill()
                    await bounded(child.wait(), 3)
        except BaseException as exc:
            errors.append('own child termination: ' + repr(exc))
        child_stopped = child is None or child.returncode is not None
        receipt['child_stopped'] = child_stopped
        try:
            if child is not None and child_stopped and failure is not None:
                dispatch = operation / 'dispatch.jsonl'
                owned = set()
                if dispatch.exists():
                    for line in dispatch.read_bytes().splitlines(keepends=True):
                        if line.endswith(b'\n'):
                            try:
                                r = json.loads(line)
                                require(type(r['port']) is int and isinstance(r['request_id'], str), 'bad ownership log')
                                owned.add((r['port'], r['request_id']))
                            except Exception as exc:
                                errors.append('ownership log: ' + repr(exc))
                cancellations = [http(session, i, '/cancel', dict(request_id=rid), timeout=3)
                    for i in binding['instances'] for port, rid in owned if i['port'] == port]
                if cancellations:
                    results = await bounded(asyncio.gather(*cancellations, return_exceptions=True), 8)
                    errors.extend('owned cancel: ' + repr(r) for r in results if isinstance(r, BaseException))
        except BaseException as exc:
            errors.append('owned cancellation: ' + repr(exc))
        receipt['restoration'] = {}
        try:
            require(child_stopped, 'native restoration requires verified child exit')
            restoration = await bounded(asyncio.gather(*(restore(session, i) for i in binding['instances']), return_exceptions=True), 60)
            receipt['restoration'] = {i['id']: (dict(complete=False, error=repr(r)) if isinstance(r, BaseException) else r)
                for i, r in zip(binding['instances'], restoration)}
        except BaseException as exc:
            errors.append('native restoration: ' + repr(exc))
        try:
            require(child_stopped, 'clock restoration requires verified child exit')
            from ecopadg.serving.backend import ClockOwner
            clocks = await asyncio.to_thread(ClockOwner, hardware, tuple(range(8)))
            await bounded(clocks.close(), 10)
            receipt['clock_restore_complete'] = True
        except BaseException as exc:
            receipt['clock_restore_complete'] = False
            receipt['clock_restore_error'] = repr(exc)
        ended = time.time()
        try:
            await asyncio.sleep(.1)
            await asyncio.to_thread(sampler.stop)
        except BaseException as exc:
            errors.append('sampler stop: ' + repr(exc))
        pdir = operation / 'power'; pdir.mkdir()
        try:
            save_raw(pdir, [], sampler.samples, sampler.utilization_samples,
                power_source=sampler.power_source, power_metadata=sampler.power_metadata)
            with (pdir / 'clocks.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['t_s'] + [f'gpu{i}_sm_mhz' for i in range(8)])
                writer.writerows([t, *v] for t, v in sampler.frequency_samples)
        except BaseException as exc:
            errors.append('raw power save: ' + repr(exc))
        receipt.update(operation_start_s=started, operation_end_s=ended, finished_s=time.time(),
            sampling_error=sampler.error, power_evidence=power_evidence(sampler.samples, sampler.power_source, sampler.power_metadata))
        try:
            receipt['full_operation_energy_j'] = trapezoid_energy(clip_power_window(sampler.samples, started, ended, pad_s=0)) if started else None
        except Exception as exc:
            receipt['integration_error'] = repr(exc)
        try:
            after = await identity(session, binding)
            write(operation / 'identity.after.json', after)
            validate_binding(binding)
        except BaseException as exc:
            errors.append('final source/process identity: ' + repr(exc))
        receipt['measurement_valid'] = bool(failure is None and not errors
            and len(receipt['restoration']) == len(binding['instances']) and all(r['complete'] for r in receipt['restoration'].values())
            and receipt['clock_restore_complete'] and not sampler.error and not receipt.get('integration_error')
            and receipt['power_evidence']['power_source_verified'])
        write(operation / 'receipt.json', receipt)
    require(receipt['measurement_valid'], 'invalid measurement/cleanup preserved: ' + str(receipt.get('error')))
    return receipt


async def sweep(args, binding):
    import aiohttp
    from ecopadg.measure.backends import PynvmlBackend
    validate_binding(binding)
    main_references.verify(args.main_reference_map, args.main_reference_sha256, args.manifest, args.binding,
        args.system, args.dataset or ('alpaca', 'sharegpt', 'longbench'))
    spec = read(args.manifest)
    require(spec['model'] == binding['model'] and args.system == binding['system']
        and spec['protocol_id'] == PROTOCOL and binding['files'].get(str(args.manifest.resolve())) == sha(args.manifest),
        'model/system/workload manifest not bound')
    selected_datasets = set(args.dataset or ('alpaca', 'sharegpt', 'longbench'))
    require(selected_datasets <= set(binding['configs']), 'deployment does not bind every selected dataset')
    rows = [r for r in spec['cells'] if r['system'] == args.system and r['phase'] == args.phase
            and r['dataset'] in selected_datasets]
    require(rows and all(r['seed'] == 701 and r['trace_duration_s'] == 100 for r in rows), 'wrong declared cells')
    output = Path(binding['output'])
    output.mkdir(parents=True, exist_ok=True)
    invocation = output / 'invocations' / (args.system + '-' + args.phase + '-' + str(time.time_ns()) + '.json')
    state = dict(started_s=time.time(), system=args.system, phase=args.phase, completed=[], complete=False,
        protocol_id=PROTOCOL, manifest_sha256=sha(args.manifest), binding_sha256=sha(args.binding), pid=os.getpid(),
        selected_datasets=sorted(selected_datasets), declared_selected_cells=len(rows),
        main_reference_map=str(args.main_reference_map), main_reference_sha256=args.main_reference_sha256)
    write(invocation, state)
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    task = asyncio.current_task()
    interrupted = False
    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            for row in rows:
                if (output / 'STOP').exists() or time.time() > GLOBAL_DEADLINE - 400:
                    state['stopped_at_boundary'] = True; break
                cp = output / 'checkpoints' / (row['cell_id'] + '.json')
                if cp.exists():
                    record = read(cp)
                    require(record['row'] == row and sha(record['receipt']) == record['receipt_sha256'], 'checkpoint changed')
                    require(read(record['receipt'])['measurement_valid'] is True, 'invalid checkpoint')
                    for path, digest in record['artifacts'].items():
                        require(sha(path) == digest, 'checkpoint artifact changed: ' + path)
                    continue
                validate_binding(binding)
                if args.phase == 'scale':
                    main_references.verify(args.main_reference_map, args.main_reference_sha256,
                        args.manifest, args.binding, args.system, selected_datasets, row=row)
                state['current_cell'] = row['cell_id']; write(invocation, state)
                receipt = await run_one(session, binding, row, output, hardware)
                rp = output / 'operations' / row['cell_id'] / 'receipt.json'
                artifacts = {str(path): sha(path) for base in (rp.parent, output / 'cells' / row['cell_id'])
                    for path in base.rglob('*') if path.is_file()}
                write(cp, dict(row=row, receipt=str(rp), receipt_sha256=sha(rp), completed_s=time.time(),
                    artifacts=artifacts, measurement_valid=True, work_complete=receipt['summary'].get('work_complete')))
                state['completed'].append(row['cell_id']); write(invocation, state)
                if len(state['completed']) >= args.max_cells:
                    break
        state['complete'] = True
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        state['finished_s'] = time.time(); write(invocation, state)
    print(json.dumps({k: v for k, v in state.items() if k != 'completed'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--system', required=True)
    p.add_argument('--dataset', action='append', choices=('alpaca', 'sharegpt', 'longbench'),
        help='Physical deployment group only; original workload declaration and rows remain unchanged')
    p.add_argument('--phase', choices=('scale',), default='scale')
    p.add_argument('--max-cells', type=int, default=1)
    p.add_argument('--main-reference-map', type=Path, required=True)
    p.add_argument('--main-reference-sha256', required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    require(args.max_cells == 1, 'scale adapter executes at most one new CP per invocation')
    require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'scale driver must acquire a fresh lease')
    main_references.verify(args.main_reference_map, args.main_reference_sha256, args.manifest, args.binding,
        args.system, args.dataset or ('alpaca', 'sharegpt', 'longbench'))
    binding = read(args.binding)
    host = Path(binding['host_release'])
    sys.path[:0] = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    os.environ['PYTHONPATH'] = ':'.join(sys.path[:3])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    validate_binding(binding)
    if args.run:
        from ecopadg.serving.campaign import node_lease
        with node_lease():
            asyncio.run(sweep(args, binding))
    else:
        print(json.dumps(dict(binding_valid=True, hardware_actions=False)))


if __name__ == '__main__':
    main()
