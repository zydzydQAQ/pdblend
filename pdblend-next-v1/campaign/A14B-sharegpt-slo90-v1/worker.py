"""One fresh-process physical cell. The parent must already own the node lease.

``--out`` is a unique attempt's run_one output root; its sibling
``point-result.json`` retains the verdict and original receipt summary even
when execution fails. This module never calls a historical sweep or retries.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
NODE_LOCK = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')
PROTOCOL = 'a14b-sharegpt-slo90-v1'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_local(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('slo90_worker_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def start_ticks(pid):
    return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])


def verify_inherited_lease():
    """Observe the actual inherited flock; never acquire or unlock it here."""
    raw = os.environ.get('PDBLEND_NODE_LOCK_FD')
    require(raw is not None and raw.isdecimal() and int(raw) >= 3,
            'a valid inherited PDBLEND_NODE_LOCK_FD is required')
    fd = int(raw)
    actual, expected = os.fstat(fd), NODE_LOCK.stat()
    require((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
            'inherited descriptor is not the actual node lock')
    info = Path('/proc/self/fdinfo', str(fd)).read_text()
    locks = [line for line in info.splitlines() if line.startswith('lock:')]
    require(any(' FLOCK ' in line and ' WRITE ' in line for line in locks),
            'inherited node descriptor does not hold an exclusive flock')
    return dict(fd=fd, device=actual.st_dev, inode=actual.st_ino,
                lock_path=str(NODE_LOCK), fdinfo=info, worker_pid=os.getpid(),
                worker_start_ticks=start_ticks(os.getpid()), parent_pid=os.getppid(),
                parent_start_ticks=start_ticks(os.getppid()))


def make_hardware():
    # Import only after frozen-runtime and actual-lease validation.
    from ecopadg.measure.backends import PynvmlBackend
    return PynvmlBackend(power_mode='instant')


def make_session():
    import aiohttp
    return aiohttp.ClientSession(trust_env=False)


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def prepare_row(row, binding, output):
    require(row.get('protocol_id') == binding.get('protocol_id') == PROTOCOL, 'new protocol identity required')
    require(row.get('model') == binding.get('model') == '14b' and row.get('dataset') == 'sharegpt',
            'only the declared 14B ShareGPT workload is allowed')
    require(row.get('system') == binding.get('system'), 'row and binding systems differ')
    require(row.get('seed') == 701 and row.get('trace_duration_s') == 100, 'wrong seed or arrival window')
    require(row.get('source_version') and isinstance(row['source_version'], str), 'frozen source version required')
    require(row.get('executing_host') == binding.get('hostname') == socket.gethostname(),
            'row/binding/physical execution host differ')
    require(row.get('dispatch_delay_max_limit_s') == 1.0 and row.get('dispatch_delay_p99_limit_s') == .1,
            'frozen arrival-lag bounds must be max 1 s and linear p99 0.1 s')
    require(isinstance(row.get('trace'), str) and row.get('trace_sha256'), 'runtime row requires string trace and digest')
    require(isinstance(row.get('cell_id'), str) and row['cell_id'] and
            Path(row['cell_id']).name == row['cell_id'] and row['cell_id'] not in ('.', '..'),
            'cell identity must be a single path component')
    require(float(row['slo_scale']) in (.5, 2.) and row.get('n_requests', 0) > 0, 'invalid scale or offered workload')
    for key, path in (
            ('receipt_path', output / 'operations' / row['cell_id'] / 'receipt.json'),
            ('summary_path', output / 'cells' / row['cell_id'] / 'summary.json')):
        require(Path(row.get(key, path)).resolve() == path, key + ' is outside this new attempt')
    audit_row = dict(row, trace=dict(path=row['trace'], sha256=row['trace_sha256']),
                     raw_dir=str(output / 'cells' / row['cell_id']),
                     receipt_path=str(output / 'operations' / row['cell_id'] / 'receipt.json'),
                     summary_path=str(output / 'cells' / row['cell_id'] / 'summary.json'))
    return audit_row


async def run_point(binding_path, host_release, common, row_path, output):
    """Called once per worker process; returns persisted meta, including failure."""
    binding_path, host_release, common, row_path, output = map(
        lambda path: Path(path).resolve(), (binding_path, host_release, common, row_path, output))
    meta_path = output.parent / 'point-result.json'
    require(not meta_path.exists() and not meta_path.with_name(meta_path.name + '.tmp').exists(),
            'fresh point-result location required; previous attempt is retained')
    report = load_local('report')
    meta = dict(schema='a14b-sharegpt-slo90-point-result-v1', protocol_id=PROTOCOL,
                pid=os.getpid(), started_s=time.time(), complete=False, measurement_valid=False,
                hardware_initialized=False, run_one_entered=False, error=None,
                meta_path=str(meta_path), output=str(output), receipt_summary=None,
                verdict=dict(measurement_valid=False, stop_eligible=False, below_slo90=False),
                energy_windows_must_not_be_added=True)
    row = None
    audit_row = None
    interrupted = False
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    installed_signals = []

    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            task.cancel()

    try:
        require(not output.exists(), 'fresh run_one output directory required')
        row, binding = read(row_path), read(binding_path)
        audit_row = prepare_row(row, binding, output)
        meta.update(cell_id=row['cell_id'], executing_host=row['executing_host'],
                    source_version=row['source_version'], receipt_path=audit_row['receipt_path'],
                    summary_path=audit_row['summary_path'], record=audit_row)
        meta['sources'] = {str(path): report.sha(path) for path in (binding_path, row_path)}
        require(Path(binding['host_release']).resolve() == host_release and
                Path(binding['executor']).resolve() == common / 'run.py', 'runtime arguments differ from binding')
        require(report.sha(row['trace']) == row['trace_sha256'], 'new workload trace digest differs')
        meta['node_lease_before'] = verify_inherited_lease()
        adapter = load_local('runtime_adapter')
        runtime = adapter.load_runtime(host_release, common)
        runtime.validate_binding(binding)
        meta['node_lease_before_hardware'] = verify_inherited_lease()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop)
            installed_signals.append(sig)
        hardware = await asyncio.to_thread(make_hardware)
        meta['hardware_initialized'] = True
        async with make_session() as session:
            meta['run_one_entered'] = True
            # The unchanged executor owns its child, cancellation, native drain,
            # clock restoration and both raw energy windows.
            await runtime.run_one(session, binding, row, output, hardware)
        require(all(report.sha(path) == digest for path, digest in meta['sources'].items()),
                'binding or single-point declaration changed during execution')
        meta['node_lease_after'] = verify_inherited_lease()
    except BaseException as exc:
        meta['error'] = repr(exc)
    finally:
        for sig in installed_signals:
            loop.remove_signal_handler(sig)
        if audit_row:
            receipt_path, summary_path = Path(audit_row['receipt_path']), Path(audit_row['summary_path'])
            receipt, summary = {}, {}
            try:
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    meta['receipt_sha256'] = report.sha(receipt_path)
                if summary_path.exists():
                    summary = read(summary_path)
                    meta['summary_sha256'] = report.sha(summary_path)
                else:
                    summary = receipt.get('summary', {})
                meta['receipt_summary'] = summary
                meta['receipt_error'] = receipt.get('error')
                meta['observed_primary_energy_j'] = summary.get('energy_j')
                meta['observed_full_operation_energy_j'] = receipt.get('full_operation_energy_j')
                verdict = report.validate_result(summary, receipt, audit_row)
                # Even an otherwise valid saved receipt cannot erase a worker
                # exception, changed lease, or cancellation after measurement.
                if meta['error'] or interrupted:
                    verdict.update(measurement_valid=False, stop_eligible=False, below_slo90=False,
                                   slo_target_met=False, status='technical_failure',
                                   technical_error=meta['error'] or 'worker interrupted')
                meta['verdict'] = verdict
                meta['measurement_valid'] = verdict['measurement_valid']
                if not verdict['measurement_valid'] and not meta['error']:
                    meta['error'] = verdict['technical_error']
            except BaseException as exc:
                meta['audit_error'] = repr(exc)
                meta['measurement_valid'] = False
                meta['verdict'] = dict(measurement_valid=False, stop_eligible=False, below_slo90=False,
                                       technical_error=repr(exc), status='technical_failure')
        meta.update(complete=True, interrupted=interrupted, finished_s=time.time(),
                    exit_status=0 if meta['measurement_valid'] else 2)
        atomic_write(meta_path, report.clean(meta))
    return report.clean(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    point = commands.add_parser('run-point')
    for name in ('binding', 'host-release', 'common', 'row', 'out'):
        point.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(run_point(args.binding, args.host_release, args.common, args.row, args.out))
    print(json.dumps({key: result.get(key) for key in ('meta_path', 'cell_id', 'measurement_valid', 'error', 'exit_status')}))
    return result['exit_status']


if __name__ == '__main__':
    raise SystemExit(main())
