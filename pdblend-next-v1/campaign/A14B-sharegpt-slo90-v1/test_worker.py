"""Pure CPU tests. Runtime, NVML and HTTP are replaced; no GPU action occurs."""
import asyncio
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('slo90_worker_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load('worker')
fixtures = load('test_report')


@pytest.fixture
def leased(tmp_path, monkeypatch):
    path = tmp_path / 'node-experiment.lock'
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(worker, 'NODE_LOCK', path)
        monkeypatch.setenv('PDBLEND_NODE_LOCK_FD', str(handle.fileno()))
        yield path, handle


def test_actual_inherited_flock_verified_without_unlocking(leased):
    path, handle = leased
    evidence = worker.verify_inherited_lease()
    assert evidence['fd'] == handle.fileno() and evidence['inode'] == path.stat().st_ino
    with path.open('a') as competitor:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_unlocked_or_wrong_descriptor_rejected(tmp_path, monkeypatch):
    path = tmp_path / 'node.lock'
    with path.open('a') as unlocked, (tmp_path / 'other.lock').open('a') as wrong:
        monkeypatch.setattr(worker, 'NODE_LOCK', path)
        monkeypatch.setenv('PDBLEND_NODE_LOCK_FD', str(unlocked.fileno()))
        with pytest.raises(ValueError, match='exclusive flock'):
            worker.verify_inherited_lease()
        monkeypatch.setenv('PDBLEND_NODE_LOCK_FD', str(wrong.fileno()))
        with pytest.raises(ValueError, match='actual node lock'):
            worker.verify_inherited_lease()
    monkeypatch.delenv('PDBLEND_NODE_LOCK_FD')
    with pytest.raises(ValueError, match='inherited'):
        worker.verify_inherited_lease()


def setup_attempt(tmp_path, monkeypatch, *, after_receipt_error=None, adapter_error=None):
    stage = tmp_path / 'cpu-fixture'
    stage.mkdir()
    summary, receipt, record, requests = fixtures.fixture(stage)
    # Below the new predeclared lag limits. These are CPU fixtures only.
    for request in requests:
        request['actual_dispatch_s'] = str(float(request['planned_arrival_s']) + .05)
        request['dispatch_delay_s'] = '.05'
    fixtures.write_csv(stage / 'bench.csv', requests)
    summary['dispatch_delay_max_s'] = .05
    out = tmp_path / 'attempt' / 'results'
    row = dict(record, protocol_id=worker.PROTOCOL, model='14b', dataset='sharegpt', seed=701,
               trace_duration_s=100, n_requests=10, trace=record['trace']['path'],
               trace_sha256=record['trace']['sha256'], executing_host=socket.gethostname(),
               dispatch_delay_max_limit_s=1., dispatch_delay_p99_limit_s=.1,
               receipt_path=str(out / 'operations' / record['cell_id'] / 'receipt.json'),
               summary_path=str(out / 'cells' / record['cell_id'] / 'summary.json'))
    host, common = tmp_path / 'host', tmp_path / 'common'
    binding = dict(protocol_id=worker.PROTOCOL, model='14b', system='pdblend',
                   hostname=socket.gethostname(), host_release=str(host), executor=str(common / 'run.py'))
    binding_path, row_path = tmp_path / 'binding.json', tmp_path / 'row.json'
    binding_path.write_text(json.dumps(binding))
    row_path.write_text(json.dumps(row))
    calls = []

    class Session:
        async def __aenter__(self):
            calls.append('session')
            return self

        async def __aexit__(self, *exc):
            calls.append('session_closed')

    async def run_one(session, actual_binding, actual_row, output, hardware):
        calls.append('run_one')
        assert isinstance(actual_row['trace'], str) and actual_row == row
        assert actual_binding == binding and output == out
        operation = output / 'operations' / actual_row['cell_id']
        cell = output / 'cells' / actual_row['cell_id']
        operation.mkdir(parents=True)
        cell.mkdir(parents=True)
        for name in ('bench.csv', 'power.csv'):
            shutil.copyfile(stage / name, cell / name)
        (cell / 'summary.json').write_text(json.dumps(summary))
        (operation / 'receipt.json').write_text(json.dumps(receipt))
        if after_receipt_error:
            raise after_receipt_error
        return receipt

    def validate_binding(actual):
        assert actual == binding
        calls.append('validate_binding')

    def runtime_loader(actual_host, actual_common):
        assert actual_host == host and actual_common == common
        calls.append('load_runtime')
        if adapter_error:
            raise adapter_error
        return SimpleNamespace(validate_binding=validate_binding, run_one=run_one,
                               sweep=lambda *args: pytest.fail('historical sweep must not execute'))

    original_loader = worker.load_local
    monkeypatch.setattr(worker, 'load_local', lambda name: SimpleNamespace(load_runtime=runtime_loader)
                        if name == 'runtime_adapter' else original_loader(name))
    monkeypatch.setattr(worker, 'make_hardware', lambda: calls.append('hardware') or object())
    monkeypatch.setattr(worker, 'make_session', Session)
    return dict(paths=(binding_path, host, common, row_path, out), row=row, binding=binding,
                summary=summary, receipt=receipt, requests=requests, calls=calls, stage=stage)


def test_success_uses_one_runtime_and_persists_original_receipt(tmp_path, monkeypatch, leased):
    case = setup_attempt(tmp_path, monkeypatch)
    result = asyncio.run(worker.run_point(*case['paths']))
    assert result['measurement_valid'], result
    assert result['exit_status'] == 0 and result['receipt_summary'] == case['summary']
    assert result['verdict']['dispatch_delay_p99_s'] == pytest.approx(.05)
    assert case['calls'] == ['load_runtime', 'validate_binding', 'hardware', 'session', 'run_one', 'session_closed']
    assert Path(result['meta_path']).is_file() and result['receipt_sha256']
    with pytest.raises(ValueError, match='previous attempt'):
        asyncio.run(worker.run_point(*case['paths']))


def test_lost_or_missing_lease_prevents_runtime_and_hardware(tmp_path, monkeypatch):
    case = setup_attempt(tmp_path, monkeypatch)
    monkeypatch.delenv('PDBLEND_NODE_LOCK_FD', raising=False)
    result = asyncio.run(worker.run_point(*case['paths']))
    assert result['exit_status'] == 2 and not result['hardware_initialized']
    assert not case['calls'] and 'inherited' in result['error']
    assert Path(result['meta_path']).is_file()


@pytest.mark.parametrize('error', [RuntimeError('native cleanup failed'), asyncio.CancelledError('stop requested')])
def test_execution_exception_cannot_be_hidden_by_valid_saved_summary(tmp_path, monkeypatch, leased, error):
    case = setup_attempt(tmp_path, monkeypatch, after_receipt_error=error)
    result = asyncio.run(worker.run_point(*case['paths']))
    assert result['exit_status'] == 2 and not result['verdict']['stop_eligible']
    assert result['receipt_summary'] == case['summary']
    assert result['observed_primary_energy_j'] == 80000 and result['observed_full_operation_energy_j'] == 90000
    assert result['receipt_sha256'] and result['summary_sha256']
    assert case['calls'][-1] == 'session_closed'


def test_mixed_runtime_import_rejected_before_hardware(tmp_path, monkeypatch, leased):
    case = setup_attempt(tmp_path, monkeypatch, adapter_error=ValueError('different serving runtime already imported'))
    result = asyncio.run(worker.run_point(*case['paths']))
    assert result['exit_status'] == 2 and not result['hardware_initialized']
    assert case['calls'] == ['load_runtime'] and 'already imported' in result['error']


def test_p99_lag_failure_does_not_modify_producer_summary_or_establish_endpoint(tmp_path, monkeypatch, leased):
    case = setup_attempt(tmp_path, monkeypatch)
    for request in case['requests']:
        request['actual_dispatch_s'] = str(float(request['planned_arrival_s']) + .2)
        request['dispatch_delay_s'] = '.2'
    fixtures.write_csv(case['stage'] / 'bench.csv', case['requests'])
    case['summary']['dispatch_delay_max_s'] = .2
    original = copy.deepcopy(case['summary'])
    result = asyncio.run(worker.run_point(*case['paths']))
    assert result['exit_status'] == 2 and 'arrival-lag bound' in result['error']
    assert result['receipt_summary'] == original and not result['verdict']['below_slo90']
    assert 'dispatch_delay_p99_s' not in result['receipt_summary']  # no fabricated producer field


def test_cli_in_fresh_process_without_lease_retains_error_meta(tmp_path, monkeypatch):
    case = setup_attempt(tmp_path, monkeypatch)
    binding, host, common, row, out = case['paths']
    environment = dict(os.environ)
    environment.pop('PDBLEND_NODE_LOCK_FD', None)
    child = subprocess.run([sys.executable, str(Path(worker.__file__)), 'run-point', '--binding', str(binding),
                            '--host-release', str(host), '--common', str(common), '--row', str(row), '--out', str(out)],
                           env=environment, capture_output=True, text=True, timeout=10)
    assert child.returncode == 2, child.stderr
    meta = json.loads((out.parent / 'point-result.json').read_text())
    assert not meta['hardware_initialized'] and not meta['verdict']['stop_eligible']


def test_actual_lock_fd_survives_fresh_process_inheritance(leased):
    path, handle = leased
    code = ('import importlib.util; from pathlib import Path; '
            f's=importlib.util.spec_from_file_location("w", {str(Path(worker.__file__))!r}); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            f'm.NODE_LOCK=Path({str(path)!r}); print(m.verify_inherited_lease()["fd"])')
    child = subprocess.run([sys.executable, '-c', code], pass_fds=(handle.fileno(),),
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stderr
    assert int(child.stdout) == handle.fileno()
