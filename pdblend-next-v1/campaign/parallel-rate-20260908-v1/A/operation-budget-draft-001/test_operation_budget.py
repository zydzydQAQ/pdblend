"""Actual executor CPU transactions; synthetic timing is never a measured bound."""
import asyncio
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

s = importlib.util.spec_from_file_location('operation_budget_executor', Path(__file__).with_name('capacity_executor.py'))
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


@dataclass
class Proposal:
    action: str = 'restore'
    gpus: tuple = (5,)
    remove_id: object = None
    duration_upper_s: float = 26.485642433
    action_energy_upper_j: float = 22981.6585


def fixture(tmp_path, monkeypatch, *, timeout=None, elapsed=27., native_failure=False):
    clock = [1000.]
    monkeypatch.setattr(m, 'time', SimpleNamespace(time=lambda: clock[0]))
    initial = [dict(id='original6', gpus=[6]), dict(id='original7', gpus=[7])]
    inventory = m.Inventory(tmp_path / 'inventory.json', initial, dict(cpu_only=True))

    class Backend:
        def allocate(self, gpus): return dict(id='synthetic-added5', gpus=list(gpus))
        async def begin_measurement(self, transaction): return object()
        async def end_measurement(self, meter):
            return dict(measurement_valid=True, duration_s=elapsed, energy_j=24000.)
        async def assert_spare(self, gpus): pass
        async def start(self, instance, before, limit):
            clock[0] += 19.
            return dict(cpu_only=True)
        async def verify(self, instance, limit):
            clock[0] = 1000. + elapsed
            if native_failure: raise ValueError('actual native/clock uncertainty remains fatal')
            if clock[0] >= limit: raise asyncio.TimeoutError()
            return dict(provenance=dict(cpu_only=True))
        async def stop_if_owned(self, instance, limit): pass

    class Adapter:
        async def reserve(self, operation, instance, proposal): pass
        async def commit(self, remove, add): pass
        async def unfreeze(self, iid): pass
        def contains(self, iid): return False

    async def validate(proposal): return True
    executor = m.PhysicalCapacityExecutor(Backend(), Adapter(), inventory, deadline_s=None,
        max_residents=8, validate_proposal=validate, lease_check=lambda: None,
        physical_operation_timeout_s=timeout)
    return executor, inventory


def test_original_empirical_budget_reproduces_failure_and_rollback(tmp_path, monkeypatch):
    executor, inv = fixture(tmp_path, monkeypatch)
    with pytest.raises(asyncio.TimeoutError): asyncio.run(executor.execute(Proposal()))
    assert executor.failed and len(inv.value['active_instances']) == 2
    assert any(e['kind'] == 'rollback_complete' for e in inv.value['events'])


def test_separate_safety_budget_completes_all_verification_and_records_miss(tmp_path, monkeypatch):
    executor, inv = fixture(tmp_path, monkeypatch, timeout=120)
    result = asyncio.run(executor.execute(Proposal()))
    assert result['execution_verified'] and len(inv.value['active_instances']) == 3
    miss = next(e for e in inv.value['events'] if e['kind'] == 'empirical_transition_estimate_exceeded')
    assert miss['actual_duration_s'] == 27 and miss['planning_duration_s'] == Proposal().duration_upper_s
    assert miss['request_or_native_failure_waived'] is False


def test_safety_budget_is_still_finite_and_rolls_back(tmp_path, monkeypatch):
    executor, inv = fixture(tmp_path, monkeypatch, timeout=120, elapsed=121)
    with pytest.raises(asyncio.TimeoutError): asyncio.run(executor.execute(Proposal()))
    assert executor.failed and len(inv.value['active_instances']) == 2


def test_native_or_frequency_failure_is_not_waived(tmp_path, monkeypatch):
    executor, inv = fixture(tmp_path, monkeypatch, timeout=120, native_failure=True)
    with pytest.raises(ValueError, match='uncertainty'): asyncio.run(executor.execute(Proposal()))
    assert executor.failed and len(inv.value['active_instances']) == 2


@pytest.mark.parametrize('timeout', [float('inf'), float('nan'), 0, -1, 361, True])
def test_unbounded_or_malformed_safety_budget_rejected(tmp_path, monkeypatch, timeout):
    with pytest.raises(ValueError): fixture(tmp_path, monkeypatch, timeout=timeout)


def test_safety_budget_cannot_truncate_empirical_estimate(tmp_path, monkeypatch):
    executor, inv = fixture(tmp_path, monkeypatch, timeout=20)
    with pytest.raises(ValueError): asyncio.run(executor.execute(Proposal()))
    assert not inv.value['events'] and len(inv.value['active_instances']) == 2
