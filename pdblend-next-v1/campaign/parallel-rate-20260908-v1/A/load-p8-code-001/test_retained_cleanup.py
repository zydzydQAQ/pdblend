"""An already-failed control task cannot waive actual terminal ownership checks."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import copy
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retained_cleanup import resource_barrier


def fixture():
    initial = [dict(id='original6'), dict(id='original7')]
    clock = SimpleNamespace(files=[SimpleNamespace(closed=True)], pending_physical_commands={},
                            physical_command_uncertainty=[], applied={})
    controller = SimpleNamespace(active={}, request_tasks=set(), clock_owner=clock,
        backend=SimpleNamespace(inflight_actions=0), capacity_task=SimpleNamespace(done=lambda: True))
    inventory = dict(complete=True, transition_inflight=False, active_instances=initial,
                     initial_ids=['original6', 'original7'], known_instances={'extra5': dict(state='stopped_after_failure')})
    service = SimpleNamespace(inventory=SimpleNamespace(value=inventory))
    return controller, service, initial


def test_finished_failed_task_does_not_prevent_independent_safe_restoration():
    c, s, initial = fixture()
    assert resource_barrier(c, s, initial, True)


@pytest.mark.parametrize('change', ['capacity_unknown', 'active_request', 'request_task', 'physical_action',
    'live_task', 'clock_fd', 'pending_clock', 'clock_uncertainty', 'clock_applied',
    'inventory_incomplete', 'inflight', 'wrong_initial', 'extra_unstopped'])
def test_unknown_active_or_unrestored_physical_state_cannot_resume(change):
    c, s, initial = fixture()
    if change == 'active_request': c.active['r'] = {}
    elif change == 'request_task': c.request_tasks.add(1)
    elif change == 'physical_action': c.backend.inflight_actions = 1
    elif change == 'live_task': c.capacity_task.done = lambda: False
    elif change == 'clock_fd': c.clock_owner.files[0].closed = False
    elif change == 'pending_clock': c.clock_owner.pending_physical_commands['unknown'] = 1
    elif change == 'clock_uncertainty': c.clock_owner.physical_command_uncertainty.append('write')
    elif change == 'clock_applied': c.clock_owner.applied[5] = 2520
    elif change == 'inventory_incomplete': s.inventory.value['complete'] = False
    elif change == 'inflight': s.inventory.value['transition_inflight'] = True
    elif change == 'wrong_initial': s.inventory.value['active_instances'] = [dict(id='other')]
    elif change == 'extra_unstopped': s.inventory.value['known_instances']['extra5']['state'] = 'started_unpublished'
    with pytest.raises(ValueError): resource_barrier(c, s, initial, change != 'capacity_unknown')


def test_same_physical_source_modules_as_actual_P8():
    import hashlib
    host = HERE.parents[1] / 'hosts/14b-capacity-p8'
    for name in ('capacity_executor.py', 'capacity_backend.py', 'capacity_runtime.py', 'capacity_planner.py', 'capacity_certificate.py'):
        assert (HERE / name).read_bytes() == (host / name).read_bytes()
