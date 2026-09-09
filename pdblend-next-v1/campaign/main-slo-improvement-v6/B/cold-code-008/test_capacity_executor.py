import asyncio
import json
from pathlib import Path
import sys
import time
import hashlib
import os
import fcntl
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_executor import Inventory, PhysicalCapacityExecutor


class Adapter:
    def __init__(self, initial):
        self.instances = {i['id']:i for i in initial}
        self.frozen = set()
        self.events = []
    async def reserve(self, action, instance, proposal):
        if action == 'remove':
            self.frozen.add(instance['id'])
    async def commit(self, removed, added):
        self.instances = {k:v for k,v in self.instances.items() if k not in removed}
        self.instances.update({i['id']:i for i in added})
        self.events.append(('commit', tuple(removed), tuple(i['id'] for i in added)))
    async def unfreeze(self, iid):
        self.frozen.discard(iid)
    async def freeze_and_require_idle(self, instance):
        self.frozen.add(instance['id'])
    def contains(self, iid):
        return iid in self.instances


class Backend:
    def __init__(self, inventory):
        self.inventory = inventory
        self.fail = None
        self.calls = []
        self.n = 0
    def allocate(self, gpus):
        self.n += 1
        return dict(id='new'+str(self.n), gpus=list(gpus), tp=2, port=34000+self.n,
                    kv_port=55000+32*self.n, native_kind='v3', container_name='owned'+str(self.n))
    async def assert_spare(self, gpus):
        pass
    async def start(self, instance, before, limit):
        raw = json.loads(self.inventory.path.read_text())
        assert raw['known_instances'][instance['id']]['state'] == 'start_intent'
        self.calls.append('start')
        return dict(container={'id':'cid'})
    async def verify(self, instance, limit):
        if self.fail == 'verify':
            raise RuntimeError('bad numerical output')
        if self.fail == 'cancel':
            raise asyncio.CancelledError()
        return dict(provenance={'actual_source':'verified'})
    async def drain_idle(self, instance, limit):
        self.calls.append('drain')
    async def stop(self, instance, limit):
        self.calls.append('stop')
    async def stop_if_owned(self, instance, limit):
        self.calls.append('rollback_stop')
        if self.fail == 'rollback':
            raise RuntimeError('stop unconfirmed')
    async def is_same_running(self, instance):
        return False
    async def resume_idle(self, instance, limit):
        self.calls.append('resume')
    async def begin_measurement(self, transaction):
        if self.fail == 'meter':
            raise RuntimeError('instant power unavailable')
        return 'meter'
    async def end_measurement(self, meter):
        return dict(measurement_valid=self.fail != 'energy', energy_j=123., duration_s=1.)


def setup(tmp_path):
    initial = [dict(id='original0', gpus=[0,1]), dict(id='original1', gpus=[2,3])]
    inventory = Inventory(tmp_path/'inventory.json', initial, {})
    adapter = Adapter(initial)
    backend = Backend(inventory)
    executor = PhysicalCapacityExecutor(backend, adapter, inventory,
        deadline_s=time.time()+1000, max_residents=4, lease_check=lambda:None)
    return executor, backend, adapter, inventory


def execute(executor, action='restore', iid=None):
    return executor._execute(action, (4,5), iid, 90, kind='developer_calibration')


def test_add_stop_real_ownership_and_top_level_provenance(tmp_path):
    e,b,a,i = setup(tmp_path)
    result = asyncio.run(execute(e))
    assert result['execution_verified']
    assert i.value['active_instances'][-1]['provenance'] == {'actual_source':'verified'}
    assert len(a.instances) == 3
    asyncio.run(execute(e, 'remove', 'new1'))
    assert len(a.instances) == 2
    assert i.value['known_instances']['new1']['state'] == 'stopped'
    assert b.calls == ['start','drain','stop']
    assert not a.frozen and not i.value['transition_inflight']


def test_sampler_start_failure_performs_no_physical_action(tmp_path):
    e,b,a,i = setup(tmp_path)
    b.fail = 'meter'
    with pytest.raises(RuntimeError, match='power'):
        asyncio.run(execute(e))
    assert b.calls == [] and not a.frozen and not i.value['transition_inflight']
    assert e.failed


@pytest.mark.parametrize('failure', ['verify','cancel'])
def test_failed_or_cancelled_add_stops_owned_new_only(tmp_path, failure):
    e,b,a,i = setup(tmp_path)
    b.fail = failure
    with pytest.raises((RuntimeError, asyncio.CancelledError)):
        asyncio.run(execute(e))
    assert b.calls == ['start','rollback_stop']
    assert len(a.instances) == 2 and i.value['known_instances']['new1']['state'] == 'stopped_after_failure'
    assert e.failed and not i.value['transition_inflight']


def test_invalid_energy_disables_future_policy(tmp_path):
    e,b,a,i = setup(tmp_path)
    b.fail = 'energy'
    with pytest.raises(ValueError, match='energy invalid'):
        asyncio.run(execute(e))
    assert e.failed and len(a.instances) == 3


def test_cleanup_is_separate_from_policy_shrink(tmp_path):
    e,b,a,i = setup(tmp_path)
    asyncio.run(execute(e))
    asyncio.run(e.finish_to_initial())
    assert len(a.instances) == 2 and i.value['complete']
    assert any(x['kind']=='transition_begin' and x['scope']=='measurement_cleanup' for x in i.value['events'])


def test_never_remove_original_resident(tmp_path):
    e,b,a,i = setup(tmp_path)
    asyncio.run(execute(e))
    with pytest.raises(ValueError, match='original'):
        asyncio.run(e._execute('remove', (0,1), 'original0', 90, kind='developer_calibration'))
    assert len(a.instances) == 3


def test_start_cannot_consume_cleanup_deadline(tmp_path):
    e,b,a,i = setup(tmp_path)
    e.deadline_s = time.time()+100
    with pytest.raises(ValueError, match='deadline'):
        asyncio.run(execute(e))
    assert not b.calls


def test_event_preserves_command_timestamp_without_duplicate_keywords(tmp_path):
    e,b,a,i = setup(tmp_path)
    event = i.event('physical_command', at_s=123., argv=['docker','inspect','owned'])
    assert event['at_s'] == 123. and event['kind'] == 'physical_command'


def test_cleanup_resolves_unpublished_create_intents(tmp_path):
    e,b,a,i = setup(tmp_path)
    created = b.allocate((4,5))
    i.intent(created, 'start', 'test')
    asyncio.run(e.finish_to_initial())
    assert b.calls == ['rollback_stop']
    assert i.value['known_instances']['new1']['state'] == 'stopped_after_failure'
    assert i.value['complete']


def test_pending_cleanup_failure_cannot_mark_complete(tmp_path):
    e,b,a,i = setup(tmp_path)
    i.intent(b.allocate((4,5)), 'start', 'test')
    b.fail = 'rollback'
    with pytest.raises(RuntimeError, match='unconfirmed'):
        asyncio.run(e.finish_to_initial())
    assert not i.value['complete']


def test_failed_remove_rollback_keeps_target_frozen(tmp_path):
    e,b,a,i = setup(tmp_path)
    asyncio.run(execute(e))
    async def broken_stop(instance, limit):
        raise RuntimeError('daemon ambiguous')
    b.stop = broken_stop
    b.fail = 'rollback'
    with pytest.raises(RuntimeError, match='daemon'):
        asyncio.run(execute(e,'remove','new1'))
    assert 'new1' in a.frozen and e.failed


def test_actual_inherited_parent_flock_is_required(tmp_path):
    lock_path = tmp_path/'node.lock'
    release = tmp_path/'release.json'
    release.write_text('{"approved": true}')
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        st = os.fstat(lock.fileno())
        ticks = int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        claim = dict(schema='capacity-parent-lease-authority-v1',fd=lock.fileno(),holder_pid=os.getpid(),
            holder_start_ticks=ticks,lock_path=str(lock_path),lock_device=st.st_dev,lock_inode=st.st_ino,
            capacity_inventory_path='expected-inventory',expected_job_path='expected-job',
            invocation=dict(path=str(release),sha256=digest(release)))
        authority = tmp_path/'authority.json'
        authority.write_text(json.dumps(claim))
        program = ('from capacity_executor import check_lease;check_lease('+repr(str(lock_path))+
            ',authority='+repr(dict(path=str(authority),sha256=digest(authority)))+
            ',expected_inventory="expected-inventory",expected_job_path="expected-job")')
        env = dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parent),PYTHONDONTWRITEBYTECODE='1')
        good = subprocess.run([sys.executable,'-c',program],env=env,pass_fds=(lock.fileno(),),
                              text=True,capture_output=True)
        assert good.returncode == 0,good.stderr
        bad = subprocess.run([sys.executable,'-c',program],env=env,text=True,capture_output=True)
        assert bad.returncode != 0


def test_expired_proposal_is_dropped_without_poisoning_or_allocating(tmp_path):
    from capacity_executor import ProposalDropped
    from types import SimpleNamespace
    e,b,a,i=setup(tmp_path)
    async def stale(_): return False
    e.validate_proposal=stale
    proposal=SimpleNamespace(action='restore',gpus=(4,5),remove_id=None,duration_upper_s=10)
    with pytest.raises(ProposalDropped):
        asyncio.run(e.execute(proposal))
    assert not b.calls and b.n==0 and not e.failed
    assert not i.value['transition_inflight'] and not a.frozen
    # A subsequent independently recomputed declaration can still execute.
    assert asyncio.run(execute(e))['execution_verified'] is True
