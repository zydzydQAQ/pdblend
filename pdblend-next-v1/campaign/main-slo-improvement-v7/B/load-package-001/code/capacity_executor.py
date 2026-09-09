"""Physical mixed-replica transactions, with durable ownership before side effects.

The adapter locks only admission/commit boundaries. Existing replicas keep serving
while an independent new replica boots. Backend methods are bounded and source
pinned; a failed transition is never reported as a successful policy decision.
"""
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed(reference):
    require(sha(reference['path']) == reference['sha256'], 'changed capacity input: ' + reference['path'])
    return json.loads(Path(reference['path']).read_text())


def durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_lease(path='/root/workspace/pdblend/new-results/campaigns/node-experiment.lock', *, authority=None,
                expected_inventory=None, expected_job_path=None):
    """Do not manufacture ownership by taking an unowned lock during a cell."""
    wanted = os.stat(path)
    open_inode = False
    for fd in Path('/proc/self/fd').iterdir():
        try:
            info = fd.stat()
            if (info.st_dev, info.st_ino) == (wanted.st_dev, wanted.st_ino):
                open_inode = True
        except OSError:
            pass
    inode = f'{os.major(wanted.st_dev):02x}:{os.minor(wanted.st_dev):02x}:{wanted.st_ino}'
    locks = [line.split() for line in Path('/proc/locks').read_text().splitlines()]
    holders = [int(row[4]) for row in locks if 'FLOCK' in row and 'WRITE' in row and inode in row and '->' not in row]
    require(len(holders) == 1, 'original node lease must have exactly one exclusive holder')
    holder = holders[0]
    if open_inode and holder == os.getpid():
        return
    require(authority is not None, 'measurement child needs explicit parent lease authority')
    claim = fixed(authority)
    require(claim.get('schema') == 'capacity-parent-lease-authority-v1'
            and claim.get('holder_pid') == holder and claim.get('lock_path') == path
            and claim.get('lock_device') == wanted.st_dev and claim.get('lock_inode') == wanted.st_ino,
            'actual parent lease differs from this invocation authority')
    fd = claim['fd']
    require(type(fd) is int and fd >= 0, 'actual inherited descriptor required')
    own = os.fstat(fd)
    require((own.st_dev, own.st_ino) == (wanted.st_dev, wanted.st_ino), 'inherited FD targets another inode')
    for owner in ('self', str(holder)):
        info = Path('/proc', owner, 'fdinfo', str(fd)).read_text().splitlines()
        require(any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line
                    and inode in line and str(holder) in line.split() for line in info),
                'descriptor does not share the parent actual FLOCK open-file description')
    require(claim.get('capacity_inventory_path') == expected_inventory
            and claim.get('expected_job_path') == expected_job_path,
            'parent lease authority is bound to another job/inventory')
    parent = os.getpid()
    ancestors = []
    while parent > 1:
        fields = Path('/proc', str(parent), 'stat').read_text().rsplit(')', 1)[1].split()
        if parent == holder:
            require(int(fields[19]) == claim['holder_start_ticks'], 'parent PID was reused')
            break
        ancestors.append(parent)
        parent = int(fields[1])
    require(parent == holder and ancestors, 'actual lease holder is not an ancestor of this measurement')
    fixed(claim['invocation'])  # The immutable release, not the circular live job.


class Inventory:
    """All known IDs are retained forever; active IDs change only after verification."""
    def __init__(self, path, initial, identity):
        self.path = Path(path)
        require(not self.path.exists(), 'new dynamic cell inventory required')
        self.value = dict(schema='capacity-live-inventory-v1', identity=identity,
                          pid=os.getpid(), version=0, initial_ids=[i['id'] for i in initial],
                          active_instances=list(initial), known_instances={}, events=[],
                          transition_inflight=False, complete=False)
        for instance in initial:
            self.value['known_instances'][instance['id']] = dict(instance, state='initial_active',
                owner_kind='retained_original', verified=True, changed_s=time.time())
        self.flush()

    def flush(self):
        self.value['updated_s'] = time.time()
        durable(self.path, self.value)

    def event(self, kind, **payload):
        value = dict(payload)
        value.setdefault('at_s', time.time())
        value['kind'] = kind
        self.value['events'].append(value)
        self.flush()
        return value

    def intent(self, instance, action, transaction):
        old = self.value['known_instances'].get(instance['id'], {})
        self.value['known_instances'][instance['id']] = {**old, **instance,
            'state':action + '_intent', 'owner_kind':old.get('owner_kind', 'created_for_cell'),
            'transaction':transaction, 'changed_s':time.time()}
        self.event(action + '_intent', instance_id=instance['id'], transaction=transaction)

    def verified(self, instance, state, **proof):
        old = self.value['known_instances'].get(instance['id'], {})
        self.value['known_instances'][instance['id']] = {**old, **instance,
            'state':state, 'verified':True, 'changed_s':time.time(), **proof}
        self.flush()

    def publish(self, instances):
        require(all(i['id'] in self.value['known_instances']
                    and self.value['known_instances'][i['id']].get('verified') is True for i in instances),
                'cannot publish unverified physical capacity')
        self.value['active_instances'] = list(instances)
        self.value['version'] += 1
        self.flush()


class ProposalDropped(ValueError):
    """A stale pre-hardware proposal is safe to recompute next epoch."""


class PhysicalCapacityExecutor:
    def __init__(self, backend, adapter, inventory, *, deadline_s, max_residents,
                 cleanup_reserve_s=120., validate_proposal=None, lease_check=check_lease):
        require(max_residents in (4, 8), 'authorized TP1/TP2 eight-card maximum required')
        self.backend, self.adapter, self.inventory = backend, adapter, inventory
        self.deadline_s, self.max_residents = deadline_s, max_residents
        self.cleanup_reserve_s = cleanup_reserve_s
        self.validate_proposal, self.lease_check = validate_proposal, lease_check
        self.lock = asyncio.Lock()
        self.inflight = None
        self.failed = False

    async def execute(self, proposal):
        require(not self.failed, 'prior transition failed; no automatic retry')
        require(self.validate_proposal is not None, 'measured capacity calibration required')
        return await self._execute(proposal.action, proposal.gpus, proposal.remove_id,
            proposal.duration_upper_s, proposal=proposal, kind='policy_transition')

    async def calibrate(self, operation, gpus, *, remove_id=None, declaration):
        """Explicit development measurement, never a certified planner proposal."""
        spec = fixed(declaration)
        require(spec.get('schema') == 'capacity-calibration-action-v1' and spec.get('authorized') is True
                and spec.get('operation') == operation and tuple(spec['gpus']) == tuple(gpus)
                and spec.get('deadline_s') == self.deadline_s
                and spec.get('automatic_retries') is False,
                'exact authorized calibration action required')
        require(0 < spec['work_budget_s'] <= 360, 'bounded calibration work budget required')
        return await self._execute(operation, tuple(gpus), remove_id, spec['work_budget_s'],
                                   kind='developer_calibration', declaration=declaration)

    async def _execute(self, operation, gpus, remove_id, budget_s, *, proposal=None,
                       kind, declaration=None):
        require(operation in ('restore', 'remove'), 'only mixed capacity addition/removal is supported')
        async with self.lock:
            self.lease_check()
            if proposal is not None and not await self.validate_proposal(proposal):
                raise ProposalDropped('expired/stale physical proposal before any hardware')
            if proposal is not None and time.time() + budget_s + self.cleanup_reserve_s >= self.deadline_s:
                raise ProposalDropped('deadline no longer covers proposal and cleanup')
            require(time.time() + budget_s + self.cleanup_reserve_s < self.deadline_s,
                    'same-day deadline cannot cover work and cleanup')
            before = list(self.inventory.value['active_instances'])
            require(2 <= len(before) <= self.max_residents, 'authorized resident count differs')
            if operation == 'restore':
                require(len(before) < self.max_residents, 'eight-GPU maximum reached')
                require(not set(gpus) & {g for i in before for g in i['gpus']}, 'restore GPUs are in use')
                instance = self.backend.allocate(gpus)
            else:
                require(len(before) > 2 and remove_id not in self.inventory.value['initial_ids'],
                        'the two original residents must remain')
                instance = next(i for i in before if i['id'] == remove_id)
                require(tuple(instance['gpus']) == tuple(gpus), 'remove allocation changed')
            transaction = uuid.uuid4().hex
            limit = min(time.time() + budget_s, self.deadline_s - self.cleanup_reserve_s)
            # Reservation rechecks pending work and all native/controller residuals
            # under the admission lock, then freezes only a removal target.
            if proposal is not None:
                if not await self.validate_proposal(proposal):
                    raise ProposalDropped('expired/stale physical proposal before reservation')
            await self.adapter.reserve(operation, instance, proposal)
            self.inventory.value['transition_inflight'] = True
            self.inventory.intent(instance, 'start' if operation == 'restore' else 'stop', transaction)
            self.inventory.event('transition_begin', transaction=transaction, scope=kind,
                operation=operation, proposal=asdict(proposal) if proposal else None,
                declaration=declaration, before_ids=[i['id'] for i in before])
            started_s = time.time()
            meter = None
            target = None
            safe_to_unfreeze = True
            physical_attempted = False
            try:
                meter = await self.backend.begin_measurement(transaction)
                physical_attempted = True
                if operation == 'restore':
                    await self.backend.assert_spare(gpus)
                    physical = await self.backend.start(instance, before, limit)
                    instance.update(physical)
                    self.inventory.verified(instance, 'started_unpublished')
                    proof = await self.backend.verify(instance, limit)
                    instance['provenance'] = proof['provenance']
                    self.inventory.verified(instance, 'ready_unpublished', physical_proof=proof)
                    target = before + [instance]
                    await self.adapter.commit((), [instance])
                else:
                    await self.backend.drain_idle(instance, limit)
                    # Detach routes before stopping the process. No request may
                    # reserve its KV after the freeze was installed.
                    await self.adapter.commit((instance['id'],), [])
                    await self.backend.stop(instance, limit)
                    self.inventory.verified(instance, 'stopped', stopped_s=time.time())
                    target = [i for i in before if i['id'] != instance['id']]
                self.inventory.publish(target)
                result = dict(transaction=transaction, scope=kind, operation=operation,
                              started_s=started_s, finished_s=time.time(), execution_verified=True,
                              live_instances=target, instance_id=instance['id'])
                self.inventory.event('physical_commit', **result)
            except BaseException as exc:
                self.failed = True
                self.inventory.event('transition_failed', transaction=transaction, error=repr(exc))
                if physical_attempted:
                    recovery = asyncio.create_task(self._rollback(operation, instance, before, transaction))
                    try:
                        try:
                            await asyncio.shield(recovery)
                        except asyncio.CancelledError:
                            await recovery
                    except BaseException as cleanup:
                        safe_to_unfreeze = False
                        self.inventory.event('rollback_failed', transaction=transaction, error=repr(cleanup))
                raise
            finally:
                self.inventory.value['transition_inflight'] = False
                try:
                    if meter is not None:
                        measurement = await self.backend.end_measurement(meter)
                        self.inventory.event('transition_measurement', transaction=transaction, **measurement)
                except BaseException:
                    self.failed = True
                    raise
                finally:
                    if safe_to_unfreeze:
                        await self.adapter.unfreeze(instance['id'])
                    self.inventory.flush()
            result['measurement'] = measurement
            if measurement.get('measurement_valid') is not True:
                self.failed = True
            require(measurement.get('measurement_valid') is True, 'physical transition energy invalid')
            return result

    async def _rollback(self, operation, instance, before, transaction):
        limit = min(time.time() + self.cleanup_reserve_s, self.deadline_s)
        if operation == 'restore':
            if self.adapter.contains(instance['id']):
                await self.adapter.freeze_and_require_idle(instance)
                await self.adapter.commit((instance['id'],), [])
            await self.backend.stop_if_owned(instance, limit)
            self.inventory.verified(instance, 'stopped_after_failure')
            self.inventory.publish(before)
        else:
            # If a stop failed while the exact original process remains alive,
            # recover its original route. Otherwise retain the safe >=2 layout;
            # this is a recorded partial rollback, never a fake restored count.
            if await self.backend.is_same_running(instance):
                await self.backend.resume_idle(instance, limit)
                if not self.adapter.contains(instance['id']):
                    await self.adapter.commit((), [instance])
                self.inventory.verified(instance, 'active_after_rollback')
                self.inventory.publish(before)
            else:
                await self.backend.stop_if_owned(instance, limit)
                self.inventory.verified(instance, 'stopped_after_failure')
                self.inventory.publish([i for i in before if i['id'] != instance['id']])
                self.inventory.event('rollback_retained_safe_capacity', transaction=transaction,
                    full_previous_layout_restored=False, automatic_restart_attempted=False)
        self.inventory.event('rollback_complete', transaction=transaction)

    async def finish_to_initial(self):
        require(not self.lock.locked(), 'transition must quiesce before measurement cleanup')
        for instance in list(self.inventory.value['active_instances']):
            if instance['id'] not in self.inventory.value['initial_ids']:
                await self._execute('remove', tuple(instance['gpus']), instance['id'],
                    min(60., max(1., self.deadline_s-time.time()-self.cleanup_reserve_s-1)),
                    kind='measurement_cleanup')
        # A daemon may have created an object before returning its ID. Such an
        # intent is absent from routing, but still belongs to this measurement.
        for instance in list(self.inventory.value['known_instances'].values()):
            if (instance['id'] not in self.inventory.value['initial_ids']
                    and instance.get('state') not in ('stopped', 'stopped_after_failure')):
                self.lease_check()
                limit = min(time.time()+self.cleanup_reserve_s, self.deadline_s)
                self.inventory.value['transition_inflight'] = True
                self.inventory.intent(instance, 'stop', uuid.uuid4().hex)
                try:
                    await self.backend.stop_if_owned(instance, limit)
                    self.inventory.verified(instance, 'stopped_after_failure')
                finally:
                    self.inventory.value['transition_inflight'] = False
                    self.inventory.flush()
        require({i['id'] for i in self.inventory.value['active_instances']} == set(self.inventory.value['initial_ids']),
                'original two-instance layout not restored')
        self.inventory.value['complete'] = True
        self.inventory.event('measurement_cleanup_complete', policy_benefit_claim=False)
