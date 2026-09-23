from contextlib import contextmanager
from dataclasses import replace

import pytest

from pdblend.control.reshard import (
    NativeReshardBackend, QuarantineRequired, ReshardCoordinator, TopologyTarget,
)
from pdblend.control.tp_modes import ResidentTPRouter, PoolMember, TransitionCost, SlowTPController


def test_resident_router_never_crosses_tp_or_generation():
    router = ResidentTPRouter([
        PoolMember('p1', 'm', 1, 1, 'tp1', 1), PoolMember('p2', 'm', 2, 1, 'tp2', 2)])
    assert router.route('a', 100, prefer_tp=2)['tp'] == 2
    assert router.route('b', 100, prefer_tp=4) is None


def test_resident_pd_requires_complete_matching_pair():
    p = PoolMember('p', 'm', 2, 1, 'same', 1, role='P')
    wrong_generation = PoolMember('d', 'm', 2, 1, 'same', 2, role='D')
    router = ResidentTPRouter([p, wrong_generation])
    assert router.route('a', 1024, prefer_pd=True) is None
    for changed in ({'generation': 1, 'tp': 1}, {'generation': 1, 'pool_id': 'elsewhere'},
                    {'generation': 1, 'model_id': 'another'}):
        router = ResidentTPRouter([p, replace(wrong_generation, **changed)])
        assert router.route('a', 1024, prefer_pd=True, model_id='m') is None
    router = ResidentTPRouter([p, replace(wrong_generation, generation=1)])
    result = router.route('a', 1024, prefer_pd=True)
    assert result['path'] == 'PD' and result['instance_ids'] == ['p', 'd']
    assert router.inflight() == {'p': 1, 'd': 1}
    assert router.release('a', generation=1, terminal_ack=True)
    assert router.inflight() == {'p': 0, 'd': 0}


def test_resident_capacity_generation_and_cancel_ownership():
    router = ResidentTPRouter([PoolMember('a', 'm', 1, 1, 'tp1', 3,
                                         capacity_tokens=2048, max_concurrent=2)])
    first = router.route('one', 1024, max_tokens=64)
    first['instance_ids'].clear()  # callers cannot mutate the ownership record
    assert router.route('two', 1024, max_tokens=64) is None
    router.set_accepting('a', False)
    assert router.route('one', 1024, max_tokens=64)['generation'] == 3
    assert router.route('new', 10) is None
    for generation, ack in ((2, True), (3, False)):
        with pytest.raises(ValueError, match='terminal ACK'):
            router.release('one', generation=generation, terminal_ack=ack)
    assert router.inflight()['a'] == 1
    assert router.release('one', generation=3, terminal_ack=True)
    assert not router.release('one', generation=3, terminal_ack=True)
    assert router.inflight()['a'] == 0


def test_resident_uses_length_capacity_and_measured_score():
    small = PoolMember('a', 'm', 1, 1, 'tp1', 1, capacity_tokens=1024)
    large = PoolMember('b', 'm', 2, 1, 'tp2', 1, capacity_tokens=8192)
    router = ResidentTPRouter([small, large], score=lambda m, n, tokens, reqs: m.tp + reqs)
    assert router.route('short', 128)['tp'] == 1
    assert router.route('long', 2048)['tp'] == 2
    assert router.route('oversized', 9000) is None


class ReceiptBackend:
    """Explicit test double, never a source of hardware qualification."""
    def __init__(self, corrupt=None):
        self.corrupt = corrupt or (lambda operation, receipt: None)
        self.operations = []
        self.lease_retained = False

    @contextmanager
    def gpu_lock(self, gpu_ids, transaction_id):
        try:
            yield {'locked': True, 'lease_id': 'test-lease', 'gpu_ids': gpu_ids,
                   'transaction_id': transaction_id}
        except QuarantineRequired:
            self.lease_retained = True
            raise

    def execute(self, operation, *, source, target, generation, source_generation,
                transaction_id, timeout_s):
        self.operations.append(operation)
        identity = dict(operation=operation, generation=generation, transaction_id=transaction_id)

        def acks(topology):
            return [dict(identity, rank=rank, gpu_id=gpu, ok=True)
                    for rank, gpu in enumerate(topology.gpu_ids)]

        receipt = dict(identity, rank_acks=acks(source if operation in ('drain', 'retire', 'rollback') else target))
        if operation == 'prepare':
            receipt.update(prepared=True, preparation='metadata_only')
        elif operation == 'drain':
            receipt.update(drained=True, inflight=0, live_kv_blocks=0, pending_transfers=0,
                           source_generation=source_generation)
        elif operation == 'transfer':
            receipt.update(weights_verified=True, live_kv_transferred=False, method='weight_transfer')
        elif operation == 'verify':
            receipt.update(output_ok=True, kv_ok=True, cancel_ok=True, live_kv_blocks=0,
                           golden_output_sha256='a' * 64, output_sha256='a' * 64)
        elif operation == 'activate':
            receipt.update(active=True)
        elif operation == 'retire':
            receipt.update(retired=True)
        elif operation == 'rollback':
            receipt.update(rolled_back=True, source_restored=True, target_quarantined=True,
                           source_generation=source_generation, target_rank_acks=acks(target))
        self.corrupt(operation, receipt)
        return receipt


def _targets():
    return TopologyTarget('m', 1, gpu_ids=(0,)), TopologyTarget('m', 2, gpu_ids=(0, 1))


def test_drained_cross_tp_is_distinct_from_symmetric_online_pd():
    backend = ReceiptBackend()
    coordinator = ReshardCoordinator(backend)
    source, target = _targets()
    result = coordinator.transition(source, target)
    assert result.state == 'complete' and coordinator.active == target
    assert coordinator.active_generation == 1 and not result.formal_eligible
    assert backend.operations == ['prepare', 'drain', 'transfer', 'verify', 'activate', 'retire']
    back = coordinator.transition(target, source)
    assert back.state == 'complete' and coordinator.active_generation == 2
    with pytest.raises(ValueError, match='newer'):
        coordinator.transition(source, target, generation=1)


@pytest.mark.parametrize('field', ['inflight', 'live_kv_blocks', 'pending_transfers'])
def test_cross_tp_cannot_transfer_until_all_source_work_drains(field):
    def corrupt(operation, receipt):
        if operation == 'drain':
            receipt[field] = 1
    backend = ReceiptBackend(corrupt)
    coordinator = ReshardCoordinator(backend)
    source, target = _targets()
    result = coordinator.transition(source, target)
    assert result.state == 'rollback' and 'transfer' not in backend.operations
    assert coordinator.active == source and coordinator.active_generation == 0


@pytest.mark.parametrize('failure', ['rank_missing', 'rank_stale', 'rank_duplicate', 'gpu_mismatch', 'bare_boolean'])
def test_native_ack_must_bind_every_rank_to_current_transaction(failure):
    def corrupt(operation, receipt):
        if operation != 'transfer':
            return
        if failure == 'rank_missing':
            receipt['rank_acks'].pop()
        elif failure == 'rank_stale':
            receipt['rank_acks'][0]['transaction_id'] = 'old'
        elif failure == 'rank_duplicate':
            receipt['rank_acks'][1]['rank'] = 0
        elif failure == 'gpu_mismatch':
            receipt['rank_acks'][0]['gpu_id'] = 7
        else:
            receipt.pop('rank_acks')
            receipt['rank_ack'] = True
    backend = ReceiptBackend(corrupt)
    coordinator = ReshardCoordinator(backend)
    result = coordinator.transition(*_targets())
    assert result.state == 'rollback' and 'activate' not in backend.operations


def test_uncertain_rollback_quarantines_and_retains_gpu_lease():
    def corrupt(operation, receipt):
        if operation == 'activate':
            receipt['active'] = False
        if operation == 'rollback':
            receipt['target_rank_acks'].pop()
    backend = ReceiptBackend(corrupt)
    coordinator = ReshardCoordinator(backend)
    result = coordinator.transition(*_targets())
    assert result.state == 'quarantine' and coordinator.active is None
    assert coordinator.quarantined_gpus == {0, 1} and backend.lease_retained
    with pytest.raises(RuntimeError, match='recovery'):
        coordinator.transition(*_targets())


def test_target_golden_output_failure_never_activates():
    def corrupt(operation, receipt):
        if operation == 'verify':
            receipt['output_sha256'] = 'b' * 64
    backend = ReceiptBackend(corrupt)
    result = ReshardCoordinator(backend).transition(*_targets())
    assert result.state == 'rollback' and 'activate' not in backend.operations


def test_native_bridge_does_not_fill_missing_receipts():
    backend = ReceiptBackend()
    bridge = NativeReshardBackend(lambda *args, **kwargs: {}, backend.gpu_lock)
    result = ReshardCoordinator(bridge).transition(*_targets())
    assert result.state == 'quarantine'


def test_transition_costs_are_finite_and_full_cost_is_recorded():
    with pytest.raises(ValueError):
        TransitionCost(startup_j=float('nan'))
    with pytest.raises(ValueError):
        TransitionCost(rollback_j=-1)
    controller = SlowTPController(ReshardCoordinator(ReceiptBackend()))
    cost = TransitionCost(startup_j=1, drain_j=2, weight_transfer_j=3, kv_transfer_j=4, rollback_j=5)
    result = controller.request(*_targets(), cost=cost)
    assert result['total_transition_j'] == 15 and not result['formal_eligible']
