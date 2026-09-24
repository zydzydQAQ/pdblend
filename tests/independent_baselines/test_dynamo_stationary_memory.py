"""Synthetic CPU contracts: arithmetic cannot grant native target activation."""
from copy import deepcopy
import math

import pytest

from pdblend_baselines.dynamollm.stationary_memory import (
    ANCHOR, COMPONENTS, PHASES, audit_peak_budget, owner_allocation_inventory,
    owner_inventory_from_kv_receipt,
)
from pdblend_baselines.dynamollm.stationary_ipc import digest
from pdblend_baselines.dynamollm.stationary_tensors import tensor_plan
from tests.independent_baselines.test_dynamo_stationary_tensors import GEOMETRY, shapes


REF = dict(path='/synthetic/cpu-only.json', sha256='a'*64)


def fixture(source_tp=1, target_tp=2):
    plan = tensor_plan(source_gpus=[f'GPU-{i}' for i in range(source_tp)],
        target_gpus=[f'GPU-{i}' for i in range(target_tp)],
        source_shapes=shapes(source_tp), target_shapes=shapes(target_tp), geometry=GEOMETRY)
    owners = []
    for rank in range(source_tp):
        parameters, cursor = {}, 4096
        for name, shape in plan['source_shapes'].items():
            nbytes = math.prod(shape)*2
            parameters[name] = dict(segment_address=4096, allocation_bytes=None,
                                    storage_ptr=cursor, storage_bytes=nbytes)
            cursor += nbytes
        for row in parameters.values():
            row['allocation_bytes'] = cursor-4096+256
        owners.append(dict(source_rank=rank, gpu_uuid=f'GPU-{rank}',
            process=dict(pid=101+rank, start_ticks=1, boot_id='cpu-only'),
            parameters=parameters, evidence=REF))
    boards = []
    for gpu in sorted(set(plan['source_gpu_uuids']+plan['target_gpu_uuids'])):
        phases = {}
        for phase in PHASES:
            values = {name:dict(bytes=512, kind='conservative_increment_bound', evidence=REF) for name in COMPONENTS}
            remote = sum(p['bytes'] for p in plan['pieces']
                         if p['target_gpu_uuid'] == gpu and p['kind'] == 'direct_gpu_transfer')
            values['remote_weight_allocator_bytes']['bytes'] = max(512, remote+256)
            phases[phase] = values
        boards.append(dict(gpu_uuid=gpu, anchor_stage=ANCHOR, driver_api='cudaMemGetInfo',
            evidence=REF, observed_at_s=100., free_bytes=8_000_000, total_bytes=10_000_000,
            phases=phases))
    return plan, owners, boards


def review(plan, owners, boards, **kw):
    return audit_peak_budget(plan, owners, boards, fleet_gpu_uuids=[b['gpu_uuid'] for b in boards],
                             now_s=100.25, max_age_s=.5, **kw)


def packet(plan, owner):
    source, views = {}, []
    for name, shape in plan['source_shapes'].items():
        stride = [math.prod(shape[i+1:]) for i in range(len(shape))]
        source[name] = dict(shape=shape, stride=stride, storage_offset=0,
                            storage_ptr=owner['parameters'][name]['storage_ptr'])
    for piece in plan['pieces']:
        if piece['source_rank'] != 0 or piece['target_rank'] != 0 or piece['kind'] != 'retain_on_gpu':
            continue
        shape = list(piece['source_shape'])
        if piece['axis'] is not None:
            shape[piece['axis']] = piece['length']
        allocation = owner['parameters'][piece['parameter']]
        src = source[piece['parameter']]
        descriptor = dict(schema='dynamo-cuda-ipc-descriptor/v1', gpu_uuid=owner['gpu_uuid'],
            dtype='bfloat16', shape=shape, stride=src['stride'],
            tensor_offset=0 if piece['axis'] is None else piece['source_offset']*src['stride'][piece['axis']],
            storage_size_bytes=allocation['storage_bytes'],
            storage_offset_bytes=allocation['storage_ptr']-allocation['segment_address'],
            allocation_bytes=allocation['allocation_bytes'], logical_bytes=piece['bytes'],
            source_storage_ptr=allocation['storage_ptr'])
        views.append(dict(piece=piece, descriptor=descriptor))
    value = dict(owner=owner['process'], consumer=dict(pid=201, start_ticks=1, boot_id='cpu-only'),
        source_rank=0, target_rank=0, gpu_uuid='GPU-0', plan_sha256=plan['plan_sha256'],
        source_storage=source, views=views)
    value['packet_sha256'] = digest(value)
    return value


@pytest.mark.parametrize('source_tp,target_tp', [(1,2),(2,1),(4,2),(2,4)])
def test_full_original_owner_segments_remain_counted_for_expansion_and_contraction(source_tp, target_tp):
    plan, owners, boards = fixture(source_tp, target_tp)
    result = review(plan, owners, boards)
    assert result['capacity_arithmetic_passed']
    assert len(result['inventory']['owner_full_allocation_bytes_by_gpu']) == source_tp
    assert result['inventory']['owner_unique_allocations'] == source_tp
    assert not result['target_activation_authorized'] and not result['hardware_verified'] and not result['formal_eligible']
    for board in result['boards']:
        for phase in board['phases'].values():
            assert phase['physical_peak_upper_bound_bytes'] == board['observed_driver_used_bytes']+phase['required_additional_bytes']
    assert result['credited_source_kv_release_bytes'] == result['credited_owner_weight_release_bytes'] == 0


def test_ipc_aliases_are_deduplicated_against_full_original_segment_not_logical_slice():
    plan, owners, _ = fixture()
    first = packet(plan, owners[0]);second = deepcopy(first)
    second['consumer']['pid'] = 202
    second['packet_sha256'] = digest({k:v for k,v in second.items() if k != 'packet_sha256'})
    result = owner_allocation_inventory(plan, owners, [first, second])
    expected = next(iter(owners[0]['parameters'].values()))['allocation_bytes']
    assert result['owner_full_allocation_bytes_by_gpu']['GPU-0'] == expected
    assert result['imported_alias_count'] == 2*len(first['views'])
    assert result['imported_alias_additional_weight_bytes'] == 0


@pytest.mark.parametrize('fault', ['parameter_missing','conflicting_segment_size','imported_allocation_changed','foreign_consumer_device'])
def test_owner_and_alias_inventory_cannot_hide_real_backing(fault):
    plan, owners, _ = fixture();value = packet(plan, owners[0])
    if fault == 'parameter_missing':owners[0]['parameters'].pop(next(iter(owners[0]['parameters'])))
    elif fault == 'conflicting_segment_size':next(iter(owners[0]['parameters'].values()))['allocation_bytes'] += 1
    elif fault == 'imported_allocation_changed':value['views'][0]['descriptor']['allocation_bytes'] += 1
    else:value['target_rank'] = 1
    value['packet_sha256'] = digest({k:v for k,v in value.items() if k != 'packet_sha256'})
    with pytest.raises(ValueError):owner_allocation_inventory(plan, owners, [value])


@pytest.mark.parametrize('api', ['torch.cuda.memory_allocated','torch.cuda.memory_reserved','target_process_peak'])
def test_target_process_allocator_totals_cannot_stand_in_for_physical_board_free(api):
    plan, owners, boards = fixture();boards[0]['driver_api'] = api
    with pytest.raises(ValueError, match='process Torch'):review(plan, owners, boards)


@pytest.mark.parametrize('component', COMPONENTS)
def test_missing_peak_component_stays_unknown_not_zero(component):
    plan, owners, boards = fixture()
    boards[0]['phases']['target_serving'][component]['bytes'] = None
    result = review(plan, owners, boards)
    assert not result['capacity_arithmetic_passed']
    assert result['boards'][0]['phases']['target_serving']['headroom_bytes'] is None
    assert 'GPU-0/target_serving/'+component in result['unknown_components']


def test_zero_requires_explicit_absence_but_target_kv_cannot_be_absent_while_serving():
    plan, owners, boards = fixture();bound=boards[0]['phases']['target_serving']['target_kv_allocator_bytes']
    bound['bytes']=0
    with pytest.raises(ValueError, match='zero is not'):review(plan, owners, boards)
    bound.update(kind='structurally_absent', reason='caller says no KV')
    with pytest.raises(ValueError, match='active native target'):review(plan, owners, boards)


def test_valid_structural_absence_is_explicit_and_does_not_add_phantom_bytes():
    plan, owners, boards = fixture()
    boards[0]['phases']['target_serving']['other_process_peak_growth_bytes'] = dict(
        bytes=0,kind='structurally_absent',reason='all other processes held quiescent throughout phase',evidence=REF)
    assert review(plan, owners, boards)['capacity_arithmetic_passed']


def test_target_fragment_bytes_are_a_floor_not_the_entire_peak_budget():
    plan, owners, boards = fixture()
    boards[1]['phases']['target_serving']['remote_weight_allocator_bytes']['bytes']=1
    with pytest.raises(ValueError, match='missing target fragments'):review(plan, owners, boards)


@pytest.mark.parametrize('fault', ['future','stale','process_used_omits_owner','target_already_created'])
def test_board_anchor_must_be_current_and_include_pinned_source(fault):
    plan, owners, boards = fixture()
    if fault == 'future':boards[0]['observed_at_s']=101
    elif fault == 'stale':boards[0]['observed_at_s']=99
    elif fault == 'process_used_omits_owner':boards[0]['free_bytes']=boards[0]['total_bytes']
    else:boards[0]['anchor_stage']='target_already_running'
    with pytest.raises(ValueError):review(plan, owners, boards)


def test_board_limit_applies_to_each_device_and_each_phase_without_pooling():
    plan, owners, boards = fixture();boards[1]['free_bytes']=100
    result=review(plan, owners, boards)
    assert not result['capacity_arithmetic_passed']
    assert len(result['over_budget_phases'])==len(PHASES)
    assert all(name.startswith('GPU-1/') for name in result['over_budget_phases'])


def kv_receipt(plan, owner):
    """Metadata shaped like the native receipt; explicitly a CPU oracle."""
    identities, sizes, blocks = {}, {}, []
    for name, row in owner['parameters'].items():
        identities[name] = dict(storage_ptr=row['storage_ptr'], device='cpu',
                                shape=plan['source_shapes'][name])
        sizes[name] = row['storage_bytes']
        blocks.append(dict(address=row['storage_ptr'], size=row['storage_bytes'],
            state='active_allocated', segment_address=row['segment_address'],
            segment_bytes=row['allocation_bytes']))
    tail = blocks[-1]['address']+blocks[-1]['size']
    segment_end = blocks[0]['segment_address']+blocks[0]['segment_bytes']
    blocks.append(dict(address=tail, size=segment_end-tail, state='inactive',
        segment_address=blocks[0]['segment_address'], segment_bytes=blocks[0]['segment_bytes']))
    before = dict(gpu_uuid=owner['gpu_uuid'], at_s=100., device_index=0, blocks=blocks)
    after = dict(deepcopy(before), at_s=100.1)
    return dict(schema='dynamo-source-kv-workspace/v1', cpu_oracle=True,
        status='released', source_execution_blocked=True, source_admission_reopened=False,
        requires_process_isolation=False, error=None, plan_sha256=plan['plan_sha256'],
        source_rank=owner['source_rank'], gpu_uuid=owner['gpu_uuid'],
        source_owner_process=owner['process'], original_weight_identity=identities,
        original_weight_storage_bytes=sizes,
        observations=dict(before_release=before, after_release=after),
        source_TP_group_reconfigured=False, target_TP_group_initialized=False,
        target_engine_activated=False, hardware_qualified=False,
        full_tp_switch_qualified=False, formal_eligible=False,
        original_weight_copy_bytes=0, host_weight_staging_bytes=0, driver_free_memory_credit_bytes=0)


def test_kv_adapter_counts_full_original_segment_before_and_after_detach_without_credit():
    plan, owners, _ = fixture();receipt = kv_receipt(plan, owners[0])
    with pytest.raises(ValueError, match='CPU oracle'):
        owner_inventory_from_kv_receipt(plan, receipt, evidence=REF)
    owner = owner_inventory_from_kv_receipt(plan, receipt, evidence=REF, allow_cpu_oracle=True)
    assert owner['parameters'] == owners[0]['parameters']
    assert not owner['transitive_owner_graph_qualified'] and not owner['formal_eligible']
    inventory = owner_allocation_inventory(plan, [owner])
    assert inventory['owner_unique_allocations'] == 1
    assert inventory['source_weight_release_credit_bytes'] == 0


@pytest.mark.parametrize('fault', ['plan','process','uuid','rank','unblocked','error',
    'restored','activated','credit','missing_parameter','shape','logical_slice_only',
    'inactive_weight','pending_weight','ambiguous_backing','incomplete_segment',
    'segment_changed','time_reversed'])
def test_kv_adapter_rejects_unbound_incomplete_or_no_longer_live_original_allocations(fault):
    plan, owners, _ = fixture();receipt = kv_receipt(plan, owners[0])
    name = next(iter(receipt['original_weight_identity']))
    after = receipt['observations']['after_release']
    if fault == 'plan':receipt['plan_sha256'] = 'b'*64
    elif fault == 'process':receipt['source_owner_process']['start_ticks'] = 0
    elif fault == 'uuid':receipt['gpu_uuid'] = 'GPU-other'
    elif fault == 'rank':receipt['source_rank'] = True
    elif fault == 'unblocked':receipt['source_execution_blocked'] = False
    elif fault == 'error':receipt['error'] = 'partial release'
    elif fault == 'restored':receipt['status'] = 'restored'
    elif fault == 'activated':receipt['target_engine_activated'] = True
    elif fault == 'credit':receipt['driver_free_memory_credit_bytes'] = 123
    elif fault == 'missing_parameter':receipt['original_weight_identity'].pop(name)
    elif fault == 'shape':receipt['original_weight_identity'][name]['shape'] = [1]
    elif fault == 'logical_slice_only':receipt['original_weight_storage_bytes'][name] = 2
    elif fault == 'inactive_weight':after['blocks'][0]['state'] = 'inactive'
    elif fault == 'pending_weight':after['blocks'][0]['state'] = 'active_awaiting_free'
    elif fault == 'ambiguous_backing':after['blocks'].append(deepcopy(after['blocks'][0]))
    elif fault == 'incomplete_segment':after['blocks'].pop()
    elif fault == 'segment_changed':
        for row in after['blocks']:row['segment_bytes'] += 1
        after['blocks'][-1]['size'] += 1
    else:after['at_s'] = 99.
    with pytest.raises(ValueError):
        owner_inventory_from_kv_receipt(plan, receipt, evidence=REF, allow_cpu_oracle=True)


def test_actual_source_metadata_requires_cuda_pointer_namespace_without_granting_hardware_proof():
    plan, owners, _ = fixture();receipt = kv_receipt(plan, owners[0]);receipt['cpu_oracle'] = False
    with pytest.raises(ValueError, match='source process CUDA device'):
        owner_inventory_from_kv_receipt(plan, receipt, evidence=REF)
    for row in receipt['original_weight_identity'].values():row['device'] = 'cuda:0'
    owner = owner_inventory_from_kv_receipt(plan, receipt, evidence=REF)
    assert not owner['hardware_verified'] and not owner['formal_eligible']
