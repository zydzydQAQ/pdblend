"""CPU audit of stationary-transition physical-board memory budgets.

This validates an explicit development contract, not the truth of hardware
receipts. It cannot authorize target activation. Driver-free memory is the
anchor; owner weights and IPC aliases are already charged there. All future
peak increments need evidence, including non-Torch and other-process growth.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .stationary_ipc import digest, need, validate_descriptor


PHASES = ('target_bootstrap', 'fragment_transport', 'target_serving', 'source_rollback')
COMPONENTS = (
    'remote_weight_allocator_bytes', 'target_kv_allocator_bytes',
    'new_cuda_context_bytes', 'new_nccl_and_ipc_mapping_bytes',
    'nonweight_buffers_bytes', 'activation_and_kernel_workspace_bytes',
    'transport_temporary_bytes', 'allocator_slack_bytes',
    'other_process_peak_growth_bytes', 'source_kv_restore_bytes', 'guard_bytes',
)
ANCHOR = 'sources_pinned_after_kv_release_before_any_target_allocation'


def _bytes(value, label):
    need(type(value) is int and value >= 0, label + ': explicit nonnegative integer bytes required')
    return value


def _process(value):
    need(isinstance(value, dict) and type(value.get('pid')) is int and value['pid'] > 0
         and type(value.get('start_ticks')) is int and value['start_ticks'] > 0
         and isinstance(value.get('boot_id'), str) and value['boot_id'],
         'PID/start-time/boot identity required for physical allocations')
    return digest(value)


def _ref(value):
    need(isinstance(value, dict) and isinstance(value.get('path'), str) and value['path']
         and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
         and all(c in '0123456789abcdef' for c in value['sha256']),
         'explicit bound evidence reference required')


def _weight_allocation(observation, identity, storage_bytes, *, gpu_uuid):
    """Join a source-process storage to a complete allocator segment."""
    need(observation['gpu_uuid'] == gpu_uuid, 'source allocator observation UUID differs')
    ptr = _bytes(identity['storage_ptr'], 'original storage address')
    size = _bytes(storage_bytes, 'original full storage')
    need(ptr > 0 and size > 0, 'original weight storage is absent')
    matches = [b for b in observation['blocks'] if b['state'] == 'active_allocated'
               and b['address'] <= ptr and ptr+size <= b['address']+b['size']]
    need(len(matches) == 1, 'source allocator must identify one active original weight backing')
    match = matches[0]
    address, allocation = match['segment_address'], match['segment_bytes']
    _bytes(address, 'segment address'); _bytes(allocation, 'full source segment')
    rows = sorted((b for b in observation['blocks'] if b['segment_address'] == address),
                  key=lambda b: b['address'])
    cursor = address
    for row in rows:
        need(row['segment_bytes'] == allocation and row['address'] == cursor
             and type(row['size']) is int and row['size'] > 0,
             'source allocator segment boundaries are incomplete or inconsistent')
        cursor += row['size']
    need(cursor == address+allocation, 'source allocator must cover the complete original segment')
    return dict(segment_address=address, allocation_bytes=allocation,
                storage_ptr=ptr, storage_bytes=size)


def owner_inventory_from_kv_receipt(plan, receipt, *, evidence, allow_cpu_oracle=False):
    """Adapt one released original-source receipt without inferring physical bytes.

    The reference binds the entire receipt, but this CPU adapter does not read
    or independently replay it. A consumer must verify that binding itself.
    Imported consumer virtual pointers are never joined to a source snapshot.
    """
    _ref(evidence)
    need(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
         and receipt.get('plan_sha256') == plan['plan_sha256'], 'KV receipt tensor plan differs')
    need(receipt.get('schema') == 'dynamo-source-kv-workspace/v1'
         and type(receipt.get('cpu_oracle')) is bool
         and (receipt['cpu_oracle'] is False or allow_cpu_oracle is True),
         'actual source KV receipt required; CPU oracle requires explicit opt-in')
    need(receipt.get('status') == 'released' and receipt.get('source_execution_blocked') is True
         and receipt.get('source_admission_reopened') is False
         and receipt.get('requires_process_isolation') is False and receipt.get('error') is None,
         'released source weights require a blocked, error-free native transaction')
    need(all(receipt.get(k) is False for k in ('source_TP_group_reconfigured',
         'target_TP_group_initialized', 'target_engine_activated', 'hardware_qualified',
         'full_tp_switch_qualified', 'formal_eligible'))
         and all(type(receipt.get(k)) is int and receipt[k] == 0 for k in
         ('original_weight_copy_bytes', 'host_weight_staging_bytes', 'driver_free_memory_credit_bytes')),
         'source-only receipt cannot imply target activation or inferred release credits')
    rank = receipt['source_rank']
    need(type(rank) is int and 0 <= rank < len(plan['source_gpu_uuids'])
         and receipt['gpu_uuid'] == plan['source_gpu_uuids'][rank], 'original source rank/UUID differs')
    _process(receipt['source_owner_process'])
    identities, sizes = receipt['original_weight_identity'], receipt['original_weight_storage_bytes']
    need(set(identities) == set(sizes) == set(plan['source_shapes']),
         'KV receipt must enumerate every original source parameter')
    before, after = (receipt['observations'][key] for key in ('before_release', 'after_release'))
    need(all(type(o['at_s']) in (int, float) and math.isfinite(o['at_s']) for o in (before, after))
         and before['at_s'] <= after['at_s'], 'source release observations are out of order')
    parameters = {}
    for name, identity in identities.items():
        need(identity['shape'] == plan['source_shapes'][name]
             and math.prod(identity['shape'])*2 <= _bytes(sizes[name], 'full original storage bytes'),
             'original weight shape/storage differs from the exact tensor plan')
        if receipt['cpu_oracle'] is False:
            need(all(type(o['device_index']) is int and o['device_index'] >= 0
                     and identity['device'] == 'cuda:'+str(o['device_index']) for o in (before, after)),
                 'weight pointer must belong to the source process CUDA device')
        first = _weight_allocation(before, identity, sizes[name], gpu_uuid=receipt['gpu_uuid'])
        last = _weight_allocation(after, identity, sizes[name], gpu_uuid=receipt['gpu_uuid'])
        need(first == last, 'original source allocation changed while detaching KV')
        parameters[name] = last
    return dict(source_rank=rank, gpu_uuid=receipt['gpu_uuid'], process=receipt['source_owner_process'],
        evidence=evidence, parameters=parameters, scope='original_dense_source_model_only',
        transitive_owner_graph_qualified=False, hardware_verified=False, formal_eligible=False)


def owner_allocation_inventory(plan, owners, ipc_packets=()):
    """Count full owner allocator segments once; imported aliases allocate no weight.

    Every source parameter must be mapped to its actual enclosing cudaMalloc
    segment, including parameters that export no retained view. Descriptors
    refer to that source process address space, never a consumer virtual pointer.
    """
    need(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
         'stationary tensor plan changed')
    need(len(owners) == len(plan['source_gpu_uuids']), 'complete source owner rank inventory required')
    ranks, segments, by_rank = set(), {}, {}
    for owner in owners:
        rank = owner['source_rank']
        need(type(rank) is int and 0 <= rank < len(owners) and rank not in ranks,
             'source owner rank duplicated or missing')
        ranks.add(rank)
        gpu = owner['gpu_uuid']
        need(gpu == plan['source_gpu_uuids'][rank], 'owner physical GPU differs')
        process = _process(owner['process'])
        _ref(owner['evidence'])
        parameters = owner['parameters']
        need(set(parameters) == set(plan['source_shapes']), 'full original owner parameter inventory required')
        by_rank[rank] = owner
        for name, allocation in parameters.items():
            address = _bytes(allocation['segment_address'], 'segment address')
            size = _bytes(allocation['allocation_bytes'], 'allocation')
            storage = _bytes(allocation['storage_ptr'], 'storage address')
            storage_bytes = _bytes(allocation['storage_bytes'], 'storage')
            need(address > 0 and size > 0 and address <= storage
                 and storage + storage_bytes <= address + size
                 and math.prod(plan['source_shapes'][name]) * 2 <= storage_bytes,
                 'source parameter does not fit actual original allocation')
            key = (gpu, process, address)
            need(key not in segments or segments[key] == size, 'same physical allocation has conflicting sizes')
            segments[key] = size
    keys = list(segments)
    for i, key in enumerate(keys):
        for other in keys[i+1:]:
            if key[:2] == other[:2]:
                need(key[2] + segments[key] <= other[2] or other[2] + segments[other] <= key[2],
                     'owner allocator segments overlap; aliases must identify the same segment')
    alias_count, alias_logical_bytes = 0, 0
    for packet in ipc_packets:
        need(packet['packet_sha256'] == digest({k: v for k, v in packet.items() if k != 'packet_sha256'})
             and packet['plan_sha256'] == plan['plan_sha256'], 'IPC packet/plan binding differs')
        owner = by_rank[packet['source_rank']]
        need(packet['owner'] == owner['process'] and packet['gpu_uuid'] == owner['gpu_uuid'],
             'IPC alias refers to another source process or physical GPU')
        _process(packet['consumer'])
        need(packet['consumer'] != packet['owner'] and type(packet['target_rank']) is int
             and 0 <= packet['target_rank'] < len(plan['target_gpu_uuids'])
             and plan['target_gpu_uuids'][packet['target_rank']] == owner['gpu_uuid'],
             'IPC import requires a distinct consumer on the same physical GPU')
        for row in packet['views']:
            piece, descriptor = row['piece'], row['descriptor']
            need(piece in plan['pieces'] and piece['kind'] == 'retain_on_gpu'
                 and piece['source_rank'] == packet['source_rank']
                 and piece['target_rank'] == packet['target_rank'], 'foreign retained fragment')
            allocation = owner['parameters'][piece['parameter']]
            validate_descriptor(descriptor, gpu_uuid=owner['gpu_uuid'], piece=piece,
                                source=packet['source_storage'][piece['parameter']])
            need(descriptor['source_storage_ptr'] == allocation['storage_ptr']
                 and descriptor['storage_size_bytes'] == allocation['storage_bytes']
                 and descriptor['source_storage_ptr'] - descriptor['storage_offset_bytes'] == allocation['segment_address']
                 and descriptor['allocation_bytes'] == allocation['allocation_bytes'],
                 'IPC alias must map the same full owner allocation')
            alias_count += 1
            alias_logical_bytes += descriptor['logical_bytes']
    per_gpu = defaultdict(int)
    for (gpu, _, _), size in segments.items():
        per_gpu[gpu] += size
    return dict(owner_full_allocation_bytes_by_gpu=dict(per_gpu), owner_unique_allocations=len(segments),
        imported_alias_count=alias_count, imported_alias_logical_bytes=alias_logical_bytes,
        imported_alias_additional_weight_bytes=0, source_weight_release_credit_bytes=0,
        scope='original_dense_source_model_only', transitive_owner_graph_qualified=False,
        hardware_verified=False, formal_eligible=False)


def audit_peak_budget(plan, owners, boards, *, fleet_gpu_uuids, now_s, max_age_s, ipc_packets=()):
    """Require an explicit peak-increment bound at each phase on every board.

    Sources are already charged in CURRENT driver used memory. No anticipated
    source KV release, allocation-cache release, or owner exit is credited. A
    zero requires a structural explanation and a bound reference; None/missing
    remains unknown. Evidence files must be independently replayed by a future
    hardware consumer before this arithmetic can support activation.
    """
    inventory = owner_allocation_inventory(plan, owners, ipc_packets)
    need(len(set(fleet_gpu_uuids)) == len(fleet_gpu_uuids) and fleet_gpu_uuids
         and set(plan['source_gpu_uuids'] + plan['target_gpu_uuids']) <= set(fleet_gpu_uuids),
         'complete physical fleet UUID inventory required')
    need(len(boards) == len(fleet_gpu_uuids) and {b['gpu_uuid'] for b in boards} == set(fleet_gpu_uuids),
         'one observation per physical fleet GPU required')
    need(type(now_s) in (int, float) and math.isfinite(now_s)
         and type(max_age_s) in (int, float) and math.isfinite(max_age_s) and max_age_s > 0,
         'explicit finite observation freshness bound required')
    reviewed, unknown, over_budget = [], [], []
    for board in boards:
        gpu = board['gpu_uuid']
        need(board['anchor_stage'] == ANCHOR, 'driver free must precede every target allocation')
        need(board['driver_api'] in ('cudaMemGetInfo', 'nvmlDeviceGetMemoryInfo'),
             'process Torch allocator totals cannot replace board driver free memory')
        _ref(board['evidence'])
        observed = board['observed_at_s']
        need(type(observed) in (int, float) and math.isfinite(observed)
             and 0 <= now_s - observed <= max_age_s, 'board memory observation stale or from the future')
        total, free = _bytes(board['total_bytes'], 'total'), _bytes(board['free_bytes'], 'free')
        need(total > 0 and free <= total, 'driver memory totals inconsistent')
        used = total - free
        owner_bytes = inventory['owner_full_allocation_bytes_by_gpu'].get(gpu, 0)
        need(owner_bytes <= used, 'observed board use cannot omit complete pinned owner allocations')
        phases = board.get('phases', {})
        need(not set(phases) - set(PHASES), 'unrecognized phase cannot hide peak allocations')
        minimum_remote = sum(p['bytes'] for p in plan['pieces']
                             if p['target_gpu_uuid'] == gpu and p['kind'] == 'direct_gpu_transfer')
        checked_phases = {}
        for phase in PHASES:
            values = phases.get(phase, {})
            need(not set(values) - set(COMPONENTS), 'unknown memory component')
            amounts, missing = {}, []
            for component in COMPONENTS:
                bound = values.get(component)
                if bound is None or bound.get('bytes') is None:
                    missing.append(component)
                    continue
                amount = _bytes(bound['bytes'], component)
                _ref(bound['evidence'])
                if amount == 0:
                    need(bound.get('kind') == 'structurally_absent' and isinstance(bound.get('reason'), str)
                         and bound['reason'].strip() and component != 'guard_bytes',
                         'zero is not a substitute for an unknown allocation/workspace bound')
                else:
                    need(bound.get('kind') in ('measured_increment_peak', 'conservative_increment_bound'),
                         'peak increment must be observed or conservatively bounded')
                amounts[component] = amount
            if gpu in plan['target_gpu_uuids'] and phase == 'target_serving':
                for component in ('target_kv_allocator_bytes', 'new_cuda_context_bytes',
                                  'nonweight_buffers_bytes', 'activation_and_kernel_workspace_bytes'):
                    if component in amounts:
                        need(amounts[component] > 0, 'active native target cannot declare '+component+' structurally absent')
            if gpu in plan['source_gpu_uuids'] and phase == 'source_rollback' and 'source_kv_restore_bytes' in amounts:
                need(amounts['source_kv_restore_bytes'] > 0, 'original source KV restoration cannot be structurally absent')
            if phase in ('fragment_transport', 'target_serving') and 'remote_weight_allocator_bytes' in amounts:
                need(amounts['remote_weight_allocator_bytes'] >= minimum_remote,
                     'missing target fragments require actual destination allocations plus allocator overhead')
            label = gpu + '/' + phase
            if missing:
                unknown.extend(label + '/' + item for item in missing)
                checked_phases[phase] = dict(known=False, missing=missing,
                    required_additional_bytes=None, physical_peak_upper_bound_bytes=None, headroom_bytes=None)
            else:
                additional = sum(amounts.values())
                if additional > free:
                    over_budget.append(label)
                checked_phases[phase] = dict(known=True, components=amounts,
                    required_additional_bytes=additional, physical_peak_upper_bound_bytes=used+additional,
                    headroom_bytes=free-additional, fits_observed_board=additional <= free)
        reviewed.append(dict(gpu_uuid=gpu, observed_driver_used_bytes=used,
            observed_driver_free_bytes=free, pinned_full_owner_allocation_bytes=owner_bytes,
            owner_weight_bytes_already_in_driver_used=True, phases=checked_phases))
    return dict(schema='dynamo-stationary-memory-budget-review/v1', scope='CPU_evidence_contract_only',
        inventory=inventory, boards=reviewed, unknown_components=unknown, over_budget_phases=over_budget,
        capacity_arithmetic_passed=not unknown and not over_budget, evidence_replay_required=True,
        credited_source_kv_release_bytes=0, credited_owner_weight_release_bytes=0,
        hardware_verified=False, target_activation_authorized=False, formal_eligible=False)
