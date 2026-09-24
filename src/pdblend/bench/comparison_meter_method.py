"""System-independent audit of the explicitly selected isolated sampler.

The frozen EcoServe auditor keeps its original copy for evidence continuity.
This module imports no system policy or controller.
"""
from .comparison_acceptance import _bound, _need, _finite, _equal


def audit_isolated_meter_method(point, identity, startup, native, metering, raw_refs):
    """Validate only an explicitly selected future process-isolated method."""
    _need(point.get('metering_execution') == identity.get('metering_execution') == 'isolated_process',
          'metering execution mode differs from frozen engine identity')
    receipt = _bound(raw_refs.get('metering_method'))
    _need(receipt.get('schema') == 'isolated-comparison-meter/v1'
          and receipt.get('process_start_method') == 'spawn' and receipt.get('test_factory') is False
          and type(receipt.get('parent_pid')) is int and receipt['parent_pid'] > 0
          and type(receipt.get('child_pid')) is int and receipt['child_pid'] > 0
          and receipt['parent_pid'] != receipt['child_pid'], 'independent production meter process is unproven')
    _need(receipt.get('gpu_ids') == list(range(8)) and receipt.get('gpu_uuids') == identity['fleet_gpu_uuids']
          and receipt.get('polling_interval_s') == .1 and receipt.get('maximum_interpolation_gap_s') == 1.
          and receipt.get('sample_clocks') is True and receipt.get('additional_frequency_observation') is True
          and receipt.get('public_snapshot_fields_modified') is False
          and receipt.get('additional_snapshot_fields') == ['frequency_samples']
          and receipt.get('rpc_scope') == 'outside_service_and_drain_tail', 'isolated sampler policy or snapshot semantics differ')
    source = _bound(startup['source_manifest'])
    implementations = {
        'wrapper': ('pdblend.bench.isolated_comparison_meter', 'IsolatedComparisonMeter'),
        'public_sampler': ('pdblend.bench.comparison_metering', 'ComparisonMeteringSession'),
        'power_sampler': ('pdblend.measure.power', 'PowerSampler'),
        'backend': ('pdblend.measure.backends', 'PynvmlBackend'),
        'factory': ('pdblend.bench.comparison_metering', 'ComparisonMeteringSession'),
        'sampler': ('pdblend.bench.comparison_metering', 'ComparisonMeteringSession')}
    for key, (module, name) in implementations.items():
        observed = receipt.get(key, {}); relative = module.replace('.', '/')+'.py'
        _need(observed.get('module') == module and observed.get('name') == name
              and observed.get('sha256') == source['files'].get(relative) and observed.get('sha256')
              and isinstance(observed.get('path'), str) and observed['path'].endswith('/'+relative),
              'isolated process implementation differs from bound source: '+key)
    origin, tail = native['service_started_s'], metering['tail_end_s']
    _need(all(_finite(receipt.get(k)) for k in ('start_requested_s','started_s','receipt_observed_s'))
          and receipt['start_requested_s'] <= receipt['started_s'] <= origin <= tail <= receipt['receipt_observed_s']
          and receipt.get('window_guard_active') is False and not receipt.get('error'),
          'isolated process observation lifetime differs')
    status = receipt.get('status')
    _need((status == 'running' and receipt.get('child_alive') is True)
          or (status == 'stopped' and receipt.get('child_alive') is False and receipt.get('child_exitcode') == 0
              and _finite(receipt.get('finished_s')) and tail <= receipt['finished_s'] <= receipt['receipt_observed_s']),
          'isolated child failed or exited without a clean boundary')
    guards = receipt.get('local_window_guards', [])
    _need(guards and all(_finite(g.get('begin_s')) and _finite(g.get('end_s'))
          and receipt['started_s'] <= g['begin_s'] <= g['end_s'] <= receipt['receipt_observed_s'] for g in guards)
          and all(a['end_s'] <= b['begin_s'] for a,b in zip(guards,guards[1:]))
          and sum(g['begin_s'] <= origin and tail <= g['end_s'] for g in guards) == 1,
          'local guard does not cover the entire service and drain tail')
    commands = receipt.get('commands', [])
    _need(commands and [r.get('sequence') for r in commands] == list(range(1,len(commands)+1)),
          'isolated meter RPC sequence is missing or duplicated')
    for command in commands:
        start, end = command.get('requested_s'), command.get('finished_s')
        _need(command.get('operation') in ('snapshot','stop') and command.get('passed') is True
              and not command.get('error') and _finite(start) and _finite(end)
              and receipt['started_s'] <= start <= end <= receipt['receipt_observed_s']
              and all(end <= g['begin_s'] or start >= g['end_s'] for g in guards),
              'meter RPC failed or overlaps a guarded service/drain interval')
    _need(all(a['finished_s'] <= b['requested_s'] for a,b in zip(commands,commands[1:]))
          and any(r['requested_s'] >= tail for r in commands), 'meter snapshot does not follow the complete tail')
    observations = receipt.get('liveness_observations', [])
    _need(len(observations) == len(commands)+1, 'child sampler liveness receipts incomplete')
    for obs, command in zip(observations, [dict(operation='start',sequence=0,
            requested_s=receipt['start_requested_s'],finished_s=receipt['started_s'])]+commands):
        _need(obs.get('child_pid') == receipt['child_pid'] and obs.get('operation') == command['operation']
              and obs.get('sequence') == command['sequence'] and _finite(obs.get('observed_s'))
              and command['requested_s'] <= obs['observed_s'] <= command['finished_s']
              and not obs.get('sampler_error')
              and obs.get('sampler_thread_alive') is (command['operation'] != 'stop'),
              'child sampler error, liveness or RPC receipt differs')
    startup_ref = startup.get('metering_method_startup')
    startup_method = _bound(startup_ref)
    preflight = _bound(startup.get('metering_startup_preflight'))
    _need(preflight.get('method') == startup_ref, 'startup hardware preflight uses another method receipt')
    for key in ('schema','process_start_method','parent_pid','child_pid','test_factory','gpu_ids','gpu_uuids',
                'polling_interval_s','maximum_interpolation_gap_s','start_requested_s','started_s',
                'sample_clocks','additional_frequency_observation','public_snapshot_fields_modified',
                'additional_snapshot_fields',*implementations):
        _need(_equal(startup_method.get(key), receipt.get(key)), 'startup/window meter identity differs: '+key)
    _need(_finite(startup_method.get('receipt_observed_s'))
          and receipt['started_s'] <= startup_method['receipt_observed_s'] <= guards[0]['begin_s']
          and not startup_method.get('local_window_guards')
          and startup_method.get('commands') == commands[:len(startup_method.get('commands', []))]
          and startup_method.get('liveness_observations') == observations[:len(startup_method.get('liveness_observations', []))],
          'same-child startup probe is not a prefix before all measured windows')
    from .comparison_meter_preflight import qualify_startup_snapshot
    raw_startup = _bound(preflight.get('raw_power'), power=True)
    replayed = qualify_startup_snapshot(raw_startup, startup_method, identity['fleet_gpu_uuids'])
    _need(all(_equal(value,preflight.get(key)) for key,value in replayed.items()),
          'startup hardware preflight differs from its bound original observations')
    return receipt
