"""Independent replay of fresh-node physical work, ownership and eight-GPU energy."""
import json
import math
from pathlib import Path
import fresh_support as f
from capacity_certificate import raw_measurement, close, idle_result, derive_group, build, validate
from calibration_compatibility import validate_compatibility
import capacity_load_calibrate as driver


def saved_identity(rows, instances, start, end, restored=False):
    from ecopadg.serving.completion_policy import engine_residual
    by_id = {r['runtime']['id']: r for r in rows}
    f.need(len(rows) == len(by_id) == len(instances) and set(by_id) == {i['id'] for i in instances},
           'saved original native identities differ')
    for instance in instances:
        row = by_id[instance['id']]
        actual, container, raw = row['container'], instance['container'], row['runtime']
        f.need(actual['Id'] == container['id'] and actual['Image'] == container['image']
               and actual['Name'].lstrip('/') == container['name'] and actual['State']['Running']
               and actual['State']['StartedAt'] == container['StartedAt']
               and actual['State']['Pid'] == instance['host_pid'], 'saved container/source owner differs')
        f.need(all(row['provenance'].get(k) == v for k, v in instance['provenance'].items()),
               'saved imported native source differs')
        stamp = raw['timestamp']
        f.need(start <= stamp <= end and not engine_residual(raw, stamp), 'native snapshot not healthy/empty in stage')
        counts = [raw.get(k) for k in ('transfer_send_started', 'transfer_send_completed', 'transfer_send_failed')]
        f.need(raw.get('transfer_send_counters_observed') is True and raw.get('transfer_inflight_sends_observed') is True
               and raw.get('transfer_send_healthy') is True and raw.get('transfer_inflight_sends') == 0
               and all(type(v) is int and v >= 0 for v in counts) and counts[0] == counts[1] and counts[2] == 0,
               'native sender unhealthy')
        f.need(raw.get('scheduler_budget_pending') is None, 'unsettled native budget')
        caches = [r.get('controls', {}).get('runtime') for r in raw.get('scheduler_io', [])]
        f.need(len(caches) == instance.get('scheduler_cache_count', 1)
               and all(c and c.get('generation') == raw['generation'] and c.get('error') is None for c in caches),
               'native scheduler ACK missing')
        if restored:
            f.need(raw['accepting'] and raw['scheduler_budget_effective']['max_num_batched_tokens']
                   == instance['restore_budget_tokens'], 'retained native service budget not restored')


def audit_rows(result, spec, inventory):
    raw = raw_measurement(result['raw_measurement'])
    trace = f.checked(result['trace'])
    config = f.checked(spec['config'])
    n = trace['n_requests']
    paths = [p for p in result['artifacts'] if Path(p).name == 'requests.json']
    f.need(len(paths) == 1 and all(f.sha(p) == h for p, h in result['artifacts'].items()), 'phase artifact differs')
    rows = f.read(paths[0])
    f.need(n > 0 and len(rows) == len(trace['requests']) == len(trace['prompts']) == n
           == result['n_expected'] == result['n_rows'], 'whole request denominator differs')
    driver.validate_trace(trace, spec['demand_domain_sha256'])
    epoch = result['actual_arrival_epoch_s']
    f.need(close(result['measured_arrival_duration_s'], trace['duration_s'])
           and raw['measurement_start_s'] <= epoch and raw['measurement_end_s'] >= epoch + trace['duration_s'],
           'full arrival window not covered by energy')
    good = 0
    for index, (row, work, prompt) in enumerate(zip(rows, trace['requests'], trace['prompts'])):
        f.need(row['idx'] == index and str(row['request_id']) == str(index), 'request identity/order differs')
        f.need(row['success'] == 1 and row['token_ids_verified'] == 1 and row['request_timeout'] is False
               and row['generated_tokens'] == row['output_len'] == work['output_len']
               and row['input_tokens'] == row['prompt_len'] == work['prompt_len'] == len(prompt),
               'fresh qualification contains incomplete or altered work')
        f.need(math.isclose(row['planned_arrival_s'], epoch + work['arrival_s'], rel_tol=0, abs_tol=1e-6)
               and close(row['request_deadline_s'] - row['planned_arrival_s'], 120), 'arrival/deadline changed')
        f.need(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                   for k in ('ttft_s', 'tpot_s')), 'unobserved latency')
        ok = row['ttft_s'] < config['slo_ttft_s'] and row['tpot_s'] < config['slo_tpot_s']
        f.need(row['slo_ok'] == int(ok), 'strict joint SLO differs')
        good += ok
    f.need(result['complete'] and result['work_complete'] and result['native_idle']
           and not result.get('failed_requests') and not result.get('request_timeouts')
           and result['n_good'] == good and close(result['slo_attainment'], good / n)
           and close(result['energy_j'], raw['energy_j']), 'raw phase metrics differ')
    return dict(n_expected=n, n_good=good, slo_attainment=good / n, energy_j=raw['energy_j'])


def audit_stage(out, spec_ref, mode):
    out = Path(out)
    spec = f.checked(spec_ref)
    cap = f.checked(spec['capacity_binding'])
    validate_compatibility(spec, cap)
    f.need(spec['mode'] == mode and all(f.sha(p) == h for p, h in spec['files'].items()), 'stage source closure differs')
    state = f.read(out / 'status.json')
    owner_ref = f.ref(out.parent / (mode + '.process.json'))
    owner = f.checked(owner_ref)
    f.need(owner['pid'] == state['pid'] and owner['exitcode'] == 0 and owner['startticks']
           and owner['started_s'] <= state['started_s'] <= state['finished_s'] <= owner['finished_s'],
           'original driver owner lifecycle differs')
    expected = 27 if mode == 'layout_calibration' else 1
    f.need(state['complete'] and state['cleanup_complete'] and not state.get('error')
           and not state.get('cleanup_errors') and state['finished_s'] and f.no_live_pid(state['pid'], owner['startticks'])
           and len(state['completed']) == expected and len({r['path'] for r in state['completed']}) == expected,
           'stage work/owner/cleanup is incomplete')
    f.need(f.read(out / 'spec-reference.json') == spec_ref, 'executed declaration differs')
    inventory = f.read(out / 'inventory.json')
    original = f.checked(spec['original_binding'])
    instances = original['instances']
    f.need(inventory['complete'] and not inventory['transition_inflight'] and inventory['pid'] == state['pid']
           and inventory['identity'] == cap['identity'] and inventory['active_instances'] == instances,
           'physical inventory did not return to exact originals')
    f.need(not any(e['kind'] in ('transition_failed', 'rollback_failed') for e in inventory['events']),
           'physical transition failed')
    f.need(all(v.get('state') == 'stopped' for k, v in inventory['known_instances'].items()
               if k not in inventory['initial_ids']), 'owned extra process remains')
    outer = raw_measurement(state['full_operation_measurement'])
    f.need(state['started_s'] <= outer['measurement_start_s'] < outer['measurement_end_s'] <= state['finished_s'],
           'outer raw window outside owning stage')
    metrics = []
    expected_source = dict(original_binding=spec['original_binding'], capacity_binding=spec['capacity_binding'],
                           config=spec['config'], host_manifest=f.ref(Path(spec['host_release']) / 'manifest.json'))
    measurements = [state['full_operation_measurement']]
    for reference in state['completed']:
        result = f.checked(reference)
        f.need(result['source'] == expected_source and Path(reference['path']).is_relative_to(out), 'foreign phase evidence')
        measurements.append(result['raw_measurement'])
        if result.get('phase_kind') == 'idle':
            idle_result(reference, cap['identity'], inventory)
        else:
            metrics.append(dict(result=reference, **audit_rows(result, spec, inventory)))
    actual_config = f.read(out / 'runtime-config.json')
    f.need(actual_config == dict(f.checked(spec['config']), journal=str(out / 'control.jsonl'),
                                capacity_inventory_path=str(out / 'inventory.json')), 'actual control configuration differs')
    for name in ('identity.before.json', 'identity.after.json'):
        saved_identity(f.read(out / name), instances, state['started_s'], state['finished_s'])
    restoration = f.checked(state['retained_restoration'])
    f.need(state['retained_restoration_complete'] and restoration['passed'] and restoration['capacity_cleanup_complete']
           and restoration['original_controller_failure_not_waived'] and restoration['experiment_requests_sent'] == 0,
           'retained native restoration incomplete')
    saved_identity(restoration['before'], instances, state['started_s'], state['finished_s'])
    saved_identity(restoration['after'], instances, state['started_s'], state['finished_s'], restored=True)
    measurements.extend(e['receipt'] for e in inventory['events'] if e['kind'] == 'transition_measurement')
    terminal = f.read(out / 'isolated-observers-terminal.json')
    f.need(terminal['complete'] and not terminal['errors'] and terminal['measurement_adapter'] == spec['measurement_adapter'],
           'isolated sampler did not terminate')
    hooks = f.load(spec['measurement_hooks']['path'], 'newA_dynamic_saved_power_hooks')
    dirs = {Path(r['directory']) for r in terminal['isolated_samplers']}
    f.need(dirs == set((out / 'isolated-samplers').glob('sampler-*')) and dirs,
           'isolated sampler omitted from terminal')
    f.need(hooks.completed_artifacts(dirs, expected_source['host_manifest'], spec['measurement_adapter'])
           == terminal['artifacts'], 'isolated raw/IPC evidence differs')
    seen = set()
    for reference in measurements:
        raw = raw_measurement(reference)
        f.need(raw['measurement_adapter'] == spec['measurement_adapter'] and len(raw['isolated_samplers']) == 1,
               'phase requires its own isolated sampler')
        directory = Path(raw['isolated_samplers'][0]['directory'])
        f.need(directory in dirs and directory not in seen, 'sampler attributed twice or foreign sampler')
        seen.add(directory)
    f.need(seen == dirs, 'isolated observer missing a physical measurement counterpart')
    evidence = None
    if mode != 'layout_calibration':
        control = [json.loads(line) for line in (out / 'control.jsonl').read_text().splitlines()]
        dispatch = [json.loads(line) for line in (out / 'engine-dispatch.jsonl').read_text().splitlines()]
        evidence = driver.autonomous_gate_evidence(inventory, dispatch, control, metrics[0]['n_expected'])
        raw = f.checked(f.checked(state['completed'][0])['raw_measurement'])
        commits = [e for e in inventory['events'] if e['kind'] == 'physical_commit']
        f.need(all(e['execution_verified'] and raw['measurement_start_s'] <= e['started_s']
                   <= e['finished_s'] <= raw['measurement_end_s'] for e in commits),
               'autonomous transition outside measured request window')
        if mode == 'automatic_underload_gate':
            f.need(evidence == f.read(out / 'autonomous-gate-evidence.json') and metrics[0]['n_expected'] == 752,
                   'autonomous752 gate evidence differs')
    return dict(schema='new-A-fresh-capacity-stage-audit-v1', passed=True, independently_recomputed=True,
                mode=mode, spec=spec_ref, status=f.ref(out / 'status.json'), inventory=f.ref(out / 'inventory.json'),
                process_terminal=owner_ref,
                identity=cap['identity'], metrics=metrics, autonomous_evidence=evidence,
                whole_operation_energy_j=outer['energy_j'], isolated_observers=len(dirs),
                source_files=f.tree_files(f.HERE / 'driver'))


def build_certificate(out, spec_ref, destination):
    out, destination = Path(out), Path(destination)
    spec = f.checked(spec_ref)
    capref = spec['capacity_binding']
    identity = f.checked(capref)['identity']
    groups = []
    def group(name, kind, members, **fields):
        reference = f.save(destination / (name + '.json'), dict(schema='capacity-evidence-group-v1',
            identity=identity, capacity_binding=capref, kind=kind, members=members, **fields))
        derive_group(reference, identity)
        return reference
    for layout, key in ((2, 'high2'), (3, 'high3')):
        groups.append(group('layout' + str(layout), 'layout',
                            [f.ref(out / f'cycle-{n}-{key}-layout{layout}/result.json') for n in (1, 2, 3)]))
    savings = []
    for key in ('idle', 'low', 'low40'):
        members = [dict(source=f.ref(out / f'cycle-{n}-{key}-layout3/result.json'),
                        target=f.ref(out / f'cycle-{n}-{key}-layout2/result.json')) for n in (1, 2, 3)]
        savings.append(group('saving-' + key, 'idle_savings' if key == 'idle' else 'savings', members,
                             **(dict(inventory=f.ref(out / 'inventory.json')) if key == 'idle' else {})))
    groups.append(group('savings-grid', 'savings_grid', savings, demand_domain_sha256=spec['demand_domain_sha256']))
    for operation in ('restore_cold', 'remove'):
        paths = [out / (f'cycle-{n}-under_load-layout2to3/result.json' if operation == 'restore_cold'
                        else f'cycle-{n}-remove.json') for n in (1, 2, 3)]
        groups.append(group(operation, 'transition', [dict(result=f.ref(p), inventory=f.ref(out / 'inventory.json'))
                                                      for p in paths], operation=operation, gpus=[5]))
    reference = build(identity, groups, destination / 'certificate.json')
    validate(f.checked(reference), identity)
    return reference
