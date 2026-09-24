"""Raw acceptance of one homogeneous, canonical PDblend observation.

The controller, exact client token receipts and native cleanup are separate
evidence. A plan estimate is never substituted for measured energy. This audit
does not qualify a profile, inherit a failed qualification, or prove optimality.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .comparison_acceptance import _bound, _need, _finite, _equal, _state, _drained
from .comparison_native_acceptance import (native_topology, audit_native_startup,
    audit_native_reset, audit_native_metrics, audit_native_meter)
from .resident_session import digest

ROLES = {'P', 'D', 'M', 'idle', 'L1', 'off'}
ACTIVE = {'P', 'D', 'M'}
RAW_REFS = ('trace', 'outcomes', 'power', 'native_result', 'canonical_requests',
    'metering', 'startup_qualification', 'reset', 'drain', 'controller', 'routes',
    'native_cleanup', 'transition_measurements', 'frequencies')
JOURNALS = {'outcomes', 'controller', 'routes', 'frequencies'}


def _read_raw(ref, name):
    if name != 'frequencies':
        return _bound(ref, journal=name in JOURNALS, power=name == 'power')
    _need(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'frequency evidence binding is missing')
    path = Path(ref['path']); contents = path.read_bytes()
    _need(hashlib.sha256(contents).hexdigest() == ref.get('sha256'), 'frequency evidence checksum differs')
    # The frozen sampler's rows are arrays, not object-shaped compact journals.
    return [json.loads(line) for line in contents.splitlines() if line.strip()]


def _inputs(point, instances):
    """Consume the explicit formal loader; a component receipt cannot pass it."""
    from .independent_dispatch import validate
    from pdblend.profile.query.versions import load_profile
    from pdblend.profile.query.runtime import require_planner_components
    from pdblend.planner.pool import Plan
    inputs = point['inputs']
    checked = validate(point, inputs)
    _need(checked['formal_eligible'], 'own source/profile/workload/mechanism/energy qualifications are incomplete')
    _need(inputs['trace'] == point['trace'], 'dispatcher trace binding differs')
    topology = {(r['tp'], r['pp']) for r in instances.values()}
    _need(len(topology) == 1, 'heterogeneous PD deployment requires a separate pool audit')
    tp, pp = next(iter(topology))
    config = checked['config']; choice = _bound(inputs['offline_choice'])
    selected = config.get('profile')
    if isinstance(selected, dict):
        _bound(selected); selected = selected['path']
    _need(isinstance(selected, str), 'explicit PD profile selection is missing')
    path = (Path(inputs['system_config']['path']).parent / selected).resolve()
    matches = [r for r in inputs['profiles'] if Path(r['path']).resolve() == path]
    _need(len(matches) == 1, 'PD profile is not uniquely hash-bound')
    _bound(matches[0])
    _need(choice.get('profile_sha256') == matches[0]['sha256'], 'offline plan profile differs')
    loaded = load_profile(path, system='pdblend', model_id=point['model_id'], tp=tp, pp=pp, usage='formal')
    require_planner_components(loaded.model, allow_pd=True, allow_dvfs=True)
    plan = Plan(**choice['plan'])
    _need((plan.tp, plan.pp) == (tp, pp) and set(plan.counts) <= ROLES
          and all(type(v) is int and v >= 0 for v in plan.counts.values())
          and sum(plan.counts.values()) == len(instances), 'offline plan inventory differs')
    key = json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':'))
    _need(not plan.profile_key or plan.profile_key == key, 'offline profile identity differs')
    return dict(choice=choice, profile_key=key, frequencies=list(loaded.model.freqs),
                calibration=loaded.manifest_fields(), qualification_bindings=checked['qualification_bindings'])


def _boundary(native, outcomes, metering):
    origin = native.get('service_started_s')
    _need(_finite(origin) and _equal(native.get('service_ended_s'), origin+150)
          and _equal(native.get('window_s'), 150), 'actual 150s service boundary is missing')
    ends = [r.get('finished_s') for r in outcomes]
    _need(ends and all(_finite(t) for t in ends), 'complete request terminal cohort is missing')
    request_end, finished = native.get('request_finished_s'), native.get('finished_s')
    _need(_finite(request_end) and _finite(finished) and max(origin+150, *ends) <= request_end
          and request_end <= finished <= metering['tail_end_s'], 'request/controller/native tail is not fully metered')
    _need(not native.get('quarantined_instances') and native.get('native_cleanup_complete') is True,
          'native cleanup incomplete or routing remains quarantined')
    return origin


def _inventory_reset(reset, instances, identity, startup, origin):
    receipt = reset.get('pdblend_inventory_reset', {})
    before = receipt.get('before', {}); missing = {r.get('instance_id') for r in before.get('off_instances', [])}
    _need(receipt.get('passed') is True and receipt.get('allocated_to_previous_service') is False
          and set(receipt.get('inventory_instance_ids', [])) == set(instances)
          and set(receipt.get('restored_instances', [])) == missing
          and receipt.get('engine_loads') == len(missing), 'pre-window resident inventory restoration differs')
    start, end = receipt.get('started_s'), receipt.get('finished_s')
    _need(_finite(start) and _finite(end) and start <= before.get('tail_end_s', -1) <= end <= origin,
          'inventory restoration lies outside the fresh reset boundary')
    caps = receipt.get('capabilities', {})
    _need(set(caps) == set(instances), 'restored native capability fleet is incomplete')
    for iid, spec in instances.items():
        cap = caps[iid]; initial = startup['capabilities'][iid]
        _need(all(cap.get(k) == initial.get(k) for k in ('model_id','model_hash','tokenizer_hash',
              'image_digest','source_revision','engine_revision','tp','pp','gpu_uuids'))
              and cap.get('supported') is True, 'restored endpoint identity differs')
        state = cap.get('state', {}); _state(state, spec['tp'], state.get('response_at_s'))
        _need(start <= state['native_at_s'] <= end, 'restored endpoint state is stale')


def _native_phase(row, instances, generation):
    iid = row['instance']; operation = row['operation']; spec = instances[iid]
    receipt = row.get('native_receipt', {})
    if operation == 'native_drain':
        _need(receipt.get('acknowledged') is True and receipt.get('drained') is True,
              'parking lacks a native drain ACK')
        state = receipt
    else:
        ack, state = receipt.get('control', {}), receipt.get('state', {})
        _need(ack.get('acknowledged') is True and ack.get('generation') == generation
              and all(state.get(k) is True for k in ('accepting', 'admit_prefill', 'admit_decode')),
              'wake lacks a native admission ACK')
    _need(_state(state, spec['tp'], row['finished_s']) == generation
          and state['native_at_s'] >= row['started_s'], 'native phase epoch or freshness differs')


def _controller(events, native, instances, identity, reset, selected, transitions):
    _need(events and all(_finite(r.get('t')) for r in events)
          and all(a['t'] <= b['t'] for a, b in zip(events, events[1:])), 'controller journal time order is incomplete')
    _need(not any(r.get('kind') in ('transition_failed', 'park_failed') for r in events),
          'controller action failed')
    phases = [r for r in events if r.get('kind') == 'transition_phase']
    bare = [{k:v for k,v in r.items() if k not in ('kind', 't')} for r in phases]
    summary = native.get('controller', {})
    _need(_equal(summary.get('transition_phases'), bare)
          and summary.get('events') == dict(Counter(r['kind'] for r in events)), 'controller summary differs from raw actions')
    measured = transitions.get('phases', [])
    _need(len(measured) == len(bare) and all(all(_equal(v, m.get(k)) for k,v in row.items()
          if k != 'energy_status') for row,m in zip(bare, measured)), 'transition report lost or changed a physical action')
    # The old local transition attribution is diagnostic only. Total energy
    # comes exclusively from the independent uninterrupted eight-board meter.
    _need(transitions.get('incremental_energy_j') is None
          and transitions.get('formal_eligible') is False, 'transition estimates were promoted into measured savings')
    completed = [r for r in events if r.get('kind') == 'transition_complete']
    tids = {r.get('transition_id'):r for r in completed}
    _need(len(tids) == len(completed) and None not in tids, 'transition ownership is missing or duplicated')
    allowed = {'route_publish', 'clock_set', 'clock_reset', 'proxy_drain', 'native_drain',
               'stop', 'start', 'ready', 'park', 'unpark', 'native_resume'}
    for row in phases:
        iid = row.get('instance'); complete = tids.get(row.get('transition_id'), {})
        _need(iid in instances and row.get('operation') in allowed and row.get('status') == 'passed'
              and not row.get('error') and _finite(row.get('started_s')) and _finite(row.get('finished_s'))
              and complete.get('started_s', float('inf')) <= row['started_s'] <= row['finished_s']
              <= complete.get('finished_s', -1) <= native['finished_s'], 'physical action ownership/status/time differs')
        expected = [identity['fleet_gpu_uuids'].index(u) for u in instances[iid]['gpu_uuids']]
        _need(row.get('gpus') == expected and row.get('generation') == reset['generation'][iid],
              'physical action placement or epoch differs')
        if row['operation'].startswith('native_'):
            _native_phase(row, instances, reset['generation'][iid])
    plans = [r for r in events if r.get('kind') == 'plan']
    _need(plans and len(plans) == len(completed) and plans[0]['t'] <= native['service_started_s'],
          'initial plan or complete transition sequence is missing')
    previous = {iid:'M' for iid in instances}
    for plan, complete in zip(plans, completed):
        roles, counts = plan.get('roles', {}), plan.get('counts', {})
        _need(set(roles) == set(instances) and set(roles.values()) <= ROLES and set(counts) <= ROLES
              and all(type(v) is int and v >= 0 for v in counts.values())
              and all(counts.get(r, 0) == list(roles.values()).count(r) for r in ROLES),
              'planned role inventory differs')
        _need(counts.get('M', 0) >= min(4, len(instances)) and counts.get('idle', 0) == 0
              and type(plan.get('tau')) is int and plan['tau'] >= 0,
              'canonical PD mixed reserve or routing threshold differs')
        _need(complete['started_s'] <= plan['t'] <= complete['finished_s'], 'plan was not published within its transition')
        pid = plan.get('plan_identity', {})
        _need((pid.get('tp'), pid.get('pp')) == (next(iter(instances.values()))['tp'], 1)
              and pid.get('generation') == next(iter(reset['generation'].values()))
              and pid.get('profile_key') == selected['profile_key'], 'plan topology/epoch/profile identity differs')
        _need(all(plan.get('f_'+r) in selected['frequencies'] for r in ACTIVE), 'plan frequency is outside its qualified profile')
        for iid, role in roles.items():
            owned = [p for p in phases if p['transition_id'] == complete['transition_id'] and p['instance'] == iid]
            ops = [p['operation'] for p in owned]; before = previous[iid]
            _need(all(a['finished_s'] <= b['started_s'] for a,b in zip(owned,owned[1:])),
                  'one instance has overlapping physical actions')
            if before in ACTIVE and role not in ACTIVE:
                required = ['route_publish', 'proxy_drain', 'native_drain']
                required += ['stop', 'clock_reset'] if role == 'off' else ['clock_reset', 'park'] if role == 'L1' else []
                _need(all(o in ops for o in required) and [ops.index(o) for o in required] == sorted(ops.index(o) for o in required),
                      'parking precedes routing/native KV release')
            if before not in ACTIVE and role in ACTIVE:
                required = (['start', 'ready'] if before == 'off' else ['unpark'] if before == 'L1' else [])
                required += ['clock_set', 'native_resume', 'route_publish']
                _need(all(o in ops for o in required) and [ops.index(o) for o in required] == sorted(ops.index(o) for o in required),
                      'wake publication precedes readiness/native admission')
            if before == 'off' and role == 'L1':
                required = ['start', 'ready', 'native_drain', 'park']
                _need(all(o in ops for o in required) and [ops.index(o) for o in required] == sorted(ops.index(o) for o in required),
                      'off-to-L1 repark precedes native readiness or KV release')
            if role == 'off' and before != 'off':
                _need('stop' in ops, 'off role has no owned physical stop')
            if before != role and before in ACTIVE and role in ACTIVE:
                _need(any(r.get('kind') == 'reroute' and r.get('instance') == iid
                          and r.get('from_role') == before and r.get('to_role') == role
                          and r.get('existing_requests_pinned') is True
                          and complete['started_s'] <= r['t'] <= complete['finished_s'] for r in events),
                      'role change lacks pinned-request routing evidence')
        previous = roles
        _need(set(complete.get('affected', [])) == {p['instance'] for p in phases
              if p['transition_id'] == complete['transition_id']}, 'completed transition affected inventory differs')
    first = plans[0]; choice = selected['choice']['plan']
    _need(all(_equal(first.get(k), choice.get(k)) for k in ('counts', 'f_P', 'f_D', 'f_M', 'tau')),
          'initial deployed plan differs from frozen offline choice')
    stops = [r for r in events if r.get('kind') == 'stop']
    _need(len(stops) == 1 and stops[0]['t'] >= native['request_finished_s']
          and stops[0].get('roles') == previous == summary.get('final_roles') == native.get('final_roles'),
          'controller final roles or stop boundary differs')
    forecasts = [r for r in events if r.get('kind') == 'forecast'
                 and native['service_started_s'] <= r['t'] < native['service_ended_s']]
    _need(len(forecasts) >= 2 and all(r.get('decision_reason') for r in forecasts),
          'canonical periodic controller observations are missing')
    return plans, completed


def _frequencies(samples, plans, completed, instances, identity, origin):
    _need(samples and all(isinstance(r, (list, tuple)) and len(r) == 2 and _finite(r[0])
          and len(r[1]) == 8 for r in samples)
          and all(a[0] < b[0] for a,b in zip(samples, samples[1:])), 'actual physical frequency samples are incomplete')
    for index, plan in enumerate(plans):
        # Plan logging happens just before transition_complete. Settle from
        # the completed physical boundary, not that earlier publication time.
        left = max(origin, completed[index]['finished_s']+1.)
        right = min(origin+150, completed[index+1]['started_s'] if index+1 < len(plans) else origin+150)
        if right <= left: continue
        rows = [r for r in samples if left <= r[0] < right]
        _need(rows and rows[0][0] <= left+1 and rows[-1][0] >= right-1
              and all(b[0]-a[0] <= 1 for a,b in zip(rows,rows[1:])), 'stable plan frequency observations contain a gap')
        for iid, role in plan['roles'].items():
            if role not in ACTIVE | {'L1'}: continue
            indices = [identity['fleet_gpu_uuids'].index(u) for u in instances[iid]['gpu_uuids']]
            requested = 210 if role == 'L1' else plan['f_'+role]
            _need(all(_finite(r[1][g]) and abs(r[1][g]-requested) <= 30 for r in rows for g in indices),
                  'actual active/parked clock differs from executed plan')


def _routes(routes, outcomes, trace, instances, reset, native):
    expected = {f'r{i}':r for i,r in enumerate(trace['requests'])}
    _need(len({r.get('request_id') for r in routes}) == len(routes)
          and {r.get('request_id') for r in routes} <= expected.keys(), 'route cohort has foreign/duplicate requests')
    by_id = {r['request_id']:r for r in routes}
    _need(len(outcomes) == len(expected) and {r.get('idx') for r in outcomes} == set(range(len(expected))),
          'complete unique client outcome cohort required')
    for outcome in outcomes:
        rid = 'r'+str(outcome['idx']); request = expected[rid]; route = by_id.get(rid)
        _need(outcome.get('arrival_s') == request['arrival_s'] and outcome.get('input_tokens') == len(request['prompt'])
              and outcome.get('max_tokens') == request['max_tokens'] and outcome.get('sampling_seed') == 701,
              'actual client workload/seed differs')
        _need(_finite(outcome.get('submitted_s')) and native['service_started_s']+request['arrival_s']
              <= outcome['submitted_s'] <= outcome['finished_s'], 'client submission precedes its offered arrival')
        failed = bool(outcome.get('error'))
        if route is None:
            _need(failed and outcome.get('completion_tokens') == 0 and not outcome.get('token_events'),
                  'submitted non-rejected request has no routing receipt')
            continue
        p,d = route.get('prefill_instance'),route.get('decode_instance')
        _need(p in instances and d in instances and (route.get('tp'),route.get('pp')) == (instances[p]['tp'],1)
              == (instances[d]['tp'],1) and route.get('generation') == reset['generation'][p] == reset['generation'][d],
              'request topology/generation differs from physical peers')
        _need(route.get('input_tokens') == len(request['prompt']) and _finite(route.get('submitted_s'))
              and outcome['submitted_s'] <= route['submitted_s'] <= outcome['finished_s'], 'route identity or request interval differs')
        terminal = route.get('terminal_state'); path = route.get('path')
        _need((path == 'M' and p == d) or (path == 'PD' and p != d), 'unknown or collapsed P/D route')
        if not failed:
            _need(terminal == 'completed' and path == outcome.get('path')
                  and p == outcome.get('prefill') and d == outcome.get('decode'), 'successful stream route differs')
        else:
            _need(terminal in ('completed','rejected_before_engine','cancelled_acknowledged'),
                  'failed request retains uncertain native ownership')
        if terminal == 'cancelled_acknowledged':
            receipts = route.get('route_estimate', {}).get('native_cancel_receipts', {})
            _need(receipts and set(receipts) <= {p,d}, 'native cancellation ownership is missing or foreign')
            request_ids = {r.get('request_id') for r in receipts.values()}
            _need(len(request_ids) == 1 and None not in request_ids, 'native cancel request identity differs')
            engine_id = next(iter(request_ids))
            _need(engine_id == rid if path != 'PD' else engine_id.endswith('_'+rid),
                  'native cancellation ACK belongs to another request')
            _need(p in receipts and (path != 'PD' or outcome['completion_tokens'] <= 1 or d in receipts),
                  'native cancellation omits an observed submitted peer')
            from pdblend.online.native_control import validate_state
            for iid, receipt in receipts.items():
                _need(receipt.get('instance_id') == iid and receipt.get('acknowledged') is True
                      and receipt.get('cancelled') is True and receipt.get('generation') == route['generation'],
                      'native cancellation ACK differs')
                validate_state(receipt.get('native_state'), generation=route['generation'], tp=instances[iid]['tp'],
                               request_id=engine_id, observed_after_s=route['submitted_s'])
                state = receipt['native_state']
                _need(_state(state, instances[iid]['tp'], state.get('response_at_s'), drained=False) == route['generation']
                      and all(engine_id not in r.get('retained', {}).get('held_requests', [])
                              and not r.get('retained', {}).get('receiving_transactions') for r in state['ranks']),
                      'cancelled rank still owns request KV')
        if terminal == 'rejected_before_engine':
            _need(failed and outcome['completion_tokens'] == 0, 'rejected-before-engine receipt has delivered tokens')


def _routing_roles(routes, events, instances):
    """Replay publication ownership; an in-progress publish admits either side.

    A route can already be executing while its owner changes roles. Only the
    dispatch timestamp is checked, so pinned old requests remain legitimate.
    """
    plans = [r for r in events if r.get('kind') == 'plan']
    completed = [r for r in events if r.get('kind') == 'transition_complete']
    target = {c['transition_id']:p['roles'] for p,c in zip(plans, completed)}
    publishes = [r for r in events if r.get('operation') == 'route_publish']
    for route in routes:
        roles = {iid:{'M'} for iid in instances}; stamp = route['submitted_s']
        for row in publishes:
            iid = row['instance']; role = target[row['transition_id']][iid]
            if row['finished_s'] <= stamp: roles[iid] = {role}
            elif row['started_s'] <= stamp: roles[iid].add(role)
        p,d = route['prefill_instance'],route['decode_instance']
        _need(('M' in roles[p] and p == d) if route['path'] == 'M'
              else ('P' in roles[p] and 'D' in roles[d]), 'dispatch used an unpublished or parked role')


def _drain(native, drain, cleanup, instances, identity, reset, metering, phases, source_revision=None):
    roles = native['final_roles']; live = {i:s for i,s in instances.items() if roles[i] != 'off'}
    off = {i:s for i,s in instances.items() if roles[i] == 'off'}
    _need(drain.get('policy_off_preserved') is True and drain.get('final_roles') == roles
          and set(drain.get('inventory_instance_ids', [])) == set(instances)
          and set(drain.get('live_instance_ids', [])) == set(live), 'resident final inventory or off preservation differs')
    generations, times = _drained(drain, live)
    _need(times and min(times) >= native['finished_s'] and max(times) <= drain['tail_end_s']
          == metering['tail_end_s'], 'final native release lies outside measured tail')
    _need(set(cleanup) == set(live), 'runner native cleanup omitted a live rank')
    for iid, receipt in cleanup.items():
        _need(receipt.get('acknowledged') is True and receipt.get('drained') is True
              and _state(receipt, instances[iid]['tp'], receipt.get('response_at_s')) == generations[iid]
              == reset['generation'][iid] and native['request_finished_s'] <= receipt['native_at_s'] <= native['finished_s'],
              'runner native cleanup generation/time differs')
    absent = drain.get('off_instances', [])
    _need(len(absent) == len(off) and {r.get('instance_id') for r in absent} == set(off), 'physical off inventory incomplete')
    for row in absent:
        iid = row['instance_id']; devices = row.get('physical_gpus', []); stop = row.get('owned_stop', {})
        _need(row.get('role') == 'off' and row.get('process_alive') is False and row.get('process_state') == 'off'
              and row.get('compute_processes_gone') is True and stop.get('kind') == 'stop'
              and native['finished_s'] <= row.get('checked_s', -1) <= metering['tail_end_s'], 'off process removal unproven')
        _need([r.get('uuid') for r in devices] == off[iid]['gpu_uuids']
              and all(r.get('local_index') == identity['fleet_gpu_uuids'].index(r['uuid'])
                      and r.get('compute_pids') == [] for r in devices), 'off GPU has wrong UUID or surviving compute processes')
        stops = [p for p in phases if p.get('instance') == iid and p.get('operation') == 'stop']
        _need(stops and stops[-1]['started_s'] <= stop.get('t_s', -1) <= stops[-1]['finished_s'],
              'off process stop differs from controller physical action')
    accounting = drain.get('engine_load_accounting', {}); starts = accounting.get('starts', [])
    _need(accounting.get('engine_loads') == len(starts) and all(r.get('kind') == 'start'
          and r.get('instance') in instances and _finite(r.get('t_s')) and type(r.get('pid')) is int for r in starts),
          'actual engine load accounting is incomplete')
    window_starts = [r for r in starts if reset['pdblend_inventory_reset']['finished_s'] <= r['t_s'] <= metering['tail_end_s']]
    physical_starts = [r for r in phases if r.get('operation') == 'start']
    _need(drain.get('window_engine_loads') == len(window_starts) == len(physical_starts)
          and all(sum(p['instance'] == r['instance'] and p['started_s'] <= r['t_s'] <= p['finished_s']
                      for p in physical_starts) == 1 for r in window_starts), 'window reload count differs from actual process starts')
    epochs = drain.get('restart_epoch_receipts', [])
    for start in window_starts:
        matches = [e for e in epochs if e.get('instance_id') == start['instance'] and e.get('pid') == start['pid']
                   and start['t_s'] <= e.get('started_s', -1)]
        _need(len(matches) == 1, 'restarted process has no unique native epoch restoration')
        row = matches[0]; iid = row['instance_id']; cap = row.get('capability', {})
        _need(cap.get('gpu_uuids') == instances[iid]['gpu_uuids']
              and all(cap.get(k) == identity[k] for k in ('model_hash','tokenizer_hash','image_digest'))
              and source_revision and cap.get('source_revision') == source_revision
              and cap.get('tp') == instances[iid]['tp'] and cap.get('pp') == 1,
              'restarted process has a different model or GPU identity')
        state = row.get('state', {}); before = row.get('before', {}); control = row.get('control', {})
        _need(_state(state, instances[iid]['tp'], row.get('finished_s')) == reset['generation'][iid]
              and _state(before, instances[iid]['tp'], before.get('response_at_s')) <= reset['generation'][iid]
              and control.get('acknowledged') is True and control.get('generation') == reset['generation'][iid]
              and state.get('accepting') is False and state.get('admit_prefill') is False
              and state.get('admit_decode') is False and state.get('role') == 'mixed' and state.get('mode') == 'temporal'
              and row['started_s'] <= state['native_at_s'] <= row['finished_s'] <= metering['tail_end_s'],
              'restarted native epoch/admission restoration differs')


def audit_pdblend_window(point, engine_identity, startup_qualification, reset, native_result,
                         canonical_metrics, metering, drain, raw_refs):
    failures, checked, data = {}, [], {}
    def gate(name, fn):
        try: value = fn()
        except (ValueError, TypeError, KeyError, OSError, IndexError, AttributeError, OverflowError, RuntimeError) as exc:
            failures[name] = str(exc); return None
        checked.append(name); return value
    for name in RAW_REFS:
        value = gate('raw.'+name, lambda name=name:_read_raw(raw_refs.get(name), name))
        if value is not None: data[name] = value
    for name, value in (('native_result',native_result), ('startup_qualification',startup_qualification),
                         ('reset',reset), ('metering',metering), ('drain',drain)):
        gate('binding.'+name, lambda name=name,value=value:_need(_equal(data[name],value), 'supplied receipt differs from bound raw file'))
    gate('binding.trace', lambda:_need(raw_refs.get('trace') == point.get('trace'), 'point trace binding differs'))
    instances = gate('pdblend.inventory', lambda:native_topology(point, engine_identity))
    if instances is not None:
        gate('pdblend.full_physical_inventory', lambda:_need([u for r in instances.values() for u in r['gpu_uuids']]
              == engine_identity['fleet_gpu_uuids'], 'PD planning must account for every metered GPU'))
    selected = gate('pdblend.formal_inputs', lambda:_inputs(point, instances))
    origin = gate('pdblend.actual_window', lambda:_boundary(native_result, data['outcomes'], metering))
    gate('pdblend.startup', lambda:audit_native_startup(point, engine_identity, startup_qualification, instances, raw_refs))
    gate('pdblend.reset', lambda:audit_native_reset(reset, instances, origin))
    gate('pdblend.inventory_restoration', lambda:_inventory_reset(reset, instances, engine_identity, startup_qualification, origin))
    control = gate('pdblend.controller_actions', lambda:_controller(data['controller'], native_result, instances,
              engine_identity, reset, selected, data['transition_measurements']))
    gate('pdblend.physical_clocks', lambda:_frequencies(data['frequencies'], *control, instances, engine_identity, origin))
    gate('pdblend.request_routes', lambda:_routes(data['routes'], data['outcomes'], data['trace'], instances, reset, native_result))
    gate('pdblend.published_route_roles', lambda:_routing_roles(data['routes'], data['controller'], instances))
    gate('pdblend.native_release_and_off', lambda:_drain(native_result, drain, data['native_cleanup'], instances,
         engine_identity, reset, metering, [r for r in data['controller'] if r.get('kind') == 'transition_phase'],
         _bound(startup_qualification['source_manifest'])['source_sha256']))
    reduced = gate('pdblend.canonical_metrics', lambda:audit_native_metrics(point, data['trace'], data['outcomes'], None,
                   origin, data['canonical_requests'], canonical_metrics))
    gate('metering.raw_eight_gpu_window', lambda:audit_native_meter(engine_identity, data['power'], metering, origin))
    valid = not failures
    return dict(schema='pdblend-single-observation-acceptance-v1', scope='canonical_policy+single_observation',
        evidence_valid=valid, formal_eligible=valid, slo_pass=reduced['slo_pass'] if reduced else False,
        missing_gates=list(failures), gate_failures=failures, checked_gates=checked,
        optimality_established=False, per_request_kv_transaction_audited=False,
        inherited_artifact_flags_unchanged=True, evidence_sha256=digest(raw_refs))
