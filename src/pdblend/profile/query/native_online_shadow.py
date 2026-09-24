"""Causal, CPU-only development queries from native layout calibration logs.

The collector bypasses Router and Controller. Its acquire/client/release times
are an explicit shadow-observer mapping, never purported actual controller
callbacks. No planner, power model, hardware action or qualification is run.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
import math
import time

from pdblend.planner.forecast import Forecast, Forecaster
from pdblend.online.observations import backlog_snapshot
from pdblend.online.router import RequestRecord
from pdblend.profile.collection.native_timing_plan import binding, digest, read_bound

SCHEMA = 'pdblend-native-online-shadow-ledger/v1'
PRIOR_SCHEMA = 'pdblend-native-online-shadow-prior/v1'
MODEL = 'Qwen2.5-32B-Instruct'
SEMANTIC_FILES = ('pdblend/planner/forecast.py', 'pdblend/online/observations.py',
                  'pdblend/online/router.py', 'pdblend/online/controller.py',
                  'pdblend/bench/run.py', 'pdblend/online/policies.py')
UNKNOWN_CONTROLLER = ['actual_controller_callback_times', 'controller_start_and_replan_times',
                      'shield_state', 'plan_hold_and_down_vote_state', 'current_online_plan',
                      'causal_actual_frequency_read_completion_times']


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _prior(raw, receipt):
    if receipt is None:
        return None
    _need(receipt.get('schema') == PRIOR_SCHEMA and receipt.get('model_id') == MODEL
          and receipt.get('dataset') == raw['point']['dataset']
          and receipt.get('selection_split') == 'calibration_training'
          and receipt.get('evaluation_used') is False and receipt.get('holdout_used') is False
          and _finite(receipt.get('frozen_s')) and receipt['frozen_s'] < raw['service_started_s'],
          'prior must be independently bound training knowledge frozen before this window')
    # Require its original source artifact; merely asserting an early timestamp
    # on an after-the-fact forecast is insufficient provenance.
    _need(isinstance(receipt.get('inputs'), list) and receipt['inputs'], 'prior source inputs missing')
    for ref in receipt['inputs']:
        source = read_bound(ref)
        _need(source.get('evaluation_used') is False and source.get('holdout_used') is False
              and _finite(source.get('frozen_s')) and source['frozen_s'] <= receipt['frozen_s'],
              'prior source does not establish pre-window training provenance')
    value = receipt['forecast']
    _need(set(value) == {f.name for f in fields(Forecast)}, 'prior Forecast fields incomplete')
    _need(isinstance(value['backlog'], (list,tuple)) and not value['backlog'] and value['inflight'] == 0,
          'prior cannot invent initial active requests')
    for key in ('rate_rps', 'input_mean', 'input_p95', 'output_mean', 'peak_rps', 'recent_rate_rps'):
        _need(_finite(value[key]) and value[key] >= 0, 'prior numeric field invalid: ' + key)
    _need(_finite(value['trend_rps']), 'prior trend invalid')
    return Forecast(**value)


def _native_states(raw, at_s, ids, max_age_s):
    observed = {}
    for row in raw.get('state_observations', []):
        received = row.get('received_s')
        if _finite(received) and received < at_s:
            iid = row.get('instance_id')
            if iid in ids and (iid not in observed or received > observed[iid]['received_s']):
                observed[iid] = row
    result = {}
    for iid in ids:
        row = observed.get(iid)
        state = row.get('state', {}) if row else {}
        native_at = state.get('native_at_s')
        missing = []
        if row is None:
            missing.append('no_received_state')
        elif not _finite(native_at) or native_at > row['received_s']:
            missing.append('native_state_time_invalid')
        elif at_s - native_at > max_age_s:
            missing.append('native_state_stale')
        for key in ('running', 'all_queue'):
            if not isinstance(state.get(key), list):
                missing.append('native_' + key + '_missing')
        result[iid] = dict(received_s=row['received_s'] if row else None,
            native_at_s=native_at, age_s=at_s-native_at if _finite(native_at) else None,
            running=state.get('running') if not missing else None,
            all_queue=state.get('all_queue') if not missing else None, missing=missing,
            interpretation='last_received_sample_not_instantaneous_router_backlog')
    return result


def _forecast_view(forecaster, at_s, prior_known):
    forecast = asdict(forecaster.forecast(at_s))
    missing = []
    if not prior_known:
        # Real PD bootstraps a prior. Do not replace it with a cold-start rate,
        # a whole observed trace, requested output lengths or future completions.
        for key in ('rate_rps', 'trend_rps'):
            forecast[key] = None
        missing.append('bootstrap_forecast_prior')
        if not forecaster.arrivals:
            forecast['input_mean'] = forecast['input_p95'] = None
        age = at_s - forecaster.traffic_start if forecaster.traffic_start is not None else 0.
        if age < forecaster.window_s:
            for key in ('output_mean', 'outputs', 'length_pairs'):
                forecast[key] = None
    return forecast, missing


def replay_window(raw, *, decision_offsets_s=None, prior=None, period_s=10., max_state_age_s=1.):
    """Recreate left-limit observer queries; all observations have time < query.

    Uses real Forecaster.arrive/finish/forecast and backlog_snapshot. Client
    receive times stand in for observer token times; acquire stands in for
    dispatch. These are declared mappings, not measured online callback times.
    A missing bootstrap prior yields partial queries, never inferred values.
    """
    _need(raw.get('schema') == 'pdblend-native-request-cycle-window/v1'
          and raw.get('system') == 'pdblend'
          and raw.get('trace', {}).get('schema') == 'pdblend-native-layout-energy-trace/v1'
          and raw['trace'].get('model_id') == MODEL
          and raw['point'].get('purpose') in ('training', 'holdout')
          and raw['trace'].get('evaluation_used_for_selection') is False,
          'only native32 layout calibration/holdout raw is accepted')
    phase = raw['point']['purpose']
    _need(raw['trace'].get('selection_split') == 'calibration_' + phase,
          'raw trace and point split disagree')
    start, end = raw['service_started_s'], raw['service_end_s']
    _need(_finite(start) and _finite(end) and end > start and _finite(period_s) and period_s > 0
          and _finite(max_state_age_s) and max_state_age_s > 0, 'invalid shadow time domain')
    if decision_offsets_s is None:
        decision_offsets_s = [i * period_s for i in range(math.ceil((end-start)/period_s))]
    offsets = list(decision_offsets_s)
    _need(offsets and all(_finite(t) and 0 <= t < end-start for t in offsets)
          and all(b > a for a, b in zip(offsets, offsets[1:])), 'shadow query times must increase within service')
    specs = {r['spec']['instance_id']: r['spec'] for r in raw['actual_launch']}
    _need(len(specs) == 4 and all(s.get('tp') == 2 and s.get('pp') == 1 for s in specs.values()),
          'shadow scope is four native TP2 replicas')
    requests = {r['req_id']: r for r in raw['trace']['requests']}
    clients = {r['request_id']: r for r in raw['client_requests']}
    _need(len(clients) == len(raw['client_requests']), 'duplicate native client IDs')
    timeline = []
    offline_issues = []
    for i, route in enumerate(raw.get('routes', [])):
        _need(route.get('request_id') in clients and route.get('instance_id') in specs
              and route.get('event') in ('acquire', 'release') and _finite(route.get('at_s')),
              'route identity/time missing')
        timeline.append((route['at_s'], 0 if route['event']=='acquire' else 3, i, route['event'], route))
    for client in clients.values():
        rid = client['request_id']
        _need(client['req_id'] in requests and client['instance_id'] in specs, 'client cohort identity differs')
        if 'events' not in client:
            offline_issues.append('missing_token_journal:' + rid)
        if not _finite(client.get('finished_s')):
            offline_issues.append('missing_finish_notification:' + rid)
        for i, event in enumerate(client.get('events', [])):
            if not _finite(event.get('received_s')):
                offline_issues.append('missing_token_receive_time:' + rid)
                continue
            timeline.append((event['received_s'], 1, i, 'token', dict(event, request_id=rid)))
        if _finite(client.get('finished_s')):
            timeline.append((client['finished_s'], 2, 0, 'finish', client))
    timeline.sort(key=lambda e: e[:3])
    known_prior = _prior(raw, prior)
    # Canonical PD enables bootstrap. Its *presence* resets bin_start on the
    # first arrival, independently of its numeric values. This reference prior
    # exercises that real branch; every output that depends on its invented
    # numeric values is censored by _forecast_view. It is never a data imputation.
    forecaster = Forecaster(initial=known_prior if known_prior is not None
                            else Forecast(0.,0.,0.,0.,0.,0))
    router = SimpleNamespace(active={iid: [] for iid in specs})
    active, seen, finished, token_counts, issues = {}, set(), {}, {}, set()
    cursor, queries = 0, []
    for offset in offsets:
        at_s = start + offset
        while cursor < len(timeline) and timeline[cursor][0] < at_s:
            stamp, _, _, kind, event = timeline[cursor]; cursor += 1
            rid = event['request_id']; client = clients[rid]
            iid = client['instance_id']; request = requests[client['req_id']]
            if kind == 'acquire':
                _need(rid not in seen and event['instance_id'] == iid, 'duplicate/mismatched route acquire')
                seen.add(rid); token_counts[rid] = 0
                record = RequestRecord(rid, 'M', iid, iid, len(request['prompt']), request['max_tokens'], stamp,
                    tp=2, pp=1, pool_id=specs[iid].get('pool_id', ''), generation=specs[iid].get('generation', 0))
                active[rid] = record; router.active[iid].append(record)
                forecaster.arrive(record.input_tokens, stamp, request_id=rid)
                # Do not inspect eventual finished_s/terminal/error/journal
                # completeness here. They may describe an unobserved future.
            elif kind == 'token':
                _need(rid in active, 'token precedes ownership or follows release')
                ids = event.get('token_ids')
                if not isinstance(ids, list) or not all(type(t) is int for t in ids):
                    issues.add('missing_exact_token_ids:' + rid)
                    continue
                token_counts[rid] += len(ids)
                if event.get('token_index') != token_counts[rid]:
                    issues.add('incomplete_token_index:' + rid)
                record = active[rid]; record.tokens_so_far = token_counts[rid]
                if ids:
                    record.last_token_s = stamp
                    if record.first_token_s is None:
                        record.first_token_s = stamp
                if event.get('finished') is True:
                    finished[rid] = dict(terminal_at_s=stamp)
            elif kind == 'finish':
                _need(rid in active, 'finish notification lacks live ownership')
                success = rid in finished and not event.get('error')
                tokens_known = not any(issue.endswith(':' + rid) for issue in issues)
                forecaster.finish(token_counts[rid] if success else 0, stamp,
                                  request_id=rid, input_tokens=len(request['prompt']))
                if success and not tokens_known:
                    issues.add('completion_length_unknown:' + rid)
                finished.setdefault(rid, {}).update(notified=True, success=success)
                if not success:
                    issues.add('native_cleanup_ack_history_missing:' + rid)
            else:
                _need(rid in active and event['instance_id'] == iid, 'release lacks owned request')
                if finished.get(rid, {}).get('success') is True:
                    router.active[iid].remove(active.pop(rid))
                else:
                    # Collector releases a counter even on exception. Router
                    # retains uncertain native work until cancellation ACK.
                    issues.add('native_cleanup_ack_history_missing:' + rid)
        forecaster.set_backlog(backlog_snapshot(router))
        forecast, missing = _forecast_view(forecaster, at_s, prior is not None)
        missing.extend(sorted(issues))
        if issues:
            # A count computed from a partial journal cannot become a completed
            # output-length observation. Expose no candidate fields derived
            # from those provisional arithmetic values.
            for key in ('output_mean','outputs','length_pairs'):
                forecast[key] = None
            if any(i.startswith('native_cleanup_ack_history_missing:') for i in issues):
                forecast['inflight'] = None
        backlog = forecast.pop('backlog')
        if issues:
            # Known observed records remain inspectable, but incomplete journal
            # or cleanup history cannot produce precise candidate backlog.
            candidate_backlog = None
        else:
            candidate_backlog = backlog
        states = _native_states(raw, at_s, specs, max_state_age_s)
        for iid, state in states.items():
            missing.extend(iid + ':' + item for item in state['missing'])
        summary = dict(pending_prefill_tokens=sum(w['waiting_prefill_tokens'] for w in backlog),
            remaining_decode_tokens=sum(w['remaining_output_tokens'] for w in backlog),
            occupied_kv_tokens=sum(w['kv_tokens'] for w in backlog)) if candidate_backlog is not None else None
        queries.append(dict(at_s=at_s,offset_s=offset,forecast=forecast,backlog=candidate_backlog,
            observed_owned_records=backlog,backlog_summary=summary,native_states=states,
            native_state_used_as_router_backlog=False,missing=missing,
            candidate_queries=[dict(counts={'M':4},frequency_mhz=f,forecast=forecast,
                backlog=candidate_backlog,backlog_summary=summary,feasibility=None,energy_prediction_j=None)
                for f in (1500,2520)],actual_action=False,energy_label=None,
            requested_collection_frequency_mhz=raw['point']['frequency_mhz'],causal_observed_frequency_mhz=None))
    return dict(schema=SCHEMA,scope='development_causal_shadow_queries_only',phase=phase,
        model_id=MODEL,dataset=raw['point']['dataset'],raw_plan_sha256=raw['plan_sha256'],queries=queries,
        schedule='declared_grid_not_actual_controller_decisions',boundary='all_observations_strictly_before_query',
        event_mapping={'arrive':'route.acquire.at_s','token':'client.events.received_s',
                       'finish':'client.finished_s with past terminal token journal',
                       'ownership_release':'successful route.release.at_s; uncertainty retained'},
        prior_known=prior is not None,actual_controller_missing=UNKNOWN_CONTROLLER,
        bootstrap_branch='canonical_PD_initial_present_unknown_values_censored',
        offline_reconstruction_issues=sorted(set(offline_issues)),
        offline_issues_affect_causal_query_values=False,
        observer_equivalent_to_actual_router=False,actual_online_actions_replayed=False,
        raw_protocol_formal_audited=False,energy_labels_created=False,model_selected=False,
        domain_qualified=False,formal_eligible=False,holdout_used_for_fit=False)


def _signature(query, dataset, frequency):
    backlog = query['backlog']
    # Request IDs and arbitrary engine names are not workload features.
    workload = None if backlog is None else sorted(
        ([w['input_tokens'], w['remaining_output_tokens'], w['waiting_prefill_tokens'], w['kv_tokens'], w['branch']]
         for w in backlog))
    return digest(dict(dataset=dataset,frequency_mhz=frequency,forecast=query['forecast'],backlog=workload))


def _write_new(out, value):
    import json
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return binding(out)


def _source(manifest_ref):
    manifest = read_bound(manifest_ref)
    _need(digest(manifest['files']) == manifest['source_sha256'], 'source manifest aggregate differs')
    root = Path(__file__).parents[3]
    frozen = Path(manifest_ref['path']).parent
    for name in SEMANTIC_FILES:
        _need(binding(root/name)['sha256'] == manifest['files'].get(name)
              == binding(frozen/name)['sha256'], 'replayed observer source differs: ' + name)
    return manifest


def build_ledger(plan_ref, raw_refs, *, source_manifest, out, phase='training', training_ledger=None,
                 prior_refs=None, period_s=10., max_state_age_s=1.):
    """Freeze a training development domain or validate holdout against it.

    Exact training state signatures are diagnostic observations, not a fitted
    or interpolated support region. Holdout never adds signatures or fits a model.
    """
    _need(phase in ('training','holdout'), 'unknown shadow split')
    _need((training_ledger is None) == (phase == 'training'), 'holdout needs frozen training ledger; training cannot consume one')
    plan = read_bound(plan_ref); source = _source(source_manifest)
    _need(plan.get('schema') == 'pdblend-native-layout-energy-plan/v1' and plan.get('model_id') == MODEL
          and plan.get('evaluation_used_for_selection') is False, 'layout calibration plan required')
    _need(raw_refs and len({(r['path'],r['sha256']) for r in raw_refs})==len(raw_refs), 'raw inventory empty or duplicated')
    prior_refs = prior_refs or {}; priors = {d: read_bound(r) for d,r in prior_refs.items()}
    result = dict(schema=SCHEMA,phase=phase,plan=plan_ref,source_manifest=source_manifest,
        implementation=binding(__file__),period_s=period_s,max_state_age_s=max_state_age_s,
        prior_refs=prior_refs,raw_inputs=list(raw_refs),windows=[],formal_eligible=False,
        scope='development_queries_not_actual_controller_or_energy_labels',model_selected=False,
        energy_labels_created=False,domain_qualified=False,holdout_used_for_fit=False)
    signatures = set(); seen = set()
    for reference in raw_refs:
        raw = read_bound(reference); point = raw['point']; key = digest(point)
        _need(point in plan['points'] and point['purpose'] == phase and key not in seen
              and raw['plan_sha256'] == digest(plan), 'raw split, point or immutable plan differs')
        seen.add(key)
        caps = raw.get('capabilities', {})
        _need(len(caps)==4 and all(c.get('source_revision')==source['source_sha256'] for c in caps.values()),
              'raw does not bind the actual replayed source')
        replay = replay_window(raw,prior=priors.get(point['dataset']),period_s=period_s,max_state_age_s=max_state_age_s)
        replay['raw'] = reference
        for query in replay['queries']:
            query['development_state_signatures'] = [_signature(query,point['dataset'],f) for f in (1500,2520)]
            if not query['missing']:
                signatures.update(query['development_state_signatures'])
        result['windows'].append(replay)
    result['complete_planned_phase_inventory'] = seen == {digest(p) for p in plan['points'] if p['purpose']==phase}
    if phase == 'training':
        result.update(training_domain=dict(exact_observed_state_signatures=sorted(signatures),
            interpolation_qualified=False,partial_queries_excluded=True),frozen_s=time.time())
    else:
        trained = read_bound(training_ledger)
        _need(trained.get('schema')==SCHEMA and trained.get('phase')=='training'
              and trained.get('plan')==plan_ref and trained.get('source_manifest')==source_manifest
              and trained.get('implementation')==result['implementation']
              and trained.get('prior_refs')==prior_refs and trained.get('period_s')==period_s
              and trained.get('max_state_age_s')==max_state_age_s,
              'holdout replay does not match frozen training observer/provenance')
        domain = set(trained['training_domain']['exact_observed_state_signatures'])
        for window in result['windows']:
            for query in window['queries']:
                query['training_exact_state_seen'] = [s in domain for s in query['development_state_signatures']]
        result.update(training_ledger=training_ledger,training_domain_sha256=digest(trained['training_domain']),
            training_domain_modified=False,validation_only=True,new_training_signatures_added=0)
    return _write_new(out,result)
