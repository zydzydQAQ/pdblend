"""Auditable endpoint handoff observations, distinct from device copy time.

Existing runtime receipts contain real P-first/D-submit/D-first timestamps.
Their differences are nonnegative client intervals including HTTP, scheduling,
KV handling and the first D step. No subtraction of an ordinary request is
used as a physical target. The optional request helper uses already owned
clients; this module never launches a fleet or changes routing thresholds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import statistics

from .native_timing_plan import binding, read_bound
from .native_timing_replay import Resolver
from .native_runtime_audit import bound, replay_runtime
from .native_runtime_topology import validate_frequencies
from .native_runtime_collect import write_new
from pdblend.profile.query.native_composition import _source_files, _translated, IDENTITY


METRICS = ('first_to_second_output_s', 'client_dispatch_s', 'decode_submit_to_first_s')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def endpoint_intervals(result):
    """Validate exact carried token evidence; reject reversed clocks, never clip."""
    pre, combined = result.get('prefill'), result.get('combined')
    need(isinstance(pre, dict) and isinstance(combined, dict), 'complete P/D payload required')
    n = combined.get('completion_tokens')
    need(type(n) is int and n >= 2 and combined.get('pd_protocol') == 'carry_first_token',
         'at least two carry-first-token outputs required')
    need(pre.get('error') is None and combined.get('error') is None
         and pre.get('usage_received') is True and combined.get('usage_received') is True
         and combined.get('stream_done') is True and pre.get('stream_done') is False,
         'failed or partial request cannot supply a handoff target')
    need(pre.get('completion_tokens') == 1 and combined.get('decode_completion_tokens') == n - 1
         and pre.get('instance_id') != combined.get('instance_id')
         and isinstance(pre.get('request_id'), str) and pre['request_id']
         and pre.get('request_id') == combined.get('request_id'), 'P/D identity or token conservation differs')
    times, ids = combined.get('token_times_s'), combined.get('token_ids')
    need(isinstance(times, list) and len(times) == n
         and isinstance(ids, list) and len(ids) == n
         and all(type(token) is int and token >= 0 for token in ids), 'exact logical token events required')
    submit, first, dispatch, second, finish = (combined.get(name) for name in
        ('submitted_s', 'first_token_s', 'decode_submitted_s', 'decode_first_token_s', 'finished_s'))
    stamps = [submit, first, dispatch, second, *times[2:], finish]
    need(all(finite(value) for value in stamps) and all(a <= b for a, b in zip(stamps, stamps[1:])),
         'handoff timestamps must be finite and causally ordered; negative gaps are not clipped')
    need(pre.get('submitted_s') == submit and pre.get('first_token_s') == first
         and pre.get('finished_s') == first and pre.get('token_times_s') == [first]
         and pre.get('token_ids') == ids[:1] and times[:2] == [first, second],
         'carried first token or D-first timestamp differs')
    return dict(first_to_second_output_s=second-first, client_dispatch_s=dispatch-first,
        decode_submit_to_first_s=second-dispatch, output_tokens=n,
        timestamps=dict(submitted_s=submit, prefill_first_s=first, decode_submitted_s=dispatch,
                        decode_first_s=second, finished_s=finish),
        physical_copy_time=False, includes_http_scheduling_kv_and_first_decode=True,
        clock_domain='same_client_wall_clock_no_cross_device_cuda_subtraction')


def proxy_intervals(record):
    """Read raw server timestamps rather than its max(0, delta) convenience field."""
    need(record.get('path') == 'PD' and not record.get('error'), 'successful PD proxy record required')
    first, second = record.get('pd_handoff_started_s'), record.get('first_decode_token_s')
    need(finite(first) and finite(second) and first <= second,
         'proxy handoff timestamps absent or reversed')
    need(record.get('first_token_s') == first, 'proxy P token boundary differs')
    return dict(first_to_second_output_s=second-first, physical_copy_time=False,
                observation_boundary='proxy_after_P_write_to_first_D_arrival',
                usable_for_independent_training=False,
                reason='requires separately bound calibration split, raw token usage, clocks and native epochs')


async def collect_endpoint_pair(transfer, prefill_client, decode_client, prompt, *,
                                output_tokens, tag, purpose, seed, persist):
    """Minimal collector on already owned clients; persist before validation.

    This does not grant fleet/clock/cleanup qualification. The resident caller
    must bind those receipts and a frozen candidate before invoking holdout.
    ``persist`` must synchronously write a new immutable raw receipt and return
    its binding. A failure is retained and raised, never dropped or retried here.
    """
    from pdblend.engine.client import pd_complete
    need(purpose in ('training', 'holdout') and type(seed) is int,
         'explicit independent training/holdout request required')
    need(type(output_tokens) is int and output_tokens >= 2
         and prompt and all(type(n) is int and n >= 0 for n in prompt), 'explicit token budgets required')
    ordinary = []
    try:
        for i in range(2):
            ordinary.append(await decode_client.complete(prompt, output_tokens, tag+'-M'+str(i),
                            token_diagnostics=True, seed=seed))
        pre, combined = await pd_complete(transfer, prefill_client, decode_client, prompt,
            output_tokens, tag+'-PD', token_diagnostics=True, seed=seed)
    except BaseException as exc:
        persist(dict(schema='pdblend-endpoint-handoff-request/v1', purpose=purpose, seed=seed,
            prompt=list(prompt), output_tokens=output_tokens, error=repr(exc),
            result=dict(ordinary=[asdict(row) for row in ordinary], prefill=None, combined=None),
            physical_copy_time=False, component_qualified=False))
        raise
    result = dict(ordinary=[asdict(row) for row in ordinary], prefill=asdict(pre),
                  combined=asdict(combined) if combined is not None else None)
    reference = persist(dict(schema='pdblend-endpoint-handoff-request/v1', purpose=purpose,
        seed=seed, prompt=list(prompt), output_tokens=output_tokens, result=result,
        physical_copy_time=False, component_qualified=False))
    # A successful result must name bytes already saved by the caller.
    raw = read_bound(reference)
    need(raw['result'] == result and raw['purpose'] == purpose and raw['seed'] == seed
         and raw['prompt'] == list(prompt) and raw['output_tokens'] == output_tokens,
         'persist callback did not bind the actual raw request')
    measured = endpoint_intervals(result)
    need(measured['output_tokens'] == output_tokens
         and combined.prompt_tokens == pre.prompt_tokens == len(prompt), 'actual request shape differs')
    for row in ordinary:
        need(not row.error and row.stream_done and row.usage_received
             and row.prompt_tokens == len(prompt) and row.completion_tokens == output_tokens
             and row.token_ids == combined.token_ids, 'ordinary/PD logical token golden differs')
    return dict(raw=reference, observations=measured, fleet_clock_cleanup_qualified=False)


def training_candidate(windows, *, identity, source):
    """Fit only supplied training windows; no holdout can influence predictions."""
    need(windows and all(row.get('purpose') == 'training' for row in windows),
         'candidate fitting accepts training only, never holdout/evaluation')
    groups = defaultdict(list)
    for row in windows:
        groups[row['input_tokens']].append(row)
    nodes = []
    for length, rows in sorted(groups.items()):
        need(len(rows) == 3 and {r['repeat'] for r in rows} == {0, 1, 2},
             'each handoff node needs the three original training repeats')
        samples = [s for row in rows for s in row['observations']]
        need(samples and all(s['output_tokens'] == 16 for s in samples), 'legacy domain is 16 output tokens only')
        nodes.append(dict(input_tokens=length, output_tokens=16,
            requested_f_P_mhz=2520, requested_f_D_mhz=2520,
            actual_clock_domain_qualified=all(row.get('physical_clocks_qualified') is True for row in rows),
            predictions={metric: statistics.median(statistics.median(s[metric] for s in row['observations'])
                         for row in rows) for metric in METRICS},
            training_requests=len(samples),
            observed_training_max_first_gap_s=max(s['first_to_second_output_s'] for s in samples),
            p99_risk_qualified=False, short_output_generalization_qualified=False))
    return dict(schema='pdblend-endpoint-handoff-training-candidate/v1', identity=identity,
        source_manifest=source, estimator='median_of_three_training_window_medians', nodes=nodes,
        selection_split='calibration', evaluation_used_for_selection=False,
        physical_copy_time=False, runtime_router_threshold_changed=False,
        qualification_scope='historical_calibration_diagnostic_not_prospective_frozen_holdout',
        component_qualified=False, full_profile_qualified=False)


def extract_runtime(completion_ref, source_manifest_ref):
    """Independently replay old runtime raw evidence before extracting durations."""
    source = _source_files(source_manifest_ref, Resolver())
    report = _translated(read_bound(completion_ref), Resolver())
    audit = replay_runtime(report)
    need(audit['raw_components_complete'], 'runtime raw evidence does not replay: ' + repr(audit['errors']))
    need(report.get('runtime_plan', {}).get('selection_split') == 'calibration_and_independent_holdout'
         and report['runtime_plan'].get('evaluation_used_for_selection') is False,
         'bound calibration/independent holdout declaration required')
    caps = list(report['initial_capabilities'].values())
    need(caps and all(cap['source_revision'] == source['source_sha256'] for cap in caps),
         'source manifest differs from actual native runtime')
    identity = {key: 'pdblend' if key == 'system' else caps[0][key] for key in IDENTITY}
    rows = bound(report['journal'], lines=True)
    requests = {row['tag']: row for row in rows if row['kind'] == 'transfer_request'}
    power = bound(report['power'])
    windows = []
    for row in rows:
        if row['kind'] != 'transfer':
            continue
        # An idle/static frequency label is not proof of both active endpoint clocks.
        clock_error = None
        try:
            validate_frequencies(power, row['gpus'], row['started_s'], row['finished_s'], 2520)
        except ValueError as exc:
            clock_error = str(exc)
        observations = []
        for index in range(len(row['timings'])):
            request = requests[f"transfer-{row['input_tokens']}-{row['repeat']}-measure-{index}"]
            need(request['purpose'] == row['purpose'], 'request/window split differs')
            value = endpoint_intervals(request['result'])
            need(row['started_s'] <= value['timestamps']['submitted_s']
                 <= value['timestamps']['finished_s'] <= row['finished_s'],
                 'request escapes the original measurement window')
            observations.append(dict(request_tag=request['tag'], **value))
        windows.append(dict(input_tokens=row['input_tokens'], repeat=row['repeat'],
            purpose=row['purpose'], prefill_instance=row['prefill_instance'],
            decode_instance=row['decode_instance'], physical_gpus=row['gpus'], observations=observations,
            requested_frequency_mhz=2520, physical_clocks_qualified=clock_error is None,
            physical_clock_error=clock_error))
    need(len(windows) == 12, 'legacy transfer requires three shapes and four original repeats')
    training = [row for row in windows if row['purpose'] == 'training']
    candidate = training_candidate(training, identity=identity, source=source_manifest_ref)
    comparisons = []
    for row in windows:
        if row['purpose'] != 'holdout':
            continue
        node = next(node for node in candidate['nodes'] if node['input_tokens'] == row['input_tokens'])
        for metric in METRICS:
            observed = statistics.median(s[metric] for s in row['observations'])
            prediction = node['predictions'][metric]
            comparisons.append(dict(input_tokens=row['input_tokens'], metric=metric,
                training_prediction=prediction, heldout=observed,
                relative_error=abs(prediction-observed)/observed if observed > 0 else None))
    return dict(schema='pdblend-endpoint-handoff-extraction/v1', completion=completion_ref,
        source_manifest=source_manifest_ref, journal=report['journal'], power=report['power'],
        native_runtime_raw_replayed=True, native_runtime_original_holdout_passed=audit.get('holdout_passed', False),
        identity=identity, physical_pair_uuid_inventory=report['lease'], candidate=candidate,
        physical_frequency_coverage_qualified=all(row['physical_clocks_qualified'] for row in windows),
        holdout_comparisons=comparisons, windows=windows,
        evaluation_used_for_selection=False, hardware_executed=False,
        physical_copy_time=False, full_profile_qualified=False, component_qualified=False,
        missing_for_short_output_tuning=['independent_exact_output_2_4_8_measurements',
            'cross_frequency_pair_coverage', 'prospective_training_candidate_freeze_before_holdout'])


def collection_plan(ledger_ref, model_id, *, output_tokens=(2, 4, 8, 16)):
    """Declare fresh exact query nodes from non-evaluation planner provenance."""
    ledger = read_bound(ledger_ref)
    need(ledger.get('evaluation_read') is False, 'cannot select handoff nodes using evaluation')
    rows = [row for row in ledger['ledgers'] if row['model_id'] == model_id]
    need(rows and all(row.get('selection_split') in ('calibration', 'tuning') for row in rows),
         'model-owned calibration/tuning query ledger required')
    need(output_tokens and len(set(output_tokens)) == len(output_tokens)
         and all(type(n) is int and n >= 2 for n in output_tokens), 'distinct output budgets >=2 required')
    lengths = sorted({query['args'][0] for row in rows for query in row['queries']
        if query['method'] == 'transfer_seconds' and query.get('finite_arguments') is True})
    need(lengths and all(type(n) is int and n > 0 for n in lengths), 'no exact PD transfer query nodes for model')
    frequencies = sorted({f for row in rows for f in row['frequency_scope']})
    need(frequencies and all(type(f) is int and f > 0 for f in frequencies), 'explicit clock domain required')
    points = [dict(input_tokens=length, output_tokens=n, f_P_mhz=p, f_D_mhz=d)
              for length in lengths for n in output_tokens for p in frequencies for d in frequencies]
    return dict(schema='pdblend-independent-endpoint-handoff-plan/v1', model_id=model_id,
        query_ledger=ledger_ref, selection_split='calibration', evaluation_used_for_selection=False,
        points=points, training_repeats=3, holdout_repeats=3,
        training_seed=9911, holdout_seed=9912, settle_s=2., measure_min_s=5.,
        min_requests_per_window=34, min_requests_per_node_per_split=102,
        actual_clock_tolerance_mhz=15, maximum_clock_sample_gap_s=1.,
        exact_pair_only=True, outstanding_pd_requests=1, other_instances_loaded_idle=True,
        estimand='same_client_P_first_to_D_first_including_D_first_step', physical_copy_time=False,
        prospective_order=['training_only', 'freeze_bound_training_candidate', 'independent_holdout_only'],
        golden_check='two_ordinary_complete_requests_match_carried_logical_token_ids_per_prompt',
        observer_entrypoint='pdblend.profile.collection.native_handoff.endpoint_intervals',
        request_entrypoint='pdblend.engine.client.pd_complete',
        source_and_lease_entrypoint='pdblend.profile.collection.native_runtime_collect.validate_inventory',
        pair_collector_entrypoint='pdblend.profile.collection.native_handoff.collect_endpoint_pair',
        collector_ready=False, full_profile_qualified=False,
        concrete_collector_work=['extend resident runner transfers with explicit output budget/clock pairs/fresh seeds',
            'persist bound candidate after all training nodes and before starting independent holdout',
            'extend each window until minimum request count; retain all failures and native drains',
            'bind both endpoint clocks and physical UUIDs; restore fleet and replay terminal cleanup'],
        pure_device_observer_work=['instrument real connector send/receive/install CUDA events per rank and request',
            'measure only same-device event elapsed time, retain overlap and never sum cross-device timestamps'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract-runtime')
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    value = (extract_runtime(binding(args.completion), binding(args.source_manifest))
             if args.command == 'extract-runtime' else collection_plan(binding(args.ledger), args.model_id))
    write_new(args.out, value)


if __name__ == '__main__':
    main()
