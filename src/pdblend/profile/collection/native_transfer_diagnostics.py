"""Read-only decomposition of the existing signed second-output diagnostic.

This does not fit a transfer model, change acceptance thresholds, or infer
unobserved CUDA/NCCL timestamps from client events. Physical transfer remains
unknown, including when the signed endpoint difference happens to be positive.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .native_runtime_audit import validate_transfer_payload


def decompose(result, *, input_tokens, prefill_instance, decode_instance):
    measured = validate_transfer_payload(result, input_tokens=input_tokens,
        prefill_instance=prefill_instance, decode_instance=decode_instance)
    first, second = result['ordinary']
    pre, dec = result['prefill'], result['combined']
    values = dict(
        ordinary_first_output_s=first['first_token_s'] - first['submitted_s'],
        ordinary_first_to_second_s=first['token_times_s'][1] - first['first_token_s'],
        pd_prefill_client_s=pre['first_token_s'] - pre['submitted_s'],
        client_bridge_s=dec['decode_submitted_s'] - pre['finished_s'],
        decode_submit_to_next_output_s=dec['decode_first_token_s'] - dec['decode_submitted_s'],
        second_ordinary_second_output_s=second['token_times_s'][1] - second['submitted_s'])
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError('client segment is negative or nonfinite')
    values['prefill_endpoint_difference_s'] = values['pd_prefill_client_s'] - values['ordinary_first_output_s']
    values['decode_endpoint_difference_s'] = values['decode_submit_to_next_output_s'] - values['ordinary_first_to_second_s']
    reconstructed = values['prefill_endpoint_difference_s'] + values['client_bridge_s'] + values['decode_endpoint_difference_s']
    if not math.isclose(reconstructed, measured['overhead_s'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('signed endpoint decomposition differs')
    values['ordinary_reference_difference_s'] = values['second_ordinary_second_output_s'] - measured['mixed_second_output_s']
    return dict(**measured, **values, reconstructed_overhead_s=reconstructed,
        physical_transport_s=None, physical_transport_identifiable=False,
        formal_eligible=False)


def review(report, original_audit):
    """Consume immutable original report/audit, preserve every failed holdout."""
    ref = report['journal']
    raw = Path(ref['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('runtime journal checksum differs')
    journal = [json.loads(line) for line in raw.splitlines() if line.strip()]
    values = []
    for row in journal:
        if row.get('kind') != 'transfer_request':
            continue
        epoch = row['native_epoch']
        part = decompose(row['result'], input_tokens=row['input_tokens'],
            prefill_instance=epoch['prefill_instance'], decode_instance=epoch['decode_instance'])
        values.append(dict(sequence=row['sequence'], tag=row['tag'], repeat=row['repeat'],
            input_tokens=row['input_tokens'], purpose=row['purpose'],
            measured_pair='-measure-' in row['tag'], **part))
    groups = defaultdict(list)
    for row in values:
        if row['measured_pair']:
            groups[(row['input_tokens'], row['repeat'], row['purpose'])].append(row)
    summary = []
    for (length, repeat, purpose), rows in sorted(groups.items()):
        fields = [k for k, v in rows[0].items() if k.endswith('_s') and type(v) in (int, float)]
        summary.append(dict(input_tokens=length, repeat=repeat, purpose=purpose, pairs=len(rows),
            medians={k: statistics.median(r[k] for r in rows) for k in fields},
            median_note='Medians summarize each field separately; the additive identity is verified per request.'))
    comparisons = original_audit['holdout_comparisons']
    transfers = [r for r in comparisons if r['component'].startswith('transfer_')]
    for row in transfers:
        length = int(row['component'].split('_')[1])
        observed = {s['repeat']: s['medians']['overhead_s'] for s in summary if s['input_tokens'] == length}
        if ([observed[i] for i in range(3)] != row['training'] or observed[3] != row['heldout']):
            raise ValueError('original transfer holdout does not replay exactly')
    return dict(schema='pdblend-native-transfer-diagnostic-review/v1', journal=ref,
        journal_kinds=dict(Counter(r['kind'] for r in journal)), decomposition=values, summaries=summary,
        original_transfer_holdout=transfers, original_holdout_passed=original_audit['holdout_passed'],
        original_holdout_error_summary=original_audit['holdout_error_summary'],
        unchanged_holdout_limits=report['runtime_plan']['holdout_limits'],
        total_component_comparisons=len(comparisons),
        nontransfer_max_relative_error=max(r['relative_error'] for r in comparisons if not r['component'].startswith('transfer_')),
        physical_transport_observed=False, physical_transport_recoverable_from_client_journal=False,
        negative_delta_preserved=True, training_refitted=False, qualification_changed=False,
        can_continue_unrelated_sampling=bool(report['ready_for_timing'] and report['safe_restore_passed']),
        formal_eligible=False)


def proposed_plan():
    """Prospective protocol specification; no job, hardware or qualification."""
    return dict(schema='pdblend-native-transport-proposed-plan/v1', status='design_requires_worker_instrumentation',
        selection_split='new_calibration_and_new_independent_holdout', evaluation_used_for_selection=False,
        old_failed_holdout_reused_for_fit=False, old_results_unchanged=True,
        input_tokens=[512, 2048, 7168], logical_output_tokens=16, training_repeats=3, holdout_repeats=1,
        training_seed=9701, holdout_seed=9702, settle_s=2., measure_s=5., active_frequency_mhz=2520,
        holdout_limits=dict(mean_relative_error=.10, p95_relative_error=.20, max_relative_error=.25),
        estimator='median of request observations per training window, then median of three window values',
        candidate_freeze='all nine training windows complete and candidate SHA frozen before any new holdout workload',
        execution_scope='one outstanding same-TP P/D pair on the existing eight-GPU native32 eager fleet; other instances stay loaded and idle',
        worker_extension='new PD-only subclass/hook; exact public engine/connector/client bytes unchanged',
        collection_interface='collect_native_transport(specs,fleet,meter,sampler,out,plan); caller owns sampler/lease',
        native_channels=[
            'exact per-rank request/layer IDs, GPU UUID, generation, shapes, dtypes, source/destination slots and transferred byte counts',
            'source layer gather events and send-queue/batch envelopes, preserving existing asynchronous scheduling',
            'CUDA events directly around actual ncclSend/ncclRecv enqueue on their existing streams; no new in-window synchronize',
            'send batch tensor-ID map, per-peer ordered receive chunks and receiver-loaded tensor IDs; strict isolated one-request join',
            'receiver load/injection events on compute stream; bind all attention layers and ranks before D compute',
            'native P compute/first-output, send completion, D load/compute boundaries plus unchanged client token timeline'],
        estimands={
            'kv_nccl_device_s': 'per request maximum over TP ranks of sum of its receive CUDA intervals; includes peer rendezvous, not pure link bandwidth',
            'kv_source_prepare_s': 'source gather and batching observations, reporting overlaps without summing an end-to-end critical path',
            'kv_receiver_install_s': 'receiver load/injection interval, separated from D compute and HTTP',
            'handoff_critical_path': 'event dependency graph retaining overlap of P compute, send, receive and injection',
            'second_output_delta_s': 'unchanged signed client diagnostic only; never a physical duration'},
        exclusions=['no truncation of negative endpoint differences', 'no absolute value as transport cost',
                    'no holdout-dependent threshold change or candidate refit',
                    'no summing sender and receiver NCCL durations for the same transfer',
                    'no adding already-overlapped transport to prefill without an explicit latency-model derivation'],
        ready_for_timing_rule='safety only: hooks removed, events harvested outside window, measurement stopped, complete original fleet/generation restored, clock2520 observed, all-rank drain, sampler healthy',
        remaining_gates=['CPU observer-semantic/no-op parity tests', 'same-fleet first exact-token functional sample',
                         'all-rank complete native byte/layer/event provenance', 'new frozen physical-metric holdout',
                         'explicit composition semantics for carried-first-token TTFT and second output',
                         'load/concurrency-domain independent serving holdout'],
        transfer_seconds_replacement_authorized=False, component_qualified=False, formal_eligible=False)
