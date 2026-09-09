"""Strict whole-batch evidence and exact-budget lookup; no GPU operations."""
import hashlib
import json
import math
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def ack(state, generation, tokens, seqs):
    require(not state.get('error') and not state.get('runtime_error'), 'runtime error')
    require(state.get('generation') == state.get('acknowledged_generation') == generation,
            'generation lacks an actual owner ACK')
    require(state.get('scheduler_budget_pending') is None, 'budget pending')
    require(state.get('scheduler_budget_effective') ==
            dict(max_num_batched_tokens=tokens, max_num_seqs=seqs), 'effective budget mismatch')
    caches = [io.get('controls', {}).get('runtime') for io in state.get('scheduler_io', [])]
    require(caches and all(cache and cache.get('generation') == generation
                          and cache.get('error') is None for cache in caches), 'cache ACK mismatch')


def verify_restored(actual, initial, generation):
    budget = initial['scheduler_budget_effective']
    ack(actual, generation, budget['max_num_batched_tokens'], budget['max_num_seqs'])
    require(all(actual.get(key) == initial.get(key) for key in
                ('role', 'mode', 'admit_prefill', 'admit_decode')), 'initial role/mode/admission not restored')


def verify_drain(drain, generation):
    require(drain.get('drained') is True and drain.get('generation') == generation + 1,
            'owner drain proof missing')
    require(drain.get('drain_proof_type') == 'synchronous_put_owner_barrier'
        and drain.get('send_counters_verified') is True, 'verified v3 owner/send drain proof missing')
    require(isinstance(drain.get('transfer_observed_s'), (float, int))
        and math.isfinite(drain['transfer_observed_s']), 'owner transfer observation time missing')
    transfers = drain.get('transfers')
    require(isinstance(transfers, list) and transfers, 'all-rank transfer observations missing')
    for transfer in transfers:
        require(transfer.get('send_counters_observed') is True and transfer.get('send_healthy') is True
            and transfer.get('listener_alive') is True and transfer.get('inflight_sends') == 0
            and transfer.get('inflight_receives') == 0 and transfer.get('buffered_tensors') == 0
            and not transfer.get('allocations'), 'transport unhealthy or not fully drained')
        require(all(type(transfer.get(key)) is int and transfer[key] >= 0
                    for key in ('send_started', 'send_completed', 'send_failed'))
            and transfer['send_failed'] == 0 and transfer['send_started'] == transfer['send_completed'],
            'send accounting incomplete or failed')


def interference_evidence(raw, events):
    """Observed phase eligibility never discards a completed batch or its energy."""
    spec, rows = raw['spec'], raw['requests']
    offsets = spec['arrival_offsets_s']
    delays = [row['dispatch_s'] - row['planned_arrival_s'] for row in rows]
    limit = spec['arrival_lateness_limit_s']
    reasons = []
    if limit is None:
        reasons.append('arrival lateness tolerance was not declared')
    elif any(delay > limit for delay in delays):
        reasons.append('actual dispatch exceeded declared arrival tolerance')
    late = [index for index, offset in enumerate(offsets) if offset > min(offsets)]
    if not late:
        reasons.append('no predeclared staggered arrival')
    pairs = []
    for later in late:
        second = rows[later]
        found = False
        for earlier, first in enumerate(rows):
            if offsets[earlier] >= offsets[later]:
                continue
            first_id, second_id = first['request_id'], second['request_id']
            produced_before_plan = first['token_received_s'][0] < second['planned_arrival_s']
            produced_after_arrival = first['token_received_s'][-1] > second['dispatch_s']
            prior_decode = [event for event in events if event.get('decode', 0) > 0
                and first_id in event.get('request_ids', [])
                and isinstance(event.get('finished_s'), (int, float))
                and event['finished_s'] <= second['planned_arrival_s']]
            overlap = [event for event in events if event.get('prefill', 0) > 0
                and event.get('decode', 0) > 0
                and {first_id, second_id} <= set(event.get('request_ids', []))
                and isinstance(event.get('started_s'), (int, float))
                and event['started_s'] >= second['dispatch_s']]
            if produced_before_plan and produced_after_arrival and prior_decode and overlap:
                found = True
                pairs.append(dict(earlier_request_index=earlier, later_request_index=later,
                    prior_owner_decode_steps=len(prior_decode), owner_overlap_steps=len(overlap)))
        if not found:
            reasons.append('no verified preexisting decode and owner overlap for request ' + str(later))
    return dict(interference_valid=bool(late and not reasons), interference_ineligible_reasons=reasons,
                preexisting_decode_pairs=pairs, dispatch_delay_s=delays)


def derive(raw, power, metadata, clocks, events):
    from ecopadg.measure.power import trapezoid_energy
    from ecopadg.metrics import clip_power_window
    from ecopadg.serving.measurement import power_evidence
    require(raw.get('system') == 'pdblend' and raw.get('profile_kind') == 'whole_batch',
            'not a PDB whole-batch profile')
    require(raw.get('sampling_error') is None, 'power sampling failed')
    require(raw.get('identity_before') == raw.get('identity_after'), 'run identity changed')
    identity = raw.get('identity_before') or {}
    require(identity.get('model_files_sha256') and identity.get('engine_image'), 'model/image byte identity missing')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', identity['engine_image']), 'invalid immutable image identity')
    for key in ('model_files_sha256', 'source_files_at_import', 'live_vllm_and_serving_source_sha256',
                'measurement_source_sha256'):
        mapping = identity.get(key)
        require(isinstance(mapping, dict) and mapping and all(
            isinstance(path, str) and re.fullmatch(r'[0-9a-f]{64}', sha) for path, sha in mapping.items()),
            'missing/corrupted content identity: ' + key)
    spec = raw['spec']; generation = raw['generation']
    require(spec['target_gpus'] and len(set(spec['target_gpus'])) == len(spec['target_gpus'])
        and all(type(gpu) is int and gpu in range(8) for gpu in spec['target_gpus']), 'invalid target GPU identities')
    offsets = spec.get('arrival_offsets_s')
    require(isinstance(offsets, list) and len(offsets) == spec['batch_size']
        and all(isinstance(offset, (int, float)) and math.isfinite(offset) and offset >= 0 for offset in offsets)
        and min(offsets) == 0, 'declared arrival offsets missing or invalid')
    require(type(spec.get('preexisting_decode_required')) is bool
        and 'arrival_lateness_limit_s' in spec and spec.get('arrival_trigger'), 'declared arrival conditions missing')
    limit = spec['arrival_lateness_limit_s']
    require(limit is None or isinstance(limit, (int, float)) and math.isfinite(limit) and limit >= 0,
            'invalid arrival lateness condition')
    ack(raw['runtime_before'], generation, spec['budget_tokens'], spec['max_num_seqs'])
    ack(raw['runtime_after_requests'], generation, spec['budget_tokens'], spec['max_num_seqs'])
    requests = raw.get('requests', [])
    require(len(requests) == spec['batch_size'] == len(spec['input_lengths']) == len(spec['output_lengths']),
            'batch work missing')
    require(len({row['request_id'] for row in requests}) == len(requests), 'duplicate request evidence')
    start, end = raw['measurement_start_s'], raw['measurement_end_s']
    require(math.isfinite(start) and math.isfinite(end) and start < end, 'invalid measurement interval')
    for index, row in enumerate(requests):
        require(row.get('success') and row.get('http_status') == 200 and row.get('done_marker'),
                'request did not complete its stream')
        require(len(row.get('prompt_token_ids', [])) == spec['input_lengths'][index]
            and row.get('usage', {}).get('prompt_tokens') == spec['input_lengths'][index]
            and row.get('usage', {}).get('completion_tokens') == spec['output_lengths'][index]
            and len(row.get('output_token_ids', [])) == spec['output_lengths'][index], 'request work mismatch')
        times = row.get('token_received_s', [])
        require(len(times) == spec['output_lengths'][index] and times
            and all(math.isfinite(t) for t in times)
            and all(b >= a for a, b in zip(times, times[1:])), 'token timing evidence missing')
        require(start <= row['dispatch_s'] <= times[0] <= times[-1] <= row['stream_end_s'] <= end,
                'token/measurement timing mismatch')
        require(math.isclose(row.get('planned_arrival_s', -1), start + offsets[index], abs_tol=1e-6)
            and row['dispatch_s'] >= row['planned_arrival_s']
            and math.isclose(row.get('dispatch_delay_s', -1), row['dispatch_s'] - row['planned_arrival_s'], abs_tol=1e-6),
            'arrival plan or dispatch delay provenance mismatch')
    drain = raw.get('drain', {})
    verify_drain(drain, generation)
    require(raw['client_end_s'] <= raw['drain_response_s'] <= end, 'drain tail omitted')
    require(raw['runtime_drained'].get('active') == raw['runtime_drained'].get('running') ==
            raw['runtime_drained'].get('waiting') == 0, 'residual request work')
    ack(raw['runtime_drained'], generation + 1, spec['budget_tokens'], spec['max_num_seqs'])
    provenance = power_evidence(power, raw.get('power_source'), metadata)
    require(provenance['power_source_verified'], 'instant eight-GPU power provenance missing')
    require(len(power) >= 2 and all(len(watts) == 8 for _, watts in power), 'eight-GPU power missing')
    clipped = clip_power_window(power, start, end, pad_s=0)
    energy_per_gpu = [trapezoid_energy([(t, [watts[gpu]]) for t, watts in clipped]) for gpu in range(8)]
    require(events and any(event.get('tokens', 0) for event in events), 'owner execution events missing')
    require(all(event.get('generation') == generation and event.get('mode') == 'continuous'
        and event.get('role') == 'mixed' and type(event.get('tokens')) is int
        and 0 <= event['tokens'] <= spec['budget_tokens'] for event in events), 'owner step exceeds budget or generation')
    actual_ids = {row['request_id'] for row in requests}
    require(set(rid for event in events for rid in event.get('request_ids', [])) == actual_ids,
            'owner events contain missing or unrelated work')
    in_window = [(t, values) for t, values in clocks if start <= t <= raw['client_end_s']]
    require(in_window and all(len(values) == 8 and all(math.isfinite(value) and value >= 0 for value in values)
                             for _, values in in_window), 'observed clocks missing')
    observed_target = [[values[gpu] for gpu in spec['target_gpus']] for _, values in in_window]
    at_target = sum(all(abs(value - spec['clock_command_mhz']) <= 15 for value in values)
                    for values in observed_target)
    require(at_target >= 3, 'commanded frequency not observed on target GPUs during work')
    key = dict(system='pdblend', profile_kind='whole_batch', identity_sha256=digest(raw['identity_before']), **spec)
    interference = interference_evidence(raw, events)
    return dict(schema_version=1, system='pdblend', profile_kind='whole_batch', valid=True,
        legacy_profile_point_compatible=False, profile_key=key, profile_key_sha256=digest(key),
        generation=generation, energy_all_eight_gpus_j=sum(energy_per_gpu),
        energy_target_gpus_j=sum(energy_per_gpu[gpu] for gpu in spec['target_gpus']),
        energy_per_gpu_j=energy_per_gpu, duration_s=end-start,
        client_duration_s=raw['client_end_s']-start, drain_tail_s=end-raw['client_end_s'],
        first_token_latency_s=[row['token_received_s'][0]-row['dispatch_s'] for row in requests],
        command_target_observed_samples=at_target, clock_samples_during_work=len(in_window),
        observed_target_clock_min_mhz=min(min(values) for values in observed_target),
        observed_target_clock_max_mhz=max(max(values) for values in observed_target),
        energy_boundary='release of entire batch through observed owner drain; setup excluded',
        **provenance, **interference)


def lookup(profiles, exact_key):
    require(exact_key.get('system') == 'pdblend' and exact_key.get('profile_kind') == 'whole_batch',
            'whole-batch PDB key required')
    require(type(exact_key.get('budget_tokens')) is int and exact_key['budget_tokens'] > 0,
            'explicit budget required; no cross-budget fallback')
    require(exact_key.get('identity_sha256'), 'explicit model/image/source identity required')
    matches = []
    for profile in profiles:
        require(profile.get('valid') is True and profile.get('legacy_profile_point_compatible') is False,
                'invalid or incompatible profile')
        require(profile.get('profile_key_sha256') == digest(profile.get('profile_key')), 'profile key corrupted')
        if profile['profile_key'] == exact_key:
            if not exact_key.get('preexisting_decode_required') or profile.get('interference_valid') is True:
                matches.append(profile)
    require(matches, 'no exact identity/workload/frequency/budget profile; interpolation is not implemented')
    return matches
