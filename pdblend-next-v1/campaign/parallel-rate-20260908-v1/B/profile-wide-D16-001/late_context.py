"""B TP2 source-ordered context evidence. CPU-only; never creates ProfilePoint.

The unchanged shared context algorithm sees nonempty model steps. Every original
line, including empty steps, retains its ordinal, hash and actual time. Empty
steps contribute no tokens; finish spacings and energy keep their elapsed time.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import context_export

ROOT = Path(__file__).resolve().parent
INSTANCE = 'nextv3b0'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def finite(x): return type(x) in (int, float) and math.isfinite(x)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def source_record(actual, provenance, expected, *, observed_s, contract_path=None):
    """Call only after the original full model/container/live-source identity gate.

    Actual includes freshly read seven scheduler/default sources, not bundled
    review bytes. The extra provenance read supplies the actual engine PID.
    """
    path = Path(contract_path) if contract_path else ROOT/'source-order-contract.json'
    contract = read(path)
    require(all(actual.get(k) == v for k, v in expected.items()), 'complete B source/model identity differs')
    require(provenance.get('instance_id') == INSTANCE and provenance.get('tp') == actual.get('tp') == 2
            and provenance.get('model') == actual.get('model')
            and provenance.get('source_files_at_import') == actual.get('source_files_at_import'),
            'fresh engine provenance differs from fully verified identity')
    require(type(provenance.get('pid')) is int and provenance['pid'] > 0 and finite(observed_s),
            'actual engine process/time missing')
    files = {p: v['sha256'] for p, v in contract['files'].items()}
    require(actual.get('engine_image') == contract['image'] and actual.get('target_gpus') == [0, 1]
            and actual.get('tp') == contract['tp'] == 2, 'B TP2 image/GPU namespace differs')
    require(all(actual['live_vllm_and_serving_source_sha256'].get(p) == value for p, value in files.items()),
            'live scheduler/default source does not match reviewed ordering')
    logger = contract['engine_logger_source']
    require(actual['source_files_at_import'].get(logger) == contract['engine_logger_sha256']
            and actual['live_vllm_and_serving_source_sha256'].get(logger) == contract['engine_logger_sha256'],
            'loaded/live B engine logger source differs')
    return dict(schema=1, instance_id=INSTANCE, tp=2, target_gpus=[0, 1], engine_pid=provenance['pid'],
                container_id=actual['container_id'], container_started_at=actual['container_started_at'],
                container_pid=actual['container_pid'], engine_image=actual['engine_image'],
                identity_sha256=digest(actual), contract_sha256=sha(path), files=files,
                engine_logger_sha256=contract['engine_logger_sha256'], observed_s=observed_s,
                read_only=True, actual_live_source_hashes=True, loaded_logger_verified=True,
                physical_kv_directly_observed=False)


def validate_source_pair(before, after, raw, *, expected, contract_path=None):
    path = Path(contract_path) if contract_path else ROOT/'source-order-contract.json'
    contract = read(path)
    identities = (raw.get('identity_before'), raw.get('identity_after'))
    require(identities[0] == identities[1] and isinstance(identities[0], dict), 'complete run identity changed/missing')
    require(all(identities[0].get(k) == v for k, v in expected.items()), 'run identity not the expected B source/model')
    for record in (before, after):
        require(record.get('read_only') is True and record.get('actual_live_source_hashes') is True
                and record.get('loaded_logger_verified') is True and record.get('instance_id') == INSTANCE
                and record.get('tp') == 2 and record.get('target_gpus') == [0, 1]
                and record.get('engine_image') == contract['image']
                and record.get('identity_sha256') == digest(identities[0])
                and record.get('contract_sha256') == sha(path)
                and record.get('files') == {p: v['sha256'] for p, v in contract['files'].items()}
                and record.get('engine_logger_sha256') == contract['engine_logger_sha256']
                and type(record.get('engine_pid')) is int and record['engine_pid'] > 0,
                'actual before/after source-order receipt missing/inconsistent')
    a = {k: v for k, v in before.items() if k != 'observed_s'}
    b = {k: v for k, v in after.items() if k != 'observed_s'}
    require(a == b and all(before.get(k) == identities[0].get(k) and before.get(k) is not None
                           for k in ('container_id', 'container_pid', 'container_started_at', 'engine_image')),
            'engine/container process/source changed between actual source receipts')
    require(finite(before.get('observed_s')) and finite(after.get('observed_s'))
            and before['observed_s'] <= raw['measurement_start_s'] < raw['measurement_end_s'] <= after['observed_s'],
            'actual source receipts do not enclose point')
    return dict(verified=True, before_observed_s=before['observed_s'], after_observed_s=after['observed_s'],
                contract_sha256=sha(path), identity_sha256=digest(identities[0]), engine_pid=before['engine_pid'],
                source_order_inferred_from_verified_code=True, physical_kv_directly_observed=False)


def parse_events(payload):
    """No slicing of the original post-warmup owner suffix; byte-level mapping."""
    events, refs, offset = [], [], 0
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        require(line.strip(), 'blank/missing owner event line')
        event = json.loads(line)
        require(isinstance(event, dict), 'owner event must be an object')
        events.append(event)
        refs.append(dict(original_owner_event_index=len(events)-1, original_line_number=line_number,
                         original_byte_offset=offset, original_line_bytes=len(line),
                         original_line_sha256=hashlib.sha256(line).hexdigest()))
        offset += len(line)
    require(events, 'complete original owner suffix absent')
    return events, refs


def normalized_work(raw):
    spec = raw['spec']; batch = spec['batch_size']
    require(type(batch) is int and batch > 0 and spec.get('tp') == 2 and spec.get('target_gpus') == [0, 1]
            and spec.get('budget_tokens') == 8192 and spec.get('max_num_seqs') == 32,
            'B TP2 / 8192 / 32 namespace required')
    inputs, outputs = spec.get('input_lengths'), spec.get('output_lengths')
    require(isinstance(inputs, list) and isinstance(outputs, list) and len(inputs) == len(outputs) == batch
            and all(type(x) is int and x > 0 for x in inputs + outputs)
            and len(set(inputs)) == len(set(outputs)) == 1, 'homogeneous actual declared work missing')
    normalized = copy.copy(raw)
    normalized['spec'] = dict(spec, input_tokens=inputs[0], output_tokens=outputs[0])
    return normalized


def reconstruct(raw, payload, *, source_order_verified):
    normalized = normalized_work(raw)
    events, refs = parse_events(payload)
    nonempty, mapping, empty = [], [], []
    previous_end = None
    for event, ref in zip(events, refs):
        start, end = event.get('started_s'), event.get('finished_s')
        require(type(event.get('generation')) is int and event['generation'] == raw['generation']
                and event.get('role') == 'mixed' and event.get('mode') == 'continuous'
                and finite(start) and finite(end)
                and raw['measurement_start_s'] <= start < end <= raw['measurement_end_s']
                and (previous_end is None or start >= previous_end-1e-6),
                'complete owner sequence has generation/time/order inconsistency')
        previous_end = end
        if event.get('tokens') == 0:
            require(type(event['tokens']) is int and type(event.get('prefill')) is int
                    and type(event.get('decode')) is int and event['prefill'] == event['decode'] == 0
                    and event.get('request_ids') == [], 'zero-token owner step contains unaccounted work')
            empty.append(dict(ref, event=event, contributes_decode_tokens=0))
        else:
            nonempty.append(event); mapping.append(ref)
    result = context_export.reconstruct(normalized, nonempty, generation=raw['generation'],
                                        token_budget=8192, source_order_verified=source_order_verified)
    late = result['late_window']
    filtered = list(late['owner_event_indices'])
    late['nonempty_owner_event_indices'] = filtered
    late['owner_event_indices'] = [mapping[i]['original_owner_event_index'] for i in filtered]
    for step in late['logical_context_by_step']:
        step['nonempty_owner_event_index'] = step['owner_event_index']
        step.update(mapping[step['owner_event_index']])
        step['owner_event_index'] = step['original_owner_event_index']
    late['empty_steps_inside_window'] = [r for r in empty if late['start_s'] <= r['event']['started_s']
                                        and r['event']['finished_s'] <= late['end_s']]
    late['consecutiveness_semantics'] = '65 nonempty full-batch decode steps; zero-work steps retained in time/energy, not counted as decode'
    result.update(full_owner_event_count=len(events), nonempty_owner_event_count=len(nonempty),
                  original_events_sha256=hashlib.sha256(payload).hexdigest(), original_line_mapping=refs,
                  empty_owner_steps=empty, nonempty_to_original_mapping=mapping,
                  raw_spec_sha256=digest(raw['spec']), normalized_spec=normalized['spec'], tp2_rank_count=2,
                  owner_count=1, owner_identity_source='full source/process proof and one acknowledged runtime scheduler',
                  original_empty_steps_retained=True, empty_elapsed_removed=False)
    return result, events


def read_power(path):
    with Path(path).open() as handle:
        return [(float(r['t_s']), [float(r[f'gpu{i}_w']) if r.get(f'gpu{i}_w') else None for i in range(8)])
                for r in csv.DictReader(handle)]


def integrate(power, start, end):
    from ecopadg.metrics import clip_power_window
    from ecopadg.measure.power import trapezoid_energy
    require(finite(start) and finite(end) and start < end and len(power) >= 2
            and all(finite(t) and len(v) == 8 and all(finite(x) and x >= 0 for x in v) for t, v in power)
            and all(b[0] > a[0] for a, b in zip(power, power[1:]))
            and power[0][0] <= start < end <= power[-1][0], 'eight-GPU power absent/invalid/unbracketed')
    clipped = clip_power_window(power, start, end, pad_s=0)
    per_gpu = [trapezoid_energy([(t, [v[i]]) for t, v in clipped]) for i in range(8)]
    return dict(all_eight_gpu_energy_j=sum(per_gpu), target_gpus=[0, 1],
                target_gpus_energy_j=sum(per_gpu[:2]), energy_per_gpu_j=per_gpu,
                target_gpus_mean_power_w=sum(per_gpu[:2])/(end-start), duration_s=end-start)


def attach_tp2_window(window, power, clocks, frequency):
    start, end = window['start_s'], window['end_s']
    # Preserve measured energy even when actual-clock qualification subsequently fails.
    window.update(integrate(power, start, end))
    require(clocks and all(finite(t) for t, _ in clocks)
            and all(b[0] > a[0] for a, b in zip(clocks, clocks[1:]))
            and clocks[0][0] <= start < end <= clocks[-1][0], 'actual clock stream unbracketed')
    active = [(t, v) for t, v in clocks if start <= t <= end]
    require(len(active) >= 3 and all(len(v) == 8 and all(finite(x) and x >= 0 for x in v) for _, v in active),
            'actual eight-board clock observations missing')
    observed = [[v[g] for _, v in active] for g in (0, 1)]
    window.update(actual_target_clock_min_mhz=[min(v) for v in observed],
                  actual_target_clock_max_mhz=[max(v) for v in observed],
                  actual_clock_samples=len(active), actual_clock_times_s=[t for t, _ in active],
                  frequency_command_mhz=frequency, energy_includes_interstep_gaps=True,
                  energy_is_not_net_incremental=True)
    require(all(abs(x-frequency) <= 15 for values in observed for x in values),
            'one or both actual TP2 GPU clocks differ during measured window')
    window['actual_both_target_clocks_valid'] = True


def full_decode_windows(raw, events, power, clocks):
    ids = {r['request_id'] for r in raw['requests']}; runs, run = [], []
    for index, event in enumerate(events):
        if event['tokens'] == 0:
            continue  # Absolute timestamps and original indices remain unchanged.
        if event['prefill'] == 0 and event['decode'] == len(ids) and set(event['request_ids']) == ids:
            run.append((index, event))
        else:
            if run: runs.append(run)
            run = []
    if run: runs.append(run)
    windows = []
    for run in runs:
        if len(run) < 2: continue
        gaps = [b[1]['finished_s']-a[1]['finished_s'] for a, b in zip(run, run[1:])]
        window = dict(start_s=run[0][1]['started_s'], end_s=run[-1][1]['finished_s'],
                      nonempty_decode_steps=len(run), original_owner_event_indices=[i for i, _ in run],
                      empty_steps_between=sum(events[i]['tokens'] == 0 for i in range(run[0][0], run[-1][0]+1)),
                      finish_spacing_values_s=gaps, finish_spacing_max_s=max(gaps))
        attach_tp2_window(window, power, clocks, raw['spec']['clock_command_mhz'])
        windows.append(window)
    require(windows, 'complete actual full-batch decode windows absent')
    return windows


def audit_point(path, before, after, *, expected, original_evidence, tp2_validator, contract_path=None,
                expected_spec=None):
    path = Path(path)
    result = dict(point_id=path.name, valid=False, whole_batch_energy=None, errors=[], profile_point_generated=False)
    sources = {}
    def use(name):
        p = path/name; sources[str(p)] = sha(p); return p
    stage = 'raw_power'
    try:
        raw = read(use('raw.json')); power = read_power(use('power.csv'))
        result['whole_batch_energy'] = integrate(power, raw['measurement_start_s'], raw['measurement_end_s'])
        result['warmup'] = dict(present=bool(raw.get('warmup')), included_in_point_primary_energy=False,
                                included_in_outer_operation_energy='must verify original outer interval after cleanup')
        stage = 'original_whole_batch'
        metadata = read(use('power-metadata.json')); clocks = read(use('clocks.json'))
        payload = use('events.jsonl').read_bytes(); events, _ = parse_events(payload)
        require(not raw.get('error'), 'original point execution failed')
        if expected_spec is not None:
            require(raw['spec'] == expected_spec, 'actual point spec differs from original declaration')
        result['original_whole_batch'] = original_evidence.derive(raw, power, metadata, clocks, events)
        result['original_tp2_observation'] = tp2_validator(raw, events)
        stage = 'actual_source_order'
        proof = validate_source_pair(before, after, raw, expected=expected, contract_path=contract_path)
        result['source_order_proof'] = proof
        stage = 'complete_per_id_context'
        late, events = reconstruct(raw, payload, source_order_verified=proof['verified'])
        result['late_context'] = late
        stage = 'late_and_full_decode_tp2_power_clock'
        attach_tp2_window(late['late_window'], power, clocks, raw['spec']['clock_command_mhz'])
        result['full_decode_windows'] = full_decode_windows(raw, events, power, clocks)
        result['valid'] = True
    except Exception as error:
        result['errors'].append(dict(stage=stage, error=repr(error)))
    result['source_sha256'] = sources
    if not all(sha(p) == value for p, value in sources.items()):
        result['valid'] = False; result['errors'].append(dict(stage='immutable_read', error='raw sources changed during audit'))
    return result


def derive_all(root, *, original_evidence, tp2_validator):
    """Called after original helper final identity/cleanup. Writes only new evidence."""
    root = Path(root); contract_path = root/'source-order-contract.json'
    paths = sorted((root/'source-order').glob('*.json'))
    report = dict(schema=1, complete=False, points=[], profile_point_generated=False,
                  scope='B TP2 empirical observations; not a certified profile or GPU correctness replacement')
    before = read(paths[0]) if len(paths) == 2 else {}
    after = read(paths[-1]) if len(paths) == 2 else {}
    report['source_receipts'] = {str(p): sha(p) for p in paths}
    expected = read(root/'expected-identity.json')
    campaign = read(root/'results/campaign.json')
    for index in range(6):
        path = root/'results'/f'point-{index:04d}'
        items = campaign.get('points', [])
        spec = items[index]['spec'] if index < len(items) else None
        result = audit_point(path, before, after, expected=expected, original_evidence=original_evidence,
                             tp2_validator=tp2_validator, contract_path=contract_path, expected_spec=spec)
        if not (spec and spec.get('batch_size') == 16 and spec.get('input_lengths') == [2304]*16
                and spec.get('output_lengths') == [640]*16 and spec.get('clock_command_mhz') == (1500 if index < 3 else 2520)
                and spec.get('budget_tokens') == 8192 and spec.get('max_num_seqs') == 32
                and spec.get('target_gpus') == [0, 1] and spec.get('tp') == 2
                and spec.get('seed') == 0 and spec.get('temperature') == 0
                and spec.get('arrival_offsets_s') == [0.]*16):
            result['valid'] = False
            result['errors'].append(dict(stage='declaration', error='not the six prescribed B16 2304->640 work points'))
        if result['valid']:
            late=result['late_context']
            per_id=late['per_request']
            if not (late['complete_prefill_token_sum']==36864 and late['complete_decode_token_sum']==10224
                    and len(per_id)==16 and all(v['total_observed_decode_steps']==639
                        and v['complete_work_attention_upper']==2943 and v['attention_after_max']>=2632 and v['declared_bucket_edge']==2944 for v in per_id.values())):
                result['valid']=False
                result['errors'].append(dict(stage='complete_domain',error='full per-ID decode or shared context2633 domain absent'))
        if path.exists():
            write_new(path/'late-context.json', result)
            result = dict(point_id=path.name, valid=result['valid'], errors=result['errors'],
                          late_context_sha256=sha(path/'late-context.json'),
                          whole_batch_energy=result['whole_batch_energy'])
        report['points'].append(result)
    report['complete'] = campaign.get('complete') is True and len(paths) == 2 and all(p['valid'] for p in report['points'])
    write_new(root/'late-context-evidence.json', report)
    return report


def warmup_boundaries(root, start, end):
    """All warmup work is inside the separate outer operation, never added twice."""
    rows = []
    for index in range(6):
        path = Path(root)/'results'/f'point-{index:04d}'/'raw.json'
        item = dict(point_id=path.parent.name, valid=False, raw_sha256=None)
        try:
            item['raw_sha256'] = sha(path); raw = read(path); warm = raw['warmup']
            item.update(dispatch_s=warm.get('dispatch_s'), stream_end_s=warm.get('stream_end_s'))
            require(warm.get('success') is True and warm.get('done_marker') is True
                    and len(warm.get('prompt_token_ids', [])) == 128
                    and len(warm.get('output_token_ids', [])) == len(warm.get('token_received_s', [])) == 64
                    and warm.get('usage', {}).get('prompt_tokens') == 128
                    and warm.get('usage', {}).get('completion_tokens') == 64,
                    'original128->64 warmup incomplete')
            require(finite(start) and finite(end) and start <= warm['dispatch_s'] < warm['stream_end_s']
                    <= raw['measurement_start_s'] < raw['measurement_end_s'] <= end,
                    'warmup/point not enclosed by full outer energy window')
            item['valid'] = True
        except Exception as error:
            item['error'] = repr(error)
        rows.append(item)
    return dict(valid=all(row['valid'] for row in rows), warmups=rows,
                outer_start_s=start, outer_end_s=end, warmup_count_expected=6,
                all_warmups_in_outer_energy=all(row['valid'] for row in rows),
                primary_excludes_warmup=True, primary_and_outer_overlap=True,
                do_not_add_primary_and_outer=True)
