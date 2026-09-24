"""Recompute bounded transition energy from immutable native per-GPU samples."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import json
import math
from pathlib import Path

from .deployment import save, sha


def integrate(readings, start, end, *, max_gap_s=.5):
    """Trapezoidal integration, bracketed at both ends, without extrapolation."""
    if not all(math.isfinite(v) for v in (start, end, max_gap_s)) or end <= start or max_gap_s <= 0:
        raise ValueError('positive finite measurement window required')
    rows = sorted(readings)
    if len(rows) < 2 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('at least two unique ordered power timestamps required')
    if any(not math.isfinite(t+p) or p <= 0 for t, p in rows):
        raise ValueError('finite positive actual power required')
    ts = [r[0] for r in rows]
    if start < ts[0] or end > ts[-1]:
        raise ValueError('power samples do not bracket transition window')

    def boundary(t):
        i = bisect_left(ts, t)
        if ts[i] == t:
            return rows[i]
        a, b = rows[i-1], rows[i]
        if b[0]-a[0] > max_gap_s:
            raise ValueError('transition power sample gap exceeds protocol')
        return t, a[1]+(b[1]-a[1])*(t-a[0])/(b[0]-a[0])

    selected = [boundary(start), *(r for r in rows if start < r[0] < end), boundary(end)]
    if any(b[0]-a[0] > max_gap_s for a, b in zip(selected, selected[1:])):
        raise ValueError('transition power sample gap exceeds protocol')
    energy = sum((a[1]+b[1])*.5*(b[0]-a[0]) for a, b in zip(selected, selected[1:]))
    return dict(energy_j=energy, duration_s=end-start, samples=len(selected),
        maximum_sample_gap_s=max(b[0]-a[0] for a, b in zip(selected, selected[1:])),
        interpolation='linear only between bracketing actual sensor samples', extrapolation=False)


def derive(attempt):
    attempt = Path(attempt).resolve()
    completion = json.loads((attempt/'completion.json').read_text())
    if (completion.get('system') != 'dynamollm'
            or completion.get('status') != 'passed' or completion.get('transition_complete') is not True
            or completion.get('own_cleanup_complete') is not True):
        raise ValueError('passed native transition, output and cleanup evidence required')
    evidence = {str(attempt/'completion.json'): sha(attempt/'completion.json')}
    for name, expected in completion['artifacts'].items():
        path = attempt/name
        if sha(path) != expected:
            raise ValueError('native transition artifact SHA differs: '+name)
        evidence[str(path)] = expected
    capabilities = json.loads((attempt/'capabilities.json').read_text())
    if any(capabilities[key].get('engine_revision') != 'vllm-0.10.1.1'
           or capabilities[key].get('model_id') != completion['model_id']
           for key in ('source', 'normal_target')):
        raise ValueError('actual native engine/model identity differs')
    from pdblend.results.journal import iter_journal
    rows = list(iter_journal(attempt/completion.get('journal_path','events.jsonl')))
    transitions = [row for row in rows if row['event']=='dynamo_transition' and row.get('phase')=='complete']
    if len(transitions) != 1:
        raise ValueError('exactly one complete native transition required')
    event = transitions[0]
    transition = event['transition']
    from .reconfiguration import Transition
    from .transition_probe_v1 import audit_receipts
    spec = Transition(**{**transition, 'source_ids':tuple(transition['source_ids']),
        'source_layout':tuple(map(tuple, transition['source_layout'])),
        'target_layout':tuple(map(tuple, transition['target_layout'])),
        'target_shapes':tuple(transition['target_shapes'])})
    golden = json.loads((attempt/'goldens.json').read_text())[str(completion['target']['tp'])]
    audit = audit_receipts(rows, transaction=spec, source=completion['source'],
        target=completion['target'], golden=golden)
    if audit.get('passed') is not True:
        raise ValueError('native transition raw re-audit failed: '+repr(audit))
    end = event['at_s']
    start = end-event['duration_s']
    prepared = [r for r in rows if r['event']=='dynamo_gpu_target_prepared'
                and r.get('transaction_id') == transition['transaction_id']]
    if len(prepared) != 1:
        raise ValueError('one real target transfer timing required')
    transfer = prepared[0]
    if not start <= transfer['transfer_started_s'] < transfer['transfer_finished_s'] <= end:
        raise ValueError('physical transfer timestamps fall outside transaction')
    uuids = completion['gpu_uuids']
    touched = {g for group in (*transition['source_layout'], *transition['target_layout']) for g in group}
    if {int(g) for g in uuids} != touched or len(set(uuids.values())) != len(touched):
        raise ValueError('transaction GPU inventory differs from actual telemetry UUIDs')
    by_gpu = {gpu: [] for gpu in touched}
    for row in rows:
        if row['event'] != 'dynamo_power':
            continue
        gpu = row.get('gpu')
        if gpu not in by_gpu or row.get('gpu_uuid') != uuids[str(gpu)]:
            raise ValueError('power sensor is outside original transaction UUID group')
        if 'error' in row:
            if start <= row['at_s'] <= end:
                raise ValueError('actual power sampling error during transition')
            continue
        if row.get('source') != 'nvml:field:186:scope:0:mW' or row.get('sensor_timestamp_us', 0) <= 0:
            raise ValueError('native instantaneous power sensor identity missing')
        t = row['sensor_timestamp_us']/1e6
        # Repeated reads of one actual sample add no independent evidence.
        sample = (t, row['power_w'])
        if by_gpu[gpu] and t == by_gpu[gpu][-1][0]:
            if sample != by_gpu[gpu][-1]:
                raise ValueError('same sensor timestamp carries conflicting power')
            continue
        by_gpu[gpu].append(sample)
    windows = {}
    for label, a, b in [('complete_transition', start, end),
                        ('weight_transfer', transfer['transfer_started_s'], transfer['transfer_finished_s'])]:
        per_gpu = {str(g): integrate(readings, a, b) for g, readings in by_gpu.items()}
        windows[label] = dict(start_s=a, end_s=b, duration_s=b-a,
            energy_j=sum(r['energy_j'] for r in per_gpu.values()), per_gpu=per_gpu)
    loaded = completion.get('loaded_drain')
    envelope = dict(max_input_tokens=0,max_output_tokens=0,active_requests=0)
    if loaded:
        if ('loaded-drain.json' not in completion['artifacts'] or not loaded.get('measured')
                or json.loads((attempt/'loaded-drain.json').read_text()) != loaded):
            raise ValueError('loaded-drain independent raw receipt missing')
        live=loaded['live'];ids=live['request_ids'];outcomes=loaded['outcomes']
        records=[r for r in rows if r['event']=='dynamo_loaded_drain_live']
        done=[r for r in rows if r['event']=='dynamo_loaded_drain_outcome']
        if (len(records)!=1 or len(set(ids))!=live['batch'] or len(outcomes)!=live['batch']
                or {r['request_id'] for r in outcomes}!=set(ids)
                or {r['request_id'] for r in done}!=set(ids)
                or not live['observed_s']<=start<=live['observed_s']+1
                or any(not r.get('ok') or not r.get('finished')
                       or len(r['token_ids'])!=live['output_tokens'] for r in outcomes)
                or any(r['finished_s']>end for r in outcomes)
                or any(not live['state']['kv_allocations'].get(rid) for rid in ids)):
            raise ValueError('actual live-KV/output/drain chronology differs')
        envelope=loaded['workload_envelope']
        if envelope!={'max_input_tokens':live['input_tokens'],'max_output_tokens':live['output_tokens'],
                      'max_batch':live['batch']}:
            raise ValueError('loaded-drain workload envelope differs from real requests')
    return dict(schema='dynamo-native-transition-components-v1', system='dynamollm',
        model_id=completion['model_id'], engine_revision='vllm-0.10.1.1',
        source_sha256=completion['source_sha256'], image_digest=completion['image_digest'],
        source_tps=sorted(map(len, transition['source_layout'])),
        target_tps=sorted(map(len, transition['target_layout'])),
        measurement='hardware', gpu_uuids=uuids, native_receipts=audit, windows=windows,
        duration_s=event['duration_s'], energy_j=windows['complete_transition']['energy_j'],
        window_alignment='completion wall timestamp minus measured monotonic transaction duration',
        stationary_weight_bytes=transfer['stationary_weight_bytes'],
        workload_envelope=envelope,
        drain_under_load_measured=bool(loaded), full_policy_cost_eligible=bool(loaded),
        golden_workload=dict(input_tokens=len(golden['prompt']), output_tokens=len(golden['token_ids'])),
        evidence=evidence, raw_unchanged=True, formal_eligible=False, energy_comparable=False,
        limitations=([('measured live workload only; no larger input/output/batch extrapolation') if loaded else
                      'empty-KV migration only; golden is not a loaded drain measurement',
            'one measured layout only; no extrapolation to multi-replica relay or retire-only',
            'group power includes every original leased GPU; not comparable eight-GPU energy']))


def prepare(attempt, out):
    value = derive(attempt)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'audit.json', value)
    cost = {k:value[k] for k in ('system','model_id','engine_revision','source_sha256',
        'source_tps','target_tps','measurement','energy_j','duration_s','workload_envelope')}
    cost.update(audit_path=str(out/'audit.json'), audit_sha256=sha(out/'audit.json'),
                full_policy_cost_eligible=value['full_policy_cost_eligible'],
                component_only=not value['full_policy_cost_eligible'])
    save(out/'cost-components.json', cost)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare(args.attempt, args.out)
    print(json.dumps({key:value[key] for key in ('model_id','energy_j','duration_s','full_policy_cost_eligible')}))


if __name__ == '__main__':
    main()
