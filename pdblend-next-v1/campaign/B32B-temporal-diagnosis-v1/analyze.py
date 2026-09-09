"""CPU-only audit of immutable diagnostic outputs and raw eight-GPU energy."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def diff(a, b):
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return dict(index_zero_based=i, output_position_one_based=i+1, reference_token=x, observed_token=y)
    return None


def integral(rows, start, end):
    assert rows[0][0] <= start < end <= rows[-1][0]
    total = [0.] * 8
    for (a, va), (b, vb) in zip(rows, rows[1:]):
        assert b > a and len(va) == len(vb) == 8
        lo, hi = max(a, start), min(b, end)
        if lo >= hi:
            continue
        for i in range(8):
            vlo = va[i] + (vb[i] - va[i]) * (lo - a) / (b - a)
            vhi = va[i] + (vb[i] - va[i]) * (hi - a) / (b - a)
            total[i] += (vlo + vhi) * .5 * (hi - lo)
    return total


def main():
    status = json.loads((ROOT / 'status.json').read_text())
    phases = status['phases']
    references = phases[0].get('outputs', [])
    comparisons = []
    all_steps_valid = True
    for phase in phases:
        raw = (ROOT / (phase['name'] + '.events.jsonl')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == phase['event_sha256']
        assert not raw or raw.endswith(b'\n')
        events = [json.loads(x) for x in raw.splitlines() if x]
        assert all(0 <= x['tokens'] <= 8192 for x in events)
        for request in phase.get('requests', []):
            rid = request['request_id']
            steps = [dict(output_position_one_based=j+1, **event) for j, event in enumerate(
                x for x in events if rid in x.get('request_ids', []) and x.get('tokens', 0))]
            tokens = request.get('response', {}).get('token_ids', [])
            slot = request['slot']
            difference = diff(references[slot], tokens) if len(references) > slot else None
            steps_valid = len(steps) == len(tokens) == 64
            all_steps_valid &= steps_valid
            pair_step = next((s['output_position_one_based'] for s in steps if len(s['request_ids']) > 1), None)
            difference_step = next((s for s in steps if difference and s['output_position_one_based'] == difference['output_position_one_based']), None)
            comparisons.append(dict(phase=phase['name'], slot=slot, prompt_tokens=len(request['body']['prompt']),
                request_id=rid, token_ids=tokens, first_difference=difference, exact_equal=difference is None,
                model_steps=len(steps), full_steps_and_output=steps_valid, first_shared_batch_output_position=pair_step,
                first_difference_owner_step=difference_step, owner_steps=steps))
        phase['audit_temporal_overlap_steps'] = sum(bool(e['prefill'] and e['decode']) for e in events) if phase['mode'] == 'temporal' else None
    pairwise = []
    for i, a in enumerate(comparisons):
        for b in comparisons[i+1:]:
            if a['slot'] == b['slot']:
                pairwise.append(dict(left=a['phase'], right=b['phase'], slot=a['slot'],
                    first_difference=diff(a['token_ids'], b['token_ids'])))
    with (ROOT / 'power/power.csv').open() as f:
        reader = csv.reader(f); header = next(reader)
        rows = [(float(r[0]), list(map(float, r[1:9]))) for r in reader]
    assert header[:9] == ['t_s'] + ['gpu' + str(i) + '_w' for i in range(8)]
    per_gpu = integral(rows, status['measurement_start_s'], status['measurement_end_s'])
    difference = sum(per_gpu) - status['total_node_energy_j']
    report = dict(complete=status.get('complete'), diagnostic_cases_complete=status.get('diagnostic_cases_complete'),
        request_count=len(comparisons), total_output_tokens=sum(len(x['token_ids']) for x in comparisons),
        exact_match_to_first_single_count=sum(x['exact_equal'] for x in comparisons),
        every_request_64_owner_steps=all_steps_valid, comparisons=comparisons, pairwise=pairwise,
        temporal_overlap_steps={p['name']: p['audit_temporal_overlap_steps'] for p in phases if p['mode'] == 'temporal'},
        per_gpu_energy_j=per_gpu, total_node_energy_j=sum(per_gpu), reported_difference_j=difference,
        duration_s=status['measurement_end_s']-status['measurement_start_s'],
        max_power_sample_gap_s=max(b[0]-a[0] for a,b in zip(rows,rows[1:])),
        measurement_valid=status.get('measurement_valid'), cleanup_complete=status.get('cleanup_complete'),
        phase_energy_j={p['name']: sum(integral(rows,p['started_s'],p['finished_s'])) for p in phases},
        limits=['Exact original equality gate retained; no tolerance added.',
            'Owner steps and transfer health do not prove KV content equality or close-logit numerical cause.',
            'Original validator did not preserve its output token sequences; this is a new bounded diagnostic.',
            'Explicit 8192/32 declaration inherited from original cleanup; all sampling/input keys unchanged.',
            'Energy is full eight-GPU observation cost, not a performance comparison.'])
    assert abs(difference) < 1e-6
    (ROOT / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('request_count','total_output_tokens','exact_match_to_first_single_count',
        'every_request_64_owner_steps','total_node_energy_j','reported_difference_j','duration_s','measurement_valid','cleanup_complete')}))
    for c in comparisons:
        print(c['phase'], c['slot'], c['first_difference'], 'first_batch2_position', c['first_shared_batch_output_position'])


if __name__ == '__main__':
    main()
