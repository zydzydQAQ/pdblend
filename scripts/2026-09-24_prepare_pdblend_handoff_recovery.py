#!/usr/bin/env python3
"""Prepare independent endpoint handoff recovery; raw CPU replay only, never enqueue."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from pdblend.profile.collection.native_handoff import collection_plan, extract_runtime
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_timing_plan import binding, read_bound


def prepare(out, inventory, ledger):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('new immutable handoff preparation directory required')
    attempts = read_bound(binding(inventory))
    ledger_ref = binding(ledger)
    plans = {size: collection_plan(ledger_ref, f'Qwen2.5-{size.upper()}-Instruct')
             for size in ('7b', '14b')}
    out.mkdir(parents=True)
    plan_refs = {size: write_new(out/(size+'-plan.json'), plan) for size, plan in plans.items()}
    reviews = []
    for attempt in attempts:
        if not attempt.get('runtime'):
            continue
        reference = attempt['runtime']['completion']
        inputs = read_bound(attempt['input_manifest'])
        source_ref = inputs['source_manifest']
        row = dict(job_id=attempt['job_id'], model_id=attempt['model_id'],
                   completion=reference, source_manifest=source_ref)
        try:
            result = extract_runtime(reference, source_ref)
            row.update(extracted=True, extraction=write_new(out/'extracted'/
                (attempt['job_id']+'.json'), result), nodes=result['candidate']['nodes'],
                holdout_comparisons=result['holdout_comparisons'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row.update(extracted=False, error=str(exc), component_qualified=False)
        reviews.append(row)
    result = dict(schema='pdblend-endpoint-handoff-recovery-preparation/v1',
        prior_inventory=binding(inventory), plans=plan_refs, raw_replays=reviews,
        implementation=binding(ROOT/'src/pdblend/profile/collection/native_handoff.py'),
        builder=binding(__file__), hardware_executed=False, enqueued=False,
        evaluation_used_for_selection=False, online_threshold_changed=False,
        physical_copy_time=False, full_profile_qualified=False, component_qualified=False,
        ready_cpu_entrypoints=[
            'python -m pdblend.profile.collection.native_handoff extract-runtime --completion FILE --source-manifest FILE --out NEW.json',
            'python -m pdblend.profile.collection.native_handoff prepare --ledger FILE --model-id MODEL --out NEW.json'],
        ready_resident_pair_entrypoint='pdblend.profile.collection.native_handoff.collect_endpoint_pair',
        next_gpu_integration='Use the bound per-model exact-node plan with the existing native resident fleet; '
            'persist each pair, complete all training then freeze candidate before any holdout; '
            'attach physical clock/native epoch/drain/terminal cleanup receipts. This manifest is not a GPU job.',
        scope_32b='Current non-evaluation canonical query ledger selects all-M and contains no PD transfer query; '
                  'legacy 32B endpoint data remains diagnostic, no new PD domain invented.',
        device_copy_observer_missing='Current public client/proxy fields contain no CUDA send/receive/install '
            'events; implementing same-device per-rank connector events is required to claim device copy time.')
    write_new(out/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, default=ROOT/
        'results/2026-09-24/pdblend-profile-recovery-v4/attempt-evidence.json')
    parser.add_argument('--ledger', type=Path, default=ROOT/
        'results/2026-09-24/pdblend-offline-readiness/all-models-all-scales-confirmed32-v2/query-ledger.json')
    args = parser.parse_args()
    result = prepare(args.out, args.inventory, args.ledger)
    print('Prepared', args.out, 'raw replay successes:', sum(r['extracted'] for r in result['raw_replays']))


if __name__ == '__main__':
    main()
