#!/usr/bin/env python3
"""Prepare a bounded pure-power pilot and a raw-backed gap review; never enqueue."""
import argparse
import json
from pathlib import Path

from pdblend.bench.resident_session import write_new
from pdblend.profile.collection.native_power_plan import binding,read_bound,build_plan,short_context_limit

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,default=ROOT/'results/2026-09-24/pdblend-offline-readiness/all-models-all-scales-v1/query-ledger.json')
    parser.add_argument('--provenance',type=Path,default=ROOT/'results/2026-09-24/pdblend-offline-readiness/all-models-all-scales-v1/bindings.json')
    args=parser.parse_args();plan=build_plan(binding(args.ledger),binding(args.provenance))
    manifest=ROOT/'results/2026-09-22/three-model/calibration-candidates/7b-tp1-4ddc49563cef6321c0b5/manifest.json'
    prior=read_bound(binding(manifest));raw_path=Path(prior['training_raw'])
    raw=read_bound(dict(path=str(raw_path),sha256=prior['training_raw_sha256']));windows=[]
    for freq in (1500,2520):
        row=min((d for d in raw['decode'] if d['freq_mhz']==freq and d['batch']==1),key=lambda d:d['context_tokens'])
        for rep in row['repeats']:
            ref=dict(path=str(raw_path.parent/rep['samples_file']),sha256=rep['samples_sha256'])
            sample=read_bound(ref)
            observed=row['context_tokens']+sum(a+b for a,b in zip(sample['start_token_counts'],sample['end_token_counts']))/(2*len(sample['start_token_counts']))
            if observed!=rep['effective_context_tokens']:raise ValueError('historical actual context differs')
            windows.append(dict(frequency_mhz=freq,prompt_tokens=row['context_tokens'],effective_context_tokens=observed,
                step_seconds=rep['step_seconds'],window_s=rep['end_s']-rep['start_s'],
                start_token_counts=sample['start_token_counts'],end_token_counts=sample['end_token_counts'],raw_sample=ref))
    short=min(q['shape']['context_tokens'] for q in plan['actual_queries'] if q['shape']['role']=='decode')
    low=[q for q in plan['actual_queries'] if 'decode_power_w' in q['methods'] and 1<q['shape']['batch']<4]
    ledger=read_bound(plan['query_ledger'])
    files=['src/pdblend/profile/collection/native_power_'+part+'.py' for part in ('plan','collect','audit')]
    files+=['src/pdblend_runtime/native_v1.py','src/pdblend_runtime/serve.py','src/pdblend/profile/query/power_table.py',
        'src/pdblend/profile/query/native_timing.py','src/pdblend/profile/query/composition.py','src/pdblend/planner/pool.py']
    review=dict(schema='pdblend-native-power-readiness/v1',hardware_executed=False,evaluation_used=False,
        formal_eligible=False,full_profile_qualified=False,planner_policy_modified=False,parameters_fitted=False,
        query_ledger=plan['query_ledger'],query_provenance=plan['query_provenance'],
        historical_training_manifest=binding(manifest),historical_training_raw=binding(raw_path),historical_windows=windows,
        short_context=short_context_limit(short),historical_evidence_is_native_equivalence_proof=False,
        raw_ledger_outcomes=[{k:r.get(k) for k in ('model_id','dataset','rate_scale','target_rate_rps','status',
            'feasible_development_candidates','error','unsupported_queries')} for r in ledger['ledgers']],
        actual_low_fractional_power_queries=low,
        obstacles=[dict(gate='current_tuning_coverage',reason='Timing rejects finite short/long queries; further power queries remain partly hidden by short-circuiting. All recorded development candidate counts are zero; this is not an optimality result.'),
            dict(gate='continuous_short_context_power',reason='Historical B1 windows advance mean context by231/236 tokens. A native pilot verifies its own lower bound; nominal prompt cannot qualify121. Historical speed alone is not proof of native impossibility.'),
            dict(gate='low_fractional_batch',reason='The existing power table isolates B1 from B>=4; its low-batch extension accepts B2/B3 only exactly. Observed power queries require1<B<4, demanding a new validated composition.'),
            dict(gate='fixed_shape_forward_interface',reason='The forward measurement scope installs observation hooks; start requires a drained scheduler and samples reads CUDA events. No supported RPC repeatedly executes retained true-KV shapes without advancing state.'),
            dict(gate='prefill_energy_semantics',reason='Prefill request-cycle means include dispatch/idle gaps; multiplying by native CUDA latency is not qualified kernel energy.'),
            dict(gate='formal_composition',reason='Native timing is development-only. Source compatibility, independent holdout, runtime components, mixed energy composition and complete tuning coverage are separate gates.')],
        implementation={p:binding(ROOT/p) for p in files})
    args.out.mkdir(parents=True,exist_ok=False)
    write_new(args.out/'pilot-plan.json',plan);review['plan']=binding(args.out/'pilot-plan.json')
    write_new(args.out/'review.json',review)
    print(json.dumps(dict(points=len(plan['points']),windows=sum(p['repeats'] for p in plan['points']),
        actual_query_shapes=len(plan['actual_queries']),fractional_low_batch_queries=len(low),out=str(args.out.resolve()))))


if __name__=='__main__':main()
