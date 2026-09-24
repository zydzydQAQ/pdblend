#!/usr/bin/env python3
"""Prepare only immutable CPU request-cycle plans; never enqueue or launch GPUs."""
import argparse
from pathlib import Path

from pdblend.profile.collection.native_serving_cycles import build_cycle_plan,cycle_trace
from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_runtime_collect import write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--model',choices=['Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct','Qwen2.5-32B-Instruct'],required=True)
    parser.add_argument('--dataset',choices=['alpaca','sharegpt','longbench'],default='alpaca')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('use a new immutable cycle-plan directory')
    plan=build_cycle_plan(binding(args.ledger),binding(args.provenance),model_id=args.model,dataset=args.dataset)
    preview=[]
    for index,point in enumerate(plan['points']):
        trace=cycle_trace(plan,point);requests=trace['requests']
        preview.append(dict(index=index,purpose=point['purpose'],frequency_mhz=point['frequency_mhz'],
            arrival_family=point['arrival_family'],target_rate_rps=point['target_rate_rps'],
            actual_rate_rps=trace['rate_rps'],requests=len(requests),
            real_prompt_range=[min(len(r['prompt']) for r in requests),max(len(r['prompt']) for r in requests)],
            real_output_range=[min(r['max_tokens'] for r in requests),max(r['max_tokens'] for r in requests)]))
    plan_ref=write_new(args.out/'plan.json',plan)
    root=Path(__file__).resolve().parents[1]
    files=['src/pdblend/profile/collection/native_serving_cycles.py',
        'src/pdblend/profile/collection/native_serving_cycles_audit.py',
        'src/pdblend/profile/query/native_cycle_model.py',
        'src/pdblend/profile/query/native_query_replay.py',__file__]
    review=dict(schema='pdblend-native-request-cycle-preparation/v1',plan=plan_ref,model_id=args.model,
        preview=preview,implementation={str(name):binding(root/name) for name in files},
        cpu_only=True,hardware_executed=False,formal_eligible=False,full_profile_qualified=False,
        same_fleet=True,engine_loads_by_this_collector=0,initial_training_windows=8,
        training_service_seconds=480,holdout_deferred_until_training_only_candidate_frozen=True,
        no_active_kernel_power_claim=True,no_evaluation_used=True,
        next_action='root_may_integrate_training_after_runtime_powerpilot_and_timing_on_the_same_fleet',
        remaining_gates=plan['remaining_gates'])
    write_new(args.out/'review.json',review)
    print(plan_ref['path'])


if __name__=='__main__':main()
