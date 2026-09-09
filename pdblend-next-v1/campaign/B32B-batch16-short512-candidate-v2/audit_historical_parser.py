"""Read-only parser regression on existing B D8 raw; never new D16 observations."""
import importlib.util
import json
from pathlib import Path
import sys

import late_context as lc

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def main():
    evidence = load('historical_original_b_evidence', ROOT.parent/'budget-profiling-v2-candidate/evidence.py')
    tp2 = load('historical_original_b_observation', ROOT/'observation.frozen.py')
    sources = {}; points = []
    def use(path):
        path = Path(path); sources[str(path)] = lc.sha(path); return path
    cases = [('B32B-batch-coverage-observation-v2-r4', range(6, 12), 512),
             ('B32B-batch-coverage-long4096-v1', range(6), 4096)]
    for name, indices, n in cases:
        campaign = ROOT.parent/name
        outer = lc.read(use(campaign/'status.json'))
        for index in indices:
            path = campaign/'results'/f'point-{index:04d}'
            raw = lc.read(use(path/'raw.json'))
            power = lc.read_power(use(path/'power.csv'))
            clocks = lc.read(use(path/'clocks.json'))
            metadata = lc.read(use(path/'power-metadata.json'))
            payload = use(path/'events.jsonl').read_bytes()
            events, refs = lc.parse_events(payload)
            original = evidence.derive(raw, power, metadata, clocks, events)
            tp2.validate_tp2_observation(raw, events)
            lc.require(raw['spec']['batch_size'] == 8 and raw['spec']['input_lengths'] == [n]*8
                       and raw['spec']['output_lengths'] == [256]*8, 'historical D8 work differs')
            # Explicit mechanical test under reviewed ordering. Original runs did
            # not capture all new default-source receipts, so this is not a full
            # new-protocol source-qualified logical-context observation.
            result, original_events = lc.reconstruct(raw, payload, source_order_verified=True)
            lc.attach_tp2_window(result['late_window'], power, clocks, raw['spec']['clock_command_mhz'])
            windows = lc.full_decode_windows(raw, original_events, power, clocks)
            warm = raw['warmup']
            lc.require(warm['success'] is True and len(warm['prompt_token_ids']) == 128
                       and len(warm['output_token_ids']) == len(warm['token_received_s']) == 64
                       and outer['measurement_start_s'] <= warm['dispatch_s'] < warm['stream_end_s']
                       <= raw['measurement_start_s'] < raw['measurement_end_s'] <= outer['measurement_end_s'],
                       'original warmup/primary not enclosed by original outer energy')
            points.append(dict(source=str(path), actual_batch=8, input_tokens=n, prescribed_output_tokens=256,
                frequency_mhz=raw['spec']['clock_command_mhz'], original_whole_batch_valid=original['valid'],
                mechanical_parser_passed=True, future_source_protocol_qualified=False,
                source_order_interpretation='conditional reconstruction under reviewed immutable ordering; historical default-source before/after receipts absent',
                full_original_event_count=len(events), empty_step_indices=[r['original_owner_event_index'] for r in result['empty_owner_steps']],
                complete_prefill_tokens=result['complete_prefill_token_sum'],
                per_id_decode_counts={rid: v['total_observed_decode_steps'] for rid, v in result['per_request'].items()},
                reconstructed_attention_after_min=min(v['attention_after_min'] for v in result['per_request'].values()),
                reconstructed_attention_after_max=max(v['attention_after_max'] for v in result['per_request'].values()),
                physical_kv_observed=False, full_decode_run_steps=[w['nonempty_decode_steps'] for w in windows],
                late_tp2_gpu01_energy_j=result['late_window']['target_gpus_energy_j'],
                late_both_target_clocks_valid=result['late_window']['actual_both_target_clocks_valid'],
                whole_batch_eight_gpu_energy_j=original['energy_all_eight_gpus_j'],
                warmup128_to64_complete=True, warmup_in_outer_energy=True, warmup_in_primary_energy=False,
                new_batch16_evidence=False, profile_point_generated=False))
    lc.require(all(lc.sha(path) == value for path, value in sources.items()), 'historical raw changed after audit')
    report = dict(schema=1, cpu_only=True, actual_new_gpu_runs=0, points=points, parser_regressions_passed=len(points),
                  source_sha256_before=sources, source_sha256_after_identical=True,
                  source_limitation='Current three-default-source read-only B receipt is not proof of historical before/after defaults',
                  new_batch16_data_available=False, performance_comparison_or_profile_generated=False,
                  actual_historical_order='2520 x3 then1500 x3 in each original D8 campaign; future D16 preserves1500 x3 then2520 x3')
    lc.write_new(ROOT/'historical-parser-evidence.json', report)
    print(json.dumps(dict(parser_regressions_passed=len(points), new_gpu_runs=0, historical_sources_unchanged=True)))


if __name__ == '__main__': main()
