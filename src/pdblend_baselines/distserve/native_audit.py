"""Bind independent native stage receipts to the original DistServe simulator.

The synthetic fixed-shape history is a CPU integration diagnostic. A partial
PP1 table never becomes a complete offline deployment choice or GPU evidence.
No author scheduling/simulation source is changed by this adapter.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from ..native_profile import DIST_FREQS, audit as audit_raw
from .planning import best_config, binary_goodput, capability_matrix, enumerate_configs
from .simulator import MeasuredLatency, OfficialSimulator

MODELS = {
    'Qwen2.5-7B-Instruct': (28, 28, (1, 2, 4)),
    'Qwen2.5-14B-Instruct': (48, 40, (1, 2, 4, 8)),
    'Qwen2.5-32B-Instruct': (64, 40, (1, 2, 4, 8)),
}
IDENTITY_FIELDS = ('model_hash', 'tokenizer_hash', 'engine_version', 'image_digest', 'source_revision')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_profile(path, model):
    value = json.loads(Path(path).read_text())
    if value.get('system') != 'distserve' or value.get('model') != model:
        raise ValueError('independent DistServe system/model identity differs')
    meta = value.get('metadata', {})
    if meta.get('engine_version') != 'vllm-0.10.1.1':
        raise ValueError('new-stack engine identity required')
    for name in ('model_hash', 'tokenizer_hash', 'source_revision'):
        if not re.fullmatch('[0-9a-f]{64}', meta.get(name, '')):
            raise ValueError('complete SHA256 provenance required: '+name)
    if not re.fullmatch('sha256:[0-9a-f]{64}', meta.get('image_digest', '')):
        raise ValueError('immutable image digest required')
    tp, pp = meta.get('tp'), meta.get('pp')
    if type(tp) is not int or tp not in MODELS[model][2] or pp != 1:
        raise ValueError('unsupported_engine: native rank events qualify only legal TP / PP1')
    uuids = meta.get('gpu_uuids', [])
    if len(uuids) != tp or len(set(uuids)) != tp or any(not str(u).startswith('GPU-') for u in uuids):
        raise ValueError('profile GPU UUID group differs from TP')
    for row in value.get('rows', []):
        if type(row.get('frequency_mhz')) is not int or row['frequency_mhz'] not in DIST_FREQS:
            raise ValueError('native profile frequency differs from six-bin protocol')
        for rank in row.get('sample', {}).get('ranks', []):
            if rank.get('tp') != tp or rank.get('pp') != 1:
                raise ValueError('native rank topology differs from metadata')
            for sample in rank.get('samples', []):
                if (sample.get('system') != 'distserve' or sample.get('tp') != tp or sample.get('pp') != 1
                        or sample.get('rank') != rank.get('rank') or sample.get('pp_rank') != 0
                        or sample.get('tp_rank') != rank.get('rank')):
                    raise ValueError('raw sample system/rank/topology identity differs')
    audit_raw(path)  # Reconstruct points from all-rank events and check their SHA.
    return value


def capability_rows(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and 'capabilities' in value:
        value = value['capabilities']
    if isinstance(value, dict) and 'state' in value:
        value = [value]
    if not isinstance(value, list):
        raise ValueError('native evidence requires capability or probe capabilities list')
    return [dict(row, evidence_path=str(Path(path).resolve()), evidence_sha256=file_sha(path)) for row in value]


def capacity_receipt(profile, capabilities):
    meta = profile['metadata']
    matches = [row for row in capabilities if row.get('model_id') == profile['model']
        and row.get('tp') == meta['tp'] and row.get('pp') == 1
        and row.get('gpu_uuids') == meta['gpu_uuids']
        and all(row.get(key) == meta[key] for key in IDENTITY_FIELDS)]
    if not matches:
        return None
    values = set()
    for row in matches:
        state = row.get('state', {})
        capacity = state.get('total_kv_tokens')
        if (row.get('supported') is not True or row.get('native_evidence_complete') is not True
                or type(capacity) is not int or capacity <= 0
                or state.get('tp') != meta['tp'] or state.get('pp') != 1):
            raise ValueError('native capacity evidence incomplete')
        values.add(capacity)
    if len(values) != 1:
        raise ValueError('native capacity changed between matching receipts')
    return dict(tp=meta['tp'], pp=1, capacity_tokens=values.pop(),
                evidence_path=matches[0]['evidence_path'], evidence_sha256=matches[0]['evidence_sha256'])


class ExactNativeLatency(MeasuredLatency):
    """Original measured provider, restricted to observed batch/context cells."""
    def stage_latency(self, role, tp, pp, stage, batch, inputs, contexts):
        if pp != 1 or stage != 0:
            raise ValueError('unsupported_engine: PP wire/host service is unmeasured')
        # Upstream Request.current_context_len starts at prompt length before
        # its first decode. Native CUDA shape includes the current input token.
        native_contexts = tuple(c+1 for c in contexts) if role == 'decode' else contexts
        context = max(native_contexts, default=0)
        maximum = max(inputs, default=0)
        matching = [point for point in self.points
            if (point['role'], point['tp'], point['pp'], point['stage_index'], point['batch'], point['max_context_tokens'])
               == (role, tp, pp, stage, batch, context)
            and (role != 'prefill' or point['max_input_tokens'] == maximum)]
        # Native collector uses homogeneous requests in each batch. A mixed
        # prefill-length batch cannot inherit the maximum-length timing alone.
        if role == 'prefill' and len(set(inputs)) != 1:
            matching = []
        if role == 'decode' and len(set(native_contexts)) != 1:
            matching = []
        if not matching:
            raise ValueError(f'missing_profile: exact {role} TP{tp}PP{pp} batch={batch} context={context}')
        return MeasuredLatency(matching).stage_latency(role, tp, pp, stage, batch, inputs, native_contexts)


def run(paths, evidence_paths, *, model, input_tokens=128, output_tokens=16, requests=8,
        gpu_budget=8, rate=1., ttft=5., tpot=.15, maximum_rate=1., epsilon=.25):
    if model not in MODELS:
        raise ValueError('explicit supported model required')
    if (type(gpu_budget) is not int or not 2 <= gpu_budget <= 8 or
            any(type(n) is not int or n < 1 for n in (input_tokens, requests))
            or type(output_tokens) is not int or not 2 <= output_tokens <= 512
            or input_tokens+output_tokens > 8192 or not paths):
        raise ValueError('bounded PP1 history / eight-GPU budget required')
    if any(not math.isfinite(v) or v <= 0 for v in (rate, ttft, tpot, maximum_rate, epsilon)):
        raise ValueError('finite positive rate, SLO and search limits required')
    if epsilon >= maximum_rate:
        raise ValueError('search epsilon must permit at least one real SimPy trial')
    profiles = [read_profile(path, model) for path in paths]
    reference = profiles[0]['metadata']
    if any(any(p['metadata'][key] != reference[key] for key in IDENTITY_FIELDS) for p in profiles):
        raise ValueError('cross-profile model/tokenizer/engine/image/source provenance differs')
    if len({p['metadata']['tp'] for p in profiles}) != len(profiles):
        raise ValueError('one immutable independent profile per TP required')
    capabilities = [row for path in evidence_paths for row in capability_rows(path)]
    receipts = [r for p in profiles if (r := capacity_receipt(p, capabilities)) is not None]
    capacities = {(r['tp'], 1): r['capacity_tokens'] for r in receipts}
    layers, heads, tps = MODELS[model]
    configs = enumerate_configs(layers=layers, attention_heads=heads, allowed_tps=tps,
                                num_nodes=1, gpus_per_node=gpu_budget)
    supported = {(tp, 1) for tp in tps if not (model.endswith('32B-Instruct') and tp == 1)}
    history = [(input_tokens, output_tokens)]*requests
    frequencies = []
    for frequency in DIST_FREQS:
        points = [point for p in profiles for row in p['rows'] if row['frequency_mhz'] == frequency for point in row['points']]
        roles = {(p['tp'], p['pp'], p['role']) for p in points}
        measured = {pair for pair in capacities if all((*pair, role) in roles for role in ('prefill', 'decode'))}
        matrix = capability_matrix(configs, supported_pairs=supported, measured_pairs=measured)
        latency = ExactNativeLatency(points)
        simulator = OfficialSimulator(history, latency=latency, capacities=capacities, seed=701, max_events=100000)
        searches = []
        for row in matrix:
            _, tp, pp, td, pd = row['config']
            if pp != 1 or pd != 1:
                row['reason'] = 'PP stage wire/host timing and native P/D correctness unsupported'
            elif tp != td:
                row.update(status='unsupported_engine', reason='asymmetric TP KV remap is unqualified')
            elif (tp, 1) not in supported:
                row['reason'] = 'TP1PP1 32B BF16 weights exceed one L20 memory budget'
            elif row['status'] == 'missing_profile':
                row['reason'] = 'missing own stage samples or identity-bound native capacity receipt'
            if row['status'] != 'covered':
                continue
            observations = []
            def observed(config, offered_rate):
                outcome = simulator(config, offered_rate)
                observations.append(dict(rate_rps=offered_rate, **outcome))
                return outcome
            result = binary_goodput(row['config'], observed, ttft_s=ttft, tpot_s=tpot,
                                    max_per_gpu_rate=maximum_rate, epsilon=epsilon)
            result['native_simpy_observations'] = observations
            if result['status'] != 'complete':
                row.update(status='missing_profile' if 'missing_profile:' in result.get('error', '') else 'simulation_failed',
                           reason=result.get('error'))
            searches.append(result)
        chosen, goodput = best_config({tuple(s['config']): s['best_per_gpu_rate'] for s in searches})
        frequencies.append(dict(frequency_mhz=frequency, matrix=matrix, searches=searches,
            local_diagnostic_candidate=None if chosen is None else dict(config=chosen, per_gpu_goodput=goodput),
            status_counts=dict(Counter(row['status'] for row in matrix))))
    any_success = any(s['status'] == 'complete' for f in frequencies for s in f['searches'])
    return dict(schema='distserve-native-search-audit-v1', system='distserve', model_id=model,
        status='diagnostic_complete' if any_success else 'inconclusive', audit_valid=True,
        formal_eligible=False, energy_comparable=False, gpu_qualified=False,
        complete_offline_choice=False, complete_search_space=False, replica_allocation_performed=False,
        gpu_budget=gpu_budget, seed=701,
        history=dict(kind='synthetic_fixed_shape_functional_probe', shapes=history,
                     campaign_trace_used=False, target_rate_rps=rate),
        slo=dict(ttft_s=ttft, tpot_s=tpot, predicate='original separate strict 90th percentiles'),
        provenance=reference, profiles=[dict(path=str(Path(path).resolve()), sha256=file_sha(path)) for path in paths],
        native_capacity_receipts=receipts, frequencies=frequencies,
        limitations=['single-seed CPU mechanism evidence; no GPU deployment decision',
            'native profiles lack independent holdout/power/interference qualification',
            'frequency identity is declared by the native collector sweep; raw clock samples are absent',
            'PP and asymmetric TP transfer remain unsupported',
            'only exact observed homogeneous batch/context cells; no interpolation or extrapolation',
            'upstream decode context budget=50000 and 2ms+0.0001ms/token Ray overhead retained',
            'simulator request flow is not physical KV transfer or cancellation evidence'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, action='append', required=True)
    parser.add_argument('--native-evidence', type=Path, action='append', default=[])
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=16)
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--gpu-budget', type=int, default=8)
    parser.add_argument('--rate', type=float, default=1.)
    parser.add_argument('--ttft', type=float, default=5.)
    parser.add_argument('--tpot', type=float, default=.15)
    parser.add_argument('--maximum-rate', type=float, default=1.)
    parser.add_argument('--epsilon', type=float, default=.25)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.profile, args.native_evidence, model=args.model, input_tokens=args.input_tokens,
        output_tokens=args.output_tokens, requests=args.requests, gpu_budget=args.gpu_budget,
        rate=args.rate, ttft=args.ttft, tpot=args.tpot, maximum_rate=args.maximum_rate, epsilon=args.epsilon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'audit_valid', 'complete_offline_choice', 'formal_eligible')}))
    return 0 if result['status'] == 'diagnostic_complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
