"""Active 45-point comparison, with one deployment decision per system/trace.

The historical topology catalogue stays available for mechanism audits. This
specification does not expand offline choices into duplicate evaluation runs.
Rate anchors are measured using calibration/tuning only; absent bindings remain
explicit blockers rather than falling back to historical 7B rates or profiles.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .campaign import DATASETS, SYSTEMS, PDBLEND_TP
from .client import load_split, poisson_trace
from ..seed_config import SEED_POLICY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError('immutable artifact differs: '+str(path))
    else:
        with path.open('x') as output: output.write(text)


def deployment(system, model):
    fixed = 2 if '32B' in model else 1
    candidates = [dict(tp=tp, pp=1) for tp in PDBLEND_TP[model]]
    if system in ('mixed', 'ecoserve'):
        return dict(mode='fixed_tp', tp=fixed, pp=1, gpu_budget=8,
            topology_source='explicit_fixed_baseline', dynamic_tp=False)
    if system == 'distserve':
        return dict(mode='offline_tp', candidates=[dict(prefill=v, decode=v) for v in candidates],
            gpu_budget=8, selection_split=['calibration','tuning'],
            scope='qualified_symmetric_TP_PP1_subset', complete_paper_reproduction=False,
            require_own_stage_profile=True, selected=None, online_reshard=False)
    if system == 'dynamollm':
        return dict(mode='native_control', initial_tp=fixed, pp=1, gpu_budget=8,
            candidates=candidates, require_qualified_control_space=True,
            control_periods_s=dict(ScaleInst=1800, ScaleShard=300, ScaleFreq=5),
            short_window_dynamic_tp_benefit_claim=False)
    if system == 'pdblend':
        return dict(mode='offline_tp', candidates=candidates, gpu_budget=8,
            selection_split=['calibration','tuning'], selected=None,
            own_profiles_only=True, pp=1)
    raise ValueError('unknown independent system')


def load_anchor(path, model, dataset, corpus_sha):
    path = Path(path); value = json.loads(path.read_text())
    if (value.get('status') != 'passed' or value.get('complete') is not True
            or value.get('model_id') != model or value.get('evaluation_used_for_selection') is not False
            or value.get('cleanup_errors') or value.get('selection_splits') != ['calibration','tuning']):
        raise ValueError('rate anchor lacks completed same-model calibration/tuning evidence')
    row = value['anchors'][dataset]
    if row['corpus_sha256'] != corpus_sha or row['base_rate_rps'] <= 0:
        raise ValueError('rate anchor belongs to a different corpus or has invalid rate')
    # Container paths are remapped explicitly, never silently searched by name.
    marker = '/output/anchor/'
    recorded = row['confirmation_path']
    if not recorded.startswith(marker):
        raise ValueError('unknown anchor output namespace')
    confirmation = path.parent/recorded[len(marker):]
    if sha(confirmation) != row['confirmation_sha256']:
        raise ValueError('rate confirmation checksum differs')
    measured = json.loads(confirmation.read_text())
    if measured['split'] != 'tuning' or measured['metrics']['passed'] is not True:
        raise ValueError('independent tuning confirmation did not pass')
    return dict(path=str(path.resolve()), sha256=sha(path), base_rate_rps=row['base_rate_rps'],
        scope=row['scope'], capacity_exact=False, confirmation_sha256=row['confirmation_sha256'])


def build(out: Path, corpus_root: Path, anchors=None):
    anchors = anchors or {}; points = []; traces = []
    for size in ('7b', '14b', '32b'):
        model = 'Qwen2.5-'+size.upper()+'-Instruct'
        corpus = corpus_root/f'2026-09-22-{size}-v1'
        manifest = json.loads((corpus/'manifest.json').read_text())
        if manifest['model_name'] != model or manifest.get('complete') is not True:
            raise ValueError('model-owned corpus missing: '+model)
        for dataset, slo in DATASETS.items():
            corpus_sha = sha(corpus/f'{dataset}.json')
            if corpus_sha != manifest['dataset_sha256'][dataset]:
                raise ValueError('corpus checksum mismatch')
            rate, anchor, trace_binding = None, None, None
            if size in anchors:
                anchor = load_anchor(anchors[size], model, dataset, corpus_sha)
                rate = .5*anchor['base_rate_rps']
                requests = poisson_trace(load_split(corpus, dataset, 'evaluation'), rate, 300., 701, source=dataset)
                trace = dict(schema='five-system-evaluation-trace-v1', model_id=model, dataset=dataset,
                    seed=701, single_seed=True, seed_policy=SEED_POLICY, duration_s=300., rate_rps=rate,
                    corpus_sha256=corpus_sha, corpus_manifest_sha256=sha(corpus/'manifest.json'),
                    corpus_tokenizer_sha256=manifest['tokenizer_sha256'],
                    output_workload='same model-tokenized capped reference length for all systems',
                    selection_split='evaluation', slo=slo, anchor=anchor,
                    requests=[asdict(r) for r in requests])
                path = out/'traces'/f'{size}-{dataset}-x0.5-seed701.json'
                write_new(path, trace)
                trace_binding = dict(path=str(path.resolve()), sha256=sha(path), requests=len(requests))
                traces.append(trace_binding)
            for system in SYSTEMS:
                topology = deployment(system, model)
                blockers = ['native_execution_preflight','mechanism_receipts','power_provenance']
                if not anchor: blockers.insert(0, 'missing_rate_anchor')
                if system != 'mixed': blockers.append('independent_profile_coverage_and_calibration')
                if system in ('distserve','pdblend'): blockers.append('offline_choice_not_deployed')
                points.append(dict(name=f'{size}-{system}-{dataset}-x0.5-seed701', model_id=model,
                    system=system, dataset=dataset, scale=.5, seed=701, single_seed=True,
                    seed_policy=SEED_POLICY, duration_s=300., topology=topology,
                    trace=trace_binding, rate_rps=rate, slo=slo, status='waiting_cpu_preparation',
                    blockers=blockers, formal_eligible=False, runner='independent_native_dispatch',
                    gpu_count=8, exclusive=True, reserve_host=True))
    value = dict(schema='first-five-system-batch-v1', priority='3_models_3_datasets_x05_5_systems',
        seeds=[701], seed_policy=SEED_POLICY, single_seed=True, duration_s=300., points=points,
        traces=traces, summary=dict(points=len(points), trace_sets=9, frozen_trace_sets=len(traces),
            pure_service_window_s=45*300, ready_points=0),
        identity_protocol='common_metering_runtime_and_workload_plus_independent_system_source_profile',
        long_mechanism=dict(dynamollm_duration_s=2100, separate_from_short_comparison=True),
        later_scales=[.25,.75,1.], later_pdblend_modes=['fixed_tp','offline_tp','resident_hetero_tp','slow_reshard_tp'])
    write_new(out/'spec.json', value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--corpus-root', type=Path, default=Path('datasets/prepared'))
    p.add_argument('--anchor', action='append', default=[], help='7b=/host/path/completion.json')
    a = p.parse_args()
    anchors = dict(item.split('=',1) for item in a.anchor)
    value = build(a.out.resolve(), a.corpus_root.resolve(), anchors)
    print(json.dumps(value['summary']))


if __name__ == '__main__': main()
