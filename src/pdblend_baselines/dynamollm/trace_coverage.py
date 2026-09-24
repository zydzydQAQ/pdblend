"""Bind frozen comparison traces to own BERT predictions and measured geometry.

This audit reads evaluation work only to check coverage. It neither fits a
predictor nor chooses a deployment from evaluation latency/energy. Proposed
measurements are separate, unqualified geometry; they never become estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path

from .coverage_inventory import FREQUENCIES, cached_rows
from .deployment import save, sha
from .policy import PERIODS, classify
from .portable_profile import validate_profile
from .prediction_cache import prompt_sha
from .run_v1 import load_trace
from .validation import COMPARISON_DURATIONS, MODELS


def bind_trace(trace_path, *, cache, corpus):
    """Match prompt bytes and work limits, never infer corpus indices by order."""
    trace_path, cache, corpus = map(Path, (trace_path, cache, corpus))
    trace = json.loads(trace_path.read_text())
    if (trace.get('schema') != 'five-system-evaluation-trace-v1'
            or trace.get('selection_split') != 'evaluation'
            or trace.get('duration_s') not in COMPARISON_DURATIONS or trace.get('seed') != 701):
        raise ValueError('frozen seed701 150/300-second comparison trace required')
    load_trace(trace_path, trace['duration_s'])  # Validate monotonic arrivals and fixed work.
    if len({row.get('idx') for row in trace['requests']}) != len(trace['requests']):
        raise ValueError('unique frozen trace request indices required')
    dataset = trace['dataset']
    binding, cached = cached_rows(cache, corpus, dataset, 'evaluation')
    manifest = json.loads((corpus/'manifest.json').read_text())
    if (binding['model_identity']['model'] != trace['model_id']
            or trace['corpus_manifest_sha256'] != sha(corpus/'manifest.json')
            or trace['corpus_sha256'] != sha(corpus/(dataset+'.json'))
            or trace['corpus_tokenizer_sha256'] != manifest['tokenizer_sha256']):
        raise ValueError('trace model/corpus/tokenizer identity differs from cache')
    predictor = Path(binding['predictor'])/'manifest.json'
    if sha(predictor) != binding['predictor_manifest_sha256']:
        raise ValueError('cached predictor manifest changed')
    matches = defaultdict(list)
    for row in cached:
        matches[(row['prompt_tokens_sha256'], row['actual_output_limit'])].append(row)
    rows = []
    for index, request in enumerate(trace['requests']):
        key = prompt_sha(request['prompt']), request['max_tokens']
        candidates = matches.get(key, [])
        if (not candidates or len({r['predicted_output'] for r in candidates}) != 1
                or any(r['input_tokens'] != len(request['prompt']) for r in candidates)):
            raise ValueError('trace request lacks unambiguous own cached prediction: '+str(index))
        row = candidates[0]
        rows.append(dict(request_index=index, trace_idx=request['idx'],
            arrival_s=request['arrival_s'], input_tokens=len(request['prompt']),
            prompt_tokens_sha256=key[0], corpus_indices=[r['request_index'] for r in candidates],
            predicted_output=row['predicted_output'], actual_output_limit=request['max_tokens'],
            output_limit_used_for_prediction=False,
            shape=classify(len(request['prompt']), row['predicted_output'])))
    return dict(schema='dynamo-frozen-trace-predictions-v1', system='dynamollm',
        model_id=trace['model_id'], dataset=dataset, seed=701, duration_s=trace['duration_s'],
        rate_rps=trace['rate_rps'], slo=trace['slo'], trace_path=str(trace_path.resolve()),
        trace_sha256=sha(trace_path), prediction_cache=str(cache.resolve()),
        cache_completion_sha256=sha(cache/'completion.json'),
        cache_binding_sha256=sha(cache/'binding.json'),
        predictor_manifest_sha256=binding['predictor_manifest_sha256'],
        corpus_manifest_sha256=trace['corpus_manifest_sha256'],
        predictions=rows, predictor_fitted_here=False, evaluation_outputs_used_for_prediction=False,
        prediction_cache_used_for_runtime=False, formal_eligible=False)


def supports(cells, target):
    """Geometry-only equivalent of PaperProfiles' complete rectangle rule."""
    if not cells:
        return False
    axes = [sorted({shape[axis] for shape in cells}) for axis in range(3)]
    alternatives = []
    for values, wanted in zip(axes, target):
        pairs = [(lo,) if lo == hi else (lo, hi)
                 for lo in values if lo <= wanted for hi in values if hi >= wanted]
        if not pairs:
            return False
        alternatives.append(pairs)
    return any(all(corner in cells for corner in product(*bounds))
               for bounds in product(*alternatives))


def geometry(points):
    result = defaultdict(set)
    for point in points:
        result[(point['tp'], point['frequency_mhz'])].add((point['input_tokens'],
            point.get('output_tokens', point.get('context_tokens', 0)-point['input_tokens']),
            point['batch']))
    return result


def audit_predictions(bound, points, *, tps, max_batch=16):
    cells = geometry(points)
    requests = bound['predictions']
    shapes = Counter((r['input_tokens'], r['predicted_output'],
                      max(r['predicted_output'], r['actual_output_limit'])) for r in requests)
    memo = {}
    def covered(tp, frequency, n, o, batch):
        key = tp, frequency, n, o, batch
        if key not in memo:
            memo[key] = supports(cells[(tp, frequency)], (n, o, batch))
        return memo[key]
    topologies = []
    for tp in tps:
        batches = []
        for batch in range(1, max_batch+1):
            by_frequency = []
            for frequency in FREQUENCIES:
                predicted = sum(count for (n, o, end), count in shapes.items()
                                if covered(tp, frequency, n, o, batch))
                envelope = sum(count for (n, o, end), count in shapes.items()
                               if covered(tp, frequency, n, o, batch)
                               and covered(tp, frequency, n, end, batch))
                by_frequency.append(dict(frequency_mhz=frequency,
                    predicted_requests_supported=predicted, work_envelope_requests_supported=envelope,
                    all_requests_supported=envelope == len(requests)))
            batches.append(dict(batch=batch, frequencies=by_frequency,
                all_requests_supported=all(r['all_requests_supported'] for r in by_frequency)))
        topologies.append(dict(tp=tp, pp=1, batches=batches,
            fully_supported_batches=[b['batch'] for b in batches if b['all_requests_supported']]))
    arrivals = [r['arrival_s'] for r in requests]
    def peak_arrivals(window):
        left = maximum = 0
        for right, at in enumerate(arrivals):
            while arrivals[left] <= at-window:
                left += 1
            maximum = max(maximum, right-left+1)
        return maximum
    return dict(dataset=bound['dataset'], trace_sha256=bound['trace_sha256'], requests=len(requests),
        unique_query_shapes=len(shapes), input_bounds=[min(r['input_tokens'] for r in requests),
                                                     max(r['input_tokens'] for r in requests)],
        predicted_output_counts=dict(Counter(r['predicted_output'] for r in requests)),
        fixed_work_output_bounds=[min(r['actual_output_limit'] for r in requests),
                                  max(r['actual_output_limit'] for r in requests)],
        shape_counts=dict(Counter(r['shape'] for r in requests)),
        arrival_bursts={str(window):peak_arrivals(window) for window in (1.,5.,10.)},
        arrival_bursts_are_not_concurrency=True, native_batch_observed=False,
        configured_admission_batch_limit=max_batch, topologies=topologies,
        mixed_batch_scope='Necessary per-request boundary coverage; heterogeneous-batch holdout is separate',
        interpolation_accuracy_qualified=False, formal_eligible=False)


def load_measured(sources, model_id):
    points, receipts = [], []
    for source in sources:
        value = validate_profile(profile=source['path'], artifact_root=source['artifact_root'],
            recorded_root=source.get('recorded_root','/output'))
        if value['model_id'] != model_id:
            raise ValueError('own profile model differs from trace')
        points.extend(value['points'])
        receipts.append(value['portable_profile_receipt'])
    return points, receipts


def prepare(*, index, cache, corpus, source_index, pending_plan, out, max_batch=16):
    index, cache, corpus, source_index, pending_plan, out = map(Path,
        (index, cache, corpus, source_index, pending_plan, out))
    if out.exists():
        raise FileExistsError('immutable Dynamo coverage output already exists')
    trace_index = json.loads(index.read_text())
    if trace_index.get('kind') != 'frozen_evaluation_trace_index' or trace_index.get('seed') != 701:
        raise ValueError('frozen shared trace index required')
    bound_traces = []
    for row in trace_index['traces']:
        if (sha(row['path']) != row['sha256'] or row.get('split') != 'evaluation'
                or row.get('seed') != 701 or row.get('duration_s') not in COMPARISON_DURATIONS):
            raise ValueError('shared trace checksum differs')
        bound = bind_trace(row['path'], cache=cache, corpus=corpus)
        if (bound['model_id'] != row['model_id'] or bound['dataset'] != row['dataset']
                or len(bound['predictions']) != row['requests'] or bound['duration_s'] != row['duration_s']):
            raise ValueError('trace index model/dataset differs')
        bound_traces.append(bound)
    models = {b['model_id'] for b in bound_traces}
    durations = {b['duration_s'] for b in bound_traces}
    if len(models) != 1 or len(durations) != 1:
        raise ValueError('one model and one duration per independent coverage package required')
    duration = next(iter(durations))
    model_id = next(iter(models))
    source_value = json.loads(source_index.read_text())
    points, receipts = load_measured(source_value['profiles'], model_id)
    plan = json.loads(pending_plan.read_text())
    if (plan.get('system') != 'dynamollm' or plan.get('model_id') != model_id
            or plan.get('schema') != 'dynamo-missing-profile-cells-v1'
            or plan.get('fit_from_holdout') is not False):
        raise ValueError('independent same-model pending plan required')
    pending = [dict(p, tp=plan['tp']) for p in plan['points']]
    measured_keys = {(p['tp'],p['frequency_mhz'],p['input_tokens'],
                      p['context_tokens']-p['input_tokens'],p['batch']) for p in points}
    # Reuse the pre-existing frozen envelope. Evaluation labels cannot enlarge
    # this candidate plan or turn its prospective corners into fitted values.
    extension = [dict(p, batch=max_batch) for p in plan['points']
                 if (plan['tp'],p['frequency_mhz'],p['input_tokens'],p['output_tokens'],max_batch)
                 not in measured_keys]
    out.mkdir(parents=True)
    measured_audits, prospective_audits = [], []
    for bound in bound_traces:
        dataset = bound['dataset']
        path = out/(dataset+'-predictions.json')
        save(path, bound)
        measured_audits.append(audit_predictions(bound, points, tps=MODELS[model_id], max_batch=max_batch))
        prospective_audits.append(audit_predictions(bound, [*points,*pending],
                                                    tps=MODELS[model_id], max_batch=max_batch))
        initial_tp = min(MODELS[model_id])
        base = 19000
        config = dict(system='dynamollm', model_id=model_id,
            model_path='/models/'+model_id, tokenizer='/models/'+model_id,
            trace=bound['trace_path'], trace_sha256=bound['trace_sha256'],
            dynamo_predictor_dir=json.loads((cache/'binding.json').read_text())['predictor'],
            profiles=None, qualified_profiles_required=True,
            profile_sources_index=str(source_index.resolve()), pending_profile_plan=str(pending_plan.resolve()),
            node_gpus=list(range(8)), legal_tp=list(MODELS[model_id]),
            instances=[dict(id='dynamo-initial-'+str(i),gpus=list(range(i*initial_tp,(i+1)*initial_tp)),
                tp=initial_tp,pp=1,role='mixed',shape='LL',frequency_mhz=2520,generation=0,
                port=base+2*i,url='http://127.0.0.1:'+str(base+2*i)) for i in range(8//initial_tp)],
            base_port=base,target_port=base+32,store_port=base+80,max_num_seqs=max_batch,
            slo_ttft_s=bound['slo']['ttft_s'],slo_tpot_s=bound['slo']['tpot_s'],
            periods_s=dict(PERIODS),dynamo_reference_tp=4,
            dynamo_require_full_mechanisms=True,mode='comparison',duration_s=duration,seed=701,
            prediction_binding=dict(path=str(path.resolve()),sha256=sha(path),audit_only=True),
            evidence_class='waiting_independent_qualification',formal_eligible=False,
            required_missing_assets=['qualified_merged_profile','workload_coverage_receipt',
                'calibration_shape_demands','weekly_history_workload_mapping',
                'transition_costs_for_enabled_directions','target_tp_goldens',
                'original_cycle_action_receipts','stationary_weight_retention'])
        save(out/(dataset+'-config.json'), config)
    save(out/'batch-extension-candidate.json',dict(schema='dynamo-missing-profile-cells-v1',
        system='dynamollm',model_id=model_id,tp=plan['tp'],pp=1,fit_from_holdout=False,
        points=extension,source_envelope_plan=str(pending_plan.resolve()),
        source_envelope_sha256=sha(pending_plan),prior_cells_recollect=False,
        status='cpu_candidate_only',gpu_preflight_required=True,formal_eligible=False,
        sufficiency='Geometric batch bridge only; nonlinear batch/interpolation holdout must pass before use',
        selection_used_evaluation_outputs=False))
    result = dict(schema='dynamo-frozen-trace-coverage-v1', system='dynamollm',model_id=model_id,
        seed=701,duration_s=duration,trace_index=dict(path=str(index.resolve()),sha256=sha(index)),
        cache_completion_sha256=sha(cache/'completion.json'),raw_revalidated_cells=len(points),
        raw_profile_receipts=receipts,measured_coverage=measured_audits,
        pending_plan=dict(path=str(pending_plan.resolve()),sha256=sha(pending_plan),cells=len(pending),
            status='waiting_gpu_validation',prospective_geometry_only=True,coverage=prospective_audits),
        batch_extension_candidate_cells=len(extension),batch_extension_enqueued=False,
        fit_performed=False,evaluation_used_for_fitting=False,formal_eligible=False,
        status='inconclusive',gpu_actions_started=False,
        blockers=['batch>1 independent profile coverage','independent interpolation and heterogeneous-batch holdout',
            'enabled target TP profile/transition coverage','complete original-period control assets',
            'stationary weight retention','full lifecycle energy protocol'])
    save(out/'review.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('index','cache','corpus','source-index','pending-plan','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-batch',type=int,default=16)
    args=parser.parse_args(argv)
    if not 2 <= args.max_batch <= 16:
        parser.error('explicit admission limit between 2 and 16 required')
    result=prepare(**vars(args))
    print(json.dumps({k:result[k] for k in ('status','raw_revalidated_cells','batch_extension_candidate_cells')}))


if __name__=='__main__':
    main()
