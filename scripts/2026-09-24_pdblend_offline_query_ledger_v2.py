#!/usr/bin/env python3
"""Capture actual PDblend planner queries using frozen non-evaluation inputs.

CPU development preflight only. Never fit profiles, launch engines, lower the
policy's M floor, select on evaluation data, or grant formal qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

from pdblend.bench.client import Request, SLOS
from pdblend.bench.run import offline_forecast
from pdblend.bench.resident_session import write_new
from pdblend.online.policies import get_policy
from pdblend.planner.pool import PoolPlanner, PlannerConfig, SLO
from pdblend.profile.query.versions import load_profile

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = ROOT/'results/2026-09-22/three-model/queue-attempts'
QUERIES = {'prefill_seconds', 'prefill_marginal_seconds', 'prefill_power_w',
    'prefill_energy_j', 'step_seconds', 'decode_supported', 'decode_power_supported',
    'decode_power_w', 'token_energy_j', 'mixed_power_supported', 'mixed_power_w',
    'static_power_w', 'wake_seconds', 'transfer_seconds'}


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_bound(ref):
    if bound(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('immutable input checksum differs: '+ref['path'])
    return json.loads(Path(ref['path']).read_text())


def json_safe(value):
    """Encode nonfinite planner sentinels without altering invocation arguments."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def has_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return any(has_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    return False


def confirmed_longbench_recovery(path, prior_ref):
    """Use the explicit confirmed recovery while preserving inherited traces."""
    from pdblend.bench.longbench_anchor_recovery import validate_prior
    from pdblend.bench.longbench_mixed_combo import classify_recovery
    path=Path(path).resolve();ref=bound(path);result=read_bound(ref)
    preflight_ref=dict(path=str(path.parent/'preflight.json'),sha256=result['artifact_sha256']['preflight.json'])
    preflight=read_bound(preflight_ref)
    if result['prior_inputs']['receipts']['completion']!=prior_ref:
        raise ValueError('recovery is not descended from the named original anchor')
    inherited,failed_rate=validate_prior(result['prior_inputs'],preflight)
    if inherited!=preflight['inherited_anchors'] or failed_rate!=preflight['prior_failed_tuning_rate_rps']:
        raise ValueError('recovery inherited confirmations differ')
    if classify_recovery(result,preflight,path.parent)!='confirmed':
        raise ValueError('recovery has no independently confirmed LongBench rate')
    if result.get('model_id')!='Qwen2.5-32B-Instruct' or result.get('evaluation_used_for_selection') is not False:
        raise ValueError('recovery model or selection identity differs')
    return ref,result,preflight_ref


class QueryLog:
    def __init__(self, base):
        self.base, self.rows = base, {}

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if name not in QUERIES or not callable(value):
            return value

        def query(*args, **kwargs):
            key = json.dumps(json_safe([name, args, kwargs]), sort_keys=True, allow_nan=False)
            row = self.rows.setdefault(key, dict(method=name, args=json_safe(args), kwargs=json_safe(kwargs),
                finite_arguments=not has_nonfinite([args, kwargs]), query_count=0, unsupported_count=0, errors=[], formal_qualified=False))
            row['query_count'] += 1
            try:
                answer = value(*args, **kwargs)
            except ValueError as exc:
                row['unsupported_count'] += 1
                if str(exc) not in row['errors']:
                    row['errors'].append(str(exc))
                raise
            if answer is False and name.endswith('_supported'):
                row['unsupported_count'] += 1
            return answer
        return query


def run(args):
    output = args.out.resolve()
    if (output/'bindings.json').exists():
        raise FileExistsError('use a new output directory; existing evidence is immutable')
    from pdblend.profile.collection.native_frequency_domain import validate_domain,domain_fields
    domain_ref=bound(args.frequency_domain) if getattr(args,'frequency_domain',None) else None
    domain=validate_domain(read_bound(domain_ref)) if domain_ref else None
    if domain:
        if args.models != [domain['model_id'].split('-')[1].lower()] or tuple(args.frequencies)!=tuple(domain['frequencies_mhz']):
            raise ValueError('new frequency ledger model/endpoints differ from explicit domain')
    elif not args.frequencies or any(f not in (1500,2520) for f in args.frequencies):
        raise ValueError('nonlegacy frequencies require a new explicit PD frequency domain')
    bindings, ledgers = {}, []
    started = time.time()
    for size in args.models:
        model_id = 'Qwen2.5-'+size.upper()+'-Instruct'
        manifest_path = ROOT/'results/2026-09-23/native-ab-smoke-prepared-v3/inputs'/size/'manifest.json'
        manifest_ref = bound(manifest_path)
        manifest = read_bound(manifest_ref)
        profile_ref = manifest['inputs']['profile']
        profile = read_bound(profile_ref)
        tp, pp = profile['tp'], profile['pp']
        loaded = load_profile(profile_ref['path'], system='pdblend', model_id=model_id, tp=tp, pp=pp,
                              usage='development')
        anchor_paths = list(ATTEMPTS.glob(f'mixed-rate-anchor-{size}-e954f2cc36b8/attempt-*/anchor/completion.json'))
        if len(anchor_paths) != 1:
            raise ValueError('ambiguous/missing explicitly named rate-anchor attempt')
        anchor_ref = bound(anchor_paths[0]); anchor = read_bound(anchor_ref)
        if anchor.get('evaluation_used_for_selection') is not False or anchor.get('model_id') != model_id:
            raise ValueError('anchor is not bound to non-evaluation model-owned selection')
        bindings[size] = dict(profile=profile_ref, input_manifest=manifest_ref, rate_anchor=anchor_ref,
                              model_verification=manifest['inputs']['model_verification'], datasets={})
        recovery=None
        if size=='32b' and getattr(args,'longbench_recovery_completion',None):
            recovery_ref,recovery,recovery_preflight=confirmed_longbench_recovery(args.longbench_recovery_completion,anchor_ref)
            bindings[size]['longbench_recovery']=dict(completion=recovery_ref,preflight=recovery_preflight)
        for dataset in args.datasets:
            selected_anchor=recovery if dataset=='longbench' and recovery else anchor
            selected = selected_anchor.get('anchors', {}).get(dataset)
            if not selected:
                ledgers.append(dict(model_id=model_id, dataset=dataset, status='blocked_missing_tuning_anchor',
                                    formal_eligible=False, queries=[]))
                continue
            relative = Path(selected['confirmation_path']).relative_to('/output/anchor')
            confirmation_root=Path(recovery_ref['path']).parent if dataset=='longbench' and recovery else anchor_paths[0].parent
            confirmation_path = confirmation_root/relative
            confirmation_ref = dict(path=str(confirmation_path.resolve()), sha256=selected['confirmation_sha256'])
            confirmation = read_bound(confirmation_ref)
            trace_path = confirmation_path.parent/'requests.json'
            trace_ref = dict(path=str(trace_path.resolve()), sha256=confirmation['trace_sha256'])
            trace = read_bound(trace_ref)
            if (confirmation.get('split') != 'tuning' or confirmation.get('dataset') != dataset
                    or confirmation['metrics'].get('passed') is not True
                    or trace.get('seed') != selected_anchor['tuning_seed']):
                raise ValueError('independent tuning confirmation identity differs')
            requests = [Request(**row) for row in trace['requests']]
            if any(row.source != dataset for row in requests):
                raise ValueError('tuning content is not owned by the named real dataset')
            forecast = offline_forecast(requests)
            bindings[size]['datasets'][dataset] = dict(confirmation=confirmation_ref, tuning_trace=trace_ref)
            for scale in args.scales:
                log = QueryLog(loaded.model)
                policy = get_policy('pdblend')
                cfg = policy.planner_config(PlannerConfig(slots=8//tp, slo=SLO(*SLOS[dataset]),
                                            freqs=tuple(args.frequencies), max_num_seqs=32))
                if cfg.min_m_instances != 4:
                    raise ValueError('canonical PDblend M floor changed; review before collecting')
                target = selected['base_rate_rps']*scale
                fc = replace(forecast, rate_rps=target)
                error, candidates = None, []
                try:
                    # Enumeration captures rejected branches too. No plan is
                    # selected, exported for deployment, or optimized here.
                    candidates = PoolPlanner(log, cfg).candidates(fc)
                except ValueError as exc:
                    error = str(exc)
                rows = sorted(log.rows.values(), key=lambda r:(-r['unsupported_count'], -r['query_count'],
                                                               r['method'], str(r['args'])))
                ledgers.append(dict(model_id=model_id, dataset=dataset, tp=tp, pp=pp,
                    rate_scale=scale, target_rate_rps=target, trace_requests=len(requests),
                    selection_split='tuning', formal_eligible=False, hardware_executed=False,
                    policy='pdblend', min_m_instances=cfg.min_m_instances, slots=cfg.slots,
                    frequency_scope=list(cfg.freqs), engine_max_num_seqs=32,
                    **({'frequency_domain_sha256':domain_fields(domain)['frequency_domain_sha256']} if domain else {}),
                    forecast=dict(input_mean=fc.input_mean, input_p95=fc.input_p95, output_mean=fc.output_mean),
                    status='development_enumeration' if error is None else 'development_query_blocked', error=error,
                    feasible_development_candidates=len(candidates), deployable_plan_selected=False,
                    query_count=sum(r['query_count'] for r in rows), unique_queries=len(rows),
                    unsupported_queries=sum(bool(r['unsupported_count']) for r in rows), queries=rows))
    result = dict(schema='pdblend-offline-query-ledger/v2', scope='development_non_evaluation_preflight',
        formal_eligible=False, hardware_executed=False, evaluation_read=False, parameters_fitted=False,
        offline_plan_selected=False, min_m_floor_overridden=False, ledgers=ledgers,
        limits=['Calls are actual existing planner invocations, not scheduler observations.',
                'The profile is an unqualified legacy candidate; a successful numeric query is not qualified.',
                'Two-frequency scope is prospective and cannot promote an original six-frequency profile.',
                'Rejected branches stop early; rerun the ledger after new components to expose downstream gaps.',
                'Fractional decode batches are model queries, not launchable integer-batch GPU samples.',
                'Existing bounded optimization components are not attached; their raw audits remain separate.'])
    if domain:result.update(frequency_domain_ref=domain_ref,**domain_fields(domain))
    gaps = []
    for ledger in ledgers:
        for query in ledger['queries']:
            if query['unsupported_count']:
                gaps.append(dict(model_id=ledger['model_id'],dataset=ledger['dataset'],
                    rate_scale=ledger['rate_scale'],**query,
                    next_action=('collect_or_validate_this_domain_without_clamping_or_extrapolation'
                                 if query['finite_arguments'] else
                                 'resolve_upstream_unsupported_domain_never_sample_nonfinite_sentinels')))
    supplement = dict(schema='pdblend-minimum-supplement/v2', formal_eligible=False,
        scope='observed_first_blockers_only_not_a_full_profile', missing_queries=gaps,
        required_even_when_queries_numerically_pass=['native_cuda_timing_and_independent_holdout',
            'runtime_capacity_static_wake_transfer_dvfs_raw_audit', 'scoped_full_profile_composition_audit',
            'actual_tuning_workload_query_coverage', 'pdblend_action_kv_generation_and_window_energy_acceptance'],
        no_exact_gpu_point_count_yet='Resolve actual domain/interpolation rules before translating model queries to a GPU grid.')
    write_new(output/'query-ledger.json', result)
    write_new(output/'minimum-supplement.json', supplement)
    modules = ['src/pdblend/planner/pool.py','src/pdblend/planner/forecast.py','src/pdblend/online/policies.py',
               'src/pdblend/bench/run.py','src/pdblend/profile/query/versions.py','src/pdblend/profile/query/model.py']
    if domain:modules.append('src/pdblend/profile/collection/native_frequency_domain.py')
    receipt = dict(schema='pdblend-offline-readiness-bindings/v1', formal_eligible=False,
        hardware_executed=False, evaluation_read=False, execution_started_s=started, execution_finished_s=time.time(),
        inputs=bindings, script=bound(__file__), implementation={name:bound(ROOT/name) for name in modules},
        outputs={p.name:bound(p) for p in (output/'query-ledger.json',output/'minimum-supplement.json')})
    if domain:receipt.update(frequency_domain_ref=domain_ref,**domain_fields(domain))
    if (output/'README.md').exists():
        receipt['outputs']['README.md'] = bound(output/'README.md')
    write_new(output/'bindings.json', receipt)
    print(json.dumps(dict(out=str(output), runs=len(ledgers), actual_queries=sum(x.get('query_count',0) for x in ledgers),
                         distinct_first_gaps=len(gaps), formal_eligible=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=('7b','14b','32b'), default=['7b'])
    parser.add_argument('--datasets', nargs='+', choices=tuple(SLOS), default=list(SLOS))
    parser.add_argument('--frequencies', nargs='+', type=int, default=[1500,2520])
    parser.add_argument('--frequency-domain',type=Path,help='Explicit PD-only domain manifest; no old query relabeling')
    parser.add_argument('--scales', nargs='+', type=float, choices=(.25,.5,.75,1.), default=[.5])
    parser.add_argument('--longbench-recovery-completion',type=Path,
                        help='Explicit confirmed 32B recovery; original Alpaca/ShareGPT inputs remain bound')
    run(parser.parse_args())
