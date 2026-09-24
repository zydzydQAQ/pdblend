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


class QueryLog:
    def __init__(self, base):
        self.base, self.rows = base, {}

    def __getattr__(self, name):
        value = getattr(self.base, name)
        if name not in QUERIES or not callable(value):
            return value

        def query(*args, **kwargs):
            key = json.dumps([name, args, kwargs], sort_keys=True, allow_nan=False)
            row = self.rows.setdefault(key, dict(method=name, args=args, kwargs=kwargs,
                query_count=0, unsupported_count=0, errors=[], formal_qualified=False))
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
        for dataset in args.datasets:
            selected = anchor.get('anchors', {}).get(dataset)
            if not selected:
                ledgers.append(dict(model_id=model_id, dataset=dataset, status='blocked_missing_tuning_anchor',
                                    formal_eligible=False, queries=[]))
                continue
            relative = Path(selected['confirmation_path']).relative_to('/output/anchor')
            confirmation_path = anchor_paths[0].parent/relative
            confirmation_ref = dict(path=str(confirmation_path.resolve()), sha256=selected['confirmation_sha256'])
            confirmation = read_bound(confirmation_ref)
            trace_path = confirmation_path.parent/'requests.json'
            trace_ref = dict(path=str(trace_path.resolve()), sha256=confirmation['trace_sha256'])
            trace = read_bound(trace_ref)
            if (confirmation.get('split') != 'tuning' or confirmation.get('dataset') != dataset
                    or confirmation['metrics'].get('passed') is not True
                    or trace.get('seed') != anchor['tuning_seed']):
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
                    forecast=dict(input_mean=fc.input_mean, input_p95=fc.input_p95, output_mean=fc.output_mean),
                    status='development_enumeration' if error is None else 'development_query_blocked', error=error,
                    feasible_development_candidates=len(candidates), deployable_plan_selected=False,
                    query_count=sum(r['query_count'] for r in rows), unique_queries=len(rows),
                    unsupported_queries=sum(bool(r['unsupported_count']) for r in rows), queries=rows))
    result = dict(schema='pdblend-offline-query-ledger/v1', scope='development_non_evaluation_preflight',
        formal_eligible=False, hardware_executed=False, evaluation_read=False, parameters_fitted=False,
        offline_plan_selected=False, min_m_floor_overridden=False, ledgers=ledgers,
        limits=['Calls are actual existing planner invocations, not scheduler observations.',
                'The profile is an unqualified legacy candidate; a successful numeric query is not qualified.',
                'Two-frequency scope is prospective and cannot promote an original six-frequency profile.',
                'Rejected branches stop early; rerun the ledger after new components to expose downstream gaps.',
                'Fractional decode batches are model queries, not launchable integer-batch GPU samples.',
                'Existing bounded optimization components are not attached; their raw audits remain separate.'])
    gaps = []
    for ledger in ledgers:
        for query in ledger['queries']:
            if query['unsupported_count']:
                gaps.append(dict(model_id=ledger['model_id'],dataset=ledger['dataset'],
                    rate_scale=ledger['rate_scale'],**query,
                    next_action='collect_or_validate_this_domain_without_clamping_or_extrapolation'))
    supplement = dict(schema='pdblend-minimum-supplement/v1', formal_eligible=False,
        scope='observed_first_blockers_only_not_a_full_profile', missing_queries=gaps,
        required_even_when_queries_numerically_pass=['native_cuda_timing_and_independent_holdout',
            'runtime_capacity_static_wake_transfer_dvfs_raw_audit', 'scoped_full_profile_composition_audit',
            'actual_tuning_workload_query_coverage', 'pdblend_action_kv_generation_and_window_energy_acceptance'],
        no_exact_gpu_point_count_yet='Resolve actual domain/interpolation rules before translating model queries to a GPU grid.')
    write_new(output/'query-ledger.json', result)
    write_new(output/'minimum-supplement.json', supplement)
    modules = ['src/pdblend/planner/pool.py','src/pdblend/planner/forecast.py','src/pdblend/online/policies.py',
               'src/pdblend/bench/run.py','src/pdblend/profile/query/versions.py','src/pdblend/profile/query/model.py']
    receipt = dict(schema='pdblend-offline-readiness-bindings/v1', formal_eligible=False,
        hardware_executed=False, evaluation_read=False, execution_started_s=started, execution_finished_s=time.time(),
        inputs=bindings, script=bound(__file__), implementation={name:bound(ROOT/name) for name in modules},
        outputs={p.name:bound(p) for p in (output/'query-ledger.json',output/'minimum-supplement.json')})
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
    parser.add_argument('--frequencies', nargs='+', type=int, choices=(1500,2520), default=[1500,2520])
    parser.add_argument('--scales', nargs='+', type=float, choices=(.25,.5,.75,1.), default=[.5])
    run(parser.parse_args())
