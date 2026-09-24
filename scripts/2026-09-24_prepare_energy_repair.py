#!/usr/bin/env python3
"""Freeze paired controller repairs, preserving original traces and controls."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.comparison_pdblend_observation import validate_observation_inputs
from pdblend.bench.comparison_runtime import pdblend_window_resources
from pdblend.bench.independent_dispatch import request_rows
from pdblend.bench.pdblend_observation_plan import decode_plan, encode_plan
from pdblend.bench.pdblend_runtime_options import DEFAULTS, ARTIFACTS
from pdblend.bench.resident_session import digest, engine_signature, write_new
from pdblend.bench.run import offline_forecast
from pdblend.control.policies import get_policy
from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
from pdblend.profile.query.versions import load_profile

ROOT = Path(__file__).resolve().parents[1]
CASES = ('7b-pdblend-sharegpt-x0.25-seed701', '7b-pdblend-sharegpt-x0.5-seed701',
         '7b-pdblend-alpaca-x0.5-seed701', '7b-pdblend-alpaca-x1-seed701',
         '7b-pdblend-longbench-x1-seed701')


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_hashes(files):
    runtime = digest({k: v for k, v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))})
    measurement = digest({k: v for k, v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')})
    return runtime, measurement


def startup_choice(point, ceiling):
    old = load_bound(point['inputs']['offline_choice'])
    initial = decode_plan(old, observation=True)
    profile = point['inputs']['profiles'][0]
    loaded = load_profile(profile['path'], system='pdblend', model_id=point['model_id'],
                          tp=initial.tp, pp=initial.pp, usage='development')
    if ceiling not in loaded.model.freqs:
        raise ValueError('candidate ceiling lacks explicit profile support')
    cfg = get_policy('pdblend').planner_config(PlannerConfig(
        slots=len(point['engine_identity']['instances']), slo=SLO(**point['slo']),
        freqs=tuple(f for f in loaded.model.freqs if f <= ceiling), max_num_seqs=32))
    cfg.preserve_overload_capacity = True
    forecast = offline_forecast(request_rows(load_bound(point['inputs']['planning_trace'])))
    plan = PoolPlanner(loaded.model, cfg).evaluate(initial.counts, ceiling, ceiling, ceiling,
                                                 initial.tau, forecast, strict=False)
    if plan is None:
        plan = replace(initial, f_P=ceiling, f_D=ceiling, f_M=ceiling,
                       power_w=float('inf'), ttft_s=float('inf'), tpot_s=float('inf'),
                       detail={'startup_prediction': 'unavailable'})
    plan = replace(plan, tp=initial.tp, pp=initial.pp, pool_id=initial.pool_id,
                   generation=initial.generation, profile_key=initial.profile_key)
    return dict(old, **encode_plan(plan), parent_choice=point['inputs']['offline_choice'],
                startup_policy='same_initial_counts_with_candidate_sustainable_clock',
                candidate_clock_mhz=ceiling, clock_qualification_pending=True,
                evaluation_used_for_selection=False)


def prepare(parent_path, out, *, ceiling=2100, cases=CASES, controls=True):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent_ref = binding(parent_path)
    parent = load_bound(parent_ref)
    execution = load_bound(parent['execution_inputs'])
    freezer = load_module(ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    source, revision = freezer.freeze_source(ROOT/'src', out/'sources')
    source_ref = binding(source/'manifest.json')
    runtime_sha, measurement_sha = method_hashes(load_bound(source_ref)['files'])
    options = dict(DEFAULTS, **{key: None for key in ARTIFACTS}, safety_max_freq=ceiling,
                   startup_safety=True, deadline_safety=True)
    by_name = {p['name']: p for p in parent['points']}
    points, groups, jobs = [], [], []
    # Candidate first provides prompt feedback. Controls are explicitly rerun,
    # separately frozen; subsequent repetitions counterbalance source order.
    for arm in (('candidate', 'control') if controls else ('candidate',)):
        selected = []
        for case in cases:
            original = by_name[case]
            point = deepcopy(original)
            point.update(name=case+'-energy-repair-'+arm+'-r0', run_id=out.name,
                         status='prepared', blockers=[], formal_eligible=False)
            point['repair_experiment'] = dict(case=case, arm=arm, repeat=0,
                same_evaluation_trace=True, source_pairing='separate_resident_sessions',
                selection_basis='predeclared_mechanism_repairs', clock_qualification_pending=arm=='candidate')
            if arm == 'candidate':
                if point['engine_identity']['runtime_source_sha256'] != runtime_sha:
                    raise ValueError('controller repair unexpectedly changed inference engine source')
                choice = startup_choice(original, ceiling)
                choice['runtime_options'] = options
                config = load_bound(original['inputs']['system_config'])
                config['pdblend_runtime'] = options
                choice_path, config_path = out/'choices'/(point['name']+'.json'), out/'configs'/(point['name']+'.json')
                write_new(choice_path, choice)
                write_new(config_path, config)
                point.update(revision=revision, source_manifest=source_ref)
                point['engine_identity']['measurement_source_sha256'] = measurement_sha
                point['inputs'].update(source_manifest=source_ref, offline_choice=binding(choice_path),
                                       system_config=binding(config_path))
                point['optimization_version'] = dict(source_manifest=source_ref,
                    profile=point['inputs']['profiles'][0], requested=options,
                    qualification='development_only', clock_qualification_pending=True)
            validate_observation_inputs(point, point['inputs'])
            specs = [SimpleNamespace(tp=r['tp'], pp=r['pp'], generation=0)
                     for r in point['engine_identity']['instances']]
            pdblend_window_resources(point, specs)
            selected.append(point)
        for model_id in dict.fromkeys(p['model_id'] for p in selected):
            members = [p for p in selected if p['model_id'] == model_id]
            identity = members[0]['engine_identity']
            if any(p['engine_identity'] != identity for p in members):
                raise ValueError('repair group engine identities differ')
            group = dict(session_id='energy-repair-'+arm+'-'+digest(members)[:16], model_id=model_id,
                         engine_identity=identity, engine_signature=engine_signature(identity),
                         points=members, gpu_count=8, exclusive=True, reserve_host=True)
            path = out/'groups'/(group['session_id']+'.json')
            write_new(path, group)
            arm_source = Path(members[0]['source_manifest']['path']).parent
            job = resident_job(group, path, root=ROOT, source=arm_source,
                image=execution['image_digest'], verification=execution['model_verification']['path'],
                campaign=out/'campaign.json', priority=900)
            job['payload'].update(system='pdblend', model_id=model_id, depends_on=[],
                after_terminal=[jobs[-1]['job_id']] if jobs else [], result_policy='all_recorded_windows/v1',
                observation_scope='pdblend_profile_unqualified_evaluation/v1', formal_eligible=False)
            jobs.append(job)
            groups.append(group)
        points.extend(selected)
    campaign = dict(parent, campaign_id=out.name, run_id=out.name, parent_campaign=parent_ref,
                    points=points, groups=groups, candidate_source=source_ref,
                    summary=dict(points=len(points), jobs=len(jobs), service_seconds=150*len(points)),
                    repair_protocol=dict(options=options, cases=list(cases), paired_controls_reexecuted=controls,
                        service_and_tail_total_energy=True, energy_integration_rule_unchanged=True,
                        frequency_telemetry_source_changed=True, formal_eligible=False))
    campaign.pop('extension_policy_refs', None)
    write_new(out/'campaign.json', campaign)
    write_new(out/'jobs.json', jobs)
    write_new(out/'preparation.json', dict(campaign=binding(out/'campaign.json'), source=source_ref,
        jobs=binding(out/'jobs.json'), preparer=binding(Path(__file__)), hardware_executed=False,
        cpu_preflight_passed=True, container_preflight_pending=True, enqueued=False))
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ceiling', type=int, default=2100)
    parser.add_argument('--cases', nargs='+', default=CASES)
    parser.add_argument('--candidate-only', action='store_true')
    args = parser.parse_args()
    result = prepare(args.parent, args.out, ceiling=args.ceiling, cases=args.cases,
                     controls=not args.candidate_only)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
