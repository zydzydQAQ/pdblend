#!/usr/bin/env python3
"""Prepare native profile recovery with raw replay and immutable jobs; never enqueue.

7B/14B timing is isolated from unqualified runtime/power auxiliaries. The 32B
existing 1500/2100 design is a separate bounded M4/TP2 layout revision, never a
replacement claim for the canonical online profile. All historical files remain
unchanged. The replay subcommand captures NEW evidence only after a job ends.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.profile.collection.native_timing_plan import binding, read_bound
from pdblend.profile.collection.native_readiness import (
    collection_reuse_decision, inspect_attempt, read_snapshot, terminal_jobs,
    verify_prepared_bindings,
)

DEFAULT_QUEUE = ROOT/'results/2026-09-22/three-model/queue.json'
PREPARATIONS = {
    '7b': ROOT/'results/2026-09-24/pdblend-native-timing-runtime-v3',
    '14b': ROOT/'results/2026-09-24/pdblend-native-timing-runtime-14b-v2',
    '32b': ROOT/'results/2026-09-24/pdblend-native-timing-layout-32b-v3',
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return binding(path)


def freezer():
    path = ROOT/'scripts/2026-09-24_prepare_pdblend_native_timing.py'
    spec = importlib.util.spec_from_file_location('native_recovery_freezer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(argv, *, depends_on=(), status='ready', reason=None):
    return dict(argv=[str(v) for v in argv], shell=shlex.join([str(v) for v in argv]),
                depends_on=list(depends_on), status=status, reason=reason)


def canonical_dependencies(size):
    """Timing reuse cannot erase the unmeasured canonical profile gates."""
    return [dict(node=size+'-canonical-power-and-handoff', kind='implementation_and_gpu_collection',
                 status='blocked_missing_qualified_collector_and_composition', depends_on=[],
                 blockers=['active_prefill_kernel_or_explicit_complete_cycle_energy_semantics',
                     'fractional_batch_and_short_actual_context_training_and_disjoint_holdout',
                     'physical_handoff_observer_and_explicit_planner_query_semantics'],
                 existing_pilot_is_sufficient=False),
            dict(node=size+'-canonical-query-selection', kind='cpu_selection',
                 status='blocked_missing_components',
                 depends_on=[size+'-terminal-replay', size+'-canonical-power-and-handoff'],
                 entrypoint='pdblend.profile.query.native_composition.audit_native_profile'),
            dict(node=size+'-selected-layout-holdout', kind='gpu_serving_energy_holdout',
                 status='blocked_missing_frozen_selection',
                 depends_on=[size+'-canonical-query-selection'],
                 entrypoint='pdblend.profile.query.native_serving_holdout.replay_serving_holdout')]


def prepare(out, *, queue=DEFAULT_QUEUE, preparations=None, after_terminal=()):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('new immutable recovery directory required')
    if (not isinstance(after_terminal, (list, tuple))
            or any(not isinstance(job_id, str) or not job_id.strip() for job_id in after_terminal)
            or len(set(after_terminal)) != len(after_terminal)):
        raise ValueError('after_terminal must contain distinct explicit job IDs')
    preparations = preparations or PREPARATIONS
    queue_value, queue_ref = read_snapshot(queue)
    jobs = terminal_jobs(queue_value)
    prior = {}
    # Fail on corrupt/stale design inputs before creating any preparation files.
    for size, path in preparations.items():
        manifest_ref = binding(Path(path)/'manifest.json')
        manifest = read_bound(manifest_ref)
        # The historical 7B runtime preparation still carries a v1 plan. Use
        # the already declared all-model v2 shape/holdout design, not a relabel.
        if size in ('7b', '14b'):
            manifest = dict(manifest, point_plan=binding(ROOT/
                f'results/2026-09-24/pdblend-native-timing-model-plans-v2/{size}-point-plan.json'))
        plan, source = verify_prepared_bindings(manifest)
        from pdblend.profile.collection.native_timing_plan_v2 import validate_plan
        validate_plan(plan)
        prior[size] = dict(manifest_ref=manifest_ref, manifest=manifest, plan=plan, source=source)
    out.mkdir(parents=True)
    queue_snapshot = write(out/'terminal-jobs.json', dict(
        schema='pdblend-native-terminal-inventory/v1', jobs=jobs,
        observed_queue_bytes=queue_ref, historical_blocked_reason_inherited=False))
    attempts_root = Path(queue).resolve().parent/'queue-attempts'
    attempts = []
    for path in sorted(attempts_root.glob('pdblend-native-timing-*/attempt-*/native-timing/completion.json')):
        attempt = path.parent.parent
        attempts.append(inspect_attempt(attempt, queue=queue,
            evidence_dir=out/'reused-timing-evidence'/attempt.parent.name/attempt.name))
    attempts_ref = write(out/'attempt-evidence.json', attempts)
    models = {}
    all_jobs = []
    graph = []
    builder = ROOT/'scripts/2026-09-24_prepare_pdblend_native_timing.py'
    freeze = freezer()
    for size, row in prior.items():
        plan, manifest = row['plan'], row['manifest']
        model = plan['model_id']
        owned = [a for a in attempts if a['model_id'] == model]
        # Qualification belongs to exact model/shape/frequency/source evidence.
        # Never skip a new frequency design because another domain passed.
        owned_for_design = [a for a in owned if
            read_bound(read_bound(a['input_manifest'])['point_plan']) == plan
            and a['source']['source_sha256'] == row['source']['source_sha256']]
        reuse = collection_reuse_decision(owned_for_design)
        reuse['all_model_raw_runtime_to_retain'] = collection_reuse_decision(owned)['raw_runtime_to_retain']
        layout = manifest.get('layout_energy_plan') if size == '32b' else None
        graph += canonical_dependencies(size)
        # The layout collector requires its own live timing-stage snapshot on
        # the same resident fleet. A reusable timing component alone cannot
        # skip this still-unqualified layout job after a later-phase failure.
        reuse['layout_requires_new_resident_timing_stage'] = bool(layout)
        if not reuse['collect_timing'] and not layout:
            models[size] = dict(model_id=model, reused_timing=reuse['qualified_timing'], reuse=reuse,
                collection_prepared=False, reason='Exact design already independently replays as qualified.',
                canonical_blockers=[r['node'] for r in canonical_dependencies(size)],
                full_profile_qualified=False, canonical_online_profile_consumer_ready=False)
            graph.append(dict(node=size+'-terminal-replay', kind='cpu_replay', status='reused_qualified',
                depends_on=[], evidence=[a['timing_evidence'] for a in reuse['qualified_timing']]))
            continue
        target = out/size/'collection'
        provenance = plan.get('query_provenance', plan.get('query_bindings'))
        argv = [sys.executable, '-B', builder, '--out', target,
                '--point-plan', manifest['point_plan']['path'],
                '--ledger', plan['query_ledger']['path'], '--bindings', provenance['path'],
                '--source-base', Path(manifest['source_manifest']['path']).parent, '--timing-first']
        # This exact-domain 32B runtime does not yet exist. Collect it once and
        # reuse the established train/freeze/select/independent-holdout chain.
        if layout:
            read_bound(layout)
            argv += ['--collect-runtime', '--layout-energy-plan', layout['path']]
        prepared = freeze.prepare(target, Path(plan['query_ledger']['path']), Path(provenance['path']),
            point_plan=Path(manifest['point_plan']['path']),
            source_base=Path(manifest['source_manifest']['path']).parent,
            timing_first=True, collect_runtime=bool(layout),
            layout_energy_plan=Path(layout['path']) if layout else None)
        new_jobs = read_bound(prepared['jobs'])
        execution_jobs = prepared['jobs']
        if after_terminal:
            for job in new_jobs:
                job['payload'] = dict(job['payload'], after_terminal=list(after_terminal))
            # Preserve the collector's immutable preparation; bind the actual
            # queue envelope separately, with terminal rather than success dependencies.
            execution_jobs = write(target.parent/'execution-jobs.json', new_jobs)
        all_jobs.extend(new_jobs)
        job_id = new_jobs[0]['job_id']
        replay = command([sys.executable, '-B', Path(__file__).resolve(), 'replay',
                          '--queue', queue, '--attempt', '{attempt_dir}',
                          '--out', '{new_replay_directory}'], depends_on=[job_id])
        entry = dict(model_id=model, tp=plan['tp'], pp=plan['pp'],
            prior_preparation=row['manifest_ref'], point_plan=prepared['point_plan'],
            source_manifest=prepared['source_manifest'], jobs=execution_jobs,
            collection_jobs=prepared['jobs'],
            frequencies_mhz=plan.get('frequency_domain', {}).get('frequencies_mhz', [1500, 2520]),
            query_ledger=plan['query_ledger'], query_provenance=provenance,
            point_count=len(plan['points']), measurement_windows=sum(p['repeats'] for p in plan['points']),
            unsupported_query_shapes=len(plan.get('unsupported_queries', [])),
            blocked_ledger_entries=len(plan.get('blocked_ledger_entries', [])),
            training_points=sum(p['purpose'] == 'training' for p in plan['points']),
            independent_holdout_points=sum(p['purpose'] == 'holdout' for p in plan['points']),
            reuse=reuse, prepare_command=command(argv), post_terminal_replay=replay,
            prepared_job_id=job_id, full_profile_qualified=False,
            canonical_online_profile_consumer_ready=False,
            selection_loader='pdblend.profile.query.native_composition.load_native_profile',
            precise_gaps=[
                'qualified_native_timing_missing',
                'full_runtime_holdout_failed; signed_handoff_can_be_nonpositive_and_is_not_physical_copy_time',
                'no_frozen_active_prefill_and_decode_power_candidate_with_independent_holdout',
                'short_context_and_fractional_occupancy_power_domain_unqualified',
                'complete_12_group_candidate_query_replay_requires_qualified_components',
                'selected_deployment_150s_serving_energy_holdout_requires_frozen_candidate_and_selection',
            ])
        if layout:
            entry.update(bounded_layout_loader='pdblend.profile.query.native_layout_profile.replay_layout_profile',
                layout_energy_plan=prepared['layout_energy_plan'],
                layout_scope='32B_M4_TP2_1500_2100_Poisson_static_layout_revision_only',
                old_runtime_reusable_in_new_frequency_domain=False,
                existing_scoped_runtime=[a['runtime']['scoped_all_m_runtime'] for a in owned
                    if a.get('runtime', {}).get('scoped_all_m_runtime')],
                bounded_layout_chain=['fresh_exact_domain_runtime', 'timing', 'freeze_timing_component',
                    'layout_training', 'freeze_power_candidate', '12_group_selection',
                    'freeze_selection', 'independent_150s_layout_holdout', 'terminal_raw_replay'])
        else:
            entry.update(runtime_recollection_in_this_job=False, power_pilot_recollection_in_this_job=False,
                         reason='Retain already collected raw evidence; failing auxiliaries must not prevent timing collection.')
        models[size] = entry
        graph += [dict(node=job_id, kind='gpu_collection', status='prepared_not_enqueued', depends_on=[],
                       after_terminal=list(after_terminal), jobs=execution_jobs, gpu_count=8, exclusive=True,
                       outputs=['native-timing/completion.json'], full_profile_qualified=False),
                  dict(node=size+'-terminal-replay', kind='cpu_replay', **replay)]
    combined_jobs = write(out/'jobs.json', all_jobs)
    dependencies = write(out/'dependencies.json', graph)
    existing_protocols = {}
    for name in ('pdblend-native-7b14b-recovery-readiness-v1/transfer-plan.json',
                 'pdblend-native-composition-readiness-v1/review.json'):
        path = ROOT/'results/2026-09-24'/name
        if path.exists():
            existing_protocols[name] = binding(path)
    report = dict(schema='pdblend-native-profile-recovery-preparation/v1',
        prepared_only=True, hardware_executed=False, enqueued=False,
        full_profile_qualified=False, formal_eligible=False, energy_comparable=False,
        historical_results_modified=False, historical_blocked_reason_inherited=False,
        after_terminal=list(after_terminal),
        terminal_inventory=queue_snapshot, attempt_evidence=attempts_ref,
        jobs=combined_jobs, dependencies=dependencies, models=models,
        existing_protocols=existing_protocols,
        execution_scope='These jobs fill components only. Canonical full-profile blockers remain explicit in dependencies.json.',
        builder=binding(__file__), implementation={name:binding(ROOT/'src/pdblend/profile/collection'/name)
            for name in ('native_readiness.py', 'native_runtime_audit.py', 'native_timing_replay.py',
                         'native_timing_stage.py')},
        no_automatic_queue_command='Preparation does not mutate queue state; jobs use normal exclusive eight-GPU lease placeholders.')
    write(out/'readiness.json', report)
    return report


def replay(attempt, queue, out):
    """Actual runnable CPU endpoint; partial timing requires terminal stage proof."""
    from pdblend.profile.collection.native_timing_replay import capture_evidence, replay_evidence
    from pdblend.profile.collection.native_timing_stage import capture_terminal_evidence, replay_terminal_evidence
    attempt, out = Path(attempt).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('new immutable replay directory required')
    report = json.loads((attempt/'native-timing/completion.json').read_text())
    out.mkdir(parents=True)
    if report.get('resident_timing_stage'):
        ref = capture_terminal_evidence(attempt, queue, out/'evidence.json')
        result = replay_terminal_evidence(ref)
    else:
        ref = capture_evidence(attempt, queue, out/'evidence.json')
        result = replay_evidence(ref)
    write(out/'replay.json', result)
    return dict(evidence=ref, replay=binding(out/'replay.json'),
                component_qualified=result.get('component_qualified',
                    (result.get('component') or {}).get('component_qualified', False)),
                full_profile_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--queue', type=Path, default=DEFAULT_QUEUE)
    p.add_argument('--after-terminal', action='append', default=[],
                   help='Wait for this existing job to terminate, regardless of success; repeat as needed')
    p = sub.add_parser('replay')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--queue', type=Path, default=DEFAULT_QUEUE)
    p.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    result = (prepare(args.out, queue=args.queue, after_terminal=args.after_terminal) if args.command == 'prepare'
              else replay(args.attempt, args.queue, args.out))
    print(json.dumps(dict(out=str(args.out.resolve()), full_profile_qualified=False,
                         models=list(result.get('models', {})), hardware_executed=False)))


if __name__ == '__main__':
    main()
