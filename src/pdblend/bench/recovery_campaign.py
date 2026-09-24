"""Immutable, paired diagnostic windows for SLO recovery changes.

Controls execute the actual historical source; candidates use a complete new
snapshot. Neither this preparation nor its descriptive report grants profile
qualification or substitutes observations for missing instrument uncertainty.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

from .comparison_campaign import binding, load_bound
from .comparison_jobs import resident_job
from .resident_session import digest, engine_signature, file_sha, write_new

CASES = (
    ('7b', 'longbench', 1., ('control', 'all_m', 'recovery')),
    ('14b', 'longbench', 1., ('control', 'all_m', 'handoff')),
    ('7b', 'sharegpt', .25, ('control', 'shield')),
    ('32b', 'alpaca', 1., ('control', 'shield')),
)


def arm_options(arm):
    if arm not in {'all_m', 'recovery', 'handoff', 'shield'}:
        raise ValueError('control must use the frozen historical source')
    return dict(shield_mode='budget_aware' if arm == 'shield' else 'legacy',
        slo_routing=arm in {'recovery', 'handoff'},
        preserve_overload_capacity=arm == 'recovery', safety_recovery=arm == 'recovery',
        experiment_mode='freeze_initial_all_m' if arm == 'all_m' else 'adaptive')


def method_hashes(files):
    runtime = {k: v for k, v in files.items() if k.startswith(('pdblend_runtime/', 'pdblend/engine/'))}
    measurement = {k: v for k, v in files.items() if k.startswith('pdblend/measure/') or k in (
        'pdblend/bench/comparison_metrics.py', 'pdblend/bench/comparison_metering.py', 'pdblend/bench/client.py')}
    return digest(runtime), digest(measurement)


def all_m_choice(point):
    from .independent_dispatch import request_rows
    from .run import offline_forecast
    from pdblend.online.policies import get_policy
    from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
    from pdblend.profile.query.versions import load_profile
    config = load_bound(point['inputs']['system_config'])
    choice = load_bound(point['inputs']['offline_choice'])
    initial = Plan(**choice['plan'])
    profile = config['profile']
    load_bound(profile)
    loaded = load_profile(profile['path'], system='pdblend', model_id=point['model_id'],
                          tp=initial.tp, pp=1, usage='development')
    prior = load_bound(point['inputs']['planning_trace'])
    if prior.get('selection_split') not in {'calibration', 'tuning'}:
        raise ValueError('fixed all-M estimates require independent planning data')
    cfg = get_policy('pdblend').planner_config(PlannerConfig(slots=len(point['engine_identity']['instances']),
        slo=SLO(**point['slo']), freqs=loaded.model.freqs, max_num_seqs=32))
    f = max(loaded.model.freqs)
    plan = PoolPlanner(loaded.model, cfg).evaluate({'M': cfg.slots}, f, f, f, 0,
                                                 offline_forecast(request_rows(prior)), strict=False)
    if plan is None or not all(math.isfinite(getattr(plan, k)) for k in ('power_w', 'ttft_s', 'tpot_s')):
        raise ValueError('fixed all-M timing estimates unavailable; do not manufacture an offline choice')
    plan = replace(plan, tp=initial.tp, pp=initial.pp, profile_key=initial.profile_key)
    return dict(choice, plan=asdict(plan), diagnostic_selection='fixed_all_m_max_frequency',
                evaluation_used_for_selection=False)


def prepare(base_path, out, *, root, repeats=3, models=('7b', '14b', '32b'), after_terminal=()):
    if type(repeats) is not int or repeats < 1:
        raise ValueError('paired diagnostic campaign requires at least one repetition')
    if not models or set(models) - {'7b', '14b', '32b'}:
        raise ValueError('invalid models')
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('new campaign directory required')
    base_ref = binding(base_path)
    base = load_bound(base_ref)
    execution = load_bound(base['execution_inputs'])
    control_source = Path(execution['source'])
    spec = importlib.util.spec_from_file_location('recovery_source_freezer',
        root/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(freezer)
    control_manifest = load_bound(binding(control_source/'manifest.json'))
    freezer.verify_snapshot(control_source, control_manifest['files'])
    # All preparation reads are CPU-only; no Docker, GPU, worker or global CSV side effects.
    out.mkdir(parents=True)
    candidate_source, candidate_sha = freezer.freeze_source(root/'src', out/'sources')
    candidate_manifest = load_bound(binding(candidate_source/'manifest.json'))
    if method_hashes(candidate_manifest['files']) != method_hashes(control_manifest['files']):
        raise ValueError('common engine or numerical metering changed; cannot pair these snapshots')
    points, groups, jobs = [], [], []
    base_points = {p['name']: p for p in base['points']}
    for repeat in range(repeats):
        for model in models:
            # Counterbalance controller-source order between independently reset repeats.
            for source_kind in (('control', 'candidate') if repeat % 2 == 0 else ('candidate', 'control')):
                source = control_source if source_kind == 'control' else candidate_source
                revision = source.name
                selected = []
                for size, dataset, scale, arms in CASES:
                    if size != model:
                        continue
                    name = f'{size}-pdblend-{dataset}-x{scale:g}-seed701'
                    parent = base_points[name]
                    for arm in arms:
                        if (arm == 'control') != (source_kind == 'control'):
                            continue
                        point = deepcopy(parent)
                        point['name'] = f'{name}-{arm}-r{repeat}'
                        point.update(revision=revision, source_manifest=binding(source/'manifest.json'),
                            status='prepared', formal_eligible=False, profile_qualified=False,
                            result_policy='all_recorded_windows/v1',
                            recovery_experiment=dict(case=name, arm=arm, repeat=repeat,
                                parent_campaign=base_ref, parent_point_sha256=digest(parent),
                                fixed_output_and_trace=True, qualification='diagnostic_only'))
                        point['inputs']['source_manifest'] = point['source_manifest']
                        if arm != 'control':
                            config = load_bound(parent['inputs']['system_config'])
                            config['pdblend_runtime'] = arm_options(arm)
                            path = out/'configs'/(point['name']+'.json')
                            write_new(path, config); point['inputs']['system_config'] = binding(path)
                            if arm == 'all_m':
                                choice_path = out/'choices'/(point['name']+'.json')
                                write_new(choice_path, all_m_choice(parent))
                                point['inputs']['offline_choice'] = binding(choice_path)
                        from .comparison_pdblend_observation import validate_observation_inputs
                        validate_observation_inputs(point, point['inputs'])
                        if arm != 'control':
                            from .comparison_runtime import pdblend_window_resources
                            specs = [SimpleNamespace(tp=r['tp'], pp=r['pp'], generation=0)
                                     for r in point['engine_identity']['instances']]
                            pdblend_window_resources(point, specs)
                        # Verify all directly consumed immutable inputs before publishing a job.
                        for key in ('trace', 'system_config', 'offline_choice', 'planning_trace', 'source_manifest'):
                            load_bound(point['inputs'][key])
                        selected.append(point); points.append(point)
                if not selected:
                    continue
                if repeat % 2:
                    selected.reverse()
                identity = selected[0]['engine_identity']
                if any(p['engine_identity'] != identity for p in selected):
                    raise ValueError('cannot reuse incompatible physical fleets')
                group = dict(session_id='recovery-'+digest([p['name'] for p in selected]+[revision])[:20],
                    model_id=selected[0]['model_id'], engine_identity=identity,
                    engine_signature=engine_signature(identity), points=selected,
                    gpu_count=8, exclusive=True, reserve_host=True)
                path = out/'groups'/(group['session_id']+'.json'); write_new(path, group)
                job = resident_job(group, path, root=root, source=source, image=execution['image_digest'],
                    verification=execution['model_verification']['path'], campaign=out/'campaign.json',
                    priority=1800-len(jobs))
                job['payload'].update(system='pdblend', model_id=group['model_id'],
                    after_terminal=list(after_terminal), depends_on=[jobs[-1]['job_id']] if jobs else [],
                    observation_scope='pdblend_profile_unqualified_evaluation/v1',
                    result_policy='all_recorded_windows/v1', formal_eligible=False)
                groups.append(group); jobs.append(job)
    campaign = dict(schema='pdblend-recovery-campaign/v1', campaign_id=out.name,
        parent_campaign=base_ref, seed=701, repeats=repeats, duration_s=150.,
        control_source=binding(control_source/'manifest.json'),
        candidate_source=binding(candidate_source/'manifest.json'),
        primary_energy_scope='eight_gpu_service_plus_complete_tail',
        statistical_claim=False, formal_eligible=False, points=points, groups=groups,
        repeat_scope='single_run_diagnostic' if repeats == 1 else 'repeated_diagnostic',
        summary=dict(points=len(points), jobs=len(jobs), service_seconds=len(points)*150))
    write_new(out/'campaign.json', campaign); write_new(out/'jobs.json', jobs)
    write_new(out/'preflight.json', dict(schema='pdblend-recovery-preflight/v1', passed=True,
        campaign=binding(out/'campaign.json'), jobs=binding(out/'jobs.json'),
        candidate_source_sha256=candidate_sha, models=list(models), repeats=repeats,
        common_runtime_and_meter_unchanged=True, hardware_executed=False, enqueued=False,
        qualifications_inherited=False, profile_missing_gates=[
            'legacy_profile_not_formally_qualified', 'native_profile_holdout_and_query_coverage_not_established']))
    return campaign


def report(campaign_path, queue_path):
    campaign = json.loads(Path(campaign_path).read_text())
    jobs = json.loads((Path(campaign_path).parent/'jobs.json').read_text())
    queue = json.loads(Path(queue_path).read_text())
    entries = queue['jobs']; entries = entries.values() if isinstance(entries, dict) else entries
    by_id = {j['job_id']: j for j in entries}
    attempts = Path(queue_path).parent/(Path(queue_path).stem+'-attempts')
    rows, states, unreadable = [], [], []
    expected_points = {p['name']: p for p in campaign['points']}
    expected_groups = {g['session_id']: g for g in campaign['groups']}
    verified_files = {}
    def verify_file(path, expected):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        actual = verified_files.get(key)
        if actual is None:
            actual = file_sha(path)
            after = path.stat()
            if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
                raise ValueError('artifact changed while reading: '+str(path))
            verified_files[key] = actual
        if actual != expected:
            raise ValueError('artifact checksum differs: '+str(path))
    for job in jobs:
        state = by_id.get(job['job_id'], {'status': 'not_enqueued'})
        states.append(dict(job_id=job['job_id'], status=state['status'], last_error=state.get('last_error')))
        for path in sorted((attempts/job['job_id']).glob('attempt-*/session/windows/*/receipt.json')):
            try:
                receipt = json.loads(path.read_text())
                point = json.loads((path.parent/'point.json').read_text())
                if not isinstance(receipt, dict) or not isinstance(point, dict):
                    raise ValueError('receipt and point must be objects')
            except (OSError, ValueError) as exc:
                unreadable.append(dict(path=str(path), error=str(exc)))
                continue
            metrics = receipt.get('result', {}).get('metrics', {})
            audit = receipt.get('result', {}).get('acceptance', {})
            runtime = None
            errors = []
            try:
                expected = expected_points.get(point.get('name'))
                if expected != point or receipt.get('point_sha256') != digest(point):
                    raise ValueError('window does not match its frozen campaign point')
                group = expected_groups[job['payload']['session_id']]
                if (receipt.get('session_id') != group['session_id']
                        or receipt.get('engine_signature') != group['engine_signature']
                        or point not in group['points']):
                    raise ValueError('window source/engine/session ownership differs')
                lease = json.loads((path.parents[3]/'manifest.json').read_text())
                if (lease.get('job_id') != job['job_id'] or lease.get('immutable') is not True
                        or lease['payload'].get('source_sha256') != point['revision']):
                    raise ValueError('window is not bound to the expected immutable source lease')
                result_file = json.loads((path.parent/'result.json').read_text())
                if result_file != receipt.get('result'):
                    raise ValueError('receipt metrics differ from the immutable result')
                if not receipt.get('artifacts'):
                    raise ValueError('window lacks artifact bindings')
                for name, sha in receipt['artifacts'].items():
                    artifact = (path.parent/name).resolve()
                    if not artifact.is_relative_to(path.parent.resolve()):
                        raise ValueError('artifact escapes window directory')
                    verify_file(artifact, sha)
                for ref in [point['trace'], point['source_manifest'], *point['inputs']['profiles'],
                            *(point['inputs'][k] for k in ('system_config', 'planning_trace', 'offline_choice'))]:
                    verify_file(ref['path'], ref['sha256'])
                if json.loads((path.parent/'drain.json').read_text()).get('passed') is not True:
                    raise ValueError('final native drain did not pass')
                native_ref = audit.get('raw_refs', {}).get('native_result')
                if native_ref is not None:
                    verify_file(native_ref['path'], native_ref['sha256'])
                    runtime = json.loads(Path(native_ref['path']).read_text()).get('pdblend_runtime')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(str(exc))
            service, tail = metrics.get('energy_service_j'), metrics.get('energy_tail_j')
            energy = service+tail if all(type(v) in (int, float) and math.isfinite(v)
                                          and v >= 0 for v in (service, tail)) else None
            good = metrics.get('good_output_tokens', 0)
            required = {'metering.raw_eight_gpu_window', 'pdblend.canonical_metrics'}
            gates = set(audit.get('checked_gates', []))
            canonical_valid = not errors and 'pdblend.canonical_metrics' in gates
            energy_valid = not errors and required <= gates
            measurement_valid = not errors and audit.get('measurement_evidence_valid') is True
            metadata = expected_points.get(point.get('name'), {}).get('recovery_experiment',
                dict(case=None, arm=None, repeat=None, qualification='unrecognized_point'))
            rows.append(dict(**metadata, receipt=binding(path),
                point_sha256=receipt.get('point_sha256'), metrics=metrics, energy_service_tail_j=energy,
                j_per_good_token=energy/good if energy is not None and type(good) is int and good > 0 else None,
                artifact_valid=not errors, artifact_errors=errors, measurement_valid=measurement_valid,
                canonical_metrics_valid=canonical_valid, energy_metrics_valid=energy_valid,
                runtime_receipt=runtime,
                cleanup_passed=receipt.get('cleanup_passed', False),
                qualification_audit=audit))
    comparisons = []
    for case in dict.fromkeys(p['recovery_experiment']['case'] for p in campaign['points']):
        arms = list(dict.fromkeys(p['recovery_experiment']['arm'] for p in campaign['points']
                                 if p['recovery_experiment']['case'] == case))
        pairs = [('control', arm) for arm in arms if arm != 'control']
        if 'all_m' in arms:
            pairs += [('all_m', arm) for arm in arms if arm not in ('control', 'all_m')]
        if {'handoff', 'first_gap_guard'} <= set(arms):
            pairs.append(('handoff', 'first_gap_guard'))
        for reference_arm, arm in pairs:
            controls = [r for r in rows if r['case'] == case and r['arm'] == reference_arm]
            candidates = [r for r in rows if r['case'] == case and r['arm'] == arm]
            complete = all(len(a) == campaign['repeats'] and
                {r['repeat'] for r in a} == set(range(campaign['repeats'])) for a in (controls, candidates))
            slo_pass = complete and all(r['canonical_metrics_valid'] and r['metrics'].get('slo_pass') is True and r['cleanup_passed']
                                         for r in controls+candidates)
            energy_complete = complete and all(r['energy_metrics_valid'] and r['cleanup_passed']
                and r['energy_service_tail_j'] is not None and r['energy_service_tail_j'] > 0
                for r in controls+candidates)
            saving = (1-statistics.mean(r['energy_service_tail_j'] for r in candidates)/
                        statistics.mean(r['energy_service_tail_j'] for r in controls)) if energy_complete else None
            comparisons.append(dict(case=case, reference_arm=reference_arm, arm=arm,
                complete=complete, both_arms_slo_pass=slo_pass,
                measurement_qualified=complete and all(r['measurement_valid'] for r in controls+candidates),
                observed_energy_saving=saving, improvement_proven=False,
                reason='instrument_uncertainty_and_profile_qualification_required' if complete else 'incomplete_repeats'))
    return dict(schema='pdblend-recovery-report/v1', campaign=binding(campaign_path), jobs=states,
                windows=rows, unreadable_windows=unreadable, comparisons=comparisons, formal_eligible=False,
                all_jobs_terminal=all(s['status'] in {'succeeded', 'failed', 'cancelled'} for s in states))


def container_preflight(campaign_path, out, root):
    """Validate both frozen interpreters without GPU access or engine startup."""
    campaign = json.loads(Path(campaign_path).read_text())
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('new container preflight output required')
    out.mkdir(parents=True)
    results, seen = [], set()
    for group in campaign['groups']:
        point = group['points'][0]
        key = (point['revision'], point['model_id'])
        if key in seen:
            continue
        seen.add(key)
        source = Path(point['source_manifest']['path']).parent
        group_path = Path(campaign_path).parent/'groups'/(group['session_id']+'.json')
        command = ['docker', 'run', '--rm', '--network=none', '--entrypoint', '/opt/venv/bin/python',
            '-v', f'{source}:/opt/pdblend-src:ro', '-v', f'{root}:{root}:ro',
            '-e', 'PYTHONPATH=/opt/pdblend-src', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            point['engine_identity']['image_digest'], '-B', str(root/'scripts/2026-09-24_preflight_recovery.py'),
            '--group', str(group_path.resolve())]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
        row = dict(group=binding(group_path), command=command, returncode=completed.returncode,
            stdout=completed.stdout, stderr=completed.stderr, hardware_executed=False)
        write_new(out/(group['session_id']+'.json'), row)
        results.append(row)
        if completed.returncode:
            break
    value = dict(schema='pdblend-recovery-container-preflight/v1', campaign=binding(campaign_path),
        groups=results, passed=len(results) == len({(p['revision'],p['model_id']) for p in campaign['points']})
            and all(r['returncode'] == 0 for r in results), hardware_executed=False)
    write_new(out/'completion.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--base', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path.cwd()); p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--models', nargs='+', choices=['7b', '14b', '32b'], default=['7b', '14b', '32b'])
    p.add_argument('--after-terminal', action='append', default=[])
    p = sub.add_parser('report')
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p = sub.add_parser('preflight')
    p.add_argument('--campaign', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == 'prepare':
        value = prepare(args.base, args.out, root=args.root, repeats=args.repeats,
                        models=args.models, after_terminal=args.after_terminal)
        print(json.dumps(value['summary']))
    elif args.command == 'preflight':
        value = container_preflight(args.campaign, args.out, args.root)
        print(json.dumps(dict(passed=value['passed'], groups=len(value['groups']), out=str(args.out))))
        if not value['passed']:
            raise SystemExit(1)
    else:
        value = report(args.campaign, args.queue)
        if args.out:
            write_new(args.out, value)
        print(json.dumps({k:v for k,v in value.items() if k != 'windows'}, indent=2))


if __name__ == '__main__':
    main()
