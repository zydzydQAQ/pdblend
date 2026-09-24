#!/usr/bin/env python3
"""Freeze completed repair receipts and compare each revision with fixed baselines.

Read-only with respect to campaigns, receipts, source bundles and the GPU queue.
Every invocation requires a new --out directory; it never updates old snapshots.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pdblend.bench.comparison_campaign import binding, load_bound
from pdblend.bench.cohort_dominance import BASELINE_SYSTEMS, analyze_points, complete_energy, paired_errors, scenario
from pdblend.bench.measurement_compatibility import hydrate_receipt_evidence, load_compatibility, prepare_compatibility
from pdblend.bench.resident_session import digest, write_new
from report_cohort_energy import analyze_point, csv_write, json_write


def stamp(seconds=None):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat() if seconds is not None else datetime.now(timezone.utc).isoformat()


def realized_rate(point):
    count = point['trace'].get('requests')
    if type(count) is not int:
        count = len(load_bound(point['trace'])['requests'])
    return count / point['duration_s']


def receipt_point(path):
    ref = binding(path)
    receipt = load_bound(ref)
    point = load_bound(dict(path=str(path.parent / 'point.json'), sha256=receipt['artifacts']['point.json']))
    if digest(point) != receipt['point_sha256']:
        raise ValueError('receipt point digest differs: ' + str(path))
    return ref, receipt, point


def read_observation(path, *, supplement=False):
    ref, receipt, point = receipt_point(path)
    metrics, identity = receipt['result']['metrics'], point['engine_identity']
    source = load_bound(point['source_manifest'])
    revision = source['source_sha256']
    system = point['system']
    arm = None if supplement else point['repair_experiment']['arm']
    series = system if supplement else 'pd_' + arm + '_' + revision[:12]
    raw = dict(metrics, system=system, model_id=point['model_id'], dataset=point['dataset'],
               rate_scale=point['scale'], offered_rps=point['rate_rps'], seed=point['seed'],
               trace_sha256=point['trace']['sha256'], revision=revision, point_id=point['name'],
               receipt_path=str(path.resolve()), receipt_sha256=ref['sha256'],
               evidence_valid=receipt['result'].get('evidence_valid'),
               formal_eligible=receipt['result'].get('formal_eligible'))
    raw.update({k: identity[k] for k in ('model_hash', 'tokenizer_hash', 'image_digest',
                                        'runtime_source_sha256', 'measurement_source_sha256')})
    row, failures, provenance = analyze_point(dict(series=series, **ref,
        receipt_sha256=ref['sha256']), raw)
    row = hydrate_receipt_evidence(row)
    row.update(observation_kind='baseline_supplement' if supplement else 'repair',
               measurement_missing_gates=receipt['result'].get('missing_gates', []))
    if not supplement:
        row.update(repair_arm=arm, repair_repeat_index=point['repair_experiment']['repeat'],
                   experiment_case=point['repair_experiment']['case'])
    return row, failures, provenance, point


def read_repair(path):
    return read_observation(path)


def baseline_core_review(original, supplement):
    """Coordinator changes cannot silently change a baseline or its metering."""
    for field in ('system', 'model_id', 'dataset', 'scale', 'rate_rps', 'seed',
                  'duration_s', 'slo', 'trace'):
        if original[field] != supplement[field]:
            raise ValueError('supplement changes original condition: ' + field)
    if original['name'] != supplement['original_point_name']:
        raise ValueError('supplement original point name differs')
    if supplement['comparison_selection'] != 'first_predeclared_complete_service_and_tail_attempt':
        raise ValueError('unsupported supplement selection rule')
    allowed = {'name', 'source_manifest', 'revision', 'metering_execution', 'run_id', 'status',
               'blockers', 'result_policy', 'energy_supplement_of', 'original_point_name',
               'comparison_selection', 'formal_eligible'}
    def unchanged_settings(point):
        settings = {k: v for k, v in point.items() if k not in allowed}
        settings['inputs'] = {k: v for k, v in point['inputs'].items() if k != 'source_manifest'}
        settings['engine_identity'] = {k: v for k, v in point['engine_identity'].items()
                                       if k != 'metering_execution'}
        return settings
    if unchanged_settings(original) != unchanged_settings(supplement):
        raise ValueError('supplement changes baseline settings outside metering execution')
    endpoints, protected = [], []
    for point in (original, supplement):
        ref = point['source_manifest']
        manifest = load_bound(ref)
        files = manifest['files']
        if manifest['source_sha256'] != digest(files):
            raise ValueError('baseline source manifest identity differs')
        names = sorted(k for k in files if k.startswith((
            'pdblend_baselines/', 'pdblend_runtime/', 'pdblend/engine/', 'pdblend/measure/'))
            or k in ('pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py',
                     'pdblend/bench/comparison_metering.py'))
        required = {'pdblend/measure/power.py', 'pdblend/measure/backends.py',
                    'pdblend/bench/client.py', 'pdblend/bench/comparison_metrics.py',
                    'pdblend/bench/comparison_metering.py'}
        if not required <= set(names) or not any(k.startswith('pdblend_baselines/') for k in names):
            raise ValueError('baseline source lacks required protected implementation')
        root = Path(ref['path']).parent.resolve()
        for name in names:
            path = (root / name).resolve()
            if not path.is_relative_to(root) or binding(path)['sha256'] != files[name]:
                raise ValueError('baseline protected source checksum differs: ' + name)
        protected.append({name: files[name] for name in names})
        endpoints.append(ref)
    if protected[0] != protected[1]:
        raise ValueError('baseline core or energy measurement changed in supplement')
    return dict(source_manifests=endpoints, protected_files_sha256=digest(protected[0]),
                protected_file_count=len(protected[0]), baseline_core_and_meter_unchanged=True)


def collect_supplements(preparation_path, attempt_root):
    """Read only bound jobs, exact declared points, and numeric attempt order."""
    ref = binding(preparation_path)
    preparation = load_bound(ref)
    campaign = load_bound(preparation['campaign'])
    jobs = load_bound(preparation['jobs'])
    declared = {p['name']: p for p in campaign['points']}
    if len(declared) != len(campaign['points']):
        raise ValueError('duplicate supplement declaration')
    groups = {g['session_id']: g for g in campaign['groups']}
    rows, audits, failures, provenance, sources = [], [], [], [], []
    covered, job_ids = set(), set()
    for job_index, job in enumerate(jobs):
        job_id, payload = job['job_id'], job['payload']
        if Path(job_id).name != job_id or job_id in job_ids:
            raise ValueError('invalid or duplicate declared supplement job')
        job_ids.add(job_id)
        if Path(payload['comparison_campaign']).resolve() != Path(preparation['campaign']['path']).resolve():
            raise ValueError('supplement job campaign differs')
        group = groups[payload['session_id']]
        members = {p['name']: p for p in group['points']}
        if covered.intersection(members):
            raise ValueError('supplement point is assigned to multiple jobs')
        covered.update(members)
        for name, point in members.items():
            if declared.get(name) != point or point['system'] not in BASELINE_SYSTEMS:
                raise ValueError('job group differs from declared baseline point')
            old_ref = point['energy_supplement_of']
            _, _, original = receipt_point(Path(old_ref['path']))
            load_bound(old_ref)
            review = baseline_core_review(original, point)
            sources.append(dict(point=name, historical_receipt=old_ref, **review))
        attempts = []
        for directory in (Path(attempt_root) / job_id).glob('attempt-*'):
            match = re.fullmatch(r'attempt-(\d+)-[^/]+', directory.name)
            if not match or not directory.is_dir():
                raise ValueError('invalid supplement attempt path: ' + str(directory))
            attempts.append((int(match[1]), directory))
        if len({n for n, _ in attempts}) != len(attempts):
            raise ValueError('ambiguous supplement attempt ordinal: ' + job_id)
        for attempt, directory in sorted(attempts):
            window_root = directory / 'session' / 'windows'
            for path in window_root.glob('*/receipt.json'):
                if path.parent.name not in members:
                    raise ValueError('undeclared receipt in supplement job: ' + str(path))
            for name, point in members.items():
                path = window_root / name / 'receipt.json'
                audit = dict(job_id=job_id, job_index=job_index, attempt=attempt,
                             point=name, revision=point['revision'], historical_receipt=point['energy_supplement_of'],
                             status='receipt_missing', receipt_path=str(path.resolve()), selected=False)
                if path.exists():
                    receipt_ref, receipt, actual = receipt_point(path)
                    if actual != point:
                        raise ValueError('supplement receipt differs from predeclared point: ' + str(path))
                    audit['receipt'] = receipt_ref
                    if not isinstance(receipt.get('result', {}).get('metrics'), dict):
                        audit.update(status='no_complete_metrics')
                    else:
                        try:
                            row, details, prov, _ = read_observation(path, supplement=True)
                        except FileNotFoundError as exc:
                            # An interrupted run may bind metrics before every raw
                            # artifact exists. It cannot supply complete energy.
                            # Existing-but-mismatched hashes still fail the report.
                            audit.update(status='missing_raw_artifact', error=str(exc))
                            audits.append(audit)
                            continue
                        row.update(supplement_of=point['energy_supplement_of'],
                                   supplement_job_id=job_id, supplement_job_index=job_index,
                                   supplement_attempt=attempt, supplement_preparation=ref)
                        rows.append(row); failures.extend(details); provenance.append(prov)
                        audit.update(status='complete_energy' if complete_energy(row) is not None else 'incomplete_energy',
                                     total_energy_kj=complete_energy(row),
                                     measurement_evidence_valid=row.get('measurement_evidence_valid'),
                                     raw_eight_gpu_meter_qualified=row.get('raw_eight_gpu_meter_qualified'),
                                     all_requests_successful=row['all_requests_successful'])
                audits.append(audit)
    if covered != set(declared):
        raise ValueError('supplement declarations are not exactly covered by bound jobs')
    return rows, audits, failures, provenance, dict(preparation=ref, campaign=preparation['campaign'],
        jobs=preparation['jobs'], declared_points=len(declared), source_reviews=sources,
        declarations=[dict(point=p['name'], historical_receipt=p['energy_supplement_of']) for p in declared.values()])


def select_baseline_observations(historical, supplements):
    """Keep complete historical points; otherwise first complete attempt, never best."""
    keys = {(r['receipt_path'], r['receipt_sha256']) for r in historical if r['system'] in BASELINE_SYSTEMS}
    by_original = defaultdict(list)
    for row in supplements:
        ref = row['supplement_of']
        key = ref['path'], ref['sha256']
        if key not in keys:
            raise ValueError('supplement does not bind the selected historical receipt')
        by_original[key].append(row)
    selected, decisions = [], []
    for original in historical:
        if original['system'] not in BASELINE_SYSTEMS:
            selected.append(original)
            continue
        key = original['receipt_path'], original['receipt_sha256']
        attempts = sorted(by_original[key], key=lambda r: (r['supplement_job_index'], r['supplement_attempt']))
        for attempt in attempts:
            if attempt['system'] != original['system'] or paired_errors(original, attempt):
                raise ValueError('supplement differs from historical comparison identity')
        order = [(r['supplement_job_index'], r['supplement_attempt']) for r in attempts]
        if len(set(order)) != len(order):
            raise ValueError('duplicate attempt for one historical baseline')
        chosen, reason = original, 'historical_complete' if complete_energy(original) is not None else 'no_complete_supplement'
        if complete_energy(original) is None:
            chosen = next((r for r in attempts if complete_energy(r) is not None), original)
            if chosen is not original:
                reason = 'first_predeclared_complete_service_and_tail_attempt'
        selected.append(chosen)
        decisions.append(dict(system=original['system'], model=original['model'], dataset=original['dataset'],
            rate_scale=original['rate_scale'], selection_reason=reason,
            historical_receipt=dict(path=original['receipt_path'], sha256=original['receipt_sha256']),
            selected_receipt=dict(path=chosen['receipt_path'], sha256=chosen['receipt_sha256']),
            selected_revision=chosen['revision'], selected_attempt=chosen.get('supplement_attempt'),
            historical_energy_kj=complete_energy(original), selected_energy_kj=complete_energy(chosen),
            selected_measurement_qualified=chosen.get('measurement_evidence_valid'),
            selected_raw_meter_qualified=chosen.get('raw_eight_gpu_meter_qualified'),
            selected_formal_eligible=chosen.get('formal_eligible'),
            selected_all_requests_successful=chosen['all_requests_successful'],
            observed_supplement_attempts=len(attempts)))
    return selected, decisions


def repair_preparation_inputs(paths, attempt_root):
    """Expand immutable repair packages for a CPU-only report watcher."""
    campaigns, directories, refs = [], [], []
    for path in paths:
        ref = binding(path)
        preparation = load_bound(ref)
        campaign = load_bound(preparation['campaign'])
        jobs = load_bound(preparation['jobs'])
        if any(p['system'] != 'pdblend' or 'repair_experiment' not in p for p in campaign['points']):
            raise ValueError('repair preparation is not a declared PD repair campaign')
        campaigns.append(Path(preparation['campaign']['path']))
        for job in jobs:
            job_id = job['job_id']
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', job_id):
                raise ValueError('invalid repair job id')
            if Path(job['payload']['comparison_campaign']).resolve() != campaigns[-1].resolve():
                raise ValueError('repair job campaign differs from preparation')
            directories.append(Path(attempt_root) / job_id)
        refs.append(dict(preparation=ref, campaign=preparation['campaign'], jobs=preparation['jobs']))
    return campaigns, directories, refs


def draw(points, planned, snapshot, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    datasets = ('alpaca', 'sharegpt', 'longbench')
    models = sorted({p['model'] for p in points}, key=lambda s: int(s[:-1]))
    baseline_names = dict(mixed='Mixed', ecoserve='EcoServe', distserve='DistServe', dynamollm='DynamoLLM')
    series = list(BASELINE_SYSTEMS) + sorted({p['series'] for p in points if p['system'] == 'pdblend'})
    series = [s for s in series if any(p['series'] == s for p in points)]
    palette = ['#64748b', '#b98000', '#aa6bc1', '#ce607e', '#008577', '#0077bb', '#ee7733', '#9a6c23', '#009988']
    colors = {s: palette[i % len(palette)] for i, s in enumerate(series)}
    candidate_series = [s for s in series if 'candidate' in s]
    markers = {s: ('D' if s == candidate_series[0] and len(candidate_series) > 1 else '*')
               if 'candidate' in s else 'o' for s in series}
    labels = {}
    for s in series:
        row = next(p for p in points if p['series'] == s)
        if s in baseline_names:
            labels[s] = baseline_names[s]
        else:
            kind = ('previous' if row.get('historical_series') == 'pd_previous' else
                    'pre-repair' if row.get('observation_kind') == 'historical'
                    else row.get('repair_arm', 'repair'))
            labels[s] = f"PD {kind}: {row['revision'][:8]}"
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    specs = [('total_energy_kj', 'energy-vs-rate', 'GPU energy: service + tail', 'Energy (kJ)'),
             ('slo_attainment_pct', 'slo-attainment-vs-rate', 'Joint SLO attainment', 'Attainment (%)'),
             ('success_pct', 'success-rate-vs-rate', 'Request success rate', 'Success (%)')]
    with PdfPages(out / 'comparison.pdf') as pdf:
        for metric, filename, title, ylabel in specs:
            multi = len(models) > 1
            fig, axes = plt.subplots(len(models), 3, figsize=(17, 4.9 * len(models) + 2.3 if multi else 6.8), squeeze=False)
            fig.subplots_adjust(left=.065, right=.985, bottom=.10 if multi else .255,
                                top=.85 if multi else .69, wspace=.23, hspace=.43)
            fig.suptitle(title + ' vs realized arrival rate', x=.065, y=.98, ha='left', fontsize=21, fontweight='bold')
            fig.text(.065, .952 if multi else .905, 'Frozen: ' + snapshot['captured_at'] +
                     f" | {snapshot['repair_completed_points']} completed repair windows | revisions remain separate", fontsize=10)
            handles = [Line2D([], [], color=colors[s], marker=markers[s],
                              linestyle='--' if 'pre_repair' in s else '-', markersize=9 if 'candidate' in s else 5,
                              markerfacecolor='none' if 'candidate' in s and markers[s] == 'D' else colors[s],
                              label=labels[s]) for s in series]
            fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .935 if multi else .86), ncol=4, frameon=False, fontsize=10)
            for i, model in enumerate(models):
                for j, dataset in enumerate(datasets):
                    ax = axes[i, j]
                    panel = [p for p in points if p['model'] == model and p['dataset'] == dataset]
                    grid = sorted({(p['rate_scale'], p['realized_arrival_rate_rps']) for p in panel}
                                  | {(p['scale'], realized_rate(p)) for p in planned
                                     if p['model_id'].split('-')[1] == model and p['dataset'] == dataset})
                    xs = [r for _, r in grid]
                    missing = Counter()
                    for s in series:
                        selected = {p['rate_scale']: p for p in panel if p['series'] == s}
                        if not selected:
                            continue
                        rows = [selected.get(scale) for scale, _ in grid]
                        ys = [p[metric] if p is not None and p.get(metric) is not None else math.nan for p in rows]
                        ax.plot(xs, ys, color=colors[s], marker=markers[s],
                                linestyle='--' if 'pre_repair' in s else '-',
                                linewidth=2.2 if 'candidate' in s else 1.5,
                                markerfacecolor='none' if 'candidate' in s and markers[s] == 'D' else colors[s],
                                markeredgewidth=2 if 'candidate' in s and markers[s] == 'D' else 1,
                                markersize=12 if 'candidate' in s else 5, zorder=8 if 'candidate' in s else 3)
                        for x, y, p in zip(xs, ys, rows):
                            if p is None:
                                continue
                            if not math.isfinite(y):
                                missing[labels[s]] += 1
                            elif metric == 'total_energy_kj':
                                if not p['all_requests_successful']:
                                    ax.scatter([x], [y], marker='x', s=80, color='#d1262e', zorder=20)
                                elif not p['slo_pass']:
                                    ax.scatter([x], [y], marker='o', s=85, facecolors='none', edgecolors='#222', zorder=20)
                                if p.get('observation_kind') in ('repair', 'baseline_supplement') and p.get('measurement_evidence_valid') is not True:
                                    ax.scatter([x], [y], marker='s', s=120, facecolors='none', edgecolors='#f28e2b', zorder=21)
                    if missing:
                        ax.text(.02, .97, 'Energy NA: ' + '; '.join(f'{s} {n}' for s, n in missing.items()), transform=ax.transAxes,
                                va='top', fontsize=7.5, color='#9b3838', wrap=True)
                    ax.set_title(model + ' / ' + dataset.title(), loc='left', fontweight='bold')
                    ax.set_xlabel('Realized arrival rate (N / 150 s)')
                    ax.set_ylabel(ylabel)
                    ax.set_xticks(xs)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.3g}'))
                    ax.grid(axis='y', color='#e2e8f0')
                    ax.set_axisbelow(True)
                    if metric == 'total_energy_kj':
                        ax.set_ylim(0, max((p[metric] for p in panel if p.get(metric) is not None), default=100) * 1.16)
                    else:
                        ax.set_ylim(0, 105)
                    if metric == 'slo_attainment_pct':
                        ax.axhline(90, color='#777', linestyle='--', linewidth=1)
            fig.text(.065, .062 if multi else .135, 'Missing rates break lines. Failed baselines remain energy targets. Supplements use the first predeclared complete attempt, never lowest energy.', fontsize=9)
            fig.text(.065, .039 if multi else .09, 'Energy: all 8 GPU boards, service + tail. Red x: failed requests; hollow circle: hard SLO failed; orange square: new observation measurement not qualified.', fontsize=9)
            fig.text(.065, .016 if multi else .045, 'Single observations. A complete win also requires all four baseline energies, joint good requests >= the best baseline, and candidate measurement acceptance.', fontsize=9)
            fig.savefig(out / (filename + '.png'), dpi=200)
            fig.savefig(out / (filename + '.svg'))
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-dir', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, action='append', default=[])
    parser.add_argument('--job-dir', type=Path, action='append', default=[])
    parser.add_argument('--repair-preparation', type=Path, action='append', default=[],
                        help='Expand hash-bound repair campaign and jobs; repeat for separate revisions.')
    parser.add_argument('--attempt-root', type=Path,
                        help='Queue-attempts root for --repair-preparation; also defaults supplement attempt root.')
    parser.add_argument('--supplement-preparation', type=Path,
                        help='Hash-bound preparation.json declaring the baseline supplement campaign and jobs.')
    parser.add_argument('--supplement-attempt-root', type=Path,
                        help='Queue-attempts root; only jobs bound by supplement preparation are read.')
    parser.add_argument('--all-models', action='store_true',
                        help='Include all historical models and both historical PD revisions (implied by supplements).')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.repair_preparation and not args.attempt_root:
        parser.error('--repair-preparation requires --attempt-root')
    expanded_campaigns, expanded_jobs, repair_preparations = repair_preparation_inputs(
        args.repair_preparation, args.attempt_root)
    args.campaign = list(dict.fromkeys(args.campaign + expanded_campaigns))
    args.job_dir = list(dict.fromkeys(args.job_dir + expanded_jobs))
    if not args.campaign or not args.job_dir:
        parser.error('provide --repair-preparation, or both --campaign and --job-dir')
    if args.supplement_preparation and not args.supplement_attempt_root:
        args.supplement_attempt_root = args.attempt_root
    if bool(args.supplement_preparation) != bool(args.supplement_attempt_root):
        parser.error('--supplement-preparation and --supplement-attempt-root must be supplied together')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    captured = stamp()
    campaigns = [(binding(path), load_bound(binding(path))) for path in args.campaign]
    planned = [p for _, c in campaigns for p in c['points']]
    allowed = {(p['name'], p['revision']): p for p in planned}
    models = {p['model_id'].split('-')[1] for p in planned}
    historical_ref = binding(args.historical_dir / 'points.json')
    historical = load_bound(historical_ref)
    all_models = args.all_models or args.supplement_preparation is not None
    if all_models:
        models = {p['model'] for p in historical}
    points = []
    for row in historical:
        if row['model'] in models and (row['system'] in BASELINE_SYSTEMS or row['series'] == 'pd_current'
                                       or all_models and row['series'] == 'pd_previous'):
            row = hydrate_receipt_evidence(row)
            row['observation_kind'] = 'historical'
            row['historical_series'] = row['series']
            if row['system'] == 'pdblend':
                row['series'] = ('pd_pre_repair_previous_' if row['series'] == 'pd_previous'
                                 else 'pd_pre_repair_') + row['revision'][:12]
            points.append(row)
    historical_rows = list(points)
    supplement_rows, supplement_audits, supplement_binding = [], [], None
    supplement_failures, supplement_provenance = [], []
    if args.supplement_preparation:
        supplement_rows, supplement_audits, supplement_failures, supplement_provenance, supplement_binding = collect_supplements(
            args.supplement_preparation, args.supplement_attempt_root)
        historical_receipts = {(r['receipt_path'], r['receipt_sha256']) for r in historical_rows}
        if any((r['historical_receipt']['path'], r['historical_receipt']['sha256']) not in historical_receipts
               for r in supplement_binding['declarations']):
            raise ValueError('supplement declaration is not bound to this historical snapshot')
    points, baseline_selection = select_baseline_observations(points, supplement_rows)
    selected_refs = {p['receipt_path'] for p in points if p['system'] in BASELINE_SYSTEMS}
    for audit in supplement_audits:
        audit['selected'] = audit['receipt_path'] in selected_refs
    receipts = sorted({p.resolve() for directory in args.job_dir for p in directory.glob('attempt-*/session/windows/*/receipt.json')})
    failures, provenance, observed, incomplete_receipts = supplement_failures, supplement_provenance, set(), []
    for path in receipts:
        try:
            row, details, prov, point = read_repair(path)
        except (KeyError, FileNotFoundError) as exc:
            # Failed windows remain an explicit diagnostic, never a fabricated metric.
            incomplete_receipts.append(dict(receipt=binding(path), error=repr(exc)))
            continue
        key = point['name'], point['revision']
        if key not in allowed or point != allowed[key]:
            raise ValueError('receipt is not exactly one predeclared campaign point: ' + str(path))
        if key in observed:
            raise ValueError('multiple receipts for one source/point require explicit predeclared selection')
        observed.add(key)
        points.append(row); failures.extend(details); provenance.append(prov)
    # Review all planned candidate sources, even before their first receipt, so
    # a later frozen refresh uses precisely the same explicitly bound endpoints.
    old_refs = {json.dumps(p['evidence_source_manifest'], sort_keys=True): p['evidence_source_manifest']
                for p in points + historical_rows if p['observation_kind'] != 'repair'}
    if supplement_binding:
        for review in supplement_binding['source_reviews']:
            source = review['source_manifests'][1]
            old_refs[json.dumps(source, sort_keys=True)] = source
    candidate_refs = {json.dumps(p['source_manifest'], sort_keys=True): p['source_manifest'] for p in planned if p.get('repair_experiment', {}).get('arm') == 'candidate'}
    reviews, refused = [], []
    for new in candidate_refs.values():
        for old in old_refs.values():
            left, right = load_bound(old)['source_sha256'], load_bound(new)['source_sha256']
            try:
                value = prepare_compatibility(old, new)
            except ValueError as exc:
                refused.append(dict(left=old, right=new, error=str(exc)))
                continue
            path = out / 'measurement-compatibility' / (left[:12] + '--' + right[:12] + '.json')
            if path.exists():
                path = path.with_name(path.stem + '--' + digest([old, new])[:12] + '.json')
            write_new(path, value)
            reviews.append(load_compatibility(path))
    points.sort(key=lambda p: (p['model'], p['dataset'], p['rate_scale'], p['series']))
    for row in points + historical_rows + supplement_rows:
        row.update(configured_rate_rps=row['offered_rps'],
                   realized_arrival_rate_rps=row['offered_requests'] / row['comparison_identity']['duration_s'])
    pending = [dict(name=p['name'], revision=p['revision'], dataset=p['dataset'], rate_scale=p['scale'],
                    offered_rps=p['rate_rps'], configured_rate_rps=p['rate_rps'],
                    realized_arrival_rate_rps=realized_rate(p), arm=p['repair_experiment']['arm'])
               for p in planned if (p['name'], p['revision']) not in observed]
    result = analyze_points(points, measurement_compatibility=reviews)
    pairs = []
    for candidate in (p for p in points if p['observation_kind'] == 'repair'):
        for baseline in (p for p in points if p['system'] in BASELINE_SYSTEMS and scenario(p) == scenario(candidate)):
            errors = paired_errors(candidate, baseline, measurement_compatibility=reviews)
            ce, be = complete_energy(candidate), complete_energy(baseline)
            pairs.append(dict(candidate_revision=candidate['revision'], candidate_point=candidate['point_id'],
                dataset=candidate['dataset'], rate_scale=candidate['rate_scale'], offered_rps=candidate['offered_rps'],
                configured_rate_rps=candidate['configured_rate_rps'], realized_arrival_rate_rps=candidate['realized_arrival_rate_rps'],
                baseline=baseline['system'], baseline_revision=baseline['revision'],
                candidate_energy_kj=ce, baseline_energy_kj=be,
                saving_pct=100*(1-ce/be) if not errors and ce is not None and be is not None and be > 0 else None,
                baseline_success_pct=baseline['success_pct'], baseline_attainment_pct=baseline['slo_attainment_pct'],
                baseline_energy_missing=baseline['missing_energy_reason'], pairing_errors=errors,
                candidate_receipt=candidate['receipt_path'], baseline_receipt=baseline['receipt_path']))
    references = []
    for candidate in (p for p in points if p['observation_kind'] == 'repair'):
        for prior in (p for p in points if p['system'] == 'pdblend' and p['observation_kind'] == 'historical'
                      and scenario(p) == scenario(candidate)):
            errors = paired_errors(candidate, prior, measurement_compatibility=reviews)
            ce, pe = complete_energy(candidate), complete_energy(prior)
            references.append(dict(candidate_revision=candidate['revision'], reference_revision=prior['revision'],
                dataset=candidate['dataset'], rate_scale=candidate['rate_scale'],
                configured_rate_rps=candidate['configured_rate_rps'], realized_arrival_rate_rps=candidate['realized_arrival_rate_rps'],
                candidate_total_energy_kj=ce, reference_total_energy_kj=pe,
                saving_pct=100*(1-ce/pe) if not errors and ce is not None and pe is not None and pe > 0 else None,
                pairing_errors=errors, candidate_receipt=candidate['receipt_path'], reference_receipt=prior['receipt_path']))
    snapshot = dict(schema='pdblend-energy-repair-snapshot/v1', captured_at=captured,
        source_modified_at=max(stamp(Path(p).stat().st_mtime) for p in
            {r['receipt_path'] for r in points + historical_rows + supplement_rows}
            | {r['receipt_path'] for r in supplement_audits if 'receipt' in r}),
        historical_points=historical_ref, campaigns=[ref for ref, _ in campaigns],
        repair_preparations=repair_preparations,
        repair_completed_points=len(observed), planned_entries=len(planned),
        completed_by_revision=dict(Counter(p['revision'] for p in points if p['observation_kind'] == 'repair')),
        receipt_selection='All complete receipts in explicitly listed jobs; duplicate source/point refused; no performance selection.',
        historical_selection='Fixed historical points.json: four baselines including failed requests, plus pre-repair PD current revision.',
        pending=pending, incomplete_receipts=incomplete_receipts, refused_compatibility=refused,
        independent_repeats_available=False, raw_receipts_unchanged=True)
    baseline_rows = [p for p in points if p['system'] in BASELINE_SYSTEMS]
    baseline_conditions = {scenario(p) for p in baseline_rows}
    snapshot.update(baseline_supplements=supplement_binding,
        baseline_selection_rule='Keep complete frozen historical observation, else numeric first predeclared complete service-and-tail attempt; never select on SLO, frequency qualification, energy magnitude, or mtime.',
        baseline_selected_points=len(baseline_rows), baseline_conditions=len(baseline_conditions),
        baseline_complete_energy_points=sum(complete_energy(p) is not None for p in baseline_rows),
        baseline_supplement_selected_points=sum(p['observation_kind'] == 'baseline_supplement' for p in baseline_rows),
        baseline_supplement_observed_points=len(supplement_rows),
        baseline_energy_missing=[dict(system=p['system'], model=p['model'], dataset=p['dataset'],
            rate_scale=p['rate_scale'], receipt_path=p['receipt_path'], reason=p['missing_energy_reason'])
            for p in baseline_rows if complete_energy(p) is None])
    snapshot['rate_axis'] = 'realized_arrival_rate_rps = offered_requests / 150 s; configured_rate_rps retained separately; pairing uses unchanged trace/scale/settings'
    for name, value in [('points.json', points), ('snapshot.json', snapshot), ('dominance.json', result),
                        ('provenance.json', provenance), ('baseline-pairs.json', pairs),
                        ('historical-points.json', historical_rows), ('baseline-selection.json', baseline_selection),
                        ('supplement-observations.json', supplement_rows), ('supplement-attempts.json', supplement_audits)]:
        json_write(out / name, value)
    for name, value in [('points.csv', points), ('repair-points.csv', [p for p in points if p['observation_kind'] == 'repair']),
                        ('comparisons.csv', result['comparisons']), ('baseline-pairs.csv', pairs),
                        ('pd-reference-pairs.csv', references),
                        ('pending.csv', pending), ('request-failures.csv', failures),
                        ('baseline-selected.csv', baseline_rows), ('baseline-selection.csv', baseline_selection),
                        ('historical-points.csv', historical_rows), ('supplement-observations.csv', supplement_rows),
                        ('supplement-attempts.csv', supplement_audits)]:
        csv_write(out / name, value)
    draw(points, planned, snapshot, out)
    lines = ['# PDblend 修复结果冻结比较', '', f'快照时间：{captured}。已完成修复窗口 {len(observed)} 个；不同 revision 分开统计。', '',
        '能耗口径：八卡服务+尾部；成功率与联合 SLO attainment 分母为全部到达请求。四 baseline 保留失败点；缺能耗不补零、不宣称全 baseline 胜出。原始 receipt、计量兼容审查与配对错误均留存。', '',
        '图横轴为实测到达率 N/150 秒；配置 rate 单列保留，按原始 trace、scale、配置配对。', '',
        f"Baseline 覆盖 {len(baseline_conditions)} 个条件、{len(baseline_rows)} 个观测；完整服务+尾部能耗 {snapshot['baseline_complete_energy_points']} 个，其中预声明补跑替换 {snapshot['baseline_supplement_selected_points']} 个。缺失项见 snapshot.json，不将部分结果称为完整矩阵。", '',
        '补跑只在历史完整能耗缺失时启用，选择预声明 job 中数字序号最早的完整 attempt；不按能耗大小、SLO或频率验收择优。历史记录、所有补跑观测、逐 attempt 审计和最终选择分别保存在 historical-points、supplement-observations、supplement-attempts、baseline-selection 文件。补跑 revision 与测量资格独立保留，formal 资格不升级。', '',
        '| Revision / arm | 场景 | 配置 rate | 实测 N/150 | 总能耗 kJ | attainment | success | 测量验收 | 全目标状态 |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    by_receipt = {r['receipt_path']: r for r in result['comparisons']}
    for row in (p for p in points if p['observation_kind'] == 'repair'):
        total = 'NA' if row['total_energy_kj'] is None else f"{row['total_energy_kj']:.4f}"
        comp = by_receipt[row['receipt_path']]
        status = comp['status'] + ('；已知能耗目标未满足' if comp['energy_goal_met'] is False else '')
        lines.append(f"| {row['revision'][:12]} / {row['repair_arm']} | {row['dataset']} ×{row['rate_scale']:g} | {row['configured_rate_rps']:g} | {row['realized_arrival_rate_rps']:.6f} | {total} | {row['slo_attainment_pct']:.2f}% | {row['success_pct']:.2f}% | {row.get('measurement_evidence_valid', 'unknown')} | {status} |")
    lines.extend(['', '未完成的计划点见 pending.csv；它们包含未执行及已中断旧版本的剩余计划，不代表这些点都仍在排队。',
                  '逐 baseline 能耗、成功率、缺失项和配对原因见 baseline-pairs.csv。小于 3% 节能需要至少 3 次独立完整配对重复；当前没有重复证据。', '',
                  '![Energy](energy-vs-rate.png)', '', '![Attainment](slo-attainment-vs-rate.png)', '', '![Success](success-rate-vs-rate.png)', ''])
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(out), completed=len(observed), revisions=snapshot['completed_by_revision'],
                         compatibility_reviews=len(reviews), refused_compatibility=len(refused)), ensure_ascii=False))


if __name__ == '__main__':
    main()
