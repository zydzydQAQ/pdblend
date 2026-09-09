"""Paired development-only method search with explicit inputs and node budget.

``prepare`` performs no hardware work. It writes a Campaign manifest whose
cells restore the declared initial layout before running the common benchmark.
``run`` uses the existing Campaign and its effective authorized budget. Results
never satisfy formal acceptance, even if every development comparison passes.

The prepare manifest names ``base_config``, optional ``control_config``
(mixed_dvfs), ``corpus``, ``calibration`` (the completed independent calibration
summary), the existing ``campaign_root`` and the physical ``restore`` manifest.
It declares ``points`` as dataset/load/fraction objects, ``seeds`` (default
11/22), ``requests`` per point, ``slo_min``, ``max_slo_drop`` (fraction, not
percentage points), ``method_budget_s`` and ``cell_limit_s``. Each point can
override ``cell_limit_s``; the allocation covers the sum of its complete paired
seed groups rather than charging every point the largest limit. Every cell limit
includes restoration; complete paired groups must fit the declared allocation.

Example CLI: python -m ecopadg.serving.method_selection prepare
--manifest method.manifest.json --out /absolute/owned/root/method-search
Then run --prepared /absolute/owned/root/method-search/prepared.json, or pass
its generated campaign.json to the common campaign runner. The summarize
subcommand reads only the jobs explicitly sealed into that preparation.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

from .budget import read_budget
from .campaign import Campaign, node_lease
from .datasets import FORMAL_SEEDS, make_trace
from .evidence import common_capacity, freeze_files, sha256, validate_freeze
from ecopadg.measure.power import instant_power_verified


VARIANTS = ('pdblend-greedy', 'pdblend-joint', 'pdblend-dynamic')
ABLATIONS = ('full_frequency', 'mixed_only', 'fixed_pools')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def document(path):
    return json.loads(Path(path).read_text())


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_preparation(prepared):
    if prepared.get('seal') != seal({k: v for k, v in prepared.items() if k != 'seal'}):
        return ['prepared experiment manifest changed']
    changed = validate_freeze(prepared['files'])
    reasons = ['source, profile or input changed: ' + p for p in changed]
    source_files = {str(p.resolve()) for root in prepared.get('source_roots', []) for p in Path(root).rglob('*.py')}
    if source_files != set(prepared.get('source_files', [])):
        reasons.append('source file inventory changed')
    return reasons


def configurations(base, control=None):
    if {i.get('role', 'mixed') for i in base['instances']} != {'mixed', 'prefill', 'decode'}:
        raise ValueError('candidate configuration must declare an initial three-pool layout')
    result = {}
    for variant in VARIANTS:
        config = deepcopy(base)
        config.update(strategy=variant, allow_pd=True, dvfs=True, node_gpus=list(range(8)),
                      dynamic_pools=variant == 'pdblend-dynamic')
        if variant != 'pdblend-dynamic':
            config['slow_topology'] = False
        result[variant] = dict(config=config, candidate=True, variant=variant, ablation=None)
        for ablation in ABLATIONS:
            altered = deepcopy(config)
            if ablation == 'full_frequency':
                altered.update(dvfs=False, park_idle=False)
            elif ablation == 'mixed_only':
                altered.update(allow_pd=False, dynamic_pools=False, slow_topology=False)
                for instance in altered['instances']:
                    instance['role'] = 'mixed'
            else:
                altered.update(dynamic_pools=False, slow_topology=False)
            result[variant + '.' + ablation] = dict(config=altered, candidate=False,
                                                   variant=variant, ablation=ablation)
    if control is not None:
        if control.get('strategy') != 'mixed_dvfs' or any(
                i.get('role', 'mixed') != 'mixed' for i in control['instances']):
            raise ValueError('strong control must explicitly configure mixed_dvfs on mixed instances')
        if any(control.get(k) != base.get(k) for k in ('profiles', 'slo_ttft_s', 'slo_tpot_s')):
            raise ValueError('all methods require identical profiles and SLO definitions')
        if control.get('dvfs') is False:
            raise ValueError('mixed_dvfs control must enable its independent frequency policy')
        result['mixed_dvfs'] = dict(config=deepcopy(control), candidate=False,
                                   variant='mixed_dvfs', ablation=None)
        result['mixed_dvfs']['config']['node_gpus'] = list(range(8))
    if any(not item['config'].get('manage_clocks', True) for item in result.values()):
        raise ValueError('development energy search requires the hardware clock owner')
    for item in result.values():
        if item['config'].get('power_mode', 'instant') != 'instant':
            raise ValueError('development method selection requires instant power')
        item['config']['power_mode'] = 'instant'
    return result


def artifact_paths(config):
    paths = set()
    for key in ('profiles', 'transfer_evidence', 'interconnect'):
        if config.get(key):
            path = Path(config[key]).resolve()
            paths.add(path)
            if key != 'interconnect':
                proof = document(path)
                paths.update(Path(p).resolve() for p in proof.get('certification_artifacts', {}))
    topology = config.get('topology') or {}
    if topology.get('engine_template'):
        paths.add(Path(topology['engine_template']).resolve())
    for key, value in config.items():
        if key.endswith('_evidence') and isinstance(value, list):
            paths.update(Path(p).resolve() for p in value)
    return paths


def calibration_priors(corpus, datasets, results):
    """Freeze one causal initialization per dataset, shared by every policy."""
    priors = {}; sources = {}
    for dataset in sorted(set(datasets)):
        path = (Path(corpus) / (dataset + '.json')).resolve()
        values = [r['output_tokens'] for r in document(path)['calibration']]
        if not values or any(type(v) is not int or v < 1 for v in values):
            raise ValueError('positive completed calibration outputs required for the historical prior')
        values.sort()
        prior = values[math.ceil(.9 * len(values)) - 1]
        calibrated = [r for r in results if r.get('passed') and r['dataset'] == dataset]
        if not calibrated or any(document(r['config']).get('output_prior') != prior for r in calibrated):
            raise ValueError('historical prior differs from the independently calibrated baseline configuration')
        priors[dataset] = prior
        sources[dataset] = dict(corpus=str(path), sha256=sha256(path), split='calibration',
            quantile=.9, observations=len(values),
            configurations=freeze_files(Path(r['config']).resolve() for r in calibrated))
    return priors, sources


def config_for_point(config, dataset):
    """Materialize the predeclared historical prior without changing policy."""
    result = deepcopy(config)
    result['output_prior'] = config['output_priors'][dataset]
    return result


def prepare(manifest_path, out):
    manifest_path = Path(manifest_path).resolve()
    manifest = document(manifest_path)
    out = Path(out).resolve()
    seeds = manifest.get('seeds', [11, 22])
    if (not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds)
            or set(seeds).intersection(FORMAL_SEEDS)):
        raise ValueError('unique development seeds must exclude 101, 202 and 303')
    slo_min = manifest['slo_min']
    max_drop = manifest['max_slo_drop']
    if not (math.isfinite(slo_min) and 0 < slo_min <= 1
            and math.isfinite(max_drop) and 0 <= max_drop <= 1):
        raise ValueError('explicit valid development SLO threshold and maximum control delta required')
    count = manifest.get('requests', 64)
    if type(count) is not int or count < 1:
        raise ValueError('positive development request count required')
    points = manifest['points']
    if (not points or len({(p['dataset'], p['load']) for p in points}) != len(points)
            or any(not 0 < p['fraction'] <= 1 or p['dataset'] not in ('alpaca', 'sharegpt', 'longbench')
                   or not p['load'] or any(not (c.isalnum() or c in '_-') for c in p['load']) for p in points)):
        raise ValueError('distinct explicit dataset/load points and capacity fractions required')
    calibration_path = Path(manifest['calibration']).resolve()
    calibration = document(calibration_path)
    if not calibration.get('passed') or not calibration.get('source_unchanged'):
        raise ValueError('independent capacity calibration has not passed')
    capacities = common_capacity(calibration['results'], target=manifest.get('capacity_slo_min', .99))
    if manifest.get('common_capacity') is not None and manifest['common_capacity'] != capacities:
        raise ValueError('declared common capacity differs from independent baseline evidence')
    base_path = Path(manifest['base_config']).resolve()
    base = document(base_path)
    control_path = Path(manifest['control_config']).resolve() if manifest.get('control_config') else None
    configs = configurations(base, document(control_path) if control_path else None)
    priors, prior_sources = calibration_priors(manifest['corpus'], [p['dataset'] for p in points], calibration['results'])
    for entry in configs.values():
        config = entry['config']
        for key, expected in (('output_priors', priors), ('output_prior_sources', prior_sources)):
            if key in config and config[key] != expected:
                raise ValueError('development prior mapping differs from frozen calibration histories')
            config[key] = deepcopy(expected)
        # The template default is unused by cells; every cell is materialized
        # with its dataset's prior below, including controls and ablations.
        config['output_prior'] = priors[points[0]['dataset']]
    budget = read_budget(manifest['campaign_root'])
    allocation = manifest['method_budget_s']
    cell_limit = manifest.get('cell_limit_s', 1800)
    point_limits = [point.get('cell_limit_s', cell_limit) for point in points]
    request_timeout = manifest.get('request_timeout_s', 120)
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
           for x in (allocation, cell_limit, request_timeout, *point_limits)):
        raise ValueError('finite positive method budget and cell/request timeouts required')
    if not budget.get('started_s'):
        raise ValueError('method search must reuse the already-started authorized campaign budget')
    remaining = budget['remaining_s']
    stage_upper_bound = len(configs)*len(seeds)*sum(point_limits)
    if allocation > remaining - 60 or stage_upper_bound > allocation:
        raise ValueError('complete paired method group upper bounds exceed the explicit remaining allocation')
    restore = deepcopy(manifest['restore'])
    restore.setdefault('retained_weights', None)
    if not restore.get('ownership_root') or not restore.get('initial_instances'):
        raise ValueError('explicit physical ownership root and allowed initial instances required')
    if not str(out).startswith(str(Path(restore['ownership_root']).resolve()) + '/'):
        raise ValueError('experiment output must be inside the restoration ownership root')
    for entry in configs.values():
        topology = entry['config'].get('topology')
        if topology and not Path(topology['runtime_dir']).resolve().is_relative_to(
                Path(restore['ownership_root']).resolve()):
            raise ValueError('dynamic runtime directory must belong to the restoration ownership root')
    files = {manifest_path, calibration_path, base_path, Path(restore['engine_template']).resolve()}
    for source in prior_sources.values():
        files.add(Path(source['corpus']))
        files.update(Path(p) for p in source['configurations'])
    for calibration_result in calibration['results']:
        proof = calibration_result.get('confirmation') or {}
        if proof.get('artifact'):
            path = Path(proof['artifact']).resolve()
            # Confirmations add rate metadata without rewriting the original
            # common measurement summary; its existing fields must all agree.
            actual = document(path)
            if any(actual.get(k) != proof.get(k) for k in actual):
                raise ValueError('capacity confirmation differs from its measured artifact')
            files.add(path)
    if control_path:
        files.add(control_path)
    # Include the shared client, measurement and serving code, plus any
    # explicitly supplied fork files; no historical result directories scanned.
    repository = Path(__file__).resolve().parents[3]
    source_roots = [repository / 'src']
    fork = repository.parent / 'vllm-pd-fork/vllm'
    if fork.is_dir():
        source_roots.append(fork)
    source_files = {p.resolve() for root in source_roots for p in root.rglob('*.py')}
    files.update(source_files)
    client = repository.parent / 'benchmarks/scripts/bench_vllm.py'
    if client.is_file():
        files.add(client)
    files.update(Path(p).resolve() for p in manifest.get('source_files', []))
    for entry in configs.values():
        files.update(artifact_paths(entry['config']))
    out.mkdir(parents=True, exist_ok=False)
    for name, entry in configs.items():
        path = out / (name + '.config.json')
        write_json(path, entry.pop('config'))
        entry['config'] = str(path)
        files.add(path)
    reference = 'mixed_dvfs' if control_path else 'pdblend-joint.full_frequency'
    groups = []
    jobs = []
    stages = []
    for point, point_limit in zip(points, point_limits):
        corpus_path = Path(manifest['corpus']).resolve() / (point['dataset'] + '.json')
        corpus = document(corpus_path)
        files.add(corpus_path)
        records = corpus['development']
        if len(records) < count:
            raise ValueError('request budget exceeds independent development examples')
        point_configs = {}
        for name, entry in configs.items():
            path = out / (point['dataset'] + '.' + name + '.config.json')
            config = config_for_point(document(entry['config']), point['dataset'])
            write_json(path, config)
            files.add(path)
            point_configs[name] = str(path)
        for seed in seeds:
            selected = random.Random(seed).sample(records, count)
            rate = capacities[point['dataset']] * point['fraction']
            trace = make_trace(selected, rate, seed, dataset=point['dataset'],
                               split='development', load=point['load'])
            group = f"{point['dataset']}-{point['load']}-seed{seed}"
            trace_path = out / (group + '.trace.json')
            write_json(trace_path, trace)
            files.add(trace_path)
            group_jobs = []
            # Paired random order limits a systematic warm-node/order advantage.
            order = list(configs)
            random.Random(seed + len(groups)).shuffle(order)
            for name in order:
                job_id = group + '--' + name
                config = document(point_configs[name])
                layout = dict(restore, instances=config['instances'])
                # Explicit static IDs from any planned method are also owned;
                # dynamic IDs still need the adapter's configuration-root proof.
                allowed = {i.get('id', i.get('instance_id')): i for i in restore['initial_instances']}
                for item in configs.values():
                    for instance in document(item['config'])['instances']:
                        allowed.setdefault(instance.get('id', instance.get('instance_id')), instance)
                layout['initial_instances'] = list(allowed.values())
                job = dict(id=job_id, method=name, **configs[name], group=group,
                           dataset=point['dataset'], load=point['load'], seed=seed,
                           trace=str(trace_path), trace_sha256=sha256(trace_path),
                           n_expected=count, expected_generated_tokens=sum(r['output_len'] for r in trace['requests']),
                           out=str(out / 'runs' / job_id), restore=layout,
                           request_timeout_s=request_timeout, cell_limit_s=point_limit,
                           prepared=str(out / 'prepared.json'))
                job['config'] = point_configs[name]
                job_path = out / (job_id + '.job.json')
                write_json(job_path, job)
                files.add(job_path)
                jobs.append(dict(id=job_id, path=str(job_path)))
                group_jobs.append(job_id)
                stages.append(dict(name='method-' + seal(str(out))[:10] + '-' + job_id, argv=[sys.executable, '-m',
                    'ecopadg.serving.method_selection', 'cell', '--job', str(job_path)],
                    limit_s=point_limit, gpu=True))
            groups.append(dict(id=group, dataset=point['dataset'], load=point['load'],
                               seed=seed, rate=rate, cell_limit_s=point_limit, jobs=group_jobs))
    campaign_path = out / 'campaign.json'
    write_json(campaign_path, dict(output=str(Path(manifest['campaign_root']).resolve()),
        budget_s=budget['limit_s'], stages=stages))
    files.add(campaign_path)
    prepared = dict(schema=1, split='development', formal_eligible=False, reference=reference,
        slo_min=slo_min, max_slo_drop=max_drop, configurations=configs, groups=groups, jobs=jobs,
        output_priors=priors, output_prior_sources=prior_sources,
        common_capacity=capacities, campaign=str(campaign_path), out=str(out),
        stage_upper_bound_s=stage_upper_bound, method_budget_s=allocation,
        files=freeze_files(files), source_roots=list(map(str, source_roots)),
        source_files=sorted(map(str, source_files)),
        selection_rule='minimum dataset-equal mean paired energy ratio among candidates satisfying every run SLO constraint')
    prepared['seal'] = seal(prepared)
    write_json(out / 'prepared.json', prepared)
    summarize(out / 'prepared.json')
    return prepared


def valid_summary(summary, job):
    if any(summary.get(k) != job[k] for k in ('dataset', 'load', 'seed', 'trace_sha256',
                                             'n_expected', 'expected_generated_tokens')):
        return False
    return (summary.get('split') == 'development' and summary.get('formal_eligible') is False
        and summary.get('variant') == job['variant'] and summary.get('validity') == 'ok'
        and summary.get('completed') == job['n_expected']
        and summary.get('generated_tokens') == job['expected_generated_tokens']
        and summary.get('gpu_count') == 8 and summary.get('measurement_schema') == 2
        and instant_power_verified(summary)
        and isinstance(summary.get('energy_j'), (int, float))
        and math.isfinite(summary['energy_j']) and summary['energy_j'] > 0
        and isinstance(summary.get('slo_attainment'), (int, float))
        and math.isfinite(summary['slo_attainment']) and 0 <= summary['slo_attainment'] <= 1)


def summarize(prepared_path):
    prepared = document(prepared_path)
    reasons = check_preparation(prepared)
    rows = {}
    jobs = {}
    for entry in prepared['jobs']:
        job = document(entry['path'])
        jobs[job['id']] = job
        path = Path(job['out']) / 'selection.status.json'
        if not path.is_file():
            reasons.append('unfinished cell: ' + job['id'])
            continue
        status = document(path)
        summary_path = Path(job['out']) / 'summary.json'
        if (status.get('complete') is not True or not summary_path.is_file()
                or status.get('summary_sha256') != sha256(summary_path)):
            reasons.append('failed or changed cell: ' + job['id'])
            continue
        summary = document(summary_path)
        if not valid_summary(summary, job):
            reasons.append('invalid or unequal output work: ' + job['id'])
            continue
        rows[job['id']] = summary
    pairs = []
    feasible = {variant: True for variant in VARIANTS}
    ratios = {variant: {} for variant in VARIANTS}
    for group in prepared['groups']:
        by_method = {jobs[j]['method']: rows[j] for j in group['jobs'] if j in rows}
        complete = len(by_method) == len(prepared['configurations'])
        reference = by_method.get(prepared['reference'])
        detail = dict(group=group['id'], complete=complete, comparisons={})
        if reference is not None:
            for name, row in by_method.items():
                delta = row['slo_attainment'] - reference['slo_attainment']
                allowed = row['slo_attainment'] >= prepared['slo_min'] and delta >= -prepared['max_slo_drop'] - 1e-12
                ratio = row['energy_j'] / reference['energy_j']
                detail['comparisons'][name] = dict(energy_ratio=ratio, energy_saving=1-ratio,
                    slo_delta=delta, feasible=allowed, energy_j=row['energy_j'], slo_attainment=row['slo_attainment'])
                if name in VARIANTS:
                    feasible[name] &= allowed
                    ratios[name].setdefault(group['dataset'], []).append(ratio)
        pairs.append(detail)
    scores = {variant: statistics.mean(statistics.mean(values) for values in by_dataset.values())
              for variant, by_dataset in ratios.items() if by_dataset}
    selected = None
    if not reasons:
        available = [variant for variant in VARIANTS if feasible[variant] and variant in scores]
        if available:
            selected = min(available, key=lambda variant: (scores[variant], VARIANTS.index(variant)))
        else:
            reasons.append('no candidate satisfies every explicit development SLO constraint')
    result = dict(split='development', formal_eligible=False,
        status='development_selected' if selected else 'evidence_insufficient',
        selected_variant=selected, reference=prepared['reference'], dataset_equal_energy_ratios=scores,
        candidate_feasible=feasible, improves_development_control=bool(selected and scores[selected] < 1),
        reason='best measured feasible development candidate; this is not formal acceptance' if selected
            else 'no method selected from incomplete, changed, unequal-work or SLO-infeasible evidence',
        reasons=reasons, pairs=pairs, rows=rows, expected_cells=len(jobs), completed_valid_cells=len(rows))
    write_json(Path(prepared['out']) / 'selection.json', result)
    return result


async def run_job(job_path):
    from .calibration_setup import restore_layout
    from .cell import run_cell
    job = document(job_path)
    prepared = document(job['prepared'])
    status = dict(complete=False, split='development', formal_eligible=False)
    out = Path(job['out'])
    try:
        changes = check_preparation(prepared)
        if changes:
            raise RuntimeError('; '.join(changes))
        await restore_layout(job['restore'], out.with_name(out.name + '.preparation'))
        changes = check_preparation(prepared)
        if changes:
            raise RuntimeError('; '.join(changes))
        options = SimpleNamespace(config=Path(job['config']), strategy=None, trace=Path(job['trace']),
            out=out, split='development', dataset=job['dataset'], load=job['load'], seed=job['seed'],
            freeze=None, mechanisms=None, timeout=job['request_timeout_s'], slo_ttft_s=None, slo_tpot_s=None)
        summary = await run_cell(options)
        changes = check_preparation(prepared)
        if changes or not valid_summary(summary, job):
            raise RuntimeError('; '.join(changes) or 'incomplete or inconsistent development work')
        status.update(complete=True, summary_sha256=sha256(out / 'summary.json'))
        return summary
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'selection.status.json', status)
        summarize(job['prepared'])


def run(prepared_path):
    prepared = document(prepared_path)
    if check_preparation(prepared):
        return summarize(prepared_path)
    manifest = document(prepared['campaign'])
    campaign = Campaign(manifest['output'], manifest['budget_s'])
    try:
        for stage in manifest['stages']:
            campaign.run(stage['name'], stage['argv'], stage['limit_s'], gpu=True)
    finally:
        campaign.close()
        result = summarize(prepared_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--manifest', type=Path, required=True)
    command.add_argument('--out', type=Path, required=True)
    for action in ('run', 'summarize'):
        command = sub.add_parser(action)
        command.add_argument('--prepared', type=Path, required=True)
    command = sub.add_parser('cell')
    command.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.manifest, args.out)
        print(json.dumps(dict(campaign=result['campaign'], cells=len(result['jobs']), split='development')))
    elif args.action == 'cell':
        with node_lease():
            asyncio.run(run_job(args.job))
    else:
        result = run(args.prepared) if args.action == 'run' else summarize(args.prepared)
        print(json.dumps({k: result[k] for k in ('status', 'selected_variant', 'completed_valid_cells', 'expected_cells')}))


if __name__ == '__main__':
    main()
