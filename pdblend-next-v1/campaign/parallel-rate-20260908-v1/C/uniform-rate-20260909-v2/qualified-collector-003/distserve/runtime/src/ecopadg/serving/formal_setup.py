"""Freeze the full formal target, then execute explicit complete paired groups.

Preparation is CPU-only. The manifest names mechanisms, mechanism_collection, calibration,
method_prepared, method_selection, protocol, corpus, model_dir, image_inspect,
engine_image, source_files, profile_files, campaign_root and restore. Runtime
configs are mapped as configs[dataset][load][system]; system is one of the five
baselines or pdblend. execute_groups lists dataset/load/seed objects. Every
selected group runs all six systems, while all 30 target cells remain in the
frozen expected_cells.json and reports, including unexecuted cells.

cell_limit_s (default 1800) and dynamic_cell_limit_s (default 4800) include
physical restoration and warm-up. Their selected-group sum must fit both
formal_budget_s and the existing campaign's authorized cumulative deadline.
The original start never moves; immutable extension evidence is frozen by prefix.
"""
import argparse
import asyncio
from copy import deepcopy
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
from .datasets import make_trace, make_dynamic_trace
from .evidence import (FORMAL_SEEDS, REQUIRED_MECHANISMS, baseline_gaps, common_capacity,
    evaluation_matrix, freeze_bundle, freeze_files, formal_freeze_gaps, matrix_gaps, sha256, validate_freeze)
from .method_selection import (check_preparation, valid_summary, VARIANTS,
                               calibration_priors, config_for_point)
from . import provenance


SYSTEMS = tuple(REQUIRED_MECHANISMS) + ('pdblend',)
DATASETS = ('alpaca', 'sharegpt', 'longbench')
LOADS = ('low', 'medium', 'near_saturation')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def policy(config):
    """Ignore output filenames; never ignore model, layout or policy settings."""
    return {k: v for k, v in config.items() if k not in ('journal', 'reported_warmup_seconds')}


def checked_protocol(value):
    expected = dict(model='Qwen2.5-14B-Instruct', static_requests_per_run=500,
        formal_paired_seeds=list(FORMAL_SEEDS), datasets=list(DATASETS),
        load_fractions=[.3, .6, .9], dynamic_duration_seconds=3600,
        energy_savings_ci95_lower_target=.05, joint_slo_difference_ci95_lower_target=-.01,
        formal_baselines=list(REQUIRED_MECHANISMS), max_model_len=8192,
        dynamic_phases=[dict(start_s=i*900, end_s=(i+1)*900, length_mix=mix, capacity_fraction=fraction)
            for i, (mix, fraction) in enumerate((([.7, .2, .1], .3), ([.2, .6, .2], .6),
                                                 ([.1, .2, .7], .9), ([.7, .2, .1], .3)))])
    if any(value.get(k) != v for k, v in expected.items()) or value.get('gpu', {}).get('count') != 8:
        raise ValueError('protocol differs from the complete fixed first-round target')
    if any(not isinstance(value.get(k), (float, int)) or not math.isfinite(value[k]) or value[k] <= 0
           for k in ('slo_ttft_s', 'slo_tpot_s')):
        raise ValueError('positive common formal SLO definitions required')
    return value


def checked_selection(prepared_path, selection_path):
    prepared, selected = read(prepared_path), read(selection_path)
    if check_preparation(prepared):
        raise ValueError('development source, profile or input fingerprints changed')
    if (selected.get('status') != 'development_selected' or selected.get('split') != 'development'
            or selected.get('formal_eligible') is not False or selected.get('selected_variant') not in VARIANTS
            or selected.get('expected_cells') != len(prepared['jobs'])
            or selected.get('completed_valid_cells') != len(prepared['jobs'])):
        raise ValueError('complete development-only method selection required')
    files = set(map(Path, prepared['files'])) | {Path(prepared_path), Path(selection_path)}
    measured = {}; jobs = {}
    for entry in prepared['jobs']:
        job = read(entry['path']); jobs[job['id']] = job
        expected_config = config_for_point(read(prepared['configurations'][job['method']]['config']), job['dataset'])
        if policy(read(job['config'])) != policy(expected_config):
            raise ValueError('development cell differs from its frozen dataset policy and historical prior')
        status_path = Path(job['out']) / 'selection.status.json'
        summary_path = Path(job['out']) / 'summary.json'
        status, summary = read(status_path), read(summary_path)
        if (not status.get('complete') or status.get('summary_sha256') != sha256(summary_path)
                or not valid_summary(summary, job) or selected['rows'].get(job['id']) != summary):
            raise ValueError('method choice differs from its paired measured runs')
        measured[job['id']] = summary
        files.update((status_path, summary_path))
    ratios = {v: {} for v in VARIANTS}; feasible = {v: True for v in VARIANTS}
    for group in prepared['groups']:
        rows = {jobs[j]['method']: measured[j] for j in group['jobs']}
        reference = rows[prepared['reference']]
        for variant in VARIANTS:
            row = rows[variant]
            feasible[variant] &= (row['slo_attainment'] >= prepared['slo_min'] and
                row['slo_attainment'] - reference['slo_attainment'] >= -prepared['max_slo_drop']-1e-12)
            ratios[variant].setdefault(group['dataset'], []).append(row['energy_j']/reference['energy_j'])
    scores = {v: statistics.mean(statistics.mean(values) for values in groups.values())
              for v, groups in ratios.items()}
    available = [v for v in VARIANTS if feasible[v]]
    winner = min(available, key=lambda v: (scores[v], VARIANTS.index(v))) if available else None
    if winner != selected['selected_variant']:
        raise ValueError('development winner was changed after measurement')
    return winner, read(prepared['configurations'][winner]['config']), files


def checked_calibration(path):
    from .calibration import implementation_sources, capacity_observation_error, verified_admission_boundary
    value = read(path)
    if not value.get('passed') or not value.get('source_unchanged') or value.get('failure'):
        raise ValueError('independent baseline calibration has not passed')
    capacities = common_capacity(value['results'])
    files = {Path(path)}; current = freeze_files(implementation_sources())
    for result in value['results']:
        if not result.get('passed'): continue
        upper=result.get('infeasible_upper_observation')
        if (result.get('failure') or not upper or upper.get('rate')!=result['infeasible_upper_rps']
                or any(o.get('measurement_valid') is False for o in result.get('observations',[]))):
            raise ValueError('calibration lacks a valid observed SLO-infeasible upper bracket')
        for label,proof in (('confirmation',result['confirmation']),('upper bracket',upper)):
            artifact=Path(proof['artifact']);actual=read(artifact)
            if any(proof.get(k)!=v for k,v in actual.items()):
                raise ValueError('calibration '+label+' differs from the actual measurement')
            source=artifact.parent.parent/'source.before.json';recorded=read(source)
            if recorded!=current:
                raise ValueError('calibration implementation fingerprints changed')
            files.update(map(Path,recorded))
            admission_upper=label=='upper bracket' and verified_admission_boundary(actual)
            if (capacity_observation_error(actual,allow_admission_rejection=admission_upper) or actual.get('gpu_count')!=8
                    or (not admission_upper and actual.get('generated_tokens')!=actual.get('expected_generated_tokens'))
                    or any(actual.get(k)!=result[k] for k in ('system','dataset'))
                    or actual.get('split')!='calibration'):
                raise ValueError('independent calibration '+label+' has invalid work or measurement evidence')
            if label=='upper bracket' and not admission_upper and actual['slo_attainment']>=.99:
                raise ValueError('calibration upper bracket is not an observed SLO failure')
            trace_path=Path(proof['trace']);trace=read(trace_path)
            if (trace.get('rate')!=proof['rate'] or trace.get('split')!='calibration'
                    or trace.get('dataset')!=result['dataset']
                    or len(trace.get('requests',[]))!=actual['n_expected']
                    or actual.get('trace_sha256')!=sha256(trace_path)):
                raise ValueError('calibration '+label+' rate or trace differs from its measurement')
            config_path=Path(result['config']);runtime_path=artifact.parent/'runtime_config.json'
            if policy(read(runtime_path))!=policy(read(config_path)):
                raise ValueError('calibrated baseline configuration changed')
            files.update((artifact,source,config_path,runtime_path,Path(proof['trace'])))
    return value['results'], capacities, files


def model_files(root):
    root = Path(root).resolve()
    if root != provenance.MODEL_ROOT.resolve():
        raise ValueError('formal engine uses the fixed canonical model directory')
    weights = set(read(root/'model.safetensors.index.json')['weight_map'].values())
    if not weights or any(Path(name).name != name for name in weights):
        raise ValueError('invalid model shard index')
    required = weights | {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if any(not (root/name).is_file() for name in required):
        raise ValueError('model shard or tokenizer file missing')
    return {p.resolve() for p in root.iterdir() if p.is_file() and
            (p.name in required or p.suffix in ('.json', '.safetensors', '.model', '.txt', '.tiktoken'))}


def certified_profile_files(configs, extras):
    files = {Path(p).resolve() for p in extras}
    for config in configs:
        for key in ('profiles', 'transfer_evidence', 'interconnect'):
            if config.get(key): files.add(Path(config[key]).resolve())
        for key, values in config.items():
            if key.endswith('_evidence') and isinstance(values, list):
                files.update(Path(p).resolve() for p in values)
    pending = list(files); seen = set()
    while pending:
        path = pending.pop()
        if path in seen: continue
        seen.add(path)
        if path.suffix != '.json': continue
        value = read(path)
        if not isinstance(value, dict): continue
        for source, digest in value.get('certification_artifacts', {}).items():
            source = Path(source).resolve()
            if sha256(source) != digest:
                raise ValueError('profile certification artifact changed')
            files.add(source); pending.append(source)
    digests = {sha256(p): p for p in files}
    for config in configs:
        for key in ('role_costs', 'topology_costs', 'frequency_costs', 'measured_capacities'):
            for cost in config.get(key, []):
                if cost.get('source_sha256') not in digests:
                    raise ValueError('missing frozen measurement source for '+key)
                raw = read(digests[cost['source_sha256']])
                if (not raw.get('complete') or raw.get('sampling_error') or raw.get('errors')
                        or (key in ('role_costs', 'topology_costs') and not raw.get('passed'))):
                    raise ValueError('incomplete or failed measurement source for '+key)
    return files


def generate(manifest_path, out):
    from .cell import verify_formal_config
    manifest_path = Path(manifest_path).resolve(); manifest = read(manifest_path)
    out = Path(out).resolve()
    if out.exists(): raise ValueError('refusing to overwrite formal evidence')
    protocol = checked_protocol(read(manifest['protocol']))
    mechanisms = read(manifest['mechanisms'])
    if baseline_gaps(mechanisms): raise ValueError('baseline mechanism gate is incomplete')
    from .evidence import checked_mechanism_collection
    if not manifest.get('mechanism_collection'):
        raise ValueError('the executed mechanism collector summary is required')
    collection_path=Path(manifest['mechanism_collection']).resolve()
    mechanism_files=checked_mechanism_collection(mechanisms,manifest['mechanisms'],collection_path)
    variant, chosen, method_files = checked_selection(manifest['method_prepared'], manifest['method_selection'])
    calibrated, capacities, calibration_files = checked_calibration(manifest['calibration'])
    image = manifest['engine_image']; inspection = read(manifest['image_inspect'])
    records = inspection if isinstance(inspection, list) else [inspection]
    if len(image) != 71 or not image.startswith('sha256:') or image not in {r.get('Id') for r in records}:
        raise ValueError('immutable engine image differs from Docker inspection')
    int(image[7:], 16)
    models = model_files(manifest['model_dir'])
    corpus_paths = {d: Path(manifest['corpus'])/(d+'.json') for d in DATASETS}
    corpora = {d: read(path) for d, path in corpus_paths.items()}
    corpus_manifest = read(protocol['corpus_manifest'])
    for dataset, corpus in corpora.items():
        if corpus_manifest['datasets'][dataset]['sha256'] != sha256(corpus_paths[dataset]):
            raise ValueError('formal corpus differs from the original protocol fingerprint')
        excluded = {r['request_shape_sha256'] for split in ('calibration', 'development') for r in corpus[split]}
        if (corpus.get('dataset') != dataset or any(len(corpus['formal'][str(s)]) != 500 for s in FORMAL_SEEDS)
                or any(r['request_shape_sha256'] in excluded for s in FORMAL_SEEDS for r in corpus['formal'][str(s)])):
            raise ValueError('exactly 500 seed-specific formal examples per dataset required')
    priors, prior_sources = calibration_priors(manifest['corpus'], DATASETS, calibrated)
    if chosen.get('output_priors') != priors or chosen.get('output_prior_sources') != prior_sources:
        raise ValueError('formal prediction histories differ from development and independent calibration')
    dynamic_prior_dataset = manifest.get('dynamic_prior_dataset')
    if dynamic_prior_dataset not in DATASETS:
        raise ValueError('dynamic_prior_dataset must select one shared independently calibrated history')
    configs = {}; config_paths = set()
    points = [(d, l) for d in DATASETS for l in LOADS] + [('dynamic', 'changing')]
    for dataset, load in points:
        prior_dataset = dynamic_prior_dataset if dataset == 'dynamic' else dataset
        mapping = manifest['configs'][dataset][load]
        if set(mapping) != set(SYSTEMS): raise ValueError('every point requires all five baselines and PDBlend')
        for system in SYSTEMS:
            path = Path(mapping[system]).resolve(); config = read(path); config_paths.add(path)
            expected = variant if system == 'pdblend' else system
            if config.get('strategy') != expected or any(config.get(k) != protocol[k] for k in ('slo_ttft_s', 'slo_tpot_s')):
                raise ValueError('formal variant or SLO differs from frozen selection/protocol')
            if sorted(config.get('node_gpus', [])) != list(range(8)):
                raise ValueError('formal strategies must own all eight GPU clocks')
            if system == 'mixed_dvfs' and config.get('dvfs') is False:
                raise ValueError('strong mixed_dvfs baseline must retain frequency optimization')
            if system == 'pdblend':
                if policy(config) != policy(config_for_point(chosen, prior_dataset)):
                    raise ValueError('PDBlend policy changed after development selection')
            else:
                eligible = [r for r in calibrated if r['system'] == system and r.get('passed')
                            and r['dataset'] == prior_dataset]
                best = [r for r in eligible if r['capacity_rps'] == max(
                    x['capacity_rps'] for x in eligible if x['dataset'] == r['dataset'])]
                if not any(policy(config) == policy(read(r['config'])) for r in best):
                    raise ValueError('formal baseline differs from its best independently calibrated configuration')
            if config.get('output_prior') != priors[prior_dataset]:
                raise ValueError('all strategies must share the independently calibrated dataset prior')
            if config.get('topology', {}).get('image', image) != image:
                raise ValueError('in-run replacements must use the frozen engine image')
            configs[(dataset, load, system)] = config
    profiles = certified_profile_files(configs.values(), manifest['profile_files'])
    for path in {Path(config['profiles']) for config in configs.values()}:
        profile = read(path)
        if (profile.get('engine_image') != image or profile.get('model') != protocol['model']
                or profile.get('status') != 'validated_envelope'):
            raise ValueError('formal profiles are not a validated same-model/image envelope')
    restore = deepcopy(manifest['restore']); restore['image'] = image
    restore.setdefault('retained_weights', None)
    ownership = Path(restore['ownership_root']).resolve()
    if not out.is_relative_to(ownership): raise ValueError('formal output outside explicit restoration ownership')
    for config in configs.values():
        topology = config.get('topology')
        if topology and not Path(topology['runtime_dir']).resolve().is_relative_to(ownership):
            raise ValueError('dynamic engines must belong to formal restoration ownership')
        if topology and read(topology['engine_template']) != read(restore['engine_template']):
            raise ValueError('initial restoration differs from the measured in-run engine template')
    expected = evaluation_matrix(capacities)
    traces = {}
    for cell in expected:
        dataset, seed = cell['dataset'], cell['seed']
        traces[(dataset, cell['load'], seed)] = make_trace(corpora[dataset]['formal'][str(seed)],
            cell['rate'], seed, dataset=dataset, split='formal', load=cell['load'])
    for seed in FORMAL_SEEDS:
        trace = make_dynamic_trace({d: corpora[d]['formal'][str(seed)] for d in DATASETS},
                                  capacities, protocol['dynamic_phases'], seed)
        traces[('dynamic', 'changing', seed)] = trace
        expected.append(dict(dataset='dynamic', load='changing', seed=seed, split='formal',
            n_requests=len(trace['requests']), trace_duration_s=3600))
    if matrix_gaps(expected): raise ValueError('full formal target matrix is incomplete')
    execute = [(g['dataset'], g['load'], g['seed']) for g in manifest['execute_groups']]
    if len(execute) != len(set(execute)) or not set(execute) <= set(traces):
        raise ValueError('execution groups must be unique complete target cells')
    limits = {key: manifest.get('dynamic_cell_limit_s', 4800) if key[0] == 'dynamic'
              else manifest.get('cell_limit_s', 1800) for key in traces}
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError('finite positive formal cell limits required')
    if any(limits[key] < 3600 for key in execute if key[0] == 'dynamic'):
        raise ValueError('dynamic cell limit cannot omit its 60-minute trace')
    budget = read_budget(manifest['campaign_root'])
    if not budget.get('started_s'): raise ValueError('original campaign must already be started')
    remaining = budget['remaining_s']
    upper = sum(limits[k]*len(SYSTEMS) for k in execute)
    allocation = manifest['formal_budget_s']
    if not isinstance(allocation, (int, float)) or not math.isfinite(allocation) or not 0 <= upper <= allocation <= remaining-60:
        raise ValueError('complete paired groups exceed explicit allocation or authorized node deadline')
    out.mkdir(parents=True)
    protocol_files = mechanism_files | {Path(p) for p in budget['revision_artifacts']} | {manifest_path, Path(manifest['protocol']), Path(manifest['mechanisms']),
        Path(protocol['corpus_manifest']), Path(restore['engine_template'])} | config_paths | method_files | calibration_files | set(corpus_paths.values())
    for entries in mechanisms.values():
        protocol_files.update(Path(proof['artifact']) for proof in entries.values())
    repository = Path(__file__).resolve().parents[3]
    sources = set((repository/'src').rglob('*.py')) | set((repository.parent/'vllm-pd-fork/vllm').rglob('*.py'))
    sources.add(repository.parent/'benchmarks/scripts/bench_vllm.py')
    sources.update(Path(p) for p in manifest['source_files'])
    trace_paths = []; jobs = []; groups = []; stages = []
    allowed = list(restore['initial_instances'])
    allowed.extend(i for config in configs.values() for i in config['instances'])
    for key, trace in traces.items():
        dataset, load, seed = key; group_id = f'{dataset}-{load}-seed{seed}'
        trace_path = out/'traces'/(group_id+'.json'); write(trace_path, trace); trace_paths.append(trace_path)
        group_jobs = []
        order = list(SYSTEMS); random.Random(seed).shuffle(order)
        for system in order:
            config = configs[(dataset, load, system)]
            config_path = out/'configs'/f'{dataset}-{load}-{system}.json'
            if not config_path.exists(): write(config_path, config); protocol_files.add(config_path)
            job = dict(id=group_id+'--'+system, system=system, dataset=dataset, load=load, seed=seed,
                config=str(config_path), trace=str(trace_path), freeze=str(out/'freeze.json'),
                mechanisms=str(Path(manifest['mechanisms']).resolve()), out=str(out/'runs'/(group_id+'--'+system)),
                restore=dict(restore, instances=config['instances'], initial_instances=allowed),
                timeout=manifest.get('request_timeout_s', 900), limit_s=limits[key])
            path = out/'jobs'/(job['id']+'.json'); write(path, job); protocol_files.add(path)
            jobs.append(str(path)); group_jobs.append(str(path))
        group = dict(id=group_id, jobs=group_jobs, limit_s=limits[key]*len(SYSTEMS),
            campaign_root=str(Path(manifest['campaign_root']).resolve()), freeze=str(out/'freeze.json'),
            mechanisms=str(Path(manifest['mechanisms']).resolve()))
        path = out/'groups'/(group_id+'.json'); write(path, group); protocol_files.add(path)
        groups.append(dict(dataset=dataset, load=load, seed=seed, path=str(path), selected=key in execute))
        if key in execute:
            stages.append(dict(name='formal-'+out.name+'-'+group_id, argv=[sys.executable, '-m',
                'ecopadg.serving.formal_setup', 'group', '--group', str(path)],
                limit_s=group['limit_s'], gpu=True, formal=True, evaluation_cell=list(key)))
    stages.sort(key=lambda stage: execute.index(tuple(stage['evaluation_cell'])))
    expected_path = out/'expected_cells.json'; write(expected_path, expected); protocol_files.add(expected_path)
    campaign = dict(output=str(Path(manifest['campaign_root']).resolve()), budget_s=budget['limit_s'],
        freeze=str(out/'freeze.json'), mechanisms=str(Path(manifest['mechanisms']).resolve()), stages=stages)
    write(out/'campaign.json', campaign); protocol_files.add(out/'campaign.json')
    prepared = dict(schema=1, status='frozen_not_executed', variant=variant, expected_cells=str(expected_path),
        groups=groups, jobs=jobs, freeze=str(out/'freeze.json'), mechanisms=campaign['mechanisms'],
        mechanism_collection=str(collection_path),
        campaign=str(out/'campaign.json'), out=str(out), stage_upper_bound_s=upper, selected_groups=len(execute),
        full_target_groups=len(traces), common_capacity=capacities,
        budget_at_preparation={k:budget[k] for k in ('original_started_s','original_limit_s','original_deadline_s',
            'limit_s','deadline_s','revision_seq','authorization_sha256','revision_sha256','revision_artifacts')},
        incomplete_target_verdict='evidence_insufficient; execution subsets never replace the complete target')
    write(out/'prepared.json', prepared); protocol_files.add(out/'prepared.json')
    freeze = freeze_bundle(dict(source=sources, model=models, image=[manifest['image_inspect']],
        profiles=profiles, traces=trace_paths, protocol=protocol_files), image)
    freeze['formal_evidence']=dict(mechanisms=str(Path(manifest['mechanisms']).resolve()),
        mechanism_collection=str(collection_path),expected_cells=str(expected_path.resolve()))
    for key, config in configs.items():
        path = out/'configs'/f'{key[0]}-{key[1]}-{key[2]}.json'
        verify_formal_config(SimpleNamespace(config=path, strategy=None, slo_ttft_s=None, slo_tpot_s=None), config, freeze)
    if formal_freeze_gaps(freeze): raise ValueError('formal freeze failed its consistency gate')
    write(out/'freeze.json', freeze)
    return prepared


async def run_job(path):
    from .calibration_setup import restore_layout
    from .cell import run_cell
    path = Path(path).resolve(); job = read(path); freeze = read(job['freeze'])
    if freeze['files'].get(str(path)) != sha256(path): raise ValueError('formal job changed')
    await restore_layout(job['restore'], Path(job['out']+'.preparation'))
    return await run_cell(SimpleNamespace(config=Path(job['config']), strategy=None, trace=Path(job['trace']),
        out=Path(job['out']), split='formal', dataset=job['dataset'], load=job['load'], seed=job['seed'],
        freeze=Path(job['freeze']), mechanisms=Path(job['mechanisms']), timeout=job['timeout'],
        slo_ttft_s=None, slo_tpot_s=None))


async def run_group(path):
    path = Path(path).resolve(); group = read(path); freeze = read(group['freeze'])
    if (freeze['files'].get(str(path)) != sha256(path) or baseline_gaps(read(group['mechanisms']))
            or formal_freeze_gaps(freeze)):
        raise ValueError('formal group gate closed')
    budget = read_budget(group['campaign_root'])
    remaining = budget['remaining_s']
    write(path.parent.parent/'budget-checks'/(group['id']+'-'+str(time.time_ns())+'.json'),
        dict(checked_s=time.time(),group=str(path),required_s=group['limit_s'],remaining_s=remaining,
            passed=group['limit_s']<=remaining-60,
            **{k:budget[k] for k in ('original_started_s','original_limit_s','original_deadline_s',
                'limit_s','deadline_s','revision_seq','authorization_sha256','revision_sha256','revision_artifacts')}))
    if group['limit_s'] > remaining-60:
        raise ValueError('insufficient remaining time to start a complete paired group')
    for job_path in group['jobs']:
        job = read(job_path)
        await asyncio.wait_for(run_job(job_path), timeout=job['limit_s'])


def report(prepared_path, out):
    from .report import collect, write_report
    prepared = read(prepared_path)
    summaries = [str(Path(read(p)['out'])/'summary.json') for p in prepared['jobs']
                 if (Path(read(p)['out'])/'summary.json').is_file()]
    manifest = dict(summaries=summaries, expected_cells=prepared['expected_cells'],
                    mechanisms=prepared['mechanisms'], freeze=prepared['freeze'])
    result = collect(manifest); write_report(result, Path(out))
    return result


def run(prepared_path):
    prepared = read(prepared_path); manifest = read(prepared['campaign'])
    campaign = Campaign(manifest['output'], manifest['budget_s'])
    try:
        if manifest['stages']:
            campaign.formal_gate(read(prepared['mechanisms']), read(prepared['freeze']))
        for stage in manifest['stages']:
            campaign.run(stage['name'], stage['argv'], stage['limit_s'], gpu=True)
    finally:
        campaign.close()
        report(prepared_path, Path(prepared['out'])/'reports'/str(time.time_ns()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cmd = sub.add_parser('generate'); cmd.add_argument('--manifest', type=Path, required=True); cmd.add_argument('--out', type=Path, required=True)
    cmd = sub.add_parser('group'); cmd.add_argument('--group', type=Path, required=True)
    cmd = sub.add_parser('run'); cmd.add_argument('--prepared', type=Path, required=True)
    cmd = sub.add_parser('report'); cmd.add_argument('--prepared', type=Path, required=True); cmd.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'generate':
        result = generate(args.manifest, args.out)
        print(json.dumps({k: result[k] for k in ('status', 'variant', 'selected_groups', 'full_target_groups', 'stage_upper_bound_s')}))
    elif args.action == 'group':
        with node_lease(): asyncio.run(run_group(args.group))
    elif args.action == 'run': run(args.prepared)
    else: print(json.dumps(report(args.prepared, args.out)['verdict']))


if __name__ == '__main__': main()
