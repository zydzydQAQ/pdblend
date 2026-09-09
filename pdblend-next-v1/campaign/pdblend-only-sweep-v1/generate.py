"""CPU-only PDBlend development traces; no calibration or execution entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re

MODELS = {'7b': 'Qwen2.5-7B-Instruct', '14b': 'Qwen2.5-14B-Instruct',
          '32b': 'Qwen2.5-32B-Instruct'}
DATASETS = ('alpaca', 'sharegpt', 'longbench')
BANDS = ('low', 'medium', 'high')
STRATEGIES = ('pdblend-greedy', 'pdblend-joint', 'pdblend-dynamic')


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def frozen(path, expected):
    path = Path(path).resolve()
    if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
        raise ValueError('explicit SHA256 required')
    if digest_file(path) != expected:
        raise ValueError('frozen artifact changed: ' + str(path))
    return path


def candidate_identity(spec):
    path = frozen(spec['manifest_path'], spec['manifest_sha256'])
    manifest = read(path)
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('candidate manifest requires a nonempty frozen file inventory')
    for relative, sha in files.items():
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent):
            raise ValueError('candidate file escapes its frozen release')
        frozen(target, sha)
    return dict(manifest_path=str(path), manifest_sha256=spec['manifest_sha256'],
                source_inventory_sha256=digest(files), frozen_file_count=len(files))


def load_pool(spec, *, model, dataset):
    """Select only development records; never select or export another split."""
    path = frozen(spec['path'], spec['sha256'])
    payload = read(path)
    if payload.get('dataset') != dataset:
        raise ValueError('pool dataset mismatch')
    if payload.get('split') not in (None, 'development'):
        raise ValueError('pool must not be a calibration or formal artifact')
    kind = spec['kind']
    if kind == 'development_corpus':
        manifest_path = frozen(spec['manifest_path'], spec['manifest_sha256'])
        manifest = read(manifest_path)
        if manifest.get('model') != MODELS[model]:
            raise ValueError('pool tokenizer/model identity mismatch')
        if manifest.get('datasets', {}).get(dataset, {}).get('sha256') != spec['sha256']:
            raise ValueError('corpus file is not bound by its split manifest')
        # Corpus files co-locate split arrays. Only this named array is used;
        # neither calibration nor held-out records enter any generated trace.
        records = payload.get('development')
        scope = 'complete_existing_development_split'
        note = 'fixed development subset reused across arrival seeds and rates; no new independent corpus records'
    elif kind == 'development_trace':
        if payload.get('split') != 'development':
            raise ValueError('input trace must explicitly be split=development')
        requests, prompts, shapes = (payload.get(k, []) for k in ('requests', 'prompts', 'source_shapes'))
        if not requests or len(requests) != len(prompts) or len(requests) != len(shapes):
            raise ValueError('complete aligned development trace required')
        records = [dict(prompt=p, input_tokens=r['prompt_len'], output_tokens=r['output_len'],
                        request_shape_sha256=s) for r, p, s in zip(requests, prompts, shapes)]
        scope = 'existing_development_trace_subset'
        note = 'repeat from existing dev pool; complete development corpus unavailable; no independent corpus claim'
    else:
        raise ValueError('only development_corpus or development_trace is allowed')
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty development records required')
    normalized = []
    for r in records:
        prompt = r.get('prompt')
        n_input, n_output = r.get('input_tokens'), r.get('output_tokens')
        shape = r.get('request_shape_sha256')
        if (not isinstance(prompt, list) or not prompt
                or any(type(t) is not int or t < 0 for t in prompt)
                or type(n_input) is not int or n_input != len(prompt)
                or type(n_output) is not int or not 1 <= n_output <= 512
                or n_input + n_output > 8192
                or not isinstance(shape, str) or not re.fullmatch('[a-f0-9]{64}', shape)):
            raise ValueError('invalid tokenized development record')
        normalized.append(dict(prompt=prompt, input_tokens=n_input, output_tokens=n_output,
                               request_shape_sha256=shape))
    return normalized, dict(path=str(path), sha256=spec['sha256'], kind=kind, scope=scope,
        manifest_path=spec.get('manifest_path'), manifest_sha256=spec.get('manifest_sha256'),
        available_records=len(normalized), unique_source_shapes=len({r['request_shape_sha256'] for r in normalized}),
        unique_prompt_payloads=len({digest(r['prompt']) for r in normalized}), reuse_note=note)


def sample_pool(records, n, seed, allow_repeat):
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    if n > len(order) and allow_repeat is not True:
        raise ValueError('insufficient development records; explicit allow_repeat_from_dev_pool required')
    selected = (order * math.ceil(n / len(order)))[:n]
    return [records[i] for i in selected], selected


def make_trace(records, rate, seed, *, model, dataset, band, pool, indices, sampling_seed):
    rng = random.Random(seed)
    arrival = 0.
    requests = []
    for i, record in enumerate(records):
        if i:
            arrival += rng.expovariate(rate)
        requests.append(dict(idx=i, arrival_s=arrival, prompt_len=record['input_tokens'],
                             output_len=record['output_tokens']))
    unique_prompts = len({digest(r['prompt']) for r in records})
    return dict(schema=2, model=model, dataset=dataset, split='development', load=band,
        seed=seed, arrival_seed=seed, sampling_seed=sampling_seed, rate=rate,
        duration_s=arrival, requests=requests, prompts=[r['prompt'] for r in records],
        source_shapes=[r['request_shape_sha256'] for r in records], source_pool_indices=indices,
        pool=pool, arrival_process='Poisson: random.Random(arrival_seed).expovariate(rate)',
        arrival_seed_independence='separate PRNG streams across seeds; paired unit-rate draws across rates',
        corpus_independence_across_arrival_seeds=False,
        unique_source_shapes=len({r['request_shape_sha256'] for r in records}),
        unique_prompt_payloads=unique_prompts, repeated_prompt_count=len(records)-unique_prompts,
        within_trace_resampling=len(set(indices)) < len(indices),
        repeat_from_existing_dev_pool=pool['kind']=='development_trace' or len(set(indices))<len(indices),
        formal_eligible=False, capacity_rps=None, saturation_verified=False,
        purpose='PDBlend-only exploratory development; actual span reported, no formal >=300 s claim')


def validate_spec(spec):
    if spec.get('split') != 'development' or spec.get('purpose') != 'pdblend_only_development':
        raise ValueError('only explicit PDBlend development scope is allowed')
    if set(spec['models']) != set(MODELS):
        raise ValueError('exactly the three model groups are required')
    if type(spec.get('n_requests')) is not int or not 64 <= spec['n_requests'] <= 128:
        raise ValueError('development cells require 64 to 128 requests')
    seeds = spec['arrival_seeds']
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('at least three distinct nonnegative integer arrival seeds required')
    if type(spec.get('sampling_seed')) is not int:
        raise ValueError('independent fixed content sampling seed required')
    candidate = candidate_identity(spec['candidate'])
    models = {}
    for model, entry in spec['models'].items():
        if set(entry['datasets']) != set(DATASETS):
            raise ValueError('each model requires the same three datasets')
        variants = {}
        for variant in entry['variants']:
            name = variant['name']
            if not re.fullmatch('[a-z0-9][a-z0-9_-]*', name) or name in variants:
                raise ValueError('unique safe PDB variant names required')
            if variant.get('kind') not in ('pdblend', 'pdblend_ablation'):
                raise ValueError('only PDBlend and its declared ablations are allowed')
            path = frozen(variant['config_path'], variant['config_sha256'])
            config = read(path)
            if config.get('strategy') not in STRATEGIES:
                raise ValueError('baseline strategy forbidden in this generator')
            if any(k in config for k in ('dataset', 'trace', 'rate', 'load', 'arrival_seed')):
                raise ValueError('model policy config cannot encode dataset or load identity')
            if (config.get('evaluation_protocol') != 'evaluation-v3'
                    or config.get('slo_ttft_s') != 5. or config.get('slo_tpot_s') != .1
                    or config.get('slo_attainment_target') != .9):
                raise ValueError('fixed evaluation-v3 planned SLO 5 s/.1 s/q=.9 required')
            if Path(config.get('controller_source_release', '')).resolve() != Path(candidate['manifest_path']).parent:
                raise ValueError('every model must use the same frozen host candidate')
            variants[name] = dict(kind=variant['kind'], strategy=config['strategy'],
                config_path=str(path), config_sha256=variant['config_sha256'],
                system='pdblend' if variant['kind']=='pdblend' else 'pdblend-ablation-'+name,
                description=variant.get('description', ''), engine_source_release=config.get('engine_source_release'))
        if not variants or sum(v['kind']=='pdblend' for v in variants.values()) != 1:
            raise ValueError('one PDB candidate plus optional PDB ablations required per model')
        datasets = {}
        for dataset, item in entry['datasets'].items():
            if set(item) != {'pool', 'rates', 'rate_rationale'} or set(item['rates']) != set(BANDS):
                raise ValueError('dataset entries contain only pool, absolute rates and rationale')
            rates = [item['rates'][band] for band in BANDS]
            if any(not isinstance(values, list) or not values for values in rates):
                raise ValueError('each exploratory band requires one or more absolute rates')
            flat = [r for values in rates for r in values]
            if (any(not positive(r) or not math.isfinite(spec['n_requests']/r) for r in flat)
                    or flat != sorted(set(flat))
                    or not isinstance(item['rate_rationale'], str) or not item['rate_rationale']):
                raise ValueError('strictly increasing finite positive rates and rationale required')
            records, pool = load_pool(item['pool'], model=model, dataset=dataset)
            selected, indices = sample_pool(records, spec['n_requests'], spec['sampling_seed'],
                                             spec.get('allow_repeat_from_dev_pool', False))
            datasets[dataset] = dict(item, records=selected, indices=indices, pool=pool)
        models[model] = dict(variants=variants, datasets=datasets)
    return candidate, models


def generate(spec, out):
    candidate, models = validate_spec(spec)  # Finish checks before writing.
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'traces').mkdir()
    (out/'groups').mkdir()
    manifest = dict(schema=1, split='development', purpose='pdblend_only_development',
        formal_eligible=False, execute_baselines=False, capacity_rps=None,
        load_band_qualification='ordered exploratory absolute rates, not calibrated low/medium/saturation claims',
        candidate=candidate, n_requests=spec['n_requests'], arrival_seeds=spec['arrival_seeds'],
        sampling_seed=spec['sampling_seed'], corpus_independence_across_seeds=False,
        groups=[], cells=[], generation_spec_sha256=digest(spec))
    for model, entry in models.items():
        for dataset, item in entry['datasets'].items():
            group = dict(group_id=model+'-'+dataset, model=model, dataset=dataset,
                split='development', candidate_manifest_sha256=candidate['manifest_sha256'],
                variants=entry['variants'], pool=item['pool'], rates=item['rates'],
                rate_rationale=item['rate_rationale'], capacity_rps=None, saturation_verified=False,
                formal_eligible=False, cells=[])
            for band in BANDS:
                for rate in item['rates'][band]:
                    for seed in spec['arrival_seeds']:
                        cell_id=f'{model}-{dataset}-{band}-r{rate:g}-s{seed}'
                        trace = make_trace(item['records'], rate, seed, model=model, dataset=dataset,
                            band=band, pool=item['pool'], indices=item['indices'], sampling_seed=spec['sampling_seed'])
                        path = out/'traces'/(cell_id+'.json')
                        content = encode(trace)
                        path.write_bytes(content)
                        for name, variant in entry['variants'].items():
                            cell = dict(cell_id=cell_id+'-'+name, model=model, dataset=dataset,
                                split='development', variant=name, system=variant['system'], strategy=variant['strategy'],
                                config_path=variant['config_path'], config_sha256=variant['config_sha256'],
                                candidate_manifest_sha256=candidate['manifest_sha256'],
                                load=band, rate_rps=rate, arrival_seed=seed, sampling_seed=spec['sampling_seed'],
                                trace_path=str(path), trace_sha256=hashlib.sha256(content).hexdigest(),
                                n_requests=len(trace['requests']), trace_duration_s=trace['duration_s'],
                                unique_source_shapes=trace['unique_source_shapes'],
                                unique_prompt_payloads=trace['unique_prompt_payloads'],
                                repeated_prompt_count=trace['repeated_prompt_count'],
                                within_trace_resampling=trace['within_trace_resampling'],
                                repeat_from_existing_dev_pool=trace['repeat_from_existing_dev_pool'],
                                corpus_independence_across_seeds=False, formal_eligible=False,
                                execution_status='not_run', capacity_rps=None, saturation_verified=False)
                            group['cells'].append(cell['cell_id'])
                            manifest['cells'].append(cell)
            (out/'groups'/(group['group_id']+'.json')).write_bytes(encode(group))
            manifest['groups'].append(group)
    # Catch any source/config/pool change while materializing traces.
    validate_spec(spec)
    (out/'spec.json').write_bytes(encode(spec))
    (out/'manifest.json').write_bytes(encode(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = generate(read(args.spec), args.out)
    print(json.dumps(dict(groups=len(result['groups']), cells=len(result['cells']),
        split=result['split'], execution_status='not_run', formal_eligible=False)))


if __name__ == '__main__':
    main()
