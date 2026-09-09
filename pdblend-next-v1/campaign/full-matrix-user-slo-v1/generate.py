"""CPU-only paired development traces with user-specified dataset SLOs.

The default CLI materializes the selected matrix; use --plan-only for its small
declaration, --anchors for nine seed701 anchor traces, or filters for one cell.
No execution, calibration, output truncation, or held-out selection exists.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re


MODELS=('7b','14b','32b')
DATASETS=('alpaca','sharegpt','longbench')
SLOS={'alpaca':{'ttft_s':1.,'tpot_s':.1,'attainment_target':.9},
      'sharegpt':{'ttft_s':5.,'tpot_s':.15,'attainment_target':.9},
      'longbench':{'ttft_s':15.,'tpot_s':.2,'attainment_target':.9}}
MULTIPLIERS=('0.05','0.10','0.20','0.25','0.30','0.40','0.50','0.60','0.70',
             '0.75','0.80','0.85','0.90','0.95','1','1.10','1.20','1.5','2')
SEEDS=(701,1701,2701)
PROTOCOL='per-dataset-slo-v1'


def require(condition,reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen(path,expected):
    path=Path(path).resolve()
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected),'explicit source SHA256 required')
    require(file_digest(path) == expected,'frozen source changed: '+str(path))
    return path


def number_text(value):
    return format(Decimal(str(value)).normalize(),'f')


def unit_arrivals(n,seed):
    rng=random.Random(seed);arrival=0.;result=[arrival]
    for _ in range(1,n):
        arrival+=rng.expovariate(1.)
        result.append(arrival)
    return result


def required_count(rate,seeds,min_requests,min_span):
    """Equal N/content across seeds of this rate; do not compress Poisson time."""
    counts=[]
    for seed in seeds:
        rng=random.Random(seed);n=1;arrival=0.
        while n < min_requests or arrival/rate < min_span:
            arrival+=rng.expovariate(1.);n+=1
            require(n <= 2_000_000,'requested rate exceeds bounded materialization; declare a separate larger campaign')
        counts.append(n)
    return max(counts)


def load_spec(spec):
    require(spec.get('protocol_id') == PROTOCOL and spec.get('measurement_schema') == 3,'wrong protocol/schema')
    require(spec.get('split') == 'development' and spec.get('purpose') == 'pdblend_only_development',
            'only explicit PDB development scope allowed')
    require(spec.get('formal_eligible') is False and spec.get('execute_baselines') is False,'formal/baseline execution forbidden')
    require(spec.get('slo_by_dataset') == SLOS,'SLO differs from current user declaration')
    require(spec.get('arrival_seeds') == list(SEEDS),'required paired arrival seeds differ')
    require(type(spec.get('sampling_seed')) is int and spec.get('allow_repeat_from_dev_pool') is True,
            'explicit deterministic content seed and resampling permission required')
    require(type(spec.get('min_requests')) is int and spec['min_requests'] >= 1000
        and type(spec.get('min_planned_arrival_span_s')) in (int,float)
        and math.isfinite(spec['min_planned_arrival_span_s']) and spec['min_planned_arrival_span_s'] >= 300,
        'long cells require at least1000 requests and at least300s planned arrival span')
    require({Decimal(str(x)) for x in spec['rate_multipliers']} == {Decimal(x) for x in MULTIPLIERS}
        and len(spec['rate_multipliers']) == len(MULTIPLIERS),'undeclared rate grid; extend in a new manifest')
    require(set(spec['models']) == set(MODELS),'three model groups required')
    origin=spec['source_development_spec']
    frozen(origin['path'],origin['sha256'])
    old_spec=read(origin['path'])
    loader=spec['development_pool_reader']
    path=frozen(loader['path'],loader['sha256'])
    module_spec=importlib.util.spec_from_file_location('_frozen_original_development_pool_reader',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    groups={}
    for model in MODELS:
        entry=spec['models'][model]
        require(set(entry['datasets']) == set(DATASETS),'three datasets required per model')
        policy=entry['policy_reference'];config_path=frozen(policy['path'],policy['sha256'])
        config=read(config_path)
        require(config.get('strategy') in ('pdblend-greedy','pdblend-joint','pdblend-dynamic'),
                'non-PDB strategy forbidden')
        require(not any(key in config for key in ('dataset','trace','rate','load','arrival_seed')),
                'model policy must not encode dataset/load identity')
        for dataset in DATASETS:
            ds=entry['datasets'][dataset]
            anchor=ds['anchor_rps']
            require(type(anchor) in (float,int) and math.isfinite(anchor) and anchor > 0,'invalid absolute rate anchor')
            old=old_spec['models'][model]['datasets'][dataset]
            require(ds['pool'] == old['pool'],'pool must come from frozen development declaration')
            require(anchor == max(old['rates']['medium']),'anchor must equal declared old pilot absolute anchor')
            require(ds['anchor_semantics'] == 'old_pilot_absolute_rate; not measured mixed capacity or saturation',
                    'anchor cannot be called a measured capacity')
            records,pool=module.load_pool(ds['pool'],model=model,dataset=dataset)
            order=list(range(len(records)));random.Random(spec['sampling_seed']).shuffle(order)
            record_digests=[digest(dict(prompt=r['prompt'],input_tokens=r['input_tokens'],
                                       output_tokens=r['output_tokens'])) for r in records]
            groups[(model,dataset)]=dict(records=records,pool=pool,order=order,record_digests=record_digests,
                anchor=Decimal(str(anchor)),policy=policy,strategy=config['strategy'])
    return groups


def selected_indices(order,n):
    return (order*math.ceil(n/len(order)))[:n]


def cell_metadata(spec,model,dataset,multiplier,seed,group,n,indices,arrivals):
    multiplier=Decimal(str(multiplier));rate=group['anchor']*multiplier
    rate_float=float(rate);span=arrivals[-1]/rate_float
    require(n >= spec['min_requests'] and span >= spec['min_planned_arrival_span_s'],'long-trace minimum not met')
    cell_id=f'{model}-{dataset}-m{number_text(multiplier)}-s{seed}'
    content=digest([group['record_digests'][i] for i in indices])
    return dict(cell_id=cell_id,model=model,dataset=dataset,protocol_id=PROTOCOL,measurement_schema=3,
        split='development',system='pdblend',variant='candidate',strategy=group['strategy'],
        load='declared_absolute_rate',arrival_seed=seed,sampling_seed=spec['sampling_seed'],
        rate_multiplier=float(multiplier),rate_multiplier_decimal=number_text(multiplier),rate_rps=rate_float,
        rate_rps_decimal=number_text(rate),anchor_rps=float(group['anchor']),anchor_capacity_verified=False,
        n_requests=n,planned_arrival_span_s=span,trace_duration_s=span,content_pairing_sha256=content,
        source_indices_sha256=digest(indices),slo=spec['slo_by_dataset'][dataset],
        unique_selected_pool_records=len(set(indices)),within_trace_resampling=n > len(set(indices)),
        resampling_method='repeat the same deterministic shuffled development-pool cycle; unchanged token/output records',
        formal_eligible=False,saturation_verified=False,capacity_rps=None,execution_status='not_run',
        actual_open_loop_span_verified=False,policy_reference=group['policy'])


def make_trace(spec,cell,group,indices,arrivals):
    records=[group['records'][i] for i in indices];rate=cell['rate_rps']
    reqs=[dict(idx=i,arrival_s=arrivals[i]/rate,prompt_len=r['input_tokens'],output_len=r['output_tokens'])
          for i,r in enumerate(records)]
    prompt_hashes={digest(r['prompt']) for r in records}
    return dict(schema=2,model=cell['model'],dataset=cell['dataset'],split='development',load='declared_absolute_rate',
        protocol_id=PROTOCOL,measurement_schema=3,seed=cell['arrival_seed'],arrival_seed=cell['arrival_seed'],
        sampling_seed=spec['sampling_seed'],rate=rate,duration_s=reqs[-1]['arrival_s'],requests=reqs,
        prompts=[r['prompt'] for r in records],source_shapes=[r['request_shape_sha256'] for r in records],
        source_pool_indices=indices,pool=group['pool'],slo=cell['slo'],
        source_generation_spec_sha256=digest(spec),content_pairing_sha256=cell['content_pairing_sha256'],
        arrival_process='Poisson: cumulative random.Random(arrival_seed).expovariate(1) divided by rate',
        arrival_seed_independence='distinct PRNG streams; same unit-rate stream paired across rates',
        content_pairing='same count and exact content across seeds at one rate; fixed shared content prefix across rates',
        corpus_independence_across_arrival_seeds=False,
        n_requests=len(reqs),minimum_requests=spec['min_requests'],minimum_planned_span_s=spec['min_planned_arrival_span_s'],
        planned_arrival_span_s=reqs[-1]['arrival_s'],actual_open_loop_span_verified=False,
        unique_source_shapes=len({r['request_shape_sha256'] for r in records}),unique_prompt_payloads=len(prompt_hashes),
        repeated_prompt_count=len(records)-len(prompt_hashes),within_trace_resampling=cell['within_trace_resampling'],
        repeat_from_existing_dev_pool=True,resampling_method=cell['resampling_method'],
        output_lengths_modified=False,prompts_truncated=False,formal_eligible=False,
        execute_baselines=False,capacity_rps=None,saturation_verified=False,
        purpose='comprehensive PDB development matrix; no formal inference against frozen unmatched baselines')


def generate(spec,out,*,models=None,datasets=None,multipliers=None,seeds=None,anchors=False,plan_only=False):
    groups=load_spec(spec)
    models=list(models or MODELS);datasets=list(datasets or DATASETS)
    multipliers=[Decimal(str(x)) for x in (multipliers or ([1] if anchors else spec['rate_multipliers']))]
    seeds=list(seeds or ([701] if anchors else spec['arrival_seeds']))
    require(set(models) <= set(MODELS) and len(set(models)) == len(models),'unknown/duplicate model filter')
    require(set(datasets) <= set(DATASETS) and len(set(datasets)) == len(datasets),'unknown/duplicate dataset filter')
    require(set(seeds) <= set(SEEDS) and len(set(seeds)) == len(seeds),'unknown/duplicate arrival seed')
    require(set(multipliers) <= {Decimal(str(x)) for x in spec['rate_multipliers']}
        and len(set(multipliers)) == len(multipliers),'rate outside frozen declaration')
    out=Path(out).resolve();require(not out.exists(),'output directory must be new')
    cells=[];pending=[]
    for model in models:
        for dataset in datasets:
            group=groups[(model,dataset)]
            for multiplier in sorted(multipliers):
                rate=float(group['anchor']*multiplier)
                # Use every declared seed when selecting N, even a single-cell
                # CLI invocation, so separately generated cells still pair.
                n=required_count(rate,SEEDS,spec['min_requests'],spec['min_planned_arrival_span_s'])
                indices=selected_indices(group['order'],n)
                for seed in seeds:
                    arrivals=unit_arrivals(n,seed)
                    cell=cell_metadata(spec,model,dataset,multiplier,seed,group,n,indices,arrivals)
                    path=out/'traces'/(cell['cell_id']+'.json')
                    cell.update(trace_path=str(path),trace_sha256=None,materialized=False)
                    cells.append(cell)
                    if not plan_only:
                        pending.append((cell,group,indices,arrivals,path))
    out.mkdir(parents=True,exist_ok=False)
    if pending:
        (out/'traces').mkdir()
    for cell,group,indices,arrivals,path in pending:
        payload=encode(make_trace(spec,cell,group,indices,arrivals));path.write_bytes(payload)
        cell.update(trace_sha256=hashlib.sha256(payload).hexdigest(),trace_bytes=len(payload),materialized=True)
    load_spec(spec)  # Recheck source/config/pool identities after materialization.
    manifest=dict(kind='pdblend_full_development_matrix_v1',protocol_id=PROTOCOL,measurement_schema=3,
        generation_spec_sha256=digest(spec),generator_sha256=file_digest(Path(__file__)),split='development',
        formal_eligible=False,execute_baselines=False,source_development_spec=spec['source_development_spec'],
        declaration_scope='old absolute pilot anchors multiplied by a dense declared grid; saturation not yet verified',
        model_policy_rule='one frozen policy per model across datasets/rates; only user external SLO differs',
        slo_by_dataset=SLOS,request_slo_rule='completed request meeting TTFT and average TPOT; joint attainment target .9',
        minimum_requests=spec['min_requests'],minimum_planned_arrival_span_s=spec['min_planned_arrival_span_s'],
        minimum_span_scope='trace plan only; bench must independently verify actual open-loop arrivals and complete work',
        workload_pairing='same rate has equal N and exact content across all declared arrival seeds; rates share prefixes',
        runtime_configuration_contract=dict(evaluation_protocol='evaluation-v3',external_slo_protocol=PROTOCOL,
            protocol_metadata_locations=['runspec.slo_protocol','summary.slo_protocol'],measurement_schema=3,
            per_cell_overlay_keys=['slo_ttft_s','slo_tpot_s','slo_attainment_target']),
        materialized_cells=sum(c['materialized'] for c in cells),cells=cells,
        total_planned_arrival_span_s=sum(c['planned_arrival_span_s'] for c in cells),
        pool_provenance={m+'-'+d:g['pool'] for (m,d),g in groups.items()},
        no_prompt_or_output_length_changes=True,holdout_selected=False,
        runtime_or_gpu_execution=False)
    (out/'spec.json').write_bytes(encode(spec));(out/'manifest.json').write_bytes(encode(manifest))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',action='append',choices=MODELS)
    parser.add_argument('--dataset',action='append',choices=DATASETS)
    parser.add_argument('--rate-multiplier',action='append',type=Decimal)
    parser.add_argument('--arrival-seed',action='append',type=int,choices=SEEDS)
    parser.add_argument('--anchors',action='store_true');parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    require(not args.anchors or args.rate_multiplier is None and args.arrival_seed is None,
            '--anchors fixes multiplier1/seed701; do not override those filters')
    result=generate(read(args.spec),args.out,models=args.model,datasets=args.dataset,
        multipliers=args.rate_multiplier,seeds=args.arrival_seed,anchors=args.anchors,plan_only=args.plan_only)
    print(json.dumps(dict(out=str(args.out),cells=len(result['cells']),materialized=result['materialized_cells'],
        min_requests=min(c['n_requests'] for c in result['cells']),
        min_planned_span_s=min(c['planned_arrival_span_s'] for c in result['cells']),
        total_planned_arrival_hours=result['total_planned_arrival_span_s']/3600),indent=2))


if __name__ == '__main__':
    main()
