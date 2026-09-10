"""New, bounded fixed-window development traces; no GPU or baseline execution.

Use --rate-rps for declared probes before capacity is known. Use --rate-spec for
ten absolute rates per selected group after probes. Only new output directories
are accepted. SLO scales reuse these exact traces without changing the workload.
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

HERE=Path(__file__).resolve().parent
PROTOCOL='per-dataset-slo-fixed-window-v2'
MODELS=('7b','14b','32b');DATASETS=('alpaca','sharegpt','longbench');SEEDS=(701,1701)
SLOS={'alpaca':{'ttft_s':1.,'tpot_s':.1,'attainment_target':.9},
      'sharegpt':{'ttft_s':5.,'tpot_s':.15,'attainment_target':.9},
      'longbench':{'ttft_s':15.,'tpot_s':.2,'attainment_target':.9}}


def require(condition,reason):
    if not condition:raise ValueError(reason)


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def digest(value):return hashlib.sha256(encode(value)).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def frozen(ref):
    require(isinstance(ref,dict) and re.fullmatch('[0-9a-f]{64}',str(ref.get('sha256'))),
        'source must declare a SHA256')
    path=Path(ref['path']).resolve()
    require(file_sha(path)==ref['sha256'],'frozen source changed: '+str(path))
    return path


def number(value):return format(Decimal(str(value)).normalize(),'f')


def window_arrivals(rate,seed,window=300.):
    require(type(rate) in (int,float) and math.isfinite(rate) and rate>0,'positive finite absolute rate required')
    require(seed in SEEDS and type(seed) is int,'two declared arrival seeds only')
    require(window==300.,'this protocol fixes the arrival window at 300 seconds')
    rng=random.Random(seed);unit=0.;values=[0.]
    while True:
        unit+=rng.expovariate(1.)
        arrival=unit/rate
        if arrival>=window:return values
        values.append(arrival)
        require(len(values)<=2_000_000,'bounded materialization exceeded; do not truncate the workload')


def load_sources(spec):
    require(spec.get('protocol_id')==PROTOCOL and spec.get('measurement_schema')==3,'wrong protocol/schema')
    require(spec.get('split')=='development' and spec.get('execute_baselines') is False
        and spec.get('formal_eligible') is False,'PDB development only; no formal/baseline execution')
    require(spec.get('arrival_seeds')==list(SEEDS) and spec.get('arrival_window_s')==300
        and spec.get('minimum_requests') is None,'two-seed fixed window must not inherit old N/span gates')
    require(spec.get('slo_by_dataset')==SLOS and spec.get('slo_scales')==[.5,1.,2.],'user SLO contract differs')
    deadline=read(frozen(spec['deadline_scope']))
    require(deadline.get('arrival_window_s')==300 and deadline.get('arrival_seeds')==list(SEEDS)
        and deadline.get('rates_per_model_dataset')==10,'hard-deadline scope differs')
    pool_spec=read(frozen(spec['frozen_development_pool_declaration']))
    loader_path=frozen(spec['frozen_development_pool_loader'])
    module_spec=importlib.util.spec_from_file_location('_frozen_full_matrix_pool_loader',loader_path)
    loader=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(loader)
    # This validates the historical pool, split and source identities only. Its
    # old request minimum, three seeds and rate grid are never used to emit rows.
    groups=loader.load_spec(pool_spec)
    require(type(pool_spec['sampling_seed']) is int and pool_spec['allow_repeat_from_dev_pool'] is True,
        'explicit unchanged development resampling policy required')
    return groups,pool_spec['sampling_seed']


def rate_groups(*,models,datasets,probe_rates=None,rate_spec=None):
    require(bool(probe_rates) != bool(rate_spec),'choose explicit probe rate(s) or a frozen ten-rate spec')
    require(models and set(models)<=set(MODELS) and len(models)==len(set(models)),'unknown/duplicate models')
    require(datasets and set(datasets)<=set(DATASETS) and len(datasets)==len(set(datasets)),'unknown/duplicate datasets')
    if probe_rates:
        require(len(models)==len(datasets)==1,'explicit probe rates require one model and one dataset')
    else:
        require(rate_spec.get('protocol_id')==PROTOCOL
            and rate_spec.get('scope')=='ten_absolute_rates_frozen_after_capacity_probes',
            'grid needs its own explicit post-probe rate declaration')
        evidence=rate_spec.get('selection_evidence',{})
        require(bool(evidence),'ten-rate declaration requires source-bound capacity/probe evidence')
        for path,sha in evidence.items():frozen(dict(path=path,sha256=sha))
    result={}
    for model in models:
        for dataset in datasets:
            values=probe_rates if probe_rates else rate_spec['rates_by_model_dataset'][model][dataset]
            require(isinstance(values,list) and values,'rate list required')
            values=[Decimal(str(x)) for x in values]
            require(all(x.is_finite() and x>0 for x in values) and len(values)==len(set(values)),
                'rates must be unique positive finite absolute rates')
            require(probe_rates is not None or len(values)==10,'each selected matrix group requires ten rates')
            result[(model,dataset)]=sorted(values)
    return result


def build_trace(spec,model,dataset,rate,seed,group,sampling_seed):
    rate_float=float(rate);arrivals=window_arrivals(rate_float,seed);n=len(arrivals)
    order=group['order'];indices=(order*math.ceil(n/len(order)))[:n]
    records=[group['records'][index] for index in indices]
    work=digest([group['record_digests'][index] for index in indices])
    requests=[dict(idx=i,arrival_s=t,prompt_len=r['input_tokens'],output_len=r['output_tokens'])
        for i,(t,r) in enumerate(zip(arrivals,records))]
    repeated=n>len(set(indices));sparse=n<30
    cell_id=f'{model}-{dataset}-r{number(rate)}-s{seed}-w300'
    cell=dict(cell_id=cell_id,model=model,dataset=dataset,protocol_id=PROTOCOL,
        measurement_schema=3,split='development',system='pdblend',variant='candidate',
        strategy=group['strategy'],load='declared_absolute_rate',arrival_seed=seed,
        sampling_seed=sampling_seed,rate_rps=rate_float,rate_rps_decimal=number(rate),
        arrival_window_s=300.,trace_duration_s=300.,planned_arrival_span_s=arrivals[-1],
        n_requests=n,content_pairing_sha256=work,source_indices_sha256=digest(indices),
        slo=SLOS[dataset],slo_protocol='per-dataset-slo-v1',allowed_slo_scales=[.5,1.,2.],
        unique_selected_pool_records=len(set(indices)),within_trace_resampling=repeated,
        resampling_method='repeat unchanged deterministic shuffled development-pool cycle',
        sparse_screen=sparse,request_count_warning='fewer than30; descriptive screen only' if sparse else None,
        source_policy_reference=group['policy'],policy_execution_binding='runner freezes one actual policy per model',
        formal_eligible=False,capacity_rps=None,saturation_verified=False,
        execution_status='not_run',actual_fixed_window_verified=False)
    trace=dict(schema=2,model=model,dataset=dataset,split='development',load='declared_absolute_rate',
        protocol_id=PROTOCOL,measurement_schema=3,seed=seed,arrival_seed=seed,sampling_seed=sampling_seed,
        rate=rate_float,duration_s=300.,arrival_window_s=300.,planned_arrival_span_s=arrivals[-1],
        requests=requests,prompts=[r['prompt'] for r in records],n_requests=n,
        source_shapes=[r['request_shape_sha256'] for r in records],source_pool_indices=indices,pool=group['pool'],
        slo=SLOS[dataset],slo_protocol='per-dataset-slo-v1',allowed_slo_scales=[.5,1.,2.],
        source_generation_spec_sha256=digest(spec),content_pairing_sha256=work,
        arrival_process='first arrival anchored at zero; subsequent cumulative random.Random(seed).expovariate(1)/rate, retained strictly before300s',
        arrival_seed_independence='two independent PRNG streams; paired unit-rate stream across rates',
        content_pairing='same seed/rate reuses exact trace across candidates and SLO scales; fixed shared content prefix across seeds/rates, N may differ',
        corpus_independence_across_arrival_seeds=False,minimum_requests=None,
        minimum_planned_span_s=None,fixed_observation_window_required_s=300.,
        post_window_drain_allowance_s=120.,request_hard_timeout_s=120.,
        unique_source_shapes=len({r['request_shape_sha256'] for r in records}),
        within_trace_resampling=repeated,repeat_from_existing_dev_pool=True,
        resampling_method=cell['resampling_method'],sparse_screen=sparse,
        output_lengths_modified=False,prompts_truncated=False,formal_eligible=False,
        execute_baselines=False,actual_fixed_window_verified=False,
        purpose='24h PDB development screening; no formal inference against unmatched frozen baselines')
    return cell,trace


def generate(spec,out,*,models,datasets,seeds=SEEDS,probe_rates=None,rate_spec=None,plan_only=False):
    require(seeds and set(seeds)<=set(SEEDS) and len(seeds)==len(set(seeds)),'unknown/duplicate arrival seeds')
    rates=rate_groups(models=models,datasets=datasets,probe_rates=probe_rates,rate_spec=rate_spec)
    groups,sampling_seed=load_sources(spec)
    out=Path(out).resolve();require(not out.exists(),'output must be new; old traces/results are immutable')
    out.mkdir(parents=True,exist_ok=False)
    if not plan_only:(out/'traces').mkdir()
    cells=[]
    for (model,dataset),values in rates.items():
        for rate in values:
            for seed in seeds:
                cell,trace=build_trace(spec,model,dataset,rate,seed,groups[(model,dataset)],sampling_seed)
                path=out/'traces'/(cell['cell_id']+'.json');payload=encode(trace)
                cell.update(trace_path=str(path),trace_sha256=hashlib.sha256(payload).hexdigest(),
                    trace_bytes=len(payload),materialized=not plan_only)
                if not plan_only:path.write_bytes(payload)
                cells.append(cell)
    load_sources(spec)
    if rate_spec:
        rate_groups(models=models,datasets=datasets,rate_spec=rate_spec)
    result=dict(schema=1,kind='pdblend_fixed_window_development_v2',protocol_id=PROTOCOL,
        measurement_schema=3,split='development',formal_eligible=False,execute_baselines=False,
        generator_sha256=file_sha(Path(__file__)),generation_spec_sha256=digest(spec),
        deadline_scope=spec['deadline_scope'],scope='declared capacity probes' if probe_rates else 'selected ten-rate matrix groups',
        declared_main_cells_all_models=180,declared_main_cells_per_model=60,
        selected_cells=len(cells),materialized_cells=sum(c['materialized'] for c in cells),
        selected_matrix_complete=False,execution_complete=False,arrival_window_s=300.,
        minimum_requests=None,minimum_actual_dispatch_span_s=None,
        arrival_seeds=list(seeds),slo_by_dataset=SLOS,slo_scales=[.5,1.,2.],
        workload_pairing='reuse exact trace SHA for every candidate/scale; variable N across different arrival seeds',
        policy_rule='one runner-frozen policy per model across datasets/rates; only external SLO/scale differs',
        required_runtime_config=dict(evaluation_protocol='evaluation-v3',measurement_window_protocol=PROTOCOL,arrival_window_s=300),
        requires_actual_epoch_window_and_full_drain_evidence=True,
        holdout_selected=False,output_lengths_modified=False,prompts_truncated=False,
        rate_declaration=rate_spec,probe_rates_rps=[number(r) for r in probe_rates] if probe_rates else None,
        total_arrival_window_hours=len(cells)*300/3600.,
        nominal_window_plus_max_drain_hours=len(cells)*420/3600.,
        duration_excludes='initialization, identity checks, postmeasurement restoration; not a walltime guarantee',
        cells=cells)
    (out/'spec.json').write_bytes(encode(spec));(out/'manifest.json').write_bytes(encode(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=HERE/'fixed-window-spec.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',action='append',choices=MODELS)
    parser.add_argument('--dataset',action='append',choices=DATASETS)
    parser.add_argument('--arrival-seed',action='append',type=int,choices=SEEDS)
    parser.add_argument('--rate-rps',action='append',type=Decimal)
    parser.add_argument('--rate-spec',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    result=generate(read(args.spec),args.out,models=args.model or list(MODELS),
        datasets=args.dataset or list(DATASETS),seeds=args.arrival_seed or list(SEEDS),
        probe_rates=args.rate_rps,rate_spec=read(args.rate_spec) if args.rate_spec else None,plan_only=args.plan_only)
    print(json.dumps(dict(out=str(args.out),selected_cells=result['selected_cells'],
        materialized=result['materialized_cells'],window_hours=result['total_arrival_window_hours'],
        n_min=min(c['n_requests'] for c in result['cells']),n_max=max(c['n_requests'] for c in result['cells']),
        sparse_cells=sum(c['sparse_screen'] for c in result['cells'])),indent=2))


if __name__=='__main__':main()
