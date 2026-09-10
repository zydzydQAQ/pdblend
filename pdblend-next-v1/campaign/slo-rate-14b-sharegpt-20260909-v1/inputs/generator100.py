"""Reuse frozen development traces: seed 701, 100 s, five systems, no execution.

Every original 300 s trace is re-created with its frozen generator and pool
reader before selecting the unchanged arrival<100 prefix. No new sampler,
content generation, output prediction, or held-out selection is introduced.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
SYSTEMS=('pdblend','mixed','distserve','dynamollm','ecoserve')
MODELS=('14b','32b','7b')
DATASETS=('alpaca','sharegpt','longbench')
WINDOW=100.;SEED=701
PARENT_PROTOCOL='per-dataset-slo-fixed-window-v2'
PARENT_GENERATOR=HERE.parent/'two-seed-fixed-window-v1/generate.py'
PARENT_GENERATOR_SHA='86c6d3c25c55f60c11f961f029e4d2b37869c34f3a2859611d5e1c5997db3c1a'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):return hashlib.sha256(encode(value)).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def number(value):return format(Decimal(str(value)).normalize(),'f')


def frozen(reference):
    path=Path(reference['path']).resolve()
    require(sha(path)==reference['sha256'],'frozen input changed: '+str(path))
    return path


def parent_generator():
    frozen(dict(path=str(PARENT_GENERATOR),sha256=PARENT_GENERATOR_SHA))
    spec=importlib.util.spec_from_file_location('_frozen_window300_generator',PARENT_GENERATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def workload_hash(trace):
    return digest([digest(dict(prompt=prompt,input_tokens=req['prompt_len'],output_tokens=req['output_len']))
                   for req,prompt in zip(trace['requests'],trace['prompts'])])


def prefix_trace(original,source):
    """Keep full request bodies; only arrivals outside the new window are excluded."""
    require(original.get('protocol_id')==PARENT_PROTOCOL and original.get('arrival_window_s')==300
        and original.get('duration_s')==300 and original.get('seed')==SEED
        and original.get('split')=='development' and original.get('measurement_schema')==3,
        'not an original seed701/300s development trace')
    requests=original['requests'];n=original['n_requests']
    require(n>0 and all(len(original[k])==n for k in ('requests','prompts','source_shapes','source_pool_indices')),
        'original trace lacks full aligned workload')
    require(workload_hash(original)==original['content_pairing_sha256'],'original content digest differs')
    require(requests[0]['arrival_s']==0 and all(q['idx']==i for i,q in enumerate(requests))
        and all(0<=q['arrival_s']<300 for q in requests)
        and all(a['arrival_s']<=b['arrival_s'] for a,b in zip(requests,requests[1:])),
        'original request order or arrival domain differs')
    keep=sum(q['arrival_s']<WINDOW for q in requests)
    result=copy.deepcopy(original)
    for key in ('requests','prompts','source_shapes','source_pool_indices'):result[key]=result[key][:keep]
    result.update(protocol_id=PROTOCOL,seed=SEED,arrival_seed=SEED,n_requests=keep,
        duration_s=WINDOW,arrival_window_s=WINDOW,fixed_observation_window_required_s=WINDOW,
        planned_arrival_span_s=result['requests'][-1]['arrival_s'],
        comparison_systems=list(SYSTEMS),execute_baselines=True,
        source_300s_trace=source,source_300s_content_pairing_sha256=original['content_pairing_sha256'],
        selection='exact original seed701 request prefix with arrival_s < 100; no request body modified',
        arrival_process='first arrival anchored at zero; original cumulative exponential arrivals retained strictly before100s',
        arrival_seed_independence='one declared arrival realization; no independent-seed confidence interval',
        content_pairing='same exact trace SHA for all five systems and all SLO scales',
        minimum_requests=None,minimum_planned_span_s=None,
        post_window_drain_allowance_s=120.,request_hard_timeout_s=120.,
        unique_source_shapes=len(set(result['source_shapes'])),
        within_trace_resampling=keep>len(set(result['source_pool_indices'])),
        sparse_screen=keep<30,formal_eligible=False,
        purpose='five-system paired development comparison; current 100s runs only; historical300s excluded')
    result['content_pairing_sha256']=workload_hash(result)
    # Array equality proves that no input/output body or original arrival moved.
    require(all(result[k]==original[k][:keep] for k in ('requests','prompts','source_shapes','source_pool_indices')),
        'prefix changed a request')
    return result


def source_rows(parent,source,model):
    require(parent.get('protocol_id')==source.get('protocol_id')==PARENT_PROTOCOL
        and parent.get('model')==model and source.get('measurement_schema')==3
        and source.get('generation_spec_sha256') and source.get('generator_sha256')==PARENT_GENERATOR_SHA,
        'wrong original declaration or generator')
    rows=[r for r in source['cells'] if r['arrival_seed']==SEED]
    require(len(rows)==30 and all(r['model']==model and r['materialized'] is True for r in rows),
        'one model needs exactly thirty original seed701 workloads')
    keyed={(r['dataset'],number(r['rate_rps'])):r for r in rows}
    require(len(keyed)==30,'duplicate source workload')
    for dataset in DATASETS:
        require(len([k for k in keyed if k[0]==dataset])==10,'ten original absolute rates required per dataset')
    ordered=[]
    for row in parent['cells']:
        if row['phase']=='main' and row['arrival_seed']==SEED:
            key=(row['dataset'],number(row['rate_rps']))
            require(key in keyed and row['trace_sha256']==keyed[key]['trace_sha256'],
                'parent main workload differs from source')
            ordered.append(keyed[key])
    require(len(ordered)==30 and len({r['cell_id'] for r in ordered})==30,'parent main grid differs')
    scale_rates={d:set() for d in DATASETS}
    scale_seen=set()
    for row in parent['cells']:
        if row['phase']=='scale' and row['arrival_seed']==SEED:
            key=(row['dataset'],number(row['rate_rps']))
            require(key in keyed and row['trace_sha256']==keyed[key]['trace_sha256']
                and row['slo_scale'] in (.5,2.),'parent scale workload differs')
            scale_rates[row['dataset']].add(key[1]);scale_seen.add((*key,row['slo_scale']))
    require(all(len(v)==3 for v in scale_rates.values()) and len(scale_seen)==18
        and all((d,r,s) in scale_seen for d,rs in scale_rates.items() for r in rs for s in (.5,2.)),
        'original three scale rates and both additional scales required')
    return ordered,scale_rates


def execution_rows(workloads,scale_rates):
    main=[];scale=[]
    # Every system sees exactly the same workload order. This declaration does
    # not claim execution order is randomized, nor prescribe hardware switching.
    for system in SYSTEMS:
        for workload in workloads:
            for factor,phase,destination in ((1.,'main',main),(.5,'scale',scale),(2.,'scale',scale)):
                if phase=='scale' and number(workload['rate_rps']) not in scale_rates[workload['dataset']]:continue
                base=workload['slo'];wid=workload['workload_id']
                row=dict(workload,cell_id=f'{wid}-{system}-slo{number(factor)}',system=system,
                    strategy=None,controller_config=None,policy_binding_required=True,
                    phase=phase,part=phase,sequence=len(destination)+1,slo_scale=factor,
                    slo_ttft_s=base['ttft_s']*factor,slo_tpot_s=base['tpot_s']*factor,
                    slo_attainment_target=.9,
                    reuse_main_cell_id=None if phase=='main' else f'{wid}-{system}-slo1')
                destination.append(row)
    require(len(main)==150 and len(scale)==90,'five-system phase count differs')
    for workload in workloads:
        group=[r for r in main if r['workload_id']==workload['workload_id']]
        require({r['system'] for r in group}==set(SYSTEMS)
            and len({(r['trace_sha256'],r['content_pairing_sha256'],r['n_requests']) for r in group})==1,
            'five-system workload pairing differs')
    indexed={r['cell_id']:r for r in main}
    require(all(indexed[r['reuse_main_cell_id']]['trace_sha256']==r['trace_sha256'] for r in scale),
        'scale must reuse its exact new100s main workload')
    return main,scale


def generate(inputs,out,*,model):
    require(model in MODELS and inputs.get('protocol_id')==PROTOCOL,'explicit current protocol/model required')
    old=parent_generator();entry=inputs['models'][model]
    parent_path=frozen(entry['parent_runspec']);source_path=frozen(entry['source_manifest'])
    spec_path=frozen(entry['source_spec']);parent=read(parent_path);source=read(source_path);spec=read(spec_path)
    require(old.digest(spec)==source['generation_spec_sha256'],'original generation spec changed')
    ordered,scale_rates=source_rows(parent,source,model)
    groups,sampling_seed=old.load_sources(spec)
    out=Path(out).resolve();require(not out.exists(),'output must be new; no frozen traces/results may be overwritten')
    out.mkdir(parents=True);(out/'traces').mkdir()
    frozen_inputs=[entry['parent_runspec'],entry['source_manifest'],entry['source_spec'],ref(PARENT_GENERATOR)]
    workloads=[]
    for row in ordered:
        path=frozen(dict(path=row['trace_path'],sha256=row['trace_sha256']));original=read(path)
        _,expected=old.build_trace(spec,model,row['dataset'],Decimal(row['rate_rps_decimal']),SEED,
                                  groups[(model,row['dataset'])],sampling_seed)
        require(path.read_bytes()==old.encode(expected),'original trace differs from strict frozen pool/generator')
        source_ref=dict(path=str(path),sha256=row['trace_sha256']);frozen_inputs.append(source_ref)
        trace=prefix_trace(original,source_ref);wid=f'{model}-{row["dataset"]}-r{number(row["rate_rps"])}-s701-w100'
        trace_path=out/'traces'/(wid+'.json');trace_path.write_bytes(encode(trace))
        n=trace['n_requests']
        workloads.append(dict(workload_id=wid,cell_id=wid,model=model,dataset=row['dataset'],
            protocol_id=PROTOCOL,measurement_schema=3,split='development',load='declared_absolute_rate',
            seed=SEED,arrival_seed=SEED,rate_rps=row['rate_rps'],rate_rps_decimal=number(row['rate_rps']),
            arrival_window_s=WINDOW,trace_duration_s=WINDOW,planned_arrival_span_s=trace['planned_arrival_span_s'],
            n_requests=n,trace=str(trace_path),trace_path=str(trace_path),trace_sha256=sha(trace_path),
            trace_bytes=trace_path.stat().st_size,materialized=True,slo=trace['slo'],slo_protocol='per-dataset-slo-v1',
            allowed_slo_scales=[.5,1.,2.],content_pairing_sha256=trace['content_pairing_sha256'],
            source_indices_sha256=digest(trace['source_pool_indices']),source_300s_trace=source_ref,
            source_manifest=str(source_path),source_manifest_sha256=entry['source_manifest']['sha256'],
            sampling_seed=sampling_seed,sparse_screen=n<30,
            request_count_warning='fewer than30; descriptive single-seed point' if n<30 else 'one arrival seed; no independent-seed CI',
            within_trace_resampling=trace['within_trace_resampling'],
            unique_selected_pool_records=len(set(trace['source_pool_indices'])),
            execution_status='not_run',actual_fixed_window_verified=False,formal_eligible=False))
    # Recheck sources after construction; the saved manifest never hides a race.
    for reference in frozen_inputs:frozen(reference)
    old.load_sources(spec)
    main,scale=execution_rows(workloads,scale_rates)
    result=dict(schema=1,kind='five_system_shared_workload_and_execution_declaration',protocol_id=PROTOCOL,
        model=model,measurement_schema=3,split='development',execute_baselines=True,formal_eligible=False,
        arrival_seeds=[SEED],arrival_window_s=WINDOW,request_hard_timeout_s=120.,
        drain_after_arrival_window_s=120.,comparison_systems=list(SYSTEMS),rates_per_dataset=10,
        source_references=frozen_inputs,generator_sha256=sha(__file__),input_declaration_sha256=digest(inputs),
        declared_workloads=30,declared_main_cells=150,declared_additional_scale_cells=90,
        declared_scale_one_references=45,total_executed_new_cells_if_complete=240,
        main_arrival_window_hours=150*WINDOW/3600,additional_scale_arrival_window_hours=90*WINDOW/3600,
        max_window_plus_drain_hours=240*(WINDOW+120)/3600,
        duration_excludes='startup, transitions and restoration; maximum-drain accounting is not a runtime estimate',
        minimum_requests=None,minimum_actual_dispatch_span_s=None,
        scale_rates_by_dataset={d:sorted(rs,key=Decimal) for d,rs in scale_rates.items()},
        original_rates_unchanged=True,original_request_bodies_unchanged=True,holdout_selected=False,
        policy_rule='each system retains an independently frozen actual implementation/config; the shared workload has no policy parameters',
        comparison_scope='only new100s executions sharing exact trace SHA; old300s and historic64 results remain separate',
        energy_scope='all eight GPU boards over arrival epoch through fixed100s idle suffix and bounded actual drain; failed work included',
        statistical_scope='single arrival realization per point; sparse N<30 marked; no independent-repeat confidence claim',
        runnable=False,execution_binding_required=True,execution_complete=False,
        workloads=workloads,main=main,scale=scale,cells=main+scale)
    (out/'manifest.json').write_bytes(encode(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=HERE/'sources.json')
    parser.add_argument('--model',required=True,choices=MODELS)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=generate(read(args.inputs),args.out,model=args.model)
    print(json.dumps(dict(out=str(args.out),model=args.model,workloads=30,main=150,scale=90,
        requests_min=min(r['n_requests'] for r in result['workloads']),
        requests_max=max(r['n_requests'] for r in result['workloads']),
        sparse_workloads=sum(r['sparse_screen'] for r in result['workloads']),
        manifest_sha256=sha(args.out/'manifest.json')),indent=2))


if __name__=='__main__':main()
