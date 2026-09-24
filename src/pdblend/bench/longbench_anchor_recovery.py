"""Recover only the missing 32B LongBench anchor without rewriting prior work.

Candidate rates descend from the prior failed tuning rate. Every candidate
needs fresh calibration and independent tuning; evaluation is never selected.
Inherited anchors retain their original confirmation bytes and provenance.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

from . import rate_anchor as base
from .client import SLOS, load_split, poisson_trace
from .resident_session import file_sha, write_new
from pdblend.results.power_archive import write_power_archive

MODEL = 'Qwen2.5-32B-Instruct'
SCHEMA = 'longbench-anchor-recovery-v1'
INHERITED = ('alpaca','sharegpt')


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path),sha256=file_sha(path))


def bound(ref):
    path = Path(ref['path'])
    if file_sha(path) != ref['sha256']:
        raise ValueError('recovery input checksum differs: '+str(path))
    return json.loads(path.read_text())


def candidate_rates(failed_rate, minimum):
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in (failed_rate,minimum))
            or minimum > failed_rate/2):
        raise ValueError('recovery floor must be positive and below the failed tuning rate')
    rates=[];value=failed_rate/2
    while value >= minimum:
        rates.append(value)
        if len(rates)>8:
            raise ValueError('recovery requires a bounded maximum of eight candidates')
        value/=2
    if rates[-1] != minimum:
        raise ValueError('recovery floor must be one of the fixed halving candidates')
    return rates


def prior_inputs(prior_root):
    """Bind small immutable receipts; do not duplicate historical token logs."""
    root=Path(prior_root).resolve();prior=json.loads((root/'completion.json').read_text())
    inputs={name:binding(root/(name+'.json')) for name in ('completion','preflight','rate-anchor')}
    confirmations={}
    for dataset in INHERITED:
        anchor=prior['anchors'][dataset]
        prefix='/output/anchor/'
        path=anchor['confirmation_path']
        if not path.startswith(prefix):
            raise ValueError('unknown prior confirmation namespace')
        relative=Path(path[len(prefix):])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('prior confirmation escapes owning directory')
        receipt=root/relative
        if file_sha(receipt)!=anchor['confirmation_sha256']:
            raise ValueError('prior confirmation differs from original anchor')
        confirmations[dataset]=dict(completion=binding(receipt),requests=binding(receipt.parent/'requests.json'))
    failures=[p for p in sorted(root.glob('longbench-tuning-*/completion.json'))
              if json.loads(p.read_text()).get('metrics',{}).get('passed') is False]
    if not failures:
        raise ValueError('prior failed LongBench tuning evidence required')
    failed=min(failures,key=lambda p:json.loads(p.read_text())['rate_rps'])
    return dict(receipts=inputs,inherited=confirmations,
                failed_tuning=dict(completion=binding(failed),requests=binding(failed.parent/'requests.json')))


def validate_prior(inputs, audit):
    prior=bound(inputs['receipts']['completion'])
    initial=bound(inputs['receipts']['preflight'])
    anchors=bound(inputs['receipts']['rate-anchor'])
    if (prior.get('status')!='failed' or prior.get('complete') is not False or prior.get('cleanup_errors')
            or prior.get('error')!='RuntimeError: longbench: no independent tuning confirmation passed'
            or prior.get('selection_splits')!=['calibration','tuning']):
        raise ValueError('prior anchor is not the clean partial LongBench failure')
    keys=('model_id','model_hash','tokenizer_hash','image_digest','tp','pp','corpus_sha256',
          'corpus_tokenizer_sha256','corpus_manifest_sha256','calibration_seed','tuning_seed')
    for document in (prior,initial,anchors):
        if any(document.get(k)!=audit[k] for k in keys) or document.get('evaluation_used_for_selection') is not False:
            raise ValueError('prior anchor model/corpus/deployment identity differs')
    if prior.get('anchors')!=anchors.get('anchors') or set(prior['anchors'])!=set(INHERITED):
        raise ValueError('prior retained anchor inventory differs')
    for dataset,refs in inputs['inherited'].items():
        if dataset not in INHERITED:
            raise ValueError('unexpected inherited dataset')
        measured=bound(refs['completion']);requests=bound(refs['requests']);a=prior['anchors'][dataset]
        m=measured.get('metrics',{});slo=SLOS[dataset]
        if (a.get('model_id')!=MODEL or a.get('tp')!=2 or a.get('pp')!=1 or a.get('replicas')!=4
                or a.get('clock_mhz')!=2520 or a.get('corpus_sha256')!=audit['corpus_sha256'][dataset]
                or a.get('confirmation_sha256')!=refs['completion']['sha256']
                or measured.get('trace_sha256')!=refs['requests']['sha256']
                or measured.get('system')!='mixed' or measured.get('dataset')!=dataset
                or measured.get('split')!='tuning' or measured.get('seed')!=9702
                or measured.get('duration_s')!=120 or measured.get('rate_rps')!=a.get('base_rate_rps')
                or requests.get('seed')!=9702 or requests.get('duration_s')!=120
                or not requests.get('requests') or m.get('offered')!=len(requests['requests'])
                or m.get('correct')!=len(requests['requests'])
                or a.get('base_rate_rps',0)<=0 or a.get('x05_rate_rps')!=a['base_rate_rps']/2
                or m.get('passed') is not True or m.get('success_rate')!=1
                or m.get('joint_slo_rate',0)<.9 or m.get('ttft_p99_s',float('inf'))>slo[0]
                or m.get('tpot_p99_s',float('inf'))>slo[1]
                or (m.get('slo_ttft_s'),m.get('slo_tpot_s'))!=slo
                or measured.get('counts_reclaimed') is not True
                or measured.get('routing_policy')!='independent_least_load_fixed_tp'
                or len(measured.get('drain',[]))!=4
                or any(r.get('drain',{}).get('drained') is not True for r in measured['drain'])):
            raise ValueError('inherited independent tuning confirmation is not qualified: '+dataset)
    if set(inputs['inherited'])!=set(INHERITED):
        raise ValueError('both unchanged inherited anchors are required')
    failed=bound(inputs['failed_tuning']['completion']);requests=bound(inputs['failed_tuning']['requests'])
    if (failed.get('dataset')!='longbench' or failed.get('split')!='tuning' or failed.get('seed')!=9702
            or failed.get('duration_s')!=120 or failed.get('metrics',{}).get('passed') is not False
            or failed.get('trace_sha256')!=inputs['failed_tuning']['requests']['sha256']
            or requests.get('seed')!=9702 or requests.get('duration_s')!=120):
        raise ValueError('failed LongBench tuning identity differs')
    return deepcopy(prior['anchors']),failed['rate_rps']


def preflight(args):
    plan=json.loads(args.plan.read_text())
    if file_sha(args.plan)!=args.plan_sha256 or plan.get('schema')!=SCHEMA:
        raise ValueError('recovery plan checksum/schema differs')
    if (plan.get('model_id')!=MODEL or Path(args.model).name!=MODEL or args.tp!=2
            or plan.get('slo')!=dict(ttft_s=15.,tpot_s=.2) or SLOS['longbench']!=(15.,.2)
            or plan.get('selection_splits')!=['calibration','tuning']
            or plan.get('evaluation_used_for_selection') is not False
            or plan.get('calibration_seed')!=9701 or plan.get('tuning_seed')!=9702
            or plan.get('window_s')!=60 or plan.get('confirm_window_s')!=120):
        raise ValueError('recovery must retain the original model/SLO/split/window protocol')
    # Reuse the original CPU-only checks for tokenizer, corpus and source bytes.
    args.window,args.confirm_window,args.search_points=60.,120.,5
    audit=base.preflight(args)
    inherited,failed_rate=validate_prior(plan['prior'],audit)
    rates=candidate_rates(failed_rate,plan['minimum_rate_rps'])
    if rates!=plan.get('candidate_rates_rps'):
        raise ValueError('candidate schedule is not derived from the bound prior failed rate')
    source=bound(plan['source_manifest'])
    if source['source_sha256']!=audit['source_sha256'] or plan['image_digest']!=audit['image_digest']:
        raise ValueError('recovery execution source/image differs from frozen plan')
    audit.update(schema=SCHEMA,scope='longbench_only_anchor_recovery',prior_inputs=plan['prior'],
        prior_source_sha256=bound(plan['prior']['receipts']['preflight'])['source_sha256'],
        recovery_plan=binding(args.plan),candidate_rates_rps=rates,minimum_rate_rps=plan['minimum_rate_rps'],
        inherited_anchors=inherited,prior_failed_tuning_rate_rps=failed_rate,
        inherited_validation='bound native confirmation/request receipts; historical token logs are not rescanned',
        source_manifest=plan['source_manifest'])
    return audit


async def search(rates,observe,record):
    """Tuning failure starts a fresh lower-rate calibration, never promotion."""
    candidates=[]
    for index,rate in enumerate(rates):
        row=dict(index=index,rate_rps=rate,status='calibration_failed')
        try:
            calibration=await observe('calibration',9701,60.,rate,index)
            row['calibration']=calibration
            if calibration['metrics']['passed'] is True:
                tuning=await observe('tuning',9702,120.,rate,index)
                row['tuning']=tuning
                row['status']='confirmed' if tuning['metrics']['passed'] is True else 'tuning_failed'
        except BaseException as exc:
            row.update(status='measurement_failed',error=f'{type(exc).__name__}: {exc}')
            record(row)
            raise
        candidates.append(row)
        record(row)
        if row['status']=='confirmed':
            return row,candidates
    return None,candidates


async def run(args):
    audit=preflight(args)
    args.out.mkdir(parents=True,exist_ok=False)
    write_new(args.out/'preflight.json',audit)
    # Copy only the exact small confirmation receipts so legacy readers can
    # resolve their original /output/anchor namespace under this new root.
    for dataset,refs in audit['prior_inputs']['inherited'].items():
        relative=audit['inherited_anchors'][dataset]['confirmation_path'].removeprefix('/output/anchor/')
        target=args.out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(Path(refs['completion']['path']).read_bytes())
    specs=[base.NativeSpec(f'mixed-anchor-{i}',tuple(range(2*i,2*i+2)),args.base_port+4*i,
        args.model,tp=2,kv_connector=None,max_num_seqs=32,extra_args=('--enforce-eager',)) for i in range(4)]
    result=dict(audit,status='failed',complete=False,hardware_executed=False,capacity_exact=False,
        energy_comparable=False,selection_splits=['calibration','tuning'],engine_loads=0,
        anchors=deepcopy(audit['inherited_anchors']),windows=[],candidates=[],cleanup_errors=[])
    fleet=meter=sampler=None
    try:
        uuids=os.environ.get('PDBLEND_GPU_UUIDS','').split(',')
        if len(set(uuids))!=8 or not all(u.startswith('GPU-') for u in uuids):
            raise ValueError('recovery requires an explicit exclusive eight-GPU UUID lease')
        meter=base.Gpus(args.gpus,power_mode='instant')
        result['hardware']=base.gpu_manifest(meter,args.gpus)
        if len(result['hardware'])!=8:raise RuntimeError('eight physical boards required')
        for gpu in args.gpus:meter.unpark(gpu);meter.set_clock(gpu,2520)
        sampler=meter.sampler(interval_s=.1);sampler.start()
        fleet=base.Fleet(specs,args.out/'logs');result['hardware_executed']=True
        load_wait_started=time.time()
        with base.model_load_lock():
            load_started=time.time()
            result['load_lock_wait_s']=load_started-load_wait_started
            for spec in specs:
                fleet[spec.instance_id].start();result['engine_loads']+=1
                fleet[spec.instance_id].wait_ready(timeout_s=600)
        result['engine_load_s']=time.time()-load_started
        result['capabilities']=await base.verify_endpoints(specs)

        async def observe(split,seed,duration,rate,index):
            label=f'longbench-{split}-recovery-{index}'
            warm=await base.warmup_endpoints(specs,label)
            trace=poisson_trace(load_split(args.corpus,'longbench',split),rate,duration,seed,source='longbench')
            if not trace:
                raise RuntimeError('fixed recovery window has no offered requests: '+label)
            measured=await base.execute(specs,trace,args.out/label,duration_s=duration,slo=SLOS['longbench'],seed=seed)
            measured.update(dataset='longbench',rate_rps=rate,split=split,
                trace_sha256=file_sha(args.out/label/'requests.json'),warmup=warm,drain=await base.drain_endpoints(specs))
            write_new(args.out/label/'completion.json',measured)
            summary=dict(path=label+'/completion.json',sha256=file_sha(args.out/label/'completion.json'),
                         rate_rps=rate,split=split,seed=seed,duration_s=duration,metrics=measured['metrics'])
            result['windows'].append(summary)
            base.write(args.out/'progress.json',result)
            return summary

        def record(row):
            result['candidates'].append(row)
            write_new(args.out/f'candidate-{row["index"]}.json',row)
            base.write(args.out/'progress.json',result)
        winner,_=await search(audit['candidate_rates_rps'],observe,record)
        if winner is None:
            raise RuntimeError('LongBench recovery floor reached without independent tuning confirmation')
        confirmation=winner['tuning'];rate=winner['rate_rps']
        result['anchors']['longbench']=dict(base_rate_rps=rate,x05_rate_rps=.5*rate,model_id=MODEL,
            corpus_sha256=audit['corpus_sha256']['longbench'],
            confirmation_path='/output/anchor/'+confirmation['path'],confirmation_sha256=confirmation['sha256'],
            capacity_exact=False,scope='highest_tested_and_confirmed_passing_rate',clock_mhz=2520,tp=2,replicas=4,pp=1)
        result.update(status='passed',complete=True)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if fleet is not None:
            try:result['final_drain']=await base.drain_endpoints(specs)
            except Exception as exc:result['cleanup_errors'].append('drain: '+repr(exc))
            try:result['cleanup_errors'].extend(base.cleanup_owned(fleet,meter,sampler))
            except Exception as exc:result['cleanup_errors'].append('owned cleanup: '+repr(exc))
        elif meter is not None:
            try:meter.reset_all()
            except Exception as exc:result['cleanup_errors'].append('clock cleanup: '+repr(exc))
        if sampler is not None:
            try:
                sampler.stop()
                write_power_archive(args.out/'power.json',dict(samples=sampler.samples,
                    frequency_samples=sampler.frequency_samples,power_metadata=sampler.power_metadata,
                    power_source=sampler.power_source,error=sampler.error))
                if sampler.error:result['cleanup_errors'].append('sampler: '+sampler.error)
            except Exception as exc:result['cleanup_errors'].append('power archive: '+repr(exc))
        if result['cleanup_errors']:result.update(status='failed',complete=False)
        result['finished_s']=time.time()
        result['artifact_sha256']={str(p.relative_to(args.out)):file_sha(p) for p in args.out.rglob('*.json')}
        write_new(args.out/'rate-anchor.json',result)
        write_new(args.out/'completion.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--tp',type=int,default=2)
    parser.add_argument('--gpus',type=lambda v:[int(x) for x in v.split(',')],required=True)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--base-port',type=int,required=True);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.preflight_only:
        result=preflight(args);write_new(args.out/'preflight.json',result)
    else:result=asyncio.run(run(args))
    print(json.dumps({k:result.get(k) for k in ('status','complete','error','candidate_rates_rps')}))
    return 0 if args.preflight_only or result['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
