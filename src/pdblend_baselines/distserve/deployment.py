"""Bridge independent offline DistServe search to concrete native P/D pairs.

The first adapter supports symmetric TP / PP1 and colocated paired replicas.
It is an explicitly bounded native deployment, not a full PP reproduction.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
import time

from pdblend_runtime.probe import NativeSpec
from pdblend.results.journal import CompactJournal, payload_receipt
from .native_audit import MODELS
from .planning import best_config, binary_goodput, gpu_count
from .request_runtime import DistServeRuntime
from .run_native import load_trace, result_receipt
from .runtime import MappedDistServeTransport
from .simulator import OfficialSimulator
from .stage_surface import StageSurface


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Surfaces:
    def __init__(self, providers):self.providers=providers
    def stage_latency(self,role,tp,pp,stage,batch,inputs,contexts):
        if tp not in self.providers:raise ValueError('missing_profile: independent DistServe topology')
        return self.providers[tp].stage_latency(role,tp,pp,stage,batch,inputs,contexts)


def search_deployment(profile_paths, calibration_path, *, model, rate_rps, ttft_s, tpot_s,
                      gpu_budget=8, frequency=2520, max_per_gpu_rate=5., epsilon=.25,
                      sample_size=None):
    if model not in MODELS or not math.isfinite(rate_rps) or rate_rps<=0:
        raise ValueError('supported explicit model and positive target rate required')
    if type(gpu_budget)is not int or not 2<=gpu_budget<=8:
        raise ValueError('bounded eight-GPU deployment required')
    corpus=json.loads(Path(calibration_path).read_text())
    records=corpus.get('calibration',[])
    if not records:raise ValueError('independent calibration split is required; evaluation cannot select deployment')
    shapes=[(int(r['input_tokens']),int(r['output_tokens'])) for r in records]
    providers,bindings,missing={},[],[]
    identity=None
    for path in profile_paths:
        artifact=json.loads(Path(path).read_text())
        meta=artifact.get('identity',{})
        if meta.get('model_id')!=model:raise ValueError('DistServe stage profile model differs')
        if not artifact.get('qualified'):
            missing.append(dict(path=str(Path(path).resolve()),status='missing_profile',reason='independent calibration gates failed'))
            continue
        surface=StageSurface.load(path,frequency=frequency)
        tp=meta['tp']
        if tp in providers:raise ValueError('duplicate independent profile for one topology')
        if tp not in ((2,4) if model.endswith('32B-Instruct') else (1,2,4)) or 2*tp>gpu_budget:continue
        comparable=('model_id','model_hash','tokenizer_hash','engine_revision','image_digest')
        if identity and any(meta[k]!=identity[k] for k in comparable):
            raise ValueError('DistServe profile execution/model identities differ')
        identity=meta
        if type(meta.get('capacity_tokens'))is not int or meta['capacity_tokens']<=0:
            raise ValueError('native measured KV capacity missing')
        providers[tp]=surface
        bindings.append(dict(path=str(Path(path).resolve()),sha256=sha(path),tp=tp,pp=1,
                             profile_source_sha256=meta['source_revision']))
    result=dict(schema='distserve-native-offline-deployment-v1',system='distserve',model_id=model,
        status='missing_profile',selected=None,seed=701,gpu_budget=gpu_budget,frequency_mhz=frequency,
        rate_rps=rate_rps,slo=dict(ttft_s=ttft_s,tpot_s=tpot_s),profiles=bindings,missing=missing,
        calibration=dict(path=str(Path(calibration_path).resolve()),sha256=sha(calibration_path),split='calibration'),
        selection_split='calibration',evaluation_used_for_selection=False,
        gpu_qualified=False,formal_eligible=False,complete_reproduction=False,
        scope='qualified symmetric TP / PP1; fixed paired replica affinity',searches=[],
        excluded=['PP pipeline service/transfer','asymmetric TP KV remap','unqualified or uncovered stage profiles'])
    if not providers:return result
    simulator=OfficialSimulator(shapes,latency=_Surfaces(providers),
        capacities={(tp,1):s.identity['capacity_tokens'] for tp,s in providers.items()},seed=701,
        sample_size=sample_size,provenance=dict(split='calibration',source_sha256=sha(calibration_path)))
    for tp in sorted(providers):
        config=(1,tp,1,tp,1)
        result['searches'].append(binary_goodput(config,simulator,ttft_s=ttft_s,tpot_s=tpot_s,
            max_per_gpu_rate=max_per_gpu_rate,epsilon=epsilon))
    chosen,goodput=best_config({tuple(r['config']):r['best_per_gpu_rate'] for r in result['searches']
                              if r['status']=='complete'})
    if chosen is None or goodput<=0:
        result['status']='missing_profile' if any('missing_profile' in r.get('error','') for r in result['searches']) else 'no_feasible_deployment'
        return result
    per_pair=gpu_count(chosen);available=gpu_budget//per_pair
    requested=math.ceil(rate_rps/(goodput*per_pair));replicas=min(available,requested)
    result.update(status='ready_for_native_execution',identity=identity,
        selected=dict(config=list(chosen),tp=chosen[1],pp=1,replicas=replicas,
            per_gpu_goodput=goodput,requested_replicas=requested,total_gpu_count=replicas*per_pair,
            predicted_capacity_rps=goodput*per_pair*replicas,
            predicted_capacity_shortfall=replicas<requested),
        selection='original per-GPU goodput bisection; ties use fewer GPUs; paired replicas cover rate')
    result['deployment']=dict(pairs=[dict(prefill=f'dist-{i}-P',decode=f'dist-{i}-D') for i in range(replicas)])
    return result


def deployment_specs(plan,gpus,base_port):
    if plan.get('system')!='distserve' or plan.get('status')!='ready_for_native_execution':
        raise ValueError('DistServe offline selection has not produced a deployable plan')
    selected=plan['selected'];tp=selected['tp'];n=selected['total_gpu_count']
    if selected['config']!=[1,tp,1,tp,1] or n!=2*tp*selected['replicas']:
        raise ValueError('unsupported_engine: expected symmetric TP / PP1 replica allocation')
    if len(gpus)<n or len(set(gpus))!=len(gpus) or len(gpus)>8:
        raise ValueError('insufficient or overlapping GPU lease')
    specs=[]
    for replica in range(selected['replicas']):
        for role in ('P','D'):
            index=len(specs)
            specs.append(NativeSpec(f'dist-{replica}-{role}',tuple(gpus[index*tp:(index+1)*tp]),
                base_port+16*index,plan['model_id'],tp=tp,max_num_seqs=32,
                extra_args=('--enforce-eager',),pool_id=f'dist-pair-{replica}',
                profile_key=next(r['sha256'] for r in plan['profiles'] if r['tp']==tp)))
    return specs


async def execute_on_resident(plan,specs,trace_path,out,duration=300.,*,request_timeout=240.):
    """Consume selected resident engines; caller owns leases and energy meter."""
    if not math.isfinite(duration) or duration<=0:
        raise ValueError('positive service duration required')
    if (plan.get('system')!='distserve' or plan.get('status')!='ready_for_native_execution'
            or plan.get('selection_split')!='calibration' or plan.get('evaluation_used_for_selection') is not False):
        raise ValueError('independent calibration-only offline choice required')
    value=load_trace(trace_path)
    if value['requests'][-1]['arrival_s']>=duration:raise ValueError('trace exceeds service window')
    by_id={spec.instance_id:spec for spec in specs}
    selected=plan['selected'];tp=selected['tp']
    if (len(by_id)!=len(specs) or len(by_id)!=2*selected['replicas']
            or selected['config']!=[1,tp,1,tp,1] or selected['pp']!=1):
        raise ValueError('actual fleet differs from offline replica allocation')
    gpus=[gpu for spec in specs for gpu in spec.gpus]
    if (len(gpus)!=selected['total_gpu_count'] or len(gpus)!=len(set(gpus))
            or len(gpus)>8 or any(spec.tp!=tp or spec.pp!=1 for spec in specs)
            or set(by_id)!={f'dist-{i}-{role}' for i in range(selected['replicas']) for role in ('P','D')}):
        raise ValueError('actual fleet topology differs from independent offline choice')
    for row in plan['profiles']:
        if sha(row['path'])!=row['sha256']:raise ValueError('selected independent profile checksum changed')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    events=CompactJournal(out/'events.jsonl.gz');runtimes=[]
    def journal(pair):
        def emit(event,**fields):
            events.write(dict(event=event,replica=pair,**fields))
        return emit
    for pair in range(plan['selected']['replicas']):
        p,d=(by_id[f'dist-{pair}-{role}'] for role in ('P','D'))
        transport=MappedDistServeTransport(p.base_url,d.base_url,prefill_address=p.zmq_address,decode_address=d.zmq_address)
        runtimes.append(DistServeRuntime(transport,tp=plan['selected']['tp'],pp=1,max_batch_size=32,
                                        request_timeout_s=request_timeout,journal=journal(pair)))
    result=dict(system='distserve',status='inconclusive',complete=False,hardware_executed=False,
        formal_eligible=False,energy_comparable=False,complete_reproduction=False,seed=701,
        plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        trace_sha256=sha(trace_path),selected=plan['selected'],outcomes=[],cleanup_errors=[],
        scope=plan['scope'],gpu_batch_equivalence_qualified=False)
    tasks=[]
    try:
        result['capabilities']=await asyncio.gather(*(r.start() for r in runtimes))
        expected=plan['identity']
        for pair in result['capabilities']:
            for cap in pair.values():
                if any(cap.get(k)!=expected[k] for k in ('model_id','model_hash','tokenizer_hash','engine_revision','image_digest')):
                    raise ValueError('selected profile and deployed engine identity differ')
                if any(k in value and value[k]!=cap[k] for k in ('model_id','tokenizer_hash')):
                    raise ValueError('frozen evaluation trace model identity differs')
        result['hardware_executed']=True;started=time.monotonic();result['service_started_s']=time.time()
        async def request(index,row):
            await asyncio.sleep(max(0.,started+row['arrival_s']-time.monotonic()))
            # Author dispatch signal is queued plus in-flight prefill count.
            # Pair affinity is explicit; no unverified cross-pair KV scheduler.
            pair=min(range(len(runtimes)),key=lambda i:(len(runtimes[i].prefill.waiting)+len(runtimes[i].prefill.processing),i))
            runtime=runtimes[pair];rid=f'distserve-701-{index}'
            outcome=dict(request_id=rid,replica=pair,arrival_s=row['arrival_s'],submitted_s=time.time(),
                         scheduled_s=result['service_started_s']+row['arrival_s'],
                         input_tokens=len(row['prompt']),output_tokens=row['max_tokens'],events=[],ok=False)
            try:
                async for event in runtime.handle(dict(row,seed=701,ignore_eos=True,temperature=0),rid):
                    outcome['events'].append(event)
                native=runtime.results[rid]
                steps={receipt['step'] for receipt in native.receipts}
                expected_steps=({'prefill','release'} if row['max_tokens']==1 else
                                {'prefill','expect_load','transfer','load_ack','release'})
                outcome.update(result=result_receipt(native),native_receipts_complete=steps==expected_steps,
                               ok=native.status=='completed' and native.tokens==row['max_tokens']
                               and bool(outcome['events'] and outcome['events'][-1].get('finished'))
                               and steps==expected_steps)
            except Exception as exc:outcome['error']=repr(exc)
            arrivals=[event['received_s'] for event in outcome['events'] for _ in event['token_ids']]
            outcome['ttft_s']=arrivals[0]-outcome['scheduled_s'] if arrivals else None
            outcome['tpot_s']=(arrivals[-1]-arrivals[0])/(len(arrivals)-1) if len(arrivals)>1 else None
            outcome['finished_s']=time.time()
            outcome.update(payload_receipt(outcome.pop('events'),journal_path='events.jsonl.gz',request_id=rid))
            if 'result' in outcome:
                outcome['result'].pop('events',None);outcome['result'].pop('token_ids',None)
            result['outcomes'].append(outcome)
        tasks=[asyncio.create_task(request(i,row)) for i,row in enumerate(value['requests'])]
        await asyncio.gather(*tasks)
        await asyncio.sleep(max(0.,started+duration-time.monotonic()))
        result['service_finished_s']=time.time()
        result['complete']=len(result['outcomes'])==len(value['requests']) and all(r['ok'] for r in result['outcomes'])
        result['status']='passed' if result['complete'] else 'failed'
    except BaseException as exc:result.update(status='failed',error=repr(exc))
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        for runtime in runtimes:
            try:await runtime.close()
            except BaseException as exc:result['cleanup_errors'].append(repr(exc))
        if result['cleanup_errors']:result.update(status='failed',complete=False)
        events.close();result['events_sha256']=sha(out/'events.jsonl.gz')
        result['journal_path']='events.jsonl.gz'
        (out/'completion.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


execute_deployment=execute_on_resident
