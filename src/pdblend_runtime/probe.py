"""Real GPU integration probe for native V1 control, measurement and retained KV.

This produces mechanism evidence, never a formal five-system energy ranking.
One loaded P/D pair is reused throughout the probe and baseline measurements.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import aiohttp

from pdblend.engine.launcher import Fleet, InstanceSpec
from pdblend.bench.metering import Gpus
from pdblend.bench.gates import random_prompt


class NativeSpec(InstanceSpec):
    def command(self):
        return [sys.executable, '-B', '-m', 'pdblend_runtime.serve', *super().command()[2:]]


async def call(session, url, path, body=None):
    async with session.request('GET' if body is None else 'POST', url+path, json=body) as response:
        text=await response.text()
        if response.status != 200:
            raise RuntimeError(f'{path} {response.status}: {text[:1500]}')
        return json.loads(text)


async def generate(session, url, payload):
    events=[]
    async with session.post(url+'/baseline/generate', json=payload) as response:
        if response.status != 200:
            raise RuntimeError(await response.text())
        async for line in response.content:
            if not line.startswith(b'data:'):
                continue
            data=line[5:].strip()
            if data == b'[DONE]':
                break
            events.append(json.loads(data))
    if not events or not events[-1].get('finished'):
        raise RuntimeError('native generation has no terminal event')
    return dict(events=events, token_ids=[token for row in events for token in row['token_ids']])


async def probe(specs, result, out):
    timeout=aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        caps=[]
        for spec in specs:
            cap=await call(session,spec.base_url,'/baseline/capability')
            caps.append(cap)
            if not cap['supported'] or cap['tp'] != spec.tp:
                raise RuntimeError('native capability incomplete')
        result['capabilities']=caps
        for spec in specs:
            outcomes=[]
            for repeat in range(2):
                outcomes.append(await generate(session,spec.base_url,dict(request_id=f'ordinary-{spec.instance_id}-{repeat}',
                    prompt=list(range(100,228)),max_tokens=16,seed=701)))
            if outcomes[0]['token_ids'] != outcomes[1]['token_ids']:
                raise RuntimeError('ordinary deterministic reference differs')
            result.setdefault('ordinary',[]).append(dict(instance=spec.instance_id,outcomes=outcomes))
        # Native CUDA measurements for each independent system own their
        # actual request and timing sample. These probes are not complete fits.
        for system,scope in [('distserve','runner'),('ecoserve','forward'),('dynamollm','runner')]:
            spec=specs[0]
            await call(session,spec.base_url,'/baseline/measurement/start',dict(system=system,scope=scope))
            response=await generate(session,spec.base_url,dict(request_id='native-profile-'+system,
                prompt=list(range(100,612)),max_tokens=1 if system=='ecoserve' else 32,seed=701))
            samples=await call(session,spec.base_url,'/baseline/measurement/samples')
            if len(samples['ranks'])!=spec.tp or any(not r['samples'] for r in samples['ranks']):
                raise RuntimeError('missing CUDA event samples on native ranks')
            result.setdefault('independent_measurement_probes',{})[system]=dict(response=response,**samples)
        await call(session,specs[0].base_url,'/baseline/measurement/stop',{})
        # EcoServe's own author-style CSV uses five full-forward invocations
        # per anchor and no PDBlend fitted model. Keep this resident pair.
        from pdblend_baselines.native_profile import collect as collect_baseline
        eco_path=out/'ecoserve-prefill.csv'
        eco_args=SimpleNamespace(url=specs[0].base_url,system='ecoserve',model=caps[0]['model_id'],
            out=str(eco_path),frequency=2520,
            **{key:caps[0][key] for key in ('model_hash','tokenizer_hash','engine_version','image_digest','source_revision','gpu_uuids')})
        result['ecoserve_profile']=await asyncio.to_thread(collect_baseline,eco_args)
        from pdblend_baselines.ecoserve.mechanism_smoke import run as eco_smoke
        eco_config=dict(instances=[dict(id=s.instance_id,gpus=list(s.gpus)) for s in specs],
            eco_prefill_csv=str(eco_path),slo_ttft_s=5.,slo_tpot_s=.15,eco_initial_instances=2,
            eco_active_frequency_mhz=2520,eco_state_poll_s=.05,eco_scale_period_s=60,request_timeout_s=120)
        result['ecoserve_admission']=await eco_smoke(eco_config,{s.instance_id:s.base_url for s in specs},
            random_prompt(512,51200),'eco-native-admission',str(out/'ecoserve-admission.json'))
        # Exercise owner-thread control while requests are held, then restore
        # execution and verify that cancel removes the native scheduler state.
        spec=specs[0]
        await call(session,spec.base_url,'/baseline/control',dict(generation=1,admit_prefill=False,admit_decode=False))
        task=asyncio.create_task(generate(session,spec.base_url,dict(request_id='cancel-held',prompt=list(range(100,612)),max_tokens=128)))
        await asyncio.sleep(.2)
        held=await call(session,spec.base_url,'/baseline/state')
        if 'cancel-held' not in held['all_queue']:
            raise RuntimeError('scheduler did not retain controlled request')
        cancelled=await call(session,spec.base_url,'/baseline/cancel',dict(request_id='cancel-held'))
        task.cancel(); await asyncio.gather(task,return_exceptions=True)
        await call(session,spec.base_url,'/baseline/control',dict(generation=2,admit_prefill=True,admit_decode=True))
        result['cancel']=cancelled
        # DistServe picks a decoder after the prefill has retained native KV.
        p,d=specs[:2]
        await call(session,d.base_url,'/baseline/control',dict(generation=2))
        for length in (512,2048,7168):
            for repeat in range(3):
                prompt=random_prompt(length,100*length)
                tag=f'kv-{length}-{repeat}'
                references=[await generate(session,d.base_url,dict(request_id=tag+f'-ref-{i}',prompt=prompt,max_tokens=16,seed=701)) for i in range(2)]
                saved=await call(session,p.base_url,'/baseline/distserve/prefill',dict(request_id=tag,prompt=prompt))
                first=[token for row in saved['outputs'] for token in row['token_ids']]
                if len(first)!=1:raise RuntimeError('prefill must produce exactly one real first token')
                try:
                    for protocol in ('legacy_recompute_last_prompt','carry_first_token'):
                        tx=tag+'-'+protocol
                        target_id=f'___prefill_addr_{p.zmq_address}___decode_addr_{d.zmq_address}_{tx}'
                        digest=repeat==0
                        expected=await call(session,d.base_url,'/baseline/distserve/expect_load',dict(
                            target_request_id=target_id,transaction_id=tx,generation=2,source_tokens=length,kv_digest=digest))
                        sent=await call(session,p.base_url,'/baseline/distserve/transfer',dict(held_request_id=saved['retained_handle'],
                            target_request_id=target_id,target_address=d.zmq_address,target_tp=d.tp,generation=2,transaction_id=tx,kv_digest=digest))
                        carried=protocol=='carry_first_token'
                        decoded=await generate(session,d.base_url,dict(request_id=target_id,prompt=prompt+first if carried else prompt,
                            max_tokens=15 if carried else 16,seed=701))
                        load=await call(session,d.base_url,'/baseline/distserve/load_ack',dict(
                            target_request_id=target_id,transaction_id=tx,generation=2))
                        if (not load.get('acknowledged') or load.get('generation')!=2 or
                                load.get('transaction_id')!=tx or len(load.get('ranks',[]))!=d.tp):
                            raise RuntimeError('missing native load ACK identity')
                        comparisons=[]
                        if digest:
                            from .kv_digest import compare_digests
                            for source in sent['ranks']:
                                target=next(row for row in load['ranks'] if row['rank']==source['rank'])
                                for layer in source['layer_receipts']:
                                    received=target['digests'][str(layer['layer'])]
                                    comparisons.append(compare_digests(layer['source_digest'],received['received'],received['injected']))
                            if not comparisons or not all(c['exact_payload_match'] for c in comparisons):
                                raise RuntimeError('source/received/injected KV bytes differ')
                        combined=(first if carried else [])+decoded['token_ids']
                        row=dict(length=length,repeat=repeat,protocol=protocol,prefill=saved,expected=expected,
                            transfer=sent,decode=decoded,load=load,prompt_seed=100*length,prompt=prompt,
                            references=references,combined_token_ids=combined,kv_digests=comparisons,
                            reference_stable=references[0]['token_ids']==references[1]['token_ids'],
                            prefill_first_matches_reference=first[0]==references[0]['token_ids'][0],
                            tokens_match=references[0]['token_ids']==combined)
                        result.setdefault('kv',[]).append(row)
                        (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
                finally:
                    release=await call(session,p.base_url,'/baseline/distserve/release',dict(held_request_id=saved['retained_handle']))
                    result.setdefault('kv_releases',[]).append(release)
        # Legacy recomputation is diagnostic. The corrected protocol carries
        # the actual first output from prefill and checks all 16 output IDs.
        carried=[r for r in result['kv'] if r['protocol']=='carry_first_token']
        result['legacy_output_match']=all(r['tokens_match'] for r in result['kv'] if r['protocol']=='legacy_recompute_last_prompt')
        result['carry_first_token_output_match']=len(carried)==9 and all(r['tokens_match'] and r['reference_stable'] for r in carried)
        if not result['carry_first_token_output_match']:
            raise RuntimeError('strict carry-first-token output golden failed')
        from pdblend_baselines.distserve.mechanism_smoke import run as distserve_smoke
        result['distserve_scheduler']=await distserve_smoke(SimpleNamespace(
            prefill_url=p.base_url,decode_url=d.base_url,target_address=d.zmq_address,tp=d.tp,
            generation=2,prompt_tokens=512,max_tokens=32,request_id='distserve-native-scheduler',
            carry_first_token=True,out=str(out/'distserve-scheduler.json')))
        # Reuse the two resident instances for independent baseline samples.
        # Each collector owns its inputs, fit, raw artifacts and GPU clocks.
        # This sparse development envelope does not qualify a full campaign.
        (out/'native-primitives.json').write_text(json.dumps(dict(
            status='passed',complete=True,formal_eligible=False,
            validated=['ordinary_output','native_control_cancel','same_tp_kv_bytes',
                       'strict_carry_first_token','distserve_scheduler','ecoserve_admission'],
            source_sha256=result['source_sha256'],image_digest=result['image_digest'],
            at_s=time.time()),indent=2)+'\n')
        # Check the public PDBlend client/router while the same pair is hot.
        # Its qualification is separate from the independent baseline chain.
        from .public_pd_probe import run as public_pd_smoke
        result['public_mixed_pd_protocol']=await public_pd_smoke(specs,out/'public-mixed-pd.json')
        dist_args=SimpleNamespace(url=p.base_url,system='distserve',model=caps[0]['model_id'],
            out=str(out/'distserve-profile.json'),batches=[1,4,8],contexts=[128,512,2048,4096],
            **{key:caps[0][key] for key in ('model_hash','tokenizer_hash','engine_version','image_digest','source_revision','gpu_uuids')})
        from pdblend_baselines.dynamollm.profile_v1 import collect as collect_dynamo
        dyn_args=SimpleNamespace(model=Path(d.model),gpus=list(d.gpus),tp=d.tp,base_port=d.port,
            instance_id=d.instance_id,existing_url=d.base_url,out=out/'dynamollm-profile',
            freqs=[900,1200,1500,1800,2100,2520],inputs=[512],outputs=[64],batches=[1],
            settle=2.,measure=5.,resume=False,label_corpus_root=None)
        outcomes=await asyncio.gather(asyncio.to_thread(collect_baseline,dist_args),
                                      collect_dynamo(dyn_args),return_exceptions=True)
        for system, outcome in zip(('distserve','dynamollm'),outcomes):
            result.setdefault('independent_profiles',{})[system]=dict(
                error=repr(outcome)) if isinstance(outcome,BaseException) else outcome
        dyn_path=out/'dynamollm-profile/completion.json'
        dyn_completion=json.loads(dyn_path.read_text()) if dyn_path.is_file() else {}
        result['independent_profile_collection']=dict(
            distserve='failed' if isinstance(outcomes[0],BaseException) else 'native_samples_collected',
            dynamollm=dyn_completion.get('status','failed'),
            dynamollm_holdout_passed=dyn_completion.get('complete',False),
            formal_profile_qualified=False)
        (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        # Calibration errors retain their own inconclusive status. They do
        # not invalidate already proven native primitives; actual drain or
        # engine failures below still fail this mechanism job.
        result['drain']=[await call(session,s.base_url,'/baseline/drain',dict(timeout_s=30)) for s in specs]
        result['events']=[await call(session,s.base_url,'/baseline/events?after_seq=0') for s in specs]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--gpus',required=True);parser.add_argument('--base-port',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); gpus=[int(x) for x in args.gpus.split(',')]
    if len(gpus)!=2*args.tp:
        raise ValueError('native probe requires two disjoint TP groups')
    # Each TP rank binds KV port + rank. Reserve a separate range per engine.
    specs=[NativeSpec(f'native{i}',tuple(gpus[i*args.tp:(i+1)*args.tp]),args.base_port+16*i,
                      args.model,tp=args.tp,max_num_seqs=32,extra_args=('--enforce-eager',)) for i in range(2)]
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',complete=False,formal_eligible=False,energy_comparable=False,
                model=args.model,tp=args.tp,pp=1,seed=701,started_s=time.time(),specs=[asdict(s) for s in specs],
                image_digest=os.environ.get('PDBLEND_IMAGE_ID'),source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'),
                evidence_class='native_execution_probe',mechanism_qualification='pending_gpu')
    meter=Gpus(gpus);sampler=meter.sampler(interval_s=.1);fleet=Fleet(specs,out/'logs')
    try:
        sampler.start()
        result['startup']=fleet.start_all(timeout_s=1200)
        asyncio.run(probe(specs,result,out))
        result.update(status='passed',complete=True,mechanism_qualification='native_primitives_passed')
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        from .cleanup import cleanup_owned
        result['cleanup_errors']=cleanup_owned(fleet,meter,sampler)
        if result['cleanup_errors']:
            result.update(status='failed',complete=False)
        result.update(finished_s=time.time(),energy_j=sampler.total_energy_j(),sampler_error=sampler.error)
        (out/'power.json').write_text(json.dumps(dict(samples=sampler.samples,frequency_samples=sampler.frequency_samples)))
        (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','model','tp','error')},indent=2),flush=True)
    raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
