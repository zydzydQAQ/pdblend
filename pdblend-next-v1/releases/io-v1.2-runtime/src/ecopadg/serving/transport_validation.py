"""Exact KV tensor/cache validation, separate from cross-TP BF16 numerics."""
import argparse
import asyncio
import json
from pathlib import Path
import uuid

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler
from .backend import ClockOwner
from .campaign import node_lease
from .profiling import HardwareProfiler
from .measurement import power_evidence


def check_events(events,nonce,source_tp,target_tp):
    events=[e for e in events if e.get('nonce')==nonce or e.get('tensor_id','').startswith(nonce+'#')]
    def tensors(direction):
        selected=[e for e in events if e['direction']==direction]
        return selected,{(e['tensor_id'],e['target_rank']):(e['sha256'],e['shape'],e['dtype']) for e in selected}
    sends,exported=tensors('export');receives,imported=tensors('import')
    expected=2*max(source_tp,target_tp)+target_tp
    readback=[e for e in events if e['direction']=='cache_readback']
    return dict(tensor_count=len(exported),expected_tensor_count=expected,
        transport_bit_exact=len(sends)==len(receives)==len(exported)==expected and exported==imported,
        cache_bit_exact=len(readback)==target_tp and {e['target_rank'] for e in readback}==set(range(target_tp))
                        and all(e['bit_exact'] for e in readback))


async def validate(args):
    topology=json.loads(args.topology.read_text())
    a,b=topology['prefill'],topology['decode']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists(): raise ValueError('refusing to overwrite transport evidence')
    hardware=await asyncio.to_thread(PynvmlBackend,power_mode='instant');clocks=ClockOwner(hardware,range(8))
    sampler=PowerSampler(range(8),interval=.02,backend=hardware,sample_clocks=True)
    result=dict(cases=[],complete=False,purpose='diagnostic tensor hash/cache readback; not performance')
    sampler.start()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),trust_env=False) as session:
        profiler=HardwareProfiler(session,topology,args.runtime_dir)
        async def control(instance,role):
            raw=await profiler.call(instance,'/runtime')
            if not raw.get('diagnostic_transport') or raw.get('diagnostic_recompute'):
                raise ValueError('bit-exact validation requires transport diagnostics and no recomputation')
            return await profiler.call(instance,'/control',dict(generation=raw['generation']+1,
                role=role,mode='continuous',admit_prefill=True,admit_decode=True))
        def records(nonce):
            values=[]
            for instance in (a,b):
                for path in args.runtime_dir.glob(instance['id']+'.control.json.transport-validation.*.jsonl'):
                    values.extend(json.loads(line) for line in path.read_text().splitlines()
                                  if nonce in line)
            return values
        try:
            result['engine_provenance']=await profiler.provenance()
            await clocks.set(sorted(set(a['gpus']+b['gpus'])),2520,verify_rise=False)
            await asyncio.sleep(.1)
            if sampler.error or len(sampler.samples)<2:
                raise RuntimeError('instant transport diagnostic power preflight failed: '+str(sampler.error))
            for source,target in ((a,b),(b,a)):
                await profiler.call(source,'/prepare-peers',dict(peers=[target['id']]))
                for n in args.inputs:
                    body=dict(prompt=([9707,1879,13]*(n//3+1))[:n],max_tokens=32,
                              temperature=0,ignore_eos=True,stream=False)
                    await control(source,'mixed');reference=await profiler.call(source,'/v1/completions',body)
                    await control(target,'mixed');target_reference=await profiler.call(target,'/v1/completions',body)
                    await control(source,'prefill');await control(target,'decode')
                    nonce=uuid.uuid4().hex
                    producer=await profiler.call(source,'/v1/completions',dict(body,max_tokens=1),
                        f"pdb:{nonce}:p:{source['id']}:{target['id']}")
                    decoded=await profiler.call(target,'/v1/completions',body,
                        f"pdb:{nonce}:d:{source['id']}:{target['id']}")
                    case=check_events(await asyncio.to_thread(records,nonce),nonce,source['tp'],target['tp'])
                    case.update(source_tp=source['tp'],target_tp=target['tp'],input_tokens=n,nonce=nonce,
                        actual_output_tokens=decoded['usage']['completion_tokens'],
                        first_token_equal=producer['token_ids']==decoded['token_ids'][:1],
                        matches_source=decoded['token_ids']==reference['token_ids'],
                        matches_target=decoded['token_ids']==target_reference['token_ids'])
                    case['passed']=(case['transport_bit_exact'] and case['cache_bit_exact'] and case['first_token_equal']
                        and case['actual_output_tokens']==len(decoded['token_ids'])==32
                        and (source['tp']!=target['tp'] or (case['matches_source'] and case['matches_target'])))
                    result['cases'].append(case);print(json.dumps(case),flush=True)
                    if not case['passed']: raise RuntimeError('KV transport correctness failed')
            result['complete']=True
        finally:
            await asyncio.to_thread(sampler.stop)
            await clocks.close()
            result.update(power_samples=sampler.samples,power_source=sampler.power_source,
                power_metadata=sampler.power_metadata,frequency_samples=sampler.frequency_samples,sampling_error=sampler.error)
            proof=power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
            result['power_source_verified']=proof['power_source_verified']
            result['passed']=(result['complete'] and all(c['passed'] for c in result['cases'])
                              and not sampler.error and proof['power_source_verified'])
            await asyncio.to_thread(args.out.write_text,json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('transport diagnostic incomplete or invalid power source')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology',type=Path,required=True)
    parser.add_argument('--runtime-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inputs',type=int,nargs='+',default=[32,256,1024,4096,7168])
    args=parser.parse_args()
    if min(args.inputs)<1 or max(args.inputs)+32>8192: parser.error('inputs must fit the configured context')
    with node_lease(): asyncio.run(validate(args))


if __name__=='__main__': main()
