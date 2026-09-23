"""Independent PP1 P/D scheduler and native execution contract probe."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from .runtime import MappedDistServeTransport, DistServeCapabilityError
from .policy import Request, PrefillScheduler, DecodeScheduler


def ack(value,name):
    if not isinstance(value,dict) or value.get('acknowledged') is not True:
        raise DistServeCapabilityError('missing '+name+' acknowledgement')
    return value

def load_ack(value, identity, tp):
    value=ack(value,'D load')
    for key in ('request_id','transaction_id','generation','ranks'):
        if key not in value: raise DistServeCapabilityError('D load receipt missing '+key)
    if value['request_id']!=identity['target_request_id'] or value['transaction_id']!=identity['transaction_id'] or value['generation']!=identity['generation']:
        raise DistServeCapabilityError('D load receipt identity/generation mismatch')
    ranks=value['ranks']
    if not isinstance(ranks,list) or len(ranks)!=tp or sorted(r.get('rank') for r in ranks)!=list(range(tp)):
        raise DistServeCapabilityError('D load rank ACK coverage incomplete')
    if any(r.get('acknowledged') is not True or r.get('generation')!=identity['generation'] or r.get('transaction_id')!=identity['transaction_id'] or r.get('target_request_id')!=identity['target_request_id'] or not isinstance(r.get('expected_layers'),int) or r.get('expected_layers')<1 or r.get('loaded_layers')!=r.get('expected_layers') for r in ranks):
        raise DistServeCapabilityError('D load rank ACK identity mismatch')
    return value


async def run(args):
    transport=MappedDistServeTransport(args.prefill_url,args.decode_url)
    started=time.monotonic();arrival=time.time();receipts=[];events=[];cleanup=[]
    rid=args.request_id;tx='tx-'+rid;prompt=list(range(100,100+args.prompt_tokens))
    p=None;d_id=None;released=False;carry_tokens=[];first_at=None;error=None
    carry=bool(args.carry_first_token);target_prompt=len(prompt)+int(carry)
    ps=ds=None
    try:
        if args.max_tokens<2:raise ValueError('mechanism probe needs at least two output tokens')
        caps={};states={}
        for role in ('P','D'):
            cap=await transport.capability(role);state=await transport.state(role)
            if cap.get('supported') is not True or state.get('native_evidence_complete') is not True:
                raise DistServeCapabilityError(role+' native evidence unavailable')
            if cap.get('tp')!=args.tp or cap.get('pp')!=1:
                raise DistServeCapabilityError('symmetric PP1 topology required')
            if state.get('generation')!=args.generation or state.get('acknowledged_generation')!=args.generation:
                raise DistServeCapabilityError('P/D generation mismatch')
            if any(type(state.get(k)) is not int or state[k]<=0 for k in ('free_blocks','block_size','max_num_batched_tokens')):
                raise DistServeCapabilityError(role+' KV/batch capacity receipt missing')
            caps[role]=cap;states[role]=state
        for key in ('model_hash','tokenizer_hash','image_digest','source_revision'):
            if not caps['P'].get(key) or caps['P'][key]!=caps['D'].get(key):
                raise DistServeCapabilityError('P/D '+key+' identity missing or different')
        ps=PrefillScheduler(max_batch_size=1,max_tokens_per_batch=states['P']['max_num_batched_tokens'],
            num_gpu_blocks=states['P']['free_blocks'],block_size=states['P']['block_size'])
        ds=DecodeScheduler(max_batch_size=1,max_tokens_per_batch=states['D']['max_num_batched_tokens'],
            num_gpu_blocks=states['D']['free_blocks'],block_size=states['D']['block_size'])
        request=Request(rid,len(prompt),args.max_tokens,{'prompt_token_ids':prompt});ps.add(request)
        batch=ps.next_batch(free_gpu_blocks=states['P']['free_blocks'])
        if len(batch)!=1 or batch[0] is not request:
            raise DistServeCapabilityError('independent P scheduler did not admit request')
        receipts.append(dict(step='p_admission',request_id=rid,observed_capacity=states['P']))
        p=ack(await transport.prefill(dict(request_id=rid,prompt_token_ids=prompt,seed=701)),'P prefill')
        receipts.append(dict(step='prefill',receipt=p));ps.complete(batch)
        source_address=p.get('source_address')
        if not source_address:raise DistServeCapabilityError('P omitted KV source address')
        first=[token for row in p.get('outputs',[]) for token in row.get('token_ids',[])]
        if carry and len(first)!=1:raise DistServeCapabilityError('P omitted exactly one real first output token')
        if carry:
            carry_tokens=first
            first_at=next((row.get('at_s') for row in p['outputs'] if row.get('token_ids')),None)
        request.kv_handle=p['retained_handle'];ds.add_bridge(request)
        if ds.accept_next(free_gpu_blocks=states['D']['free_blocks']) is not request:
            raise DistServeCapabilityError('independent D scheduler did not select completed prefill')
        d_id=f'{rid}-D___prefill_addr_{source_address}___decode_addr_{args.target_address}_tag'
        receipts.append(dict(step='d_selection',request_id=rid,target_request_id=d_id,observed_capacity=states['D']))
        identity=dict(target_request_id=d_id,transaction_id=tx,generation=args.generation)
        expected=ack(await transport.expect_load(dict(identity,source_tokens=len(prompt))),'D expect_load')
        receipts.append(dict(step='expect_load',receipt=expected))
        sent=ack(await transport.transfer(dict(identity,held_request_id=p['retained_handle'],
                    target_address=args.target_address,target_tp=args.tp)),'P transfer')
        receipts.append(dict(step='transfer',receipt=sent))
        async for event in transport.generate_decode(dict(request_id=d_id,prompt=prompt+carry_tokens,
                        max_tokens=args.max_tokens-int(carry),seed=701)):
            events.append(event)
        if not events or not events[-1].get('finished'):raise DistServeCapabilityError('D stream incomplete')
        tokens=carry_tokens+[token for event in events for token in event.get('token_ids',[])]
        if len(tokens)!=args.max_tokens:raise DistServeCapabilityError('D output token count mismatch')
        if first_at is None:first_at=next((event.get('at_s') for event in events if event.get('token_ids')),None)
        receipts.append(dict(step='load_ack',receipt=load_ack(await transport.load_ack(identity),identity,args.tp)))
        receipts.append(dict(step='release',receipt=ack(await transport.release(dict(held_request_id=p['retained_handle'])),'P release')))
        released=True;ds.finish(rid);ps.release(rid)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    finally:
        if d_id and error:
            try:cleanup.append(dict(step='cancel_decode',receipt=await transport.cancel('D',dict(request_id=d_id))))
            except Exception as exc:cleanup.append(dict(step='cancel_decode',error=str(exc),quarantined=True))
        if p and p.get('retained_handle') and not released:
            try:cleanup.append(dict(step='release_source',receipt=await transport.release(dict(held_request_id=p['retained_handle']))))
            except Exception as exc:cleanup.append(dict(step='release_source',error=str(exc),quarantined=True))
    artifact=dict(schema='distserve-mechanism-smoke-v2',status='failed' if error else 'passed',complete=not error,
        mechanism_validated=not error,formal_eligible=False,energy_comparable=False,
        mode='carry_first_token' if carry else 'legacy',request_id=rid,transaction_id=tx,
        events=events,receipts=receipts,cleanup=cleanup,source_tokens=len(prompt),target_prompt_tokens=target_prompt,
        combined_token_ids=carry_tokens+[token for event in events for token in event.get('token_ids',[])],
        first_token_at_s=first_at,ttft_s=first_at-arrival if first_at is not None else None,
        finished=bool(events and events[-1].get('finished')),elapsed_s=time.monotonic()-started,error=error)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(artifact,indent=2)+'\n')
    if error:raise RuntimeError(error)
    return artifact


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prefill-url',required=True);p.add_argument('--decode-url',required=True)
    p.add_argument('--target-address',required=True);p.add_argument('--tp',type=int,default=1);p.add_argument('--generation',type=int,default=2)
    p.add_argument('--prompt-tokens',type=int,default=128);p.add_argument('--max-tokens',type=int,default=32)
    p.add_argument('--request-id',default='distserve-smoke');p.add_argument('--carry-first-token',action='store_true');p.add_argument('--out',required=True)
    print(json.dumps(asyncio.run(run(p.parse_args(argv))),sort_keys=True))


if __name__=='__main__':main()
