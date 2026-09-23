"""Public Mixed/PDBlend protocol checks on an already loaded native pair.

Token diagnostics qualify correctness only.  These requests are not service
energy/profile measurements or an automatic planner qualification.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from aiohttp import web

from pdblend.bench.gates import random_prompt
from pdblend.engine.client import EngineClient, PDTransfer, pd_complete
from pdblend.proxy.router import Router
from pdblend.proxy.server import Proxy
from pdblend.results.receipts import request_record_receipt as record_receipt


def check_completion(value, expected, *, prompt_tokens, output_tokens):
    """Usage and exact IDs must agree, including empty/special token text."""
    return dict(
        no_error=value.error is None,
        stream_done=value.stream_done is True,
        usage_received=value.usage_received is True,
        token_ids_match=value.token_ids == expected,
        output_count=value.completion_tokens == output_tokens,
        exact_ids_present=isinstance(value.token_ids, list) and len(value.token_ids) == output_tokens,
        original_prompt_count=value.prompt_tokens == prompt_tokens,
        first_token_present=value.first_token_s is not None,
        finished=value.finished_s is not None,
    )


async def run(specs, out: Path):
    p,d=specs[:2]
    transfer=PDTransfer('P2pNcclConnector',{s.instance_id:s.zmq_address for s in specs})
    metadata={s.instance_id:dict(tp=s.tp,pp=s.pp,pool_id='public-smoke',model_id=Path(s.model).name)
              for s in specs}
    router=Router([s.instance_id for s in specs],pd_threshold_tokens=1,instance_metadata=metadata)
    proxy=Proxy({s.instance_id:s.base_url for s in specs},router=router,transfer=transfer)
    runner=web.AppRunner(proxy.app)
    result=dict(status='running',complete=False,formal_eligible=False,energy_comparable=False,
                automatic_planner_qualified=False,diagnostic_logprobs=True,seed=701,
                scope='public_mixed_and_symmetric_pd_protocol',started_s=time.time(),rows=[])

    def save():
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')

    try:
        await runner.setup()
        port=p.port+48
        await web.TCPSite(runner,'127.0.0.1',port).start()
        async with EngineClient(p.instance_id,p.base_url) as pc, \
                   EngineClient(d.instance_id,d.base_url) as dc, \
                   EngineClient('public-proxy',f'http://127.0.0.1:{port}') as public:
            for length in (512,2048,7168):
                prompt=random_prompt(length,100*length)
                refs=[await dc.complete(prompt,16,f'public-ref-{length}-{r}',
                                        token_diagnostics=True,seed=701) for r in range(2)]
                refchecks=[check_completion(x,refs[0].token_ids,prompt_tokens=length,output_tokens=16) for x in refs]
                row=dict(input_tokens=length,reference=[asdict(x) for x in refs],reference_checks=refchecks)
                result['rows'].append(row)
                save()
                if not all(all(c.values()) for c in refchecks):
                    raise RuntimeError('public reference is incomplete or not repeatable')
                pre,combined=await pd_complete(transfer,pc,dc,prompt,16,f'public-client-{length}',
                                               token_diagnostics=True,seed=701)
                row['client_prefill']=asdict(pre)
                row['client_pd']=asdict(combined) if combined else None
                row['client_checks']=(check_completion(combined,refs[0].token_ids,
                    prompt_tokens=length,output_tokens=16) if combined else dict(no_error=False))
                save()
                if not all(row['client_checks'].values()):
                    raise RuntimeError('public EngineClient carry-first-token golden failed')
                for mode,roles in (('PD',{p.instance_id:'P',d.instance_id:'D'}),
                                   ('M',{p.instance_id:'M',d.instance_id:'M'})):
                    router.set_roles(roles,pd_threshold_tokens=1)
                    value=await public.complete(prompt,16,f'public-proxy-{mode}-{length}',
                        token_diagnostics=True,pdblend_token_diagnostics=True,seed=701)
                    record=router.records[-1] if router.records else None
                    checks=check_completion(value,refs[0].token_ids,prompt_tokens=length,output_tokens=16)
                    checks['route_matches']=record is not None and record.path==mode
                    checks['router_accounting']=(record is not None and record.error is None and
                                                record.completion_tokens==16 and record.input_tokens==length)
                    row['proxy_'+mode]=dict(completion=asdict(value),record=record_receipt(record),checks=checks)
                    save()
                    if not all(checks.values()):
                        raise RuntimeError('public proxy '+mode+' token/usage/route golden failed')
            # A one-token completion must not create an unconsumed remote KV.
            prompt=random_prompt(128,701)
            single_ref=await pc.complete(prompt,1,'public-single-reference',token_diagnostics=True,seed=701)
            reference_checks=check_completion(single_ref,single_ref.token_ids,prompt_tokens=128,output_tokens=1)
            result['single_token']=dict(reference=asdict(single_ref),reference_checks=reference_checks)
            save()
            if not all(reference_checks.values()):
                raise RuntimeError('single-token reference is incomplete')
            _,single=await pd_complete(transfer,pc,dc,prompt,1,'public-single-client',token_diagnostics=True,seed=701)
            checks=check_completion(single,single_ref.token_ids,prompt_tokens=128,output_tokens=1)
            checks['no_remote_protocol']=single.pd_protocol=='single_engine_no_remote_kv'
            result['single_token'].update(completion=asdict(single),checks=checks)
            save()
            if not all(checks.values()):
                raise RuntimeError('single-token no-remote-KV path failed')
            router.set_roles({p.instance_id:'P',d.instance_id:'D'},pd_threshold_tokens=1)
            single_proxy=await public.complete(prompt,1,'public-proxy-single',token_diagnostics=True,
                pdblend_token_diagnostics=True,seed=701)
            record=router.records[-1] if router.records else None
            proxy_checks=check_completion(single_proxy,single_ref.token_ids,prompt_tokens=128,output_tokens=1)
            proxy_checks['route_matches']=(record is not None and record.path=='P_ONLY' and
                record.prefill_instance==record.decode_instance==p.instance_id)
            proxy_checks['router_accounting']=(record is not None and record.error is None and
                record.completion_tokens==1 and record.input_tokens==128 and
                all(load.inflight_seqs==load.inflight_prefill_tokens==0 for load in router.loads.values()) and
                not any(router.active.values()))
            result['single_token']['proxy']=dict(completion=asdict(single_proxy),
                record=record_receipt(record),checks=proxy_checks)
            save()
            if not all(proxy_checks.values()):
                raise RuntimeError('single-token proxy P_ONLY path failed')
        result.update(status='passed',complete=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
    finally:
        await runner.cleanup()
        result.update(finished_s=time.time(),router_records=[record_receipt(r) for r in router.records])
        save()
    return result
