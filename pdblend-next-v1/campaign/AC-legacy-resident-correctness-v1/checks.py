"""Small real legacy TP checks; caller owns deployment, sampling and bounded cleanup."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

FIELDS=('generation','role','mode','admit_prefill','admit_decode')

def require(ok,why):
    if not ok:raise RuntimeError(why)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def difference(reference,observed):
    for i in range(max(len(reference),len(observed))):
        a=reference[i] if i<len(reference) else None;b=observed[i] if i<len(observed) else None
        if a!=b:return dict(position_one_based=i+1,reference=a,observed=b)
    return None

def check_ack(raw,instance,generation=None):
    require(raw.get('id')==instance['id'] and not raw.get('error') and not raw.get('runtime_error'),'wrong/unhealthy owner')
    require(type(raw.get('generation')) is int and raw['generation']==raw.get('acknowledged_generation'),'legacy owner ACK missing')
    if generation is not None:require(raw['generation']==generation,'wrong owner generation')
    require(raw.get('transport_healthy') is True,'unhealthy TP transport')
    for field in ('timestamp','transfer_observed_s'):
        require(type(raw.get(field)) in (int,float) and 0<=time.time()-raw[field]<=1.,'stale actual '+field)

def is_idle(raw):
    fields=('active','running','waiting','kv_allocations','transfer_allocations','transfer_buffered_tensors','transfer_inflight_receives')
    return all(k in raw for k in fields) and not any(raw[k] for k in fields)

def check_ranks(ranks,tp):
    require(isinstance(ranks,list) and len(ranks)==tp,'all native TP ranks are required')
    for r in ranks:
        require(all(type(r.get(k)) is int and r[k]==0 for k in ('buffered_tensors','inflight_receives','buffered_gpu_bytes'))
            and isinstance(r.get('allocations'),dict) and not r['allocations'] and r.get('listener_alive') is True,
            'legacy native rank residue or missing observation')

def tokens(reply,prompt_length,output_length=64):
    ids=reply.get('token_ids');require(isinstance(ids,list) and len(ids)==output_length and all(type(i)is int for i in ids),'actual token IDs missing')
    require(reply.get('usage',{}).get('prompt_tokens')==prompt_length and reply['usage'].get('completion_tokens')==output_length,'prescribed work changed')
    return ids

def body(length):
    return dict(prompt=([9707,1879,13]*(length//3+1))[:length],max_tokens=64,temperature=0,top_p=1,ignore_eos=True,seed=0,stream=False)

class Checks:
    def __init__(self,session,binding,out):
        self.session=session;self.binding=binding;self.instances=binding['instances'];self.out=Path(out)
        require(not self.out.exists(),'new correctness output required');self.out.mkdir(parents=True)
        self.owned=set();self.tasks=[];self.log=(self.out/'http.jsonl').open('x',buffering=1);self.ownlog=(self.out/'owned.jsonl').open('x',buffering=1)
        self.state=dict(complete=False,passed=False,checks={},requests=[],started_s=time.time(),legacy_sender_counter_observation='not exposed; synchronous PUT and strict ACK source is bound',numerical_gate='exact output IDs; no logits or bit-exact KV claim',temporal_original_v3_failure_preserved=True)
        self.save()
    def save(self):write(self.out/'checks.json',self.state)
    async def http(self,i,path,payload=None,rid=None,timeout=45):
        import aiohttp
        record=dict(instance=i['id'],route=path,body=payload,request_id=rid,started_s=time.time())
        try:
            async with self.session.request('GET' if payload is None else 'POST',i['url']+path,json=payload,
                headers={'X-Request-Id':rid} if rid else None,timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                text=await response.text();record.update(status=response.status,response=json.loads(text) if text.startswith(('{','[')) else text)
                require(response.status==200,f"{i['id']}{path}: {text[:500]}");return record['response']
        except BaseException as exc:record['error']=repr(exc);raise
        finally:record['finished_s']=time.time();self.log.write(json.dumps(record,allow_nan=False)+'\n')
    async def runtime(self,i):
        r=await self.http(i,'/runtime');check_ack(r,i);return r
    async def idle(self,i,timeout=15):
        end=time.monotonic()+timeout
        while True:
            r=await self.runtime(i)
            if is_idle(r):return r
            require(time.monotonic()<end,'owned request/transfer not idle');await asyncio.sleep(.02)
    async def control(self,i,**changes):
        before=await self.runtime(i);payload={k:before[k] for k in FIELDS};payload.update(changes,generation=before['generation']+1)
        require('scheduler_budget' not in payload,'legacy must not receive v3 budget controls')
        result=await self.http(i,'/control',payload);after=await self.runtime(i);check_ack(after,i,payload['generation'])
        require(result==payload and all(after.get(k)==v for k,v in payload.items()),'legacy control state differs')
        return dict(before=before,command=payload,after=after)
    def own(self,i,rid):
        pair=(i['id'],rid)
        if pair not in self.owned:self.ownlog.write(json.dumps(dict(instance_id=i['id'],request_id=rid,created_s=time.time()))+'\n')
        self.owned.add(pair)
    async def generate(self,i,length,label,rid=None,payload=None):
        rid=rid or 'legacycheck-'+uuid.uuid4().hex;payload=payload or body(length);self.own(i,rid)
        row=dict(instance_id=i['id'],request_id=rid,label=label,body=payload,dispatch_s=time.time());self.state['requests'].append(row);self.save()
        try:
            result=await self.http(i,'/v1/completions',payload,rid,timeout=120);row['response']=result
            values=tokens(result,length,payload['max_tokens']);self.owned.discard((i['id'],rid));return values
        except BaseException as exc:row['error']=repr(exc);raise
        finally:row['finished_s']=time.time();self.save()
    def task(self,coro):
        t=asyncio.create_task(coro);self.tasks.append(t);return t
    async def run(self):
        a=self.instances[0]
        for i in self.instances:await self.idle(i);await self.control(i,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        references={};self.state['ordinary']=references
        for length in (128,7168):
            replies=[]
            for i in self.instances:replies.append(await self.generate(i,length,'ordinary-'+str(length)))
            references[str(length)]={'token_ids':replies,'first_differences':[difference(replies[0],x) for x in replies]};self.save()
            require(not any(references[str(length)]['first_differences']),'ordinary cross-replica output differs')
        self.state['checks']['ordinary_cross_replica_exact']=True;self.save()
        await self.control(a,role='prefill')
        for i in self.instances[1:]:await self.control(i,role='decode')
        await self.http(a,'/prepare-peers',dict(peers=[i['id'] for i in self.instances[1:]]))
        pd=[];self.state['pd']=pd
        for target in self.instances[1:]:
            for length in (128,7168):
                nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{a['id']}:{target['id']}";did=f"pdb:{nonce}:d:{a['id']}:{target['id']}"
                self.own(target,did)
                await self.generate(a,length,'pd-producer',pid,dict(body(length),max_tokens=1))
                result=await self.generate(target,length,'pd-consumer',did)
                row=dict(source=a['id'],target=target['id'],prompt_length=length,token_ids=result,first_difference=difference(references[str(length)]['token_ids'][0],result));pd.append(row);self.save()
                require(row['first_difference'] is None,'real PD output differs')
                await self.idle(a);await self.idle(target)
            nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{a['id']}:{target['id']}";did=f"pdb:{nonce}:d:{a['id']}:{target['id']}";self.own(target,did)
            await self.generate(a,128,'unconsumed-producer',pid,dict(body(128),max_tokens=1))
            cancellation=await self.http(target,'/cancel',dict(request_id=did));check_ranks(cancellation.get('transfers'),target['tp']);await self.idle(target);self.owned.discard((target['id'],did))
            self.state.setdefault('cancelled_unconsumed_kv',[]).append(dict(target=target['id'],request_id=did,result=cancellation));self.save()
        self.state['checks']['pd_exact_all_declared_pairs']=True;self.state['checks']['cancel_all_tp_ranks']=True;self.save()
        for i in self.instances:await self.idle(i);await self.control(i,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        b=self.instances[1];refs=[await self.generate(b,n,'temporal-single-reference') for n in (96,192)]
        phase=self.state['temporal']=dict(reference_token_ids=refs,original_sequence='first allocated >=1; close prefill; second queued .1s; reopen',complete=False);self.save()
        await self.control(b,mode='temporal',admit_prefill=True)
        rid1='legacycheck-'+uuid.uuid4().hex;rid2='legacycheck-'+uuid.uuid4().hex
        t1=self.task(self.generate(b,96,'temporal-first',rid1));end=time.monotonic()+30
        while True:
            r=await self.runtime(b)
            if rid1 in r['kv_allocations'] and r['running']==1:phase['first_allocated']=r;break
            require(not t1.done() and time.monotonic()<end,'original allocation gate missed');await asyncio.sleep(.005)
        phase['close_prefill']=await self.control(b,admit_prefill=False)
        t2=self.task(self.generate(b,192,'temporal-second',rid2));await asyncio.sleep(.1);held=await self.runtime(b);phase['held']=held
        require(rid2 not in held['kv_allocations'] and held['waiting']>=1,'temporal prefill leaked into closed window')
        phase['open_prefill']=await self.control(b,admit_prefill=True)
        outputs=await asyncio.gather(t1,t2);phase.update(token_ids=outputs,first_differences=[difference(x,y) for x,y in zip(refs,outputs)],complete=True);self.save()
        require(not any(phase['first_differences']),'legacy temporal exact-output gate failed; retain outputs/events')
        self.state['checks']['temporal_exact']=True;self.state['passed']=True;self.save()
    async def cleanup(self):
        errors=[]
        for t in self.tasks:
            if not t.done():t.cancel()
        if self.tasks:await asyncio.gather(*self.tasks,return_exceptions=True)
        by_id={i['id']:i for i in self.instances}
        for iid,rid in list(self.owned):
            try:await self.http(by_id[iid],'/cancel',dict(request_id=rid),timeout=5)
            except BaseException as exc:errors.append('own cancel '+repr(exc))
        restorations={}
        async def restore(i):
            result={'errors':[]}
            try:
                # Reopen a failed temporal hold before waiting for residual work.
                await self.control(i,admit_prefill=True,admit_decode=True)
                before=await self.idle(i,timeout=15);proof=await self.http(i,'/drain',dict(expected_generation=before['generation']),timeout=35)
                result['proof']=proof;require(proof.get('drained') is True and proof.get('accepting') is False and proof.get('generation')==before['generation']+1 and proof.get('drain_proof_type')=='synchronous_put_owner_barrier','legacy barrier missing');check_ranks(proof.get('transfers'),i['tp'])
            except BaseException as exc:result['errors'].append('native proof '+repr(exc))
            finally:
                try:
                    result['resume']=await self.control(i,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
                    after=await self.idle(i);require(after.get('accepting') is True,'legacy resume still paused');result['after']=after
                except BaseException as exc:result['errors'].append('resume '+repr(exc))
            result['complete']=not result['errors'];return result
        values=await asyncio.gather(*(restore(i) for i in self.instances),return_exceptions=True)
        for i,v in zip(self.instances,values):restorations[i['id']]=dict(complete=False,error=repr(v)) if isinstance(v,BaseException) else v
        self.state['cleanup']={'errors':errors,'instances':restorations,'complete':not errors and all(r['complete'] for r in restorations.values())};self.save();return self.state['cleanup']['complete']
    def close(self):self.log.close();self.ownlog.close()
