"""A's original 5xTP1 prefill -> TP2 decode routes, with separate TP oracles."""
import uuid
from checks import Checks as OriginalChecks,body,difference,require,check_ranks

class Checks(OriginalChecks):
    async def run(self):
        sources=self.instances[:-1];target=self.instances[-1]
        self.state.update(route_scope='five TP1 prefills to one TP2 decoder; GPU5 idle is still measured',reference_scope='TP1 and TP2 local references remain distinct; imported PD compared exactly with decoder TP2 local golden',checks={},ordinary={})
        for i in self.instances:await self.idle(i);await self.control(i,role='mixed',mode='continuous',admit_prefill=True,admit_decode=True)
        refs=self.state['ordinary']
        for length in (128,7168):
            tp1=[await self.generate(i,length,'ordinary-tp1-'+str(length)) for i in sources]
            tp2=[await self.generate(target,length,'ordinary-tp2-local-repeat-'+str(j)) for j in range(2)]
            refs[str(length)]=dict(tp1_token_ids=tp1,tp2_token_ids=tp2,tp1_replica_first_differences=[difference(tp1[0],x) for x in tp1],tp2_repeat_first_difference=difference(*tp2),cross_tp_first_difference_diagnostic_only=difference(tp1[0],tp2[0]));self.save()
            require(not any(refs[str(length)]['tp1_replica_first_differences']),'same-TP1 replicas differ')
            require(refs[str(length)]['tp2_repeat_first_difference'] is None,'same TP2 local golden repeats differ')
        self.state['checks']['ordinary_cross_replica_exact']=True;self.state['checks']['ordinary_reference_scope']='five TP1 replicas plus two repeats on the same TP2 decoder; no cross-TP equivalence claim';self.save()
        await self.control(target,role='decode')
        for source in sources:await self.control(source,role='prefill')
        rows=self.state['pd']=[]
        for source in sources:
            await self.http(source,'/prepare-peers',dict(peers=[target['id']]))
            for length in (128,7168):
                nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{source['id']}:{target['id']}";did=f"pdb:{nonce}:d:{source['id']}:{target['id']}"
                self.own(target,did)
                await self.generate(source,length,'cross-tp-producer',pid,dict(body(length),max_tokens=1))
                result=await self.generate(target,length,'cross-tp-consumer',did)
                row=dict(source=source['id'],source_tp=1,target=target['id'],target_tp=2,prompt_length=length,output_length=64,token_ids=result,first_difference_to_local_tp2=difference(refs[str(length)]['tp2_token_ids'][0],result),first_difference_to_local_tp1=difference(refs[str(length)]['tp1_token_ids'][sources.index(source)],result));rows.append(row);self.save()
                require(row['first_difference_to_local_tp2'] is None,'imported cross-TP KV changed decoder-local TP2 exact output')
                await self.idle(source);await self.idle(target)
            nonce=uuid.uuid4().hex;pid=f"pdb:{nonce}:p:{source['id']}:{target['id']}";did=f"pdb:{nonce}:d:{source['id']}:{target['id']}";self.own(target,did)
            await self.generate(source,128,'unconsumed-cross-tp-producer',pid,dict(body(128),max_tokens=1))
            cancelled=await self.http(target,'/cancel',dict(request_id=did));check_ranks(cancelled.get('transfers'),2);await self.idle(target);self.owned.discard((target['id'],did));self.state.setdefault('cancelled_unconsumed_kv',[]).append(dict(source=source['id'],target=target['id'],result=cancelled));self.save()
        self.state['checks'].update(pd_exact_all_declared_pairs=True,cancel_all_tp_ranks=True);self.state['passed']=True;self.save()
