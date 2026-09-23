import asyncio
import time

import pytest

from pdblend_baselines.dynamollm.loaded_drain import LoadedDrain


class Transport:
    def __init__(self, *, finish_early=False):
        self.live={};self.release=asyncio.Event();self.finish_early=finish_early;self.cancelled=[]
    async def state(self,iid):
        return dict(timestamp=time.time(),free_kv_tokens=65536,kv_allocations=dict(self.live),
                    evidence_complete=True)
    async def stream(self,iid,payload):
        rid=payload['request_id'];self.live[rid]=[[1,2,3]]
        yield dict(token_ids=[1],finished=False)
        if not self.finish_early:await self.release.wait()
        yield dict(token_ids=[2]*(payload['max_tokens']-1),finished=True)
        self.live.pop(rid)
    async def cancel(self,iid,rid):
        self.cancelled.append(rid);self.live.pop(rid,None)


@pytest.mark.asyncio
async def test_loaded_probe_requires_actual_live_kv_then_complete_source_output():
    transport=Transport();rows=[]
    loader=LoadedDrain(transport,'source',lambda event,**r:rows.append(dict(event=event,**r)),
                       input_tokens=128,output_tokens=16,batch=2)
    observation=await loader.start()
    assert len(observation['state']['kv_allocations'])==2
    transport.release.set()
    result=await loader.finish();await loader.close()
    assert result['measured'] and all(r['ok'] for r in result['outcomes'])
    assert not transport.cancelled and not transport.live
    assert result['workload_envelope']==dict(max_input_tokens=128,max_output_tokens=16,max_batch=2)


@pytest.mark.asyncio
async def test_completed_before_snapshot_is_not_loaded_drain_evidence():
    transport=Transport(finish_early=True)
    loader=LoadedDrain(transport,'source',lambda *a,**k:None,input_tokens=128,output_tokens=16,batch=1)
    with pytest.raises(RuntimeError,match='ended before'):
        await loader.start()
    await loader.close()


@pytest.mark.asyncio
async def test_failed_transition_cancels_and_awaits_owned_live_source_requests():
    transport=Transport()
    loader=LoadedDrain(transport,'source',lambda *a,**k:None,input_tokens=128,output_tokens=16,batch=2)
    await loader.start();await loader.close()
    assert len(transport.cancelled)==2 and not transport.live
    assert all(task.done() for task in loader.tasks)
