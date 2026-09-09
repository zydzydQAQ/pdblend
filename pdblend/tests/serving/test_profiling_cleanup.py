import asyncio

import pytest

from ecopadg.serving.profiling import HardwareProfiler
from ecopadg.serving import interference,prefill_batch


def profiler(tmp_path):
    a=dict(id='p',tp=1,gpus=[0],port=1)
    b=dict(id='d',tp=1,gpus=[1],port=2)
    for i in (a,b):(tmp_path/(i['id']+'.control.events.jsonl')).write_text('')
    p=HardwareProfiler(None,dict(mixed=a,prefill=a,decode=b),tmp_path)
    return p,a,b


def state():
    return dict(running=0,waiting=0,free_kv_tokens=100000,
        transfer_allocations={},transfer_bytes_per_token=16,free_transfer_bytes=1000000)


@pytest.mark.parametrize('kind',['mixed','interference','prefill_batch'])
def test_cancel_after_first_gate_commits_still_restores_admission(tmp_path,kind):
    async def run():
        p,a,b=profiler(tmp_path);gate=asyncio.Event();actions=[]
        async def call(instance,path,body=None,rid=None):return state()
        async def control(instance,**changes):
            actions.append((instance['id'],changes))
            if changes.get('admit_decode') is False or changes.get('admit_prefill') is False:
                gate.set();await asyncio.Event().wait()
            return {}
        p.call=call;p.control=control
        work=(p.run(1,1,32,'mixed') if kind=='mixed' else
            interference.run_case(p,a,tmp_path/'p.control.events.jsonl',1,1,8,32) if kind=='interference'
            else prefill_batch.case(p,1,1))
        task=asyncio.create_task(work);await asyncio.wait_for(gate.wait(),1);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        restored={i for i,c in actions if c.get('admit_prefill') is True and c.get('admit_decode') is True}
        assert restored==({'p','d'} if kind=='prefill_batch' else {'p'})
    asyncio.run(run())


def test_cancel_between_producer_and_consumer_cancels_both_phase_ids(tmp_path):
    async def run():
        p,a,b=profiler(tmp_path);cancelled=[];completions=[]
        async def call(instance,path,body=None,rid=None):
            if path=='/v1/completions':
                completions.append(rid);raise asyncio.CancelledError()
            if path=='/cancel':cancelled.append((instance['id'],body['request_id']))
            return state()
        async def control(instance,**changes):return {}
        p.call=call;p.control=control
        with pytest.raises(asyncio.CancelledError):await p.run(1,1,32,'pd')
        assert len(completions)==1 and ':p:p:d' in completions[0]
        nonce=completions[0].split(':')[1]
        assert set(cancelled)=={('p',f'pdb:{nonce}:p:p:d'),('d',f'pdb:{nonce}:d:p:d')}
    asyncio.run(run())


def test_failed_cancel_does_not_skip_other_ids_local_tasks_or_either_restore(tmp_path):
    async def run():
        p,a,b=profiler(tmp_path);actions=[];ended=asyncio.Event()
        async def request():
            try:await asyncio.Event().wait()
            finally:ended.set()
        task=asyncio.create_task(request());await asyncio.sleep(0);p.tasks=[task]
        async def call(instance,path,body=None,rid=None):
            actions.append(('cancel',instance['id']))
            if instance is a:raise RuntimeError('first cancel failed')
            return {}
        async def control(instance,**changes):actions.append(('restore',instance['id']));return {}
        p.call=call;p.control=control
        with pytest.raises(RuntimeError,match='cleanup incomplete'):
            await p.cleanup((a,b),cancel_ids=[(a,'one'),(b,'two')])
        assert ended.is_set() and task.done()
        assert set(actions)=={('cancel','p'),('cancel','d'),('restore','p'),('restore','d')}
    asyncio.run(run())
