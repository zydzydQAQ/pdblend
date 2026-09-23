import asyncio
import json
from types import SimpleNamespace

from pdblend.profile.sampling_epochs import SamplingEpochs
from pdblend.profile.wave import atomic_json


def test_six_system_groups_concurrent_retirement_keeps_window_guards(tmp_path):
    names=['pdblend-32b','pdblend-14b','pdblend-7b','dynamo-32b','dynamo-14b','dynamo-7b']
    root=tmp_path/'cohort';root.mkdir();atomic_json(root/'cohort.json',dict(cohort_id='cpu-six-member-test',members=names))
    peers=[]
    for name in names:
        out=tmp_path/name;out.mkdir()
        uuids=[name+':0',name+':1'] if '32b' in name else [name+':0']
        p=SimpleNamespace(out_dir=out,raw=dict(environment=dict(gpu_uuids=uuids)),
            profile_key=SimpleNamespace(as_dict=lambda:dict(system='cpu-protocol-test')),_checkpoint=lambda:None)
        async def probe(profiler,phase,wave):
            # CPU-only protocol receipts exercise the real cross-system wave
            # barrier. They are confined to pytest tmp_path, never GPU results.
            if phase=='parallel':
                wave.write('parallel-window-0-ready',dict(ready=True))
                await wave.wait('parallel-window-0-ready')
            return dict(instances=[dict(step_seconds=.02,power_w=100,repeats=[dict(start_s=1.,end_s=6.)])],
                gpu_uuids=profiler.raw['environment']['gpu_uuids'])
        peers.append(SamplingEpochs(root,name,p,timeout_s=5,poll_s=.001,probe_callback=probe))
    async def check():
        await asyncio.gather(*(p.ready() for p in peers))
        before={p.member:p.qualification_guard() for p in peers}
        retire=[asyncio.create_task(p.retire()) for p in peers[:2]]
        await asyncio.sleep(.02)
        assert all(not task.done() for task in retire)
        for p in peers[2:]:assert p.qualification_guard()==before[p.member]
        next_windows=[asyncio.create_task(p.window_boundary()) for p in peers[2:]]
        await asyncio.gather(*retire)
        peers[0].released()
        assert json.loads((root/'epoch-state.json').read_text())['phase']=='cleanup'
        assert not any(t.done() for t in next_windows)
        peers[1].released();fresh=await asyncio.gather(*next_windows)
        assert all(value['epoch']==1 and set(value['layout'])==set(names[2:]) for value in fresh)
        assert all(value['qualification_sha256']!=before[p.member]['qualification_sha256'] for p,value in zip(peers[2:],fresh))
        await asyncio.gather(*(p.retire() for p in peers[2:]))
        for p in peers[2:]:p.released()
        state=json.loads((root/'epoch-state.json').read_text())
        assert state['phase']=='complete' and not state['active'] and not state['errors']
    asyncio.run(check())
