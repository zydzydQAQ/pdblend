import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from pdblend.profile.sampling_epochs import SamplingEpochs
from pdblend.profile.wave import ProfileWave, atomic_json


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


@pytest.mark.parametrize('delayed_burst_s, expected_pass', [(3.456, True), (7.5, False)])
def test_six_heterogeneous_probes_check_actual_overlap_after_barrier(
        tmp_path, monkeypatch, delayed_burst_s, expected_pass):
    """Exercise six real callback/barrier loops with different burst tails.

    Time is scaled only to keep this CPU regression short. Start/end receipts
    come from the callbacks' observed wall clock, not a fabricated passed flag.
    A longer qualification window must still reject a sufficiently late peer.
    """
    names = ['pdblend-32b', 'pdblend-14b', 'pdblend-7b',
             'dynamo-32b', 'dynamo-14b', 'dynamo-7b']
    root = tmp_path/'wave'; root.mkdir()
    atomic_json(root/'wave.json', dict(cohort_id='six-heterogeneous-timing', coordinator=True,
        members=names, synchronize_parallel_windows=True, qualification_measure_s=8.))
    real_sleep, scale = asyncio.sleep, .02
    async def scaled_sleep(seconds):
        await real_sleep(seconds*scale)
    monkeypatch.setattr(asyncio, 'sleep', scaled_sleep)

    async def scenario():
        start = time.monotonic()
        waves, profiles = [], []
        for name in names:
            wave = ProfileWave(root, name, timeout_s=5.)
            out = tmp_path/name; out.mkdir()
            profiler = SimpleNamespace(out_dir=out, raw=dict(environment=dict(gpu_uuids=[name])),
                profile_key=SimpleNamespace(as_dict=lambda:dict(system='cpu-test')),
                _checkpoint=lambda:None)
            async def probe(profile, phase, owner=wave):
                repeats = []
                for repeat in range(3):
                    if phase == 'parallel':
                        marker = f'parallel-window-{repeat}-ready'
                        owner.write(marker, dict(ready=True))
                        await owner.wait(marker)
                        # PDB already has continuous decode running. Dynamo
                        # 32B may complete a whole warmup burst after release.
                        tail = delayed_burst_s if owner.member == 'dynamo-32b' else (
                            .7 if owner.member.startswith('dynamo-') else 0.)
                        await asyncio.sleep(tail)
                        began = (time.monotonic()-start)/scale
                        await asyncio.sleep(owner.qualification_measure_s)
                        ended = (time.monotonic()-start)/scale
                    else:
                        began, ended = 0., owner.qualification_measure_s
                    repeats.append(dict(start_s=began, end_s=ended))
                return dict(instances=[dict(step_seconds=.02, power_w=100., repeats=repeats)],
                            gpu_uuids=profile.raw['environment']['gpu_uuids'])
            wave.probe = probe
            waves.append(wave); profiles.append(profiler)
        await asyncio.gather(*(wave.qualify(profile) for wave, profile in zip(waves, profiles)))
        for wave, profile in zip(waves, profiles):
            receipt = json.loads((profile.out_dir/'samples/external-interference.json').read_text())
            assert receipt['qualification_measure_s'] == 8.
            assert all(row['passed'] for row in receipt['comparisons'])
            assert receipt['common_measurement_windows']['minimum_s'] == 2.
            assert receipt['common_measurement_windows']['passed'] is expected_pass
            assert wave.parallel is expected_pass
            assert len(receipt['common_measurement_windows']['overlap_seconds']) == 3
    asyncio.run(scenario())
