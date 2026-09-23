import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from pdblend.profile.sampling_epochs import SamplingEpochs
from pdblend.profile.wave import atomic_json, ProfileWave


def members(tmp_path, monkeypatch, *, parallel=True):
    shared = tmp_path/'wave'; shared.mkdir()
    atomic_json(shared/'cohort.json', dict(cohort_id='cpu-protocol-test', members=['a','b']))
    async def qualify(wave, profiler):
        wave.parallel = parallel and len(wave.members)>1
        path = profiler.out_dir/'samples/external-interference.json'
        atomic_json(path, dict(complete=True, cross_job=True, passed=wave.parallel,
            fallback=None if wave.parallel else 'serial_cohort', members=wave.members))
        profiler.raw['external_interference'] = dict(complete=True, cross_job=True,
            passed=wave.parallel, samples_file='samples/external-interference.json',
            samples_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(ProfileWave, 'qualify_external', qualify)
    result = []
    for name in ('a','b'):
        out = tmp_path/name; out.mkdir()
        profiler = SimpleNamespace(raw=dict(environment=dict(gpu_uuids=['GPU-'+name])),
            out_dir=out, _checkpoint=lambda:None)
        result.append(SamplingEpochs(shared, name, profiler, timeout_s=2, poll_s=.001))
    return result


def test_retirement_waits_for_window_and_physical_cleanup_then_requalifies(tmp_path, monkeypatch):
    a,b = members(tmp_path, monkeypatch)
    async def check():
        await asyncio.gather(a.ready(), b.ready())
        old = b.qualification_guard()
        parent_before = json.dumps(b.profiler.raw, sort_keys=True)
        retire = asyncio.create_task(a.retire())
        await asyncio.sleep(.01)
        assert not retire.done()
        assert b.qualification_guard() == old  # current complete window remains valid
        next_window = asyncio.create_task(b.window_boundary({}, 1, 'decode'))
        await retire
        assert not next_window.done()  # cleanup cannot overlap the new probe
        a.released()
        new = await next_window
        assert new['epoch'] == 1 and new['layout'] == {'b':['GPU-b']}
        assert new['qualification_path'] != old['qualification_path']
        assert json.dumps(b.profiler.raw, sort_keys=True) == parent_before
        await b.retire(); b.released()
        assert json.loads((a.root/'epoch-state.json').read_text())['phase'] == 'complete'
    asyncio.run(check())


def test_failed_parallel_probe_serializes_until_member_is_cleaned(tmp_path, monkeypatch):
    a,b = members(tmp_path, monkeypatch, parallel=False)
    async def check():
        task_a = asyncio.create_task(a.ready()); task_b = asyncio.create_task(b.ready())
        await task_a
        assert not task_b.done()
        await a.retire(); a.released()
        await task_b
        assert b.qualification_guard()['measured_mode'] == 'serial_cohort'
    asyncio.run(check())


def test_mutated_qualification_cannot_be_used_for_new_window(tmp_path, monkeypatch):
    a,b = members(tmp_path, monkeypatch)
    async def check():
        await asyncio.gather(a.ready(), b.ready())
        from pathlib import Path
        Path(a.binding['qualification_path']).write_text('{}')
        with pytest.raises(RuntimeError, match='bytes changed'):
            a.qualification_guard()
    asyncio.run(check())


def test_failed_epoch_immediately_interrupts_pending_wave_barrier(tmp_path, monkeypatch):
    a,b = members(tmp_path, monkeypatch)
    path = a.root/'qualification-0000'; path.mkdir()
    atomic_json(path/'wave.json', dict(cohort_id='test', coordinator=True, members=['a','b']))
    async def check():
        wave = ProfileWave(path, 'b', timeout_s=10)
        waiting = asyncio.create_task(wave.wait('isolated', ['a']))
        await asyncio.sleep(.01)
        a.fail(RuntimeError('native rank crashed'))
        with pytest.raises(RuntimeError, match='native rank crashed'):
            await asyncio.wait_for(waiting, .5)
    asyncio.run(check())
