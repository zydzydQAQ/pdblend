"""Keep qualified power peers resident through unequal/serial measurements."""
import asyncio
import json

import pytest

from pdblend.profile.wave import ProfileWave


def waves(tmp_path, *, parallel=True, keep=True, timeout=2):
    (tmp_path/'wave.json').write_text(json.dumps(dict(members=['a', 'b'], cohort_id='power-pair',
        coordinator=True, keep_peers_resident_until_all_done=keep)))
    values = [ProfileWave(tmp_path, name, timeout_s=timeout) for name in ('a', 'b')]
    for value in values: value.parallel = parallel
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize('parallel', [True, False])
async def test_finished_member_holds_resident_scope_until_all_samples_end(tmp_path, parallel):
    a, b = waves(tmp_path, parallel=parallel)
    first_done, second_started, finish_second = asyncio.Event(), asyncio.Event(), asyncio.Event()
    exited = []
    async def first():
        async with a.measurement(): first_done.set()
        exited.append('a')
    async def second():
        async with b.measurement():
            second_started.set(); await finish_second.wait()
        exited.append('b')
    tasks = [asyncio.create_task(first()), asyncio.create_task(second())]
    await asyncio.wait_for(first_done.wait(), 1)
    await asyncio.wait_for(second_started.wait(), 1)
    assert (tmp_path/'a.done.json').is_file()  # Serial successor was admitted.
    assert not exited                       # Caller cannot stop/reset/release.
    finish_second.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert sorted(exited) == ['a', 'b']


@pytest.mark.asyncio
async def test_peer_failure_wins_even_when_done_file_exists(tmp_path):
    a, b = waves(tmp_path)
    b.write('done', {'time': 1}); b.write('error', {'error': 'failed during checkpoint'})
    with pytest.raises(RuntimeError, match='peer failed'):
        async with a.measurement(): pass
    assert (tmp_path/'a.error.json').is_file()


@pytest.mark.asyncio
async def test_missing_peer_times_out_and_records_failure(tmp_path):
    a, _ = waves(tmp_path, timeout=.01)
    with pytest.raises(TimeoutError, match='waiting for done'):
        async with a.measurement(): pass
    assert (tmp_path/'a.error.json').is_file()


@pytest.mark.asyncio
async def test_other_waves_preserve_existing_nonbarrier_completion(tmp_path):
    a, _ = waves(tmp_path, keep=False)
    async with a.measurement(): pass
    assert (tmp_path/'a.done.json').is_file()
    assert not (tmp_path/'b.done.json').exists()
