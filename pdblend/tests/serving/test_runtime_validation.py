import asyncio
import json

import pytest
from aiohttp import web

from ecopadg.serving.engine import EngineService
from ecopadg.serving.runtime_validation import verify_role_rollback


def original_state():
    return dict(role='decode', mode='continuous', admit_prefill=False,
                admit_decode=True, generation=7, acknowledged_generation=7,
                accepting=True, error=None, runtime_error=None)


def test_real_control_failure_consumes_versions_and_rejects_delayed_command(tmp_path):
    """Exercise the actual HTTP control transaction without loading a GPU model."""
    async def run():
        service = EngineService(dict(id='test', runtime_dir=str(tmp_path)))
        before = original_state()
        service.state = {key: before[key] for key in
                         ('role', 'mode', 'admit_prefill', 'admit_decode', 'generation')}
        service.snapshot = dict(before)
        attempts = []

        def commit(payload):
            attempts.append(dict(payload))
            if payload['role'] != service.state['role']:
                raise RuntimeError('transport has not drained')
            service.state = dict(payload)
            service.snapshot = dict(payload, acknowledged_generation=payload['generation'])

        class Request:
            async def json(self):
                return dict(proposal)

        service.commit_state = commit
        proposal = dict(service.state, role='mixed', generation=8)
        try:
            with pytest.raises(web.HTTPConflict, match='Conflict') as failure:
                await service.control(Request())
            assert failure.value.text == 'control rolled back: transport has not drained'
            after = json.loads((await service.status(None)).text)
            result = verify_role_rollback(before, after, proposal['generation'])
            assert result['expected_rollback_generation'] == 9
            assert [attempt['generation'] for attempt in attempts] == [8, 9]
            with pytest.raises(web.HTTPConflict) as replay:
                await service.control(Request())
            assert replay.value.text == 'expected next generation'
            replayed = json.loads((await service.status(None)).text)
            assert verify_role_rollback(before, replayed, 8) == result
            assert len(attempts) == 2  # A stale proposal never reaches the commit.
        finally:
            service.worker.shutdown(wait=True, cancel_futures=True)

    asyncio.run(run())


@pytest.mark.parametrize('mutation', [
    {'role': 'mixed'},
    {'mode': 'temporal'},
    {'admit_prefill': True},
    {'admit_decode': False},
    {'generation': 7, 'acknowledged_generation': 7},
    {'generation': 8, 'acknowledged_generation': 8},
    {'generation': 10, 'acknowledged_generation': 10},
    {'acknowledged_generation': 8},
    {'runtime_error': 'scheduler failed'},
    {'error': 'executor failed'},
    {'accepting': False},
])
def test_rollback_rejects_control_mutation_unconfirmed_version_or_unhealthy_engine(mutation):
    before = original_state()
    after = dict(before, generation=9, acknowledged_generation=9)
    after.update(mutation)
    with pytest.raises(RuntimeError):
        verify_role_rollback(before, after, 8)


def test_rollback_cannot_certify_an_out_of_order_proposal():
    before = original_state()
    with pytest.raises(RuntimeError, match='next generation'):
        verify_role_rollback(before, dict(before, generation=10, acknowledged_generation=10), 9)


@pytest.mark.parametrize('fault',['partial_write','cancel_during_restore'])
def test_validation_cleanup_still_saves_raw_and_resets_clocks_after_cleanup_fault(tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    from ecopadg.serving import runtime_validation as validation
    actions=[];restore_started=None
    class Hardware:pass
    class Clocks:
        def __init__(self,*args):pass
        async def set(self,*args,**kwargs):pass
        async def close(self):actions.append('clock restored')
    class Sampler:
        def __init__(self,*args,**kwargs):self.samples=[];self.frequency_samples=[];self.error=None
        def start(self):pass
        def stop(self):actions.append('sampler stopped')
    class Session:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    class Profiler:
        def __init__(self,*args):self.count=0;self.tasks=[]
        async def provenance(self):return []
        async def call(self,*args,**kwargs):
            return dict(original_state(),active=0,running=0,waiting=0,kv_allocations={},transfer_allocations={})
        async def cleanup(self,*args,**kwargs):actions.append('requests cancelled')
        async def control(self,*args,**kwargs):
            self.count+=1
            if self.count==1:raise ValueError('stop before GPU mechanism work')
            actions.append('control restore started');restore_started.set()
            if fault=='cancel_during_restore':await asyncio.Event().wait()
    monkeypatch.setattr(validation,'PynvmlBackend',Hardware)
    monkeypatch.setattr(validation,'ClockOwner',Clocks);monkeypatch.setattr(validation,'PowerSampler',Sampler)
    monkeypatch.setattr(validation.aiohttp,'ClientSession',Session);monkeypatch.setattr(validation,'HardwareProfiler',Profiler)
    if fault=='partial_write':
        def failed_write(*args):raise OSError('partial raw write failed')
        monkeypatch.setattr(validation,'write_json',failed_write)
    topology=tmp_path/'topology.json';topology.write_text(json.dumps(dict(
        prefill=dict(id='a',tp=1,gpus=[0]),decode=dict(id='b',tp=1,gpus=[1]))))
    args=SimpleNamespace(topology=topology,out=tmp_path/'out',runtime_dir=tmp_path)
    async def run():
        nonlocal restore_started
        restore_started=asyncio.Event()
        task=asyncio.create_task(validation.validate(args))
        if fault=='cancel_during_restore':
            await asyncio.wait_for(restore_started.wait(),1);task.cancel()
        with pytest.raises(asyncio.CancelledError if fault=='cancel_during_restore' else ValueError):await task
    asyncio.run(run())
    assert {'requests cancelled','control restore started','sampler stopped','clock restored'}<=set(actions)
    raw=json.loads((args.out/'raw.json').read_text())
    assert not raw['complete'] and not raw['passed'] and raw['errors']
