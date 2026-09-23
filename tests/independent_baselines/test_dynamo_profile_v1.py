import copy
import json
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import profile_v1
from pdblend_baselines.dynamollm.deployment import save, sha
from pdblend_baselines.dynamollm.label_v1 import actual_label
from pdblend_baselines.dynamollm.profiles import PaperProfiles


def window(tp=2, batch=1, multiplier=1.):
    native = {'ranks': [dict(rank=rank, samples=[dict(role='decode', batch=batch, system='dynamollm',
        measurement_scope='runner', tp=tp, pp=1, gpu_elapsed_ms=10.) for _ in range(8)]) for rank in range(tp)]}
    return dict(settle_s=2., started_s=10., finished_s=15., native=native, gpu_ids=list(range(tp)),
        power=[dict(timestamp=t, gpu=gpu, gpu_uuid=f'GPU-{gpu}', frequency_mhz=900,
                    power_w=100*multiplier, source='nvml:field:186:scope:0:mW')
               for gpu in range(tp) for t in (11., 12.)],
        requests=[dict(submitted_s=11., first_token_s=11.+.1*multiplier,
                       finished_s=11.+.25*multiplier, tokens=16, ok=True)])


def test_reduce_native_window_requires_all_rank_steps_and_power_geometry():
    result = profile_v1.reduce_window(window(), tp=2, batch=1, frequency=900)
    assert result['power_w'] == 200
    assert result['iteration_s'] == pytest.approx(.01)
    assert result['prefill_s'] == pytest.approx(.09)
    value = window()
    value['native']['ranks'][1]['samples'].pop()
    with pytest.raises(ValueError, match='8 actual decode'):
        profile_v1.reduce_window(value, tp=2, batch=1, frequency=900)
    value = window()
    value['power'] = value['power'][:2]
    with pytest.raises(ValueError, match='2 group power'):
        profile_v1.reduce_window(value, tp=2, batch=1, frequency=900)


def test_fit_holds_out_entire_window_and_does_not_admit_out_of_error_budget():
    point = dict(batch=1, frequency_mhz=900)
    fitted = profile_v1.fit_cell([window(), window(), window()], window(), tp=2, point=point)
    assert fitted['holdout_passed']
    failed = profile_v1.fit_cell([window(), window(), window()], window(multiplier=2.), tp=2, point=point)
    assert not failed['holdout_passed']
    assert failed['holdout_errors']['power'] == .5
    with pytest.raises(ValueError, match='three independent'):
        profile_v1.fit_cell([window()], window(), tp=2, point=point)


def test_resume_checks_raw_checksums_and_exact_identity(tmp_path):
    identity = dict(system='dynamollm', model_id='Qwen2.5-32B-Instruct', tp=2,
                    gpu_uuids={'0':'GPU-a', '1':'GPU-b'})
    point = dict(batch=1, frequency_mhz=900)
    raw = tmp_path/'raw'/'sample.json'
    save(raw, window())
    save(tmp_path/'capability.json', {'generation': 0})
    key = profile_v1.key_for(identity, point)
    save(tmp_path/'cells'/(key+'.json'), dict(identity=identity, point=point,
         artifacts={'raw/sample.json': sha(raw)}, fit={'holdout_passed': True},
         capability_sha256=sha(tmp_path/'capability.json')))
    assert len(profile_v1.completed_cells(tmp_path, identity)) == 1
    with pytest.raises(ValueError, match='identity changed'):
        profile_v1.completed_cells(tmp_path, dict(identity, tp=4))
    raw.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        profile_v1.completed_cells(tmp_path, identity)


@pytest.mark.asyncio
@pytest.mark.parametrize('external,epoch', [(False,False), (True,False), (False,True)])
async def test_collector_executes_three_repeats_and_holdout_with_one_resident_engine(tmp_path, monkeypatch, external, epoch):
    calls = []
    monkeypatch.setattr(profile_v1, 'model_identity', lambda path: dict(model='Qwen2.5-32B-Instruct'))
    class Telemetry:
        def __init__(self, gpus, journal): self.uuids={0:'GPU-a',1:'GPU-b'}
        def start(self): calls.append('meter-start')
        def clock(self, *args): calls.append('clock')
        async def close(self): calls.append('meter-close')
    class Transport:
        def __init__(self, specs, *args): self.instances={s['id']:s for s in specs}
        async def start(self): pass
        async def close(self): calls.append('http-close')
        async def state(self, iid):
            import time
            return dict(total_kv_tokens=8192, free_kv_tokens=8192, active=0, running=0, waiting=0,
                generation=2, acknowledged_generation=2, timestamp=time.time(),
                evidence_complete=True, transport_healthy=True, accepting=False,
                kv_allocations={}, transfer_allocations={})
        async def json(self, iid, path, payload=None, **kwargs):
            if path == '/baseline/measurement/stop':
                calls.append('measurement-stop');return {}
            if path == '/baseline/dynamollm/drain':
                calls.append('native-drain')
                return dict(ranks=[dict(rank=r,ok=True,generation=2,drained=True,
                    cuda_synchronized=True,active_weight_sessions=0) for r in range(2)])
            assert path == '/baseline/capability'
            return dict(model_id='Qwen2.5-32B-Instruct',engine_revision='vllm-0.10.1.1',tp=2,pp=1,
                        gpu_uuids=['GPU-a','GPU-b'], state={'generation':2})
    class Lifecycle:
        def __init__(self, *args): pass
        async def start(self, spec): calls.append('model-load')
        async def close(self): calls.append('model-close')
    async def measure(*args, **kwargs):
        calls.append(('window', kwargs['repeat']))
        kwargs['ownership']['started'] = True
        return window()
    monkeypatch.setattr(profile_v1, 'GroupTelemetry', Telemetry)
    monkeypatch.setattr(profile_v1, 'V1Transport', Transport)
    monkeypatch.setattr(profile_v1, 'SubprocessLifecycle', Lifecycle)
    monkeypatch.setattr(profile_v1, 'measure_window', measure)
    async def golden(*args, **kwargs): return {}
    monkeypatch.setattr(profile_v1, 'collect_golden', golden)
    args = SimpleNamespace(model=tmp_path/'model',tp=2,gpus=[0,1],out=tmp_path/'profile',resume=False,
        base_port=17000,batches=[1],freqs=[900],inputs=[128],outputs=[16],settle=2,measure=5,label_corpus_root=None,
        existing_url='http://resident:17000' if external else None)
    if epoch:
        from pdblend_baselines.dynamollm import profile_epochs
        class Epochs:
            retirement_permitted=False
            def __init__(self, *a): pass
            async def ready(self): calls.append('epoch-ready')
            async def measure(self, point, repeat, **kwargs):
                return await measure(repeat=repeat,ownership={'started':False})
            async def retire(self): self.retirement_permitted=True;calls.append('epoch-retire')
            def released(self): calls.append('epoch-released')
            def fail(self, error): raise AssertionError('epoch unexpectedly failed: '+repr(error))
        monkeypatch.setattr(profile_epochs,'DynamoEpochs',Epochs)
        args.sampling_epoch_root=tmp_path/'cohort'
        args.sampling_member='dynamo-32b'
    result = await profile_v1.collect(args)
    assert calls.count('model-load') == int(not external) and calls.count('model-close') == int(not external)
    assert calls.count('measurement-stop') == int(external)
    assert [row[1] for row in calls if isinstance(row, tuple)] == [0,1,2,3]
    assert not result['coverage']['all_six_frequencies']
    assert not result['hardware_qualified'] and not result['formal_eligible']
    if epoch:
        from pdblend.experimentation.worker import receipt
        proofs=receipt(args.out,['completion.json','epoch-drain.json','epoch-release.json'])
        assert len(proofs)==3
        assert calls.index('epoch-retire') < calls.index('native-drain') < calls.index('model-close') < calls.index('epoch-released')
    own = PaperProfiles.load(args.out/'profile.json')
    assert own.query(2,900,128,144,1).decode_s == pytest.approx(.01)


def test_optional_natural_label_never_uses_reference_count_and_records_censoring():
    source = dict(prompt=[1,2,3], output_tokens=207)
    measured = actual_label(source, [dict(token_ids=list(range(12)), finished=True, finish_reason='stop')])
    assert measured['output_tokens'] == 12 and not measured['ignore_eos']
    capped = actual_label(source, [dict(token_ids=list(range(512)), finished=True, finish_reason='length')])
    assert capped['output_tokens'] == 512 and capped['right_censored']
    with pytest.raises(ValueError, match='truncation'):
        actual_label(source, [dict(token_ids=[1], finished=True, finish_reason='length')])
