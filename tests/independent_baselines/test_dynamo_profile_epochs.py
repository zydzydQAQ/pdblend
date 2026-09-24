import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend_baselines.dynamollm import profile_epochs, profile_v1
from pdblend_baselines.dynamollm.deployment import save, sha


def receipt(root, mode='parallel'):
    layout = {'dynamo-7b':['GPU-own'], 'pdblend-14b':['GPU-peer']}
    probe = dict(gpu_uuids=['GPU-own'], artifacts={})
    value = dict(complete=True, cross_job=True, passed=mode=='parallel',
        fallback=None if mode=='parallel' else 'serial_cohort',
        cohort_id='test:epoch:0', member='dynamo-7b', members=list(layout),
        isolated=[probe, {}], parallel=[probe, {}])
    path = root/'qualification-epochs/0000/samples/external-interference.json'
    save(path, value)
    return dict(epoch=0, epoch_id=value['cohort_id'], qualification_path=str(path),
        qualification_sha256=sha(path), layout=layout,
        layout_sha256=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        measured_mode=mode)


@pytest.mark.parametrize('mode', ['parallel','serial_cohort'])
def test_bound_window_rechecks_receipt_layout_and_mode(tmp_path, mode):
    binding = receipt(tmp_path, mode)
    binding['qualification_path'] = str(Path(binding['qualification_path']).relative_to(tmp_path))
    value = dict(started_s=3, finished_s=8, sampling_epoch=dict(binding,
        guard_before_at_s=2.9, guard_after_at_s=8.1))
    assert profile_epochs.validate_window_qualification(value, tmp_path)['sha256'] == binding['qualification_sha256']
    altered = copy.deepcopy(value)
    altered['sampling_epoch']['layout']['dynamo-7b'] = ['GPU-other']
    with pytest.raises(ValueError, match='layout checksum'):
        profile_epochs.validate_window_qualification(altered, tmp_path)
    altered = copy.deepcopy(value)
    altered['sampling_epoch']['guard_after_at_s'] = 7.9
    with pytest.raises(ValueError, match='chronology'):
        profile_epochs.validate_window_qualification(altered, tmp_path)
    (tmp_path/binding['qualification_path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        profile_epochs.validate_window_qualification(value, tmp_path)


@pytest.mark.asyncio
async def test_repeat_restores_clock_after_boundary_and_rejects_epoch_change(tmp_path, monkeypatch):
    events = []
    binding = receipt(tmp_path)
    class Epoch:
        async def window_boundary(self, point, repeat, phase):
            events.append(('boundary', repeat, phase))
        def qualification_guard(self):
            events.append('guard')
            return dict(binding)
    async def measured(*args, **kwargs):
        events.append('measured')
        import time
        return dict(started_s=time.time(), finished_s=time.time(), repeat=kwargs['repeat'])
    monkeypatch.setattr(profile_v1, 'measure_window', measured)
    owner = object.__new__(profile_epochs.DynamoEpochs)
    owner.output, owner.iid, owner.ownership = tmp_path, 'native', {}
    owner.transport = SimpleNamespace(instances={'native':{'gpus':[0]}})
    owner.telemetry = SimpleNamespace(clock=lambda g,f:events.append(('clock',g,f)))
    owner.epochs = Epoch()
    point = dict(frequency_mhz=900)
    value = await owner.measure(point, 3, settle_s=2, measure_s=5)
    assert events == [('boundary',3,'holdout'),('clock',[0],900),'guard','measured','guard']
    assert value['sampling_epoch']['qualification_path'].startswith('qualification-epochs/')
    assert profile_epochs.validate_window_qualification(value,tmp_path)
    calls = 0
    def changed():
        nonlocal calls
        calls += 1
        return dict(binding, epoch=calls)
    owner.epochs.qualification_guard = changed
    with pytest.raises(RuntimeError, match='crossed'):
        await owner.measure(point, 0, settle_s=2, measure_s=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('probe_seconds', [5., 8.])
async def test_probe_uses_own_collector_three_repeats_with_barrier(tmp_path, monkeypatch, probe_seconds):
    windows, barriers = [], []
    async def measured(*args, **kwargs):
        await kwargs['before_measure']()
        windows.append(kwargs)
        return dict(started_s=10.+kwargs['repeat']*10, finished_s=15.+kwargs['repeat']*10)
    monkeypatch.setattr(profile_v1,'measure_window',measured)
    monkeypatch.setattr(profile_v1,'reduce_window',lambda *a,**kw:dict(iteration_s=.01,power_w=123.))
    class Wave:
        qualification_measure_s = probe_seconds
        def write(self, phase, data): barriers.append(('write',phase))
        async def wait(self, phase): barriers.append(('wait',phase))
    owner = object.__new__(profile_epochs.DynamoEpochs)
    owner.iid, owner.ownership = 'dynamo-own', {}
    owner.transport = SimpleNamespace(instances={owner.iid:dict(tp=1,gpus=[0])})
    owner.telemetry = SimpleNamespace(clock=lambda g,f:None)
    facade = profile_epochs.QualificationFacade(tmp_path,dict(gpu_uuids={'0':'GPU-own'}))
    result = await owner.probe(facade,'parallel',Wave())
    assert [row['repeat'] for row in windows] == [0,1,2]
    assert all(row['settle_s']==2 and row['measure_s']==probe_seconds for row in windows)
    assert len(barriers)==6 and len(result['artifacts'])==3
    assert result['instances'][0]['step_seconds']==.01
    assert result['point']['batch']==1
    for name, expected in result['artifacts'].items():
        assert sha(tmp_path/name)==expected


@pytest.mark.asyncio
async def test_required_epoch_refuses_unqualified_job_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_v1,'model_identity',lambda _:dict(model='Qwen2.5-7B-Instruct'))
    monkeypatch.delenv('PDBLEND_SAMPLING_EPOCH_ROOT',raising=False)
    monkeypatch.delenv('PDBLEND_PROFILE_MEMBER',raising=False)
    def forbidden(*args): raise AssertionError('GPU telemetry must not be touched')
    monkeypatch.setattr(profile_v1,'GroupTelemetry',forbidden)
    args=SimpleNamespace(model=tmp_path,tp=1,gpus=[0],out=tmp_path/'output',
        resume=False,inputs=[128],outputs=[16],batches=[1],freqs=[900],require_sampling_epochs=True)
    with pytest.raises(ValueError,match='requires a frozen'):
        await profile_v1.collect(args)
