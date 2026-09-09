import asyncio
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest

from ecopadg.serving import profiling


@pytest.mark.parametrize('interval', [None,.005,.001,1.])
def test_profile_passes_requested_interval_to_sampler_and_preserves_raw(tmp_path,monkeypatch,interval):
    calls=[]
    class Hardware:
        def __init__(self,**kwargs):
            assert kwargs==dict(power_mode='instant')
        def power_limit_w(self,gpu):return 350
    class Clocks:
        def __init__(self,*args):self.applied={}
        async def close(self):calls.append('clocks_restored')
    class Sampler:
        samples=[];frequency_samples=[];error=None
        def __init__(self,gpus,**kwargs):
            calls.append(('sampler',tuple(gpus),kwargs['interval']))
            assert isinstance(kwargs['backend'],Hardware) and kwargs['sample_clocks']
        def start(self):calls.append('sampler_started')
        def stop(self):calls.append('sampler_stopped')
    class Profiler:
        def __init__(self,*args):pass
        async def provenance(self):return [dict(image_id='test-image')]
    for name,value in [('PynvmlBackend',Hardware),('ClockOwner',Clocks),
                       ('PowerSampler',Sampler),('HardwareProfiler',Profiler)]:
        monkeypatch.setattr(profiling,name,value)
    topology=tmp_path/'topology.json'
    topology.write_text(json.dumps(dict(mixed=dict(id='m',tp=1,gpus=[0]))))
    # No inference is needed to verify the actual sampler construction and
    # normal measurement finalization, including the persisted raw metadata.
    args=SimpleNamespace(topology=topology,runtime_dir=tmp_path,out=tmp_path/'out',
        power_mode='instant',output_tokens=64,frequencies=[])
    if interval is not None:args.power_sample_interval_s=interval
    asyncio.run(profiling.profile(args))
    expected=.02 if interval is None else interval
    assert calls==[('sampler',tuple(range(8)),expected),'sampler_started',
                   'sampler_stopped','clocks_restored']
    raw=json.loads((args.out/'raw.json').read_text())
    assert raw['complete'] and raw['cleanup_complete']
    assert raw['requested_power_sample_interval_s']==expected


@pytest.mark.parametrize('value', ['nan','inf','-inf','0','-0.01','0.0009','1.001'])
def test_invalid_cli_interval_is_rejected_before_lease_or_gpu(tmp_path,monkeypatch,value):
    def forbidden(*args,**kwargs):pytest.fail('invalid interval reached hardware/lifecycle')
    for name in ('PynvmlBackend','node_lease','run_async_cli'):
        monkeypatch.setattr(profiling,name,forbidden)
    monkeypatch.setattr(sys,'argv',['profiling','--topology',str(tmp_path/'missing.json'),
        '--runtime-dir',str(tmp_path),'--out',str(tmp_path/'out'),
        '--power-sample-interval-s='+value])
    with pytest.raises(SystemExit) as error:profiling.main()
    assert error.value.code==2 and not (tmp_path/'out').exists()


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-.01,0,.0009,1.001])
def test_invalid_direct_interval_is_rejected_before_reading_topology_or_gpu(monkeypatch,value):
    def forbidden(*args,**kwargs):pytest.fail('invalid direct interval opened hardware')
    monkeypatch.setattr(profiling,'PynvmlBackend',forbidden)
    with pytest.raises(ValueError,match='power sample interval'):
        asyncio.run(profiling.profile(SimpleNamespace(power_sample_interval_s=value)))


@pytest.mark.parametrize('option,expected', [([],.02),(['--power-sample-interval-s','.005'],.005)])
def test_cli_exposes_interval_without_changing_workload_or_power_defaults(tmp_path,monkeypatch,option,expected):
    captured=[]
    async def profile(args):captured.append(args)
    monkeypatch.setattr(profiling,'profile',profile)
    monkeypatch.setattr(profiling,'node_lease',nullcontext)
    monkeypatch.setattr(profiling,'run_async_cli',lambda awaitable,**kwargs:asyncio.run(awaitable))
    monkeypatch.setattr(sys,'argv',['profiling','--topology',str(tmp_path/'topology.json'),
        '--runtime-dir',str(tmp_path),'--out',str(tmp_path/'out'),*option])
    profiling.main()
    args=captured[0]
    assert args.power_sample_interval_s==expected and args.power_mode=='average'
    assert args.inputs==[2048,4096,7168] and args.batches==[1,4,8,16,32]
    assert args.frequencies==[900,1500,2100,2520] and args.output_tokens==512
