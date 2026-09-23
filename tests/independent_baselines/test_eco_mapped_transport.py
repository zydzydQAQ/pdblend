import asyncio
import pytest
from pdblend_baselines.ecoserve.runtime import MappedEcoServeTransport,EcoServeCapabilityError


@pytest.mark.asyncio
async def test_concurrent_instance_routing_and_clock_stay_on_group():
    transport=MappedEcoServeTransport({'a':'http://a','b':'http://b'})
    transport.bind_specs({'a':{'gpus':[0,1]},'b':{'gpus':[2,3]}})
    calls=[]
    for identifier,client in transport.clients.items():
        async def call(method,path,body=None,identifier=identifier):
            await asyncio.sleep(.01 if identifier=='a' else 0)
            calls.append((identifier,method,path,body))
            return dict(acknowledged=True,identifier=identifier)
        client._call=call
    states=await asyncio.gather(transport.state('a'),transport.state('b'))
    assert [s['identifier'] for s in states]==['a','b']
    await transport.json('b','/control',dict(generation=1))
    await transport.clock([2,3],1800)
    assert calls[-2]==('b','POST','/baseline/control',dict(generation=1))
    assert calls[-1]==('b','POST','/baseline/clock',dict(frequency_mhz=1800))
    with pytest.raises(EcoServeCapabilityError,match='whole TP'):
        await transport.clock([1,2],1800)
