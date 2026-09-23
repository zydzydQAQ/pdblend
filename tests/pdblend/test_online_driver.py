"""CPU native-contract exercise; never marks synthetic endpoints as hardware evidence."""
import asyncio
import socket

from pdblend.bench.online_qualification import qualify
from synthetic import synthetic_model
from test_e2e import FakeFleet, FakeGpus


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_online_driver_abort_reuse_and_real_control_sequence_on_cpu(tmp_path):
    fleet = FakeFleet(2)
    for instance in fleet.instances.values():
        instance.spec.pool_id = 'cpu-test'
        instance.spec.profile_key = ''
        instance.spec.model = 'synthetic'
        instance.spec.base_url = f'http://127.0.0.1:{free_port()}'
    async def run():
        for instance in fleet.instances.values():
            await instance.serve()
        try:
            return await qualify(fleet, FakeGpus([0, 1]), synthetic_model(), tmp_path, free_port())
        finally:
            for instance in fleet.instances.values():
                await instance.runner.cleanup()
    result = asyncio.run(run())
    assert result['functional_passed'], result
    assert not result['hardware_qualified']
    assert result['checks']['native_cancel_and_kv_cleanup']
    assert result['checks']['tokens_during_control']
    assert len(result['terminal_native']) == 2
