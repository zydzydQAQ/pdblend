"""Real MappedDistServeTransport against actual native FastAPI SSE routes."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
_spec = importlib.util.spec_from_file_location('distserve_native_http_fixture',
    Path(__file__).with_name('test_ecoserve_native_http.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)

from pdblend_baselines.distserve.runtime import MappedDistServeTransport


@pytest.mark.asyncio
async def test_actual_mapped_decode_is_async_iterable_and_uses_decode_endpoint(tmp_path, monkeypatch):
    async with _fixture.native_services(monkeypatch, ('P', 'D')) as (engines, endpoints):
        transport = MappedDistServeTransport(endpoints['P'], endpoints['D'])
        values = [event async for event in transport.generate_decode(dict(
            request_id='actual-mapped-decode', prompt=list(range(100, 228)),
            max_tokens=16, seed=701, ignore_eos=True))]
        assert len(values) == 16 and values[-1]['finished']
        assert values[-1]['choices'][0]['finish_reason'] == 'length'
        assert [row['token_index'] for row in values] == list(range(1, 17))
        assert [token for row in values for token in row['token_ids']] == list(range(100, 116))
        assert engines['P'].outputs == []
        assert len(engines['D'].outputs) == 16
        assert not engines['D'].scheduler.native_state()['all_queue']
