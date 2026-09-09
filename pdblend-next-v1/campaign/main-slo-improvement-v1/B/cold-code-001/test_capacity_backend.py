import asyncio
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_backend import PinnedDockerBackend
from capacity_executor import Inventory


class Response:
    status = 200
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def json(self):
        return {'id':'actual'}


class Session:
    def __init__(self):
        self.calls = []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response()
    get = post


def test_request_uses_actual_header_identity_for_later_cancellation(tmp_path):
    session = Session()
    backend = object.__new__(PinnedDockerBackend)
    backend.controller = SimpleNamespace(session=session)
    backend.inventory = Inventory(tmp_path/'inventory.json', [], {})
    asyncio.run(backend.request(dict(id='new',url='http://127.0.0.1:1234'), '/v1/completions',
        dict(prompt=[1],max_tokens=1,request_id='controlled-id'), limit=time.time()+10))
    assert session.calls[0][1]['headers'] == {'X-Request-Id':'controlled-id'}


def test_expired_native_operation_is_rejected_before_dispatch(tmp_path):
    session = Session()
    backend = object.__new__(PinnedDockerBackend)
    backend.controller = SimpleNamespace(session=session)
    backend.inventory = Inventory(tmp_path/'inventory.json', [], {})
    with pytest.raises(ValueError, match='deadline'):
        asyncio.run(backend.request(dict(id='new',url='http://127.0.0.1:1234'), '/runtime', limit=time.time()-1))
    assert not session.calls
