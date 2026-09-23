import asyncio

import pytest

from pdblend_baselines.distserve.runtime import DistServeCapabilityError, DistServeRuntime
from pdblend_baselines.ecoserve.runtime import EcoServeCapabilityError, validate_native_state


def state():
    return {"free_kv_tokens": 4096, "total_kv_tokens": 4096, "block_size": 16,
            "generation": 2, "acknowledged_generation": 2, "native_at_s": 1.0,
            "native_evidence_complete": True, "error": None, "runtime_error": None}


class FakeDist:
    def __init__(self, good=True): self.good, self.calls = good, []
    async def capability(self): return {"supported": self.good, "tp": 1, "pp": 1}
    async def state(self): return state()
    async def prefill(self, request):
        self.calls.append("prefill")
        return {"acknowledged": True, "kv_handle": "kv-" + request.request_id}
    async def decode(self, request):
        self.calls.append("decode")
        return {"acknowledged": True, "generated_tokens": 2,
                "release": {"acknowledged": True}}
    async def cancel(self, request_id): return {"acknowledged": True}


def test_distserve_rejects_obsolete_single_endpoint_decode_receipt_transport():
    async def run():
        transport = FakeDist()
        runtime = DistServeRuntime(transport, num_gpu_blocks=256)
        # A synthetic decode/release dict cannot replace separate native P/D,
        # transaction/rank receipts and actual SSE. The HTTP integration tests
        # exercise the replacement execution contract end to end.
        with pytest.raises(DistServeCapabilityError, match='explicit P/D KV addresses'):
            await runtime.start()
        assert not transport.calls
    asyncio.run(run())


def test_distserve_fails_closed_when_capability_missing():
    async def run():
        with pytest.raises(DistServeCapabilityError):
            await DistServeRuntime(FakeDist(False)).start()
    asyncio.run(run())


def test_eco_state_requires_complete_native_evidence():
    validate_native_state(state())
    bad = state(); bad["native_evidence_complete"] = False
    with pytest.raises(EcoServeCapabilityError): validate_native_state(bad)
    bad = state(); bad["acknowledged_generation"] = 1
    with pytest.raises(EcoServeCapabilityError): validate_native_state(bad)
