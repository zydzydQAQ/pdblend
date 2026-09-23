import hashlib

import pytest
import torch

from pdblend_runtime.kv_digest import compare_digests, digest_paged_kv, digest_tensor


def identity(stage):
    return dict(transaction_id="tx", generation=2, request_id="decode-id", rank=1,
                layer="model.layers.0.self_attn.attn", stage=stage)


def test_p2p_gather_matches_received_and_injected_bytes_across_physical_slots():
    source = torch.arange(2 * 4 * 16 * 2 * 8).reshape(2, 4, 16, 2, 8).to(torch.bfloat16)
    source_slots = torch.tensor([17, 18, 19, 48])
    payload = source.reshape(2, 64, -1)[:, source_slots]
    destination = torch.zeros_like(source)
    target_slots = torch.tensor([32, 33, 34, 15])
    destination.reshape(2, 64, -1)[:, target_slots] = payload
    sent = digest_paged_kv(source, source_slots, **identity("source"))
    received = digest_tensor(payload, **identity("received"))
    injected = digest_paged_kv(destination, target_slots, **identity("injected"))
    assert sent["byte_count"] == 2 * 4 * 16 * 2
    assert sent["physical_slot_sha256"] != injected["physical_slot_sha256"]
    assert compare_digests(sent, received, injected)["exact_payload_match"]
    assert not compare_digests(sent, received, injected)["formal_eligible"]
    destination.reshape(2, 64, -1)[0, target_slots[-1], 0] = -1
    corrupted = digest_paged_kv(destination, target_slots, **identity("injected"))
    assert compare_digests(sent, received, corrupted)["status"] == "mismatch"


def test_bfloat16_is_hashed_without_lossy_cast_and_handles_noncontiguous_payload():
    bits = torch.tensor([0, -32768, 32704, 32641, 1, 2, 3, 4], dtype=torch.int16).reshape(2, 2, 2)
    payload = bits.view(torch.bfloat16).transpose(1, 2)
    digest = digest_tensor(payload, **identity("source"))
    expected = bits.transpose(1, 2).contiguous().numpy().tobytes()
    assert digest["sha256"] == hashlib.sha256(expected).hexdigest()


def test_stale_and_missing_receipts_never_compare_equal():
    payload = torch.ones(2, 2, 4, dtype=torch.float16)
    source = digest_tensor(payload, **identity("source"))
    received = digest_tensor(payload, **identity("received"))
    assert compare_digests(source, received)["status"] == "matched"
    received["generation"] = 1
    assert compare_digests(source, received)["status"] == "inconclusive"
    assert compare_digests({}, {})["status"] == "inconclusive"
    assert compare_digests(source, None)["status"] == "inconclusive"


def test_matching_digests_with_impossible_tensor_metadata_are_not_evidence():
    payload = torch.ones(2, 2, 4, dtype=torch.float16)
    source = digest_tensor(payload, **identity("source"))
    received = digest_tensor(payload, **identity("received"))
    source["byte_count"] = received["byte_count"] = 0
    assert compare_digests(source, received)["status"] == "inconclusive"


@pytest.mark.parametrize("slots", [[0, 0], [-1], [64], [], [1.0]])
def test_invalid_slots_fail_closed(slots):
    with pytest.raises(ValueError):
        digest_paged_kv(torch.zeros(2, 4, 16, 8), slots, **identity("source"))


def test_pinned_pool_descriptor_is_not_transfer_evidence():
    with pytest.raises(TypeError):
        digest_tensor((1234, torch.bfloat16, (2, 7168, 512)), **identity("received"))


def test_mla_layout_explicit():
    source = torch.arange(4 * 16 * 8).reshape(4, 16, 8).float()
    sent = digest_paged_kv(source, [0, 31], is_mla=True, **identity("source"))
    received = digest_tensor(source.reshape(64, 8)[[0, 31]], is_mla=True, **identity("received"))
    assert sent["shape"] == [2, 8]
    assert compare_digests(sent, received)["exact_payload_match"]
