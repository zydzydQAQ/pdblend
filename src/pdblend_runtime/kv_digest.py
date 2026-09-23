"""Optional exact-byte diagnostics for symmetric P2P KV handoffs.

These helpers synchronize a tensor copy to CPU.  Enable them only in dedicated
correctness probes, never in a performance or power measurement window.  A
matching digest proves transport of the inspected tensor, not output golden
equivalence, slot correctness, lifetime correctness, or whole-request coverage.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from typing import Any


def _identity(transaction_id, generation, request_id, rank, layer, stage):
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("transaction_id must be a nonempty string")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must name the target decode request")
    if type(generation) is not int or generation < 0:
        raise ValueError("generation must be a nonnegative integer")
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if not isinstance(layer, (str, int)) or isinstance(layer, bool) or str(layer) == "":
        raise ValueError("layer must identify an attention layer")
    if stage not in ("source", "received", "injected"):
        raise ValueError("stage must be source, received, or injected")
    return dict(transaction_id=transaction_id, generation=generation,
                request_id=request_id, rank=rank, layer=str(layer), stage=stage)


def digest_tensor(tensor, *, transaction_id: str, generation: int,
                  request_id: str, rank: int, layer: str | int, stage: str,
                  is_mla: bool = False) -> dict[str, Any]:
    """Hash actual P2P payload bytes without converting BF16 values to FP32.

    Non-MLA P2P sends ``[2, token_count, flattened_head_dimension]``; MLA sends
    ``[token_count, flattened_head_dimension]``.  A pinned-pool descriptor
    ``(address, dtype, shape)`` is not a tensor and is deliberately rejected.
    Call this after ``recv_tensor`` has completed its H2D copy, or on the source
    gathered tensor before it is sent.  A caller using another CUDA stream must
    establish the same producer-event dependency used by the actual transfer.
    """
    import torch

    identity = _identity(transaction_id, generation, request_id, rank, layer, stage)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("digest requires the actual tensor, not a pool descriptor or address")
    if tensor.layout != torch.strided or tensor.ndim != (2 if is_mla else 3):
        raise ValueError("tensor does not have the P2P payload layout")
    if not is_mla and tensor.shape[0] != 2:
        raise ValueError("non-MLA payload must contain both K and V")
    if tensor.numel() == 0 or any(size <= 0 for size in tensor.shape):
        raise ValueError("empty KV cannot establish transfer correctness")
    # View the original representation as bytes: numpy cannot directly expose
    # BF16, and float casts would hide payload-bit differences (including NaNs).
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(order="C")
    expected = tensor.numel() * tensor.element_size()
    if len(raw) != expected:
        raise RuntimeError("KV byte length differs from tensor metadata")
    return dict(schema=1, **identity, dtype=str(tensor.dtype), shape=list(tensor.shape),
                byte_order=sys.byteorder, byte_count=expected,
                token_count=int(tensor.shape[0 if is_mla else 1]), is_mla=bool(is_mla),
                sha256=hashlib.sha256(raw).hexdigest(),
                evidence_scope="one_rank_one_layer_exact_payload", formal_eligible=False)


def digest_paged_kv(kv_layer, slot_mapping, *, is_mla: bool = False, **identity) -> dict[str, Any]:
    """Gather the same logical slots as P2pNcclEngine.extract_kv_from_layer.

    Physical source and destination slot IDs may differ.  Their slot hashes are
    recorded independently, while comparison uses the logical payload digest.
    For ``stage='injected'`` call immediately after load, before the next forward
    recomputes the final prompt token and changes the destination KV in place.
    """
    import torch

    if not isinstance(kv_layer, torch.Tensor) or kv_layer.ndim < (3 if is_mla else 4):
        raise ValueError("paged KV must be a tensor with page and slot dimensions")
    if not is_mla and kv_layer.shape[0] != 2:
        raise ValueError("paged non-MLA KV must contain K and V")
    slots = torch.as_tensor(slot_mapping)
    if slots.ndim != 1 or slots.numel() == 0 or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot_mapping must be a nonempty integer vector")
    slots = slots.detach().cpu().to(torch.int64)
    pages, page_size = (kv_layer.shape[:2] if is_mla else kv_layer.shape[1:3])
    if int(slots.min()) < 0 or int(slots.max()) >= pages * page_size:
        raise ValueError("slot_mapping contains an out-of-range slot")
    if torch.unique(slots).numel() != slots.numel():
        raise ValueError("slot_mapping repeats a token slot")
    if is_mla:
        payload = kv_layer.reshape(pages * page_size, -1)[slots, ...]
    else:
        payload = kv_layer.reshape(2, pages * page_size, -1)[:, slots, ...]
    result = digest_tensor(payload, is_mla=is_mla, **identity)
    encoded_slots = json.dumps(slots.tolist(), separators=(",", ":")).encode()
    result["physical_slot_sha256"] = hashlib.sha256(encoded_slots).hexdigest()
    result["physical_slot_count"] = int(slots.numel())
    return result


def compare_digests(source: dict, received: dict, injected: dict | None = None) -> dict:
    """Fail closed on missing/stale evidence; never qualify an output golden."""
    receipts = [source, received] + ([] if injected is None else [injected])
    expected_stages = ["source", "received"] + ([] if injected is None else ["injected"])
    fields = ("transaction_id", "generation", "request_id", "rank", "layer",
              "dtype", "shape", "byte_order", "byte_count", "token_count", "is_mla")
    errors = []
    for receipt, stage in zip(receipts, expected_stages):
        if not isinstance(receipt, dict):
            errors.append(f"missing_{stage}_receipt")
            continue
        try:
            _identity(*(receipt[k] for k in ("transaction_id", "generation", "request_id",
                                             "rank", "layer")), receipt["stage"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_{stage}_identity")
        digest = receipt.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"invalid_{stage}_digest")
        if receipt.get("stage") != stage or any(k not in receipt for k in fields):
            errors.append(f"incomplete_{stage}_receipt")
        shape = receipt.get("shape")
        mla = receipt.get("is_mla")
        dtype = receipt.get("dtype")
        item_bytes = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4,
                      "torch.float64": 8, "torch.int8": 1, "torch.uint8": 1,
                      "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1}.get(dtype) if isinstance(dtype, str) else None
        valid_shape = (type(mla) is bool and isinstance(shape, list) and
                       len(shape) == (2 if mla else 3) and
                       all(type(x) is int and x > 0 for x in shape) and
                       (mla or shape[0] == 2))
        if (receipt.get("schema") != 1 or not valid_shape or item_bytes is None or
                receipt.get("byte_order") not in ("little", "big") or
                (valid_shape and (receipt.get("token_count") != shape[0 if mla else 1] or
                                  receipt.get("byte_count") != math.prod(shape) * (item_bytes or 0)))):
            errors.append(f"invalid_{stage}_tensor_metadata")
        if isinstance(source, dict) and any(receipt.get(k) != source.get(k) for k in fields):
            errors.append(f"{stage}_identity_or_layout_mismatch")
    match = not errors and all(x["sha256"] == source["sha256"] for x in receipts)
    return dict(status="inconclusive" if errors else ("matched" if match else "mismatch"),
                exact_payload_match=bool(match), errors=errors,
                injection_checked=injected is not None,
                evidence_scope="one_rank_one_layer_transport_only", formal_eligible=False,
                output_golden_verified=False)
