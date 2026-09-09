# -*- coding: utf-8 -*-
"""runner:vLLM 实例命令构造与 GPU 分配。"""
from __future__ import annotations

import json
import pytest

from ecopadg.runner import assign_gpus, build_vllm_cmd, kv_transfer_config
from ecopadg.types import ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL, InstanceSpec, Partition


def test_kv_transfer_config_json():
    cfg = kv_transfer_config(role="kv_producer", rank=0, parallel_size=2)
    d = json.loads(cfg)
    assert d["kv_connector"] == "PyNcclConnector"
    assert d["kv_role"] == "kv_producer"
    assert d["kv_rank"] == 0
    assert d["kv_parallel_size"] == 2


def test_mixed_cmd_chunked_prefill():
    inst = InstanceSpec(role=ROLE_MIXED, model="/models/Qwen2.5-32B-Instruct",
                        tp=2, gpus=(0, 1))
    cmd = build_vllm_cmd(inst, port=8000, engine_v1=False)
    assert "--enable-chunked-prefill" in cmd  # sarathi 式混批(V0 显式开)
    assert "--tensor-parallel-size" in cmd and "2" in cmd
    assert "--port" in cmd and "8000" in cmd
    cmd_v1 = build_vllm_cmd(inst, port=8001, engine_v1=True)
    assert "--enable-chunked-prefill" not in cmd_v1  # V1 默认 chunked
    strict = build_vllm_cmd(
        inst, port=8002, engine_v1=False, strict_padg=True)
    assert "--no-enable-chunked-prefill" in strict
    assert "--enable-chunked-prefill" not in strict
    with pytest.raises(ValueError):
        build_vllm_cmd(inst, port=8003, engine_v1=True, strict_padg=True)


def test_prefill_decode_cmd_kv_roles():
    pre = InstanceSpec(role=ROLE_PREFILL, model="m", tp=2, gpus=(0, 1))
    dec = InstanceSpec(role=ROLE_DECODE, model="m", tp=2, gpus=(2, 3))
    cp = build_vllm_cmd(pre, port=8100, engine_v1=False, kv_pair=(0, 2))
    cd = build_vllm_cmd(dec, port=8200, engine_v1=False, kv_pair=(1, 2))
    assert "kv_producer" in " ".join(cp)
    assert "kv_consumer" in " ".join(cd)
    assert "--enable-chunked-prefill" not in cp  # prefill 实例不混批(只 FCFS token 批)
    cp_port = build_vllm_cmd(
        pre, port=8100, engine_v1=False, kv_pair=(0, 2, 14579)
    )
    transfer = cp_port[cp_port.index("--kv-transfer-config") + 1]
    assert json.loads(transfer)["kv_port"] == 14579


def test_assign_gpus_no_overlap():
    p = Partition(n_mixed=1, n_prefill=1, n_decode=1, tp_mixed=2,
                  tp_prefill=2, tp_decode=2, gpus_total=8)
    a = assign_gpus(p, list(range(8)))
    flat = [g for grp in a.values() for pair in grp for g in pair]
    assert len(flat) == 6
    assert len(set(flat)) == 6  # 无重叠
    assert len(a["mixed"]) == 1 and len(a["mixed"][0]) == 2
    assert len(a["prefill"]) == 1 and len(a["decode"]) == 1
