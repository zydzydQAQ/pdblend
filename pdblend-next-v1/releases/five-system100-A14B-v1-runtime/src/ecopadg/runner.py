# -*- coding: utf-8 -*-
"""runner:vLLM 实例生命周期(命令构造 / GPU 分配)。

统一 batching:mixed 实例 = chunked prefill(sarathi 式混批,V0 显式开、V1 默认);
prefill/decode 实例 = PyNcclConnector KV 传输对(连续 batching 语义);
prefill 实例不加 chunked 标志(只做 FCFS token 批)。
"""
from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ecopadg.gpu_topo import Placement, place_partition
from ecopadg.types import ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL, InstanceSpec, Partition


def kv_transfer_config(
    role: str,
    rank: int,
    parallel_size: int,
    kv_port: Optional[int] = None,
) -> str:
    """PyNcclConnector KV 传输配置 JSON(vLLM 0.9.2 官方 PD 方案)。"""
    if role not in ("kv_producer", "kv_consumer"):
        raise ValueError("role 必须是 kv_producer/kv_consumer")
    config = dict(
        kv_connector="PyNcclConnector",
        kv_role=role,
        kv_rank=rank,
        kv_parallel_size=parallel_size,
    )
    if kv_port is not None:
        config["kv_port"] = int(kv_port)
    return json.dumps(config)


def gpu_env(gpus: Sequence[int]) -> str:
    """CUDA_VISIBLE_DEVICES 字符串。"""
    return ",".join(str(int(g)) for g in gpus)


def build_vllm_cmd(inst: InstanceSpec, port: int, engine_v1: bool = True,
                   kv_pair: Optional[Sequence[int]] = None,
                   strict_padg: bool = False) -> List[str]:
    """构造 vllm serve 命令。

    strict_padg 只允许 V0 mixed，并显式关闭 chunked prefill，使一个
    scheduler step 内 prefill/decode 互斥；V1 强制 chunked，不能伪装成 PaDG。
    kv_pair=(kv_rank, kv_parallel_size)，仅空间 PD 角色使用。
    """
    if strict_padg and engine_v1:
        raise ValueError("strict temporal PaDG 需要 V0 engine")
    cmd = ["vllm", "serve", inst.model,
           "--host", "0.0.0.0", "--port", str(int(port)),
           "--max-model-len", str(int(inst.max_model_len)),
           "--gpu-memory-utilization", str(inst.gpu_mem_util),
           "--trust-remote-code",
           "--no-enable-prefix-caching",
           "--tensor-parallel-size", str(int(inst.tp))]
    if inst.role == ROLE_MIXED and not engine_v1:
        cmd += (["--no-enable-chunked-prefill"] if strict_padg
                else ["--enable-chunked-prefill"])
    if inst.role in (ROLE_PREFILL, ROLE_DECODE):
        if kv_pair is None:
            raise ValueError("PD 实例必须提供 kv_pair=(rank, parallel_size)")
        if len(kv_pair) not in (2, 3):
            raise ValueError("kv_pair 必须是 (rank,size[,port])")
        rank, psize = int(kv_pair[0]), int(kv_pair[1])
        kv_port = int(kv_pair[2]) if len(kv_pair) == 3 else None
        role_str = "kv_producer" if inst.role == ROLE_PREFILL else "kv_consumer"
        cmd += [
            "--kv-transfer-config",
            kv_transfer_config(role_str, rank, psize, kv_port),
        ]
    return cmd


def assign_gpus(partition: Partition,
                gpus: Sequence[int],
                numa_of: Optional[Mapping[int, int]] = None,
                *, force_interleave: bool = False) -> Dict[str, List[List[int]]]:
    """按 NUMA 同岛配对 P/D;探拓扑失败时段交错。不再先铺完 P 再铺 D。"""
    return place_partition(
        partition, gpus, numa_of=numa_of,
        force_interleave=force_interleave).as_assign()


def assign_gpus_placement(
        partition: Partition,
        gpus: Sequence[int],
        numa_of: Optional[Mapping[int, int]] = None,
        *, force_interleave: bool = False) -> Placement:
    """与 assign_gpus 同一放置,附带每段 NUMA / kv_cross_numa。"""
    return place_partition(
        partition, gpus, numa_of=numa_of,
        force_interleave=force_interleave)
