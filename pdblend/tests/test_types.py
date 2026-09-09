# -*- coding: utf-8 -*-
"""types:Instance/Segment/Partition/SystemConfig 基础契约。"""
from __future__ import annotations

from ecopadg.types import (
    ROLE_DECODE, ROLE_MIXED, ROLE_PREFILL, InstanceSpec, Partition,
    Segment, SloSpec, SystemConfig,
)


def test_instance_spec_defaults():
    inst = InstanceSpec(role=ROLE_MIXED, model="Qwen2.5-32B", tp=2, gpus=(0, 1))
    assert inst.role == ROLE_MIXED
    assert inst.max_model_len == 8192
    assert inst.gpu_mem_util == 0.85
    assert inst.gpus == (0, 1)


def test_roles_distinct():
    assert len({ROLE_PREFILL, ROLE_DECODE, ROLE_MIXED}) == 3


def test_segment_holds_pair():
    seg = Segment(prefill=InstanceSpec(role=ROLE_PREFILL, model="m", tp=2, gpus=(0, 1)),
                  decode=InstanceSpec(role=ROLE_DECODE, model="m", tp=2, gpus=(2, 3)))
    assert seg.prefill.role == ROLE_PREFILL
    assert seg.decode.role == ROLE_DECODE


def test_partition_gpu_math():
    p = Partition(n_mixed=2, n_prefill=1, n_decode=1,
                  tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=8)
    assert p.total_gpus() == 8
    assert p.pairs() == 1
    p2 = Partition(n_mixed=0, n_prefill=2, n_decode=2,
                   tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=8)
    assert p2.pairs() == 2


def test_partition_invalid_gpu_count():
    p = Partition(n_mixed=1, n_prefill=1, n_decode=1,
                  tp_mixed=2, tp_prefill=2, tp_decode=2, gpus_total=4)
    assert p.total_gpus() == 6 > p.gpus_total  # 超配应能被发现


def test_slo_spec_and_defaults():
    slo = SloSpec(ttft_s=5.0, tpot_s=0.15)
    cfg = SystemConfig(model="Qwen2.5-32B")
    assert cfg.slo == slo
    assert SloSpec.from_dataset("sharegpt") == slo
    assert SloSpec.from_dataset("longbench").ttft_s == 15.0
    assert cfg.prefill_freq == 2520
    assert 2520 in cfg.freq_candidates
    assert cfg.gpu_count == 8
    assert cfg.target_attainment == 0.9
    assert cfg.slo_non_inferior_pp == 0.01
    assert cfg.baseline_att != cfg.baseline_att  # 未测 = nan
    assert cfg.global_period_s == 30.0
    p_pp = Partition(n_mixed=1, tp_mixed=2, pp_mixed=2, gpus_total=8)
    assert p_pp.total_gpus() == 4
