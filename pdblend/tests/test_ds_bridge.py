# -*- coding: utf-8 -*-
"""ds_bridge:DistServe 桥接门控与 least-loaded 路由语义移植。"""
from __future__ import annotations

from ecopadg.ds_bridge import (
    LeastLoadedRouter, SegmentBridge, blocks_needed, should_accept,
)


def test_blocks_needed():
    assert blocks_needed(prompt_len=31, block_size=16) == 2
    assert blocks_needed(prompt_len=32, block_size=16) == 2
    assert blocks_needed(prompt_len=33, block_size=16) == 3


def test_should_accept_port():
    # DistServe: waiting_blocks < max*threshold 且 所需块 <= 可用块
    assert should_accept(waiting_blocks=4, max_blocks=100, threshold=0.5,
                         blocks_needed=2, avail_blocks=10)
    assert not should_accept(waiting_blocks=50, max_blocks=100, threshold=0.5,
                             blocks_needed=2, avail_blocks=10)  # 等待块达阈值
    assert not should_accept(waiting_blocks=4, max_blocks=100, threshold=0.5,
                             blocks_needed=2, avail_blocks=1)  # 可用块不足


def test_bridge_fifo_accept_until_blocked():
    br = SegmentBridge(waiting_block_prop_threshold=0.5, max_blocks=100,
                        block_size=16)
    for i in range(3):
        br.on_prefill_done(dict(rid=i, prompt_len=31 + 32 * i))
    # decode 侧等待块 40(>=50 的一半?40 < 50 OK)
    got = br.try_accept(waiting_blocks=40, avail_blocks=1000)
    assert [g["rid"] for g in got] == [0, 1, 2]
    assert len(br.unaccepted) == 0


def test_bridge_stops_at_first_reject():
    br = SegmentBridge(waiting_block_prop_threshold=0.5, max_blocks=100,
                        block_size=16)
    br.on_prefill_done(dict(rid=0, prompt_len=31))
    br.on_prefill_done(dict(rid=1, prompt_len=31))
    # 首条可接受,但接受后 decode 等待块超阈值 → 第二条拒绝(与 post_process 的 break 一致)
    got = br.try_accept(waiting_blocks=49, avail_blocks=1000)
    assert [g["rid"] for g in got] == [0]
    assert len(br.unaccepted) == 1


def test_least_loaded_prefill():
    r = LeastLoadedRouter(n_prefill=3, n_decode=2)
    assert r.route_prefill([5, 1, 3]) == 1
    # round-robin 平局推进(独立 router,计数从 0 起)
    r2 = LeastLoadedRouter(n_prefill=3, n_decode=2)
    assert r2.route_prefill([2, 2, 2]) == 0
    assert r2.route_prefill([2, 2, 2]) == 1
    assert r2.route_prefill([2, 2, 2]) == 2


def test_least_loaded_decode():
    r = LeastLoadedRouter(n_prefill=2, n_decode=2)
    assert r.route_decode([3, 0]) == 1
    r2 = LeastLoadedRouter(n_prefill=2, n_decode=2)
    assert r2.route_decode([0, 0]) == 0
    assert r2.route_decode([0, 0]) == 1
