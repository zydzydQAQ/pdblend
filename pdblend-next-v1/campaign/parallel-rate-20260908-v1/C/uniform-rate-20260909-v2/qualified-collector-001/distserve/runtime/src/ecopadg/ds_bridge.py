# -*- coding: utf-8 -*-
"""DS-PD 桥接门控与 least-loaded 路由 —— 移植 DistServe 两阶段调度器语义。

源码参照(DistServe,Apache-2.0):
  distserve/decoding_stage_scheduler.py::DecodingStageFCFSScheduler.post_process
    - unaccepted 桥队列:prefill 完成但 decode 未接受的请求;
    - should_accept:waiting_blocks < max_blocks * waiting_block_prop_threshold
      且该请求所需块 <= 可用块;循环中每接受一条即触发迁移并入等待队列,
      遇到拒绝即 break(与原文一致);
  distserve/scheduler.py::Scheduler._find_best_worker_and_queue
    - least-loaded 实例选择,平局 round-robin。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence


def blocks_needed(prompt_len: int, block_size: int = 16) -> int:
    """请求占用的 KV 块数(DistServe block 粒度 16 token)。"""
    return (int(prompt_len) + block_size - 1) // block_size


def should_accept(waiting_blocks: int, max_blocks: int, threshold: float,
                  blocks_needed: int, avail_blocks: int) -> bool:
    """DistServe post_process::should_accept 原语义移植。"""
    return (waiting_blocks < max_blocks * threshold
            and blocks_needed <= avail_blocks)


class SegmentBridge:
    """段内桥接:prefill 完成 → unaccepted 队列 → decode 有容量才迁移。"""

    def __init__(self, waiting_block_prop_threshold: float = 0.5,
                 max_blocks: int = 100, block_size: int = 16):
        self.threshold = float(waiting_block_prop_threshold)
        self.max_blocks = int(max_blocks)
        self.block_size = int(block_size)
        self.unaccepted: Deque[dict] = deque()

    def on_prefill_done(self, req: dict) -> None:
        """prefill 完成后请求进入桥队列(等待 decode 接受)。"""
        self.unaccepted.append(req)

    def try_accept(self, waiting_blocks: int, avail_blocks: int) -> List[dict]:
        """post_process 循环移植:逐条判断、接受即迁移,遇到拒绝即停。
        返回本次接受(触发迁移)的请求列表;waiting_blocks 随接受累加(与原文
        一致:接受后的请求计入 decode 等待队列)。"""
        accepted: List[dict] = []
        wb = int(waiting_blocks)
        while self.unaccepted:
            req = self.unaccepted[0]
            need = blocks_needed(int(req.get("prompt_len", 0)), self.block_size)
            if not should_accept(wb, self.max_blocks, self.threshold,
                                 need, avail_blocks):
                break
            self.unaccepted.popleft()
            accepted.append(req)
            wb += need
        return accepted


class LeastLoadedRouter:
    """least-loaded 路由(DistServe Scheduler 同款,round-robin 平局)。"""

    def __init__(self, n_prefill: int, n_decode: int):
        self.n_prefill = int(n_prefill)
        self.n_decode = int(n_decode)
        self._prefill_tt = 0
        self._decode_tt = 0

    @staticmethod
    def _pick(loads: Sequence[int], counter: int) -> int:
        if not loads:
            raise ValueError("loads 不能为空")
        m = min(loads)
        tied = [i for i, v in enumerate(loads) if v == m]
        if len(tied) == 1:
            return tied[0]
        c = counter % len(loads)
        for i in tied:  # 优先取 >= c 的最小 tied,否则回绕取最小
            if i >= c:
                return i
        return tied[0]

    def route_prefill(self, loads: Sequence[int]) -> int:
        idx = self._pick(loads, self._prefill_tt % self.n_prefill)
        self._prefill_tt += 1
        return idx

    def route_decode(self, loads: Sequence[int]) -> int:
        idx = self._pick(loads, self._decode_tt % self.n_decode)
        self._decode_tt += 1
        return idx
