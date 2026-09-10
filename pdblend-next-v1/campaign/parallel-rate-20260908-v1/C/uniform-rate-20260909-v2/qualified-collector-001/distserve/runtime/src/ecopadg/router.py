# -*- coding: utf-8 -*-
"""Router 纯逻辑:准入门控(排队/开合)与逐请求遥测记录。"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, List, Optional

from ecopadg.types import SloSpec


def prompt_tokens(prompt: str, body: Optional[dict] = None,
                  tokenizer=None) -> int:
    """优先 tokenizer / 请求体 token 数;启发式只作回退。"""
    if body:
        for k in ("prompt_len", "prompt_tokens"):
            v = body.get(k)
            if v not in (None, ""):
                try:
                    return max(int(v), 0)
                except (TypeError, ValueError):
                    pass
    if tokenizer is not None and prompt:
        try:
            return max(len(tokenizer.encode(
                prompt, add_special_tokens=False)), 0)
        except Exception:
            pass
    return estimate_prompt_tokens(prompt, body)


def estimate_prompt_tokens(prompt: str, body: Optional[dict] = None) -> int:
    """优先用请求体里的 token 数;否则 CJK 按字、拉丁按词或 chars/4。"""
    if body:
        for k in ("prompt_len", "prompt_tokens"):
            v = body.get(k)
            if v not in (None, ""):
                try:
                    return max(int(v), 0)
                except (TypeError, ValueError):
                    pass
    if not prompt:
        return 0
    n_cjk = sum(1 for ch in prompt if ord(ch) > 0x2E80)
    rest = max(len(prompt) - n_cjk, 0)
    return max(1, n_cjk + max(len(prompt.split()), rest // 4))


def classify_slo(ttft_s: Optional[float], tpot_s: Optional[float],
                 slo: SloSpec) -> bool:
    """严格小于 SLO 才算达标(成功 = TTFT 与 TPOT 同时满足)。"""
    if ttft_s is None or tpot_s is None:
        return False
    return ttft_s < slo.ttft_s and tpot_s < slo.tpot_s


class RequestRecord:
    """一条请求的遥测记录(与 bench.csv 逐请求 schema 对齐)。"""

    def __init__(self, rid: int, arrival_s: float, prompt_len: int,
                 output_len: int):
        self.rid = rid
        self.arrival_s = float(arrival_s)
        self.prompt_len = int(prompt_len)
        self.output_len = int(output_len)
        self.release_s: Optional[float] = None
        self.ttft_s: Optional[float] = None
        self.tpot_s: Optional[float] = None
        self.latency_s: Optional[float] = None
        self.success: bool = False
        self.error: str = ""

    def slo_ok(self, slo: SloSpec) -> bool:
        return classify_slo(self.ttft_s, self.tpot_s, slo)

    def as_row(self, slo: SloSpec) -> dict:
        return dict(rid=self.rid, success=int(self.success),
                    prompt_len=self.prompt_len, output_len=self.output_len,
                    latency_s=(round(self.latency_s, 6)
                               if self.latency_s is not None else ""),
                    ttft_s=(round(self.ttft_s, 6)
                            if self.ttft_s is not None else ""),
                    tpot_s=(round(self.tpot_s, 6)
                            if self.tpot_s is not None else ""),
                    slo_ok=int(self.slo_ok(slo)), error=self.error)


class AdmissionGate:
    """代理侧准入队列:FIFO;关闭时不出队;支持单次释放上限。"""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._open = True
        self._q: Deque[RequestRecord] = deque()
        self.log: List = []  # (t, open_bool)

    def is_open(self) -> bool:
        return self._open

    def set_open(self, open_: bool, now: float) -> None:
        self._open = bool(open_)
        self.log.append((float(now), self._open))

    def submit(self, rec: RequestRecord) -> None:
        self._q.append(rec)

    def pending(self) -> int:
        return len(self._q)

    def release_due(self, now: float, limit: Optional[int] = None) -> List[RequestRecord]:
        """开门时按 FIFO 释放(最多 limit 条),并记录 release_s。"""
        if not self._open:
            return []
        out: List[RequestRecord] = []
        while self._q and (limit is None or len(out) < limit):
            rec = self._q.popleft()
            rec.release_s = float(now)
            out.append(rec)
        return out


def admit_request(gate: AdmissionGate, sched, rec: RequestRecord) -> None:
    """C0.1:AdmissionGate 与 PaDGWindowScheduler.submit 必须同时入队.

    gate 负责 FIFO 遥测/release_s;sched 负责窗口释放时机。
    二者按到达顺序入队,窗口释放后按相同条数 FIFO 从 gate 对齐出队。
    """
    gate.submit(rec)
    sched.submit(rec.rid, rec.arrival_s, rec.prompt_len, rec.output_len)
