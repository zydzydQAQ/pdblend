"""Isolated EcoServe author-policy port, milliseconds throughout this module.

Derived from MLSysU/EcoServe e7d7f7fe29e20c0218afac305f157dde4513de76,
Apache-2.0. The exact sources/license/hashes are in baselines/ecoserve/references.
The unusual max, next-instance branch, block rounding and integer truncation
are intentional author behavior, not the paper's alternative mean policy.
"""
from collections import deque
import csv
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import time


class PrefillProfile:
    def __init__(self, points, *, source_sha256=None):
        self.points = dict(points)
        if not {16, 4096} <= self.points.keys():
            raise ValueError('official EcoServe profile requires both 16 and 4096 anchors')
        if any(type(k) is not int or k <= 0 or not math.isfinite(v) or v <= 0
               for k, v in self.points.items()):
            raise ValueError('positive measured token lengths and milliseconds required')
        self.source_sha256 = source_sha256

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = path.read_bytes()
        rows = csv.DictReader(raw.decode().splitlines())
        return cls({int(row['Length']): float(row['Prefill Time']) for row in rows},
                   source_sha256=hashlib.sha256(raw).hexdigest())

    def predict_ms(self, num_tokens):
        if type(num_tokens) is not int or num_tokens < 1:
            raise ValueError('positive token count required')
        if num_tokens < 16:
            return int(self.points[16] * num_tokens / 16)
        return int(self.points.get(num_tokens, self.points[4096] * num_tokens / 4096))


@dataclass
class RequestState:
    request_id: str
    arrival_time: float
    num_iterations: int
    ttft: float
    predict_time: int
    predict_length: int
    prefill_blocks: int


@dataclass
class InstanceState:
    instance_id: str
    requests: deque = field(default_factory=deque)
    waiting_queue: list = field(default_factory=list)
    free_blocks: int = 0
    prefill_mode: bool = False
    schedule_time: float = 0
    max_predict_time: int = 0


class OfficialMacro:
    def __init__(self, identifiers, profile, ttft_ms, tpot_ms, *, now_ms=None):
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError('nonempty distinct EcoServe members required')
        self.now_ms = now_ms or (lambda: time.time() * 1000)
        self.profile = profile
        self.TTFT, self.TPOT = ttft_ms, tpot_ms
        if any(not math.isfinite(v) or v <= 0 for v in (ttft_ms, tpot_ms)):
            raise ValueError('positive finite millisecond SLOs required')
        self.instance_count = len(identifiers)
        self.instance_states = [InstanceState(i, schedule_time=self.now_ms()) for i in identifiers]
        self.prefill_instance = 0
        self.controls = []

    @property
    def identifiers(self):
        return tuple(i.instance_id for i in self.instance_states)

    def update_state(self, state):
        """Consume a native scheduler event, before execution, exactly as upstream."""
        instance = self.instance_states[state.instance_id]
        instance.free_blocks = state.free_blocks
        instance.prefill_mode = state.prefill_mode
        finished = []
        for request in instance.requests:
            if request.request_id not in state.all_queue and request.num_iterations != 0:
                finished.append(request)
                continue
            if not state.prefill_mode and request.request_id in state.schedule_queue:
                if request.num_iterations == 0:
                    request.ttft = state.schedule_time - request.arrival_time
                request.num_iterations += 1
        for request in finished:
            instance.requests.remove(request)

    def _get_predict_time(self, num_tokens):
        return self.profile.predict_ms(num_tokens)

    def _check_constraints(self, num_blocks, predict_time):
        saved, pending = [], []
        need_blocks, need_time = num_blocks, predict_time
        instance = self.instance_states[self.prefill_instance]
        for request in instance.requests:
            if request.request_id in instance.waiting_queue:
                pending.append(self.TTFT)
                need_blocks += request.prefill_blocks
                need_time += request.predict_time
            else:
                saved.append(request.ttft + request.num_iterations * self.TPOT
                             - instance.schedule_time + request.arrival_time)
        if need_blocks > instance.free_blocks:
            return False
        if saved:
            saved = [max(saved)]
        remaining = pending + saved
        if not remaining:
            return True
        if min(remaining) > need_time:
            return True
        if need_time > self.TTFT:
            return False
        instance = self.instance_states[(self.prefill_instance + 1) % self.instance_count]
        saved = [request.ttft + request.num_iterations * self.TPOT - self.now_ms()
                 + request.arrival_time for request in instance.requests]
        if not saved:
            return False
        return max(saved) < self.TTFT * (self.instance_count - 1) / self.instance_count

    def _switch_instance(self):
        now = self.now_ms()
        self.instance_states[self.prefill_instance].waiting_queue = []
        self.controls.append((self.prefill_instance, True))
        selected = (self.prefill_instance + 1) % self.instance_count
        self.controls.append((selected, False))
        self.instance_states[selected].schedule_time = now
        return selected

    def schedule(self, request):
        now = self.now_ms()
        count = request.prompt_len
        prediction = self._get_predict_time(count)
        # Deliberately preserve upstream (n+16)//16, including exact multiples.
        blocks = (count + 16) // 16
        state = RequestState(request.request_id, now, 0, self.TTFT, prediction, -1, blocks)
        selected = self.prefill_instance if self._check_constraints(blocks, prediction) else self._switch_instance()
        self.prefill_instance = selected
        self.instance_states[selected].requests.append(state)
        self.instance_states[selected].waiting_queue.append(request.request_id)
        return selected

    def forget_cancelled(self, request_id):
        """Neutral cancellation cleanup; scheduler tokens are never synthesized."""
        for instance in self.instance_states:
            instance.requests = deque(r for r in instance.requests if r.request_id != request_id)
            instance.waiting_queue[:] = [rid for rid in instance.waiting_queue if rid != request_id]


class OutputBuffer:
    """Port of author Instance._send_outputs: flush only when output arrives."""
    def __init__(self):
        self.send_output = True
        self.ttft_ms = 0
        self.prefill_time_ms = 0
        self.pending = []

    def control(self, send_output, ttft_ms, *, now_ms):
        self.send_output, self.ttft_ms, self.prefill_time_ms = send_output, ttft_ms, now_ms

    def receive(self, request_id, event, *, now_ms, unfinished):
        self.pending.append((request_id, event))
        if self.send_output or now_ms - self.prefill_time_ms > self.ttft_ms or unfinished == 0:
            result, self.pending = self.pending, []
            return result
        return []

    def cancel(self, request_id):
        self.pending[:] = [(rid, item) for rid, item in self.pending if rid != request_id]
