"""CPU workload design check, explicitly not native execution or GPU evidence."""
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from .controller import EcoServeController


class _Replay(EcoServeController):
    """Run the original scale_once thresholds over modeled request progress.

    Only the CPU workload-design check uses modeled progress. Native execution
    uses the unmodified EcoServeRuntime and validates actual scheduler events.
    """
    def __init__(self, config, journal, clock, decode_factor):
        super().__init__(config, None, journal)
        self.clock = clock
        self.work = []
        self.prefill_free = {iid: 0.0 for iid in self.specs}
        self.decode_step_s = self.profile.points[16] / 1000 * decode_factor
        self.candidates = []

    async def refresh(self, **kwargs):
        now = self.clock[0]
        for row in self.work:
            if not row['first_observed'] and row['first'] <= now:
                self.ttft_history.append((row['first'], row['first'] - row['arrival']))
                row['first_observed'] = True
            request = row['request']
            request.num_iterations = max(0, min(row['output'], int((now-row['first'])/self.decode_step_s)+1)) if now >= row['first'] else 0
            if request.num_iterations:
                request.ttft = (row['first'] - row['arrival']) * 1000
        for iid, member in self.members.items():
            live = [row for row in self.work if row['instance'] == iid and row['finish'] > now]
            member.requests = deque(row['request'] for row in live)
            member.waiting_queue = [row['request'].request_id for row in live if row['first'] > now]
            # This is an explicit CPU capacity assumption, not a measured KV
            # inventory and never a hardware qualification receipt.
            member.free_blocks = max(0, (32 * 8192 - sum(row['input'] + row['output'] for row in live)) // 16)
            self.states[iid] = dict(accepting=True, running=[row['request'].request_id for row in live],
                                   waiting=[], kv_allocations={})

    async def add_member(self, identifier, *, trigger):
        before = [list(group.identifiers) for group in self.groups]
        after = [list(group) for group in before]
        self._layout_add(after, identifier)
        self._set_layout(after)
        self.candidates.append(dict(at_s=self.clock[0], operation='add', trigger=trigger,
                                    before=before, after=after, split=len(after)>len(before)))
        return True

    async def remove_member(self, identifier, *, trigger):
        before = [list(group.identifiers) for group in self.groups]
        after = self._layout_remove(identifier)
        self._set_layout(after)
        self.candidates.append(dict(at_s=self.clock[0], operation='remove', trigger=trigger,
                                    before=before, after=after, merge=len(after)<len(before)))
        return True

    async def admit(self, index, arrival, prompt, count):
        await self.refresh()
        group = min(self.groups, key=lambda g: (sum(len(m.requests) for m in g.instance_states), g.identifiers))
        request_id = f'cpu-model-{index}'
        selected = group.schedule(SimpleNamespace(request_id=request_id, prompt_len=len(prompt)))
        iid = group.identifiers[selected]
        request = next(row for row in self.members[iid].requests if row.request_id == request_id)
        first = max(arrival, self.prefill_free[iid]) + self.profile.predict_ms(len(prompt)) / 1000
        self.prefill_free[iid] = first
        self.work.append(dict(instance=iid, request=request, input=len(prompt), output=count, arrival=arrival,
                             first=first, finish=first+count*self.decode_step_s, first_observed=False))


async def replay(config, rows, duration=300.0, decode_factor=1.0):
    """Keep real five-second decision ticks and sixty-second history semantics."""
    clock, journal = [0.0], []
    def emit(kind, **fields):
        journal.append(dict(kind=kind, **fields))
    fake_time = SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
    with patch('pdblend_baselines.ecoserve.controller.time', fake_time):
        controller = _Replay(config, emit, clock, decode_factor)
        # The author macro takes a clock callback, so simulated time stays local
        # to this CPU instance and never changes its decision formula.
        original_group = controller._group
        def group(ids):
            result = original_group(ids)
            result.now_ms = lambda: clock[0] * 1000
            return result
        controller._group = group
        controller._set_layout([list(g.identifiers) for g in controller.groups])
        index = 0
        for tick in range(int(duration * 4) + 1):
            clock[0] = tick / 4
            while index < len(rows) and rows[index][0] <= clock[0]:
                await controller.admit(index, *rows[index])
                index += 1
            await controller.refresh()
            if clock[0] > 0 and clock[0] % controller.period == 0:
                await controller.scale_once()
    split = any(row.get('split') and row['trigger'] == 'mean_ttft' for row in controller.candidates)
    merge = any(row.get('merge') and row['trigger'] == 'saved_tpot' for row in controller.candidates)
    return dict(status='cpu_candidate_replay_passed' if split and merge else 'cpu_candidate_inconclusive',
                hardware_executed=False, native_evidence=False, original_policy_thresholds=True,
                period_s=controller.period, history_window_s=controller.history_window,
                decode_step_assumption_s=controller.decode_step_s,
                assumptions=['CSV 16-token prefill forward latency is a decode-step proxy for sensitivity analysis.',
                             'Per-engine prefill is serial; modeled KV capacity is 32 x 8192 tokens.',
                             'Native decoding, buffering and member drain timing must be verified on GPU.'],
                candidates=controller.candidates, split_candidate=split, merge_candidate=merge,
                observations=journal)
