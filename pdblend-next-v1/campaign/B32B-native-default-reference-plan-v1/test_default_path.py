"""CPU only: actual default scheduler composition with explicit fake allocator leaves.

These checks prove a scheduling construction, never logits, KV bytes, or CUDA.
The default method and its result/budget classes are compiled from the actual
SHA-bound container export. Only GPU/block-allocation leaves are stand-ins.
"""
import ast
import hashlib
import importlib.util
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, List, Set, Tuple, Sequence as GenericSequence

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'B32B-temporal-engine-review-v1/actual-sources'
SCHEDULER_SHA = 'ed93fc25d69d0bbbce123c48598ff06d123339f1ed161e92abdc38373a22835e'


def actual_functions():
    path = SOURCE / 'core/scheduler.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SCHEDULER_SHA
    tree = ast.parse(path.read_text())
    names = {'SchedulingBudget', 'ScheduledSequenceGroup', 'SchedulerOutputs',
             'SchedulerRunningOutputs', 'SchedulerSwappedInOutputs', 'SchedulerPrefillOutputs'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    nodes += [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_schedule_default']
    ns = dict(dataclass=dataclass, field=field, List=List, Set=Set, Tuple=Tuple,
              GenericSequence=GenericSequence, SequenceGroup=object,
              LoRARequest=object, PromptAdapterRequest=object)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def actual_runtime():
    p = SOURCE / 'pdblend_runtime.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '4cd252ecc481cc81059708661802807f3f51ebcda785c31fc432b74272046347'
    spec = importlib.util.spec_from_file_location('readonly_actual_runtime', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSequence:
    lora_request = prompt_adapter_request = None
    lora_int_id = 0
    def __init__(self, rid, n):
        self.request_id, self.prompt_length, self.outputs = rid, n, 0
    def get_max_num_running_seqs(self): return 1
    def uses_prompt_embeds(self): return False


class SchedulingConstruction:
    def __init__(self):
        self.ns = actual_functions()
        self.scheduler_config = SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=32,
                                                policy='fcfs', chunked_prefill_enabled=False)
        self.running, self.waiting, self.swapped = deque(), deque(), deque()
        self.lora_enabled = False

    def _schedule_prefills(self, budget, loras, enable_chunking):
        assert enable_chunking is False
        out = self.ns['SchedulerPrefillOutputs'].create_empty()
        while self.waiting:
            seq = self.waiting.popleft()
            assert budget.can_schedule(num_new_tokens=seq.prompt_length, num_new_seqs=1)
            budget.add_num_seqs(seq.request_id, 1)
            budget.add_num_batched_tokens(seq.request_id, seq.prompt_length)
            out.seq_groups.append(self.ns['ScheduledSequenceGroup'](seq, seq.prompt_length))
        return out

    def _schedule_running(self, budget, loras, enable_chunking):
        assert enable_chunking is False
        out = self.ns['SchedulerRunningOutputs'].create_empty()
        while self.running:
            seq = self.running.popleft()
            budget.add_num_batched_tokens(seq.request_id, 1)
            out.decode_seq_groups.append(self.ns['ScheduledSequenceGroup'](seq, 1))
            out.decode_seq_groups_list.append(seq)
        return out

    def _schedule_swapped(self, budget, loras):
        assert not self.swapped
        return self.ns['SchedulerSwappedInOutputs'].create_empty()

    def _schedule_default(self): return self.ns['_schedule_default'](self)

    def step(self, *, hidden=False):
        if hidden:
            result = actual_runtime().schedule(self, dict(role='mixed', mode='temporal',
                        admit_prefill=False, admit_decode=True))
        else:
            result = self._schedule_default()
        signature = (result.num_prefill_groups,
                     len(result.scheduled_seq_groups) - result.num_prefill_groups,
                     result.num_batched_tokens,
                     tuple(s.seq_group.request_id for s in result.scheduled_seq_groups))
        for row in result.scheduled_seq_groups: row.seq_group.outputs += 1
        # This is a stand-in for normal end-of-request release, not a KV proof.
        self.running = deque(s for s in self.running if s.outputs < 64)
        return signature


def replay(*, hidden=False, admission_after=5):
    s = SchedulingConstruction()
    first, second = FakeSequence('first', 96), FakeSequence('second', 192)
    s.waiting.append(first)
    rows = []
    while len(rows) < 130:
        if first.outputs == (2 if hidden else admission_after) and second not in s.waiting and second.outputs == 0:
            s.waiting.append(second)
        rows.append(s.step(hidden=hidden and 2 <= first.outputs < admission_after))
        if first.outputs == second.outputs == 64:
            return rows
    raise AssertionError('finite complete work not reached')


EXPECTED = ([(1, 0, 96, ('first',))] + [(0, 1, 1, ('first',))] * 4
            + [(1, 0, 192, ('second',))] + [(0, 2, 2, ('first', 'second'))] * 59
            + [(0, 1, 1, ('second',))] * 4)


def test_native_deferred_admission_matches_exact_69_step_trace():
    assert replay() == EXPECTED


def test_actual_runtime_hiding_and_native_deferred_admission_same_construction():
    assert replay(hidden=True) == replay() == EXPECTED


def test_one_step_earlier_admission_fails_same_trajectory_requirement():
    assert replay(admission_after=4) != EXPECTED


def test_default_prefill_excludes_decode_without_hiding_running():
    s = SchedulingConstruction()
    first, second = FakeSequence('first', 96), FakeSequence('second', 192)
    first.outputs = 5
    s.running.append(first)
    running = s.running
    s.waiting.append(second)
    assert s.step() == (1, 0, 192, ('second',))
    assert s.running[0] is first and first.outputs == 5
    assert running[0] is first  # native method retained the actual seq object
