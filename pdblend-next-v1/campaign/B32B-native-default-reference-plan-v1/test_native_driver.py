import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import pytest
from native_driver import run_reference
from test_default_path import SchedulingConstruction, FakeSequence

ROOT = Path(__file__).resolve().parent
REQUESTS = json.loads((ROOT.parent / 'B32B-temporal-solo-pair-observation-v1/spec.json').read_text())['requests'][:4]


class FakeEngine:
    """API stand-in: actual default policy composition; synthetic tokens only."""
    def __init__(self, fail_at=None, mismatch_shape=False):
        self.s = SchedulingConstruction()
        self.s.schedule = self.schedule
        self.scheduler = [self.s]
        self.seq, self.tokens, self.calls, self.fail_at = {}, {}, 0, fail_at
        self.mismatch_shape = mismatch_shape
    def schedule(self):
        result = self.s._schedule_default()
        return [SimpleNamespace(request_id=x.seq_group.request_id) for x in result.scheduled_seq_groups], result, False
    def add_request(self, rid, prompt, params):
        assert params == dict(temperature=0, top_p=1, max_tokens=64, ignore_eos=True, seed=0)
        self.seq[rid] = FakeSequence(rid, len(prompt['prompt_token_ids']))
        self.tokens[rid] = []
        self.s.waiting.append(self.seq[rid])
    def step(self):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError('injected model step failed')
        _, result, _ = self.s.schedule()
        if self.mismatch_shape: result.num_batched_tokens += 1
        out = []
        for row in result.scheduled_seq_groups:
            seq = row.seq_group
            seq.outputs += 1
            self.tokens[seq.request_id].append(1000 + seq.outputs)
            out.append(SimpleNamespace(request_id=seq.request_id, finished=seq.outputs == 64,
                outputs=[SimpleNamespace(token_ids=list(self.tokens[seq.request_id]))]))
        self.s.running = deque(s for s in self.s.running if s.outputs < 64)
        return out
    def has_unfinished_requests(self): return bool(self.s.running or self.s.waiting or self.s.swapped)


def test_actual_driver_uses_engine_outputs_and_four_full64_with_69_pair_steps():
    e = FakeEngine(); original = e.s.schedule; events = []
    result = run_reference(e, REQUESTS, lambda **k: k, events.append, lambda: None)
    assert result['complete'] and e.calls == 197 and e.s.schedule == original
    assert all(v == list(range(1001, 1065)) for v in result['token_ids_by_request_uuid'].values())
    assert len([x for x in events if x['kind'] == 'output']) == 256
    assert [x['kind'] for x in events].count('add_intent') == 4


def test_actual_driver_deadline_failure_restores_record_only_wrapper():
    e = FakeEngine(); original = e.s.schedule
    def guard():
        if e.calls == 3: raise TimeoutError('actual phase deadline')
    with pytest.raises(TimeoutError): run_reference(e, REQUESTS, lambda **k: k, lambda x: None, guard)
    assert e.s.schedule == original and e.calls == 3 and e.has_unfinished_requests()
    # Outer native/abort ownership is still necessary, not fabricated here.


def test_actual_driver_step_failure_preserves_error_for_parent_cleanup():
    e = FakeEngine(fail_at=2); original = e.s.schedule
    with pytest.raises(RuntimeError, match='injected'):
        run_reference(e, REQUESTS, lambda **k: k, lambda x: None, lambda: None)
    assert e.s.schedule == original and e.has_unfinished_requests()


def test_actual_driver_does_not_accept_chunked_configuration():
    e = FakeEngine(); e.s.scheduler_config.chunked_prefill_enabled = True
    with pytest.raises(RuntimeError, match='native default'):
        run_reference(e, REQUESTS, lambda **k: k, lambda x: None, lambda: None)
    assert e.calls == 0
