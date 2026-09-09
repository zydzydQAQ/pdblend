# -*- coding: utf-8 -*-
"""router:准入门控(FIFO/开合)与 SLO 判定。"""
from __future__ import annotations

from ecopadg.router import (
    AdmissionGate, RequestRecord, admit_request, classify_slo,
    estimate_prompt_tokens,
)
from ecopadg.types import SloSpec
from tests.conftest import FakeClock


SLO = SloSpec(ttft_s=5.0, tpot_s=0.1)


def test_estimate_prompt_tokens_body_and_cjk():
    assert estimate_prompt_tokens("hello world", {"prompt_len": 42}) == 42
    assert estimate_prompt_tokens("你好世界", {}) >= 4
    assert estimate_prompt_tokens("hello world", {}) >= 2


def test_classify_slo_boundaries():
    assert classify_slo(ttft_s=4.9, tpot_s=0.09, slo=SLO)
    assert not classify_slo(ttft_s=5.0, tpot_s=0.09, slo=SLO)  # 严格小于
    assert not classify_slo(ttft_s=4.9, tpot_s=0.10, slo=SLO)
    assert not classify_slo(ttft_s=5.1, tpot_s=0.05, slo=SLO)


def test_request_record_defaults():
    r = RequestRecord(rid=0, arrival_s=1.0, prompt_len=100, output_len=20)
    assert not r.success
    assert r.release_s is None
    assert not r.slo_ok(SLO)


def test_gate_fifo_and_closed():
    clock = FakeClock(0.0)
    g = AdmissionGate(clock=clock)
    g.submit(RequestRecord(rid=0, arrival_s=0.0, prompt_len=10, output_len=10))
    g.submit(RequestRecord(rid=1, arrival_s=0.1, prompt_len=10, output_len=10))
    g.set_open(False, now=0.2)
    assert g.release_due(now=0.3) == []
    assert g.pending() == 2
    g.set_open(True, now=0.4)
    out = g.release_due(now=0.5)
    assert [r.rid for r in out] == [0, 1]  # FIFO
    assert g.pending() == 0


def test_gate_release_limit():
    g = AdmissionGate(clock=FakeClock(0.0))
    for i in range(5):
        g.submit(RequestRecord(rid=i, arrival_s=0.0, prompt_len=10, output_len=10))
    out = g.release_due(now=1.0, limit=2)
    assert [r.rid for r in out] == [0, 1]
    assert g.pending() == 3


def test_gate_log():
    g = AdmissionGate(clock=FakeClock(0.0))
    g.set_open(False, now=1.0)
    g.set_open(True, now=2.0)
    assert g.log == [(1.0, False), (2.0, True)]


def test_admit_request_wires_gate_and_scheduler(synth_opmodel, fake_clock):
    """C0.1:gate.submit 与 sched.submit 必须同时发生,释放条数对齐。"""
    from ecopadg.online_scheduler import PaDGWindowScheduler
    from ecopadg.types import SystemConfig
    gate = AdmissionGate()
    sched = PaDGWindowScheduler(
        synth_opmodel, SystemConfig(model="syn"), clock=fake_clock)
    rec = RequestRecord(rid=7, arrival_s=0.0, prompt_len=100, output_len=10)
    admit_request(gate, sched, rec)
    assert gate.pending() == 1
    assert len(sched.buffer) == 1
    d = sched.step(now=0.5)
    assert 7 in d.released
    rel = gate.release_due(now=0.5, limit=len(d.released))
    assert [r.rid for r in rel] == [7]
