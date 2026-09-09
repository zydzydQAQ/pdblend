import asyncio
import json
import math

import pytest
from ecopadg.metrics import (bench_time_bounds, summarize_bench, clip_power_window,
                            energy_summary, slo_attainment, classify_run_validity)
from ecopadg.types import SloSpec
from benchmarks.scripts.bench_vllm import send_request


def test_arrivals_define_throughput_and_failures_stay_in_denominator():
    rows = [dict(request_id="0", arrival_s=100, finish_s=101, latency_s=1,
                 ttft_s=.2, tpot_s=.05, generated_tokens=17),
            dict(request_id="1", arrival_s=110, finish_s=112, latency_s=2,
                 ttft_s=.2, tpot_s=.05, generated_tokens=37)]
    slo = SloSpec(5, .1)
    assert bench_time_bounds(rows) == (100, 112)
    summary = summarize_bench(rows, slo, n_expected=3)
    assert summary["req_throughput"] == pytest.approx(2/12)
    assert slo_attainment(rows, slo, n_expected=3) == pytest.approx(2/3)
    assert classify_run_validity(rows, n_expected=3) == "invalid_run"
    assert slo_attainment([dict(rows[0], error="timeout")], slo) == 0
    assert slo_attainment([dict(rows[0], slo_ok=1, ttft_s=6)], slo) == 0


def test_integrates_all_cards_and_interpolates_boundaries():
    samples = [(0, [10, 20]), (2, [30, 40]), (4, [10, 20])]
    clipped = clip_power_window(samples, 1, 3, pad_s=0)
    assert clipped == [(1, [20, 30]), (2, [30, 40]), (3, [20, 30])]
    with pytest.raises(ValueError):
        clip_power_window(samples, -1, 3, pad_s=0)
    from ecopadg.measure.power import trapezoid_energy
    assert trapezoid_energy(clipped) == pytest.approx(120)


class Response:
    status = 200
    def __init__(self, events):
        self.events = events
        self.content = self
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    def __aiter__(self):
        return self.lines()
    async def lines(self):
        for event in self.events:
            yield ("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\n").encode()


class Session:
    def __init__(self, events):
        self.events = events
    def post(self, *args, **kwargs):
        return Response(self.events)


@pytest.mark.parametrize('status,message,code',[
    (429,{'error':{'type':'admission_rejection','code':'admission_deadline'}},'admission_deadline'),
    (500,{'error':{'type':'admission_rejection','code':'admission_deadline'}},None),
    (429,{'error':{'type':'engine_error','code':'admission_deadline'}},None),
    (429,{'error':{'type':'admission_rejection','code':'unknown'}},None),
    (503,'connection failure',None),
])
def test_unknown_server_failures_are_not_capacity_evidence(status,message,code):
    class FailedResponse(Response):
        async def text(self): return json.dumps(message)
    response=FailedResponse([]);response.status=status
    class FailedSession:
        def post(self,*args,**kwargs): return response
    result=asyncio.run(send_request(FailedSession(),'http://unused','model',[1],2))
    assert not result['success'] and result['admission_rejection']==code
    assert result['http_status']==status and result['generated_tokens']==0


def test_actual_usage_and_empty_text_token_are_measured():
    events = [dict(choices=[dict(text="")], token_ids=[100]),
              dict(choices=[dict(text="hello")], token_ids=[101],
                   usage=dict(completion_tokens=2, prompt_tokens=9)), "[DONE]"]
    result = asyncio.run(send_request(Session(events), "http://unused", "model", "x", 2))
    assert result["success"]
    assert result["generated_tokens"] == 2
    assert result["token_ids"] == [100, 101]
    assert len(result["token_itl"]) == 1
    assert result["ttft"] is not None


@pytest.mark.parametrize("events,reason", [
    ([dict(choices=[dict(text="hello")]), "[DONE]"], "missing_token_usage"),
    ([dict(choices=[dict(text="hello")])], "truncated_stream"),
    ([dict(choices=[dict(text="hello")], usage=dict(completion_tokens=1,prompt_tokens=1)), "[DONE]"], "incomplete_output")])
def test_no_planned_length_or_chunks_as_actual_work(events, reason):
    result = asyncio.run(send_request(Session(events), "http://unused", "model", "x", 2))
    assert not result["success"]
    assert reason in result["error"]


@pytest.mark.parametrize('second_index,reason',[(1,'duplicate_token_event'),(3,'duplicate_token_event'),
                                               (2,'token_stream_usage_mismatch')])
def test_lost_or_repeated_stream_tokens_do_not_pass_workload_validation(second_index,reason):
    events=[dict(choices=[dict(text='a')],token_ids=[10],token_index=1),
            dict(choices=[dict(text='b')],token_ids=[11],token_index=second_index,
                 usage=dict(completion_tokens=3,prompt_tokens=1)), '[DONE]']
    result=asyncio.run(send_request(Session(events),'http://unused','model','x',3))
    assert not result['success'] and reason in result['error']
