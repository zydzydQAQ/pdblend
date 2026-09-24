"""Exact natural context completion is unsupported; errors remain failures."""
import json
import threading

import pytest

from pdblend_baselines.distserve import stage_collect as module
from tests.independent_baselines.test_distserve_stage_collect_window import harness


def natural(prompt=7808):
    budget=8192-prompt
    return dict(request_id='r',input_tokens=prompt,requested_output_tokens=budget,tokens=budget,
        finished=True,stream_complete=True,done_marker=True,finish_reason='length',
        terminal_usage=dict(prompt_tokens=prompt,completion_tokens=budget))


def test_exact_context_exhaustion_is_distinct_from_failed_or_short_work():
    with pytest.raises(module.ContextWindowExhausted):module.check_decode_terminal({'r':natural()},[])
    module.check_decode_terminal({'r':dict(finished=False)},[])


@pytest.mark.parametrize('change',[
    dict(tokens=383),dict(requested_output_tokens=383),dict(input_tokens=7807),
    dict(stream_complete=False),dict(done_marker=False),dict(finish_reason='stop'),
    dict(error='TimeoutError()',error_at_s=123.),dict(terminal_usage={}),
])
def test_early_termination_timeout_or_incomplete_budget_is_not_unsupported(change):
    row=natural();row.update(change)
    with pytest.raises(RuntimeError) as error:module.check_decode_terminal({'r':row},[])
    assert not isinstance(error.value,module.ContextWindowExhausted)


def test_another_failed_request_cannot_hide_behind_natural_context_exhaustion():
    with pytest.raises(RuntimeError,match='failed'):
        module.check_decode_terminal({'natural':natural(),'bad':dict(error='socket timeout',finished=False)},[])


@pytest.mark.parametrize('bad',['none','truncated','bad_index','error','wrong_reason'])
def test_real_stream_retains_token_progress_and_exact_terminal_budget(monkeypatch,bad):
    event=dict(token_ids=[10,11],token_index=2,finished=True,finish_reason='length',
               usage=dict(prompt_tokens=8190,completion_tokens=2))
    if bad=='bad_index':event['token_index']=3
    if bad=='error':event['error']='native failure'
    if bad=='wrong_reason':event['finish_reason']='stop'
    lines=[b'data: '+json.dumps(event).encode()+b'\n']
    if bad!='truncated':lines.append(b'data: [DONE]\n')
    class Response:
        def __enter__(self):return iter(lines)
        def __exit__(self,*args):pass
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:Response())
    state=dict(tokens=0,finished=False)
    module._stream('http://cpu',dict(request_id='r',prompt=[1]*8190,max_tokens=2),state,threading.Lock())
    assert state['request_id']=='r' and state['requested_output_tokens']==2
    assert state['finished_s']>=state['submitted_s']
    if bad in ('none','wrong_reason'):
        assert state['tokens']==2 and len(state['events'])==1 and state['stream_complete']
    if bad=='none':
        with pytest.raises(module.ContextWindowExhausted):module.check_decode_terminal({'r':state},[])
    else:
        with pytest.raises(RuntimeError) as error:module.check_decode_terminal({'r':state},[])
        assert not isinstance(error.value,module.ContextWindowExhausted)


def test_context_exhaustion_keeps_raw_progress_and_later_windows_still_run(harness,tmp_path,monkeypatch):
    spec,meter,point,state,clock=harness
    original_sleep=clock.sleep
    exhausted=[False]
    def sleep(seconds):
        original_sleep(seconds)
        if seconds==5 and not exhausted[0]:
            for row in state['states'].values():row.update(natural(1024))
            state['active']=False;exhausted[0]=True
    monkeypatch.setattr(clock,'sleep',sleep)
    raw=module.window(spec,meter,point,tmp_path/'ended.json')
    assert raw['status']=='unsupported_engine' and raw['reason']=='context_window_exhausted'
    assert len(raw['client_requests'])==2 and raw['sample']['ranks']
    assert raw['end_s']-raw['start_s']==5 and raw['settle_finished_s']-raw['settle_started_s']==2
    assert module.rows_from_window(raw)==[] and not raw['cleanup_errors']
    later=module.window(spec,meter,point,tmp_path/'later.json')
    assert later['status']=='measured' and module.rows_from_window(later)


def test_decode_error_after_measurement_still_fails_whole_window(harness,tmp_path,monkeypatch):
    spec,meter,point,state,clock=harness;original=clock.sleep
    def sleep(seconds):
        original(seconds)
        if seconds==5:next(iter(state['states'].values())).update(error='TimeoutError()',error_at_s=clock.time())
    monkeypatch.setattr(clock,'sleep',sleep)
    raw=module.window(spec,meter,point,tmp_path/'error.json')
    assert raw['status']=='failed' and 'context_window_exhausted'!=raw.get('reason')
    assert module.rows_from_window(raw)==[]


def test_owned_cleanup_done_is_recorded_separately_but_transport_timeout_is_never_hidden(monkeypatch):
    event=dict(token_ids=[10],token_index=1,finished=False,finish_reason=None,
               usage=dict(prompt_tokens=8190,completion_tokens=1))
    class Response:
        def __enter__(self):return iter([b'data: '+json.dumps(event).encode()+b'\n',b'data: [DONE]\n'])
        def __exit__(self,*args):pass
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:Response())
    state=dict(tokens=0,finished=False,cleanup_cancel_requested_s=1.)
    payload=dict(request_id='r',prompt=[1]*8190,max_tokens=2)
    module._stream('http://cpu',payload,state,threading.Lock())
    assert state['cleanup_stream_closed'] and not state['stream_complete'] and not state.get('error')
    def timeout(*args,**kwargs):raise TimeoutError('real transport timeout after cancellation')
    monkeypatch.setattr(module,'urlopen',timeout)
    failed=dict(tokens=0,finished=False,cleanup_cancel_requested_s=1.,cleanup_cancel_receipt=dict(acknowledged=True))
    module._stream('http://cpu',payload,failed,threading.Lock())
    assert 'TimeoutError' in failed['error'] and not failed['stream_complete']
