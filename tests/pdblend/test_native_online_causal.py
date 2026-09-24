from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from pdblend.planner.forecast import Forecast,Forecaster
from pdblend.online.observations import backlog_snapshot
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_timing_plan import digest
from pdblend.profile.collection.native_online_causal import CausalForecaster,replay_causal_forecasts,plain


def observe(tmp_path,*,finish=True):
    now=[100.];router=SimpleNamespace(active={'m0':[]})
    prior=Forecast(2.,0.,10.,20.,30.,0,inputs=(10,20),outputs=(20,40),length_pairs=((10,20),(20,40)))
    ref=write_new(tmp_path/'prior.json',dict(forecast=plain(asdict(prior))))
    observed=CausalForecaster(initial=prior,prior_ref=ref,router=router,identity=dict(model_id='Qwen2.5-32B-Instruct'),clock=lambda:now[0])
    base=Forecaster(initial=prior)
    assert asdict(observed.forecast())==asdict(base.forecast(now[0]))
    now[0]=101.;observed.arrive_request(10,40,request_id='r');base.arrive(10,now[0],request_id='r')
    record=SimpleNamespace(request_id='r',input_tokens=10,max_tokens=40,tokens_so_far=2,first_token_s=101.1,path='M',pool_id='pool')
    router.active['m0']=[record];now[0]=102.
    observed.set_backlog(backlog_snapshot(router));base.set_backlog(backlog_snapshot(router))
    assert asdict(observed.forecast())==asdict(base.forecast(now[0]))
    prefix=deepcopy(observed.receipt())
    if finish:
        now[0]=105.;observed.finish_request(4,request_id='r',input_tokens=10);base.finish(4,now[0],request_id='r',input_tokens=10)
        router.active['m0']=[];observed.set_backlog(backlog_snapshot(router));base.set_backlog(())
    now[0]=110.;assert asdict(observed.forecast())==asdict(base.forecast(now[0]))
    return observed,prefix,record


def test_actual_callback_prefix_replays_known_prior_and_budget(tmp_path):
    observed,prefix,_=observe(tmp_path);result=replay_causal_forecasts(observed.receipt())
    assert result['replay_passed'] and len(result['queries'])==3
    backlog=result['queries'][1]['forecast']['backlog'][0]
    assert backlog['remaining_output_tokens']==38 and backlog['kv_tokens']==12
    assert not result['online_policy_qualified'] and not result['prior_selection_provenance_qualified']
    assert replay_causal_forecasts(prefix)['queries']==result['queries'][:2]


def test_future_finish_presence_and_metadata_do_not_change_past_query(tmp_path):
    (tmp_path/'a').mkdir();(tmp_path/'b').mkdir()
    a,prefix_a,record=observe(tmp_path/'a',finish=True);b,prefix_b,_=observe(tmp_path/'b',finish=False)
    assert replay_causal_forecasts(prefix_a)['queries']==replay_causal_forecasts(prefix_b)['queries']
    before=a.receipt();record.finished_s=10000.;record.final_completion_tokens=999
    assert a.receipt()==before
    assert 'finished_s' not in prefix_a['events'][2]['records'][0]


@pytest.mark.parametrize('damage',['future_time','forecast_value','backlog_tokens','future_first_token','source','prior'])
def test_replay_rejects_tampered_prefix_and_identity(tmp_path,damage):
    observed,_,_=observe(tmp_path);receipt=observed.receipt()
    if damage=='future_time':receipt['events'][0]['at_s']=1000.
    elif damage=='forecast_value':receipt['events'][-1]['forecast']['rate_rps']+=.1
    elif damage=='backlog_tokens':receipt['events'][2]['records'][0]['tokens_so_far']=8
    elif damage=='future_first_token':receipt['events'][2]['records'][0]['first_token_s']=10000.
    elif damage=='source':receipt['source']['forecaster']['sha256']='other'
    else:receipt['initial']['rate_rps']=77.
    receipt['event_digest']=digest(receipt['events'])
    with pytest.raises(ValueError):replay_causal_forecasts(receipt)


def test_bootstrap_cannot_be_guessed_and_router_snapshot_cannot_be_substituted(tmp_path):
    router=SimpleNamespace(active={});now=[1.]
    with pytest.raises(ValueError,match='raw prior'):
        CausalForecaster(initial=Forecast(1,0,1,1,1,0),prior_ref=None,router=router,identity={},clock=lambda:now[0])
    observed=CausalForecaster(initial=None,prior_ref=None,router=router,identity={},clock=lambda:now[0])
    from pdblend.planner.forecast import InFlightWork
    with pytest.raises(ValueError,match='live Router'):
        observed.set_backlog([InFlightWork('ghost',10,5)])
    with pytest.raises(ValueError,match='future'):
        observed.forecast(2.)
    assert not observed.receipt()['events']
