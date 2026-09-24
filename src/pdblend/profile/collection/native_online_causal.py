"""Observe real controller callbacks, with no prediction or hardware actions.

This opt-in adapter must replace a *fresh* Forecaster before Router listeners
are installed. Unlike offline shadow mappings it records actual invocation
order. Its receipts establish replayability, not model or online qualification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import time

from pdblend.online.observations import backlog_snapshot
from pdblend.planner.forecast import Forecast,Forecaster,InFlightWork
from .native_timing_audit import need,finite
from .native_timing_plan import binding,digest,read_bound

SCHEMA='pdblend-native-causal-forecaster/v1'
OPTIONS=dict(short_s=30.,long_s=120.,window_s=120.,default_input=512.,default_output=128.,bin_s=60.,epoch_s=1800.)
RECORD_FIELDS=('request_id','input_tokens','max_tokens','tokens_so_far','first_token_s','path','pool_id')
CONTROL_FIELDS=('roles','freqs','_first_traffic_s','_last_plan_change_s','_down_candidate_key','_down_votes',
                'min_plan_hold_s','down_plan_votes','home_margin','freeze','hold_initial','dynamic_m_floor',
                '_m_floor','_risk_windows','_quiet_windows','_last_pressure')


def plain(value):
    return json.loads(json.dumps(value,allow_nan=False))


def forecast_value(value):
    if value is None:return None
    row=deepcopy(value)
    row['backlog']=tuple(InFlightWork(**v) for v in row.get('backlog',[]))
    for key in ('inputs','outputs'):row[key]=tuple(row.get(key,()))
    row['length_pairs']=tuple(tuple(p) for p in row.get('length_pairs',()))
    return Forecast(**row)


def router_view(router):
    # Only counters already owned by Router; finished_s and future outcome
    # metadata are deliberately absent. Cancellation reservations remain active.
    active={r.request_id:r for rows in router.active.values() for r in rows}
    records=[{key:getattr(record,key) for key in RECORD_FIELDS} for record in active.values()]
    need(all(type(r['max_tokens']) is int and type(r['tokens_so_far']) is int
             and 0<=r['tokens_so_far']<=r['max_tokens'] for r in records),
         'causal router token prefix or known budget invalid')
    return plain(records)


def records_backlog(records,*,observed_s=None):
    need(len({r['request_id'] for r in records})==len(records),'duplicate causal request records')
    need(all(type(r['input_tokens']) is int and r['input_tokens']>0
             and type(r['max_tokens']) is int and type(r['tokens_so_far']) is int
             and 0<=r['tokens_so_far']<=r['max_tokens']
             and (r['first_token_s'] is None or (finite(r['first_token_s'])
                  and (observed_s is None or r['first_token_s']<=observed_s))) for r in records),
         'causal backlog token prefix or first-token receipt is invalid/future')
    return backlog_snapshot(SimpleNamespace(active={'observed':[SimpleNamespace(**r) for r in records]}))


def controller_view(controller):
    value={key:deepcopy(getattr(controller,key)) for key in CONTROL_FIELDS}
    value['plan_now']=asdict(controller.plan_now) if controller.plan_now is not None else None
    value['initial_plan']=asdict(controller.initial_plan) if controller.initial_plan is not None else None
    # This is an observation, not a replay of shield/gating decisions. Missing
    # internals stay explicitly missing until an online controller audit exists.
    value['shield_state']=None
    value['missing']=['shield_internal_state','actual_clock_read_completion_times','planner_candidate_actions']
    return plain(value)


class CausalForecaster:
    """Drop-in fresh Forecaster observer; no public engine/router changes."""
    def __init__(self,*,initial,prior_ref,router,identity,options=None,clock=time.time,controller=None,sink=None):
        options=dict(OPTIONS,**(options or {}))
        need(set(options)==set(OPTIONS) and all(finite(v) and v>0 for v in options.values()),
             'explicit positive forecaster options required')
        initial_value=plain(asdict(initial)) if initial is not None else None
        if prior_ref is not None:
            prior=read_bound(prior_ref)
            need(prior.get('forecast')==initial_value,'causal prior binding differs from actual initial forecast')
        else:
            need(initial is None,'nonempty bootstrap requires a raw prior binding')
        self._clock,self._router,self._controller,self._sink=clock,router,controller,sink
        self._base=Forecaster(initial=forecast_value(initial_value),**options)
        self._events=[];self._last=None
        from pdblend.planner import forecast as forecast_module
        from pdblend.online import observations as observation_module
        self._header=dict(schema=SCHEMA,identity=deepcopy(identity),options=options,initial=initial_value,
            prior_ref=deepcopy(prior_ref),created_s=clock(),source={
                'forecaster':binding(forecast_module.__file__),'backlog_snapshot':binding(observation_module.__file__),
                'observer':binding(__file__)},
            actual_callback_mapping=True,prior_selection_provenance_qualified=False,
            online_energy_qualified=False,online_policy_qualified=False,formal_eligible=False)

    def __getattr__(self,name):return getattr(self._base,name)

    def _time(self,now):
        observed=self._clock();at=observed if now is None else now
        need(finite(observed) and finite(at) and self._header['created_s']<=at<=observed
             and (self._last is None or observed>=self._last),'causal callback time is future or reordered')
        return at,observed

    def _record(self,kind,at,observed,**value):
        row=plain(dict(sequence=len(self._events),kind=kind,at_s=at,observed_s=observed,**value))
        self._events.append(row);self._last=observed
        if self._sink is not None:self._sink(deepcopy(row))

    def arrive_request(self,input_tokens,max_tokens,*,request_id):
        return self.arrive(input_tokens,request_id=request_id,known_max_tokens=max_tokens)

    def arrive(self,input_tokens,now=None,*,request_id=None,known_max_tokens=None):
        at,seen=self._time(now)
        self._base.arrive(input_tokens,at,request_id=request_id)
        self._record('arrive',at,seen,input_tokens=input_tokens,request_id=request_id,known_max_tokens=known_max_tokens)

    def finish_request(self,output_tokens,*,request_id,input_tokens):
        return self.finish(output_tokens,request_id=request_id,input_tokens=input_tokens)

    def finish(self,output_tokens,now=None,*,request_id=None,input_tokens=None):
        at,seen=self._time(now)
        self._base.finish(output_tokens,at,request_id=request_id,input_tokens=input_tokens)
        self._record('finish',at,seen,output_tokens=output_tokens,request_id=request_id,input_tokens=input_tokens)

    def set_backlog(self,items):
        at,seen=self._time(None);items=tuple(items);records=router_view(self._router)
        need(items==records_backlog(records,observed_s=seen),'backlog differs from live Router prefix')
        self._base.set_backlog(items)
        self._record('backlog',at,seen,records=records,backlog=[asdict(r) for r in items])

    def forecast(self,now=None):
        at,seen=self._time(now);result=self._base.forecast(at)
        context=controller_view(self._controller()) if self._controller is not None else None
        self._record('forecast',at,seen,forecast=asdict(result),controller=context)
        return result

    def receipt(self):
        events=deepcopy(self._events)
        return plain(dict(self._header,events=events,event_digest=digest(events),complete=True,
                          completeness_scope='events_observed_by_this_adapter_only_not_external_request_cohort'))


def replay_causal_forecasts(receipt):
    """Replay exact callback prefixes; never invent absent native/clock signals."""
    need(receipt.get('schema')==SCHEMA and receipt.get('complete') is True
         and receipt.get('formal_eligible') is False and receipt.get('online_policy_qualified') is False
         and digest(receipt['events'])==receipt['event_digest'],'causal receipt schema/content differs')
    from pdblend.planner import forecast as forecast_module
    from pdblend.online import observations as observation_module
    for key,module in [('forecaster',forecast_module),('backlog_snapshot',observation_module)]:
        need(binding(module.__file__)['sha256']==receipt['source'][key]['sha256'],'causal semantics source differs')
    need(binding(__file__)['sha256']==receipt['source']['observer']['sha256'],'causal observer source differs')
    if receipt['prior_ref'] is not None:
        need(read_bound(receipt['prior_ref']).get('forecast')==receipt['initial'],'causal prior raw differs')
    else:need(receipt['initial'] is None,'unbound causal bootstrap')
    options=receipt['options'];need(set(options)==set(OPTIONS) and all(finite(v) and v>0 for v in options.values()),
                                  'causal forecaster configuration incomplete')
    base=Forecaster(initial=forecast_value(receipt['initial']),**options);queries=[];last=receipt['created_s']
    for sequence,row in enumerate(receipt['events']):
        at,seen=row['at_s'],row['observed_s']
        need(row['sequence']==sequence and finite(at) and finite(seen) and last<=seen
             and receipt['created_s']<=at<=seen,'causal replay ordering/time differs')
        last=seen;kind=row['kind']
        if kind=='arrive':base.arrive(row['input_tokens'],at,request_id=row['request_id'])
        elif kind=='finish':base.finish(row['output_tokens'],at,request_id=row['request_id'],input_tokens=row['input_tokens'])
        elif kind=='backlog':
            items=records_backlog(row['records'],observed_s=seen)
            need(plain([asdict(r) for r in items])==row['backlog'],'causal backlog raw prefix differs')
            base.set_backlog(items)
        elif kind=='forecast':
            value=plain(asdict(base.forecast(at)))
            need(value==row['forecast'],'causal forecast does not reproduce from observed prefix')
            queries.append(dict(sequence=sequence,at_s=at,forecast=value,controller=row['controller']))
        else:raise ValueError('unknown causal callback')
    return dict(replay_passed=True,queries=queries,formal_eligible=False,online_policy_qualified=False,
        online_energy_qualified=False,prior_selection_provenance_qualified=False,
        missing=['external_request_cohort_completeness','prior_training_selection_provenance',
                 'native_state_and_clock_receipts','actual_planner_and_hysteresis_action_replay'])
