"""Opt-in local model-runner CUDA events, harvested only at a safe boundary.

These events measure actual local stage execution, including its TP collectives.
They do not measure PP network transfer or replace full request latency records.
"""
import time
from types import MethodType


class TimedCalls:
    def __init__(self,event_factory):self.event_factory=event_factory;self.pending=[]

    def around(self,function,metadata,*args,**kwargs):
        start=self.event_factory();end=self.event_factory();row=dict(metadata,started_s=time.time(),succeeded=False)
        start.record()
        try:
            result=function(*args,**kwargs);row['succeeded']=True;return result
        except BaseException as exc:row['error']=repr(exc);raise
        finally:
            end.record();row['returned_s']=time.time();self.pending.append((row,start,end))

    def collect(self):
        # Caller must have stopped all physical rank execution. No event wait is
        # inserted between model calls or into the live inference critical path.
        pending=self.pending;self.pending=[];rows=[]
        for row,start,end in pending:
            end.synchronize();rows.append(dict(row,gpu_elapsed_ms=start.elapsed_time(end)))
        return rows


def input_metadata(model_input):
    def seq(name):
        value=getattr(model_input,name,None)
        return None if value is None else list(value)
    tokens=getattr(model_input,'input_tokens',None)
    return dict(seq_lens=seq('seq_lens'),query_lens=seq('query_lens'),
        is_prompt=getattr(model_input,'is_prompt',None),
        input_tokens=None if tokens is None else int(tokens.numel()),
        virtual_engine=getattr(model_input,'virtual_engine',None))


def install(worker):
    import torch
    from vllm.distributed import get_pp_group,get_tp_group
    worker=getattr(worker,'worker',worker)
    timer=TimedCalls(lambda:torch.cuda.Event(enable_timing=True));worker._distserve_stage_timer=timer
    runner=worker.model_runner;original=runner.execute_model
    identity=dict(rank=worker.rank,pp_rank=get_pp_group().rank_in_group,tp_rank=get_tp_group().rank_in_group)
    from .pipeline_service_profile import install as install_service
    service=install_service(worker,get_pp_group(),identity)
    def measured(self,*args,**kwargs):
        model_input=args[0] if args else kwargs['model_input']
        metadata=input_metadata(model_input);service.mark_model(metadata)
        return service.phase('model_runner',timer.around,original,dict(identity,**metadata),*args,**kwargs)
    runner.execute_model=MethodType(measured,runner)


def collect(worker):
    worker=getattr(worker,'worker',worker)
    timer=getattr(worker,'_distserve_stage_timer',None)
    if timer is None:raise RuntimeError('stage profiling was not enabled before engine execution')
    service=getattr(worker,'_distserve_service_timer',None)
    return dict(rank=worker.rank,rows=timer.collect(),service_calls=service.collect() if service else [],
        service_scope='actual worker CPU prepare/cache/PP recv/model/PP send and total; recv includes upstream wait, send includes backpressure; not pure wire or simulator service',
        scope='CUDA events around local model_runner.execute_model; TP collectives included, PP wire/queue time excluded')


def reduce_rank_timings(ranks,*,tp,pp,is_prompt,steps):
    import math
    import statistics
    if len(ranks)!=tp*pp or {r['rank'] for r in ranks}!=set(range(tp*pp)):
        raise ValueError('all unique physical stage ranks required')
    values=[]
    for rank in sorted(ranks,key=lambda r:r['rank']):
        leader_id=rank['rank']//tp*tp;leader=next(r for r in ranks if r['rank']==leader_id)
        if len(rank['rows'])!=len(leader['rows']):raise ValueError('TP rank call counts differ')
        rows=[]
        for row,reference in zip(rank['rows'],leader['rows']):
            if (row.get('input_tokens')!=reference.get('input_tokens') or
                    row.get('virtual_engine')!=reference.get('virtual_engine') or
                    row.get('is_prompt') is not None and row['is_prompt'] is not reference.get('is_prompt')):
                raise ValueError('TP broadcast shape evidence differs from its leader call')
            if reference.get('is_prompt') is is_prompt:rows.append(row)
        if len(rows)!=steps or any(not r['succeeded'] or not math.isfinite(r['gpu_elapsed_ms']) or
                r['gpu_elapsed_ms']<=0 or r['rank']!=rank['rank'] for r in rows):
            raise ValueError('native phase count or timing sample is invalid')
        delays=[r['gpu_elapsed_ms'] for r in rows]
        if len({(r['pp_rank'],r['tp_rank']) for r in rows})!=1:raise ValueError('physical rank ownership changed')
        values.append(dict(rank=rank['rank'],pp_rank=rows[0]['pp_rank'],tp_rank=rows[0]['tp_rank'],
            samples=len(delays),maximum_gpu_ms=max(delays),median_gpu_ms=statistics.median(delays),
            phase_source_rank=leader_id))
    if {(r['pp_rank'],r['tp_rank']) for r in values}!={(p,t) for p in range(pp) for t in range(tp)}:
        raise ValueError('pipeline/tensor ownership coverage incomplete')
    return values
