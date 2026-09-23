"""Actual PP worker CPU phases, separate from GPU model CUDA timing.

Receive waits include upstream producer delay; send waits can include downstream
backpressure. These inclusive samples must not be fed to a stage simulator as
pure service latency without a separately validated queue/wire decomposition.
"""
import threading
import time
from types import MethodType


class ServiceCalls:
    def __init__(self,identity):self.identity=dict(identity);self.local=threading.local();self.pending=[]

    def around(self,function,*args,**kwargs):
        previous=getattr(self.local,'row',None)
        if previous is not None:raise RuntimeError('nested physical worker execute_model call')
        begin=time.perf_counter();row=dict(self.identity,started_s=time.time(),phases=[],succeeded=False,is_model_call=False)
        self.local.row=row
        try:
            value=function(*args,**kwargs);row['succeeded']=True;return value
        except BaseException as exc:row['error']=repr(exc);raise
        finally:
            row.update(finished_s=time.time(),elapsed_s=time.perf_counter()-begin,
                recv_includes_upstream_wait=True,send_includes_downstream_backpressure=True,
                simulator_provider_eligible=False)
            self.pending.append(row);self.local.row=previous

    def phase(self,name,function,*args,**kwargs):
        row=getattr(self.local,'row',None)
        if row is None:return function(*args,**kwargs)
        start=time.perf_counter();phase=dict(phase=name,started_s=time.time(),succeeded=False)
        try:
            value=function(*args,**kwargs);phase['succeeded']=True;return value
        except BaseException as exc:phase['error']=repr(exc);raise
        finally:
            phase.update(finished_s=time.time(),elapsed_s=time.perf_counter()-start);row['phases'].append(phase)

    def mark_model(self,metadata):
        row=getattr(self.local,'row',None)
        if row is not None:row.update(metadata,is_model_call=True)

    def collect(self):
        rows=self.pending;self.pending=[];return rows


def install(worker,pp_group,identity):
    timer=ServiceCalls(identity);worker._distserve_service_timer=timer
    original=worker.execute_model
    def execute(self,*args,**kwargs):return timer.around(original,*args,**kwargs)
    worker.execute_model=MethodType(execute,worker)
    for instance,name,phase in ((worker,'prepare_input','prepare_input'),(worker,'execute_worker','cache_ops'),
        (pp_group,'recv_tensor_dict','pp_recv'),(pp_group,'send_tensor_dict','pp_send')):
        original_phase=getattr(instance,name)
        def measured(self,*args,_original=original_phase,_phase=phase,**kwargs):
            return timer.phase(_phase,_original,*args,**kwargs)
        setattr(instance,name,MethodType(measured,instance))
    return timer
