"""Bind stage identity per generator advance; never leak across SimPy yields.

The official Worker generator bodies and scheduling decisions stay unchanged.
"""
from ._sim.estimators.time_estimator import PIPE_STAGE


class StageIterator:
    def __init__(self,generator,stage):self.generator=generator;self.stage=stage
    def __iter__(self):return self
    def _advance(self,method,*args):
        token=PIPE_STAGE.set(self.stage)
        try:return method(*args)
        finally:PIPE_STAGE.reset(token)
    def __next__(self):return self._advance(next,self.generator)
    def send(self,value):return self._advance(self.generator.send,value)
    def throw(self,*args):return self._advance(self.generator.throw,*args)
    def close(self):return self._advance(self.generator.close)


def bind_worker_stages(cluster):
    for worker in cluster.get_all_workers():
        for name in ('do_prefill','do_decode'):
            original=getattr(worker,name)
            def bound(*args,_original=original,_stage=worker.pipe_rank,**kwargs):
                return StageIterator(_original(*args,**kwargs),_stage)
            setattr(worker,name,bound)
