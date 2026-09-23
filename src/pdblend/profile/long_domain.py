"""Union of a frozen short-context model and exact-batch long observations.

Unmeasured gaps are deliberately preserved. No nearest batch/frequency,
cross-regime interpolation, or context extrapolation is available.
"""
from copy import deepcopy
from pathlib import Path

from .long_context_followup import KIND,predict


class LongDomainUnion:
    def __init__(self,base,candidate):
        if (candidate.get('kind')!=KIND or candidate.get('holdout_used') is not False or
            candidate.get('batch_interpolation_qualified') is not False or
            tuple(candidate.get(k) for k in ('system','model_id','tp','pp')) !=
                (base.system,Path(base.model).name,base.tp,base.pp)):
            raise ValueError('long candidate system/model/topology/domain identity differs')
        if not base.bounded_coverage or not base.decode_overrides:
            raise ValueError('short model must have explicit bounded domains')
        expected={f'{f}/{b}' for f in base.freqs for b in candidate['exact_batches']}
        if set(candidate['nodes'])!=expected:raise ValueError('long candidate exact frequency/batch nodes incomplete')
        self.base,self.candidate=base,deepcopy(candidate)

    def __getattr__(self,name):return getattr(self.base,name)

    def _long_supported(self,batch,ctx,f):
        if f not in self.base.freqs:return False
        try:
            predict(self.candidate,'step_seconds',f,batch,ctx)
            predict(self.candidate,'power_w',f,batch,ctx)
            return True
        except (ValueError,TypeError):return False

    def decode_supported(self,batch,ctx,f):
        return self.base.decode_supported(batch,ctx,f) or self._long_supported(batch,ctx,f)

    def decode_power_supported(self,batch,ctx,f):
        return (self.base.decode_supported(batch,ctx,f) and self.base.decode_power_supported(batch,ctx,f)) or self._long_supported(batch,ctx,f)

    def step_seconds(self,batch,ctx,f):
        if self.base.decode_supported(batch,ctx,f):return self.base.step_seconds(batch,ctx,f)
        return predict(self.candidate,'step_seconds',f,batch,ctx)

    def decode_power_w(self,batch,f,*,ctx=None):
        if ctx is None:raise ValueError('explicit decode context required')
        if self.base.decode_supported(batch,ctx,f):return self.base.decode_power_w(batch,f,ctx=ctx)
        return predict(self.candidate,'power_w',f,batch,ctx)

    def token_energy_j(self,batch,ctx,f):
        return self.step_seconds(batch,ctx,f)*self.decode_power_w(batch,f,ctx=ctx)/batch
