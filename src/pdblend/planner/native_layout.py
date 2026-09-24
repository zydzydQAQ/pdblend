"""Explicit PD energy revision; the default PoolPlanner remains untouched.

For canonical 32B minM4/slots4, the complete layout domain is M4. Feasibility
retains the timing equations; physical fleet watts are queried once after the
timing checks. Pure-role power and static watts are never queried or added.
"""
from __future__ import annotations
import json
from dataclasses import replace
from pdblend.planner.pool import PoolPlanner,Plan,mdc_wait
from pdblend.profile.collection.native_timing_audit import need,finite
from pdblend.profile.collection.native_frequency_domain import require_same_domain

REVISION='pdblend_layout_energy_v1'


class NativeLayoutPlanner(PoolPlanner):
    def __init__(self,timing_model,config,energy_model,*,workload_scope):
        need(timing_model.model=='Qwen2.5-32B-Instruct' and timing_model.tp==2 and timing_model.pp==1
             and config.slots==config.min_m_instances==4 and config.max_num_seqs==32
             and config.freqs==energy_model.frequencies==tuple(timing_model.freqs)
             and not config.pressure_controls and not config.capacity_floors,
             'missing_profile: layout revision requires canonical 32B TP2/M4/native32 policy')
        # Every canonical count choice is M4 in this exact topology. No P/D,
        # parked watts or wake predictor participates in this component.
        timing_model.require_runtime_components('capacity','clock_transition')
        if 'frequency_domain' in energy_model.candidate:
            require_same_domain(energy_model.candidate,timing_model.calibration_identity)
        self.model=timing_model;self.cfg=replace(config)
        self._peak_cache={};self._quantile_cache=(None,([],[]));self._branch_cache={}
        self._split_cache={};self._role_cache=None;self._floor=None
        self.energy_model=energy_model;self.workload_scope=dict(workload_scope);self.unsupported=[]
        self.revision=energy_model.revision

    def _decode_batch(self,rate,out_mean,ctx,servers,f,dilution=0.):
        b=1.
        for _ in range(32):
            active=max(1.,b)
            # A physical capacity exclusion is different from an unmeasured
            # executable branch; the latter blocks complete candidate replay.
            if active>self.cfg.max_num_seqs:return b
            need(self.model.decode_supported(active,ctx,f),
                 'missing_profile: executable decode timing branch outside actual hull')
            tpot=self.model.step_seconds(active,ctx,f)/max(1.-dilution,1e-6)
            nxt=max(rate*out_mean*tpot/servers,1e-6)
            if nxt>self.cfg.max_num_seqs:return nxt
            if abs(nxt-b)<1e-3*max(b,1.):return nxt
            b=nxt
        return b

    def mixed_timing(self,fc,n,f):
        """Canonical mixed-pool timing equations with no energy dependency."""
        if n<=0:return None
        in_mean,in_p95=fc.input_mean,fc.input_p95
        s_p=self.model.prefill_marginal_seconds(int(in_mean),f)
        prefill_rate=self._prefill_rate(fc.rate_rps,fc,in_mean)
        decode_rate=self._decode_rate(fc.rate_rps,fc)
        u_p=prefill_rate*s_p/n
        if u_p>=self.cfg.rho_max:return None
        ctx=max(in_mean+fc.output_mean/2.,fc.occupied_kv_tokens/max(len(fc.backlog),1))
        b=max(self._decode_batch(decode_rate,fc.output_mean,ctx,n,f,dilution=u_p),len(fc.backlog)/n)
        active_batch=max(1.,b) if (self.model.bounded_coverage or self.model.decode_power_overrides) else b
        if not self.model.decode_supported(active_batch,ctx,f):return None
        if b>min(self.cfg.peak_batch_cap,self.cfg.max_num_seqs) or max(b*(in_mean+fc.output_mean),fc.occupied_kv_tokens/n)>self.model.kv_capacity_tokens*.9:return None
        if decode_rate*fc.output_mean/n>self.cfg.rho_decode*(1.-u_p)*self._peak_decode_tps(ctx,f):return None
        step=self.model.step_seconds(active_batch,ctx,f);tpot=step/(1.-u_p)
        wait=(mdc_wait(prefill_rate,s_p,n) or 0.)+self._backlog_wait(fc,in_mean,n,f)
        ttft=wait+self.model.prefill_seconds(int(in_p95),f)+tpot
        return dict(ttft_s=ttft,tpot_s=tpot,batch=b,busy=u_p,
            tpot_miss=self._stall_miss(prefill_rate/n,fc,step,f),pending_prefill_tokens=fc.pending_prefill_tokens,
            remaining_decode_tokens=fc.remaining_decode_tokens,occupied_kv_tokens=fc.occupied_kv_tokens)

    def evaluate(self,counts,f_P,f_D,f_M,tau,fc,strict=True):
        nonzero={k:v for k,v in counts.items() if v}
        if nonzero!={'M':4} or tau!=0 or fc.backlog:
            self.unsupported.append(dict(counts=dict(counts),frequency_mhz=f_M,reason='unsupported_layout_or_backlog'))
            return None
        try:
            timing=self.mixed_timing(fc,4,f_M)
            if timing is None:return None
            slo=self.cfg.slo
            if strict and (timing['ttft_s']>slo.ttft_s*slo.safety or timing['tpot_s']>slo.tpot_s*slo.safety
                           or timing['tpot_miss']>1.-self.cfg.tail_target):return None
            power=self.energy_model.predict_layout_mean_w(model_id=self.model.model,tp=2,pp=1,
                counts=nonzero,rate_rps=fc.rate_rps,frequency_mhz=f_M,**self.workload_scope)
            need(finite(power) and power>0,'layout energy prediction invalid')
        except ValueError as exc:
            if 'missing_profile:' not in str(exc) and 'outside measured coverage' not in str(exc):raise
            self.unsupported.append(dict(counts=dict(counts),frequency_mhz=f_M,reason=str(exc)))
            return None
        detail=dict(M=dict(timing,power_w=power,energy_model='native_whole_layout_request_cycle'),
            revision=self.revision,energy_scope='whole_eight_gpu_wall_clock_mean_w',
            physical_gpu_count=8,pure_role_power_queried=False,parked_power_added_again=False,
            forecast=dict(mixed_floor=4,mixed_output_mean=fc.output_mean))
        return Plan(dict(counts),f_P,f_D,f_M,tau,power,timing['ttft_s'],timing['tpot_s'],detail,tp=2,pp=1,
                    profile_key=json.dumps(self.model.profile_key,sort_keys=True,separators=(',',':')))

    def candidates(self,fc):
        self.unsupported=[]
        return super().candidates(fc)

    def fallback(self,fc):
        high=max(self.cfg.freqs)
        value=self.evaluate({'M':4},high,high,high,0,fc,strict=False)
        need(value is not None,'missing_profile: whole-layout fallback is outside qualified timing/energy scope')
        value.detail['fallback']=True
        return value
