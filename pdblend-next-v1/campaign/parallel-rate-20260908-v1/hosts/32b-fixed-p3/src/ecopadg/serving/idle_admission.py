"""A genuinely idle first admission promises the covered clock observed now."""
import asyncio
import math
import time
from dataclasses import replace
from .state import ExpiredPlan
from .backend import ClockWriteUncertain
from .completion_policy import recovery_budget, recovery_snapshot


def ledger_idle(instance):
    return not (instance.requests or instance.running or instance.waiting
        or instance.reserved_kv_tokens or instance.reserved_transfer_bytes
        or instance.kv_allocations or instance.transfer_allocations)


async def current_clock_first_plan(controller, plan, request, *, completion_recovery=False):
    # Called only inside the existing action lock, before reserve/forward.
    if (controller.config.get('observed_idle_admission_frequency_v2') is not True
            or controller.config.get('measured_frequency_write_guard_v1') is not True):
        return plan
    if not plan.feasible or len(plan.routes)!=1:
        return plan
    route=plan.routes[0]
    if route.prefill_id!=route.decode_id:
        return plan
    snapshot=controller.state.snapshot
    instance=next((i for i in snapshot.instances if i.instance_id==route.decode_id),None)
    # Existing work owns its original phase and prefix promises unchanged.
    if instance is not None and not ledger_idle(instance):
        return plan
    if not controller.action_lock.locked():
        raise RuntimeError('first-clock observation requires the admission action lock')
    clocks=controller.backend.clocks
    if clocks is None or clocks.write_guard is None or clocks.state_lock is not controller.state.lock:
        raise RuntimeError('first-clock observation requires guarded clock and state ownership')
    async with clocks.lock, controller.state.lock:
        # Executor cancellation does not establish a known physical state.
        # Never queue observations behind an unresolved command under locks.
        if (clocks.pending_physical_commands or clocks.physical_command_uncertainty):
            clocks.clock_event(dict(kind='idle_admission_prior_physical_uncertainty',
                at_s=time.time(),safely_replannable=False,
                pending_sequences=list(clocks.pending_physical_commands),
                prior_uncertainty=list(clocks.physical_command_uncertainty),
                requires_owned_close=True,measurement_confirmation=False))
            raise ClockWriteUncertain('idle observation blocked by unresolved physical command; owned close required')
        snapshot=controller.state.snapshot
        instance=next((i for i in snapshot.instances if i.instance_id==route.decode_id),None)
        event=dict(kind='idle_first_admission_clock',at_s=time.time(),request_id=request.request_id,
            instance_id=route.decode_id,snapshot_version=snapshot.version,
            original_frequencies=[dict(instance_id=a.instance_id,frequency_mhz=a.frequency_mhz)
                                  for a in plan.frequencies],observations=[],physical_write_attempts=[])
        def reject(reason):
            event.update(allowed=False,error=reason,finished_s=time.time(),safely_replannable=True)
            clocks.clock_event(event)
            raise ExpiredPlan('idle first admission observed-clock eligibility: '+reason)
        now=time.time()
        if snapshot.version!=plan.snapshot_version or now>plan.expires_s:
            reject('original planning snapshot expired before observation')
        if instance is None:
            reject('planned instance disappeared')
        if not ledger_idle(instance):
            # A snapshot change normally catches this; never replace a promise.
            return plan
        config=controller.backend.instances.get(instance.instance_id,{})
        raw=controller.backend.last.get(instance.instance_id,{})
        if (instance.role!='mixed' or instance.mode!='continuous' or not instance.accepting
            or not instance.admit_prefill or set(config.get('gpus',()))!=set(instance.gpus)
            or config.get('tp')!=instance.tp or raw.get('active')!=0
            or not 0<=now-instance.timestamp_s<=controller.planner.telemetry_ttl_s):
            reject('idle topology, native state, or telemetry is untrusted')
        event.update(gpus=list(instance.gpus),native_active=raw['active'],ledger_requests=0,
            retained_commands={g:clocks.applied.get(g) for g in instance.gpus})
        loop=asyncio.get_running_loop()
        for gpu in instance.gpus:
            if gpu not in clocks.gpus:
                reject('GPU outside actual clock ownership')
            started=time.time()
            try:
                observed=await loop.run_in_executor(clocks.pool,clocks.hardware.current_freq,gpu)
            except Exception as exc:
                reject('actual clock read failed: '+repr(exc))
            if (isinstance(observed,bool) or not isinstance(observed,(int,float))
                or not math.isfinite(observed) or observed<=0):
                reject('invalid actual clock observation')
            event['observations'].append(dict(gpu=gpu,observed_mhz=observed,
                started_s=started,finished_s=time.time()))
        now=time.time()
        ttl=controller.planner.telemetry_ttl_s
        if not 0<=now-instance.timestamp_s<=ttl or now>plan.expires_s:
            reject('telemetry or original plan expired during clock observation')
        if any(not 0<=now-r['started_s']<=ttl for r in event['observations']):
            reject('full TP observations are not jointly fresh')
        frequencies=controller.planner.profiles.frequencies(instance.role,instance.tp)
        targets=[f for f in frequencies if all(abs(r['observed_mhz']-f)<=15 for r in event['observations'])]
        legacy_idle=False
        if not targets:
            # Preserve the original bounded deferred-wakeup path only when
            # EVERY member actually reports the hardware idle-only throttle
            # reason. A 405MHz observation is never recorded as the target.
            observed=[r['observed_mhz'] for r in event['observations']]
            if max(observed)-min(observed)>15 or not hasattr(clocks.hardware,'clock_idle'):
                reject('full TP actual frequencies disagree or lack a measured profile frequency')
            event['idle_throttle_observations']=[]
            for gpu in instance.gpus:
                try:idle=await loop.run_in_executor(clocks.pool,clocks.hardware.clock_idle,gpu)
                except Exception as exc:reject('hardware idle-throttle read failed: '+repr(exc))
                event['idle_throttle_observations'].append(dict(gpu=gpu,idle_only=idle is True,at_s=time.time()))
                if idle is not True:reject('unprofiled clock has no complete TP idle-only throttle evidence')
            targets=[a.frequency_mhz for a in plan.frequencies if a.instance_id==instance.instance_id]
            if len(targets)!=1 or targets[0] not in frequencies:
                reject('original deferred-wakeup target lacks a measured frequency')
            legacy_idle=True
        now=time.time()
        if (not 0<=now-instance.timestamp_s<=ttl or now>plan.expires_s
            or any(not 0<=now-r['started_s']<=ttl for r in event['observations'])
            or controller.backend.last.get(instance.instance_id,{}).get('active')!=0):
            reject('idle evidence expired or native work appeared before candidate selection')
        # Re-run the original functional candidate search: input/context/KV,
        # full batch, measured switching cost, TTFT/TPOT and existing deadlines.
        # The observed frequency is never substituted into a precomputed plan.
        candidate_snapshot,candidate_request=snapshot,request
        if completion_recovery:
            if (not controller.evaluation_v3 or request.hard_deadline_s is None
                    or now>=request.hard_deadline_s):
                reject('completion recovery lacks a remaining original hard deadline')
            # Exactly the existing completion policy, only for planner inputs.
            # The caller still reserves and scores the original request budget.
            candidate_snapshot=recovery_snapshot(snapshot,now)
            candidate_request=recovery_budget(request,now)
        candidates=controller.planner.candidates(candidate_snapshot,candidate_request,now)
        candidates=[p for p in candidates if len(p.routes)==1
            and p.routes[0].prefill_id==instance.instance_id==p.routes[0].decode_id
            and len(p.frequencies)==1 and p.frequencies[0].instance_id==instance.instance_id
            and p.frequencies[0].frequency_mhz in targets]
        if not candidates:
            reject('current TP frequency has no original measured feasible first-request candidate')
        chosen=min(candidates,key=lambda p:(p.routes[0].incremental_j,p.routes[0].predicted_ttft_s,
                                           p.frequencies[0].frequency_mhz))
        if time.time()>chosen.expires_s or not 0<=time.time()-instance.timestamp_s<=ttl:
            reject('candidate expired before committing observed-clock selection')
        event.update(allowed=True,completion_recovery=completion_recovery,chosen_frequency_mhz=chosen.frequencies[0].frequency_mhz,
            selection_mode='original_measured_target_with_deferred_wakeup' if legacy_idle else 'current_observed_covered_frequency',
            observed_at_target=not legacy_idle,requires_deferred_wakeup_verification=legacy_idle,
            predicted_ttft_s=chosen.routes[0].predicted_ttft_s,
            predicted_tpot_s=chosen.routes[0].predicted_tpot_s,finished_s=time.time(),
            request_input_tokens=request.input_tokens,request_output_limit=request.output_limit,
            request_arrival_s=request.arrival_s,request_ttft_s=request.ttft_s,request_tpot_s=request.tpot_s)
        clocks.clock_event(event)
        # The caller still reserves normally, writes any needed retained-clock
        # intent, and re-observes all TP members before backend.confirm/forward.
        return replace(chosen,expires_s=min(chosen.expires_s,plan.expires_s,
                request.hard_deadline_s if request.hard_deadline_s is not None else float('inf')),
            reason=('completion recovery; original SLO retained: ' if completion_recovery else '')+('idle first admission preserves original measured target with all-TP idle-only throttle and deferred wakeup verification; '
            if legacy_idle else 'idle first admission at freshly observed covered TP clock; ')+chosen.reason)
