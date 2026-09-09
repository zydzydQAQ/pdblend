"""First admission only: observe and retain a covered clock before reserving work.

No writes, no fallback, no retry classification. Each successful first admission
consumes the rule; later empty and existing-decode decisions keep their policy.
"""
import asyncio
import copy
import math
import time
from .backend import ClockWriteUncertain


def _empty(instance, raw, now, ttl):
    stamp=raw.get('timestamp')
    return (instance.role=='mixed' and instance.accepting and not instance.parked
        and not instance.requests and not instance.running and not instance.waiting
        and not instance.reserved_kv_tokens and not instance.reserved_transfer_bytes
        and not instance.kv_allocations and not instance.transfer_allocations
        and type(stamp) in (int,float) and math.isfinite(stamp) and 0<=now-stamp<=ttl
        and all(type(raw.get(k)) is int and raw[k]==0 for k in ('active','running','waiting'))
        and raw.get('kv_allocations')=={} and raw.get('transfer_allocations')=={}
        and not raw.get('error') and not raw.get('runtime_error'))


async def admission_planner(controller, snapshot):
    original=controller.planner
    if controller.config.get('observed_first_admission_frequency_v1') is not True:
        return original
    if (controller.config.get('measured_frequency_write_guard_v1') is not True
            or controller.strategy not in ('pdblend-joint','pdblend-dynamic')):
        raise ValueError('startup observation requires the guarded joint PDB runtime')
    used=getattr(controller,'startup_admission_used',set())
    aborted=getattr(controller,'startup_admission_aborted',set())
    cold=[i for i in snapshot.instances if i.role=='mixed' and i.instance_id not in used
          and not i.requests and not i.running and not i.waiting]
    if not cold:return original
    planner=copy.copy(original)
    # Missing proof blocks this cold instance; it never falls through to a low write.
    proof={i.instance_id:None for i in cold};planner.startup_frequency_restrictions=proof
    clocks=controller.backend.clocks
    if clocks is None:return planner
    async with controller.action_lock, clocks.lock, clocks.snapshot_guard():
        # A cancelled coroutine may leave a real executor write in progress.
        # Do not enqueue a frequency read behind it while holding state locks.
        if (getattr(clocks,'pending_physical_commands',{})
                or getattr(clocks,'physical_command_uncertainty',())):
            clocks.clock_event(dict(kind='first_admission_prior_physical_uncertainty',
                at_s=time.time(),safely_replannable=False,
                pending_sequences=list(getattr(clocks,'pending_physical_commands',{})),
                prior_uncertainty=list(getattr(clocks,'physical_command_uncertainty',())),
                requires_owned_close=True,measurement_confirmation=False))
            raise ClockWriteUncertain('startup observation blocked by unresolved physical command; owned close required')
        if controller.state.snapshot.version!=snapshot.version:return planner
        for instance in cold:
            iid=instance.instance_id;members=tuple(instance.gpus)
            if iid in aborted:continue
            raw=controller.backend.last.get(iid,{})
            if not _empty(instance,raw,time.time(),original.telemetry_ttl_s):continue
            if (any(g not in clocks.gpus or g in clocks.fallbacks for g in members)
                    or any(e.get('completed') is not True for e in clocks.transaction_writes)):
                continue
            commands=[clocks.applied.get(g) for g in members]
            if (not commands or any(type(f) is not int for f in commands)
                    or len(set(commands))!=1 or commands[0]!=instance.frequency_mhz):continue
            frequency=commands[0];epochs={g:clocks.epochs[g] for g in members}
            readings=[]
            try:
                loop=asyncio.get_running_loop()
                for gpu in members:
                    value=await loop.run_in_executor(clocks.pool,clocks.hardware.current_freq,gpu)
                    idle=(await loop.run_in_executor(clocks.pool,clocks.hardware.clock_idle,gpu)
                          if hasattr(clocks.hardware,'clock_idle') else False)
                    readings.append(dict(gpu=gpu,observed_mhz=value,idle=idle,at_s=time.time()))
            except Exception:
                continue  # Read failure cannot authorize a command or recovery.
            now=time.time()
            if (all(r['idle'] is True for r in readings)
                    and all(0<=now-r['at_s']<=original.telemetry_ttl_s for r in readings)
                    and all(clocks.applied.get(g)==frequency and clocks.epochs[g]==epochs[g] for g in members)):
                # A real idle P-state is not a service frequency. Preserve the
                # existing guarded command/wakeup/deferred verification path.
                proof.pop(iid,None)
                clocks.clock_event(dict(kind='first_admission_idle_wakeup_preserved',
                    instance_id=iid,readings=readings,service_frequency_proven=False))
                continue
            if any(g in clocks.deferred for g in members):continue
            if (any(type(r['observed_mhz']) not in (int,float) or not math.isfinite(r['observed_mhz'])
                    or abs(r['observed_mhz']-frequency)>15 or not 0<=now-r['at_s']<=original.telemetry_ttl_s for r in readings)
                    or any(clocks.applied.get(g)!=frequency or clocks.epochs[g]!=epochs[g] for g in members)
                    or not _empty(instance,raw,now,original.telemetry_ttl_s)):
                continue
            item=dict(frequency_mhz=frequency,observed_s=min(r['at_s'] for r in readings),
                gpus=members,snapshot_version=snapshot.version,readings=readings,
                scope='first admission only; profile/SLO and final physical confirmation still required')
            proof[iid]=item
            clocks.clock_event(dict(kind='first_admission_frequency_observation',instance_id=iid,**item))
    return planner


def committed(controller,plan):
    if controller.config.get('observed_first_admission_frequency_v1') is not True:return
    used=getattr(controller,'startup_admission_used',set())
    used.update(iid for route in plan.routes for iid in (route.prefill_id,route.decode_id))
    controller.startup_admission_used=used


def failed(controller,plan,error):
    if (controller.config.get('observed_first_admission_frequency_v1') is not True
            or not isinstance(error,ClockWriteUncertain)):return
    aborted=getattr(controller,'startup_admission_aborted',set())
    aborted.update(a.instance_id for a in plan.frequencies)
    controller.startup_admission_aborted=aborted
