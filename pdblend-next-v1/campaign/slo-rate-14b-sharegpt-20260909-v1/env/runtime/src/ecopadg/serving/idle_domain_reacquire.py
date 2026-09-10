"""Owned idle reacquisition into a measured service domain, before planning.

No unprofiled performance or transition cost is predicted. The real transaction
runs under the existing admission/clock/state locks and consumes wall time and
energy before the unchanged planner sees the original absolute request budget.
"""
import asyncio
import math
import time
from dataclasses import replace
from .backend import ClockWriteUncertain
from .completion_policy import engine_residual
from .state import ExpiredPlan


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _idle(instance):
    return not (instance.requests or instance.running or instance.waiting
        or instance.reserved_kv_tokens or instance.reserved_transfer_bytes
        or instance.kv_allocations or instance.transfer_allocations)


def native_idle(raw, instance, now, ttl):
    """The native owner ACK is the existing all-rank control barrier, not a GPU count.

    acknowledged_generations enumerates schedulers (PP), not TP ranks. Require
    every reported owner ACK, the current generation and actual transport drain.
    The bound engine configuration supplies the exact TP membership.
    """
    if not isinstance(raw, dict):
        raise ClockWriteUncertain('idle reacquisition native observation is not an object')
    residual = engine_residual(raw, now, ttl)
    generation = raw.get('generation')
    acks = raw.get('acknowledged_generations')
    if (residual or raw.get('id') != instance.instance_id
        or any(type(raw.get(k)) is not int or raw[k] != 0 for k in
               ('active', 'running', 'waiting', 'transfer_buffered_tensors',
                'transfer_inflight_receives', 'transfer_inflight_sends', 'transfer_send_failed'))
        or any(not isinstance(raw.get(k), dict) or raw[k] for k in
               ('kv_allocations', 'transfer_allocations'))
        or type(raw.get('acknowledged_generation')) is not int
        or type(raw.get('observed_control_generation')) is not int
        or 'scheduler_budget_pending' not in raw
        or type(generation) is not int or generation != instance.generation
        or raw.get('acknowledged_generation') != generation
        or raw.get('observed_control_generation') != generation
        or not isinstance(acks, list) or not acks
        or any(type(g) is not int or g != generation for g in acks)
        or raw.get('role') != 'mixed' or raw.get('mode') != 'continuous'
        or raw.get('accepting') is not True or raw.get('admit_prefill') is not True
        or raw.get('admit_decode') is not True
        or raw.get('diagnostic_recompute') is not False
        or raw.get('diagnostic_transport') is not False
        or raw.get('scheduler_budget_pending') is not None
        or raw.get('transfer_inflight_sends_observed') is not True
        or raw.get('transfer_send_counters_observed') is not True
        or raw.get('transfer_send_healthy') is not True
        or raw.get('transfer_inflight_sends') != 0
        or raw.get('transfer_send_failed') != 0
        or any(type(raw.get(k)) is not int or raw[k] < 0 for k in
               ('transfer_send_started', 'transfer_send_completed'))
        or raw.get('transfer_send_started') != raw.get('transfer_send_completed')
        or any(type(raw.get(k)) is not int or raw[k] < 0 for k in
               ('free_kv_tokens', 'free_transfer_bytes'))):
        raise ClockWriteUncertain('idle reacquisition native ownership or drain unconfirmed: '+repr(residual))
    return {k: raw[k] for k in (
        'id', 'timestamp', 'generation', 'acknowledged_generation',
        'acknowledged_generations', 'observed_control_generation', 'active',
        'running', 'waiting', 'kv_allocations', 'transfer_allocations',
        'transfer_buffered_tensors', 'transfer_inflight_receives',
        'transfer_inflight_sends', 'transfer_observed_s', 'transport_healthy',
        'transfer_send_started', 'transfer_send_completed', 'transfer_send_failed',
        'transfer_send_counters_observed', 'transfer_send_healthy',
        'role', 'mode', 'accepting', 'admit_prefill', 'admit_decode')}


async def reacquire(controller, instance, plan, request, observations, frequencies):
    """Return a confirmed snapshot/instance/target/read-set, or None off-domain.

    The caller already owns action -> clock -> state. Do not call ClockOwner.set,
    which reacquires those locks and may skip a retained but physically lost write.
    """
    clocks, backend = controller.backend.clocks, controller.backend
    maximum = controller.config.get('max_service_frequency_mhz', 2520)
    values = [r['observed_mhz'] for r in observations]
    if (not values or not all(v > maximum + 15 for v in values)
        or max(values)-min(values) > 15):
        return None
    event = dict(kind='idle_domain_reacquisition', request_id=request.request_id,
        instance_id=instance.instance_id, gpus=list(instance.gpus), target_mhz=maximum,
        started_s=time.time(), initial_observations=observations,
        retained_commands={g:clocks.applied.get(g) for g in instance.gpus},
        original_plan_expires_s=plan.expires_s, original_request_deadline_s=request.hard_deadline_s,
        deadline_extended=False, unprofiled_transition_predicted=False,
        actual_transaction_in_measurement=True, confirmed=False, physical_write_attempts=[])
    attempted = False
    confirmed = False
    current_only = False
    def ownership():
        if (not controller.action_lock.locked() or not clocks.lock.locked()
            or not controller.state.lock.locked() or clocks.state_lock is not controller.state.lock
            or clocks.write_guard is None):
            raise ClockWriteUncertain('idle reacquisition requires action/clock/state ownership')
        if clocks.pending_physical_commands or clocks.physical_command_uncertainty:
            raise ClockWriteUncertain('idle reacquisition cannot clear unresolved physical commands')
        current = next((i for i in controller.state.snapshot.instances if i.instance_id==instance.instance_id), None)
        config = backend.instances.get(instance.instance_id, {})
        if (current is None or not _idle(current) or not _idle(instance)
            or current.generation != instance.generation
            or current.role != 'mixed' or current.mode != 'continuous'
            or not current.accepting or not current.admit_prefill
            or len(set(instance.gpus)) != instance.tp
            or tuple(current.gpus) != tuple(instance.gpus)
            or set(config.get('gpus', ())) != set(instance.gpus)
            or config.get('tp') != instance.tp
            or not set(instance.gpus) <= set(clocks.gpus)
            or any(instance.instance_id in (r.prefill_id,r.decode_id)
                   for r in controller.state.reservations.values())):
            raise ClockWriteUncertain('idle reacquisition topology, ledger or reservation is not empty and owned')
        return current
    try:
        ownership()
        if (type(maximum) is not int or maximum not in frequencies
            or maximum != clocks.max_service_frequency_mhz
            or maximum != backend.max_service_frequency_mhz
            or maximum != controller.max_service_frequency_mhz):
            raise ClockWriteUncertain('idle reacquisition target lacks the exact qualified profile/domain')
        # Idle-domain recovery has no admitted/prefill promise. Its physical
        # bound is independent of the unchanged active DVFS settling bound.
        idle_budget = controller.config.get('idle_domain_reacquire_timeout_s')
        if idle_budget is None:
            idle_budget = clocks.settle_timeout_s
        elif (not _finite(idle_budget) or not clocks.settle_timeout_s <= idle_budget <= 2.):
            raise ClockWriteUncertain('idle reacquisition budget must be finite, non-bool, and within active bound..2s')
        event['idle_domain_reacquire_timeout_s'] = idle_budget
        event['original_active_settle_timeout_s'] = clocks.settle_timeout_s
        ttl = controller.planner.telemetry_ttl_s
        if any(not 0 <= time.time()-r['started_s'] <= ttl for r in observations):
            raise ExpiredPlan('idle reacquisition initial full-TP observations expired before any write')
        native_started = time.time()
        # GET is bounded by the original native backend timeout (.5 s).
        raw = await backend.json(instance.instance_id, '/runtime')
        event['native_read_started_s'] = native_started
        event['native_read_finished_s'] = time.time()
        event['native_idle'] = native_idle(raw, instance, time.time(), ttl)
        ownership()
        if time.time() > min(plan.expires_s, request.hard_deadline_s or float('inf')):
            raise ExpiredPlan('idle reacquisition plan expired before any physical write')
        # Publish only fields just observed by the native owner. This is a real
        # state change, so all prior plan versions are invalidated.
        instance = replace(instance, timestamp_s=raw['timestamp'],
            free_kv_tokens=raw['free_kv_tokens'], free_transfer_bytes=raw['free_transfer_bytes'])
        backend.last[instance.instance_id] = raw
        controller.state.snapshot = replace(controller.state.snapshot,
            version=controller.state.snapshot.version+1,
            instances=tuple(instance if i.instance_id==instance.instance_id else i
                            for i in controller.state.snapshot.instances))
        clocks.transaction_writes = []
        for gpu in instance.gpus:
            clocks.epochs[gpu] += 1
        loop = asyncio.get_running_loop()
        for gpu in instance.gpus:
            ownership()
            native_idle(raw, instance, time.time(), ttl)
            # Preserve the original physical write guard for the FULL TP group
            # immediately before each member, including a cached equal command.
            clocks.require_covered_write(instance.gpus, maximum, 'idle domain reacquisition')
            attempted = True
            await clocks.physical_write(gpu, maximum, loop)
        event['physical_write_attempts'] = list(clocks.transaction_writes)
        last_finished = max(w['finished_s'] for w in clocks.transaction_writes)
        clocks.last_write_finished_s = last_finished
        physical_deadline_s = last_finished+idle_budget
        if request.hard_deadline_s is not None:
            physical_deadline_s = min(physical_deadline_s, request.hard_deadline_s)
        deadline = time.monotonic()+physical_deadline_s-time.time()
        event['physical_confirmation_deadline_s'] = physical_deadline_s
        event['original_request_deadline_cap_s'] = request.hard_deadline_s
        event['last_command_finished_s'] = last_finished
        event['settle_timeout_s'] = clocks.settle_timeout_s
        reads = []
        def read(gpu):
            started = time.time()
            value = clocks.hardware.current_freq(gpu)
            return dict(gpu=gpu, observed_mhz=value, started_s=started,
                        finished_s=time.time(), finished_monotonic_s=time.monotonic())
        while time.monotonic() <= deadline:
            group = []
            for gpu in instance.gpus:
                ownership()
                if time.monotonic() > deadline:
                    break
                row = await loop.run_in_executor(clocks.pool, read, gpu)
                if not _finite(row['observed_mhz']) or row['observed_mhz'] <= 0:
                    raise ClockWriteUncertain('idle reacquisition invalid actual SM observation')
                group.append(row)
            reads.extend(group)
            if (len(group)==instance.tp
                and all(r['finished_monotonic_s'] <= deadline
                        and abs(r['observed_mhz']-maximum)<=15 for r in group)):
                observations = group
                confirmed = True
                break
            await asyncio.sleep(min(.01, max(0., deadline-time.monotonic())))
        # Preserve the opted-in P9 boundary semantics: one fresh final read
        # can establish the CURRENT state after a delayed journal/coroutine.
        # It cannot establish when the hardware settled; no deadline is reset.
        if (not confirmed and controller.config.get('clock_failure_fresh_confirmation_v1') is True
            and getattr(clocks, 'failure_fresh_confirmation', False)
            and (request.hard_deadline_s is None or time.time()<request.hard_deadline_s)):
            ownership()
            final_group = []
            for gpu in instance.gpus:
                row = await loop.run_in_executor(clocks.pool, read, gpu)
                if not _finite(row['observed_mhz']) or row['observed_mhz'] <= 0:
                    raise ClockWriteUncertain('idle reacquisition invalid final SM observation')
                final_group.append(row)
            ownership()
            reads.extend(final_group)
            event['final_confirmation'] = dict(
                origin='idle_domain_reacquisition', observations=final_group,
                decision_s=time.time(), original_settle_timeout_s=clocks.settle_timeout_s,
                idle_domain_reacquire_timeout_s=idle_budget,
                elapsed_after_original_deadline_s=max(0., time.monotonic()-deadline),
                confirms_current_state_only=True, settled_within_original_bound_proven=False,
                deadline_extended=False, additional_physical_writes=0)
            if all(abs(r['observed_mhz']-maximum)<=15 for r in final_group):
                observations = final_group
                confirmed = True
                current_only = True
        event['confirmation_observations'] = reads
        if not confirmed:
            raise ClockWriteUncertain('idle reacquisition not confirmed within its bounded idle recovery/request deadline')
        ownership()
        for gpu in instance.gpus:
            clocks.applied[gpu] = maximum
            clocks.deferred.pop(gpu, None)
            clocks.fallbacks.pop(gpu, None)
        backend.frequency[instance.instance_id] = maximum
        backend.parked.discard(instance.instance_id)
        backend.idle_since.pop(instance.instance_id, None)
        instance = replace(instance, frequency_mhz=maximum, parked=False)
        controller.state.snapshot = replace(controller.state.snapshot,
            version=controller.state.snapshot.version+1,
            instances=tuple(instance if i.instance_id==instance.instance_id else i
                            for i in controller.state.snapshot.instances))
        event.update(confirmed=True, physical_state_unknown=False,
            confirmed_before_original_settle_deadline=all(
                r['finished_s']<=last_finished+clocks.settle_timeout_s for r in observations),
            confirmed_within_idle_recovery_bound=not current_only,
            exceeded_original_active_settle_bound=any(
                r['finished_s']>last_finished+clocks.settle_timeout_s for r in observations),
            confirmed_current_only=current_only, finished_s=time.time(),
            original_plan_expired=time.time()>plan.expires_s,
            request_deadline_unchanged=True, replan_after_actual_transaction=True)
        clocks.clock_event(event)
        return controller.state.snapshot, instance, maximum, observations
    except BaseException as exc:
        # Once positively confirmed, an old planning expiry is a safe replan,
        # not physical uncertainty. Actual cancellation/partial writes are not.
        if attempted and not confirmed:
            clocks.physical_command_uncertainty.append(dict(
                kind='idle_domain_reacquisition_uncertain', gpus=list(instance.gpus),
                target_mhz=maximum, at_s=time.time(), error=repr(exc),
                requires_owned_close=True, physical_state_unknown=True))
        event.update(error=repr(exc), finished_s=time.time(), confirmed=confirmed,
            physical_write_attempts=list(clocks.transaction_writes) if attempted else [],
            physical_state_unknown=attempted and not confirmed,
            safely_replannable=isinstance(exc, ExpiredPlan) and not attempted)
        clocks.clock_event(event)
        if isinstance(exc, ExpiredPlan) and not attempted:
            raise
        raise ClockWriteUncertain('idle domain reacquisition did not establish safe admission: '+repr(exc)) from exc
