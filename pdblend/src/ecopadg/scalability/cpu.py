"""Real control-code scaling on explicit virtual resources, never GPU throughput.

Run ``python -m ecopadg.scalability.cpu --smoke --output /tmp/cpu-smoke``.
Formal mode requires measured profiles, transfers and transition costs. Every
cell is independent and emits raw observations plus a machine-readable summary.
"""
import argparse
import asyncio
from collections import Counter, deque
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time

from ecopadg.serving.frequency import FrequencyPlanner
from ecopadg.serving.planning_executor import PlanningExecutor
from ecopadg.serving.reconfigure import ResidentRolePlanner, search_roles
from ecopadg.serving.state import StateManager, StalePlan

from .fixtures import coverage, layout_roles, load_inputs, make_planner, make_snapshot, request
from .telemetry import (TimedLock, distribution, memory, physical_core_affinity,
                        sha256, summarize_plans)


class ReplayBackend:
    """Zero-device-I/O backend: exercise commit logic without simulated GPU time."""
    def __init__(self, state):
        self.state = state

    async def execute(self, plan):
        await self.state.apply_clocks({a.instance_id: a.frequency_mhz for a in plan.frequencies}, set())
        if plan.roles:
            actions = {a.instance_id: a for a in plan.roles}
            instances = tuple(replace(i, role=actions[i.instance_id].role,
                generation=actions[i.instance_id].expected_generation + 1)
                if i.instance_id in actions else i for i in self.state.snapshot.instances)
            await self.state.publish(instances, time.time())

    async def confirm(self, plan):
        instances = {i.instance_id: i for i in self.state.snapshot.instances}
        return (all(instances[a.instance_id].frequency_mhz == a.frequency_mhz for a in plan.frequencies)
                and all(instances[a.instance_id].role == a.role for a in plan.roles))

    async def complete(self, plan, budget):
        # These are real state methods. This completion is controller workload,
        # not an assertion about token production speed or network latency.
        await self.state.prefill_complete(budget.request_id)
        await self.state.update_budget(replace(budget, emitted=1,
                                              first_token_s=time.time(), last_token_s=time.time()))
        await self.state.release(budget.request_id, unissued=True)


async def planner_cell(planner, n, layout, active_mode, seed, args, samples):
    executor = PlanningExecutor()
    checks = []
    began = time.perf_counter()
    try:
        for index in range(args.planner_calls):
            now = time.time()
            snapshot = make_snapshot(n, layout, active_mode, seed * 100000 + index, now,
                input_tokens=args.input_tokens, output_tokens=args.output_tokens)
            pending = tuple(request(f'planner-{seed}-{index}-{j}', now,
                            args.input_tokens, args.output_tokens) for j in range(3))
            if index == 0:
                check = coverage(planner, snapshot, pending, now)
                checks.append(check)
                if not check['all_paths_covered']:
                    return {"measurement_valid": False, "invalid_reason": 'Uncovered or infeasible initial layout paths',
                            "coverage": check}
            plan, observation = await executor.run(planner.measured_plan, snapshot, pending,
                                    now=now, submitted_s=time.perf_counter())
            samples.append(dict(observation, call=index, request_id=pending[0].request_id))
            if (index + 1) % args.progress_every == 0:
                print(json.dumps({"kind": "progress", "mode": "planner", "n_instances": n,
                    "layout": layout, "active_mode": active_mode, "seed": seed,
                    "completed_calls": index + 1, "planned_calls": args.planner_calls,
                    "elapsed_s": time.perf_counter() - began}), flush=True)
    finally:
        await executor.close()
    metrics = summarize_plans(samples)
    valid = metrics['no_candidate_ratio'] == 0
    return {"measurement_valid": valid, "coverage": checks[0], "planner": metrics,
            "invalid_reason": None if valid else 'One or more snapshots took the no-candidate fast path'}


def _periodic_call(function, args, submitted):
    began = time.perf_counter()
    cpu = time.thread_time()
    plan = function(*args)
    return plan, {"latency_ms": (time.perf_counter() - began) * 1000,
                  "worker_wait_ms": max(0., (began - submitted) * 1000),
                  "cpu_ms": (time.thread_time() - cpu) * 1000}


async def concurrent_cell(planner, role_costs, n, layout, active_mode, seed, args, samples):
    began_wall = time.time()
    began = time.perf_counter()
    deadline = began + args.concurrent_seconds
    snapshot = make_snapshot(n, layout, active_mode, seed, began_wall,
                             input_tokens=args.input_tokens, output_tokens=args.output_tokens)
    check = coverage(planner, snapshot,
        (request(f'coverage-{seed}', began_wall, args.input_tokens, args.output_tokens),), began_wall)
    if not check['all_paths_covered']:
        return {"measurement_valid": False, "invalid_reason": 'Uncovered or infeasible initial layout paths',
                "coverage": check}
    state = StateManager(snapshot)
    state.lock = TimedLock(state.lock)
    commit_lock = TimedLock()
    backend = ReplayBackend(state)
    frequency = FrequencyPlanner(planner, planner.frequency_costs)
    roles = ResidentRolePlanner(role_costs)
    executor = PlanningExecutor()
    queue = deque()
    history = deque(maxlen=256)
    counters = Counter()
    end_arrivals = asyncio.Event()
    admissions_done = asyncio.Event()
    stop = asyncio.Event()
    arrival_rate = n * args.arrivals_per_instance
    token_rate = n * args.token_updates_per_instance
    periods = {'frequency': [], 'role': []}
    backlog = []

    async def arrival_producer():
        rng = random.Random(seed)
        due = began + rng.expovariate(arrival_rate)
        index = 0
        while due < deadline:
            await asyncio.sleep(max(0, due - time.perf_counter()))
            budget = request(f'arrival-{seed}-{index}', began_wall + due - began,
                             args.input_tokens, args.output_tokens)
            queue.append((budget, due))
            history.append(budget)
            counters['arrived'] += 1
            samples.append({"kind": "arrival", "request_id": budget.request_id,
                            "at_s": due - began, "delivery_lag_ms": max(0, (time.perf_counter() - due) * 1000)})
            index += 1
            due += rng.expovariate(arrival_rate)
        end_arrivals.set()

    async def admissions():
        while not stop.is_set():
            if not queue:
                if end_arrivals.is_set() and time.perf_counter() >= deadline:
                    admissions_done.set()
                    return
                await asyncio.sleep(.001)
                continue
            budget, due = queue.popleft()
            counters['inflight'] += 1
            committed = False
            failure = 'stale_retry_exhausted'
            for attempt in range(args.stale_retries + 1):
                now = time.time()
                pending = (budget,) + tuple(item[0] for item in list(queue)[:2])
                plan, observation = await executor.run(planner.measured_plan, state.snapshot,
                    pending, now=now, submitted_s=time.perf_counter())
                counters['planning_attempts'] += 1
                observation.update(request_id=budget.request_id, attempt=attempt,
                                   at_s=time.perf_counter() - began)
                samples.append(observation)
                if not plan.feasible:
                    failure = 'no_candidate' if observation['no_candidate'] else 'active_rejection'
                    counters[failure] += 1
                    break
                async with commit_lock:
                    try:
                        await state.reserve(plan, time.time(), budget)
                    except StalePlan:
                        counters['stale_retries'] += 1
                        samples.append({"kind": "stale_retry", "request_id": budget.request_id,
                                        "attempt": attempt, "at_s": time.perf_counter() - began})
                        continue
                    await backend.execute(plan)
                    if not await backend.confirm(plan):
                        raise RuntimeError('Replay commit was not confirmed')
                    committed = True
                await backend.complete(plan, budget)
                counters['successful_commits'] += 1
                if time.perf_counter() <= deadline:
                    counters['commits_in_window'] += 1
                break
            elapsed = (time.perf_counter() - due) * 1000
            counters['inflight'] -= 1
            if not committed:
                counters['rejected'] += 1
            samples.append({"kind": "admission", "request_id": budget.request_id,
                            "latency_ms": elapsed, "committed": committed,
                            "failure": None if committed else failure,
                            "at_s": time.perf_counter() - began})

    async def token_updates():
        delivered = 0
        while not stop.is_set():
            # Keep telemetry alive while pending admissions drain. Stopping it
            # at the arrival cutoff would manufacture stale-plan failures.
            due = time.perf_counter()
            target = int((due - began) * token_rate)
            updates = target - delivered
            current = [r for i in state.snapshot.instances if i.role != 'prefill'
                       for r in i.requests if r.request_id.startswith('background-')]
            for offset in range(updates):
                if not current:
                    break
                old = current[(delivered + offset) % len(current)]
                now = time.time()
                # Repeated trace states hold the selected background load fixed.
                # They are not an autoregressive GPU execution simulation.
                budget = replace(old, emitted=old.emitted,
                                 first_token_s=now - .1, last_token_s=now)
                await state.update_budget(budget)
                counters['token_updates'] += 1
                if offset % 32 == 31:
                    await asyncio.sleep(0)
            delivered = target
            now = time.time()
            await state.publish(tuple(replace(i, timestamp_s=now) for i in state.snapshot.instances), now,
                                engine_waiting={i.instance_id: 0 for i in state.snapshot.instances})
            if time.perf_counter() >= deadline and admissions_done.is_set():
                return
            await asyncio.sleep(args.telemetry_period)

    async def periodic(kind, period):
        due = began + period
        while due < deadline and not stop.is_set():
            await asyncio.sleep(max(0, due - time.perf_counter()))
            start = time.perf_counter()
            now = time.time()
            snap = state.snapshot
            source = 'queued_requests'
            if kind == 'frequency':
                function, call_args = frequency.plan, (snap, now)
            else:
                pending = tuple(item[0] for item in list(queue)[:3])
                if not pending:
                    # Explicitly replay recent arrival shapes when the queue is
                    # empty, equivalent to a bounded benchmark demand fixture.
                    pending = tuple(replace(r, request_id=f'role-{seed}-{len(periods[kind])}-{j}',
                        arrival_s=now) for j, r in enumerate(list(history)[-3:]))
                    source = 'historical_shape_fixture'
                if not pending:
                    due += period
                    continue
                function, call_args = search_roles, (planner, roles, snap, pending, now)
            plan, observation = await executor.run(_periodic_call, function, call_args, time.perf_counter())
            committed = False
            stale = False
            if plan and (plan.frequencies or plan.roles):
                async with commit_lock:
                    current = state.snapshot
                    stale = current.version != plan.snapshot_version or time.time() > plan.expires_s
                    if not stale:
                        await backend.execute(plan)
                        committed = await backend.confirm(plan)
                        if committed and kind == 'role':
                            roles.confirmed(plan, time.time())
            finished = time.perf_counter()
            observation.update(kind=kind, at_s=finished - began,
                scheduled_lag_ms=max(0., (start - due) * 1000),
                cycle_ms=(finished - start) * 1000,
                over_period=finished - due > period, period_s=period,
                action_count=0 if plan is None else len(plan.frequencies) + len(plan.roles),
                committed=committed, stale=stale, demand_source=source,
                idle_role_candidates=sum(not (i.requests or i.running or i.waiting) for i in snap.instances))
            periods[kind].append(observation)
            samples.append(observation)
            # Preserve missed deadlines without issuing a catch-up CPU storm.
            skipped = max(0, int((finished - due) // period))
            counters[kind + '_missed_ticks'] += skipped
            due += (skipped + 1) * period

    async def heartbeat():
        due = began + .01
        while not stop.is_set() and due <= deadline:
            await asyncio.sleep(max(0., due - time.perf_counter()))
            now = time.perf_counter()
            samples.append({"kind": "event_loop", "lag_ms": max(0, (now - due) * 1000),
                            "at_s": now - began})
            backlog.append({"at_s": min(now - began, args.concurrent_seconds),
                            "pending": len(queue) + counters['inflight']})
            skipped = max(0, int((now - due) // .01))
            counters['heartbeat_missed_ticks'] += skipped
            due += (skipped + 1) * .01

    tasks = [asyncio.create_task(coro()) for coro in
             (arrival_producer, admissions, token_updates, heartbeat)]
    tasks.extend([asyncio.create_task(periodic('frequency', args.frequency_period)),
                  asyncio.create_task(periodic('role', args.role_period))])
    try:
        await asyncio.sleep(args.concurrent_seconds)
        counters['pending_at_window_end'] = len(queue) + counters['inflight']
        # Keep drain bounded and report unresolved admissions; no hidden tail.
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.drain_seconds)
        except asyncio.TimeoutError:
            counters['drain_timeout'] = 1
    finally:
        stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await executor.close()
    elapsed = time.perf_counter() - began
    samples.extend(dict(row, kind='backlog') for row in backlog)
    samples.extend(state.lock.samples('state_lock'))
    samples.extend(commit_lock.samples('commit_lock'))
    plans = summarize_plans(samples)
    successful = counters['successful_commits']
    arrivals = counters['arrived']
    # A no-candidate-heavy run can expose overload, but cannot validate full
    # algorithm scaling. Raw failures remain available for diagnosis.
    reasons = []
    if not successful:
        reasons.append('No successful real-state reservations/commits')
    if not periods['frequency'] or not periods['role']:
        reasons.append('A periodic controller did not execute during the window')
    if plans['no_candidate_ratio'] and plans['no_candidate_ratio'] > .1:
        reasons.append('More than 10% of plans used the no-candidate fast path')
    pending_tail = [row for row in backlog if row['at_s'] >= args.concurrent_seconds / 2]
    backlog_growth = None
    if len(pending_tail) >= 2 and pending_tail[-1]['at_s'] > pending_tail[0]['at_s']:
        backlog_growth = ((pending_tail[-1]['pending'] - pending_tail[0]['pending']) /
                          (pending_tail[-1]['at_s'] - pending_tail[0]['at_s']))
    return {"measurement_valid": not reasons, "invalid_reason": '; '.join(reasons) or None,
        "coverage": check, "planner": plans, "elapsed_s": elapsed,
        "arrival_window_s": args.concurrent_seconds, "input_arrival_rate_rps": arrival_rate,
        "input_token_update_rate_rps": token_rate, "counts": dict(counters),
        "successful_commit_throughput_rps": counters['commits_in_window'] / args.concurrent_seconds,
        "successful_commit_throughput_including_drain_rps": successful / elapsed,
        "successful_commit_ratio": successful / arrivals if arrivals else None,
        "stale_retry_ratio": counters['stale_retries'] / max(1, counters['planning_attempts']),
        "control_latency": distribution(row['latency_ms'] for row in samples if row['kind'] == 'admission'),
        "event_loop_lag": distribution(row['lag_ms'] for row in samples if row['kind'] == 'event_loop'),
        "state_lock_wait": distribution(state.lock.wait_ms), "state_lock_hold": distribution(state.lock.hold_ms),
        "commit_lock_wait": distribution(commit_lock.wait_ms), "commit_lock_hold": distribution(commit_lock.hold_ms),
        "backlog_growth_rps_second_half": backlog_growth,
        "periodic": {kind: {"latency": distribution(r['latency_ms'] for r in rows),
                "cycle": distribution(r['cycle_ms'] for r in rows),
                "over_period_ratio": sum(r['over_period'] for r in rows) / len(rows) if rows else None,
                "stale_ratio": sum(r['stale'] for r in rows) / len(rows) if rows else None,
                "action_count": sum(r['action_count'] for r in rows),
                "committed_cycles": sum(r['committed'] for r in rows),
                "missed_ticks": counters[kind + '_missed_ticks']}
                for kind, rows in periods.items()},
        "final_role_counts": dict(Counter(i.role for i in state.snapshot.instances))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', type=Path)
    p.add_argument('--transfers', type=Path)
    p.add_argument('--frequency-costs', type=Path,
                   help='JSON list or runtime configuration containing frequency_costs')
    p.add_argument('--role-costs', type=Path,
                   help='JSON list or runtime configuration containing role_costs')
    p.add_argument('--frequencies', type=int, nargs='+')
    p.add_argument('--instances', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128])
    p.add_argument('--layouts', choices=['mixed', 'selective', 'pd'], nargs='+',
                   default=['mixed', 'selective', 'pd'])
    p.add_argument('--active-modes', choices=['per_instance4', 'total32'], nargs='+',
                   default=['per_instance4', 'total32'])
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303, 404, 505])
    p.add_argument('--modes', choices=['planner', 'concurrent'], nargs='+', default=['planner', 'concurrent'])
    p.add_argument('--planner-calls', type=int, default=1000)
    p.add_argument('--progress-every', type=int, default=100,
                   help='Emit progress every this many completed planner calls')
    p.add_argument('--concurrent-seconds', type=float, default=300.)
    p.add_argument('--arrivals-per-instance', type=float, default=2.)
    p.add_argument('--token-updates-per-instance', type=float, default=80.)
    p.add_argument('--telemetry-period', type=float, default=.1,
                   help='Production telemetry interval: 0.1 seconds')
    p.add_argument('--frequency-period', type=float, default=.5)
    p.add_argument('--role-period', type=float, default=.5)
    p.add_argument('--drain-seconds', type=float, default=5.)
    p.add_argument('--stale-retries', type=int, default=3)
    p.add_argument('--input-tokens', type=int, default=128)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--smoke', action='store_true',
                   help='One seed, N=4, three planner calls, 0.8 s replay; synthetic software fixture if no profile supplied')
    return p


def validate_args(args):
    for field in ('planner_calls', 'progress_every', 'concurrent_seconds', 'arrivals_per_instance',
                  'token_updates_per_instance', 'telemetry_period', 'frequency_period',
                  'role_period', 'drain_seconds', 'input_tokens'):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{field} must be finite and positive')
    if args.output_tokens < 33 or args.stale_retries < 0:
        raise ValueError('output_tokens must be >=33 and stale_retries >=0')
    for n in args.instances:
        layout_roles(n, 'mixed')
    for name in ('seeds', 'instances', 'layouts', 'active_modes', 'modes'):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate {name} would overwrite a cell')


async def run(args, affinity):
    profiles, transfers, frequency_costs, role_costs, provenance = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted((Path(__file__).resolve().parents[1] / 'serving').glob('*.py'))
    sources += [Path(__file__), Path(__file__).with_name('fixtures.py'), Path(__file__).with_name('telemetry.py')]
    source_hashes = {str(path.resolve()): sha256(path) for path in sources}
    summaries = []
    # Rotate layout order by seed; cell snapshots/event streams remain paired.
    for seed_index, seed in enumerate(args.seeds):
        ordered = args.layouts[seed_index % len(args.layouts):] + args.layouts[:seed_index % len(args.layouts)]
        for n in args.instances:
            for layout in ordered:
                for active_mode in args.active_modes:
                    for mode in args.modes:
                        name = f'{mode}-n{n}-{layout}-{active_mode}-seed{seed}'
                        directory = args.output / name
                        directory.mkdir(parents=True, exist_ok=False)
                        planner = make_planner(profiles, transfers, frequency_costs)
                        samples = []
                        start_cpu = time.process_time()
                        before = memory()
                        full_protocol = (not args.smoke and len(args.seeds) == 5
                            and (args.planner_calls >= 1000 if mode == 'planner'
                                 else args.concurrent_seconds >= 300))
                        base = {"schema": 1, "scope": "control_plane_replay", "mode": mode,
                            "n_instances": n, "layout": layout, "active_mode": active_mode, "seed": seed,
                            "smoke": args.smoke, "formal_evidence": full_protocol,
                            "protocol_sampling_complete": full_protocol,
                            "virtual_resources": True, "cpu": affinity, "provenance": provenance,
                            "instrumentation": 'RSS includes retained benchmark observations; peak RSS is '
                                'process-lifetime and includes earlier cells. Lock samples are retained separately.',
                            "source_hashes": source_hashes,
                            "slow_topology": {"measured": False, "reason": 'The production slow topology '
                                'allocator supports the physical 8-GPU node; it is not extrapolated in this replay.'},
                            "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
                        try:
                            result = (await planner_cell(planner, n, layout, active_mode, seed, args, samples)
                                      if mode == 'planner' else
                                      await concurrent_cell(planner, role_costs, n, layout, active_mode, seed, args, samples))
                        except Exception as exc:
                            result = {"measurement_valid": False, "invalid_reason": f'{type(exc).__name__}: {exc}'}
                        summary = dict(base, **result, process_cpu_s=time.process_time() - start_cpu,
                                       memory_before=before, memory_after=memory())
                        summary['formal_evidence'] = bool(summary['formal_evidence'] and summary['measurement_valid'])
                        (directory / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in samples))
                        (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
                        summaries.append({"cell": name, "measurement_valid": summary['measurement_valid'],
                                          "formal_evidence": summary['formal_evidence'],
                                          "summary": str((directory / 'summary.json').resolve())})
                        print(json.dumps(summaries[-1]), flush=True)
    (args.output / 'index.json').write_text(json.dumps({"scope": "control_plane_replay", "cells": summaries}, indent=2) + '\n')
    return 0 if all(row['measurement_valid'] for row in summaries) else 2


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.smoke:
        args.instances = [4]
        args.seeds = [args.seeds[0]]
        args.planner_calls = 3
        args.concurrent_seconds = .8
        args.frequency_period = args.role_period = .2
    try:
        validate_args(args)
        with physical_core_affinity() as affinity:
            return asyncio.run(run(args, affinity))
    except (ValueError, RuntimeError, FileExistsError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
