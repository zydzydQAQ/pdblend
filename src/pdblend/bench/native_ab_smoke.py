"""Bounded native functional and latency A/B checks on one leased fleet.

This development runner never awards formal performance or energy eligibility.
The control freezes the same initial plan; the candidate replans every 10 s.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import uuid

from pdblend.engine.client import EngineClient
from pdblend.engine.launcher import Fleet, make_specs
from pdblend.online.native_control import NativeControl
from pdblend.online.controller import Controller
from pdblend.online.observations import backlog_snapshot
from pdblend.online.policies import get_policy
from pdblend.planner.pool import PlannerConfig, PoolPlanner, SLO
from pdblend.profile.query.versions import load_profile
from .client import Request
from .metering import Gpus
from .online_qualification import qualify
from .run import _point, offline_forecast
from . import run as run_module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


class Cohort:
    """Per-member atomic receipts; a failed member cannot strand its peers."""
    def __init__(self, root, member, timeout_s=1800.):
        self.root, self.member, self.timeout_s = Path(root), member, timeout_s
        self.config = json.loads((self.root / 'cohort.json').read_text())
        self.members = self.config['members']
        if (not self.members or len(set(self.members)) != len(self.members)
                or member not in self.members or any(not re.fullmatch(r'[a-zA-Z0-9_-]+', x) for x in self.members)
                or self.config.get('qualification_limit') != .05):
            raise ValueError('invalid bounded smoke cohort')
        self.state = dict(member=member, status='active', stages={}, cohort_id=self.config['cohort_id'])

    def states(self):
        result = {}
        for member in self.members:
            path = self.root / (member + '.json')
            if path.exists():
                row = json.loads(path.read_text())
                if row.get('cohort_id') != self.config['cohort_id']:
                    raise ValueError('stale cohort member identity')
                result[member] = row
        return result

    def publish(self, **fields):
        self.state.update(fields, updated_s=time.time())
        write(self.root / (self.member + '.json'), self.state)

    def wait(self, predicate):
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            rows = self.states()
            if len(rows) == len(self.members) and predicate(rows):
                return rows
            time.sleep(.25)
        raise TimeoutError('cohort stage timed out; no GPU evidence invented')

    def barrier(self, stage, evidence=None):
        self.state['stages'][stage] = dict(at_s=time.time(), evidence=evidence)
        self.publish()
        return self.wait(lambda rows: all(stage in r.get('stages', {}) or r['status'] in ('failed', 'done')
                                         for r in rows.values()))

    def active_members(self):
        return [m for m, row in self.states().items() if row['status'] == 'active']

    @contextmanager
    def load_lock(self):
        with (self.root / 'load.lock').open('a') as handle:
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('cohort model load lock timed out')
                    time.sleep(.25)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_trace(path, split, duration):
    data = json.loads(Path(path).read_text())
    expected_seed = 701 if split == 'evaluation' else 9701
    if data.get('seed') != expected_seed or data.get('split') != split or data.get('duration_s') != duration:
        raise ValueError('trace identity, split or duration differs')
    rows = [Request(**row) for row in data['requests']]
    if (not rows or len({r.idx for r in rows}) != len(rows)
            or any(not math.isfinite(r.arrival_s) or not 0 <= r.arrival_s < duration
                   or r.max_tokens != 128 or r.input_tokens not in (512, 1024, 2048)
                   or any(type(t) is not int or not 0 <= t < 151643 for t in r.prompt) for r in rows)
            or any(a.arrival_s > b.arrival_s for a, b in zip(rows, rows[1:]))):
        raise ValueError('unsupported development trace shape or arrivals')
    return rows


def prepare(args):
    ids = [int(x) for x in args.gpus.split(',')]
    model_id = Path(args.model).name
    if (len(set(ids)) != len(ids) or any(i < 0 for i in ids)
            or args.tp not in (1, 2) or len(ids) != (8 if args.pd_eight else args.tp * 2)
            or args.pd_eight and (model_id != 'Qwen2.5-7B-Instruct' or args.tp != 1)):
        raise ValueError('smoke needs two TP groups, or the explicit 7B TP1 eight-GPU experiment')
    manifest = json.loads(args.input_manifest.read_text())
    if ((manifest.get('model_id'), manifest.get('tp'), manifest.get('pp')) != (model_id, args.tp, 1)
            or manifest.get('image_digest') != os.environ.get('PDBLEND_IMAGE_ID')
            or manifest.get('source_sha256') != os.environ.get('PDBLEND_SOURCE_SHA256')):
        raise ValueError('smoke immutable runtime identity differs')
    paths = dict(profile=args.profile, trace=args.trace, tuning_trace=args.tuning_trace,
                 model_verification=Path(os.environ['PDBLEND_MODEL_VERIFICATION_RECEIPT']))
    if set(manifest['inputs']) != set(paths):
        raise ValueError('exact profile/trace/tuning/model identity bindings required')
    for key, path in paths.items():
        if sha(path) != manifest['inputs'][key]['sha256']:
            raise ValueError('frozen input checksum differs: ' + key)
    source_manifest = Path(os.environ['PDBLEND_SOURCE_MANIFEST'])
    source = json.loads(source_manifest.read_text())
    if canonical(source['files']) != manifest['source_sha256'] or source['source_sha256'] != manifest['source_sha256']:
        raise ValueError('frozen source inventory identity differs')
    source_root = Path(__file__).resolve().parents[2]
    for name, digest in source['files'].items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or sha(path) != digest:
            raise ValueError('frozen source changed: ' + name)
    cohort = Cohort(args.cohort_root, args.member)
    if args.pd_eight and len(cohort.members) != 1:
        raise ValueError('eight-GPU PD development requires a one-member cohort')
    trace = read_trace(args.trace, 'evaluation', 300)
    tuning = read_trace(args.tuning_trace, 'tuning', 60)
    if sha(args.trace) == sha(args.tuning_trace):
        raise ValueError('evaluation and independent tuning inputs must differ')
    loaded = load_profile(args.profile, system='pdblend', model_id=model_id, tp=args.tp)
    model = loaded.model
    policy = get_policy('pdblend')
    if args.pd_eight:
        policy = replace(policy, allow_park=('L1',))
    slo = SLO(5., .15)
    # Shape coverage must pass at every permitted frequency before loading GPUs.
    for frequency in model.freqs:
        for n in sorted({r.input_tokens for r in trace + tuning}):
            model.prefill_seconds(n, frequency)
            model.step_seconds(1, n, frequency)
            model.step_seconds(1, n + 127, frequency)
    cfg = policy.planner_config(PlannerConfig(slots=len(ids) // args.tp, slo=slo, freqs=model.freqs))
    initial = PoolPlanner(model, cfg).plan(offline_forecast(tuning))
    if initial.detail.get('fallback') or not math.isfinite(initial.power_w):
        raise ValueError('initial tuning plan lacks feasible measured coverage')
    key = json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':'))
    initial = replace(initial, tp=args.tp, pp=1, profile_key=key)
    specs = make_specs(args.model, ids, tp=args.tp, base_port=args.base_port,
        native_control=True, profile_key=key,
        kv_connector='P2pNcclConnector' if args.pd_eight else None)
    raw_uuids = os.environ.get('PDBLEND_GPU_UUIDS', '')
    gpu_uuids = json.loads(raw_uuids) if raw_uuids.startswith('[') else raw_uuids.split(',')
    bindings = dict(manifest, gpu_uuids=gpu_uuids, gpu_count=len(ids),
                    trace_sha256=sha(args.trace), seed=701, duration_s=300,
                    cohort_id=cohort.config['cohort_id'], policy=asdict(policy),
                    initial_plan=asdict(initial), profile_selection=loaded.manifest_fields(),
                    parking_restriction='L1_only' if args.pd_eight else None,
                    comparison='frozen_initial_plan_vs_periodic_replanning',
                    scope='synthetic_development_latency_and_functional',
                    source_files_verified=len(source['files']))
    return ids, loaded, policy, slo, initial, specs, trace, tuning, cohort, bindings


async def reset_native(fleet, meter, frequency=1500):
    """Restore active weights before touching the native scheduler inventory."""
    for instance in fleet.instances.values():
        if not instance.alive():
            raise RuntimeError('resident native process exited')
        if instance.state == 'sleeping':
            await asyncio.to_thread(instance.wake_up)
    native = NativeControl({iid: i.spec for iid, i in fleet.instances.items()}, timeout_s=60.)
    drained = {iid: await native.drain(iid, 60.) for iid in fleet.instances}
    for gpu in meter.gpus:
        await asyncio.to_thread(meter.unpark, gpu)
        await asyncio.to_thread(meter.set_clock, gpu, frequency)
    resumed = {iid: await native.resume(iid, 'M') for iid in fleet.instances}
    return dict(drained=drained, resumed=resumed, at_s=time.time(), frequency=frequency)


async def drain(fleet):
    native = NativeControl({iid: i.spec for iid, i in fleet.instances.items()}, timeout_s=60.)
    # Sleeping engines still expose native scheduler receipts.
    return {iid: await native.drain(iid, 60.) for iid in fleet.instances}


async def representative_probe(fleet, meter, prompt, tag, cohort=None):
    latencies, rows = [], []
    sampler = meter.sampler(interval_s=.05)
    # Same warmup and settling in solo and parallel phases.
    async def request(instance, suffix):
        async with EngineClient(instance.spec.instance_id, instance.spec.base_url, timeout_s=120.) as client:
            value = await client.complete(prompt, 128, tag + '-' + suffix, seed=701)
        if value.error or value.completion_tokens != 128 or not value.stream_done or not value.usage_received:
            raise RuntimeError('representative native request failed: ' + repr(value.error))
        return value
    for i, instance in enumerate(fleet.instances.values()):
        await request(instance, 'warm-' + str(i))
    if cohort is not None:
        states = await asyncio.to_thread(cohort.barrier, 'parallel-warmed')
        started = max(row['stages'].get('parallel-warmed', {}).get('at_s', 0)
                      for row in states.values()) + 2.
    else:
        started = time.time() + 2.
    await asyncio.sleep(max(0., started-time.time()))
    sampler.start()
    deadline = started + 16.
    try:
        repeat = 0
        while time.time() < deadline or repeat < 3:
            values = await asyncio.gather(*(request(inst, f'{repeat}-{i}')
                                           for i, inst in enumerate(fleet.instances.values())))
            for value in values:
                latency = value.finished_s - value.submitted_s
                if value.finished_s <= deadline:
                    latencies.append(latency)
                rows.append(dict(request_id=value.request_id, latency_s=latency,
                                 completion_tokens=value.completion_tokens,
                                 submitted_s=value.submitted_s, finished_s=value.finished_s))
            repeat += 1
    finally:
        ended = time.time()
        sampler.stop()
    samples = [(t, sum(w)) for t, w in sampler.samples if started + 2 <= t <= deadline - 2]
    if sampler.error or not latencies or len(samples) < 2 or samples[-1][0] <= samples[0][0]:
        raise RuntimeError('representative group power evidence missing')
    energy = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(samples, samples[1:]))
    return dict(latency_s=latencies, median_latency_s=statistics.median(latencies),
                mean_power_w=energy / (samples[-1][0] - samples[0][0]),
                gpus=list(meter.gpus), group_gpus=list(meter.gpus), started_s=started, ended_s=ended, requests=rows,
                measurement_start_s=started+2, measurement_end_s=deadline-2,
                power_samples=sampler.samples, power_source=sampler.power_source,
                power_error=sampler.error)


def compare_probes(solo, parallel):
    deviations = {}
    for key in ('median_latency_s', 'mean_power_w'):
        left, right = solo[key], parallel[key]
        if not all(math.isfinite(x) and x > 0 for x in (left, right)):
            raise ValueError('invalid measured parallel interference value')
        deviations[key] = abs(right - left) / left
    return dict(passed=all(x <= .05 for x in deviations.values()), limit=.05,
                relative_errors=deviations)


@contextmanager
def timed_planning(path):
    """Initial CPU selection is outside this scope; calls originate in Controller."""
    original, rows = PoolPlanner.plan, []
    def measured(self, *args, **kwargs):
        started, clock = time.time(), time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            rows.append(dict(t=started, duration_s=time.perf_counter() - clock,
                             origin='periodic_controller'))
    PoolPlanner.plan = measured
    try:
        yield rows
    finally:
        PoolPlanner.plan = original
        Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


def warmup_trace(trace):
    # Separate request IDs prevent warmup/evaluation accounting collisions.
    return [replace(r, idx=1000000 + i, arrival_s=i * .1) for i, r in enumerate(trace[:6])]


class FixedPlanShieldController(Controller):
    """Freeze optimization while retaining the exact same online safety shield.

The general Controller currently gates safety execution inside its periodic
replanning branch. A fixed-plan control needs an explicit safety-only loop.
"""
    async def run(self, stop):
        if self.initial_plan is None:
            raise ValueError('fixed-plan control requires the independently selected initial plan')
        await self.execute(self.initial_plan)
        while not stop.is_set():
            await asyncio.sleep(self.tick_s)
            now = time.time()
            self.forecaster.set_backlog(backlog_snapshot(self.router))
            if self.shield is None:
                continue
            pressure = self.shield.observe(self.router.observation_records(60., now), now)
            level = self.shield.update(pressure, now)
            if level or self.shield.floor_active:
                safe = self.shield.apply(self.plan_now, pressure, self.max_freq)
                if safe.key() != self.plan_now.key():
                    self.log('fixed_shield_override', level=level)
                    await self.execute(safe)
        self.log('stop', roles=dict(self.roles))


@contextmanager
def arm_controller(arm):
    original = run_module.Controller
    if arm == 'A':
        run_module.Controller = FixedPlanShieldController
    try:
        yield
    finally:
        run_module.Controller = original


def run(args, prepared):
    ids, loaded, policy, slo, initial, specs, trace, tuning, cohort, bindings = prepared
    result = dict(status='running', complete=False, functional_passed=False, bindings=bindings,
                  formal_eligible=False, energy_comparable=False, hardware_executed=True,
                  arms={}, errors=[])
    meter, fleet = Gpus(ids), Fleet(specs, args.out / 'logs')
    cohort.publish()
    try:
        with fleet:
            try:
                # Only the group's own clocks are changed. Every instance load is
                # locked and staggered; later requests use the same processes.
                meter.reset_all()
                startup = {}
                for iid, instance in fleet.instances.items():
                    with cohort.load_lock():
                        instance.start()
                        startup[iid] = instance.wait_ready()
                pids = {iid: i.process.pid for iid, i in fleet.instances.items()}
                result.update(startup_s=startup, shared_lifecycle_pids=pids)
                cohort.barrier('loaded')
                functional_out = args.out / 'functional'
                functional_out.mkdir(parents=True, exist_ok=True)
                subset = Fleet([], functional_out / 'logs')
                subset.instances = dict(list(fleet.instances.items())[:2])
                functional = asyncio.run(qualify(subset, meter, loaded.model, functional_out, args.base_port + 100))
                write(functional_out / 'completion.json', functional)
                result['functional_passed'] = bool(functional.get('functional_passed'))
                if not result['functional_passed']:
                    raise RuntimeError('native disconnect/control qualification failed: ' + str(functional.get('error')))
                write(args.out / 'stages' / 'functional-drain.json', asyncio.run(drain(fleet)))
                write(args.out / 'stages' / 'functional-reset.json', asyncio.run(reset_native(fleet, meter)))
                if args.pd_eight:
                    from pdblend_runtime.public_pd_probe import run as public_golden
                    golden = asyncio.run(public_golden(specs[:2], args.out / 'pd-golden.json'))
                    if not golden.get('complete'):
                        raise RuntimeError('public symmetric PD token golden failed')
                    write(args.out / 'stages' / 'golden-drain.json', asyncio.run(drain(fleet)))
                    write(args.out / 'stages' / 'golden-reset.json', asyncio.run(reset_native(fleet, meter)))
                cohort.barrier('functional')
                prompt = trace[0].prompt[:512]
                # Freeze the participating loaded environment across all solo and
                # parallel measurements. Failed members remain resident and idle.
                qualified_members = cohort.active_members()
                solo = None
                for member in cohort.members:
                    cohort.barrier('solo-before-' + member)
                    if member == args.member:
                        solo = asyncio.run(representative_probe(fleet, meter, prompt, 'solo'))
                        write(args.out / 'qualification' / 'solo.json', solo)
                        write(args.out / 'stages' / 'solo-drain.json', asyncio.run(drain(fleet)))
                        asyncio.run(reset_native(fleet, meter))
                    cohort.barrier('solo-after-' + member)
                cohort.barrier('parallel-before')
                parallel = asyncio.run(representative_probe(fleet, meter, prompt, 'parallel', cohort))
                write(args.out / 'qualification' / 'parallel.json', parallel)
                comparison = compare_probes(solo, parallel)
                comparison['members_unchanged'] = qualified_members == cohort.active_members()
                comparison['passed'] &= comparison['members_unchanged']
                write(args.out / 'qualification' / 'comparison.json', comparison)
                write(args.out / 'stages' / 'parallel-drain.json', asyncio.run(drain(fleet)))
                asyncio.run(reset_native(fleet, meter))
                rows = cohort.barrier('qualified', comparison)
                remeasure = [member for member, row in rows.items() if row['status'] == 'active'
                             and not row['stages'].get('qualified', {}).get('evidence', {}).get('passed', False)]
                parallel_ok = args.member not in remeasure
                mode = 'parallel_qualified' if parallel_ok else 'parallel_unqualified'
                result.update(comparison_mode=mode, parallel_qualification_passed=parallel_ok,
                              models_requiring_serial_remeasurement=remeasure)

                def execute_arm(arm, comparison_mode):
                    arm_out = args.out / arm
                    arm_out.mkdir(parents=True, exist_ok=True)
                    write(args.out / 'stages' / (arm + '-reset.json'), asyncio.run(reset_native(fleet, meter)))
                    members_before = cohort.active_members()
                    arm_started = time.time()
                    with arm_controller(arm), timed_planning(arm_out / 'planning-times.jsonl'):
                        summary = asyncio.run(_point(fleet, meter, loaded.model, policy, slo, trace,
                            warmup_trace(trace), arm_out, args.base_port + 100, 10., 120.,
                            fixed_plan=initial if arm == 'A' else None, initial_plan=initial,
                            sampling_seed=701, planning_trace=tuning, observation_duration_s=300.))
                    interval = dict(start_s=arm_started, end_s=time.time())
                    stable = all(i.alive() and i.process.pid == pids[iid] for iid, i in fleet.instances.items())
                    unchanged = members_before == cohort.active_members()
                    summary.update(arm=arm, bindings=bindings, shared_lifecycle_pids=pids,
                        resident_processes_unchanged=stable, cohort_members_unchanged=unchanged,
                        comparison_mode=comparison_mode, formal_eligible=False, energy_comparable=False)
                    write(arm_out / 'summary.json', summary)
                    write(args.out / 'stages' / (arm + '-drain.json'), asyncio.run(drain(fleet)))
                    if not stable or not unchanged:
                        raise RuntimeError('resident process/cohort changed during an A/B window')
                    result['arms'][arm] = dict(summary=str(arm_out / 'summary.json'), sha256=sha(arm_out / 'summary.json'))
                    return interval

                # Retain the original full parallel layout for both arms. Only
                # the models failing paired interference are subsequently rerun.
                for arm in ('A', 'B'):
                    cohort.barrier(arm + '-parallel-before')
                    execute_arm(arm, mode)
                    cohort.barrier(arm + '-parallel-after')
                for owner in remeasure:
                    cohort.barrier('serial-' + owner + '-before')
                    if owner == args.member:
                        archived = args.out / 'parallel-unqualified'
                        archived.mkdir()
                        for arm in ('A', 'B'):
                            (args.out / arm).rename(archived / arm)
                        intervals = {arm: execute_arm(arm, 'serial_after_interference') for arm in ('A', 'B')}
                        write(args.out / 'qualification' / 'serial-execution.json',
                              dict(exclusive_member=True, model_id=bindings['model_id'],
                                   cohort_id=bindings['cohort_id'], intervals=intervals,
                                   group_gpu_uuids=bindings['gpu_uuids'],
                                   basis='all cohort members acknowledge before/after owner barriers'))
                        comparison['comparison_mode'] = 'serial_after_interference'
                        write(args.out / 'qualification' / 'comparison.json', comparison)
                        result['comparison_mode'] = 'serial_after_interference'
                    cohort.barrier('serial-' + owner + '-after')
                result.update(status='passed', complete=True)
            except Exception as exc:
                result.update(status='failed', complete=False)
                result['errors'].append(repr(exc))
                cohort.publish(status='failed', error=repr(exc))
                try:
                    write(args.out / 'stages' / 'failure-drain.json', asyncio.run(drain(fleet)))
                except Exception as cleanup_error:
                    result['errors'].append('failure drain: ' + repr(cleanup_error))
            finally:
                # Avoid unloading one participant during another's accepted arm.
                cohort.publish(status='done' if result['complete'] else 'failed', completed_s=time.time())
                try:
                    cohort.wait(lambda rows: all(r['status'] in ('failed', 'done') for r in rows.values()))
                except Exception as exc:
                    result.update(status='failed', complete=False)
                    result['errors'].append('terminal barrier: ' + repr(exc))
    finally:
        meter.reset_all()
        result['fleet_events'] = fleet.events()
        write(args.out / 'completion.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'gpus', 'member'):
        parser.add_argument('--' + flag, required=True)
    for flag in ('profile', 'out', 'cohort-root', 'trace', 'tuning-trace', 'input-manifest'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--pd-eight', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare(args)
        (args.out / 'trace.json').write_bytes(args.trace.read_bytes())
        (args.out / 'tuning-trace.json').write_bytes(args.tuning_trace.read_bytes())
        write(args.out / 'preflight.json', dict(status='passed', hardware_executed=False,
              bindings=prepared[-1], formal_eligible=False, energy_comparable=False,
              specs=[asdict(s) for s in prepared[5]]))
        if args.preflight_only:
            return 0
        result = run(args, prepared)
        from .native_ab_audit import audit
        verdict = audit(args.out)
        write(args.out / 'audit.json', verdict)
        result.update(execution_complete=result['complete'], audit_status=verdict['status'],
                      classification=verdict.get('classification'), audit_path=str(args.out / 'audit.json'),
                      audit_sha256=sha(args.out / 'audit.json'))
        write(args.out / 'completion.json', result)
        return 0 if result['complete'] else 1
    except Exception as exc:
        write(args.out / ('preflight.json' if args.preflight_only else 'completion.json'),
              dict(status='failed', complete=False, error=repr(exc), hardware_executed=False,
                   formal_eligible=False, energy_comparable=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
