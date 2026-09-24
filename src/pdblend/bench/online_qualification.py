"""Native abort/KV/control qualification on exactly two leased TP groups."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

from pdblend.engine.launcher import Fleet, make_specs
from pdblend.online.controller import Controller
from pdblend.online.native_control import NativeControl
from pdblend.online.qualification import qualify_live_proxy
from pdblend.online.router import Router
from pdblend.online.server import Proxy
from pdblend.online.transition_measurement import measure_transitions
from pdblend.planner.pool import Plan, PlannerConfig, PoolPlanner, SLO
from pdblend.profile.query.versions import load_profile
from pdblend.source_inventory import implementation_hashes
from pdblend.results.receipts import request_record_receipt
from .metering import Gpus
from .run import _serve_proxy


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


async def qualify(fleet, meter, model, out, port):
    specs = {iid: inst.spec for iid, inst in fleet.instances.items()}
    metadata = {iid: dict(tp=s.tp, pp=s.pp, generation=s.generation,
                         pool_id=s.pool_id, profile_key=s.profile_key,
                         model_id=Path(s.model).name) for iid, s in specs.items()}
    router = Router(specs, instance_metadata=metadata)
    native_timeout_s = 60.
    native = NativeControl(specs, timeout_s=native_timeout_s)
    proxy = Proxy({iid: s.base_url for iid, s in specs.items()}, router, native_cancel=native.cancel,
                  cancel_timeout_s=native_timeout_s+5.)
    controller = Controller(fleet, router, meter,
        PoolPlanner(model, PlannerConfig(slots=2, slo=SLO(30., 1.), freqs=model.freqs)),
        native_control=native, drain_timeout_s=60., log_path=out/'controller.jsonl')
    frequency = 1500 if 1500 in model.freqs else max(model.freqs)
    initial = Plan({'M': 2}, frequency, frequency, frequency, 0, 0., 0., 0.,
                   detail={'mechanism_forced_roles': True, 'formal_eligible': False})
    await controller.execute(initial)
    runner = await _serve_proxy(proxy, port)
    rng = random.Random(701)
    prompt = [rng.randrange(100, 5000) for _ in range(512)]

    async def control_action():
        # Wait for the continuing request to own its instance before choosing
        # the empty peer for parking. assign_roles preserves the busy owner.
        deadline = time.monotonic()+60.
        while not any(r.request_id.endswith('-reuse') and r.first_token_s and not r.finished_s
                      for rows in router.active.values() for r in rows):
            if time.monotonic() >= deadline:
                raise TimeoutError('continuing stream did not start')
            await asyncio.sleep(.01)
        before = len(controller.transition_events)
        lower = max(f for f in model.freqs if f <= frequency)
        if len([f for f in model.freqs if f < frequency]):
            lower = max(f for f in model.freqs if f < frequency)
        await controller.execute(replace(initial, counts={'M': 1, 'L1': 1}, f_M=lower))
        await controller.execute(initial)
        return dict(phases=controller.transition_events[before:],
                    actual_roles=dict(controller.roles), clocks=dict(controller.freqs),
                    native_tp_change=False)

    try:
        result = await qualify_live_proxy(proxy, f'http://127.0.0.1:{port}', prompt,
                                         max_tokens=512, timeout_s=180., control_action=control_action)
        result['terminal_native'] = {iid: await native.drain(iid, 60.) for iid in specs}
        result['transition_phases'] = controller.transition_events
        result['routes'] = [request_record_receipt(r) for r in router.records]
        return result
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--base-port', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    ids = [int(g) for g in args.gpus.split(',')]
    if len(ids) != args.tp*2 or len(set(ids)) != len(ids):
        raise ValueError('qualification needs exactly two disjoint TP groups')
    loaded = load_profile(args.profile, system='pdblend', model_id=Path(args.model).name, tp=args.tp)
    key = json.dumps(loaded.profile_key, sort_keys=True, separators=(',', ':'))
    specs = [replace(s, profile_key=key, native_control=True) for s in make_specs(args.model, ids, tp=args.tp,
             base_port=args.base_port, kv_connector=None)]
    args.out.mkdir(parents=True, exist_ok=True)
    binding = dict(profile=loaded.manifest_fields(), profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                   implementation=implementation_hashes(), specs=[asdict(s) for s in specs], seed=701,
                   image_digest=os.environ.get('PDBLEND_IMAGE_ID'),
                   gpu_uuids=os.environ.get('PDBLEND_GPU_UUIDS'), source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'))
    write(args.out/'preflight.json', dict(status='passed', hardware_executed=False, **binding))
    if args.preflight_only:
        return 0
    result = dict(status='failed', complete=False, formal_eligible=False, energy_comparable=False,
                  hardware_executed=True, bindings=binding)
    meter = Gpus(ids)
    sampler = meter.sampler(interval_s=.05)
    started = time.time()
    try:
        meter.reset_all()
        sampler.start()
        with Fleet(specs, args.out/'logs') as fleet:
            result['startup_s'] = fleet.start_all()
            result.update(asyncio.run(qualify(fleet, meter, loaded.model, args.out, args.base_port+100)))
        result.update(status='passed' if result['functional_passed'] else 'failed',
                      complete=bool(result['functional_passed']))
    except Exception as exc:
        result['error'] = repr(exc)
    finally:
        sampler.stop()
        meter.reset_all()
        result['wall_s'] = time.time()-started
        result['occupied_gpu_s'] = result['wall_s']*len(ids)
        result['transition_measurements'] = measure_transitions(result.get('transition_phases', []),
            sampler.samples, ids, sampler_error=sampler.error, power_source=sampler.power_source)
        result['metering_error'] = sampler.error
        write(args.out/'power.json', sampler.samples)
        write(args.out/'completion.json', result)
    return 0 if result.get('complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
