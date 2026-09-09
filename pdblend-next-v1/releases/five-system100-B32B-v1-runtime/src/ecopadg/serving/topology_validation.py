"""Measure explicit physical transitions while unaffected instances serve.

These forced transactions test mechanics and supply conservative switch costs.
They are neither autonomous policy decisions nor formal energy comparisons.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

import aiohttp
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler, trapezoid_energy
from ecopadg.metrics import clip_power_window
from .backend import ClockOwner, HttpEngineBackend
from .campaign import node_lease
from .evidence import sha256
from .measurement import power_evidence
from .topology import DockerLifecycle, InstanceSpec, TopologyManager, validate_layout


def measured_costs(transactions, source):
    """Repeated transitions share the largest observed time and node energy."""
    groups = {}
    for row in transactions:
        key = (tuple(sorted(row['source_tps'])), tuple(sorted(row['target_tps'])))
        cost = groups.setdefault(key, dict(source_tps=list(key[0]), target_tps=list(key[1]),
            duration_upper_s=0., energy_upper_j=0., source_sha256=source, cached_weights=True))
        cost['duration_upper_s'] = max(cost['duration_upper_s'],
            1.2*(row['finished_s']-row['started_s']))
        cost['energy_upper_j'] = max(cost['energy_upper_j'], 1.2*row['total_node_energy_j'])
    return list(groups.values())


def background_continuity(rows, started, finished):
    return [r for r in rows if r['started_s'] <= finished and r['finished_s'] >= started and r['passed']]


def instance(value):
    return InstanceSpec(value.get('instance_id', value.get('id')), value['tp'],
        tuple(value['gpus']), value['port'], value['kv_port'],
        value.get('role', 'mixed'), value.get('generation', 0))


def validate_transitions(manifest):
    """Reject an invalid later transition before touching the initial layout."""
    current = {s.instance_id: s for s in map(instance, manifest['instances'])}
    validate_layout(list(current.values()), range(8))
    background = manifest['background_instance']
    if background not in current:
        raise ValueError('background instance is absent')
    reference_tps = {current[i].tp for i in manifest['reference_instances']}
    for transition in manifest['transitions']:
        remove = transition['remove']
        added = list(map(instance, transition['add']))
        if (len(set(remove)) != len(remove) or
                not set(remove) <= set(current) or background in remove):
            raise ValueError('invalid removal or loss of the background instance')
        if not remove and not added:
            raise ValueError('empty physical transition')
        if not remove and not manifest.get('retained_weights'):
            raise ValueError('pure addition needs retained weights')
        if not {s.tp for s in added} <= reference_tps:
            raise ValueError('every target TP needs an ordinary reference')
        # Do not silently replace an ID before physical overlap validation.
        survivors = [s for i, s in current.items() if i not in remove]
        validate_layout(survivors + added, range(8))
        current = {s.instance_id: s for s in survivors + added}
    if not manifest['transitions']:
        raise ValueError('no physical transition was specified')


async def validate(manifest, out):
    validate_transitions(manifest)
    out.mkdir(parents=True, exist_ok=False)
    specs = list(map(instance, manifest['instances']))
    template = json.loads(Path(manifest['engine_template']).read_text())
    template.update(manifest.get('engine_overrides', {}))
    hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
    clocks = ClockOwner(hardware, range(8))
    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
    raw = dict(complete=False, passed=False, manifest=manifest, transactions=[],
        background=[], events=[], references={},
        purpose='forced physical mechanism and switch-cost measurement; no autonomous gain claim')
    stop = asyncio.Event()
    task = None
    manager = None

    class Journal:
        async def emit(self, event):
            raw['events'].append(event)

    async with aiohttp.ClientSession(trust_env=False) as session:
        backend = HttpEngineBackend([s.endpoint() for s in specs], session, clocks)

        async def freeze(ids, enabled):
            raw['events'].append(dict(kind='routing_freeze', ids=list(ids), enabled=enabled, at_s=time.time()))

        async def commit(ids, added):
            backend.replace_instances(ids, [s.endpoint() for s in added])

        manager = TopologyManager(backend, DockerLifecycle(out/'lifecycle', manifest['image'], template),
            specs, range(8), Journal(), freeze=freeze, commit=commit)
        prompts = [[9707, 1879, 13]*n for n in (16, 42, 85)]

        async def outputs(spec):
            result = []
            for prompt in prompts:
                response = await manager.request(spec, '/v1/completions', dict(prompt=prompt,
                    max_tokens=32, temperature=0, ignore_eos=True, stream=False))
                if len(response['token_ids']) != 32 or response['usage']['completion_tokens'] != 32:
                    raise RuntimeError('ordinary/reference output work differs')
                result.append(response['token_ids'])
            return result

        async def provenance():
            values = {}
            for key, spec in manager.specs.items():
                value = await manager.request(spec, '/provenance')
                # The /provenance endpoint includes engine code and model;
                # Docker inspection binds the actual process to the image.
                inspected = await manager.lifecycle.command('inspect', '--format', '{{.Image}}', 'pdb-v2-'+key)
                if inspected != manifest['image']:
                    raise RuntimeError('physical transition used a different engine image')
                values[key] = dict(value, image_id=inspected)
            return values

        async def background():
            spec = manager.specs[manifest['background_instance']]
            while not stop.is_set():
                started = time.time()
                try:
                    response = await manager.request(spec, '/v1/completions', dict(prompt=prompts[0],
                        max_tokens=32, temperature=0, ignore_eos=True, stream=False))
                    expected = raw['references'][str(spec.tp)][0]
                    passed = response['token_ids'] == expected and response['usage']['completion_tokens'] == 32
                    raw['background'].append(dict(started_s=started, finished_s=time.time(), passed=passed))
                    if not passed:
                        raise RuntimeError('background generation changed during reconfiguration')
                except Exception as exc:
                    raw['background'].append(dict(started_s=started, finished_s=time.time(), passed=False, error=repr(exc)))
                    raise

        try:
            await clocks.set(list(range(8)), 2520, verify_rise=False)
            for spec in specs:
                state = await manager.ready(spec)
                if any(state.get(k) for k in ('running', 'waiting', 'active', 'transfer_allocations')):
                    raise RuntimeError('initial physical layout is not drained')
                await manager.request(spec, '/control', dict(generation=state['generation']+1,
                    role='mixed', mode='continuous', admit_prefill=True, admit_decode=True))
            raw['provenance_before'] = await provenance()
            for key in manifest['reference_instances']:
                spec = manager.specs[key]
                values = await outputs(spec)
                previous = raw['references'].setdefault(str(spec.tp), values)
                if previous != values:
                    raise RuntimeError('same-TP ordinary references differ')
            if str(manager.specs[manifest['background_instance']].tp) not in raw['references']:
                raise ValueError('background TP needs an ordinary reference')
            sampler.start()
            task = asyncio.create_task(background())
            while not raw['background']:
                if task.done(): await task
                await asyncio.sleep(.01)
            for transition in manifest['transitions']:
                remove = tuple(transition['remove'])
                added = tuple(map(instance, transition['add']))
                source_tps = sorted(manager.specs[key].tp for key in remove)
                started = time.time()
                result = await manager.reconfigure(remove, added, savings_lower_j=1, cost_upper_j=0,
                    retained_weights=manifest['retained_weights'])
                finished = time.time()
                entry = dict(source_tps=source_tps, target_tps=sorted(s.tp for s in added),
                    started_s=started, finished_s=finished, result=result, output_checks=[])
                raw['transactions'].append(entry)
                for spec in added:
                    matches = await outputs(spec) == raw['references'][str(spec.tp)]
                    entry['output_checks'].append(dict(instance_id=spec.instance_id, tp=spec.tp, matches=matches))
                    if not matches: raise RuntimeError('reconstructed instance differs from ordinary reference')
                served = [r for r in raw['background'] if started <= r['finished_s'] <= finished and r['passed']]
                entry['unaffected_requests_completed'] = len(served)
                # A deletion can finish faster than one background request.
                # Observe the request crossing that interval without extending
                # the transaction's measured duration or energy boundary.
                deadline = time.monotonic()+10
                while not background_continuity(raw['background'],started,finished):
                    if task.done(): await task
                    if time.monotonic() >= deadline:
                        raise RuntimeError('no successful background request crossed the physical transition')
                    await asyncio.sleep(.01)
                entry['unaffected_requests_crossing_interval'] = len(background_continuity(raw['background'],started,finished))
                entry['background_observed_until_s'] = time.time()
                if task.done(): await task
            raw['provenance_after'] = await provenance()
            raw.update(complete=True, passed=True)
        finally:
            stop.set()
            if task:
                outcome = await asyncio.gather(task, return_exceptions=True)
                if any(isinstance(x, BaseException) for x in outcome): raw['passed'] = False
            await asyncio.sleep(.05)
            await asyncio.to_thread(sampler.stop)
            raw.update(power_samples=sampler.samples, frequency_samples=sampler.frequency_samples,
                power_source=sampler.power_source, power_metadata=sampler.power_metadata,
                commanded_frequencies=dict(clocks.applied), sampling_error=sampler.error,
                live_instances=[asdict(s) for s in manager.specs.values()])
            raw['power_evidence'] = power_evidence(sampler.samples,sampler.power_source,sampler.power_metadata)
            if sampler.error or not raw['power_evidence']['power_source_verified']: raw['passed'] = False
            for item in raw['transactions']:
                item['total_node_energy_j'] = trapezoid_energy(clip_power_window(sampler.samples,
                    item['started_s'], item['finished_s'], pad_s=0))
            await clocks.close()
            await asyncio.to_thread((out/'raw.json').write_text, json.dumps(raw))
    if not raw['passed']: raise RuntimeError('physical transition validation failed')
    source = sha256(out/'raw.json')
    costs = measured_costs(raw['transactions'], source)
    (out/'topology_costs.json').write_text(json.dumps(costs, indent=2))
    print(json.dumps(dict(passed=True, transactions=len(costs), costs=costs)))
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with node_lease():
        asyncio.run(validate(json.loads(args.manifest.read_text()), args.out))


if __name__ == '__main__':
    main()
