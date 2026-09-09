"""Bounded, exact-input TP2 temporal correctness diagnosis; no performance claims."""
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import signal
import time
import uuid

import aiohttp
from ecopadg.serving.campaign import node_lease
from ecopadg.serving.backend import ClockOwner
from ecopadg.serving.measurement import power_evidence, save_raw
from ecopadg.measure.backends import PynvmlBackend
from ecopadg.measure.power import PowerSampler, trapezoid_energy
from ecopadg.metrics import clip_power_window

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'B32B-engine-v3-candidate-v2'
RELEASE = ROOT.parents[1] / 'releases/io-v3-runtime'
PORTS = (33500, 33501)
IDS = ('nextv3b0', 'nextv3b1')
NAMES = tuple('pdb-v2-' + x for x in IDS)
IMAGE = 'sha256:fd4ba34686c028ec6ba0ae17220f833b24c2f45f29535f735066f5a7a27004c2'
BODY = dict(prompt=[9707, 1879, 13] * 32, max_tokens=64,
            temperature=0, ignore_eos=True, stream=False)
BODIES = (BODY, dict(BODY, prompt=BODY['prompt'] * 2))
BUDGET = dict(schema_version=1, max_num_batched_tokens=8192, max_num_seqs=32)
RESIDUALS = ('active', 'running', 'waiting', 'kv_allocations', 'transfer_allocations',
             'transfer_buffered_tensors', 'transfer_inflight_receives', 'transfer_inflight_sends')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_difference(a, b):
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return dict(index_zero_based=i, output_position_one_based=i+1, reference_token=x, observed_token=y)
    return None


def idle(r):
    return all(k in r for k in RESIDUALS) and not any(r[k] for k in RESIDUALS)


def ack(r, generation=None):
    require(not r.get('error') and not r.get('runtime_error'), 'engine error')
    require(r.get('transport_healthy') is True, 'unhealthy TP transport')
    require(r['generation'] == r['acknowledged_generation'], 'owner ACK mismatch')
    if generation is not None:
        require(r['generation'] == generation, 'wrong owner generation')
    require(r.get('scheduler_budget_pending') is None, 'pending budget')
    require(r.get('scheduler_budget_effective') == dict(max_num_batched_tokens=8192, max_num_seqs=32),
            'effective startup budget changed')
    owners = [x.get('controls', {}).get('runtime') for x in r.get('scheduler_io', [])]
    require(owners and all(x and x.get('generation') == r['generation'] and not x.get('error') for x in owners),
            'runtime owner/cache generations disagree')
    require(0 <= time.time() - r.get('timestamp', 0) <= 1, 'stale runtime cache')


class Diagnosis:
    def __init__(self):
        require(not (ROOT / 'status.json').exists() and not (ROOT / 'http.jsonl').exists(), 'existing run retained')
        self.log = (ROOT / 'http.jsonl').open('x', buffering=1)
        self.tasks = []
        self.owned = set()
        self.issued = []
        self.verified_for_controls = False
        self.state = dict(complete=False, scope='TP2 exact-output correctness diagnosis only',
            inherited_budget_declaration_difference='Original temporal run had no explicit budget. Its cleanup wrote 8192/32; this run inherits it without restart. Original control payload still omits budget.',
            bodies=BODIES, started_s=time.time(), phases=[], work_timeout_s=390, cleanup_timeout_s=90,
            exact_equality_required=True, no_logits_available=True, baseline_execution=False)
        self.save()

    def save(self):
        write(ROOT / 'status.json', self.state)

    async def command(self, *args, timeout=12):
        p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout)
        except BaseException:
            if p.returncode is None:
                p.kill()
                await p.wait()
            raise
        require(p.returncode == 0, out.decode(errors='replace')[-1000:])
        return out.decode(errors='replace')

    async def http(self, port, route, body=None, rid=None, label='', timeout=15):
        record = dict(port=port, route=route, request=body, request_id=rid, label=label, started_s=time.time())
        try:
            async with self.session.request('GET' if body is None else 'POST', f'http://127.0.0.1:{port}{route}',
                    json=body, headers={'X-Request-Id': rid} if rid else None,
                    timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                text = await response.text()
                try:
                    data = json.loads(text)
                except ValueError:
                    data = text
                record.update(status=response.status, body=data)
                require(response.status == 200, route + ': ' + str(data)[:400])
                return data
        except BaseException as exc:
            record['error'] = repr(exc)
            raise
        finally:
            record['finished_s'] = time.time()
            self.log.write(json.dumps(record, allow_nan=False) + '\n')

    async def runtime(self, port=33501, label='runtime'):
        return await self.http(port, '/runtime', label=label)

    async def wait_idle(self, port=33501, timeout=15):
        deadline = time.monotonic() + timeout
        while True:
            r = await self.runtime(port, 'wait-idle')
            ack(r)
            if idle(r):
                return r
            require(time.monotonic() < deadline, 'owned work did not drain')
            await asyncio.sleep(.02)

    async def control(self, port=33501, **changes):
        before = await self.runtime(port, 'control-before')
        payload = {key: before[key] for key in ('role', 'mode', 'admit_prefill', 'generation')}
        payload.update(admit_decode=before.get('admit_decode', True))
        payload.update(changes, generation=before['generation'] + 1)
        result = await self.http(port, '/control', payload, label='original-profiler-control-payload', timeout=40)
        after = await self.runtime(port, 'control-owner-ack')
        ack(after, payload['generation'])
        return dict(request=payload, response=result, owner=after)

    def task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def generate(self, phase, slot, port=33501):
        rid = 'temporaldiag-' + uuid.uuid4().hex
        row = dict(port=port, slot=slot, request_id=rid, body=BODIES[slot], dispatched_s=time.time())
        phase.setdefault('requests', []).append(row)
        self.owned.add((port, rid)); self.issued.append((port, rid)); self.save()
        try:
            reply = await self.http(port, '/v1/completions', BODIES[slot], rid, phase['name'], timeout=90)
            row['response'] = reply
            self.owned.discard((port, rid))
            require(len(reply.get('token_ids', [])) == 64, 'missing output token IDs')
            require(reply.get('usage', {}).get('prompt_tokens') == len(BODIES[slot]['prompt'])
                    and reply['usage'].get('completion_tokens') == 64, 'output work changed')
            return reply['token_ids']
        except BaseException as exc:
            row['error'] = repr(exc)
            raise
        finally:
            row['finished_s'] = time.time(); self.save()

    async def allocated(self, task, rid):
        deadline = time.monotonic() + 30
        while True:
            r = await self.runtime(label='wait-original-allocation-gate')
            if rid in r['kv_allocations'] and r['running'] == 1:
                return r
            require(not task.done() and time.monotonic() < deadline, 'first request no longer held at original allocation gate')
            await asyncio.sleep(.005)

    async def phase(self, name, mode, slots, pairing=False, port=33501):
        phase = dict(name=name, mode=mode, slots=slots, port=port, started_s=time.time(), complete=False)
        self.state['phases'].append(phase); self.save()
        index = PORTS.index(port)
        timeline = OLD / 'runtime' / (IDS[index] + '.control.events.jsonl')
        require(timeline.is_file(), 'owner event stream missing')
        offset = timeline.stat().st_size
        phase['event_offset_start'] = offset
        try:
            phase['before'] = await self.wait_idle(port)
            phase['control'] = await self.control(port, mode=mode, admit_prefill=True, admit_decode=True)
            if pairing and mode == 'temporal':
                first = self.task(self.generate(phase, slots[0], port))
                await asyncio.sleep(0)
                rid = phase['requests'][0]['request_id']
                phase['first_allocated'] = await self.allocated(first, rid)
                phase['close_prefill'] = await self.control(port, admit_prefill=False)
                second = self.task(self.generate(phase, slots[1], port))
                await asyncio.sleep(.1)
                held = phase['held'] = await self.runtime(port, 'original-temporal-held')
                second_id = phase['requests'][1]['request_id']
                require(second_id not in held['kv_allocations'] and held['waiting'] >= 1,
                        'prefill executed in original decode-only window')
                phase['open_prefill'] = await self.control(port, admit_prefill=True)
                phase['outputs'] = await asyncio.gather(first, second)
            elif pairing:
                phase['outputs'] = await asyncio.gather(*(self.task(self.generate(phase, s, port)) for s in slots))
            else:
                phase['outputs'] = [await self.generate(phase, s, port) for s in slots]
            phase['after'] = await self.wait_idle(port)
            phase['complete'] = True
        except BaseException as exc:
            phase['error'] = repr(exc)
            raise
        finally:
            # Owner writer is asynchronous. Preserve evidence even when equality or HTTP fails.
            await asyncio.sleep(.15)
            with timeline.open('rb') as f:
                f.seek(offset); content = f.read()
            (ROOT / (name + '.events.jsonl')).write_bytes(content)
            phase.update(event_offset_end=offset+len(content), event_sha256=hashlib.sha256(content).hexdigest(),
                         finished_s=time.time(), event_tail_complete=not content or content.endswith(b'\n'))
            self.save()

    async def identity(self, label):
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        for path, digest in manifest['frozen_inputs'].items():
            require(sha(path) == digest, 'frozen input changed: ' + path)
        release = json.loads((RELEASE / 'manifest.json').read_text())
        for path, digest in release['files'].items():
            require(sha(RELEASE / path) == digest, 'release bytes changed: ' + path)
        inspections = json.loads(await self.command('docker', 'inspect', *NAMES))
        records = []
        for i, obj in enumerate(inspections):
            require(obj['Image'] == IMAGE and obj['State']['Running'], 'engine image/container changed')
            env = {x.partition('=')[0]: x.partition('=')[2] for x in obj['Config']['Env'] if x.startswith('NCCL_')}
            require(env == json.loads((OLD / 'transport-environment.json').read_text())['nccl_environment'], 'TP environment changed')
            prov = await self.http(PORTS[i], '/provenance', label=label)
            expected = {str(RELEASE / p): s for p, s in release['files'].items() if p.startswith('src/ecopadg/serving/') and p.endswith('.py')}
            require(prov.get('source_files_at_import') == expected, 'imported source identity mismatch')
            require(prov.get('instance_id') == IDS[i] and prov.get('tp') == 2 and prov.get('model') == '/models/Qwen2.5-32B-Instruct', 'wrong model/TP/id')
            patches = json.loads((OLD / 'candidate-manifest.json').read_text())['image_patch_files']
            code = 'import hashlib,json,pathlib;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ' + repr(list(patches)) + '}))'
            observed = json.loads(await self.command('docker', 'exec', NAMES[i], 'python3', '-c', code))
            require(observed == patches, 'frozen image patch bytes changed')
            state = await self.runtime(PORTS[i], label)
            ack(state); require(idle(state) and state.get('accepting') is True, 'instance not idle/accepting')
            records.append(dict(container=obj, provenance=prov, runtime=state, nccl_environment=env, image_patch_sha256=observed))
        write(ROOT / ('identity.' + label + '.json'), records)
        return records

    async def work(self, hardware):
        self.state['identity_before'] = await self.identity('before')
        self.verified_for_controls = True
        self.state['verified_for_controls'] = True
        self.clocks = await asyncio.to_thread(ClockOwner, hardware, (0, 1, 2, 3))
        await self.clocks.set((0, 1, 2, 3), 2520, verify_rise=False)
        self.state['clock_policy'] = dict(gpus=[0, 1, 2, 3], target_mhz=2520, verify_rise=False)
        # The original 96/192-token bodies and all sampling keys are unchanged.
        for repeat in range(2):
            await self.phase('continuous-single-' + str(repeat), 'continuous', [0, 1])
        for repeat in range(2):
            await self.phase('continuous-pair-' + str(repeat), 'continuous', [0, 1], pairing=True)
        for repeat in range(2):
            await self.phase('temporal-original-pair-' + str(repeat), 'temporal', [0, 1], pairing=True)
        await self.phase('temporal-single', 'temporal', [0, 1])
        await self.phase('continuous-after', 'continuous', [0, 1])
        self.state['diagnostic_cases_complete'] = True

    async def restore_one(self, port):
        result = self.state.setdefault('cleanup', {}).setdefault(str(port), {})
        proof_error = None
        try:
            # Reopen in the same mode so a failed temporal hold cannot stall cleanup.
            result['admission_resume'] = await self.control(port, admit_prefill=True, admit_decode=True)
            before = await self.wait_idle(port, timeout=20)
            barrier = await self.http(port, '/drain', dict(expected_generation=before['generation']), label='native-final-drain', timeout=30)
            result['drain'] = barrier
            require(barrier.get('drained') is True and barrier.get('accepting') is False
                    and barrier.get('generation') == before['generation']+1
                    and barrier.get('drain_proof_type') == 'synchronous_put_owner_barrier', 'not actual native drain')
            ranks = barrier.get('transfers', [])
            require(len(ranks) == 2 and all(r.get('listener_alive') is True and not any(r.get(k) for k in
                ('buffered_tensors', 'inflight_receives', 'inflight_sends', 'buffered_gpu_bytes', 'allocations')) for r in ranks), 'rank transport residue')
            require(barrier.get('send_counters_verified') is True and all(r.get('send_counters_observed') is True
                and r.get('send_started') == r.get('send_completed') and r.get('send_failed') == 0 for r in ranks),
                'rank send proof incomplete')
        except BaseException as exc:
            proof_error = exc
            result['proof_error'] = repr(exc)
        finally:
            # A proof error must not strand an otherwise live engine behind /drain's admission barrier.
            try:
                current = await self.runtime(port, 'final-resume-current-generation')
                target = current['generation'] + 1
                await self.http(port, '/control', dict(generation=target, role='mixed', mode='continuous',
                    admit_prefill=True, admit_decode=True, scheduler_budget=BUDGET), label='final-8192-32-restore', timeout=30)
                r = result['restored'] = await self.wait_idle(port)
                ack(r, target)
                require(r['role'] == 'mixed' and r['mode'] == 'continuous' and r['admit_prefill'] and r['admit_decode']
                        and r.get('accepting') is True, 'not restored accepting mixed/continuous')
            except BaseException as exc:
                result['resume_error'] = repr(exc)
                raise
        if proof_error is not None:
            raise proof_error

    async def cleanup(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        # Cancel only this run's unresolved IDs; never stop either existing engine.
        async def cancel(port, rid):
            try:
                await self.http(port, '/cancel', dict(request_id=rid), label='owned-final-cancel', timeout=15)
                self.owned.discard((port, rid))
            except BaseException as exc:
                self.state.setdefault('cancel_errors', []).append(dict(port=port, request_id=rid, error=repr(exc)))
        await asyncio.gather(*(cancel(p, r) for p, r in list(self.owned)))
        results = await asyncio.gather(*(self.restore_one(p) for p in PORTS), return_exceptions=True)
        errors = [repr(x) for x in results if isinstance(x, BaseException)]
        require(not errors, 'native cleanup failed: ' + str(errors))
        self.state['cleanup_complete'] = True

    async def run(self):
        hardware = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)
        self.clocks = None
        sampler.start()
        start = end = None
        failure = None
        async with aiohttp.ClientSession(trust_env=False) as self.session:
            try:
                deadline = time.monotonic()+5
                while True:
                    rows = list(sampler.samples); meta = list(sampler.power_metadata[:len(rows)])
                    require(not sampler.error and time.monotonic() < deadline, 'eight-GPU instant sampling failed')
                    if len(rows) >= 2 and power_evidence(rows, sampler.power_source, meta)['power_source_verified']:
                        break
                    await asyncio.sleep(.02)
                start = self.state['measurement_start_s'] = time.time()
                await asyncio.wait_for(self.work(hardware), 390)
            except BaseException as exc:
                failure = exc
                self.state['error'] = repr(exc)
            finally:
                try:
                    if self.verified_for_controls:
                        await asyncio.wait_for(self.cleanup(), 90)
                        self.state['identity_after'] = await asyncio.wait_for(self.identity('after'), 20)
                except BaseException as exc:
                    self.state.update(cleanup_complete=False, incomplete_drain=True, cleanup_error=repr(exc))
                finally:
                    try:
                        if self.clocks is not None:
                            await asyncio.wait_for(self.clocks.close(), 15)
                    except BaseException as exc:
                        self.state['clock_cleanup_error'] = repr(exc)
                    end = self.state['measurement_end_s'] = time.time()
                    await asyncio.sleep(.15)
                    await asyncio.to_thread(sampler.stop)
                    destination = ROOT / 'power'; destination.mkdir()
                    save_raw(destination, [], sampler.samples, sampler.utilization_samples,
                        power_source=sampler.power_source, power_metadata=sampler.power_metadata)
                    with (destination / 'clocks.csv').open('w', newline='') as f:
                        w=csv.writer(f); w.writerow(['t_s'] + [f'gpu{i}_sm_mhz' for i in range(8)])
                        w.writerows([t] + list(v) for t, v in sampler.frequency_samples)
                    evidence = power_evidence(sampler.samples, sampler.power_source, sampler.power_metadata)
                    self.state.update(power_evidence=evidence, sampling_error=sampler.error)
                    try:
                        self.state['total_node_energy_j'] = trapezoid_energy(clip_power_window(sampler.samples, start, end, pad_s=0)) if start else None
                    except BaseException as exc:
                        self.state['integration_error'] = repr(exc)
                    self.state['measurement_valid'] = bool(start and evidence['power_source_verified'] and not sampler.error
                        and self.state.get('cleanup_complete') and not self.state.get('integration_error') and not self.state.get('clock_cleanup_error'))
                    self.state.update(complete=True, finished_s=time.time(), issued=self.issued, unresolved_owned_ids=list(self.owned))
                    self.save(); self.log.close()
        if isinstance(failure, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise failure


async def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    require(sha(ROOT / 'run.py') == manifest['run_sha256'], 'diagnostic script changed after review')
    task = asyncio.current_task()
    interrupted = False
    def stop():
        nonlocal interrupted
        if not interrupted:
            interrupted = True; task.cancel()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    await Diagnosis().run()


if __name__ == '__main__':
    with node_lease():
        asyncio.run(main())
