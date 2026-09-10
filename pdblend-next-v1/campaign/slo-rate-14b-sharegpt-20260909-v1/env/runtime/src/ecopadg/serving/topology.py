"""Serialized slow topology transactions within an explicit physical GPU budget.

A drained source retains CPU weight shards. Independent replacements prepare
concurrently; routing commits only after health, output and peer checks. No
request KV is migrated. GPU overlap requires old workers to stop first.
"""
import asyncio
from dataclasses import dataclass, asdict, replace
import json
import math
from pathlib import Path
import time
import uuid
import aiohttp

@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    tp: int
    gpus: tuple[int, ...]
    port: int
    kv_port: int
    role: str = 'mixed'
    generation: int = 0

    def endpoint(self):
        return dict(id=self.instance_id, tp=self.tp, gpus=list(self.gpus), role=self.role, url=f'http://127.0.0.1:{self.port}', port=self.port, kv_port=self.kv_port)

def validate_layout(specs, node_gpus):
    if len(set(node_gpus)) != len(node_gpus) or not set(node_gpus) <= set(range(8)):
        raise ValueError('node GPU budget must fit the authorized eight-card node')
    used = set()
    ports = set()
    ids = set()
    for spec in specs:
        if spec.tp not in (1, 2, 4, 8) or len(spec.gpus) != spec.tp or len(set(spec.gpus)) != spec.tp or (not set(spec.gpus) <= set(node_gpus)) or used.intersection(spec.gpus) or (spec.instance_id in ids) or (spec.role not in ('mixed', 'prefill', 'decode')) or (not spec.instance_id) or any((c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in spec.instance_id)):
            raise ValueError('invalid or overlapping physical instance allocation')
        own_ports = {spec.port, *range(spec.kv_port, spec.kv_port + spec.tp)}
        if len(own_ports) != spec.tp + 1 or ports.intersection(own_ports) or min(own_ports) < 1024 or (max(own_ports) > 65535):
            raise ValueError('overlapping or invalid instance ports')
        ports.update(own_ports)
        used.update(spec.gpus)
        ids.add(spec.instance_id)

def retained_cache_available(root):
    """Validate cache completeness before a pure addition has any side effect.

    Rank content hashes and model identity are checked by the existing loader
    before replacement validation/commit; no source instance is paused here.
    """
    if not root:
        return False
    try:
        root = Path(root).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        ranks = manifest['ranks']
        tp = manifest['tp']
        return bool(manifest.get('complete') and manifest.get('schema') == 1 and (tp in (1, 2, 4, 8)) and (len(ranks) == tp) and ({r['rank'] for r in ranks} == set(range(tp))) and manifest.get('model_config_sha256') and all(((root / r['file']).resolve().is_relative_to(root) and (root / r['file']).is_file() and ((root / r['file']).stat().st_size > 0) and (len(r['sha256']) == 64) for r in ranks)))
    except (OSError, ValueError, KeyError, TypeError):
        return False

class DockerLifecycle:

    def __init__(self, root, image, engine_template, *, prefix='pdb-v2-', parallelism=2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.template = dict(engine_template)
        self.prefix = prefix
        if prefix != 'pdb-v2-':
            raise ValueError('lifecycle only owns this experiment container prefix')
        self.slots = asyncio.Semaphore(parallelism)

    async def command(self, *args, timeout=180):
        async with self.slots:
            process = await asyncio.create_subprocess_exec('docker', *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                (output, _) = await asyncio.wait_for(process.communicate(), timeout)
            except BaseException:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            if process.returncode:
                raise RuntimeError(output.decode(errors='replace')[-4000:])
            return output.decode().strip()

    async def start(self, spec, peers, retained_weights=None):
        config = dict(self.template, id=spec.instance_id, tp=spec.tp, port=spec.port, kv_port=spec.kv_port, role=spec.role, initial_generation=spec.generation, peers=peers, retained_weights=retained_weights)
        path = self.root / (spec.instance_id + '.json')
        await asyncio.to_thread(path.write_text, json.dumps(config, indent=2))
        return await self.command('run', '-d', '--name', self.prefix + spec.instance_id, '--gpus', 'all', '--network', 'host', '--ipc', 'host', '-e', 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, spec.gpus)), '-e', 'NCCL_P2P_DISABLE=1', '-e', 'NCCL_SHM_DISABLE=1', '-e', 'NCCL_IB_DISABLE=1', '-e', 'NCCL_CUMEM_ENABLE=0', '-e', 'NCCL_DEBUG=WARN', '-e', 'PYTHONPATH=/root/workspace/pdblend/src', '-e', 'VLLM_HOST_IP=127.0.0.1', '-v', '/root/workspace:/root/workspace', '-v', '/root/workspace/models:/models:ro', self.image, 'python3', '-m', 'ecopadg.serving.engine', '--config', str(path))

    async def stop(self, spec):
        name = self.prefix + spec.instance_id
        try:
            await self.command('inspect', '--format', '{{.State.Running}}', name, timeout=10)
        except RuntimeError as exc:
            if 'No such object' in str(exc) or 'No such container' in str(exc):
                return
            raise
        logs = await self.command('logs', name, timeout=30)
        await asyncio.to_thread((self.root / (spec.instance_id + '.log')).write_text, logs)
        await self.command('rm', '-f', name, timeout=120)

class TopologyManager:

    def __init__(self, backend, lifecycle, specs, node_gpus, journal, *, freeze, commit):
        validate_layout(specs, node_gpus)
        (self.backend, self.lifecycle, self.journal) = (backend, lifecycle, journal)
        self.specs = {s.instance_id: s for s in specs}
        self.node_gpus = tuple(node_gpus)
        (self.freeze, self.commit) = (freeze, commit)
        self.lock = asyncio.Lock()
        self.version = 0

    async def request(self, spec, path, payload=None, timeout=60):
        kwargs = dict(timeout=aiohttp.ClientTimeout(total=timeout))
        if payload is not None:
            kwargs['json'] = payload
        method = self.backend.session.post if payload is not None else self.backend.session.get
        async with method(f'http://127.0.0.1:{spec.port}' + path, **kwargs) as response:
            if response.status != 200:
                raise RuntimeError(f'{spec.instance_id} {path}: {await response.text()}')
            return await response.json()

    async def ready(self, spec, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            try:
                state = await self.request(spec, '/runtime', timeout=1)
                if not state.get('error') and state.get('transport_healthy', True) and (state.get('generation') >= spec.generation) and (state.get('total_kv_tokens', 0) > 0):
                    return state
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f'{spec.instance_id} failed replacement health check')
            await asyncio.sleep(0.25)

    async def verify(self, spec):
        state = await self.ready(spec)
        generation = state['generation']
        if state['role'] != 'mixed':
            state = await self.request(spec, '/control', dict(generation=generation + 1, role='mixed', mode='continuous', admit_prefill=True))
        body = dict(prompt=[9707, 1879, 13] * 16, max_tokens=32, ignore_eos=True, temperature=0, stream=False)
        output = await self.request(spec, '/v1/completions', body)
        if output.get('usage', {}).get('completion_tokens') != 32 or len(output.get('token_ids', [])) != 32:
            raise RuntimeError('replacement output workload check failed')
        state = await self.request(spec, '/runtime')
        if state['role'] != spec.role:
            await self.request(spec, '/control', dict(generation=state['generation'] + 1, role=spec.role, mode='continuous', admit_prefill=True))
        return dict(instance_id=spec.instance_id, token_ids=output['token_ids'], validation='runtime output-count check; numerical cache validation is a separate gate')

    async def reconfigure(self, remove_ids, add_specs, *, savings_lower_j, cost_upper_j, retained_weights=None, capacity_recovery=None):
        recovery = capacity_recovery or {}
        capacity_ok = bool(recovery.get('source_sha256') and all((math.isfinite(recovery.get(k, float('nan'))) for k in ('current_rps', 'required_rps', 'target_rps'))) and (0 <= recovery['current_rps'] < recovery['required_rps'] <= recovery['target_rps']))
        if not all((math.isfinite(v) for v in (savings_lower_j, cost_upper_j))) or cost_upper_j < 0:
            raise ValueError('invalid switching energy')
        if savings_lower_j <= cost_upper_j and (not capacity_ok):
            raise ValueError('slow transaction does not amortize measured cost')
        async with self.lock:
            before = dict(self.specs)
            if len(set(remove_ids)) != len(remove_ids) or not set(remove_ids) <= set(before):
                raise ValueError('unknown source group')
            if not remove_ids and (not add_specs):
                raise ValueError('empty physical transition')
            if not remove_ids and (not await asyncio.to_thread(retained_cache_available, retained_weights)):
                raise ValueError('pure instance addition requires an available complete retained-weight cache')
            remove = [before[i] for i in remove_ids]
            untouched = [s for (i, s) in before.items() if i not in remove_ids]
            if set((s.instance_id for s in add_specs)) & set(before):
                raise ValueError('replacement needs fresh versioned IDs for new NCCL handles')
            validate_layout(untouched + list(add_specs), self.node_gpus)
            if not untouched and (not add_specs):
                raise ValueError('cannot remove every serving instance')
            transaction = uuid.uuid4().hex
            started = time.time()
            stopped = []
            created = []
            safe_to_resume = True
            await self.freeze(remove_ids, True)
            await self.journal.emit(dict(kind='topology_begin', transaction=transaction, at_s=started, before=[asdict(s) for s in remove], after=[asdict(s) for s in add_specs], savings_lower_j=savings_lower_j, cost_upper_j=cost_upper_j, capacity_recovery=capacity_recovery))
            try:
                if self.backend.clocks and add_specs:
                    await self.backend.clocks.set(sorted({g for s in add_specs for g in s.gpus}), getattr(self.backend, 'max_service_frequency_mhz', 2520))
                for spec in remove:
                    raw = await self.request(spec, '/runtime')
                    await self.request(spec, '/drain', dict(expected_generation=raw['generation']))
                if retained_weights is None:
                    source = remove[0]
                    raw = await self.request(source, '/runtime')
                    result = await self.request(source, '/retain-weights', dict(transaction=transaction, expected_generation=raw['generation']), timeout=600)
                    retained_weights = result['retained_weights']
                for spec in remove:
                    stopped.append(spec)
                    await self.lifecycle.stop(spec)
                peers = {s.instance_id: dict(host='127.0.0.1', tp=s.tp, kv_port=s.kv_port) for s in untouched + list(add_specs)}

                async def prepare(spec):
                    created.append(spec)
                    await self.lifecycle.start(spec, peers, retained_weights)
                    return await self.verify(spec)
                outcomes = await asyncio.gather(*(prepare(s) for s in add_specs), return_exceptions=True)
                errors = [x for x in outcomes if isinstance(x, BaseException)]
                if errors:
                    raise RuntimeError(str(errors[0]))
                for old in untouched:
                    for new in add_specs:
                        await self.request(old, '/register-peer', dict(id=new.instance_id, peer=peers[new.instance_id]))
                for new in add_specs:
                    targets = [i for i in peers if i != new.instance_id]
                    if targets:
                        await self.request(new, '/prepare-peers', dict(peers=targets))
                await self.commit(remove_ids, tuple(add_specs))
                self.specs = {s.instance_id: s for s in untouched + list(add_specs)}
                self.version += 1
                result = dict(transaction=transaction, version=self.version, committed=True, duration_s=time.time() - started, retained_weights=retained_weights, validation=outcomes, live_instances=[asdict(s) for s in self.specs.values()])
                await self.journal.emit(dict(kind='topology_commit', at_s=time.time(), **result))
                return result
            except BaseException as exc:
                await self.journal.emit(dict(kind='topology_failure', transaction=transaction, at_s=time.time(), error=repr(exc), stopped=[s.instance_id for s in stopped]))
                recovery = asyncio.create_task(self.rollback(before, remove, stopped, created, retained_weights, transaction))
                safe_to_resume = False
                try:
                    try:
                        recovered = await asyncio.shield(recovery)
                    except asyncio.CancelledError:
                        recovered = await recovery
                except BaseException as restore_error:
                    await self.journal.emit(dict(kind='topology_recovery_failed', transaction=transaction, at_s=time.time(), error=repr(restore_error), frozen_instances=list(remove_ids)))
                    raise
                safe_to_resume = True
                await self.journal.emit(dict(kind='topology_rollback', transaction=transaction, at_s=time.time(), error=str(exc), recovered=recovered, duration_s=time.time() - started, live_instances=[asdict(s) for s in self.specs.values()]))
                raise
            finally:
                if safe_to_resume:
                    await self.freeze(remove_ids, False)

    async def rollback(self, before, remove, stopped, created, retained_weights, transaction):
        for spec in created:
            try:
                await self.lifecycle.stop(spec)
            except Exception:
                await self.lifecycle.stop(spec)
        for spec in stopped:
            await self.lifecycle.stop(spec)
        if not stopped:
            published = tuple((s.instance_id for s in created if s.instance_id in getattr(self.backend, 'instances', {})))
            if published:
                await self.commit(published, ())
            for spec in remove:
                state = await self.request(spec, '/runtime')
                await self.request(spec, '/control', dict(generation=state['generation'] + 1, role=spec.role, mode='continuous', admit_prefill=True))
            self.specs = dict(before)
            if published:
                self.version += 1
            return True
        restored = [replace(s, instance_id=s.instance_id + 'r' + transaction[:8], port=s.port + 200, kv_port=s.kv_port + 200, generation=s.generation + 2) for s in stopped]
        survivors = [s for s in before.values() if s.instance_id not in {x.instance_id for x in stopped}]
        validate_layout(survivors + restored, self.node_gpus)
        peers = {s.instance_id: dict(host='127.0.0.1', tp=s.tp, kv_port=s.kv_port) for s in survivors + restored}
        if self.backend.clocks:
            await self.backend.clocks.set(sorted({g for s in restored for g in s.gpus}), getattr(self.backend, 'max_service_frequency_mhz', 2520))
        for spec in restored:
            await self.lifecycle.start(spec, peers, retained_weights)
            await self.verify(spec)
        for old in survivors:
            for new in restored:
                await self.request(old, '/register-peer', dict(id=new.instance_id, peer=peers[new.instance_id]))
        for spec in restored:
            await self.request(spec, '/prepare-peers', dict(peers=[i for i in peers if i != spec.instance_id]))
        route_ids = tuple((s.instance_id for s in stopped + created if s.instance_id in self.backend.instances)) if hasattr(self.backend, 'instances') else tuple((s.instance_id for s in stopped))
        await self.commit(route_ids, tuple(restored))
        for spec in remove:
            if spec not in stopped:
                state = await self.request(spec, '/runtime')
                await self.request(spec, '/control', dict(generation=state['generation'] + 1, role=spec.role, mode='continuous', admit_prefill=True))
        self.specs = {s.instance_id: s for s in survivors + restored}
        self.version += 1
        return True
