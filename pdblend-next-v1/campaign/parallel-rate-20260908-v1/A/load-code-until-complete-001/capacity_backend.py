"""Pinned independent mixed engines and eight-GPU physical transition measurements."""
import asyncio
import copy
import json
from pathlib import Path
import socket
import threading
import time
import uuid

from capacity_executor import check_lease, durable, fixed, require, sha


def cancel_transfer_rows(rows, tp):
    """The frozen collective returns one unlabeled record per TP worker."""
    require(isinstance(rows, list) and len(rows) == tp, 'all native TP worker replies required')
    for row in rows:
        require(all(type(row.get(k)) is int and row[k] == 0 for k in
                ('buffered_tensors', 'buffered_gpu_bytes', 'inflight_receives', 'inflight_sends', 'send_failed'))
                and row.get('listener_alive') is True and row.get('send_counters_observed') is True
                and row.get('send_healthy') is True and isinstance(row.get('allocations'), dict)
                and not row['allocations'] and type(row.get('send_started')) is int and row['send_started'] >= 0
                and type(row.get('send_completed')) is int and row['send_started'] == row['send_completed'],
                'actual native worker cancellation residue or missing sender evidence')


class TransitionMeter:
    def __init__(self, out):
        self.out = Path(out)

    async def start(self):
        from ecopadg.measure.backends import PynvmlBackend
        from ecopadg.measure.power import PowerSampler
        backend = await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        self.memory = []
        self.memory_error = None
        self.memory_stop = threading.Event()
        def memory_observer():
            try:
                while not self.memory_stop.is_set():
                    for gpu in range(8):
                        value = backend._nvml.nvmlDeviceGetMemoryInfo(backend._handle(gpu))
                        self.memory.append(dict(at_s=time.time(), gpu=gpu, used_bytes=int(value.used)))
                    self.memory_stop.wait(.1)
            except Exception as exc:
                self.memory_error = repr(exc)
        # Memory queries must not delay the eight instantaneous readings within
        # a power row. Keep the original, already validated PowerSampler path.
        self.memory_thread = threading.Thread(target=memory_observer, daemon=True)
        self.memory_thread.start()
        self.sampler = PowerSampler(range(8), interval=.02, backend=backend, sample_clocks=True)
        self.sampler.start()
        deadline = time.monotonic() + 3
        try:
            while len(self.sampler.samples) < 2:
                require(not self.sampler.error and time.monotonic() < deadline, 'transition sampler did not start')
                await asyncio.sleep(.02)
        except BaseException:
            await asyncio.to_thread(self.sampler.stop)
            self.memory_stop.set()
            await asyncio.to_thread(self.memory_thread.join, 2)
            raise
        self.started_s = time.time()
        return self

    async def finish(self):
        finished_s = time.time()
        interrupted = None
        bracket_deadline = time.monotonic() + 3
        try:
            while not self.sampler.samples or self.sampler.samples[-1][0] < finished_s:
                if self.sampler.error or time.monotonic() >= bracket_deadline:
                    break
                await asyncio.sleep(.02)
        except BaseException as exc:
            interrupted = exc

        # A separately owned thread cannot itself be cancelled by asyncio's
        # task shutdown. Keep its result until the raw writer has finished.
        outcome = {}
        written = threading.Event()
        def collect():
            try:
                outcome['result'] = self._finish_snapshot(finished_s)
            except BaseException as exc:
                outcome['error'] = exc
            finally:
                written.set()
        cleanup = threading.Thread(target=collect, daemon=True)
        cleanup.start()
        while not written.is_set():
            try:
                await asyncio.sleep(.01)
            except asyncio.CancelledError as exc:
                if interrupted is None:
                    interrupted = exc
        cleanup.join(timeout=.1)
        if 'error' in outcome:
            raise outcome['error']
        result = outcome['result']
        result['measurement_cancelled'] = isinstance(interrupted, asyncio.CancelledError)
        result['finish_interrupted'] = repr(interrupted) if interrupted is not None else None
        if interrupted is not None:
            result['measurement_valid'] = False
        durable(self.out / 'measurement.json', result)
        if interrupted is not None:
            raise interrupted
        return dict(result, receipt=dict(path=str(self.out / 'measurement.json'),
                    sha256=sha(self.out / 'measurement.json')))

    def _finish_snapshot(self, finished_s):
        from ecopadg.measure.power import trapezoid_energy
        from ecopadg.metrics import clip_power_window
        from ecopadg.serving.measurement import save_raw, power_evidence

        # The pinned sampler joins for two seconds but clears its own handle
        # even if a native read is still blocked. Retain that handle to report
        # actual observer termination instead of silently assuming it.
        power_thread = getattr(self.sampler, '_thread', None)
        stop_error = None
        try:
            self.sampler.stop()
        except Exception as exc:
            stop_error = repr(exc)
        power_alive = bool(power_thread is not None and power_thread.is_alive())
        if hasattr(self, 'memory_stop'):
            self.memory_stop.set()
            self.memory_thread.join(2)
            if self.memory_thread.is_alive():
                self.memory_error = 'memory observer did not stop within two seconds'
        memory_alive = bool(hasattr(self, 'memory_thread') and self.memory_thread.is_alive())

        # A stuck observer makes this invalid. Still copy its observed prefix
        # once, so late appends cannot mutate a written evidence set.
        samples = copy.deepcopy(self.sampler.samples)
        utilization = copy.deepcopy(self.sampler.utilization_samples)
        source = copy.deepcopy(self.sampler.power_source)
        metadata = copy.deepcopy(self.sampler.power_metadata)
        clocks = copy.deepcopy(self.sampler.frequency_samples)
        memory = copy.deepcopy(self.memory)
        self.out.mkdir(parents=True, exist_ok=False)
        save_raw(self.out, [], samples, utilization, power_source=source, power_metadata=metadata)
        durable(self.out / 'memory.json', memory)
        durable(self.out / 'clocks.json', clocks)
        try:
            evidence = power_evidence(samples, source, metadata)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            evidence = dict(power_source_verified=False, error=repr(exc))
        window_error = None
        try:
            if len(samples) < 2:
                raise ValueError('fewer than two power samples: energy unknown')
            energy = trapezoid_energy(clip_power_window(samples, self.started_s, finished_s, pad_s=0.))
        except ValueError as exc:
            energy = None
            window_error = repr(exc)
        peaks = {str(g): max((v['used_bytes'] for v in memory if v['gpu'] == g), default=None)
                 for g in range(8)}
        memory_error = getattr(self, 'memory_error', None)
        memory_complete = all(value is not None for value in peaks.values())
        return dict(measurement_valid=not self.sampler.error and not stop_error and not power_alive
                    and not memory_alive and not window_error and not memory_error and memory_complete
                    and evidence['power_source_verified'],
                    measurement_start_s=self.started_s, measurement_end_s=finished_s,
                    energy_j=energy, duration_s=finished_s-self.started_s,
                    gpu_indices=list(range(8)), power_evidence=evidence,
                    sampling_error=self.sampler.error, sampling_stop_error=stop_error,
                    power_observer_stopped=not power_alive,
                    memory_observer_stopped=not memory_alive,
                    window_error=window_error, memory_sampling_error=memory_error,
                    memory_sampling_complete=memory_complete, memory_sampling_interval_s=.1,
                    clock_samples_observed=len(clocks), peak_memory_per_gpu_bytes=peaks,
                    scope='whole-node transition; includes concurrent service; diagnostic overlap, not an extra serving-energy term',
                    artifacts={str(p):sha(p) for p in self.out.iterdir() if p.is_file()})


class PinnedDockerBackend:
    """Only creates this cell's versioned IDs; retains every stopped container.

    Mixed-only replicas have only their own peer identity. This retains actual
    local transport health/cancel observations without modifying existing engines
    or establishing cross-instance PD channels.
"""
    def __init__(self, controller, binding, inventory):
        self.controller, self.binding, self.inventory = controller, binding, inventory
        self.root = Path(binding['runtime_dir'])
        self.tp = binding['identity']['tp']
        self.owner = binding['owner_id']
        require(self.tp in (1, 2) and self.owner and self.owner.replace('-', '').isalnum(), 'invalid physical owner/TP')
        require(binding['native_kind'] in ('v3', 'legacy_sync_put'), 'native engine kind must be explicit')
        require(binding['independent_mixed_only'] is True, 'independent mixed engine protocol required')
        self.template = fixed(binding['engine_template'])
        self.service_budget = binding['service_budget_tokens']
        self.restore_budget = binding['restore_budget_tokens']
        require(self.service_budget in (2048, 8192) and self.restore_budget in (2048, 8192),
                'original per-model service/restore budgets must be explicit')
        self.oracles = fixed(binding['correctness_oracles'])
        require(self.oracles.get('schema') == 'capacity-correctness-oracles-v1'
                and self.oracles.get('model_source_identity') == self.expected_oracle_identity()
                and self.oracles.get('measured') is True, 'measured same-source numerical oracle required')
        require({c['prompt_length'] for c in self.oracles['cases']} == {128, 7168}
                and all(len(c['prompt']) == c['prompt_length'] and len(c['token_ids']) == c['max_tokens'] == 64
                        for c in self.oracles['cases']), 'short and long full64 numerical cases required')
        self.inputs = binding['files']
        require(self.inputs and all(sha(p) == h for p, h in self.inputs.items()), 'engine package changed')
        require(binding['engine_entry'] in self.inputs and binding['image'] == binding['identity']['engine_image'],
                'current engine entry/image must be pinned')
        require(binding['engine_module'] == 'ecopadg.serving.engine'
                and str(Path(binding['engine_pythonpath'])/'ecopadg/serving/engine.py') == binding['engine_entry'],
                'original module invocation/source must be explicit; direct script shadows stdlib types')
        self.counter = 0
        self.endpoint_timeout_s = 35.

    def expected_oracle_identity(self):
        return self.binding['identity']

    def engine_launch(self, instance, config_path, env):
        return ['python3', '-m', self.binding['engine_module'], '--config', str(config_path)], env

    def native_clean(self, instance, raw):
        from ecopadg.serving.completion_policy import engine_residual
        return not engine_residual(raw, time.time())

    def allocate(self, gpus):
        require(len(gpus) == self.tp and tuple(sorted(gpus)) == tuple(gpus)
                and len(set(gpus)) == len(gpus) and set(gpus) <= set(range(8)), 'physical TP allocation invalid')
        self.counter += 1
        require(self.counter <= self.binding.get('max_creations', 32), 'bounded creation inventory exhausted')
        suffix = uuid.uuid4().hex[:12]
        iid = f'cap-{self.owner}-{suffix}'
        port = self.binding['http_port_base'] + self.counter
        kv_port = self.binding['kv_port_base'] + 32*self.counter
        require(1024 <= port < 65000 and 1024 <= kv_port <= 65000-32, 'port domain exhausted')
        for own_port in (port, *range(kv_port, kv_port+32)):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', own_port))
        config = dict(copy.deepcopy(self.template), id=iid, tp=self.tp, port=port, kv_port=kv_port,
                      role='mixed', peers={iid:dict(host='127.0.0.1',tp=self.tp,kv_port=kv_port)}, retained_weights=None,
                      scheduler_budget=dict(schema_version=1, max_num_batched_tokens=self.service_budget,
                                            max_num_seqs=self.template.get('max_num_seqs', 32)),
                      runtime_dir=str(self.root / iid / 'owner'))
        require(config.get('max_model_len') == 8192, 'ordinary engine model limit differs')
        config_path = self.root / iid / 'engine.json'
        return dict(id=iid, tp=self.tp, gpus=list(gpus), port=port, kv_port=kv_port,
                    url=f'http://127.0.0.1:{port}', role='mixed', mode='continuous',
                    native_kind=self.binding['native_kind'], container_name='pdb-v2-'+iid,
                    engine_config=str(config_path), planned_config=config, owner_id=self.owner,
                    service_budget_tokens=self.service_budget, restore_budget_tokens=self.restore_budget,
                    scheduler_cache_observed=True, scheduler_cache_count=self.binding.get('scheduler_cache_count', 1))

    async def command(self, args, limit, cap=30):
        remaining = min(cap, limit-time.time())
        require(remaining > 0, 'physical command deadline expired')
        child = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(child.communicate(), remaining)
        except BaseException:
            if child.returncode is None:
                child.kill()
                await child.wait()
            raise
        record = dict(argv=args, at_s=time.time(), exitcode=child.returncode,
                      stdout=stdout.decode(errors='replace'), stderr=stderr.decode(errors='replace'))
        self.inventory.event('physical_command', **record)
        require(child.returncode == 0, 'physical command failed: '+record['stderr'][-2000:])
        return record['stdout']

    async def gpu_state(self, gpus):
        from ecopadg.measure.backends import PynvmlBackend
        def inspect():
            hw = PynvmlBackend(power_mode='instant')
            rows = []
            for g in gpus:
                h = hw._handle(g)
                mem = hw._nvml.nvmlDeviceGetMemoryInfo(h)
                procs = hw._nvml.nvmlDeviceGetComputeRunningProcesses(h)
                rows.append(dict(gpu=g, at_s=time.time(), free_bytes=int(mem.free),
                                 used_bytes=int(mem.used), process_pids=[int(p.pid) for p in procs]))
            return rows
        return await asyncio.to_thread(inspect)

    async def assert_spare(self, gpus):
        rows = await self.gpu_state(gpus)
        require(all(not row['process_pids'] for row in rows), 'unowned GPU process blocks restore')
        return rows

    async def inspect(self, instance, limit):
        rows = json.loads(await self.command(['docker', 'inspect', instance['container_name']], limit, 10))
        require(len(rows) == 1, 'one exact owned container required')
        row = rows[0]
        require(row['Name'].lstrip('/') == instance['container_name']
                and row['Image'] == self.binding['image']
                and row['Config'].get('Labels', {}).get('pdblend.capacity.owner') == self.owner,
                'created container ownership/image differs')
        if instance.get('container', {}).get('id'):
            require(row['Id'] == instance['container']['id'], 'container ID changed')
        return row

    async def clock_ownership_proof(self, instance, operation):
        """Authorize a clock write only for a freshly empty owned transition GPU."""
        require(operation in ('bootstrap', 'release'), 'unknown capacity clock operation')
        config = self.controller.config
        check_lease(authority=config.get('capacity_lease_authority'),
                    expected_inventory=config['capacity_inventory_path'],
                    expected_job_path=config.get('capacity_job_path'))
        inventory = self.inventory.value
        known = inventory['known_instances'].get(instance['id'], {})
        allowed = ('start_intent',) if operation == 'bootstrap' else (
            'stop_intent', 'start_intent', 'started_unpublished', 'ready_unpublished')
        require(inventory['transition_inflight'] is True
                and instance['id'] not in inventory['initial_ids']
                and known.get('owner_kind') == 'created_for_cell'
                and known.get('state') in allowed and known.get('transaction')
                and known.get('gpus') == instance['gpus'], 'exact owned transition clock intent required')
        routed = self.controller.backend.instances.values()
        require(not set(instance['gpus']) & {g for item in routed for g in item['gpus']},
                'clock bootstrap/release GPU remains published')
        rows = await self.assert_spare(instance['gpus'])
        require({row['gpu'] for row in rows} == set(instance['gpus'])
                and len(rows) == len(instance['gpus'])
                and all(0 <= time.time()-row['at_s'] <= 1. for row in rows),
                'fresh complete empty GPU evidence required')
        self.inventory.event('clock_'+operation+'_proof', instance_id=instance['id'],
            transaction=known['transaction'], gpus=instance['gpus'], rows=rows)
        return dict(schema='capacity-clock-'+operation+'-v1', instance_id=instance['id'],
                    transaction=known['transaction'], inventory_path=str(self.inventory.path),
                    inventory_sha256=sha(self.inventory.path), gpus=instance['gpus'], rows=rows)

    async def park_stopped(self, instance):
        clocks = self.controller.backend.clocks
        if clocks:
            epochs = dict(clocks.epochs)
            if self.controller.config.get('measured_frequency_write_guard_v1') is True:
                proof = await self.clock_ownership_proof(instance, 'release')
                await clocks.park(instance['gpus'], epochs, bootstrap=proof)
            else:
                await clocks.park(instance['gpus'], epochs)

    async def start(self, instance, before, limit):
        require(all(sha(p) == h for p, h in self.inputs.items()), 'engine source changed before create')
        path = Path(instance['engine_config'])
        require(not path.exists(), 'fresh versioned engine configuration required')
        durable(path, instance['planned_config'])
        self.inventory.event('engine_config_written', instance_id=instance['id'], path=str(path), sha256=sha(path))
        env = dict(self.binding['environment'])
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, instance['gpus']))
        require(env['PYTHONPATH'] == self.binding['engine_pythonpath'], 'engine import source differs')
        engine_argv, env = self.engine_launch(instance, path, env)
        require(engine_argv and all(type(v) is str for v in engine_argv)
                and isinstance(env, dict), 'explicit actual engine command/environment required')
        args = ['docker', 'run', '-d', '--name', instance['container_name'], '--gpus', 'all',
                '--network', 'host', '--ipc', 'host', '--label', 'pdblend.capacity.owner='+self.owner]
        for option in self.binding.get('security_options', ['label=disable']):
            args += ['--security-opt', option]
        for key, value in env.items():
            args += ['-e', key+'='+value]
        args += ['-v', '/root/workspace:/root/workspace', '-v', '/root/workspace/models:/models:ro',
                 self.binding['image']] + engine_argv
        if self.controller.backend.clocks:
            if self.controller.config.get('measured_frequency_write_guard_v1') is True:
                proof = await self.clock_ownership_proof(instance, 'bootstrap')
                await self.controller.backend.clocks.set(instance['gpus'], 2520, verify_rise=False, bootstrap=proof)
            else:
                await self.controller.backend.clocks.set(instance['gpus'], 2520)
        cid = (await self.command(args, limit, 30)).strip()
        row = await self.inspect(instance, limit)
        require(row['Id'] == cid and row['State']['Running'] and row['State']['Pid'] > 0,
                'actual new physical process absent')
        require(row['Config']['Cmd'] == engine_argv
                and row['HostConfig']['NetworkMode'] == 'host' and row['HostConfig']['IpcMode'] == 'host'
                and row['HostConfig'].get('AutoRemove') is False, 'actual creation contract differs')
        actual_env = dict(value.split('=', 1) for value in row['Config']['Env'])
        require(all(actual_env.get(k) == v for k, v in env.items()), 'actual environment differs')
        return dict(container=dict(id=cid, name=instance['container_name'], image=row['Image'],
                                   StartedAt=row['State']['StartedAt'], host_pid=row['State']['Pid']),
                    host_pid=row['State']['Pid'], config_sha256=sha(path))

    async def request(self, instance, path, payload=None, *, limit, cap=35):
        import aiohttp
        left = min(cap, limit-time.time())
        require(left > 0, 'native operation deadline expired')
        kwargs = dict(timeout=aiohttp.ClientTimeout(total=left))
        if payload is not None:
            kwargs['json'] = payload
            if payload.get('request_id'):
                kwargs['headers'] = {'X-Request-Id': payload['request_id']}
        call = self.controller.session.post if payload is not None else self.controller.session.get
        async with call(instance['url']+path, **kwargs) as response:
            require(response.status == 200, f"{instance['id']} {path} status={response.status}")
            result = await response.json()
        self.inventory.event('native_reply', instance_id=instance['id'], path=path, reply=result)
        return result

    async def ready(self, instance, limit):
        import aiohttp
        while time.time() < limit:
            try:
                raw = await self.request(instance, '/runtime', limit=limit, cap=1)
                if raw.get('transport_healthy') is True and not raw.get('error') and raw.get('total_kv_tokens', 0) > 0:
                    return raw
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                pass
            await asyncio.sleep(.2)
        raise TimeoutError('new replica did not become ready')

    async def verify(self, instance, limit):
        from ecopadg.serving.completion_policy import engine_residual
        raw = await self.ready(instance, limit)
        provenance = await self.request(instance, '/provenance', limit=limit)
        expected = dict(self.binding['expected_provenance'], instance_id=instance['id'],
                        cuda_visible_devices=','.join(map(str, instance['gpus'])), tp=instance['tp'])
        require(all(provenance.get(k) == v for k, v in expected.items()), 'actual imported source/model/TP differs')
        require(raw.get('role') == 'mixed' and raw.get('mode') == 'continuous', 'new replica role differs')
        observed_budget = raw.get('scheduler_budget_effective')
        if observed_budget is not None:
            require(observed_budget.get('max_num_batched_tokens') == self.service_budget,
                    'actual native per-model service budget differs')
        else:
            require(self.binding['native_kind'] == 'legacy_sync_put'
                    and self.binding.get('budget_evidence_mode') == 'static_engine_initialization'
                    and self.service_budget == self.restore_budget == self.template.get('max_num_batched_tokens') == 8192
                    and self.template.get('max_num_seqs') == 32,
                    'legacy missing dynamic budget ACK only supports verified static8192/32 initialization')
        replies = []
        for case in self.oracles['cases']:
            request_id = 'capacity-verify-'+uuid.uuid4().hex
            body = dict(prompt=case['prompt'], max_tokens=case['max_tokens'], ignore_eos=True,
                        temperature=0, stream=False, request_id=request_id)
            self.inventory.event('correctness_request_intent', instance_id=instance['id'], request_id=request_id)
            output = await self.request(instance, '/v1/completions', body, limit=limit, cap=60)
            require(output.get('token_ids') == case['token_ids']
                    and output.get('usage', {}).get('prompt_tokens') == case['prompt_length']
                    and output.get('usage', {}).get('completion_tokens') == case['max_tokens'],
                    'new replica numerical output/original work differs')
            replies.append(output)
        cancellation = await self.verify_cancel(instance, limit)
        raw = await self.request(instance, '/runtime', limit=limit)
        require(self.native_clean(instance, raw), 'new replica native resources not clean')
        return dict(provenance=provenance, numerical_correctness=True, ordinary_replies=replies,
                    runtime=raw, cancellation=cancellation,
                    independent_mixed_only=True, peer_reconfiguration_performed=False)

    async def verify_cancel(self, instance, limit):
        from ecopadg.serving.completion_policy import engine_residual
        import aiohttp
        request_id = 'capacity-cancel-'+uuid.uuid4().hex
        case = next(c for c in self.oracles['cases'] if c['prompt_length'] == 128)
        body = dict(prompt=case['prompt'], max_tokens=512, ignore_eos=True,
                    temperature=0, stream=False)
        self.inventory.event('correctness_request_intent', instance_id=instance['id'],
                             request_id=request_id, controlled_cancel=True)
        async def issue():
            async with self.controller.session.post(instance['url']+'/v1/completions', json=body,
                    headers={'X-Request-Id':request_id},
                    timeout=aiohttp.ClientTimeout(total=max(.1, limit-time.time()))) as response:
                return dict(status=response.status, body=await response.text(), finished_s=time.time())
        child = asyncio.create_task(issue())
        cancelled = False
        try:
            while time.time() < limit:
                raw = await self.request(instance, '/runtime', limit=limit, cap=1)
                if raw.get('active') and raw.get('running') and raw.get('kv_allocations', {}).get(request_id, 0) > 0:
                    break
                require(not child.done(), 'controlled request finished before active KV proof')
                await asyncio.sleep(.05)
            else:
                raise TimeoutError('controlled cancellation never observed own running KV')
            reply = await self.request(instance, '/cancel', dict(request_id=request_id), limit=limit)
            cancelled = True
            require(reply.get('cancelled') == request_id, 'cancellation acknowledgement differs')
            ranks = reply.get('transfers', [])
            cancel_transfer_rows(ranks, instance['tp'])
            response = await asyncio.wait_for(child, max(.1, limit-time.time()))
            require(response['status'] >= 400 and 'cancel' in response['body'].lower(),
                    'controlled request did not terminate as cancellation')
            while time.time() < limit:
                settled = await self.request(instance, '/runtime', limit=limit, cap=1)
                if not engine_residual(settled, time.time()):
                    return dict(request_id=request_id, before=raw, cancelled=reply,
                                response=response, settled=settled, verified=True)
                await asyncio.sleep(.05)
            raise TimeoutError('controlled cancellation did not release all native resources')
        finally:
            if not cancelled:
                try:
                    await self.request(instance, '/cancel', dict(request_id=request_id), limit=time.time()+10)
                except Exception as exc:
                    self.inventory.event('cancel_cleanup_error', instance_id=instance['id'],
                                         request_id=request_id, error=repr(exc))
            if not child.done():
                child.cancel()
            await asyncio.gather(child, return_exceptions=True)

    async def drain_idle(self, instance, limit):
        from ecopadg.serving.completion_policy import engine_residual
        raw = await self.request(instance, '/runtime', limit=limit)
        require(not engine_residual(raw, time.time()), 'nonempty/stale native engine cannot be stopped')
        await self.request(instance, '/drain', dict(expected_generation=raw['generation']), limit=limit)
        after = await self.request(instance, '/runtime', limit=limit)
        require(not engine_residual(after, time.time()), 'drain did not prove empty native engine')

    async def resume_idle(self, instance, limit):
        raw = await self.request(instance, '/runtime', limit=limit)
        await self.request(instance, '/control', dict(generation=raw['generation']+1,
                           role='mixed', mode='continuous', admit_prefill=True, admit_decode=True), limit=limit)

    async def stop(self, instance, limit):
        row = await self.inspect(instance, limit)
        if row['State']['Running']:
            await self.command(['docker', 'stop', '--time', '10', row['Id']], limit, 25)
        after = await self.inspect(instance, limit)
        require(not after['State']['Running'] and not after['State']['Pid'], 'physical stop unconfirmed')
        await self.assert_spare(instance['gpus'])
        await self.park_stopped(instance)

    async def stop_if_owned(self, instance, limit):
        try:
            await self.stop(instance, limit)
        except ValueError as exc:
            # A failed Docker create may have left no object at all. Only the
            # exact missing-container result plus empty GPU evidence is success.
            if 'No such object' not in str(exc) and 'No such container' not in str(exc):
                raise
            await self.assert_spare(instance['gpus'])
            await self.park_stopped(instance)

    async def is_same_running(self, instance):
        try:
            row = await self.inspect(instance, time.time()+10)
            original = instance.get('container', {})
            return bool(row['State']['Running'] and row['State']['StartedAt'] == original.get('StartedAt')
                        and row['State']['Pid'] == original.get('host_pid'))
        except (ValueError, TimeoutError):
            return False

    async def begin_measurement(self, transaction):
        return await TransitionMeter(self.root / 'transitions' / transaction).start()

    async def end_measurement(self, meter):
        return await meter.finish()
