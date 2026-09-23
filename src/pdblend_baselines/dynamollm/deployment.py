"""Dynamo-owned V1 subprocess lifecycle within an externally leased GPU group."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

PROCESS_GROUP_GRACE_S = 10

def _proc_stat_entries():
    """Return (pid, state, pgrp) for readable process stat files.

    A disappearing process is a normal /proc race.  Permission and malformed
    entries are deliberately errors: cleanup must not claim an owned group is
    gone when it could not inspect it.
    """
    entries = []
    try:
        proc_entries = list(Path('/proc').iterdir())
    except OSError as exc:
        raise RuntimeError('cannot enumerate /proc for process cleanup') from exc
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / 'stat').read_text()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError('cannot read process stat for '+entry.name) from exc
        close = raw.rfind(')')
        if close < 0:
            raise RuntimeError('malformed process stat for '+entry.name)
        fields = raw[close + 2:].split()
        # fields[0]=state, fields[2]=pgrp after the comm field.
        if len(fields) < 3:
            raise RuntimeError('short process stat for '+entry.name)
        try:
            entries.append((int(entry.name), fields[0], int(fields[2])))
        except (TypeError, ValueError) as exc:
            raise RuntimeError('invalid process stat for '+entry.name) from exc
    return entries


def proc_group_state(pgid):
    """Inspect one owned process group, distinguishing live and zombie PIDs."""
    if type(pgid) is not int or pgid <= 0:
        raise ValueError('positive process-group id required')
    members = [dict(pid=pid, state=state, pgid=group)
               for pid, state, group in _proc_stat_entries() if group == pgid]
    live = [row for row in members if row['state'] != 'Z']
    zombies = [row for row in members if row['state'] == 'Z']
    return dict(pgid=pgid, members=members, live_members=live, zombie_members=zombies,
                no_live_members=not live, all_pid_gone=not members)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_devices(gpus):
    """Resolve lease-local ordinals without accidentally using host ordinals."""
    leased = os.environ.get('PDBLEND_GPU_UUIDS', '').split(',')
    leased = [value.strip() for value in leased if value.strip()]
    if len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError('unique nonnegative local GPU indices required')
    if leased:
        if len(set(leased)) != len(leased) or any(g >= len(leased) for g in gpus):
            raise ValueError('GPU index outside physical UUID lease')
        return [leased[g] for g in gpus]
    return [str(g) for g in gpus]


class SubprocessLifecycle:
    """Only stop process groups created by this object, never global vLLM jobs."""

    def __init__(self, config, transport, journal, output):
        self.config, self.transport, self.journal = config, transport, journal
        self.output = Path(output)
        self.instances, self.processes, self.logs = {}, {}, {}
        self.base_port = int(config.get('base_port', 16000))
        self.target_port = int(config.get('target_port', self.base_port + 32))
        store_port = int(config.get('store_port', self.base_port + 80))
        if (not 1024 <= self.base_port <= 65436
                or not self.base_port <= self.target_port < self.base_port + 80
                or not self.base_port + 80 <= store_port <= self.base_port + 92):
            raise ValueError('Dynamo endpoints must fit the owned 100-port lease')

    def command(self, spec, *, dummy=False):
        tp = int(spec['tp'])
        if tp not in self.config['legal_tp'] or len(spec['gpus']) != tp:
            raise ValueError('model-specific legal TP and PP1 placement required')
        command = [sys.executable, '-m', 'pdblend_runtime.serve', self.config['model_path'],
                   '--host', '127.0.0.1', '--port', str(spec['port']),
                   '--served-model-name', 'm', '--tensor-parallel-size', str(tp),
                   '--pipeline-parallel-size', '1', '--dtype', 'bfloat16',
                   '--scheduler-cls', 'pdblend_runtime.native_v1.NativeScheduler',
                   '--worker-extension-cls', 'pdblend_baselines.dynamollm.gpu_weights.DynamoWorkerExtension',
                   '--max-model-len', str(self.config.get('max_model_len', 8192)),
                   '--max-num-seqs', str(self.config.get('max_num_seqs', 16)),
                   '--max-num-batched-tokens', str(self.config.get('max_num_batched_tokens', 8192)),
                   '--gpu-memory-utilization', str(self.config.get('gpu_memory_utilization', .85)),
                   '--no-enable-prefix-caching', '--enable-chunked-prefill', '--enforce-eager',
                   '--disable-log-requests']
        if dummy:
            command += ['--load-format', 'dummy']
        return command

    def environment(self, spec, *, dummy=False, golden=None):
        # The queue already limits Docker to the leased physical UUIDs. vLLM
        # 0.10.1.1's Platform.device_id_to_physical_device_id parses visible
        # devices with int(), so its CUDA selection must use container-local
        # ordinals. NVML still resolves these ordinals through gpu_devices().
        gpu_devices(spec['gpus'])  # Reject ordinals outside the explicit lease.
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, spec['gpus'])),
                           CUDA_DEVICE_ORDER='PCI_BUS_ID', DYNAMO_GENERATION=str(spec.get('generation', 0)),
                           DYNAMO_INSTANCE_ID=spec.get('instance_id', spec.get('id')), DYNAMO_DUMMY=str(int(dummy)),
                           DYNAMO_MODEL_ID=self.config['model_id'])
        if golden:
            environment['DYNAMO_GOLDEN_JSON'] = json.dumps(golden, allow_nan=False)
        else:
            environment.pop('DYNAMO_GOLDEN_JSON', None)
        return environment

    async def start(self, spec, *, dummy=False, golden=None):
        iid = spec.get('instance_id', spec.get('id'))
        if not iid or iid in self.processes:
            raise ValueError('new owned instance identity required')
        occupied = {gpu for row in self.instances.values() for gpu in row['gpus']}
        if occupied.intersection(spec['gpus']):
            raise ValueError('Dynamo GPU ownership overlap')
        if not set(spec['gpus']) <= set(self.config['node_gpus']):
            raise ValueError('Dynamo target lies outside lease')
        if (type(spec.get('port')) is not int
                or not self.base_port <= spec['port'] < self.base_port + 80
                or any(row['port'] == spec['port'] for row in self.instances.values())):
            raise ValueError('Dynamo HTTP endpoint collides or lies outside owned port lease')
        if dummy and not golden:
            raise ValueError('same-target-TP golden required before dummy launch')
        spec = dict(spec, id=iid, instance_id=iid, generation=int(spec.get('generation', 0)))
        spec.setdefault('url', 'http://127.0.0.1:' + str(spec['port']))
        spec['execution_config'] = dict(enforce_eager=True, cuda_graphs=False,
                                        execution_mode='eager')
        environment = self.environment(spec, dummy=dummy, golden=golden)
        self.output.mkdir(parents=True, exist_ok=True)
        log = (self.output / (iid + '.log')).open('ab')
        try:
            process = await asyncio.create_subprocess_exec(*self.command(spec, dummy=dummy),
                env=environment, stdout=log, stderr=log, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.logs[iid], self.processes[iid] = log, process
        self.instances[iid] = spec
        self.transport.instances[iid] = dict(spec)
        self.journal('dynamo_instance_start', instance_id=iid, pid=process.pid, spec=spec,
                     dummy=dummy, command=self.command(spec, dummy=dummy),
                     execution_config=spec['execution_config'], enforce_eager=True)
        deadline = time.monotonic() + self.config.get('startup_timeout_s', 600)
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError('Dynamo V1 engine exited during startup: ' + iid)
            try:
                state = await self.transport.state(iid)
                if (state.get('evidence_complete') is True and state.get('transport_healthy') is True
                        and state.get('generation') == spec['generation']):
                    self.journal('dynamo_instance_ready', instance_id=iid, state=state)
                    return dict(ready=True, instance_id=iid, generation=spec['generation'],
                                execution_config=spec['execution_config'], enforce_eager=True)
            except (OSError, RuntimeError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(.25)
        raise TimeoutError('Dynamo native readiness timed out: ' + iid)

    async def stop(self, iid):
        process = self.processes.get(iid)
        if process is None:
            if iid in self.instances:
                raise RuntimeError('refusing to acknowledge unowned process disappearance')
            return dict(absent=True, instance_id=iid, already_absent=True)
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 20)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(process.wait(), 10)
        # Descendants may outlive the process leader.  Inspect /proc so a
        # zombie does not keep an otherwise dead owned group quarantined, while
        # any live member still blocks cleanup.
        state = proc_group_state(process.pid)
        self.journal('dynamo_process_group_state', instance_id=iid, **state)
        if state['live_members']:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + PROCESS_GROUP_GRACE_S
            while time.monotonic() < deadline:
                state = proc_group_state(process.pid)
                self.journal('dynamo_process_group_state', instance_id=iid, **state)
                if state['no_live_members']:
                    break
                await asyncio.sleep(.1)
            else:
                raise RuntimeError('Dynamo process group absence unconfirmed: ' + iid)
        execution_config = self.instances[iid].get('execution_config',
                                                    dict(enforce_eager=True, cuda_graphs=False,
                                                         execution_mode='eager'))
        self.processes.pop(iid)
        self.instances.pop(iid, None)
        self.transport.instances.pop(iid, None)
        self.logs.pop(iid).close()
        receipt = dict(absent=True, instance_id=iid, pid=process.pid, returncode=process.returncode,
                       process_group_state=state,
                       no_live_members=state['no_live_members'],
                       all_pid_gone=state['all_pid_gone'],
                       absence_scope='no_live_execution', execution_config=execution_config,
                       enforce_eager=True)
        self.journal('dynamo_instance_stop', **receipt)
        return receipt

    async def close(self):
        results = await asyncio.gather(*(self.stop(iid) for iid in list(self.processes)), return_exceptions=True)
        errors = [repr(result) for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError('Dynamo owned cleanup incomplete: ' + repr(errors))
