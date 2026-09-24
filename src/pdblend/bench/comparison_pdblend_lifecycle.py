"""PDblend-only resident boundaries; policy-owned off/L1 actions stay intact."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import urllib.request

from pdblend.online.native_control import validate_state
from pdblend_baselines.resident_campaign import drain_endpoints, verify_endpoints, model_load_lock

ROLES = {'P', 'D', 'M', 'idle', 'L1', 'off'}


def _json(spec, endpoint, payload=None):
    request = urllib.request.Request(spec.base_url+'/baseline/'+endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'}, method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(request, timeout=35) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError('PDblend restart control returned a non-object')
    return result


class PDblendResidentBoundary:
    """Only the PD-owned fleet is adapted; shared launcher/controller stay fixed.

    After a window, no process is restarted and no parked clock is changed.
    The next reset restores the exact original physical inventory before the
    common reset establishes the new native generation and warmup.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        self.initial_specs = {s.instance_id:s for s in adapter.specs}
        self.last_roles = None
        self.restart_epochs = []
        self.window_start_count = None
        self._inventory()
        for instance in adapter.fleet.instances.values():
            self._install_ready_boundary(instance)
        self.refresh_load_count()

    def _inventory(self):
        fleet = self.adapter.fleet.instances
        if set(fleet) != set(self.initial_specs) or {s.instance_id for s in self.adapter.specs} != set(fleet):
            raise RuntimeError('PDblend resident instance inventory changed')
        for spec in self.adapter.specs:
            frozen = asdict(self.initial_specs[spec.instance_id]); current = asdict(spec)
            frozen.pop('generation'); current.pop('generation')
            actual = asdict(fleet[spec.instance_id].spec); actual.pop('generation')
            if current != frozen or actual != frozen:
                raise RuntimeError('PDblend resident launch topology/options changed: '+spec.instance_id)

    def refresh_load_count(self):
        events = [dict(event) for instance in self.adapter.fleet.instances.values()
                  for event in instance.events if event.get('kind') == 'start']
        if len(events) < self.adapter.load_count:
            raise RuntimeError('PDblend actual process-start evidence regressed')
        self.adapter.load_count = len(events)
        return dict(engine_loads=len(events), starts=sorted(events, key=lambda e:e['t_s']))

    def _install_ready_boundary(self, instance):
        original = instance.wait_ready
        verified_process = instance.process

        def wait_ready(*args, **kwargs):
            nonlocal verified_process
            elapsed = original(*args, **kwargs)
            # A real off->start produces a new Popen. A repeated readiness
            # check must not reset a live server's generation or admission.
            if instance.process is verified_process:
                return elapsed
            spec = instance.spec; started = time.time()
            cap = _json(spec, 'capability')
            expected = self.adapter.identity
            uuids = [expected['fleet_gpu_uuids'][g] for g in spec.gpus]
            if (cap.get('supported') is not True or cap.get('tp') != spec.tp or cap.get('pp') != spec.pp
                    or cap.get('model_id') != Path(spec.model).name
                    or any(cap.get(k) != expected[k] for k in ('model_hash','tokenizer_hash','image_digest'))
                    or cap.get('source_revision') != os.environ['PDBLEND_SOURCE_SHA256']
                    or sorted(cap.get('gpu_uuids', [])) != sorted(uuids)):
                raise RuntimeError('PDblend restarted endpoint identity differs')
            before = _json(spec, 'state')
            validate_state(before, generation=before['generation'], tp=spec.tp, pp=spec.pp,
                           drained=True, observed_after_s=started)
            if before['generation'] > spec.generation:
                raise RuntimeError('PDblend restarted native generation advanced unexpectedly')
            desired = dict(generation=spec.generation, role='mixed', mode='temporal',
                           accepting=False, admit_prefill=False, admit_decode=False)
            ack = _json(spec, 'control', desired)
            state = _json(spec, 'state')
            validate_state(state, generation=spec.generation, tp=spec.tp, pp=spec.pp,
                           drained=True, observed_after_s=started)
            if (ack.get('acknowledged') is not True or ack.get('generation') != spec.generation
                    or any(state.get(k) != v for k,v in desired.items())):
                raise RuntimeError('PDblend restarted native epoch/admission restoration failed')
            self.restart_epochs.append(dict(instance_id=spec.instance_id, pid=instance.process.pid,
                started_s=started, finished_s=time.time(), capability=cap, before=before, control=ack, state=state))
            verified_process = instance.process
            return elapsed

        instance.wait_ready = wait_ready

    async def _off_evidence(self, specs, timeout_s=30.):
        if not specs:
            return []
        backend = self.adapter.gpus.backend; nvml = backend._nvml
        expected = self.adapter.identity['fleet_gpu_uuids']
        if os.environ.get('PDBLEND_GPU_UUIDS', '').split(',') != expected:
            raise RuntimeError('PDblend physical lease changed')
        deadline = time.monotonic()+timeout_s
        while True:
            rows=[]
            for spec in specs:
                instance = self.adapter.fleet[spec.instance_id]
                if instance.alive() or instance.state != 'off':
                    raise RuntimeError('PDblend final off role still has a live/undeclared process: '+spec.instance_id)
                lifecycle = [e for e in instance.events if e.get('kind') in ('start', 'stop')]
                if not lifecycle or lifecycle[-1]['kind'] != 'stop':
                    raise RuntimeError('PDblend off instance lacks an owned process-stop receipt')
                devices=[]
                for gpu in spec.gpus:
                    handle = backend._handle(gpu); uuid = nvml.nvmlDeviceGetUUID(handle)
                    uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
                    if uuid != expected[gpu]:
                        raise RuntimeError('PDblend off GPU UUID differs from lease')
                    pids = [p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                    devices.append(dict(local_index=gpu, uuid=uuid, compute_pids=pids))
                rows.append(dict(instance_id=spec.instance_id, role='off', process_alive=False,
                    process_state=instance.state, owned_stop=dict(lifecycle[-1]), physical_gpus=devices,
                    compute_processes_gone=all(not d['compute_pids'] for d in devices), checked_s=time.time()))
            if all(row['compute_processes_gone'] for row in rows):
                return rows
            if time.monotonic() >= deadline:
                raise RuntimeError('PDblend off GPU retains compute processes: '+json.dumps(rows))
            await asyncio.sleep(.25)

    async def drain(self, roles):
        self._inventory()
        if set(roles) != set(self.initial_specs) or any(role not in ROLES for role in roles.values()):
            raise RuntimeError('PDblend final role inventory is incomplete or invalid')
        live=[]; off=[]
        for spec in self.adapter.specs:
            instance = self.adapter.fleet[spec.instance_id]
            if roles[spec.instance_id] == 'off':
                off.append(spec)
            elif not instance.alive() or instance.state != 'ready':
                raise RuntimeError('PDblend online role lost its process: '+spec.instance_id)
            else:
                live.append(spec)
        states = await drain_endpoints(live) if live else []
        absent = await self._off_evidence(off)
        return dict(passed=True, states=states, off_instances=absent, final_roles=dict(roles),
            live_instance_ids=[s.instance_id for s in live], inventory_instance_ids=sorted(self.initial_specs),
            tail_end_s=time.time(), engine_load_accounting=self.refresh_load_count(),
            restart_epoch_receipts=list(self.restart_epochs), policy_off_preserved=True)

    async def restore(self):
        started = time.time()
        roles = self.last_roles if self.last_roles is not None else {iid:'M' for iid in self.initial_specs}
        before = await self.drain(roles)
        missing = [s for s in self.adapter.specs if roles[s.instance_id] == 'off']
        load_before = self.adapter.load_count
        # This is outside the prior service/tail interval and before warmup.
        for spec in self.adapter.specs:
            for gpu in spec.gpus:
                self.adapter.gpus.unpark(gpu); self.adapter.gpus.set_clock(gpu, 2520)

        def load():
            with model_load_lock():
                for spec in missing:
                    instance = self.adapter.fleet[spec.instance_id]
                    instance.start(); instance.wait_ready(timeout_s=900)
        if missing:
            await asyncio.to_thread(load)
        self.refresh_load_count()
        caps = await verify_endpoints(self.adapter.specs)
        self.last_roles = {iid:'M' for iid in self.initial_specs}
        return dict(passed=True, before=before, restored_instances=[s.instance_id for s in missing],
            capabilities=caps, inventory_instance_ids=sorted(self.initial_specs),
            engine_loads=self.adapter.load_count-load_before, cumulative_engine_loads=self.adapter.load_count,
            started_s=started, finished_s=time.time(), allocated_to_previous_service=False)

    def begin_window(self):
        self.window_start_count = self.refresh_load_count()['engine_loads']

    async def finish_window(self, raw):
        if raw.get('quarantined_instances'):
            raise RuntimeError('PDblend native run quarantined instances')
        result = await self.drain(raw.get('final_roles', {}))
        result['window_engine_loads'] = self.adapter.load_count-self.window_start_count
        self.last_roles = dict(raw['final_roles'])
        return result
