"""Current-controller integration for physically measured mixed-only capacity."""
import asyncio
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import sys
import time

from capacity_executor import Inventory, PhysicalCapacityExecutor, fixed, require, sha
from capacity_backend import PinnedDockerBackend


def load_planner(reference):
    require(sha(reference['path']) == reference['sha256'], 'frozen capacity planner changed')
    name = 'measured_capacity_planner_v2'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, reference['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def calibration_model(module, binding):
    identity = module.Identity(**binding['identity'])
    certificate = fixed(binding['calibration'])
    require(certificate.get('schema') == 'capacity-measured-bounds-v1'
            and certificate.get('identity') == binding['identity']
            and certificate.get('measurement_verified') is True
            and certificate.get('bounds_are_empirical_not_hard_guarantees') is True,
            'same-source measured capacity certificate required')
    references = certificate.get('raw_measurements', [])
    require(references, 'calibration has no raw measurements')
    evidence_ids = set()
    for reference in references:
        raw = fixed(reference)
        require(raw.get('measurement_valid') is True and raw.get('gpu_indices') == list(range(8))
                and raw.get('power_evidence', {}).get('power_source_verified') is True
                and not raw.get('sampling_error') and raw.get('artifacts'), 'raw whole-node measurement invalid')
        require(all(sha(p) == h for p, h in raw['artifacts'].items()), 'raw calibration artifact changed')
        require(raw.get('energy_j', 0) > 0 and raw.get('measurement_end_s', 0) > raw.get('measurement_start_s', 0),
                'calibration energy/window invalid')
        evidence_ids.add(reference['sha256'])
    def evidence(value):
        require(value['raw_sha256'] in evidence_ids, 'bound does not reference validated raw evidence')
        return module.Evidence(identity, value['raw_sha256'], certified=True)
    layouts = [module.LayoutBound(module.groups(c['resident_groups']), c['demand_domain_sha256'],
               c['sustainable_rate_lower_rps'], evidence(c)) for c in certificate.get('layouts', [])]
    transitions = [module.TransitionBound(c['operation'], tuple(c['gpus']), c['duration_upper_s'],
                   c['energy_upper_j'], evidence(c), c.get('peak_memory_per_gpu_upper_bytes', 0),
                   c.get('cached_weights_sha256')) for c in certificate.get('transitions', [])]
    savings = [module.SavingsBound(module.groups(c['source_groups']), module.groups(c['target_groups']),
               c['demand_domain_sha256'], c['rate_lower_rps'], c['rate_upper_rps'],
               c['whole_node_saving_lower_w'], evidence(c)) for c in certificate.get('savings', [])]
    require(layouts and transitions and savings, 'layout, cold/remove costs and matched-load savings are required')
    require(any(t.operation == 'restore_cold' for t in transitions)
            and any(t.operation == 'remove' for t in transitions), 'cold fallback/removal calibration required')
    policy = module.Policy(**dict(binding.get('policy', {}), min_residents=2))
    return module.CapacityPlanner(identity, layouts, transitions, savings, policy)


class ControllerAdapter:
    def __init__(self, controller, inventory):
        self.controller, self.inventory = controller, inventory

    def contains(self, iid):
        return iid in self.controller.backend.instances

    def local_idle(self, iid):
        c = self.controller
        current = next((i for i in c.state.snapshot.instances if i.instance_id == iid), None)
        if current is None:
            return False
        routed = any(a.get('route') and iid in (a['route'].prefill_id, a['route'].decode_id)
                     for a in c.active.values())
        return not routed and not any((current.requests, current.running, current.waiting,
            current.kv_allocations, current.reserved_kv_tokens, current.transfer_allocations,
            current.reserved_transfer_bytes))

    async def reserve(self, operation, instance, proposal):
        c = self.controller
        async with c.action_lock:
            require(not c.control_stop.is_set() or proposal is None, 'controller already quiescing')
            require(not c.backend.inflight_actions, 'another physical control is in flight')
            if operation == 'remove':
                require(self.local_idle(instance['id']), 'new request/reservation blocks shrink')
                if proposal is not None:
                    require(not any(not a.get('route') and not a['future'].done() for a in c.active.values()),
                            'new queued request blocks shrink')
                c.frozen_instances.add(instance['id'])
                await c.refresh()

    async def freeze_and_require_idle(self, instance):
        await self.reserve('remove', instance, None)

    async def commit(self, remove_ids, added):
        c = self.controller
        async with c.action_lock:
            require(all(self.local_idle(i) for i in remove_ids), 'nonempty target at capacity commit')
            c.backend.replace_instances(remove_ids, added)
            await c.refresh()
            # Every newly published identity is now explicit to the outer runner.
            self.inventory.event('routing_commit', removed=list(remove_ids), added=[i['id'] for i in added],
                                 topology_version=c.backend.topology_version)

    async def unfreeze(self, iid):
        c = self.controller
        async with c.action_lock:
            c.frozen_instances.discard(iid)
            await c.refresh()


class CapacityService:
    def __init__(self, controller, binding, *, require_calibration=True):
        self.controller, self.binding = controller, binding
        self.module = load_planner(binding['planner_source'])
        self.identity = self.module.Identity(**binding['identity'])
        self.planner = calibration_model(self.module, binding) if require_calibration else None
        self.inventory = Inventory(controller.config['capacity_inventory_path'], controller.config['instances'],
                                   binding['identity'])
        self.adapter = ControllerAdapter(controller, self.inventory)
        self.backend = PinnedDockerBackend(controller, binding, self.inventory)
        self.executor = PhysicalCapacityExecutor(self.backend, self.adapter, self.inventory,
            deadline_s=binding['deadline_s'], max_residents=8//self.identity.tp,
            validate_proposal=self.revalidate if require_calibration else None)
        self.state = self.module.State()
        self.stopping = False
        self.tick_lock = asyncio.Lock()
        self.current_task = None
        self.last_spares = {}

    async def snapshot(self):
        c, m = self.controller, self.module
        current = c.state.snapshot
        instances = list(c.backend.instances.values())
        used = {g for i in instances for g in i['gpus']}
        free = [g for g in range(8) if g not in used]
        gpu_rows = await self.backend.gpu_state(free) if free else []
        by_gpu = {v['gpu']:v for v in gpu_rows}
        now = time.time()
        residents = []
        for i in current.instances:
            raw = c.backend.last.get(i.instance_id, {})
            known = self.inventory.value['known_instances'][i.instance_id]
            residents.append(m.Resident(i.instance_id, tuple(i.gpus), i.timestamp_s, i.generation,
                min(i.timestamp_s, known['changed_s']), role=i.role, accepting=i.accepting,
                transport_healthy=raw.get('transport_healthy') is True,
                active_requests=len(i.requests), queued_requests=max(i.waiting, i.running),
                kv_allocations=len(i.kv_allocations), reserved_kv_tokens=i.reserved_kv_tokens,
                transfer_allocations=len(i.transfer_allocations),
                inflight_transfers=int(raw.get('transfer_inflight_sends', 1))+int(raw.get('transfer_inflight_receives', 1)),
                reserved_transfer_bytes=i.reserved_transfer_bytes,
                pending_controls=int(bool(raw.get('scheduler_budget_pending')))
                    +int(raw.get('acknowledged_generation') != i.generation),
                error=raw.get('error') or raw.get('runtime_error')))
        spares = []
        for start in range(0, 8, self.identity.tp):
            group = tuple(range(start, start+self.identity.tp))
            if not set(group) <= set(free):
                self.last_spares.pop(group, None)
                continue
            rows = [by_gpu[g] for g in group]
            since = self.last_spares.setdefault(group, now)
            spares.append(m.Spare(group, min(r['at_s'] for r in rows), since,
                min(r['free_bytes'] for r in rows), gpu_processes=sum(len(r['process_pids']) for r in rows)))
        return m.Snapshot(self.identity, c.backend.topology_version, tuple(residents), tuple(spares),
                          transition_inflight=self.inventory.value['transition_inflight'])

    def demand(self, now):
        c, m = self.controller, self.module
        window = self.binding.get('rate_observation_window_s', 10.)
        require(1 <= window <= 60, 'bounded historical rate window required')
        history = [r for r in c.arrival_history if 0 <= now-r.arrival_s <= window]
        pending = [a['budget'] for a in c.active.values() if not a.get('route') and not a['future'].done()]
        all_observed = history + [a['budget'] for a in c.active.values()]
        domain = self.binding['demand_domain']
        valid = all(r.input_tokens <= domain['max_input_tokens'] and r.ttft_s >= domain['min_ttft_s']
                    and r.tpot_s >= domain['min_tpot_s'] for r in all_observed)
        # No trace, dataset/rate label or eventual output length is consulted.
        # Unknown shapes use an impossible domain, so no uncalibrated shrink occurs.
        domain_sha = domain['sha256'] if valid else '0'*64
        span = min(window, max(.001, now-c.history_started_s))
        n = len(history)
        uncertainty = self.binding.get('arrival_count_margin', 2.) * math.sqrt(n+1)
        lower, upper = max(0., n-uncertainty)/span, (n+uncertainty)/span
        return m.Demand(now, max(0., now-c.history_started_s), lower, upper, domain_sha,
                        len(pending), max([now-r.arrival_s for r in pending] or [0.]))

    async def revalidate(self, proposal):
        snapshot = await self.snapshot()
        now = time.time()
        if proposal.action == 'remove' and self.demand(now).queued_requests:
            return False
        return self.module.revalidate(proposal, snapshot, now)

    async def tick(self, now_s=None):
        if self.stopping or self.tick_lock.locked():
            return None
        require(self.planner is not None, 'calibration service cannot run automatic planning')
        async with self.tick_lock:
            if self.stopping:
                return None
            self.current_task = asyncio.current_task()
            try:
                snapshot = await self.snapshot()
                now = time.time() if now_s is None else now_s
                demand = self.demand(now)
                result = self.planner.choose(snapshot, demand, self.state, now)
                self.state = result.state
                self.inventory.event('capacity_decision', reason=result.reason, demand=asdict(demand),
                                     proposal=asdict(result.proposal) if result.proposal else None)
                if result.proposal is None:
                    return result
                if len(snapshot.residents) >= 8//self.identity.tp and result.proposal.action == 'restore':
                    return None
                execution = await self.executor.execute(result.proposal)
                self.state = self.module.committed(self.state, result.proposal, execution['finished_s'],
                                                   execution_verified=execution['execution_verified'])
                return execution
            finally:
                self.current_task = None

    async def quiesce(self):
        self.stopping = True
        task = self.current_task
        if task is not None and task is not asyncio.current_task():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task

    async def finish_to_initial(self):
        await self.quiesce()
        await self.executor.finish_to_initial()


def create_service(controller):
    c = controller.config
    require(c.get('capacity_integration_v1') is True and controller.strategy.startswith('pdblend')
            and c.get('allow_pd') is False and not c.get('transfers')
            and not c.get('slow_topology') and not c.get('dynamic_pools')
            and set(c.get('node_gpus', [])) == set(range(8)), 'independent mixed capacity protocol required')
    require(len(c['instances']) == 2 and all(i['role'] == 'mixed' for i in c['instances']),
            'exact two mixed starting instances required')
    binding = fixed(dict(path=c['capacity_binding_path'], sha256=c['capacity_binding_sha256']))
    require(binding.get('schema') == 'capacity-runtime-binding-v1' and binding.get('min_residents') == 2
            and binding.get('max_residents') == 8//binding['identity']['tp'], 'capacity bounds differ')
    return CapacityService(controller, binding)
