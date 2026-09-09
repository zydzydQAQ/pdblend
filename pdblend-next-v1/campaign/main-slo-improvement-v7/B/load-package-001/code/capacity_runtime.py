"""Current-controller integration for physically measured mixed-only capacity."""
import asyncio
from dataclasses import asdict, replace
import importlib.util
import fcntl
import math
from pathlib import Path
import sys
import time

from capacity_executor import Inventory, PhysicalCapacityExecutor, ProposalDropped, check_lease, fixed, require, sha
from capacity_backend import PinnedDockerBackend


def load_planner(reference):
    require(sha(reference['path']) == reference['sha256'], 'frozen capacity planner changed')
    name = 'measured_capacity_planner_' + reference['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, reference['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    require({'transfer_counts_known', 'removable'} <= set(module.Resident.__dataclass_fields__),
            'explicit unknown transfer and original-resident protection planner required')
    return module


def physical_backend(controller, binding, inventory):
    kind = binding.get('backend_kind', 'pinned_v3')
    if kind == 'pinned_v3':
        require(binding['native_kind'] == 'v3', 'unguarded legacy physical removal is not qualified')
        return PinnedDockerBackend(controller, binding, inventory)
    require(kind == 'guarded_legacy_self_only_v1', 'unknown physical backend')
    sources = binding['source_composition']['sources']
    reference = binding['backend_source']
    require(reference == sources['backend'], 'backend source must match explicit composition')
    for source in sources.values():
        fixed_source = Path(source['path'])
        require(sha(fixed_source) == source['sha256']
                and binding['files'].get(str(fixed_source)) == source['sha256'],
                'unfrozen isolated backend dependency')
    # Flat helper imports may already exist in an interpreter. A conflicting
    # version is an error rather than silent import-cache reuse.
    for key, module_name in [('guard', 'legacy_isolation')]:
        previous = sys.modules.get(module_name)
        if previous is not None:
            require(sha(previous.__file__) == sources[key]['sha256'], 'conflicting imported isolation helper')
    name = 'measured_legacy_backend_' + reference['sha256']
    if name not in sys.modules:
        sys.path.insert(0, str(Path(reference['path']).parent))
        spec = importlib.util.spec_from_file_location(name, reference['path'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].LegacyIsolatedBackend(controller, binding, inventory)


def calibration_model(module, binding):
    identity = module.Identity(**binding['identity'])
    certificate = fixed(binding['calibration'])
    require(certificate.get('schema') == 'capacity-measured-bounds-v1'
            and certificate.get('identity') == binding['identity']
            and certificate.get('measurement_verified') is True
            and certificate.get('bounds_are_empirical_not_hard_guarantees') is True,
            'same-source measured capacity certificate required')
    from capacity_certificate import validate
    evidence_ids = validate(certificate, binding['identity'])
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


def bounded_policy(planner, deadline_s, now_s, cleanup_reserve_s):
    recovery_s = max([c.duration_upper_s for c in planner.transitions
                      if c.operation == 'restore_cold'] or [0.])
    horizon = min(planner.policy.amortization_horizon_s,
                  deadline_s-now_s-cleanup_reserve_s-recovery_s)
    return replace(planner.policy, amortization_horizon_s=horizon) if horizon > 0 else None


def transfer_observation(raw):
    """Preserve legacy unknown sends; never synthesize an idle or busy count."""
    values = [raw.get('transfer_inflight_sends'), raw.get('transfer_inflight_receives')]
    if any(value is not None and (type(value) is not int or value < 0) for value in values):
        raise ValueError('malformed native transfer count')
    known = all(value is not None for value in values) and raw.get('transfer_inflight_sends_observed') is True
    return (sum(values), True) if known else (None, False)


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
            if proposal is not None and (c.control_stop.is_set() or c.backend.inflight_actions):
                raise ProposalDropped('controller quiescing or another control in flight')
            require(not c.backend.inflight_actions, 'another physical control is in flight')
            if operation == 'remove':
                if proposal is not None and (not self.local_idle(instance['id']) or
                        any(not a.get('route') and not a['future'].done() for a in c.active.values())):
                    raise ProposalDropped('new request, queue or reservation blocks shrink')
                require(self.local_idle(instance['id']), 'new request/reservation blocks shrink')
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
        inventory_path = Path(controller.config['capacity_inventory_path'])
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        self.owner_lock = inventory_path.with_suffix('.executor.lock').open('a')
        fcntl.flock(self.owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.inventory = Inventory(controller.config['capacity_inventory_path'], controller.config['instances'],
                                   binding['identity'])
        self.adapter = ControllerAdapter(controller, self.inventory)
        self.backend = physical_backend(controller, binding, self.inventory)
        self.executor = PhysicalCapacityExecutor(self.backend, self.adapter, self.inventory,
            deadline_s=binding['deadline_s'], max_residents=8//self.identity.tp,
            validate_proposal=self.revalidate if require_calibration else None,
            lease_check=lambda: check_lease(authority=controller.config.get('capacity_lease_authority'),
                expected_inventory=controller.config['capacity_inventory_path'],
                expected_job_path=controller.config.get('capacity_job_path')))
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
            transfers, counts_known = transfer_observation(raw)
            isolated = False
            if (not counts_known and i.instance_id not in self.inventory.value['initial_ids']
                    and hasattr(self.backend, 'isolation_candidate')):
                isolated = await self.backend.isolation_candidate(known, raw)
            residents.append(m.Resident(i.instance_id, tuple(i.gpus), i.timestamp_s, i.generation,
                min(i.timestamp_s, known['changed_s']), role=i.role, accepting=i.accepting,
                transport_healthy=raw.get('transport_healthy') is True,
                active_requests=len(i.requests), queued_requests=max(i.waiting, i.running),
                kv_allocations=len(i.kv_allocations), reserved_kv_tokens=i.reserved_kv_tokens,
                transfer_allocations=len(i.transfer_allocations),
                inflight_transfers=transfers, transfer_counts_known=counts_known,
                removable=i.instance_id not in self.inventory.value['initial_ids'],
                isolated_transport_candidate=isolated,
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
            # Capacity deficit may bypass growth cooldown, never minimum off-time.
            if now-since < (self.planner.policy.min_off_s if self.planner is not None else 30.):
                continue
            spares.append(m.Spare(group, min(r['at_s'] for r in rows), since,
                min(r['free_bytes'] for r in rows), gpu_processes=sum(len(r['process_pids']) for r in rows)))
        return m.Snapshot(self.identity, c.backend.topology_version, tuple(residents), tuple(spares),
                          transition_inflight=self.inventory.value['transition_inflight'])

    def demand(self, now):
        c, m = self.controller, self.module
        window = self.binding.get('rate_observation_window_s', 60.)
        require(window == 60., 'uniform causal sixty-second capacity rate window required')
        history = [r for r in c.arrival_history if 0 <= now-r.arrival_s <= window]
        pending = [a['budget'] for a in c.active.values() if not a.get('route') and not a['future'].done()]
        all_observed = history + [a['budget'] for a in c.active.values()]
        domains = self.binding.get('demand_domains') or [self.binding['demand_domain']]
        matches = [domain for domain in domains if all(
            r.input_tokens <= domain['max_input_tokens'] and r.ttft_s == domain['slo_ttft_s']
            and r.tpot_s == domain['slo_tpot_s'] and type(r.output_limit) is int
            and 0 < r.output_limit <= domain['max_output_limit']
            and r.input_tokens+r.output_limit <= domain['max_context_tokens'] for r in all_observed)]
        # No trace, dataset/rate label or eventual output length is consulted.
        # Unknown shapes use an impossible domain, so no uncalibrated shrink occurs.
        domain_sha = matches[0]['sha256'] if len(matches) == 1 and all_observed else '0'*64
        if all_observed:
            self.last_observed_domain = domain_sha
        elif not all_observed:
            # Only zero observed demand can retain a previous real domain; never
            # invent a shape from an empty history or a future formal trace.
            domain_sha = getattr(self, 'last_observed_domain', '0'*64)
        span = min(window, max(.001, now-c.history_started_s))
        n = len(history)
        uncertainty = self.binding.get('arrival_count_margin', 2.) * math.sqrt(n+1) if n else 0.
        lower, upper = max(0., n-uncertainty)/span, (n+uncertainty)/span
        recent_lower = trend = 0.
        if now-c.history_started_s >= 10.:
            recent = sum(0 <= now-r.arrival_s <= 5. for r in history)
            previous = sum(5. < now-r.arrival_s <= 10. for r in history)
            margin = self.binding.get('arrival_count_margin', 2.)
            recent_lower = max(0., recent-margin*math.sqrt(recent+1))/5.
            previous_upper = (previous+margin*math.sqrt(previous+1))/5.
            trend = max(0., recent_lower-previous_upper)/5.
        return m.Demand(now, max(0., now-c.history_started_s), lower, upper, domain_sha,
                        len(pending), max([now-r.arrival_s for r in pending] or [0.]),
                        recent_lower, trend,
                        min([max(0., r.arrival_s+r.ttft_s-now) for r in pending]) if pending else None)

    async def revalidate(self, proposal):
        snapshot = await self.snapshot()
        now = time.time()
        if not self.module.revalidate(proposal,snapshot,now):
            return False
        demand = self.demand(now)
        state = self.state
        if getattr(self,'previous_domain',None) != demand.domain_sha256:
            state = self.module.State(last_change_s=state.last_change_s)
        policy = bounded_policy(self.planner,self.binding['deadline_s'],now,
                                self.executor.cleanup_reserve_s)
        if policy is None:
            return False
        planner = self.module.CapacityPlanner(self.identity,self.planner.layouts,
            self.planner.transitions,self.planner.savings,policy)
        fresh = planner.choose(snapshot,demand,state,now).proposal
        return fresh is not None and all(getattr(fresh,k)==getattr(proposal,k)
            for k in ('action','gpus','remove_id','identity'))

    def startup_risk(self, snapshot, demand, now):
        """Record measured cold forecast and causal headroom policy; no SLO promise."""
        pending = [a['budget'] for a in self.controller.active.values()
                   if not a.get('route') and not a['future'].done()]
        cap,_ = self.planner.capacity(self.module.groups(i.gpus for i in snapshot.residents),demand)
        cold = [c for c in self.planner.transitions if c.operation=='restore_cold'
                and any(tuple(s.gpus)==tuple(c.gpus) for s in snapshot.spares)]
        duration = min([c.duration_upper_s for c in cold] or [float('inf')])
        available = math.isfinite(duration)
        return dict(schema='capacity-causal-startup-risk-v1',observed_s=now,
            pending_requests=len(pending),already_expired_ttft=sum(now>=r.arrival_s+r.ttft_s for r in pending),
            ttft_deadline_before_earliest_cold_ready=sum(r.arrival_s+r.ttft_s<now+duration for r in pending),
            earliest_qualified_cold_ready_estimate_s=now+duration if available else None,
            measured_cold_duration_upper_s=duration if available else None,
            current_empirical_capacity_lower_rps=cap,
            estimated_backlog_at_cold_ready_upper=(len(pending)+max(0.,demand.rate_upper_rps-cap)*duration)
                if available and cap is not None else None,
            basis='only arrived pending deadlines, past sixty-second rate and measured capacity/cold cost',
            prediction_only=True,slo_recovery_guaranteed=False,
            growth_trigger_unchanged=False,
            causal_growth=(self.planner.growth_signals(snapshot,demand,cap,now)
                if cap is not None and hasattr(self.planner,'growth_signals') else None))

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
                self.inventory.event('capacity_startup_risk',**self.startup_risk(snapshot,demand,now))
                if getattr(self, 'previous_domain', None) != demand.domain_sha256:
                    self.state = self.module.State(last_change_s=self.state.last_change_s)
                    self.previous_domain = demand.domain_sha256
                scoped_policy = bounded_policy(self.planner,self.binding['deadline_s'],
                                               now,self.executor.cleanup_reserve_s)
                if scoped_policy is None:
                    self.inventory.event('capacity_decision', reason='deadline_recovery_reserve')
                    return None
                scoped = self.module.CapacityPlanner(self.identity, self.planner.layouts,
                    self.planner.transitions, self.planner.savings, scoped_policy)
                result = scoped.choose(snapshot, demand, self.state, now)
                self.state = result.state
                self.inventory.event('capacity_decision', reason=result.reason, demand=asdict(demand),
                                     proposal=asdict(result.proposal) if result.proposal else None)
                if result.proposal is None:
                    return result
                if len(snapshot.residents) >= 8//self.identity.tp and result.proposal.action == 'restore':
                    return None
                try:
                    execution = await self.executor.execute(result.proposal)
                except ProposalDropped as exc:
                    self.inventory.event('proposal_dropped', reason=str(exc),
                        proposal=asdict(result.proposal), physical_attempted=False)
                    return None
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
