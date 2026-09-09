"""Causal remaining-work horizons shared by admission and frequency control."""
from dataclasses import replace


def admission_budget(plan, request, now=None):
    route = plan.routes[0]
    return replace(request, pending_import_s=route.import_block_s,
        pending_ready_s=(plan.created_s if now is None else now) + max(0., route.predicted_ttft_s-route.predicted_tpot_s),
        pending_frequency_mhz=next((a.frequency_mhz for a in plan.frequencies if a.instance_id==route.decode_id),None))


def resident_context(instance):
    if instance.role == 'prefill':
        return max((r.input_tokens+1 for r in instance.requests), default=0)
    return max((r.input_tokens+max(r.predicted_output,r.emitted) for r in instance.requests), default=0)


class TailModel:
    """Completion horizon; dynamic kernel energy is deliberately absent.

    Admission stores the already-promised phase readiness in each budget, so
    parallel instances do not each charge the same node-wide prefill interval.
    Unmarked analytical snapshots reconstruct a conservative FIFO phase queue
    from profiles; this also supports independent baseline policy callers.
    """
    def __init__(self, estimator, snapshot, now):
        self.estimator, self.snapshot, self.now = estimator, snapshot, now
        self.contexts = {i.instance_id:resident_context(i) for i in snapshot.instances}
        self.sources = {r.request_id:i for i in snapshot.instances if i.role=='prefill' for r in i.requests}
        self.consumers = {r.request_id for i in snapshot.instances if i.role=='decode' for r in i.requests}
        self._points = {}; self._source_ready = {}
        self._active_groups = {}
        for i in snapshot.instances:
            grouped = {}
            for r in i.requests:
                if r.emitted:
                    remaining=max(r.predicted_output-r.emitted,1)
                    old=grouped.get(r.input_tokens)
                    if old is None or remaining>old[1]: grouped[r.input_tokens]=(r,remaining)
            self._active_groups[i.instance_id]=tuple(grouped.values())
        self.tails = {i.instance_id:self.instance_tail(i) for i in snapshot.instances}

    def point(self, instance, request, frequency, batch, context=None):
        context = self.contexts[instance.instance_id] if context is None else context
        query_context = max(context, request.input_tokens+(1 if instance.role=='prefill' else request.predicted_output))
        key = (instance.role,instance.tp,frequency,request.input_tokens,query_context,batch)
        if key not in self._points:
            self._points[key] = self.estimator.profiles.lookup(*key)
        return self._points[key]

    def source_ready(self, request):
        source = self.sources.get(request.request_id)
        if source is None:
            return 0., None, None
        if source.instance_id not in self._source_ready:
            ready = {}; elapsed = 0.
            for r in source.requests:
                # A batch-one FIFO bound never invents parallel prefill work.
                p = self.point(source,r,source.frequency_mhz,1)
                if p is None or elapsed is None:
                    ready[r.request_id] = (None,None)
                    elapsed=None
                else:
                    elapsed += p.phase_time_bound('prefill')
                    ready[r.request_id] = (elapsed,p)
            self._source_ready[source.instance_id] = ready
        elapsed,p = self._source_ready[source.instance_id][request.request_id]
        return elapsed,source,p

    def instance_tail(self, instance, *, context=None, delays=None, active_delay=0., requests=None,
                      frequency=None, batch=None):
        requests=instance.requests if requests is None else requests
        frequency=instance.frequency_mhz if frequency is None else frequency
        if not requests:
            return 0.
        # A reserved decode budget already includes its source's work. Orphan
        # source work still keeps the node alive, without a duplicate charge.
        if instance.role=='prefill':
            orphan = [r for r in instance.requests if r.request_id not in self.consumers]
            values = [self.source_ready(r)[0] for r in orphan]
            return None if any(v is None for v in values) else max(values,default=0.)
        context = self.contexts[instance.instance_id] if context is None else context
        delays = delays or {}
        batch = max(1,len(requests),instance.running+instance.waiting) if batch is None else batch
        pending = [r for r in requests if not r.emitted]
        work = {}; ready = {}; elapsed = 0.
        for r in pending:
            trusted=(r.pending_ready_s is not None and r.pending_frequency_mhz==frequency)
            if instance.role=='mixed':
                pp = self.estimator.profiles.lookup('mixed',instance.tp,frequency,
                    r.input_tokens,r.input_tokens+1,1)
                if pp is None: return None
                phase_work=pp.phase_time_bound('prefill')
                if trusted:
                    phase_work=min(phase_work,max(0.,r.pending_ready_s-self.now-elapsed))
                work[r.request_id] = phase_work
                elapsed += work[r.request_id]
                inferred = elapsed
            else:
                work[r.request_id] = r.pending_import_s
                inferred = 0.
                if not trusted:
                    source_time,source,pp = self.source_ready(r)
                    if source_time is None: return None
                    if source is not None:
                        link = self.estimator.transfer_point(source.tp,instance.tp,source.gpus,instance.gpus,
                            r.input_tokens,pp.batch,None)
                        links=[link] if link is not None else []
                    else:
                        # After P completes its budget leaves that instance.
                        # An external clock manager can invalidate the original
                        # destination-clock promise; use a measured upper bound
                        # across eligible transports rather than inventing its
                        # former source/batch or trusting the stale timestamp.
                        links=[link for link in self.estimator.transfers if link.validated and link.source_sha256
                               and link.target_tp==instance.tp and link.max_input_tokens>=r.input_tokens]
                    if not links: return None
                    work[r.request_id]=max(work[r.request_id],max(link.import_seconds_upper or link.seconds_upper for link in links))
                    inferred = source_time + elapsed + max(link.seconds_upper for link in links)
                elapsed += work[r.request_id]
            ready[r.request_id] = (max(0.,r.pending_ready_s-self.now)
                if trusted else inferred)
        # Later prefills/imports delay older decode streams. Their own ready
        # timestamps already include earlier phase work, so add only successors.
        later = {}; suffix = 0.
        for r in reversed(pending):
            later[r.request_id] = suffix
            suffix += work[r.request_id]
        tails = []
        for r,remaining in self._active_groups[instance.instance_id]:
            point=self.point(instance,r,frequency,batch,context)
            if point is None: return None
            tails.append(suffix+remaining*point.iteration_s+active_delay)
        for r in pending:
            point=self.point(instance,r,frequency,batch,context)
            if point is None: return None
            phase=ready[r.request_id]+later[r.request_id]
            tails.append(phase+max(r.predicted_output-1,0)*point.iteration_s+delays.get(r.request_id,0.))
        return max(tails,default=0.)

    def after_admission(self, instance, request, frequency, *, switch_delay=0., source=None, source_delay=0., new_point=None,
                        per_instance=False):
        context = max(self.contexts[instance.instance_id],request.input_tokens+request.predicted_output)
        delays = {r.request_id:switch_delay for r in instance.requests}
        source_ids = {r.request_id for r in source.requests} if source is not None and source_delay else set()
        tails = dict(self.tails)
        if source_ids:
            for other in self.snapshot.instances:
                affected = {r.request_id:source_delay for r in other.requests if r.request_id in source_ids and not r.emitted}
                if other.instance_id==instance.instance_id:
                    for rid,delay in affected.items(): delays[rid]=delays.get(rid,0.)+delay
                elif affected and other.role=='decode':
                    tails[other.instance_id]=self.instance_tail(other,delays=affected)
        if not instance.requests and new_point is not None:
            tails[instance.instance_id]=max(0.,request.pending_ready_s-self.now)+max(request.predicted_output-1,0)*new_point.iteration_s
        else:
            tails[instance.instance_id]=self.instance_tail(instance,context=context,delays=delays,
                active_delay=switch_delay,requests=instance.requests+(request,),frequency=frequency,
                batch=max(1,len(instance.requests)+1,instance.running+instance.waiting+1))
        if any(t is None for t in tails.values()): return None
        return tails if per_instance else max(tails.values(),default=0.)
