"""Independent CPU preparation and a restricted planner for low-M experiments.

Preparation does not issue a capacity certificate. Evaluation traces and seed
701 are rejected; only subsequently validated GPU receipts can issue one.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from pdblend.planner.pool import Plan, PoolPlanner
from pdblend.planner.transitions import identity
from .capacity_workloads import corpus_inputs
from .client import Request, poisson_trace, staged_trace
from .comparison_campaign import binding
from .resident_session import digest, write_new

KIND = 'pdblend_low_m_tuning_v2'
FAMILIES = ('nominal', 'burst25', 'short_output', 'long_prompt', 'initial_burst', 'tail_burst')
CONTEXT_KEYS = ('algorithm_source_sha256', 'workload_family_sha256', 'recovery_policy_sha256')
SOURCE_PACKAGES = ('pdblend','pdblend_runtime','pdblend_baselines')
RATE_DOMAIN = 'fixed_nominal_workload_with_online_forecast_ceiling/v2'


def source_inventory(root=None):
    """Algorithm bytes only: results and generated evidence never enter the digest."""
    if root is None:
        source = Path(__file__).resolve().parents[1]
    else:
        root = Path(root).resolve()
        source = root/'src'/'pdblend' if (root/'src'/'pdblend').is_dir() else root/'pdblend'
    if not source.is_dir():
        raise ValueError('algorithm source root lacks pdblend package')
    return {str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for package in SOURCE_PACKAGES for p in sorted((source.parent/package).rglob('*.py'))}


def workload_identity(corpus, model_id, dataset='sharegpt'):
    return dict(model_id=model_id, dataset=dataset,
                corpus_manifest=corpus['corpus_manifest'], corpus_dataset=corpus['corpus_dataset'])


def validate_recovery(recovery):
    from .pdblend_runtime_options import CONTROL_OPTIONS, control_options
    from pdblend.online.policies import get_policy
    if not isinstance(recovery,dict) or set(recovery)-set(CONTROL_OPTIONS):
        raise ValueError('tuning recovery must contain only CONTROL_OPTIONS, not general runtime/artifact keys')
    recovery = control_options(get_policy('pdblend'),recovery)
    if any(recovery.get(k) is not True for k in ('startup_safety', 'deadline_safety',
            'preserve_overload_capacity', 'safety_recovery', 'slo_routing', 'capacity_floor_reserve_canonical')):
        raise ValueError('low-M tuning requires the complete canonical recovery policy')
    if recovery.get('shield_mode') != 'budget_aware' or type(recovery.get('safety_max_freq')) is not int:
        raise ValueError('low-M tuning requires budget-aware Shield and explicit frequency ceiling')
    return recovery


def rate_stages(family, duration):
    if family not in FAMILIES:
        raise ValueError('unsupported independent tuning family')
    if duration <= 0:
        raise ValueError('tuning duration must be positive')
    burst = min(30., duration/3.)
    if family == 'burst25':
        edge = (duration-burst)/2.
        return [(edge, 1.), (burst, 1.25), (edge, 1.)]
    if family == 'initial_burst':
        return [(burst, 1.25), (duration-burst, 1.)]
    if family == 'tail_burst':
        return [(duration-burst, 1.), (burst, 1.25)]
    return [(duration, 1.)]


def trace_rate_metadata(*, family, rate, duration):
    stages, start = [], 0.
    for span, scale in rate_stages(family, duration):
        stages.append(dict(start_s=start, end_s=start+span, rate_rps=rate*scale))
        start += span
    return dict(rate_stages=stages, peak_rate_rps=max(row['rate_rps'] for row in stages),
        rate_rps=sum((row['end_s']-row['start_s'])*row['rate_rps'] for row in stages)/duration,
        rate_rps_semantics='time_weighted_target_rate')


def trace_requests(records, *, family, rate, duration, seed):
    stages = rate_stages(family, duration)
    selected = records
    if family == 'short_output':
        selected = [dict(r, output_tokens=min(8, r['output_tokens'])) for r in records]
    elif family == 'long_prompt':
        threshold = sorted(len(r['prompt']) for r in records)[int(.9*(len(records)-1))]
        selected = [r for r in records if len(r['prompt']) >= threshold]
    if len(stages) > 1:
        return staged_trace(selected, rate, stages, 1., seed, source='sharegpt:tuning:'+family)
    return poisson_trace(selected, rate, duration,
                         seed, source='sharegpt:tuning:'+family)


def prepare(*, output, corpus, profile, recovery, source_root=None, seeds=(8801, 8802, 8803),
            frequencies=(2100, 1800, 1500, 1200, 900), duration_s=150., model_id='Qwen2.5-7B-Instruct'):
    from pdblend.profile.query.versions import load_profile
    recovery = validate_recovery(recovery)
    if len(set(seeds)) != 3 or 8801 not in seeds or any(type(seed) is not int or seed == 701 for seed in seeds):
        raise ValueError('exactly three distinct independent tuning seeds, excluding evaluation seed 701, required')
    if duration_s != 150.:
        raise ValueError('low-M qualification uses the complete 150s service plus measured tail')
    loaded = load_profile(profile, system='pdblend', model_id=model_id, tp=1, pp=1, usage='development')
    if not frequencies or any(f not in loaded.model.freqs or f > recovery['safety_max_freq'] for f in frequencies):
        raise ValueError('candidate frequency is outside profile or safety ceiling')
    records, corpus_manifest, corpus_refs = corpus_inputs(corpus, 'sharegpt', 'tuning')
    if corpus_manifest['model_name'] != model_id:
        raise ValueError('tuning corpus model differs from profile')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    sources = source_inventory(source_root)
    family_identity = workload_identity(corpus_refs, model_id)
    startup_plan = dict(counts={'M':4,'L1':4},f_P=recovery['safety_max_freq'],
                       f_D=recovery['safety_max_freq'],f_M=recovery['safety_max_freq'],tau=0)
    context = dict(algorithm_source_sha256=digest(sources), workload_family_sha256=digest(family_identity),
                   recovery_policy_sha256=digest(dict(control=recovery,startup_plan=startup_plan)))
    trials = []
    for rate, minimum in ((2., 2), (4., 3)):
        for seed in seeds:
            for family in FAMILIES:
                requests = trace_requests(records, family=family, rate=rate, duration=duration_s, seed=seed)
                rate_metadata = trace_rate_metadata(family=family, rate=rate, duration=duration_s)
                trace = dict(kind='pdblend_low_m_tuning_trace_v2', model_id=model_id, dataset='sharegpt',
                    selection_split='tuning', evaluation_used_for_selection=False, seed=seed,
                    family=family, nominal_rate_rps=rate, **rate_metadata,
                    duration_s=duration_s, slo=dict(ttft_s=5., tpot_s=.15), corpus=corpus_refs,
                    requests=[asdict(r) for r in requests])
                path = output/f'trace-r{rate:g}-s{seed}-{family}.json'; write_new(path, trace)
                for frequency in frequencies:
                    trial_id = f'r{rate:g}-m{minimum}-f{frequency}-s{seed}-{family}'
                    trials.append(dict(id=trial_id, seed=seed, family=family, nominal_rate_rps=rate,
                        **rate_metadata, trace=binding(path), context=context,
                        slo=trace['slo'], plan=dict(counts={'M':minimum,'L1':8-minimum},
                            f_P=recovery['safety_max_freq'], f_D=recovery['safety_max_freq'],
                            f_M=frequency, tau=0)))
    manifest = dict(kind=KIND, policy='pdblend', scope='independent_low_m_tuning/v2',
        formal_eligible=False, selection_split='tuning', evaluation_used_for_selection=False,
        identity=identity(loaded.model), profile=binding(Path(profile)),
        profile_frequencies=list(loaded.model.freqs), corpus=corpus_refs,
        workload_identity=family_identity, source_files=sources, context=context,
        recovery_policy=recovery, startup_plan=startup_plan,
        seeds=list(seeds), families=list(FAMILIES), duration_s=duration_s,
        rate_domain=RATE_DOMAIN, selection_seed=8801,
        selection_objective='nominal_selection_seed_service_tail_energy_j',
        energy_objective='eight_gpu_boards_service_and_tail', power_limit_w=350,
        certification_status='unmeasured', trials=trials)
    write_new(output/'manifest.json', manifest)
    return manifest


class TrialPlanner(PoolPlanner):
    """Uncertified experimental layout; never serialized as qualified capacity.

    Normal controller safety/recovery remains active. Model estimates screen the
    single trial layout only; final selection uses measured service+tail energy.
    """
    def __init__(self, model, config, trial):
        super().__init__(model, config)
        self.trial = trial
        self.minimum = trial['plan']['counts']['M']
        self.frequency = trial['plan']['f_M']
        self.max_rate = trial['nominal_rate_rps']*1.25
        self._trial_mode = False

    @property
    def capacity_reserve_enabled(self):
        # Experimental safety behavior, not qualified empirical evidence.
        return True

    def enforce_capacity_floor(self, plan, fc, *, force_canonical=False, preserve_restoration=False):
        counts = {role:n for role,n in plan.counts.items() if n}
        if set(counts)-{'M','L1'} or sum(counts.values()) != self.cfg.slots:
            raise ValueError('tuning canonical recovery requires the original M/L1 fleet')
        canonical = min(self.cfg.min_m_instances,self.cfg.slots)
        valid = (counts == self.trial['plan']['counts'] and plan.f_M == self.frequency
                 and plan.tau == 0 and 0 <= fc.rate_rps <= self.max_rate)
        if counts.get('M',0) >= canonical or (valid and not force_canonical):
            return plan
        restored = dict(M=canonical,L1=self.cfg.slots-canonical)
        detail = dict(plan.detail,unqualified_tuning_trial=self.trial['id'],
            capacity_estimate_available=False,capacity_insufficient=True,
            tuning_canonical_restoration=dict(from_counts=counts,to_counts=restored,
                reason='shield_capacity_reserve' if force_canonical else 'candidate_domain_exit'))
        return replace(plan,counts=restored,f_M=max(self.cfg.freqs),power_w=float('inf'),
                       ttft_s=float('inf'),tpot_s=float('inf'),detail=detail)

    def mixed_floor(self, fc):
        return self.minimum if self._trial_mode else self.cfg.min_m_instances

    def evaluate(self, counts, f_P, f_D, f_M, tau, fc, strict=True):
        low = counts.get('M', 0) < self.cfg.min_m_instances
        if low and (counts != self.trial['plan']['counts'] or f_M != self.frequency or tau != 0
                    or not 0 <= fc.rate_rps <= self.max_rate):
            return None
        self._trial_mode = low
        old_floor = self._floor; self._floor = None
        try:
            result = super().evaluate(counts, f_P, f_D, f_M, tau, fc, strict)
        finally:
            self._trial_mode = False; self._floor = old_floor
        if result:
            result.detail['unqualified_tuning_trial'] = self.trial['id']
        return result

    def plan(self, fc, current=None):
        target = self.trial['plan']
        result = self.evaluate(target['counts'], target['f_P'], target['f_D'], target['f_M'], 0, fc)
        if result is None:
            counts = {'M':self.cfg.min_m_instances,'L1':self.cfg.slots-self.cfg.min_m_instances}
            result = self.evaluate(counts, max(self.cfg.freqs), max(self.cfg.freqs), max(self.cfg.freqs), 0, fc, False)
        result = result or self.fallback(fc, current)
        return (replace(result,tp=current.tp,pp=current.pp,pool_id=current.pool_id,
                        generation=current.generation,profile_key=current.profile_key) if current else result)
