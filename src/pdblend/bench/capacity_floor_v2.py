"""Reproducible low-M evidence: independent traces, zero misses and full energy.

This grants a narrowly bound capacity domain, never profile qualification or a
claim of superiority to evaluation baselines. Stored ``passed`` flags are not
inputs to the verdict.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from .comparison_acceptance import _bound, _need, _equal, _finite, _drained, _state
from .comparison_campaign import binding
from .comparison_metering import summarize_comparison
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .low_m_tuning import KIND, FAMILIES, CONTEXT_KEYS, SOURCE_PACKAGES, RATE_DOMAIN, source_inventory, trace_requests, trace_rate_metadata, validate_recovery
from .resident_session import digest


def validate_manifest(path):
    manifest = _bound(binding(Path(path)))
    _need(manifest.get('kind') == KIND and manifest.get('policy') == 'pdblend'
          and manifest.get('scope') == 'independent_low_m_tuning/v2'
          and manifest.get('selection_split') == 'tuning'
          and manifest.get('evaluation_used_for_selection') is False
          and manifest.get('formal_eligible') is False, 'independent tuning scope required')
    seeds = manifest.get('seeds', [])
    _need(len(set(seeds)) == len(seeds) == 3 and 701 not in seeds and 8801 in seeds
          and all(type(seed) is int for seed in seeds), 'three independent non-evaluation seeds required')
    _need(manifest.get('rate_domain') == RATE_DOMAIN and manifest.get('selection_seed') == 8801
          and manifest.get('selection_objective') == 'nominal_selection_seed_service_tail_energy_j',
          'nominal workload domain and independent nominal energy objective required')
    _need(set(manifest.get('families', [])) == set(FAMILIES), 'complete low-M stress family required')
    _need(manifest.get('duration_s') == 150 and manifest.get('power_limit_w') == 350,
          'full service window and unchanged board power limit required')
    context = manifest.get('context', {})
    _need(set(context) == set(CONTEXT_KEYS), 'complete algorithm/workload/recovery identity required')
    _need(manifest.get('source_files') and digest(manifest['source_files']) == context['algorithm_source_sha256'],
          'algorithm source inventory differs')
    _need(all(name.startswith(tuple(package+'/' for package in SOURCE_PACKAGES)) and name.endswith('.py') and '..' not in Path(name).parts
              for name in manifest['source_files']), 'non-algorithm file in capacity source inventory')
    _need(digest(manifest['workload_identity']) == context['workload_family_sha256']
          and digest(dict(control=manifest['recovery_policy'],startup_plan=manifest['startup_plan']))
          == context['recovery_policy_sha256'], 'capacity context differs')
    _need(validate_recovery(manifest['recovery_policy']) == manifest['recovery_policy'],
          'complete canonical recovery controls must be explicitly bound')
    ceiling = manifest['recovery_policy']['safety_max_freq']
    _need(manifest['startup_plan'] == dict(counts={'M':4,'L1':4},f_P=ceiling,f_D=ceiling,f_M=ceiling,tau=0),
          'canonical M4 startup identity required')
    _bound(manifest['profile'])
    from pdblend.profile.query.versions import load_profile
    from pdblend.planner.transitions import identity
    loaded = load_profile(manifest['profile']['path'],system='pdblend',model_id=manifest['identity']['model_id'],
        tp=manifest['identity']['tp'],pp=manifest['identity']['pp'],usage='development')
    _need(identity(loaded.model) == manifest['identity'] and list(loaded.model.freqs) == manifest['profile_frequencies'],
          'selected tuning profile/frequency identity differs')
    from .capacity_workloads import corpus_inputs
    refs = manifest['corpus']
    records, corpus_manifest, checked = corpus_inputs(Path(refs['corpus_manifest']['path']).parent,
                                                     'sharegpt', 'tuning')
    _need(checked == refs and corpus_manifest['model_name'] == manifest['identity']['model_id'],
          'independent corpus source differs')
    _need(manifest['workload_identity'] == dict(model_id=manifest['identity']['model_id'], dataset='sharegpt',
          corpus_manifest=refs['corpus_manifest'], corpus_dataset=refs['corpus_dataset']), 'workload family differs')
    ids = []
    for trial in manifest['trials']:
        ids.append(trial['id'])
        _need(trial['seed'] in seeds and trial['family'] in FAMILIES and trial['context'] == context,
              'trial lies outside frozen tuning identity')
        _need(trial.get('slo') == dict(ttft_s=5.,tpot_s=.15), 'original ShareGPT SLO must remain unchanged')
        plan = trial['plan']; minimum = 2 if trial['nominal_rate_rps'] == 2. else 3
        _need(trial['nominal_rate_rps'] in (2., 4.) and plan['counts'] == {'M':minimum,'L1':8-minimum}
              and plan['tau'] == 0 and 0 < plan['f_M'] <= manifest['recovery_policy']['safety_max_freq'],
              'unapproved low-M candidate or frequency ceiling')
        trace = _bound(trial['trace'])
        rate_metadata = trace_rate_metadata(family=trial['family'], rate=trial['nominal_rate_rps'], duration=150.)
        _need(all(_equal(trial.get(key), value) and _equal(trace.get(key), value)
                  for key, value in rate_metadata.items()), 'tuning target rate stages, peak or mean differ')
        _need(trace.get('selection_split') == 'tuning' and trace.get('evaluation_used_for_selection') is False
              and trace.get('seed') == trial['seed'] and trace.get('family') == trial['family']
              and trace.get('duration_s') == 150 and trace.get('slo') == trial['slo']
              and trace.get('corpus') == refs and trace.get('rate_rps') == trial['rate_rps']
              and trace.get('nominal_rate_rps') == trial['nominal_rate_rps'], 'tuning trace metadata differs')
        from dataclasses import asdict
        replay = trace_requests(records, family=trial['family'], rate=trial['nominal_rate_rps'],
                                duration=150., seed=trial['seed'])
        _need(trace['requests'] == [asdict(row) for row in replay], 'trace does not replay the independent tuning generator')
    _need(ids and len(ids) == len(set(ids)), 'empty or duplicate tuning trials')
    return manifest


def validate_trial(path, *, manifest=None, manifest_ref=None):
    """Recompute from exact client tokens, eight-board samples and native release."""
    receipt = _bound(binding(Path(path)))
    _need(receipt.get('kind') == 'pdblend_low_m_trial_v2' and receipt.get('hardware_executed') is True
          and receipt.get('selection_split') == 'tuning' and receipt.get('evaluation_used_for_selection') is False,
          'real independent tuning receipt required')
    if manifest is None:
        _bound(receipt['manifest']); manifest = validate_manifest(receipt['manifest']['path'])
        manifest_ref = receipt['manifest']
    _need(receipt['manifest'] == manifest_ref and receipt.get('executed_context') == manifest['context'],
          'executed algorithm/workload/recovery context differs')
    matches = [trial for trial in manifest['trials'] if trial['id'] == receipt.get('trial_id')]
    _need(len(matches) == 1, 'unknown or duplicate trial')
    trial = matches[0]; trace = _bound(trial['trace']); refs = receipt['artifacts']
    _need(trial.get('slo') == dict(ttft_s=5.,tpot_s=.15), 'original ShareGPT SLO must remain unchanged')
    _need(receipt.get('actual_plan') == trial['plan'], 'executed candidate differs from trial')
    data = {name:_bound(refs[name], journal=name in ('outcomes', 'controller', 'frequency_readings','routes'),
                       power=name=='power') for name in ('requests','outcomes','controller','native_result',
                            'drain','native_cleanup','metering','power','frequency_readings','startup','reset',
                            'transition_measurements','routes')}
    raw = Path(refs['frequencies']['path']).read_bytes()
    import hashlib
    _need(hashlib.sha256(raw).hexdigest() == refs['frequencies']['sha256'], 'frequency evidence checksum differs')
    frequencies = [json.loads(line) for line in raw.splitlines() if line.strip()]
    native, meter, outcomes = data['native_result'], data['metering'], data['outcomes']
    origin = native['service_started_s']; boundary = origin+150.; end = meter['tail_end_s']
    _need(_finite(origin) and abs(native['service_ended_s']-boundary) <= 1e-6
          and boundary <= native['finished_s'] <= end and native.get('native_cleanup_complete') is True
          and not native.get('quarantined_instances'), 'service, terminal or native cleanup boundary differs')
    _need(len(outcomes) == len(trace['requests']) and outcomes, 'complete tuning terminal cohort required')
    for row, request in zip(sorted(outcomes, key=lambda r:r['idx']), trace['requests']):
        _need(row.get('idx') == request['idx'] and row.get('sampling_seed') == trial['seed']
              and row.get('arrival_s') == request['arrival_s'] and row.get('input_tokens') == len(request['prompt'])
              and row.get('max_tokens') == request['max_tokens'] and _finite(row.get('finished_s'))
              and origin+request['arrival_s'] <= row['submitted_s'] <= row['finished_s'] <= end,
              'actual request identity, sampling seed or metered terminal boundary differs')
    canonical = canonical_outcomes('pdblend', trace, outcomes, service_started_s=origin)
    metrics = reduce_comparison(trace, canonical, service_started_s=origin, duration_s=150.,
                                slo=(trial['slo']['ttft_s'],trial['slo']['tpot_s']))
    _need(_equal(metrics['request_metrics'], data['requests']), 'canonical tuning request rows differ')
    _need(metrics.get('token_timing_complete') is True and metrics['unresolved_requests'] == 0
          and metrics['successful_requests'] == metrics['joint_slo_requests'] == len(outcomes)
          and metrics['ttft_p99_s'] <= trial['slo']['ttft_s'] and metrics['tpot_p99_s'] <= trial['slo']['tpot_s'],
          'zero SLO misses, full success and original P99 constraints required')
    engine = receipt['engine_identity']; fleet = engine['fleet_gpu_uuids']
    instances = {row['instance_id']:row for row in engine['instances']}
    _need(len(fleet) == len(set(fleet)) == 8 and len(instances) == 8//manifest['identity']['tp']
          and all((row['tp'],row['pp']) == (manifest['identity']['tp'],manifest['identity']['pp'])
                  for row in instances.values()), 'eight-board exact TP/PP deployment required')
    from .comparison_native_acceptance import audit_native_startup, audit_native_reset
    audit_native_startup(dict(model_id=manifest['identity']['model_id']),engine,data['startup'],instances,refs)
    audit_native_reset(data['reset'],instances,origin)
    executed_source = _bound(data['startup']['source_manifest'])['files']
    algorithm = {name:sha for name,sha in executed_source.items()
                 if name.startswith(tuple(package+'/' for package in SOURCE_PACKAGES)) and name.endswith('.py')}
    _need(algorithm == manifest['source_files'], 'native executed algorithm differs from tuning source inventory')
    reduced = summarize_comparison(data['power'], gpu_uuids=fleet, origin_s=origin, duration_s=150., tail_end_s=end)
    _need(reduced['energy_comparable'] is True and _equal(reduced,meter)
          and _finite(reduced['energy_service_tail_j']), 'raw full service+tail energy is incomplete or differs')
    generations, times = _drained(data['drain'], instances)
    _need(times and native['finished_s'] <= min(times) <= max(times) <= end
          and abs(data['drain']['tail_end_s']-end) <= 1e-6, 'native rank cleanup lies outside measured tail')
    _need(set(data['native_cleanup']) == set(instances), 'native cleanup omitted a resident instance')
    for iid, cleanup in data['native_cleanup'].items():
        _need(cleanup.get('acknowledged') is True and cleanup.get('drained') is True
              and _state(cleanup,instances[iid]['tp'],cleanup.get('response_at_s')) == generations[iid]
              and native['request_finished_s'] <= cleanup['native_at_s'] <= native['finished_s'],
              'runner native cleanup or generation differs')
    from .comparison_pdblend_acceptance import _controller, _routes, _routing_roles
    events = data['controller']
    selected = dict(profile_key=json.dumps(manifest['identity']['profile_key'],sort_keys=True,separators=(',',':')),
        frequencies=manifest['profile_frequencies'],choice=dict(plan=manifest['startup_plan']))
    def experimental_floor(plan, _selected, slots):
        counts = {role:n for role,n in plan['counts'].items() if n}
        _need(set(counts) <= {'M','L1'} and sum(counts.values()) == slots and plan['tau']==0,
              'tuning only accepts candidate/canonical M-L1 layouts')
        if counts.get('M',0) >= min(4,slots): return min(4,slots)
        _need(counts == trial['plan']['counts'] and plan['f_M'] == trial['plan']['f_M'],
              'tuning low-M plan differs from explicit candidate')
        return trial['plan']['counts']['M']
    plans,completed = _controller(events,native,instances,engine,data['reset'],selected,
        data['transition_measurements'],capacity_floor_check=experimental_floor)
    _routes(data['routes'],outcomes,trace,instances,data['reset'],native,sampling_seed=trial['seed'])
    _routing_roles(data['routes'],events,instances)
    _need(plans and len(plans) == len(completed), 'every tuning plan requires completed physical transition')
    _need({key:plans[0].get(key) for key in manifest['startup_plan']} == manifest['startup_plan']
          and completed[0]['finished_s'] <= origin, 'measured tuning did not use bound canonical startup')
    actual_low_s = 0.
    for index, plan in enumerate(plans):
        counts = {k:v for k,v in plan['counts'].items() if v}
        plan_identity = plan.get('plan_identity',{})
        _need(plan_identity.get('tp') == manifest['identity']['tp'] and plan_identity.get('pp') == manifest['identity']['pp']
              and plan_identity.get('generation') == next(iter(data['reset']['generation'].values()))
              and plan_identity.get('profile_key') == json.dumps(manifest['identity']['profile_key'],sort_keys=True,separators=(',',':')),
              'tuning plan topology/generation/profile differs')
        low = counts.get('M',0) < 4
        _need(set(counts) <= {'M','L1'} and sum(counts.values()) == len(instances)
              and (not low or (counts == trial['plan']['counts'] and plan['f_M'] == trial['plan']['f_M']))
              and plan['f_M'] <= manifest['recovery_policy']['safety_max_freq'],
              'tuning controller left candidate/canonical recovery layouts or frequency ceiling')
        if low:
            left = max(origin,completed[index]['finished_s']+1.)
            right = min(boundary, completed[index+1]['started_s'] if index+1<len(completed) else boundary)
            actual_low_s += max(0.,right-left)
    _need(actual_low_s >= 30., 'candidate lacks thirty seconds of actual stable low-M exposure')
    from .comparison_pdblend_acceptance import _frequencies
    _frequencies(frequencies, plans, completed, instances, engine, origin, readings=data['frequency_readings'])
    service_readings = [row for row in data['frequency_readings'] if origin<=row.get('read_started_s',-1)
                        and row.get('read_finished_s',float('inf'))<=end]
    _need({row.get('gpu') for row in service_readings} == set(range(8))
          and all(row.get('power_limit_w') == 350 for row in service_readings),
          'unchanged 350W board power limit lacks actual observations')
    return dict(trial_id=trial['id'], seed=trial['seed'], family=trial['family'],
        nominal_rate_rps=trial['nominal_rate_rps'], rate_rps=trial['rate_rps'],
        min_m_instances=trial['plan']['counts']['M'], frequency_mhz=trial['plan']['f_M'],
        energy_service_tail_j=reduced['energy_service_tail_j'], offered_requests=len(outcomes),
        actual_offered_rate_rps=len(outcomes)/150.,
        joint_slo_requests=metrics['joint_slo_requests'], stable_low_m_s=actual_low_s,
        input_range=[min(len(r['prompt']) for r in trace['requests']),max(len(r['prompt']) for r in trace['requests'])],
        output_range=[min(r['max_tokens'] for r in trace['requests']),max(r['max_tokens'] for r in trace['requests'])],
        receipt=binding(Path(path)))


def summarize(manifest_path, receipt_paths):
    manifest = validate_manifest(manifest_path); manifest_ref = binding(Path(manifest_path))
    rows, rejected, seen = [], [], set()
    for path in receipt_paths:
        try:
            row = validate_trial(path, manifest=manifest, manifest_ref=manifest_ref)
            _need(row['trial_id'] not in seen, 'duplicate accepted trial receipt')
            seen.add(row['trial_id']); rows.append(row)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            rejected.append(dict(path=str(path), reason=str(exc)))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['nominal_rate_rps'],row['min_m_instances'],row['frequency_mhz'])].append(row)
    required = {(seed,family) for seed in manifest['seeds'] for family in FAMILIES}
    eligible = []
    for (rate,minimum,frequency), values in sorted(groups.items()):
        if {(row['seed'],row['family']) for row in values} == required:
            eligible.append(dict(nominal_rate_rps=rate,min_m_instances=minimum,frequency_mhz=frequency,
                rate_domain=RATE_DOMAIN,rate_range=[0.,1.25*rate], input_range=[min(r['input_range'][0] for r in values),max(r['input_range'][1] for r in values)],
                output_range=[min(r['output_range'][0] for r in values),max(r['output_range'][1] for r in values)],
                slo=dict(ttft_s=5.,tpot_s=.15),
                objective_energy_service_tail_j=next(r['energy_service_tail_j'] for r in values
                    if r['family']=='nominal' and r['seed']==manifest['selection_seed']),
                nominal_mean_energy_service_tail_j=mean(r['energy_service_tail_j'] for r in values if r['family']=='nominal'),
                stress_mean_energy_service_tail_j=mean(r['energy_service_tail_j'] for r in values if r['family']!='nominal'),
                trial_ids=sorted(r['trial_id'] for r in values)))
    selected = []
    for rate in sorted({row['nominal_rate_rps'] for row in eligible}):
        selected.append(min((row for row in eligible if row['nominal_rate_rps']==rate),
                            key=lambda row:(row['objective_energy_service_tail_j'],row['frequency_mhz'])))
    return dict(kind='pdblend_low_m_summary_v2',manifest=manifest_ref,rows=rows,rejected=rejected,
        eligible=eligible,selected=selected,formal_eligible=False,
        missing_trial_ids=sorted(t['id'] for t in manifest['trials'] if t['id'] not in seen))


def load_v2_floors(path, *, model):
    from pdblend.planner.pool import QualifiedCapacityFloor
    from pdblend.planner.transitions import identity
    path = Path(path); artifact = _bound(binding(path))
    _need(artifact.get('kind') == 'pdblend_capacity_floor_v2' and artifact.get('identity') == identity(model),
          'capacity-floor model/TP/profile identity mismatch')
    manifest = _bound(artifact['tuning_manifest'])
    refs = artifact.get('trial_receipts',[])
    for ref in refs: _bound(ref)
    verified = summarize(artifact['tuning_manifest']['path'], [ref['path'] for ref in refs])
    _need(verified['selected'] and artifact.get('floors') == verified['selected'],
          'capacity floor lacks complete independent zero-miss/full-energy evidence')
    _need(manifest['identity'] == identity(model), 'capacity tuning profile identity differs')
    return tuple(QualifiedCapacityFloor(identity(model)['model_id'],model.tp,model.pp,row['min_m_instances'],
        tuple(row['rate_range']),tuple(row['input_range']),tuple(row['output_range']),
        str(path.resolve())+'#sha256='+binding(path)['sha256'],qualified=True,
        profile_key=json.dumps(model.profile_key,sort_keys=True,separators=(',',':')),
        accepted_slo=(row['slo']['ttft_s'],row['slo']['tpot_s']),version=2,
        frequency_mhz=row['frequency_mhz'],context=dict(manifest['context'],nominal_rate_rps=row['nominal_rate_rps']))
        for row in verified['selected'])


def runtime_context(path, *, runtime_options, workload_family_sha256=None, nominal_rate_rps=None, source_root=None):
    """Fail closed unless executing code, actual corpus family and recovery match."""
    artifact = _bound(binding(Path(path)))
    if artifact.get('kind') != 'pdblend_capacity_floor_v2':
        return {}
    manifest = _bound(artifact['tuning_manifest'])
    from .pdblend_runtime_options import CONTROL_OPTIONS, control_options
    from pdblend.online.policies import get_policy
    runtime_options = control_options(get_policy('pdblend'),{key:runtime_options[key]
        for key in CONTROL_OPTIONS if key in runtime_options})
    context = dict(algorithm_source_sha256=digest(source_inventory(source_root)),
        workload_family_sha256=workload_family_sha256,
        recovery_policy_sha256=digest(dict(control={k:runtime_options.get(k) for k in manifest['recovery_policy']},
            startup_plan=dict(counts={'M':4,'L1':4},f_P=runtime_options.get('safety_max_freq'),
                f_D=runtime_options.get('safety_max_freq'),f_M=runtime_options.get('safety_max_freq'),tau=0))))
    return (dict(context,nominal_rate_rps=nominal_rate_rps) if context == manifest['context']
            and nominal_rate_rps in {trial['nominal_rate_rps'] for trial in manifest['trials']} else {})
