"""Explicit PD-only calibration frequency revisions; no hardware qualification.

Requested clocks are design coordinates. Observed clocks still need the same
raw coverage and +/-30 MHz checks. A new domain cannot reuse an old component
merely because one endpoint, model name or source file happens to match.
"""
from __future__ import annotations
from copy import deepcopy

from .native_timing_plan import digest, read_bound

SCHEMA = 'pdblend-native-frequency-domain/v1'
LEGACY_FREQUENCIES = (1500, 2520)
FIELDS = ('frequency_domain', 'frequency_domain_sha256')
MODEL_TP = {'Qwen2.5-7B-Instruct':1, 'Qwen2.5-14B-Instruct':1, 'Qwen2.5-32B-Instruct':2}
MODELS = tuple(MODEL_TP)


def _need(test, message):
    if not test:
        raise ValueError(message)


def make_domain(*, model_id, high_mhz, revision):
    _need(model_id in MODELS, 'frequency revision requires a model-owned PD TP topology')
    _need(type(high_mhz) is int and 1500 < high_mhz < 2520 and high_mhz % 15 == 0,
          'explicit lower high-frequency candidate in 15 MHz increments required')
    _need(isinstance(revision, str) and revision.strip() == revision and bool(revision),
          'explicit PD revision name required')
    _need(model_id!='Qwen2.5-32B-Instruct' or high_mhz==2100,
          '32B TP2 revision currently declares only 1500/2100 MHz')
    return dict(schema=SCHEMA, system='pdblend', model_id=model_id, tp=MODEL_TP[model_id], pp=1,
        revision=revision, frequencies_mhz=[1500, high_mhz], observed_tolerance_mhz=30,
        maximum_observation_gap_s=1., min_m_instances=4, engine_max_num_seqs=32,
        policy='pdblend', online_policy_changed=False, evaluation_used_for_selection=False,
        hardware_qualified=False, formal_eligible=False,
        scope='new_exact_frequency_calibration_design_not_old_profile_relabeling')


def validate_domain(domain):
    _need(isinstance(domain, dict) and domain.get('schema') == SCHEMA,
          'unknown explicit PD frequency domain')
    freqs=domain.get('frequencies_mhz', [])
    _need(isinstance(freqs, list) and len(freqs) == 2, 'two explicit frequency endpoints required')
    expected=make_domain(model_id=domain.get('model_id'), high_mhz=freqs[1], revision=domain.get('revision'))
    _need(domain == expected, 'frequency domain changes policy, thresholds or qualification')
    return deepcopy(domain)


def domain_fields(domain):
    domain=validate_domain(domain)
    return dict(frequency_domain=domain, frequency_domain_sha256=digest(domain))


def identity_frequencies(identity):
    present=[key in identity for key in FIELDS]
    _need(all(present) or not any(present), 'partial frequency domain identity')
    if not any(present):
        return LEGACY_FREQUENCIES
    domain=validate_domain(identity['frequency_domain'])
    _need(identity['frequency_domain_sha256'] == digest(domain), 'frequency domain identity checksum differs')
    _need(all(identity.get(k) == domain[k] for k in ('model_id', 'tp', 'pp')),
          'frequency domain model/topology identity differs')
    return tuple(domain['frequencies_mhz'])


def with_domain(identity, domain):
    result=dict(identity, **domain_fields(domain))
    identity_frequencies(result)
    return result


def require_same_domain(left, right):
    identity_frequencies(left);identity_frequencies(right)
    _need(all(left.get(key) == right.get(key) for key in FIELDS),
          'frequency domain differs; legacy runtime/power/timing cannot be mixed with a new revision')


def plan_frequencies(plan):
    return identity_frequencies(plan)


def point_fields(plan):
    plan_frequencies(plan)
    return ({'frequency_domain_sha256':plan['frequency_domain_sha256']}
            if 'frequency_domain' in plan else {})


def check_rows(rows, identity):
    frequencies=identity_frequencies(identity)
    for row in rows:
        _need(row.get('frequency_mhz') in frequencies, 'timing row outside exact frequency domain')
        _need(row.get('frequency_domain_sha256') == identity.get('frequency_domain_sha256'),
              'timing row domain missing or differs; old observations cannot be relabeled')
    return frequencies


def validate_collection_inputs(plan, inputs, *, collect_runtime=False, power_pilot=False,
                               request_cycles=False, layout_energy=False, bound_reader=read_bound):
    """CPU preflight before model load; unsupported supplements fail early."""
    plan_frequencies(plan)
    if 'frequency_domain' not in plan:
        _need('frequency_domain_ref' not in plan and not any(k in inputs for k in (*FIELDS,'frequency_domain_ref')),
              'legacy timing cannot declare an unbound new domain')
        return LEGACY_FREQUENCIES
    ref=inputs.get('frequency_domain_ref')
    _need(ref == plan.get('frequency_domain_ref') and ref, 'input/plan frequency domain binding differs')
    _need(validate_domain(bound_reader(ref)) == plan['frequency_domain'], 'bound frequency domain bytes differ')
    require_same_domain(plan, inputs)
    _need(inputs.get('timing_first') is True, 'new frequency domain requires explicit timing-first revision')
    if collect_runtime:
        from .native_runtime_collect import build_runtime_plan
        runtime_plan=build_runtime_plan(ref)
        _need(inputs.get('runtime_include_transfer') is False and inputs.get('runtime_plan')==runtime_plan
              and inputs.get('runtime_scope')==runtime_plan['scope'],
              'new frequency runtime requires bound scoped plan and runtime_include_transfer=False before GPU load')
    else:
        _need(not any(k in inputs for k in ('runtime_plan','runtime_include_transfer','runtime_scope')),
              'new frequency timing cannot carry undeclared runtime inputs')
    _need(not any((power_pilot, request_cycles)),
          'new frequency domain power/cycle supplements remain blocked before GPU load')
    if layout_energy:
        _need(plan['model_id']=='Qwen2.5-32B-Instruct' and plan['tp']==2 and collect_runtime,
              'new frequency layout requires the explicit 32B TP2 runtime/timing chain before GPU load')
    return plan_frequencies(plan)
