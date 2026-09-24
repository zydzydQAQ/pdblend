"""Read-only native component inventory; collection and qualification are distinct.

The inventory is a preparation input, never a profile selection. In particular,
an auxiliary holdout failure is retained even if another measured component can
be reused. No GPU, queue, historical receipt or selection is modified here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .native_timing_plan import binding, read_bound
from .native_runtime_audit import replay_runtime
from .native_timing_replay import Resolver
from pdblend.profile.query.native_composition import IDENTITY, _source_files, _translated


def read_snapshot(path):
    """Hash the bytes actually read, including a concurrently changing queue."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def terminal_jobs(queue):
    jobs = queue['jobs']
    jobs = jobs.values() if isinstance(jobs, dict) else jobs
    # Preserve only public task state; do not copy lease tokens or unrelated jobs.
    return [dict(job_id=j['job_id'], status=j['status'], attempts=j.get('attempts'),
                 last_error=j.get('last_error'), model_id=j['payload'].get('model_id'),
                 source_sha256=j['payload'].get('source_sha256'),
                 input_manifest=j['payload'].get('input_manifest'))
            for j in jobs if j['job_id'].startswith('pdblend-native-timing-')]


def runtime_readiness(reference):
    """Reconstruct holdout errors from bound raw data, never trust audit flags."""
    report = _translated(read_bound(reference), Resolver())
    audit = replay_runtime(report)
    limits = report.get('runtime_plan', {}).get('holdout_limits', {})
    comparisons = audit.get('holdout_comparisons', [])
    nonpositive = [dict(component=r['component'], metric=r['metric'],
                        training_prediction=r['training_prediction'])
                   for r in comparisons if r['training_prediction'] <= 0]
    exceeded = [r for r in comparisons if r['relative_error'] is None
                or r['relative_error'] > limits.get('max_relative_error', float('inf'))]
    caps = list(report.get('initial_capabilities', {}).values())
    identity = ({k: ('pdblend' if k == 'system' else caps[0].get(k)) for k in IDENTITY}
                if caps else None)
    return dict(completion=reference, identity=identity, raw_audit=audit,
                raw_components_complete=audit['raw_components_complete'],
                independent_holdout_passed=audit.get('holdout_passed', False),
                failed_max_error_nodes=exceeded, nonpositive_training_nodes=nonpositive,
                limits=limits, usable_as_full_runtime=False,
                full_runtime_blockers=([*audit['errors']]
                    + ([] if audit.get('holdout_passed') else ['independent_runtime_holdout_failed'])
                    + (['nonpositive_signed_handoff_predictor'] if nonpositive else [])
                    + ['model_source_domain_and_complete_composition_required']),
                reuse_scope='immutable_raw_evidence_only_until_consumer_replay_passes',
                measured_nontransfer_nodes=[r for r in comparisons
                                           if not r['component'].startswith('transfer_')])


def source_readiness(reference):
    source = _source_files(reference, Resolver())
    return dict(manifest=reference, source_sha256=source['source_sha256'],
                verified_files=len(source['files']))


def inspect_attempt(attempt, *, queue=None, evidence_dir=None):
    """Inspect one terminal native attempt without promoting partial artifacts."""
    attempt = Path(attempt).resolve()
    manifest, manifest_ref = read_snapshot(attempt/'manifest.json')
    root = attempt/'native-timing'
    completion, completion_ref = read_snapshot(root/'completion.json')
    inputs_ref = manifest['payload']['input_manifest']
    inputs = read_bound(inputs_ref)
    result = dict(attempt_manifest=manifest_ref, completion=completion_ref,
        input_manifest=inputs_ref, source=source_readiness(inputs['source_manifest']),
        model_id=inputs['model_id'], job_id=manifest['job_id'],
        status=completion.get('status'), complete=completion.get('complete', False),
        error=completion.get('error'), timing_stage=completion.get('resident_timing_stage'),
        measured_windows=completion.get('measured_windows'),
        raw_binding_count=len(completion.get('raw_bindings', {})),
        timing_component_claimed=completion.get('component_qualified', False),
        qualified_timing=False, qualified_power=False, full_profile_qualified=False,
        timing_blockers=['no_independently_replayed_terminal_timing_evidence'])
    if queue is not None and evidence_dir is not None and (
            completion.get('resident_timing_stage') or completion.get('complete') is True):
        # A new evidence manifest is allowed; original reports remain immutable.
        from .native_timing_replay import capture_evidence, replay_evidence
        from .native_timing_stage import capture_terminal_evidence, replay_terminal_evidence
        try:
            if completion.get('resident_timing_stage'):
                ref = capture_terminal_evidence(attempt, queue, Path(evidence_dir)/'evidence.json')
                audit = replay_terminal_evidence(ref)
            else:
                ref = capture_evidence(attempt, queue, Path(evidence_dir)/'evidence.json')
                audit = replay_evidence(ref)
            result['timing_evidence'] = ref
            result['qualified_timing'] = (audit.get('component') or {}).get('component_qualified') is True
            result['timing_blockers'] = ([] if result['qualified_timing']
                                          else ['timing_component_holdout_failed'])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            result['timing_blockers'] = [str(exc)]
    execution = attempt/'execution.json'
    if execution.exists():
        value, ref = read_snapshot(execution)
        result['execution'] = dict(binding=ref, status=value.get('status'),
                                   complete=value.get('complete'), returncode=value.get('returncode'))
    runtime = root/'runtime/completion.json'
    if runtime.exists():
        result['runtime'] = runtime_readiness(binding(runtime))
        identity = result['runtime']['identity']
        if identity and identity['model_id'] == 'Qwen2.5-32B-Instruct':
            # Existing formal consumer explicitly supports this predeclared
            # clock/capacity subset. It does not qualify transfer or another
            # frequency domain, even when its scoped holdout passes.
            from pdblend.profile.query.native_layout_profile import replay_layout_runtime_component
            try:
                scoped = replay_layout_runtime_component(binding(runtime), Resolver(), identity,
                    {result['source']['source_sha256']})
                result['runtime']['scoped_all_m_runtime'] = {
                    k: v for k, v in scoped.items() if k not in ('values', 'audit')}
                result['runtime']['scoped_all_m_runtime']['identity'] = identity
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                result['runtime']['scoped_all_m_runtime'] = dict(scoped_runtime_qualified=False, error=str(exc))
    result['auxiliaries'] = {}
    for name in ('power-pilot', 'request-cycles', 'layout-energy'):
        path = root/name/'completion.json'
        if not path.exists():
            continue
        value, ref = read_snapshot(path)
        result['auxiliaries'][name] = dict(completion=ref, schema=value.get('schema'),
            status=value.get('status'), complete=value.get('complete', False),
            error=value.get('error'), safe_restore_passed=value.get('safe_restore_passed'),
            claimed_component_qualified=value.get('component_qualified', False),
            usable_as_power_training_and_holdout=False,
            reason='pilot_or_unreplayed_partial_receipt_is_not_a_frozen_power_candidate')
    return result


def collection_reuse_decision(attempts):
    """Preparation may skip only an independently replayed qualified component."""
    timing = [a for a in attempts if a.get('qualified_timing') is True]
    return dict(collect_timing=not bool(timing), qualified_timing=timing,
                raw_runtime_to_retain=[a['runtime']['completion'] for a in attempts
                    if a.get('runtime', {}).get('raw_components_complete') is True],
                raw_runtime_qualifies_profile=False,
                power_pilot_qualifies_profile=False)


def verify_prepared_bindings(manifest):
    """Preparation fails closed on stale input/source refs before producing jobs."""
    plan = read_bound(manifest['point_plan'])
    source = source_readiness(manifest['source_manifest'])
    read_bound(plan['query_ledger'])
    read_bound(plan.get('query_provenance', plan.get('query_bindings')))
    if plan.get('frequency_domain_ref'):
        read_bound(plan['frequency_domain_ref'])
    return plan, source
