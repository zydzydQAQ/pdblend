"""Dispatch shared traces to the five independent system implementations.

Public metering, traces and engine lifecycle may be shared. Policy objects,
profiles and deployment choices are supplied only to their owning runner.
This module deliberately does not call the historical matrix policy registry.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .client import Request

RUNNERS = {
    'mixed':'pdblend.bench.native_mixed.execute',
    'distserve':'pdblend_baselines.distserve.run_native.execute',
    'dynamollm':'pdblend_baselines.dynamollm.run_v1.execute',
    'ecoserve':'pdblend_baselines.ecoserve.run_native.execute',
    'pdblend':'pdblend.bench.run._point',
}
REQUIRED_GATES = {'source_identity','profile_calibration','workload_coverage','mechanisms','energy_protocol'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound(binding):
    path = Path(binding['path'])
    if sha(path) != binding['sha256']:
        raise ValueError('input checksum mismatch: '+str(path))
    return json.loads(path.read_text())


def validate(point, inputs):
    """Read-only; reject substituted traces/profiles before any engine launch."""
    system = point['system']
    duration = point.get('duration_s')
    if system not in RUNNERS or point.get('seed') != 701 or duration not in (150, 300):
        raise ValueError('active comparison requires a supported native system and seed701/150s or 300s')
    trace = load_bound(inputs['trace'])
    if (trace.get('seed') != point['seed'] or trace.get('model_id') != point['model_id']
            or trace.get('dataset') != point['dataset'] or trace.get('duration_s') != duration
            or trace.get('selection_split') != 'evaluation' or trace.get('slo') != point['slo']
            or trace.get('rate_rps') != point['rate_rps'] or not trace.get('requests')):
        raise ValueError('shared frozen evaluation trace identity differs')
    for row in trace['requests']:
        if not 0 <= row['arrival_s'] < duration or not row['prompt'] or not 2 <= row['max_tokens'] <= 512:
            raise ValueError('invalid evaluation request')
    policy = load_bound(inputs['system_config'])
    if policy.get('system') != system or policy.get('model_id') != point['model_id']:
        raise ValueError('independent system configuration identity differs')
    profiles = []
    if system != 'mixed' and not inputs.get('profiles'):
        raise ValueError('missing_profile: own system profile required')
    for binding in inputs.get('profiles', []):
        profile = load_bound(binding)
        key = profile.get('profile_key', {})
        profile_model = profile.get('model_id', profile.get('model', key.get('model_id')))
        # Historical PD profiles store the actual weight directory. Match the
        # development loader's basename rule only for the explicit observation
        # scope; default/formal and every baseline retain their exact identity.
        scope = 'pdblend_profile_unqualified_evaluation/v1'
        if (system == 'pdblend' and point.get('observation_scope') == scope
                and point.get('qualification_mode') == scope and isinstance(profile_model, str)):
            profile_model = Path(profile_model).name
        if (profile.get('system', key.get('system')) != system
                or profile_model != point['model_id']):
            raise ValueError('cross-system or cross-model profile is forbidden')
        profiles.append(profile)
    if system in ('pdblend','distserve'):
        choice = load_bound(inputs['offline_choice'])
        if (choice.get('system') != system or choice.get('model_id') != point['model_id']
                or choice.get('selection_split') not in ('calibration','tuning')
                or choice.get('evaluation_used_for_selection') is not False):
            raise ValueError('offline choice must be bound to calibration/tuning only')
        if system == 'pdblend':
            prior = load_bound(inputs['planning_trace'])
            if (prior.get('selection_split') not in ('calibration','tuning')
                    or prior.get('model_id') != point['model_id'] or not prior.get('requests')):
                raise ValueError('PDBlend bootstrap needs its frozen calibration/tuning trace')
    gates = []
    for binding in inputs.get('qualifications', []):
        receipt = load_bound(binding)
        if receipt.get('system') != system or receipt.get('model_id') != point['model_id']:
            raise ValueError('qualification belongs to another system/model')
        if receipt.get('gate') == 'workload_coverage' and receipt.get('trace_sha256') != inputs['trace']['sha256']:
            raise ValueError('coverage qualification belongs to another trace')
        gates.append(dict(path=binding['path'], sha256=binding['sha256'],
                          gate=receipt.get('gate'),
                          passed=receipt.get('formal_eligible') is True))
    return dict(system=system, native_runner=RUNNERS[system], trace=trace, config=policy,
        own_profile_count=len(profiles), qualification_bindings=gates,
        formal_eligible={g['gate'] for g in gates} >= REQUIRED_GATES and all(g['passed'] for g in gates),
        hardware_executed=False)


@dataclass
class Resources:
    """Caller-owned lifecycle; Dynamo's existing runner owns its own lifecycle."""
    specs: list
    fleet: Any = None
    meter: Any = None
    pd_model: Any = None
    pd_plan: Any = None
    pd_pool_models: Any = None
    pd_planning_trace: Any = None
    proxy_port: int = 18080
    dynamo_session: Any = None
    comparison_record_tokens: bool = False


def request_rows(trace):
    return [Request(row.get('idx', i), row['arrival_s'], row['prompt'], row['max_tokens'],
                    row.get('source', trace.get('dataset', '')))
            for i, row in enumerate(trace['requests'])]


async def execute(point, inputs, resources: Resources, out: Path, *, runner_overrides=None):
    """Execute a prepared point; functional success does not open formal gates.

    Lifecycle/meter owners wrap this call from before cold start through drain
    and cleanup. Tests inject native runners to verify routing independently of
    CUDA; production imports the exact module named in RUNNERS.
    """
    from .comparison_pdblend_observation import observation_requested, validate_observation_inputs
    observation = observation_requested(point)
    audit = validate_observation_inputs(point, inputs) if observation else validate(point, inputs)
    if not observation and not audit['formal_eligible']:
        raise ValueError('inconclusive: qualification receipts do not authorize this formal point')
    system, trace, config = audit['system'], audit['trace'], audit['config']
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('refusing to overwrite independent execution')
    if runner_overrides and system in runner_overrides:
        runner = runner_overrides[system]
    else:
        import importlib
        module, name = RUNNERS[system].rsplit('.', 1)
        runner = getattr(importlib.import_module(module), name)
    trace_path = Path(inputs['trace']['path'])
    duration, slo = point['duration_s'], point['slo']
    if system == 'mixed':
        return await runner(resources.specs, request_rows(trace), out,
            duration_s=duration, slo=(slo['ttft_s'], slo['tpot_s']), seed=701)
    if system == 'ecoserve':
        out.parent.mkdir(parents=True, exist_ok=True)
        return await runner(config, {s.instance_id:s.base_url for s in resources.specs},
                            trace_path, out, duration)
    if system == 'dynamollm':
        from pdblend_baselines.dynamollm.run_v1 import load_trace
        from pdblend_baselines.dynamollm.validation import preflight
        # Preserve original controller periods even though this short window
        # cannot qualify the 1800-second ScaleInst mechanism by itself.
        config = dict(config, dynamo_require_full_mechanisms=True)
        receipt = preflight(config, mode='comparison', duration_s=duration, seed=701)
        if not receipt.get('ready'):
            raise ValueError('Dynamo native preflight failed: '+json.dumps(receipt))
        out.mkdir(parents=True, exist_ok=True)
        if resources.dynamo_session is not None:
            from pdblend_baselines.dynamollm.run_v1 import execute_on_resident
            return await execute_on_resident(config, load_trace(trace_path, duration), output=out,
                duration_s=duration, mode='comparison', receipt=receipt, session=resources.dynamo_session)
        return await runner(config, load_trace(trace_path, duration), output=out,
                            duration_s=duration, mode='comparison', receipt=receipt)
    if system == 'distserve':
        choice = load_bound(inputs['offline_choice'])
        pairs = choice['deployment']['pairs']
        by_id = {s.instance_id:s for s in resources.specs}
        if not pairs:
            raise ValueError('DistServe offline choice has no native P/D placement')
        if len(pairs) != 1:
            # Multi-pair placement needs the independent DistServe global
            # prefill-load router, never a generic policy or request dropping.
            from pdblend_baselines.distserve.deployment import execute_deployment
            return await execute_deployment(choice, resources.specs, trace_path, out, duration)
        pair = pairs[0]; p,d = by_id[pair['prefill']], by_id[pair['decode']]
        if p.tp != d.tp or p.pp != 1 or d.pp != 1 or set(p.gpus) & set(d.gpus):
            raise ValueError('unsupported_engine: first batch requires symmetric TP PP1')
        return await runner(SimpleNamespace(trace=trace_path, out=out, duration=duration,
            tp=p.tp, pp=1, prefill_url=p.base_url, decode_url=d.base_url,
            prefill_address=p.zmq_address, decode_address=d.zmq_address,
            max_batch_size=config['max_batch_size'], request_timeout=180.))
    if system == 'pdblend':
        from pdblend.control.policies import get_policy
        from pdblend.control.planner import SLO
        if resources.pd_model is None or resources.pd_plan is None:
            raise ValueError('missing independent PDBlend model or deployed offline choice')
        prior = request_rows(load_bound(inputs['planning_trace']))
        out.mkdir(parents=True, exist_ok=True)
        return await runner(resources.fleet, resources.meter, resources.pd_model,
            get_policy('pdblend'), SLO(slo['ttft_s'], slo['tpot_s']), request_rows(trace), [], out,
            resources.proxy_port, 10., 300., sampling_seed=701, initial_plan=resources.pd_plan,
            pool_models=resources.pd_pool_models, planning_trace=prior,
            observation_duration_s=duration,
            **({'comparison_wait_initial_plan': True} if observation else {}),
            **({'comparison_record_tokens': True} if resources.comparison_record_tokens else {}))
    raise AssertionError(system)
