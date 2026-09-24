"""Original baseline execution with explicitly unqualified profile inputs.

The resident Dynamo implementation is loaded unchanged in a private module
namespace. Only its configuration/qualification entry points are injected; the
ordinary module, lifecycle, controller, topology hooks and periods are untouched.
The common meter remains owned by the comparison adapter across all windows.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import time

from .comparison_campaign import PROTOCOL, binding, load_bound
from .comparison_acceptance import _need
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import summarize_comparison
from .resident_session import digest, engine_signature, file_sha, write_new
from pdblend.results.power_archive import write_power_archive

SCOPE = 'baseline_profile_unqualified_evaluation/v1'
SCHEMA = 'baseline-observation-acceptance/v1'
PROFILE_GAPS = {
    'distserve': ['own_profile_holdout_and_interference_unqualified',
                 'fixed_symmetric_deployment_not_offline_optimum'],
    'dynamollm': ['partial_single_observation_profile_coverage',
                 'original_cycle_and_stationary_weight_mechanisms_unqualified'],
}
RESULT_POLICY = 'all_recorded_windows/v1'
COMMON_GATES = {'inputs.immutable_observation', 'binding.raw', 'native.window',
                'native.bound_window', 'metrics.client_reduction', 'metering.raw_eight_gpu_window'}


def observation_requested(point):
    return point.get('observation_scope') == SCOPE


def validate_observation_inputs(point, identity):
    """Validate identities and fixed work without granting profile qualification."""
    system = point.get('system')
    _need(system in PROFILE_GAPS and observation_requested(point)
          and point.get('qualification_mode') == SCOPE,
          'explicit original-baseline observation scope required')
    _need(point.get('seed') == 701 and point.get('duration_s') == 150,
          'observation requires the frozen 150-second seed-701 protocol')
    engine_signature(identity)
    fleet = identity.get('fleet_gpu_uuids', [])
    _need(len(fleet) == len(set(fleet)) == 8
          and [u for row in identity['instances'] for u in row['gpu_uuids']] == fleet,
          'the initial inventory must cover all eight metered physical GPUs')
    _need(point.get('engine_identity') == identity, 'point engine identity differs')
    inputs = point['inputs']
    _need(inputs.get('trace') == point['trace'], 'point/input frozen traces differ')
    trace, config = load_bound(point['trace']), load_bound(inputs['system_config'])
    for key in ('model_id', 'dataset', 'rate_rps', 'slo', 'seed', 'duration_s'):
        _need(trace.get(key) == point.get(key), 'frozen trace differs: ' + key)
    _need(trace.get('selection_split') == 'evaluation', 'independent evaluation trace required')
    from pdblend_baselines.dynamollm.run_v1 import load_trace
    load_trace(point['trace']['path'], 150)  # Exact arrivals, prompts and output budgets.
    _need(config.get('system') == system and config.get('model_id') == point['model_id']
          and config.get('observation_scope') == SCOPE and config.get('formal_eligible') is False,
          'configuration must explicitly retain its unqualified observation scope')
    profiles = inputs.get('profiles', [])
    _need(bool(profiles), 'own immutable profile references required')
    for ref in profiles:
        value = load_bound(ref)
        key = value.get('profile_key', {})
        profile_model = value.get('model_id', value.get('model', key.get('model_id')))
        _need(value.get('system', key.get('system')) == system
              and isinstance(profile_model, str) and Path(profile_model).name == point['model_id'],
              'profile belongs to a different system or model')
    result = dict(config=config, trace=trace, formal_eligible=False, profile_qualified=False,
                  observation_scope=SCOPE, profile_missing_gates=list(PROFILE_GAPS[system]))
    if system == 'distserve':
        from .comparison_native_acceptance import native_topology
        instances = native_topology(point, identity)
        choice = load_bound(inputs['offline_choice'])
        selected = choice['selected']; tp, replicas = selected['tp'], selected['replicas']
        _need(choice.get('system') == system and choice.get('model_id') == point['model_id']
              and choice.get('status') == 'ready_for_native_execution'
              and choice.get('selection_split') == 'calibration'
              and choice.get('evaluation_used_for_selection') is False
              and choice.get('selection_used_dataset_requests') is False
              and choice.get('selection') == 'predeclared_fixed_native_topology_no_offline_search'
              and choice.get('offline_topology_search_performed') is False
              and choice.get('formal_eligible') is False,
              'DistServe observation must declare its fixed pre-evaluation deployment')
        _need(selected['config'] == [1, tp, 1, tp, 1] and selected['pp'] == 1
              and selected['total_gpu_count'] == 2 * tp * replicas == 8
              and set(instances) == {f'dist-{i}-{role}' for i in range(replicas) for role in ('P', 'D')}
              and all(r['tp'] == tp for r in instances.values()), 'DistServe fixed pair inventory differs')
        _need(choice['profiles'] == profiles and choice.get('trace') == point['trace']
              and choice.get('rate_rps') == point['rate_rps'] and choice.get('slo') == point['slo']
              and choice.get('frequency_mhz') == 2520 and choice.get('gpu_budget') == 8,
              'DistServe bound plan/profile/workload differs')
        _need(all(choice['identity'].get(k) == identity[k] for k in
                  ('model_hash', 'tokenizer_hash', 'image_digest'))
              and config.get('max_batch_size') == 32 and config.get('request_timeout_s') == 240.,
              'DistServe native identity or original queue limits differ')
        result['choice'] = choice
    else:
        from .comparison_dynamo_runtime import dynamo_launch_options, ENTRYPOINT, WORKER_EXTENSION
        from pdblend_baselines.dynamollm.policy import PERIODS
        _need(identity['entrypoint'] == ENTRYPOINT and identity['worker_extension'] == WORKER_EXTENSION
              and identity['dtype'] == 'bfloat16', 'Dynamo independent entrypoint/worker differs')
        _need(config.get('mode') == 'comparison' and config.get('dynamo_require_full_mechanisms') is False
              and config.get('periods_s') == PERIODS
              and not config.get('functional_profile_stage', {}).get('enabled'),
              'Dynamo must preserve comparison hooks and original real-time periods')
        _need(Path(config['trace']).resolve() == Path(point['trace']['path']).resolve()
              and config.get('slo_ttft_s') == point['slo']['ttft_s']
              and config.get('slo_tpot_s') == point['slo']['tpot_s'], 'Dynamo frozen work or SLO differs')
        _need(config.get('node_gpus') == list(range(8))
              and [g for row in config['instances'] for g in row['gpus']] == list(range(8)),
              'Dynamo initial inventory does not cover its eight-GPU lease')
        selected = [r for r in profiles if Path(r['path']).resolve() == Path(config['profiles']).resolve()]
        _need(len(selected) == 1, 'Dynamo selected profile is not uniquely bound')
        actual = {r.get('instance_id', r.get('id')): r for r in config['instances']}
        _need(set(actual) == {r['instance_id'] for r in identity['instances']}, 'Dynamo instances differ')
        for row in identity['instances']:
            native = actual[row['instance_id']]
            _need(native['tp'] == row['tp'] and native.get('pp', 1) == row['pp']
                  and [fleet[g] for g in native['gpus']] == row['gpu_uuids']
                  and row['launch_options'] == dynamo_launch_options(config), 'Dynamo placement/launch differs')
    return result


def _private_copy(module, suffix):
    """Load trusted frozen code in a separate namespace; never patch its owner."""
    name = module.__package__ + '._observation_' + suffix
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    private = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private)
    return private


def dynamo_resident_implementation():
    """Qualification dependency injection; the executable resident code is original."""
    from pdblend_baselines.dynamollm import resident
    private = _private_copy(resident, 'resident')
    original_config, original_preflight = private._config, private.preflight

    def config(value, mode):
        _need(mode == 'comparison' and value.get('observation_scope') == SCOPE
              and value.get('mode') == 'comparison' and value.get('formal_eligible') is False
              and value.get('dynamo_require_full_mechanisms') is False,
              'explicit unqualified comparison configuration required')
        result = original_config(value, mode)
        result['dynamo_require_full_mechanisms'] = False
        return result

    def preflight(value, *, mode, duration_s, seed):
        config(value, mode)
        _need(duration_s == 150 and seed == 701, 'original observational short-window protocol differs')
        assets = original_preflight(value, mode='functional', duration_s=duration_s, seed=seed)
        formal = original_preflight(value, mode='comparison', duration_s=duration_s, seed=seed)
        # ready denotes ONLY the unmodified functional asset validation. Preserve
        # the entire comparison report; do not relabel it ready or qualified.
        result = deepcopy(assets)
        result.update(observation_scope=SCOPE, readiness_scope='functional_assets_only',
            execution_mode='comparison', comparison_qualification=formal,
            formal_eligible=False, profile_qualified=False,
            profile_missing_gates=list(PROFILE_GAPS['dynamollm']))
        result['evidence'].update(observation_wrapper=binding(__file__),
                                 resident_implementation=binding(resident.__file__))
        return result

    private._config, private.preflight = config, preflight
    private.ResidentSession = _observational_session_class(private.ResidentSession)
    return private


async def execute_native_observation(point, identity, *, out, specs=None, session=None,
                                     base_port=None):
    """Run an original baseline; caller retains the resident fleet and common meter."""
    checked = validate_observation_inputs(point, identity)
    if point['system'] == 'distserve':
        from pdblend_baselines.distserve.deployment import execute_on_resident
        _need(specs is not None and session is None, 'DistServe requires its native resident fleet')
        return await execute_on_resident(checked['choice'], specs, Path(point['trace']['path']),
                                        Path(out), 150, request_timeout=240.)
    from .comparison_dynamo_runtime import lease_config
    from pdblend_baselines.dynamollm.run_v1 import execute_on_resident, load_trace
    _need(session is not None and specs is None and base_port is not None,
          'Dynamo requires its independent observational resident session')
    return await execute_on_resident(lease_config(checked['config'], base_port),
        load_trace(point['trace']['path'], 150), session=session, output=Path(out),
        duration_s=150, mode='comparison')


def restore_observational_reuse(session, raw):
    """Separate request qualification from the original successful native boundary.

    The native result is unchanged. No failed/incomplete boundary is retried or
    synthesized here; engine/layout errors must still close the original session.
    """
    origin, done = raw.get('service_started_s'), raw.get('requests_done_s')
    full = (type(origin) in (int, float) and type(done) in (int, float)
            and math.isfinite(origin) and math.isfinite(done) and done >= origin + 150)
    safe = (full and not session.closed and raw.get('own_cleanup_complete') is True
            and not raw.get('cleanup_errors') and bool(raw.get('resident_boundaries', {}).get('after')))
    restored = bool(safe and session.quarantined)
    if restored:
        session.quarantined = False
    return dict(schema='dynamo-observation-reuse/v1', result_policy=RESULT_POLICY,
        original_native_result_sha256=digest(raw), original_resident_reusable=raw.get('resident_reusable'),
        complete_service_window=full, original_after_boundary_completed=safe,
        qualification_only_quarantine_cleared=restored, reuse_permitted=safe,
        formal_eligible=False)


def _observational_session_class(original):
    """Delay only result-triggered close until the real native result is available.

    The original window still owns all request/controller/boundary operations.
    Unsafe or interrupted executions immediately take the original close path.
    """
    class ObservationSession(original):
        async def _close(self):
            if getattr(self, '_observation_window_active', False) and self.quarantined:
                self._observation_close_pending = True
                return
            await super()._close()

        async def execute_window(self, trace, *, output, **kwargs):
            _need(not getattr(self, '_observation_window_active', False), 'observational window already running')
            self._observation_window_active, self._observation_close_pending = True, False
            raw = None
            try:
                raw = await super().execute_window(trace, output=output, **kwargs)
            finally:
                self._observation_window_active = False
                if raw is None and self._observation_close_pending:
                    # Do not retain inventory after an interrupted result/serialization.
                    await super()._close()
            reuse = restore_observational_reuse(self, raw)
            close_error = None
            try:
                if self._observation_close_pending and not reuse['reuse_permitted']:
                    await super()._close()
            except BaseException as exc:
                close_error = repr(exc)
                raise
            finally:
                self.observation_reuse = reuse
                write_new(Path(output)/'observation-resident-cleanup.json', dict(
                    reuse=reuse, result_triggered_close_deferred=self._observation_close_pending,
                    session_closed=self.closed, close_error=close_error,
                    original_native_result_unchanged=True, formal_eligible=False))
            return raw

    return ObservationSession


def make_dynamo_adapter(out, *, base_port):
    """Reuse the original adapter's meter, startup identity, boundaries and close."""
    from . import comparison_dynamo_runtime as original
    implementation = dynamo_resident_implementation()
    adapter_module = _private_copy(original, 'dynamo_adapter')
    adapter_module.ResidentSession = implementation.ResidentSession

    class ObservationAdapter(adapter_module.DynamoResidentAdapter):
        def _prepare(self, point):
            checked = validate_observation_inputs(point, self.identity)
            value = implementation._config(original.lease_config(checked['config'], self.base_port), 'comparison')
            assets = implementation.preflight(value, mode='comparison', duration_s=150, seed=701)
            _need(assets['ready'], 'Dynamo independent asset preflight failed: ' + json.dumps(assets['missing_evidence']))
            return value, assets

        async def execute(self, point, out):
            self._point(point)
            raw = await execute_native_observation(point, self.identity, out=out,
                                                   session=self.dynamo_session, base_port=self.base_port)
            reuse = getattr(self.dynamo_session, 'observation_reuse', None)
            if reuse is None or reuse.get('original_native_result_sha256') != digest(raw):
                reuse = restore_observational_reuse(self.dynamo_session, raw)
            write_new(Path(out)/'observation-reuse.json', reuse)
            # execute_window already performs its own real all-rank boundary.
            # Never manufacture a fresh boundary for a quarantined/failed run.
            tail_end = time.time()
            await asyncio.sleep(.3)
            return finalize_observation(point, self.identity, out=out, native_result=raw,
                snapshot=self.monitor.snapshot(), tail_end_s=tail_end,
                startup=load_bound(binding(self.out/'qualification.json')),
                reset=self.observation_reset,
                drain=dict(passed=reuse['reuse_permitted'], observational_reuse=binding(Path(out)/'observation-reuse.json'),
                           states=raw.get('resident_boundaries', {}).get('after', {})))

        async def reset(self, point):
            self.observation_reset = await super().reset(point)
            return self.observation_reset

    return ObservationAdapter(out, base_port=base_port)


def finalize_observation(point, identity, *, out, native_result, snapshot, tail_end_s,
                         startup, reset, drain):
    """Reduce real raw evidence; never convert missing qualification into success.

    The caller takes the snapshot after a bracketing sample and owns cleanup.
    A startup/early failure yields an immutable failure report and empty metrics,
    never a synthetic full service window. Final outer drain remains mandatory.
    """
    from .comparison_runtime import read_native_measurement
    from .comparison_native_acceptance import audit_native_meter
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    refs = {'trace': point['trace']}; failures = {}; diagnostics = {}; gates = []; metrics = {}; checked = None
    def gate(name, fn, *, diagnostic=False):
        try:
            value = fn()
        except (ValueError, TypeError, KeyError, OSError, IndexError, AttributeError, RuntimeError, OverflowError) as exc:
            (diagnostics if diagnostic else failures)[name] = str(exc); return None
        if not diagnostic: gates.append(name)
        return value
    checked = gate('inputs.immutable_observation', lambda: validate_observation_inputs(point, identity))
    for name, filename, value in (('native_result', 'native-result.json', native_result),
            ('startup_qualification', 'observation-startup.json', startup),
            ('reset', 'observation-reset.json', reset), ('drain', 'native-drain.json', drain)):
        write_new(out/filename, value); refs[name] = binding(out/filename)
    # Use serialized raw identity (not int-key in-memory mappings) for replay.
    native_result = load_bound(refs['native_result'])
    write_power_archive(out/'power.json', snapshot); refs['power'] = binding(out/'power.json')
    for key, names in [('events', ('events.jsonl.gz', 'events.jsonl')),
                       ('outcomes', ('outcomes.jsonl.gz', 'outcomes.jsonl', 'outcomes.json'))]:
        path = next((out/n for n in names if (out/n).is_file()), None)
        if path is not None: refs[key] = binding(path)
    measurement = gate('native.window', lambda: read_native_measurement(point['system'], out, native_result))
    if measurement is not None:
        origin, outcomes, journal = measurement
        def window():
            _need(math.isfinite(tail_end_s) and tail_end_s >= origin + 150,
                  'actual observation did not cover the full service window')
            _need(native_result.get('system') == point['system']
                  and native_result.get('trace_sha256') == point['trace']['sha256']
                  and refs.get('events', {}).get('sha256') == native_result.get('events_sha256'),
                  'native system/trace/journal identity differs')
        gate('native.bound_window', window)
        try:
            if point['system'] == 'dynamollm':
                from .comparison_dynamo_runtime import bind_dynamo_request_indices
                outcomes, journal = bind_dynamo_request_indices(checked['trace'], outcomes, journal)
            rows = canonical_outcomes(point['system'], checked['trace'], outcomes, service_started_s=origin, journal=journal)
            metrics = reduce_comparison(checked['trace'], rows, service_started_s=origin, duration_s=150,
                slo=(point['slo']['ttft_s'], point['slo']['tpot_s']), observed_until_s=tail_end_s)
            request_metrics = metrics.pop('request_metrics')
            write_new(out/'comparison-requests.json', request_metrics)
            refs['canonical_requests'] = binding(out/'comparison-requests.json')
            gates.append('metrics.client_reduction')
            gate('metrics.client_canonical', lambda: _need(len(outcomes) == len(checked['trace']['requests'])
                and metrics['token_timing_complete'] and metrics['unresolved_requests'] == 0,
                'request cohort or exact client token times incomplete'), diagnostic=True)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            failures['metrics.client_reduction'] = str(exc)
        metering = gate('metering.raw_eight_gpu_window', lambda: summarize_comparison(snapshot,
            gpu_uuids=identity['fleet_gpu_uuids'], origin_s=origin, tail_end_s=tail_end_s, duration_s=150))
        if metering is not None:
            gate('metering.coverage', lambda: audit_native_meter(identity, snapshot, metering, origin), diagnostic=True)
            write_new(out/'comparison-metering.json', metering); refs['metering'] = binding(out/'comparison-metering.json')
            metrics.update({k: v for k, v in metering.items() if not isinstance(v, (dict, list))})
            metrics.update(tail_s=tail_end_s-origin-150, measurement_protocol_version=PROTOCOL,
                           gpu_util_coverage_fraction=metering['util_coverage_fraction'])
            for i, uuid in enumerate(identity['fleet_gpu_uuids']):
                gpu = metering['service']['utilization']['per_gpu'][uuid]
                metrics.update({f'gpu{i}_uuid': uuid, f'gpu{i}_util_mean_pct': gpu.get('mean_pct'),
                                f'gpu{i}_util_peak_pct': gpu.get('peak_pct')})
        gate('native.lifecycle', lambda: _need(reset.get('passed') is True and drain.get('passed') is True
            and not native_result.get('cleanup_errors'), 'native reset/cleanup/drain failed'), diagnostic=True)
    gate('binding.raw', lambda: [_need(file_sha(r['path']) == r['sha256'], 'raw evidence changed') for r in refs.values()])
    valid = not failures and COMMON_GATES <= set(gates)
    audit = dict(schema=SCHEMA, scope=SCOPE, point_sha256=digest(point), metrics_sha256=digest(metrics),
        evidence_valid=False, formal_eligible=False, profile_qualified=False,
        measurement_evidence_valid=valid, checked_gates=gates, missing_gates=list(failures), gate_failures=failures,
        result_policy=RESULT_POLICY, diagnostic_failures=diagnostics,
        profile_missing_gates=list(PROFILE_GAPS[point['system']]), optimality_established=False,
        raw_refs=refs, evidence_sha256=digest(refs))
    write_new(out/'observation-acceptance.json', audit)
    return dict(evidence_valid=False, formal_eligible=False, profile_qualified=False,
        measurement_evidence_valid=valid, observation_scope=SCOPE, observation_acceptance=audit,
        result_policy=RESULT_POLICY, recorded_window_complete=valid,
        metrics=metrics, missing_gates=list(failures), native_status=native_result.get('status'),
        native_error=native_result.get('error'), raw_refs=refs,
        identity=dict(model_hash=identity['model_hash'], tokenizer_hash=identity['tokenizer_hash'],
                      image_digest=identity['image_digest'], runtime_source_sha256=identity['runtime_source_sha256'],
                      measurement_source_sha256=identity['measurement_source_sha256'],
                      gpu_uuids=identity['fleet_gpu_uuids'], measurement_protocol_version=PROTOCOL))


def valid_observation_result(point, result):
    audit = result.get('observation_acceptance', {})
    return (observation_requested(point) and point.get('qualification_mode') == SCOPE
        and point.get('system') in PROFILE_GAPS and result.get('observation_scope') == SCOPE
        and all(result.get(k) is False and audit.get(k) is False for k in
                ('evidence_valid', 'formal_eligible', 'profile_qualified'))
        and result.get('measurement_evidence_valid') is True and audit.get('measurement_evidence_valid') is True
        and audit.get('schema') == SCHEMA and audit.get('scope') == SCOPE
        and result.get('result_policy') == audit.get('result_policy') == RESULT_POLICY
        and audit.get('point_sha256') == digest(point) and audit.get('metrics_sha256') == digest(result.get('metrics', {}))
        and audit.get('missing_gates') == [] and audit.get('gate_failures') == {}
        and COMMON_GATES <= set(audit.get('checked_gates', []))
        and audit.get('profile_missing_gates') == PROFILE_GAPS[point['system']]
        and audit.get('evidence_sha256') == digest(audit.get('raw_refs', {})))
