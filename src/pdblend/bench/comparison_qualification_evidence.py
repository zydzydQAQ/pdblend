"""Bound, descriptive history of component qualification attempts.

This is NOT a profile consumer or a raw qualification replay.  It proves which
immutable job/source/inputs produced a terminal report and quotes that report.
Engine, metering and query-domain equivalence to a comparison point is unknown.
No window, SLO, baseline freeze, or ranking flag is granted by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'comparison-qualification-attempt-evidence/v1'
SCOPE = 'historical_component_attempts_not_current_point_failures/v1'
COMPONENTS = ('resident_runtime', 'resident_power_pilot', 'resident_request_cycles', 'resident_layout_energy')
FIELDS = ('status', 'complete', 'hardware_executed', 'formal_eligible', 'component_qualified',
          'collection_complete', 'full_profile_qualified', 'operational_failure', 'error',
          'safe_restore', 'timing_inventory_ready', 'holdout_passed',
          'safe_restore_passed', 'ready_for_timing', 'runtime_holdout_passed',
          'runtime_measurements_valid', 'power_component_qualified', 'pure_power_component_qualified',
          'raw_components_complete', 'independent_holdout_collected', 'holdout_error_summary')
COLLECTORS = {'pdblend': {'pdblend.profile.collection.native_timing_collect'},
              'distserve': {'pdblend_baselines.distserve.stage_collect', 'pdblend_baselines.distserve.stage_cohort'}}


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ref(ref):
    _need(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'missing bound reference')
    _need(Path(ref['path']).is_absolute(), 'reference must be absolute')
    sha = ref.get('sha256')
    _need(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'invalid SHA256')
    return dict(path=str(Path(ref['path']).resolve()), sha256=sha)


def _read(ref):
    raw = Path(ref['path']).read_bytes()
    _need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'qualification evidence checksum differs: ' + ref['path'])
    return json.loads(raw)


def _arg(argv, flag):
    positions = [i for i, value in enumerate(argv) if value == flag]
    _need(len(positions) == 1 and positions[0] + 1 < len(argv), 'missing/duplicate argument: ' + flag)
    return argv[positions[0] + 1]


def _env(argv):
    rows = [argv[i + 1] for i, v in enumerate(argv[:-1]) if v == '-e' and '=' in argv[i + 1]]
    values = dict(v.split('=', 1) for v in rows)
    _need(len(rows) == len(values), 'duplicate environment key')
    return values


def host_path(argv, value):
    """Resolve a container path through the actual Docker volume bindings."""
    value = Path(value)
    _need(value.is_absolute(), 'container path is relative')
    matches = []
    for i, flag in enumerate(argv[:-1]):
        if flag not in ('-v', '--volume'):
            continue
        parts = argv[i + 1].split(':')
        _need(len(parts) in (2, 3), 'unsupported Docker volume')
        host, target = map(Path, parts[:2])
        if value.is_relative_to(target):
            matches.append((len(target.parts), host / value.relative_to(target)))
    _need(bool(matches), 'container path is not bound: ' + str(value))
    depth = max(item[0] for item in matches)
    paths = {str(p.resolve()) for n, p in matches if n == depth}
    _need(len(paths) == 1, 'ambiguous Docker volume binding')
    return paths.pop()


def _render(payload, manifest, attempt):
    ids = manifest['gpu_indices']
    uuids = manifest['gpu_uuids']
    _need(ids and len(ids) == len(uuids) == len(set(uuids)), 'invalid lease GPU inventory')
    replacements = {'{attempt_dir}': str(attempt), '{lease_id}': manifest['lease_id'],
                    '{lease_gpu_uuids}': ','.join(uuids), '{lease_gpu_indices}': ','.join(ids),
                    '{lease_local_indices}': ','.join(map(str, range(len(ids)))),
                    '{lease_port}': str(10000 + 100 * int(ids[0]))}
    argv = payload['argv'][:]
    _need(argv[:2] == ['docker', 'run'], 'unsupported execution wrapper')
    for old, new in replacements.items():
        argv = [value.replace(old, new) for value in argv]
    for i, value in enumerate(argv[:-1]):
        if value == '--gpus' and argv[i + 1] in ('all', '"device=all"'):
            argv[i + 1] = '"device=' + ','.join(uuids) + '"'
    _need(not any('{' in value or '}' in value for value in argv), 'unresolved job placeholder')
    return argv


@dataclass(frozen=True)
class QualificationEvidence:
    by_system_model: dict
    refs: tuple

    @property
    def watch_paths(self):
        return tuple(ref['path'] for ref in self.refs)


def load_qualification_evidence(ref, *, load_bound=None, file_sha=None):
    """Verify bound terminal metadata and return all revisions, never best-of-N.

    ``load_bound`` must verify a JSON reference's SHA before returning its value.
    ``file_sha`` can use the exporter's immutable-file cache for source bytes.
    ``refs`` includes every document/source file actually checked, for watching.
    Large native journals, power archives and weights are deliberately not read.
    Any provenance mismatch rejects the entire manifest, rather than silently
    dropping an inconvenient attempt. The external caller binds the manifest.
    """
    loader, hasher = load_bound or _read, file_sha or _file_sha
    refs, documents = {}, {}
    def remember(reference):
        reference = _ref(reference)
        old = refs.get(reference['path'])
        _need(old is None or old == reference, 'same path has conflicting evidence hashes')
        refs[reference['path']] = reference
        return reference
    def read(reference):
        reference = remember(reference)
        if reference['path'] not in documents:
            documents[reference['path']] = loader(reference)
        return documents[reference['path']]
    manifest_ref = remember(ref)
    manifest = read(manifest_ref)
    _need(manifest.get('schema') == SCHEMA and manifest.get('scope') == SCOPE,
          'unsupported qualification evidence scope')
    _need(manifest.get('formal_eligible') is False and manifest.get('current_point_failure') is False,
          'historical evidence cannot grant qualification or assert point failure')
    queue = read(manifest['terminal_queue_capture'])
    _need(queue.get('schema') == 'terminal-queue-metadata-capture/v1' and queue.get('capture_only') is True,
          'terminal queue metadata must be an explicitly derived capture')
    _need(all('token' not in lease for lease in queue.get('leases', {}).values()),
          'queue capture contains lease token field')
    attempts = manifest.get('attempts')
    _need(isinstance(attempts, list) and attempts, 'empty qualification history')
    seen, result = set(), {}
    for entry in attempts:
        m = read(entry['attempt_manifest'])
        attempt = Path(entry['attempt_manifest']['path']).resolve().parent
        _need(Path(entry['attempt_manifest']['path']).name == 'manifest.json', 'attempt manifest filename differs')
        _need(m.get('immutable') is True and m.get('schema') == 1, 'attempt manifest is not immutable')
        key = (m['job_id'], m['lease_id'])
        _need(key not in seen, 'duplicate qualification attempt')
        seen.add(key)
        _need(attempt.name == f"attempt-{m['attempt']:04d}-{m['lease_id']}" and attempt.parent.name == m['job_id'],
              'attempt directory identity differs')
        j, lease = queue['jobs'][m['job_id']], queue['leases'][m['lease_id']]
        _need(j.get('status') == lease.get('status') == 'failed' and j.get('attempts') == m['attempt'] > 0,
              'qualification attempt is not a terminal failed execution')
        _need(j.get('lease_id') is None and j.get('payload') == m['payload'], 'captured job payload/active lease differs')
        for name in ('job_id', 'lease_id', 'attempt', 'claimed_at', 'gpu_indices', 'gpu_uuids', 'owner', 'owner_pid'):
            _need(lease.get(name) == m.get(name), 'captured lease identity differs: ' + name)
        _need(Path(lease['attempt_dir']).resolve() == attempt, 'captured lease directory differs')
        execution = read(entry['execution'])
        _need(Path(entry['execution']['path']).resolve() == attempt / 'execution.json', 'execution belongs to another attempt')
        _need(execution.get('status') == 'failed' and execution.get('complete') is False
              and type(execution.get('returncode')) is int and execution['returncode'] != 0,
              'execution is not a terminal process failure')
        times = [m['claimed_at'], execution['started_s'], execution['finished_s'], queue['captured_at_s']]
        _need(all(type(x) in (int, float) and math.isfinite(x) for x in times) and times == sorted(times),
              'terminal execution timestamps differ')
        payload = m['payload']
        system, model = payload['system'], payload['model_id']
        _need(system in COLLECTORS, 'unsupported component system')
        argv = _render(payload, m, attempt)
        _need(argv == execution.get('argv'), 'actual execution argv differs from immutable lease payload')
        module = _arg(argv, '-m')
        _need(module in COLLECTORS[system], 'collector does not match system')
        _need(Path(_arg(argv, '--model')).name == model, 'collector model differs')
        env = _env(argv)
        source_ref, inputs_ref, plan_ref = map(_ref, (entry['source_manifest'], entry['input_manifest'], entry['point_plan']))
        for expected, actual in ((inputs_ref['path'], _arg(argv, '--input-manifest')),
                                 (plan_ref['path'], _arg(argv, '--point-plan')),
                                 (source_ref['path'], env['PDBLEND_SOURCE_MANIFEST'])):
            _need(host_path(argv, actual) == expected, 'executed source/input/plan path differs')
        source, inputs, plan = read(source_ref), read(inputs_ref), read(plan_ref)
        image = inputs.get('image_digest')
        _need(isinstance(image, str) and image.startswith('sha256:')
              and image == payload.get('image_digest') == env.get('PDBLEND_IMAGE_ID')
              and any(value == image or value.endswith('@' + image) for value in argv),
              'image identity differs')
        source_sha = _digest(source['files'])
        _need(source.get('source_sha256') == payload.get('source_sha256') == env['PDBLEND_SOURCE_SHA256']
              == inputs.get('source_sha256') == source_sha, 'source identity differs')
        source_root = Path(source_ref['path']).parent
        _need(host_path(argv, '/opt/pdblend-src') == str(source_root), 'executed source mount differs')
        entrypoint = module.replace('.', '/') + '.py'
        _need(entrypoint in source['files'], 'collector missing from source inventory')
        for name, sha in source['files'].items():
            target = (source_root / name).resolve()
            _need(target.is_relative_to(source_root), 'source file escapes snapshot')
            reference = dict(path=str(target), sha256=sha)
            previous = refs.get(reference['path'])
            remember(reference)
            if previous is None:
                _need(hasher(target) == sha, 'frozen source bytes differ: ' + name)
        _need(plan.get('system') == system and plan.get('model_id') == model
              and plan.get('evaluation_used_for_selection') is False, 'plan system/model/selection differs')
        _need(type(plan.get('tp')) is int and plan['tp'] > 0 and plan.get('pp') == 1, 'unsupported plan topology')
        if system == 'pdblend':
            _need(inputs.get('system') == system and inputs.get('model_id') == model, 'input identity differs')
            _need(_ref(inputs['point_plan']) == plan_ref and _ref(inputs['source_manifest']) == source_ref,
                  'input plan/source binding differs')
            _need(_ref(payload['input_manifest']) == inputs_ref, 'queued input binding differs')
        else:
            _need(int(_arg(argv, '--tp')) == payload.get('tp') == plan['tp'] and payload.get('pp') == plan['pp'],
                  'DistServe topology differs')
            exact = inputs['exact_inputs_sha256']
            _need(exact['point_plan'] == plan_ref['sha256'] and exact['source_manifest'] == source_ref['sha256'],
                  'DistServe exact input binding differs')
            if payload.get('exact_inputs_sha256') is not None:
                _need(payload['exact_inputs_sha256'] == exact, 'queued DistServe exact inputs differ')
        verification_ref = _ref(entry['model_verification'])
        _need(host_path(argv, env['PDBLEND_MODEL_VERIFICATION_RECEIPT']) == verification_ref['path'],
              'model verification execution binding differs')
        read(verification_ref)
        if system == 'pdblend':
            _need(_ref(inputs['model_verification']) == verification_ref, 'model verification input binding differs')
            for flag, field in (('--power-pilot-plan', 'power_pilot_plan'),
                                ('--request-cycle-plan', 'request_cycle_plan'),
                                ('--layout-energy-plan', 'layout_energy_plan')):
                if flag in argv:
                    supplement = _ref(inputs[field])
                    _need(host_path(argv, _arg(argv, flag)) == supplement['path'], 'supplement plan path differs')
                    supplement_plan = read(supplement)
                    _need(supplement_plan.get('model_id') == model
                          and supplement_plan.get('evaluation_used_for_selection') is False,
                          'supplement plan model/selection differs')
        else:
            _need(inputs['exact_inputs_sha256']['model_verification'] == verification_ref['sha256'],
                  'model verification input checksum differs')
            if module.endswith('.stage_cohort'):
                cohort_ref = _ref(entry['cohort_inputs'])
                _need(host_path(argv, _arg(argv, '--cohort-inputs')) == cohort_ref['path'], 'cohort path differs')
                cohort = read(cohort_ref)
                _need(cohort_ref['sha256'] == payload.get('cohort_sha256') == inputs.get('cohort_sha256')
                      and cohort.get('source_sha256') == source_sha, 'cohort source/input binding differs')
                member = _arg(argv, '--member')
                _need(member == inputs.get('cohort_member') == env.get('PDBLEND_PROFILE_MEMBER'),
                      'cohort member differs')
                member_plan = cohort['members'][member]
                _need(member_plan['model_id'] == model and member_plan['tp'] == plan['tp']
                      and member_plan['pp'] == plan['pp'] and member_plan['gpu_count'] == len(m['gpu_uuids']),
                      'cohort member model/topology differs')
                point = member_plan['point_plan']
                _need(str((Path(cohort_ref['path']).parent / point['path']).resolve()) == plan_ref['path']
                      and point['sha256'] == plan_ref['sha256'], 'cohort point plan differs')
        completion_ref = _ref(entry['completion'])
        out = Path(host_path(argv, _arg(argv, '--out')))
        _need(out.is_relative_to(attempt) and Path(completion_ref['path']) == out / 'completion.json',
              'completion belongs to another attempt/output')
        _need(str(Path(completion_ref['path']).relative_to(attempt)) in payload['required_receipts'],
              'completion was not a required receipt')
        completion = read(completion_ref)
        _need(completion.get('status') == 'failed' and completion.get('complete') is False
              and completion.get('formal_eligible') is False and isinstance(completion.get('error'), str),
              'completion is not a reported qualification collection failure')
        if 'point_plan' in completion:
            _need(_ref(completion['point_plan']) == plan_ref, 'completion plan differs')
        if 'system' in completion:
            _need(completion['system'] == system, 'completion system differs')
        def reported(document):
            values = {k: document[k] for k in FIELDS if k in document}
            if isinstance(document.get('measurement_qualification_gaps'), list):
                values['measurement_qualification_gap_count'] = len(document['measurement_qualification_gaps'])
            return values
        components = []
        for name in COMPONENTS:
            if name not in completion:
                continue
            component_ref = _ref(completion[name])
            _need(Path(component_ref['path']).is_relative_to(out), 'component belongs to another output')
            component = read(component_ref)
            subreports = []
            for field in ('audit', 'training', 'heldout'):
                child = component.get(field)
                if not isinstance(child, dict) or 'path' not in child:
                    continue
                child_ref = _ref(child)
                _need(Path(child_ref['path']).is_relative_to(out), 'component subreport belongs to another output')
                subreports.append(dict(kind=field, reference=child_ref, reported=reported(read(child_ref))))
            components.append(dict(component=name, reference=component_ref,
                                   reported=reported(component), subreports=subreports))
        summary = dict(job_id=m['job_id'], lease_id=m['lease_id'], attempt=m['attempt'],
            system=system, model_id=model, tp=plan['tp'], pp=plan['pp'], source_sha256=source_sha,
            scope=SCOPE, applicability='historical_related_component_only',
            current_engine_meter_domain_equivalence='unproven', raw_qualification_replayed=False,
            finished_s=execution['finished_s'], reported=reported(completion),
            components=components, evidence=dict(manifest=manifest_ref, **{k: _ref(entry[k]) for k in
                ('attempt_manifest', 'execution', 'completion', 'source_manifest', 'input_manifest', 'point_plan')}))
        result.setdefault((system, model), []).append(summary)
    for rows in result.values():
        rows.sort(key=lambda row: (row['finished_s'], row['job_id'], row['attempt']))
    return QualificationEvidence(result, tuple(refs[key] for key in sorted(refs)))


def annotate(rows, evidence):
    """Return copied rows; only receipt-less blocked placeholders get history."""
    _need(isinstance(evidence, QualificationEvidence), 'load qualification evidence once before annotation')
    result = deepcopy(rows)
    for row in result:
        if row.get('status') != 'blocked' or row.get('receipt_path') or row.get('receipt_sha256'):
            continue
        history = evidence.by_system_model.get((row.get('system'), row.get('model_id')), [])
        if history:
            row.update(qualification_attempt_count=len(history), qualification_evidence_scope=SCOPE,
                qualification_evidence_status='historical_related_failures_current_applicability_unproven',
                qualification_evidence_refs=deepcopy(history))
    return result
