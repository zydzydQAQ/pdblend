"""Read-only C native-refusal reclassification; never replace live state.

Every cache hit verifies all declared input hashes and filesystem identities.
An integrity failure raises before returning any modified report.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
DIRECTORY = ROOT / 'C/uniform-rate-20260909-v2'
ADAPTER = DIRECTORY / 'native-rejection-adapter-v1'
AUDITOR = dict(path=str(ADAPTER / 'reconstruct.py'),
    sha256='534eafae6947dafaffe0a138ed17ea933a11c0cd83275c713d83006371bdd01f')
CONTRACT = dict(path=str(ADAPTER / 'contract.py'),
    sha256='d6ee9aac1566264548fcea66f13866eaed7fba125cc4464f8a55a68d4edc4c24')
SCIENTIFIC = ('n_expected', 'completed_work_requests', 'good_requests', 'generated_tokens',
    'expected_generated_tokens', 'slo_attainment', 'producer_slo_attainment', 'request_timeouts',
    'ttft_avg_s', 'tpot_avg_s', 'energy_j', 'energy_per_gpu_j', 'gpu_util', 'gpu_util_per_gpu',
    'request_throughput_rps', 'token_throughput_tps', 'token_throughput_is_exact',
    'actual_output_tokens', 'observed_output_tokens_lower_bound', 'measurement_duration_s',
    'measurement_valid', 'work_complete', 'completion_fraction', 'failed_requests')
_CACHE = {}
_SEEN_INDICES = {}
_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def stat_key(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def file_hash(path):
    path = Path(path)
    with path.open('rb') as stream:
        before = stat_key(os.fstat(stream.fileno()))
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
        after = stat_key(os.fstat(stream.fileno()))
        need(before == after == stat_key(path.stat()), 'overlay input changed during read: ' + str(path))
    return digest.hexdigest(), before


def ref(path):
    path = str(Path(path).resolve())
    return dict(path=path, sha256=file_hash(path)[0])


def checked(reference):
    need(isinstance(reference, dict) and set(reference) >= {'path', 'sha256'}, 'immutable overlay reference required')
    path = Path(reference['path'])
    with path.open('rb') as stream:
        before = stat_key(os.fstat(stream.fileno()))
        data = stream.read()
        need(before == stat_key(os.fstat(stream.fileno())) == stat_key(path.stat()),
            'overlay JSON changed during snapshot read')
    need(hashlib.sha256(data).hexdigest() == reference['sha256'], 'changed overlay reference: ' + reference['path'])
    return json.loads(data)


def load(reference, name):
    need(file_hash(reference['path'])[0] == reference['sha256'], 'overlay auditor or contract changed')
    spec = importlib.util.spec_from_file_location(name, reference['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add(files, path, digest):
    need(isinstance(path, str) and Path(path).is_absolute()
        and isinstance(digest, str) and len(digest) == 64, 'invalid declared overlay dependency')
    need(path not in files or files[path] == digest, 'conflicting overlay dependency: ' + path)
    files[path] = digest


def dependencies(index_ref, index):
    files = {}
    add(files, str(Path(__file__).resolve()), _SOURCE_SHA)
    for path, digest in index['files'].items():
        add(files, path, digest)
    for reference in (index_ref, AUDITOR, CONTRACT, index['observation'], index['diagnosis'],
                      index['proof'], index['checkpoint'], index['original_observation']):
        add(files, reference['path'], reference['sha256'])
    cp = checked(index['checkpoint'])
    for name in ('release', 'binding', 'receipt'):
        reference = cp[name]
        if isinstance(reference, str):
            reference = dict(path=reference, sha256=cp[name + '_sha256'])
        add(files, reference['path'], reference['sha256'])
        document = checked(reference)
        for field in ('files', 'source_files', 'artifacts'):
            for path, digest in document.get(field, {}).items():
                add(files, path, digest)
    for path, digest in cp['artifacts'].items():
        add(files, path, digest)
    add(files, cp['row']['trace'], cp['row']['trace_sha256'])
    return files


def fingerprint(files, shared=None):
    shared = {} if shared is None else shared
    result = []
    for path, expected in sorted(files.items()):
        before = stat_key(Path(path).stat())
        key = (path, expected, before)
        actual = shared.get(key)
        if actual is None:
            digest, actual = file_hash(path)
            need(digest == expected, 'declared overlay input changed: ' + path)
            shared[key] = actual
        need(actual == stat_key(Path(path).stat()), 'overlay input replaced after verification: ' + path)
        result.append((path, expected, actual))
    need(all(stat_key(Path(path).stat()) == value for path, _, value in result),
        'overlay input changed while checking its closure')
    return tuple(result)


def comparable_classification(value):
    value = copy.deepcopy(value)
    # This is audit wall time, not any request or physical measurement time.
    value.get('native_queue_evidence', {}).get('raw', {}).pop('captured_s', None)
    return value


def verify_index(index_ref, index, shared):
    need(index['schema'] == 'C-native-refusal-classification-source-v1'
        and index['auditor'] == AUDITOR and index['declaration_contract'] == CONTRACT
        and index['no_gpu_actions'] is True, 'unreviewed native reclassification authority')
    files = dependencies(index_ref, index)
    before = fingerprint(files, shared)
    key = (index_ref['path'], index_ref['sha256'])
    cached = _CACHE.get(key)
    if cached and cached['fingerprint'] == before:
        return copy.deepcopy(cached['observation'])
    saved, original = checked(index['observation']), checked(index['original_observation'])
    diagnosis, proof = checked(index['diagnosis']), checked(index['proof'])
    need(saved['checkpoint'] == original['checkpoint'] == diagnosis['checkpoint'] == proof['checkpoint'] == index['checkpoint'],
        'native reclassification attempted to replace another checkpoint')
    need((saved.get('measurement_host'), saved['model'], saved['system']) == ('C', '7b', 'ecoserve')
        and saved['cell_id'] == original['cell_id'] and saved['independent_reclassification_only'] is True,
        'native overlay physical scope or identity differs')
    need(saved['original_observation'] == index['original_observation']
        and saved['diagnosis_reference'] == index['diagnosis']
        and diagnosis['native_proof'] == index['proof'], 'native derived evidence references differ')
    need(diagnosis['passed'] and diagnosis['independently_recomputed'] and diagnosis['no_unknown_errors']
        and diagnosis['no_PDB_complete_boundary_claim'], 'native diagnosis did not pass')
    actual = load(AUDITOR, '_report_native_independent_auditor').audit_checkpoint(index['checkpoint']['path'])
    for key in SCIENTIFIC:
        need(key in saved and key in actual, 'native scientific metric absent: ' + key)
        need(saved[key] == actual[key], 'derived native scientific metric differs: ' + key)
        if key in original:
            need(original[key] == actual[key], 'native reclassification changed an original metric: ' + key)
    need(comparable_classification(actual['baseline_service_failure'])
        == comparable_classification(saved['baseline_service_failure'])
        == comparable_classification(diagnosis['classification']), 'native classification differs from full recomputation')
    need(actual['zero_output_diagnosis']['native_rejections'] == proof['native_rejections']
        and actual['zero_output_diagnosis']['request_timeouts'] == proof['request_timeouts'], 'native proof counts differ')
    actual.update(failure_class=saved['failure_class'], diagnosis_reference=index['diagnosis'],
        original_observation=index['original_observation'], independent_reclassification_only=True)
    # The diagnosis reference is immutable and compares its full classification.
    # Retain that exact snapshot there, while zero_output_diagnosis contains the
    # newly recomputed proof including its actual audit wall-clock timestamp.
    actual['baseline_service_failure'] = copy.deepcopy(saved['baseline_service_failure'])
    need(load(CONTRACT, '_report_native_reviewed_contract').acceptable_baseline(actual), 'native contract rejected derived observation')
    # Rehash after the full audit; the cache never spans a changed dependency.
    after = fingerprint(files)
    need(before == after, 'native source/evidence changed during independent audit')
    _CACHE[(index_ref['path'], index_ref['sha256'])] = dict(fingerprint=after, observation=copy.deepcopy(actual))
    return actual


def apply(result):
    """Return a report copy with verified same-checkpoint classifications only."""
    need(result.get('schema') == 'uniform-rate-results-v2', 'unknown report schema')
    need(file_hash(__file__)[0] == _SOURCE_SHA, 'loaded report overlay source changed')
    need(all(Path(path).is_file() for path in _SEEN_INDICES), 'previously seen native overlay index disappeared')
    positions = {}
    for n, observation in enumerate(result['observations']):
        reference = observation['checkpoint']
        key = reference['path'], reference['sha256']
        need(key not in positions, 'duplicate primary checkpoint in report')
        positions[key] = n
    replacements, shared = {}, {}
    for path in sorted(DIRECTORY.glob('native-rejection-reconstruction-*/source-closure.json')):
        index_ref = ref(path)
        seen = _SEEN_INDICES.get(str(path))
        need(seen is None or seen == index_ref['sha256'], 'previously seen native overlay index changed')
        index = checked(index_ref)
        key = index['checkpoint']['path'], index['checkpoint']['sha256']
        if key not in positions:
            need(not any(cp_path == key[0] for cp_path, _ in positions), 'report checkpoint SHA changed beneath native overlay')
            continue
        _SEEN_INDICES[str(path)] = index_ref['sha256']
        need(key not in replacements, 'multiple derived classifications target one checkpoint')
        actual = verify_index(index_ref, index, shared)
        replacements[key] = actual, index_ref, index
    output = copy.deepcopy(result)
    affected = set()
    for key, (actual, index_ref, index) in replacements.items():
        target = output['observations'][positions[key]]
        need((target.get('measurement_host'), target['model'], target['system']) == ('C', '7b', 'ecoserve')
            and target['cell_id'] == actual['cell_id'], 'same checkpoint reported with a different physical identity')
        for metric in SCIENTIFIC:
            need(target.get(metric) == actual[metric], 'native overlay would change a displayed metric: ' + metric)
        # Preserve every existing numeric, identity and live-state field.
        for field in ('failure_class', 'diagnosis_reference', 'baseline_service_failure',
                      'zero_output_diagnosis', 'original_observation', 'independent_reclassification_only'):
            target[field] = copy.deepcopy(actual[field])
        target['native_report_overlay'] = dict(source=index_ref, observation=index['observation'],
            auditor=AUDITOR, report_contract=CONTRACT)
        affected.add((target['model'], target['dataset'], target['measurement_host']))
    if not affected:
        return output
    contract = load(CONTRACT, '_report_native_group_decisions')
    for group in output['groups']:
        identity = group['model'], group['dataset'], group['node']
        if identity not in affected:
            continue
        logical = contract.resolve_group(group['declaration'], identity[0], identity[1], actual_host=identity[2])
        rows = [r for r in output['observations'] if
            (r['model'], r['dataset'], r.get('measurement_host')) == identity]
        by_id = {r['cell_id']: r for r in rows}
        reused = {r['cell_id']: r for r in logical['reused_observations']}
        for cell_id, saved in reused.items():
            if cell_id in by_id:
                need(by_id[cell_id]['checkpoint'] == saved['checkpoint'], 'reused C checkpoint changed')
        logical['reused_observations'] = [by_id[key] for key in reused if key in by_id]
        decision = contract.select_group(logical, [r for r in rows if r['cell_id'] not in reused])
        cap = decision.get('cap_rate_rps')
        need(cap == group['cap_rate_rps'], 'native classification changed the PDB boundary')
        gaps = []
        for position in logical['positions']:
            if cap is not None and position['rate_rps'] <= cap:
                for system in contract.SYSTEMS:
                    matching = [r for r in rows if r['system'] == system and r['rate_rps'] == position['rate_rps']]
                    if not any(r.get('token_throughput_is_exact') is True for r in matching):
                        gaps.append(dict(rate_rps=position['rate_rps'], system=system, metric='exact_output_throughput'))
        group.update(decision=decision, report_contract=CONTRACT,
            missing_metric_coordinates=gaps,
            pdb_boundary_complete=decision.get('pdb_boundary_complete') is True and not group['missing_raw_metric_audits'],
            complete=decision['phase'] == 'complete' and not group['missing_raw_metric_audits'] and not gaps)
    output['complete'] = len(output['groups']) == 9 and all(g['complete'] for g in output['groups'])
    output['native_report_overlay'] = dict(schema='C-native-refusal-report-overlay-v1',
        affected_groups=[dict(model=m, dataset=d, node=n) for m, d, n in sorted(affected)],
        observation_count=len(replacements), auditor=AUDITOR, report_contract=CONTRACT,
        physical_pipeline_states_preserved=True, existing_scientific_values_preserved=True)
    return output
