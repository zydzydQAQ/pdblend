"""Independent audit of the measured export that follows the preserved Q2 failure.

This proof grants retained-weight capability only. The composition verifier must
separately reconstruct the original candidate/request/idle evidence.
"""
import ast
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p


def source_equivalence(reference):
    p.need(reference == p.ref(HERE / 'source-files.json'), 'sole-export source manifest changed')
    files = p.checked(reference)
    expected = {str(HERE / name) for name in ('source-equivalence.json', 'frequency.py', 'verify_prior.py')}
    p.need(set(files) == expected, 'unexpected sole-export source closure')
    for name, digest in files.items():
        p.need(p.sha(name) == digest, 'frozen sole-export source changed: ' + name)
    equivalence = p.checked(p.ref(HERE / 'source-equivalence.json'))
    p.need(equivalence['schema'] == 'new-A-retention-interface-only-recovery-source-v1'
           and equivalence['old_status_rewritten'] is False
           and equivalence['old_raw_rewritten'] is False, 'source recovery declaration differs')
    for key in ('original_frequency', 'retention_only', 'prior_verifier_original', 'prior_verifier'):
        item = equivalence[key]
        p.need(p.sha(item['path']) == item['sha256'], 'recovery parent source changed: ' + key)
        files[item['path']] = item['sha256']
    p.need(equivalence['retention_only'] == p.ref(HERE / 'frequency.py')
           and equivalence['prior_verifier'] == p.ref(HERE / 'verify_prior.py'), 'unexpected recovery entry')
    original = Path(equivalence['original_frequency']['path']).read_text()
    actual = Path(equivalence['retention_only']['path']).read_text()
    old_tree, new_tree = ast.parse(original), ast.parse(actual)
    functions = lambda tree: {v.name: v for v in tree.body if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old, new = functions(old_tree), functions(new_tree)
    p.need(set(old) == set(new), 'recovery added or removed helper functions')
    for name in set(old) - {'execute', 'main'}:
        p.need(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
               'recovery changed original helper: ' + name)
    old_execute = ast.get_source_segment(original, old['execute'])
    new_execute = ast.get_source_segment(actual, new['execute'])
    before = old_execute.split('    async def reference(i):', 1)[0]
    after = old_execute[old_execute.index('    if a.retain_weights:'):]
    replacement = '    # Prior loaded-clock/work evidence is independently replayed before this sole export.\n'
    p.need(new_execute == before + replacement + after,
           'setup/export/measurement/cleanup differ from original Q3 source')
    files[reference['path']] = reference['sha256']
    return equivalence, files


def resume(proof, instance, audit):
    before, control, after = proof['before'], proof['control'], proof['after']
    audit.ack(before, instance)
    audit.ack(after, instance)
    p.need(control['generation'] == before['generation'] + 1 == after['generation'],
           'retention resume lacks actual generation transition')
    expected = dict(role=instance.get('role', 'mixed'), mode='continuous', admit_prefill=True, admit_decode=True)
    p.need(all(control.get(k) == v and after.get(k) == v for k, v in expected.items())
           and after['accepting'] is True, 'retention resume did not restore native admission')


def prior_reference(state, binding_ref, profile_ref):
    prior = state['prior_frequency_proof']
    p.need(prior['passed'] is True and prior['independently_recomputed'] is True
           and prior['retained_weights_qualified'] is False, 'prior proof must not grant cache capability')
    reference = prior['original_failed_status']
    old = p.checked(reference)
    p.need(old['binding'] == binding_ref and old['profile'] == profile_ref
           and old['finished_s'] < state['started_s'], 'prior evidence identity or cleanup ordering differs')
    p.need(old['passed'] is False and old['complete'] is False and old['node_lease_held'] is False
           and old['measurement_valid'] is True and not old['cleanup_errors'], 'original clean failed status missing')
    error = old.get('error', '')
    p.need(error == prior['original_export_error'] and error.startswith('ContentTypeError(')
           and '/retain_weights' in error and 'status=404' in error,
           'prior failure was not the diagnosed terminal route error')
    p.need(prior['files'].get(reference['path']) == reference['sha256'], 'prior proof does not bind original status')
    return reference


def retained_evidence(state, binding, audit, common):
    retained = state['retained_weights']
    p.need(isinstance(retained, dict) and retained['instance_id'] == binding['instances'][0]['id'],
           'sole native export member differs')
    i = binding['instances'][0]
    audit.ack(retained['native_before'], i)
    value = p.checked(retained['manifest'])
    response, payload = retained['response'], retained['payload']
    p.need(retained['http_status'] == 200 and response['manifest'] == value
           and response['generation'] == payload['expected_generation'] == retained['native_before']['generation']
           and response['accepting'] is False, 'fresh weight export native identity differs')
    p.need(state['measurement_start_s'] <= retained['started_s'] < retained['finished_s'] <= state['measurement_end_s'],
           'native export lies outside measured operation')
    cache = Path(response['retained_weights'])
    transaction = payload['transaction']
    p.need(isinstance(transaction, str) and transaction.startswith('uniform_')
           and Path(transaction).name == transaction, 'invalid fresh export transaction')
    p.need(cache == Path(p.read(i['engine_config'])['weight_cache_root']) / transaction
           and retained['manifest']['path'] == str(cache / 'manifest.json'), 'cache outside this new owner')
    p.need(value['complete'] is True and value['tp'] == i['tp'] and len(value['ranks']) == i['tp']
           and {r['rank'] for r in value['ranks']} == set(range(i['tp'])), 'actual retained TP ranks differ')
    rank_paths = {str(cache / rank['file']) for rank in value['ranks']}
    p.need(set(retained['rank_files']) == rank_paths, 'retained rank proof set differs')
    for rank in value['ranks']:
        path = cache / rank['file']
        proof = retained['rank_files'][str(path)]
        # The frozen real exporter hashes every rank before recording its stat.
        # Rechecks retain the original verifier's immutable-stat rule.
        p.need(path.parent == cache and proof['sha256'] == rank['sha256']
               and common.stat_identity(path) == proof['stat'], 'retained rank immutable stat differs')
    resume(retained['resumed'], dict(i, role='mixed'), audit)
    p.need(retained['resumed']['before']['generation'] == response['generation'], 'export resume generation differs')
    return retained


def verify(out, binding_ref, profile_ref):
    out = Path(out)
    binding, profile = p.checked(binding_ref), p.checked(profile_ref)
    reference = p.ref(out / 'status.json')
    state = p.checked(reference)
    p.need(state['schema'] == 'fresh-legacy-retention-only-recovery-v1'
           and state['passed'] is True and state['complete'] is True and state['finished_s']
           and not state.get('error') and state['node_lease_held'] is False and not state['cleanup_errors']
           and state['measurement_valid'] is True and state['clock_restore_complete'] is True
           and not state['sampling_error'], 'terminal clean measured retention required')
    if socket.gethostname() == binding['hostname']:
        p.need(not p.active_owner(state), 'sole-export process has not exited')
    p.need(state['binding'] == binding_ref and state['profile'] == profile_ref
           and state['cost_values_recalibrated'] is False, 'retention candidate input changed')
    p.need(state['started_s'] <= state['measurement_start_s'] < state['measurement_end_s'] <= state['finished_s'],
           'retention terminal measurement ordering differs')
    p.need(state['references'] == [] and state['cases'] == [] and state['idle_wakeup'] == []
           and state['active_cases'] == {}, 'sole export must not claim repeated qualification work')
    equivalence, sources = source_equivalence(state['source_manifest'])
    prior = prior_reference(state, binding_ref, profile_ref)
    q = p.load(equivalence['retention_only'], 'independent_sole_retention_driver')
    common = q.paths(binding)
    from ecopadg.serving.measurement import power_evidence
    audit_path = q.ROOT.parent / 'AC-baseline-binding-v2/gate_evidence.py'
    audit = p.load(audit_path, 'independent_sole_retention_raw')
    before_files = audit.files(out)
    audit.identities(out, binding['instances'])
    physical = audit.power(out, state, power_evidence)
    for i in binding['instances']:
        p.need(audit.lines(out / (i['id'] + '.requests.jsonl')) == []
               and audit.lines(out / (i['id'] + '.stream.jsonl')) == [], 'sole export contains request work')
    p.need(not list((out / 'case-records').glob('*')), 'sole export contains case evidence')
    p.need(set(state['restoration']) == {i['id'] for i in binding['instances']}, 'restoration member set differs')
    for i in binding['instances']:
        restoration = state['restoration'][i['id']]
        p.need(restoration['complete'] is True and not restoration['errors'], 'native restoration incomplete')
        audit.ack(restoration['before'], i)
        common.barrier(restoration['before'], restoration['proof'], i)
        resume(restoration['resumed'], i, audit)
        p.need(restoration['resumed']['before']['generation'] >= restoration['proof']['generation'],
               'restoration resume predates native barrier')
    p.need(p.sha(state['topology']['path']) == state['topology']['sha256']
           and all('GPU' + str(i) in Path(state['topology']['path']).read_text() for i in range(8)),
           'fresh physical topology differs')
    retained = retained_evidence(state, binding, audit, common)
    for name, digest in state['isolated_artifacts'].items():
        p.need(p.sha(name) == digest, 'isolated measurement evidence changed')
    hooks_path = q.U / 'meter-runtime/sampler_hooks.py'
    hooks = p.load(hooks_path, 'independent_sole_retention_sampler')
    frozen = hooks.completed_artifacts(hooks.directories(out / 'isolated-samplers'),
        p.ref(Path(binding['host_release']) / 'manifest.json'), p.ref(q.U / 'isolated-power/manifest.json'))
    p.need(frozen == state['isolated_artifacts'], 'sampler terminal closure differs')
    p.need(audit.files(out) == before_files, 'retention evidence changed during reconstruction')
    retained_evidence(state, binding, audit, common)
    files = {**before_files, **sources}
    for item in (reference, binding_ref, profile_ref, prior, retained['manifest'], state['topology'],
                 p.ref(__file__), p.ref(audit_path), p.ref(hooks_path)):
        files[item['path']] = item['sha256']
    return dict(passed=True, independently_recomputed=True, retained_weights_qualified=True,
        repeated_candidate_cases=0, repeated_request_count=0, repeated_idle_cycles=0,
        historical_profile_costs_recalibrated=False, prior_frequency_requires_separate_reconstruction=True,
        original_failed_status=prior, retained=retained, topology=state['topology'],
        rawref=reference, physical=physical, files=files)
