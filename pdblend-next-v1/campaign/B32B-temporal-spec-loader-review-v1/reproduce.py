"""Real frozen loader/child, real failed spec bytes; CPU only, no HTTP/GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

ROOT = Path('/root/workspace/pdblend-next-v1')
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT / 'campaign/B32B-temporal-observation-attempt-001'
WRAPPER = ROOT / 'campaign/B32B-temporal-observation-execution-v2-r4'
HOOK = ATTEMPT / 'image-context/pdblend_diagnostics.py'
SPEC = ATTEMPT / 'observation-spec.json'
PARENT_SPEC = ROOT / 'campaign/B32B-temporal-observation-candidate-v1/specs/original-vs-continuous.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compact(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_case(tmp, name, payload, *, claimed_sha=None):
    path = tmp / (name + '.json')
    path.write_bytes(payload)
    mod = module(HOOK, 'peer_hook_' + name)
    warnings = []
    env = {'PDBLEND_DIAGNOSTIC_SPEC': str(path),
           'PDBLEND_DIAGNOSTIC_SPEC_SHA256': claimed_sha or sha(path)}
    with patch.dict(os.environ, env), patch.object(mod.os, 'write',
            lambda fd, data: warnings.append(data.decode()) or len(data)):
        state = mod.state()  # This is the actual enabled loader, not validate_spec alone.
        assert mod._LOADED
        if state is not None:
            assert state.writer is None and state.error is None and not state.seen
            assert state.spec == json.loads(payload) and state.spec_sha == sha(path)
            assert state.ids == frozenset(state.spec['request_ids'])
    return dict(case=name, bytes=len(payload), sha256=sha(path), loaded=state is not None,
                warnings=warnings, writer_created=False), mod, path


def main():
    paths = [HOOK, SPEC, PARENT_SPEC, ATTEMPT/'spec.json',
             ATTEMPT/'image-context/model_runner.py', WRAPPER/'prepare.py', WRAPPER/'common.py',
             WRAPPER/'child.py', WRAPPER/'test_job_integration.py',
             ROOT/'campaign/B32B-temporal-observation-candidate-v1/test_observation.py',
             ATTEMPT/'results/job.json', ATTEMPT/'results/diagnostic-binding.json',
             ATTEMPT/'results/child/full-outputs.json',
             ATTEMPT/'results/child/status.json', ATTEMPT/'results/status.json',
             ATTEMPT/'results/diagnostic-docker.log']
    inputs = {str(p): sha(p) for p in paths}
    prepared = json.loads((ATTEMPT/'spec.json').read_text())
    assert prepared['installed_diagnostic_sources'][
        '/usr/local/lib/python3.10/dist-packages/vllm/pdblend_diagnostics.py'] == sha(HOOK)
    assert prepared['installed_diagnostic_sources'][
        '/usr/local/lib/python3.10/dist-packages/vllm/worker/model_runner.py'] == sha(
            ATTEMPT/'image-context/model_runner.py')
    assert prepared['observation_spec_sha256'] == sha(SPEC)
    raw = SPEC.read_bytes()
    parsed = json.loads(raw)
    encoded = compact(parsed)
    assert len(raw) == 16432 and len(encoded) == 6009
    assert json.loads(encoded) == parsed
    child = module(WRAPPER/'child.py', 'peer_frozen_temporal_child')
    job = json.loads((ATTEMPT/'results/job.json').read_text())
    issued = json.loads((ATTEMPT/'results/child/status.json').read_text())['started_s']
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        original, failed, _ = load_case(tmp, 'original_actual_16432', raw)
        assert not original['loaded'] and 'spec byte cap' in original['warnings'][0]
        cases.append(original)
        current, _, compact_path = load_case(tmp, 'compact_same_object', encoded)
        assert current['loaded'] and not current['warnings']
        cases.append(current)
        for size, expected in [(16384, True), (16385, False)]:
            value, _, _ = load_case(tmp, 'boundary_' + str(size),
                                   encoded + b' ' * (size-len(encoded)))
            assert value['loaded'] is expected
            cases.append(value)
        bad_sha, _, _ = load_case(tmp, 'wrong_sha', encoded, claimed_sha='0'*64)
        assert not bad_sha['loaded'] and 'spec SHA mismatch' in bad_sha['warnings'][0]
        cases.append(bad_sha)
        bad = copy.deepcopy(parsed)
        bad['request_ids'] = [bad['request_ids'][0]]*2
        bad_scope, _, _ = load_case(tmp, 'invalid_duplicate_uuid', compact(bad))
        assert not bad_scope['loaded'] and 'two explicit UUID' in bad_scope['warnings'][0]
        cases.append(bad_scope)
        # Existing failed processes cannot be repaired by overwriting the spec.
        with patch.dict(os.environ, {'PDBLEND_DIAGNOSTIC_SPEC': str(compact_path),
                'PDBLEND_DIAGNOSTIC_SPEC_SHA256': sha(compact_path)}):
            assert failed.state() is None
        # Frozen child validates the object only. It accepted the actual old
        # bytes and also accepts identical compact bytes under their true SHA.
        fresh_job = copy.deepcopy(job)
        fresh_job.update(observation_spec=str(compact_path), observation_spec_sha256=sha(compact_path))
        with patch.object(child.time, 'time', lambda: issued):
            original_child = child.validate_job(job)
            compact_child = child.validate_job(fresh_job)
        assert original_child == compact_child == parsed
    status = json.loads((ATTEMPT/'results/status.json').read_text())
    log = (ATTEMPT/'results/diagnostic-docker.log').read_text()
    outputs = json.loads((ATTEMPT/'results/child/full-outputs.json').read_text())['token_ids_by_request_uuid']
    ids = {r['label']: r['request_uuid'] for r in parsed['requests']}
    assert set(outputs) == set(ids.values()) and all(len(v) == 64 for v in outputs.values())
    def difference(a, b):
        return next((dict(position_one_based=n, reference=x, observed=y)
                     for n, (x,y) in enumerate(zip(outputs[ids[a]], outputs[ids[b]]),1)
                     if x != y), None)
    diffs = {mode: [difference('golden-'+slot, mode+'-'+slot) for slot in ('first','second')]
             for mode in ('temporal','continuous')}
    assert diffs == {'temporal': [None, {'position_one_based':32,'reference':2776,'observed':4172}],
                     'continuous': [None, {'position_one_based':10,'reference':279,'observed':330}]}
    assert all(sha(p) == value for p, value in inputs.items())
    result = dict(schema=1, cpu_only=True, gpu_executed=False,
        original_bytes=len(raw), parent_declaration_bytes=len(PARENT_SPEC.read_bytes()),
        loader_byte_cap=16384, original_excess_bytes=len(raw)-16384,
        compact_bytes=len(encoded), parsed_objects_exactly_equal=True,
        same_six_requests=True, same_two_capture_uuids=True, same_full64_work=True,
        actual_hook_sha256=sha(HOOK), cases=cases, failed_loader_does_not_reload=True,
        frozen_child_accepts_original_and_compact_objects=True,
        original_child_validation_wall_clock_replayed_s=issued,
        actual_http_uuid_by_label=ids, actual_http_six_full64=True,
        actual_http_pair_differences_recomputed=diffs,
        actual_warning_count=log.count("PDB diagnostic disabled: ValueError('spec byte cap')"),
        actual_capture_complete=status['capture_complete'], actual_capture_error=status['capture_error'],
        actual_measurement_valid=status['measurement_valid'],
        retained_reported_full_operation_energy_j=status['full_operation_energy_j'],
        hook_record_cap_unchanged=14, hook_record_byte_cap_unchanged=16384,
        input_sha256=inputs, all_inputs_unchanged_after=True,
        limitations=['No model_runner import or GPU forward executed.',
                     'No capture, metadata parity, logits, or numerical root cause inferred.',
                     'Saved compact bytes contain the old output path and are CPU evidence only; never a new attempt.'])
    (OUT/'spec-bytes-for-loader-only.json').write_bytes(encoded)
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('original_bytes','parent_declaration_bytes',
        'compact_bytes','parsed_objects_exactly_equal','actual_warning_count',
        'frozen_child_accepts_original_and_compact_objects','all_inputs_unchanged_after')}))


if __name__ == '__main__':
    main()
