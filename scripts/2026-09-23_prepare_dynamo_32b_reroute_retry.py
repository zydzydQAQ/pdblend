#!/usr/bin/env python3
"""Freeze a 32B reroute-fix retry; reuse validated own cells, never enqueue."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path('/home/pdblend4')
sys.path.insert(0, str(ROOT/'src'))
from pdblend_baselines.dynamollm.portable_profile import validate_profile
from pdblend_baselines.dynamollm.prepare_v1 import merge_own_profiles
from pdblend_baselines.dynamollm.profiles import PaperProfiles

OLD_JOBS = ROOT/'results/2026-09-23/dynamo-functional-job-specs-final-v2/jobs.json'
ATTEMPT = ROOT/'results/2026-09-22/three-model/queue-attempts/dynamo-functional-32b-seed701-ff4553c99b379bfb/attempt-0001-08c68a8fd0104b4faac06ba535af5d5b'
OVERLAY = 'pdblend_baselines/dynamollm/runtime.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def freezer():
    spec = importlib.util.spec_from_file_location('reroute_source_freezer',
        ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(out, *, after_terminal=()):
    out = out.resolve()
    if out.exists():
        raise FileExistsError('refusing to replace retry inputs')
    old_job = next(j for j in json.loads(OLD_JOBS.read_text())
                   if j['payload']['model_id'] == 'Qwen2.5-32B-Instruct')
    old = old_job['payload']
    base = Path(old['source_snapshot'])
    helper = freezer()
    old_manifest = json.loads((base/'manifest.json').read_text())
    helper.verify_snapshot(base, old_manifest['files'])
    with tempfile.TemporaryDirectory(prefix='dynamo-reroute-source-') as directory:
        staging = Path(directory)/'src'
        shutil.copytree(base, staging)
        shutil.copyfile(ROOT/'src'/OVERLAY, staging/OVERLAY)
        source, source_sha = helper.freeze_source(staging, ROOT/'results/2026-09-23/dynamo-functional-sources')
    frozen = json.loads((source/'manifest.json').read_text())
    changed = sorted(name for name in frozen['files']
                     if frozen['files'][name] != old_manifest['files'].get(name))
    if changed != [OVERLAY] or set(frozen['files']) != set(old_manifest['files']):
        raise ValueError('retry source must change only the reviewed runtime')
    original_profiles = [ATTEMPT/f'dynamo/functional-profile-stage/instance-{i}/profile.json' for i in range(2)]
    validated = [validate_profile(artifact_root=ATTEMPT, recorded_root=Path('/output'), profile=p)
                 for p in original_profiles]
    if any(v['portable_profile_receipt']['cell_count'] != 9 for v in validated):
        raise ValueError('expected two complete nine-cell profiles')
    prior = json.loads((ATTEMPT/'dynamo/completion.json').read_text())
    if (prior['requests'] != 17 or prior['seed'] != 701 or prior['own_cleanup_complete'] is not True
            or prior['failures'] != ['request_output_or_real_route_missing']):
        raise ValueError('previous attempt provenance or cleanup differs')
    out.mkdir(parents=True)
    portable = []
    for i, value in enumerate(validated):
        path = out/f'instance-{i}-portable.json'
        save(path, value)
        portable.append(path)
    merged = merge_own_profiles(portable, model_id=old['model_id'])
    if len(merged['points']) != 18:
        raise ValueError('expected exact eighteen-cell reuse')
    profile = out/'profiles.json'
    save(profile, merged)
    trace = next(Path(h) for h, target, _ in old['mounts'] if target == '/trace.json')
    prediction = next(Path(h) for h, target, _ in old['mounts'] if target == '/prediction-receipt.json')
    predicted = json.loads(prediction.read_text())
    requests = json.loads(trace.read_text())['requests']
    if len(requests) != 17 or sha(trace) != old['exact_inputs_sha256']['trace']:
        raise ValueError('original seventeen-request workload changed')
    model = PaperProfiles.load(profile)
    for index, (request, row) in enumerate(zip(requests, predicted['predictions'], strict=True)):
        if (row['request_index'] != index or row['input_tokens'] != len(request['prompt'])
                or row['trace_max_tokens'] != request['max_tokens']):
            raise ValueError('original predictor binding differs')
        for frequency in model.frequencies(2):
            model.query(2, frequency, len(request['prompt']),
                        len(request['prompt'])+row['predicted_output'], 1)
    cfg = json.loads(Path(old['config_path']).read_text())
    cfg.pop('functional_profile_stage')
    cfg['profiles'] = '/reuse/profiles.json'
    cfg['profile_reuse'] = dict(source_attempt=str(ATTEMPT), original_source_sha256=old['source_sha256'],
        receipt='/reuse/profile-reuse.json', recollect=False, cells=18, formal_eligible=False)
    config = out/'config.json'
    save(config, cfg)
    inputs = {str(p): sha(p) for p in [*original_profiles, trace, prediction,
        ATTEMPT/'dynamo/completion.json', ATTEMPT/'dynamo/functional-profile-stage/merged-profile.json',
        Path(old['config_path'])]}
    for value in validated:
        for mapping in value['portable_profile_receipt']['mappings']:
            inputs[mapping['actual']] = mapping['sha256']
    receipt = dict(schema='dynamollm-profile-reuse-v1', model_id=old['model_id'], cells=18,
        old_source_sha256=old['source_sha256'], new_source_sha256=source_sha,
        changed_source_files=changed, raw_unchanged=True, raw_and_holdout_recomputed=True,
        original_attempt_status=prior['status'], original_failed_request='dynamo-701-9',
        input_sha256=inputs, profile_sha256=sha(profile),
        original_gpu_uuids=prior['gpu_uuids'],
        hardware_scope='same 8xL20 host, native actual lease UUIDs recorded separately; functional only',
        coverage=dict(tp=2, pp=1, batch=[1], input_tokens=[128,512,2048], output_tokens=[74],
            frequency_mhz=[900,1200,1500,1800,2100,2520], extrapolation=False),
        profile_source_preserved=True, formal_eligible=False, energy_comparable=False)
    save(out/'profile-reuse.json', receipt)
    identity = dict(source_sha256=source_sha, config_sha256=sha(config),
                    profile_sha256=sha(profile), reuse_sha256=sha(out/'profile-reuse.json'), trace_sha256=sha(trace))
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    job_id = 'dynamo-reroute-retry-32b-seed701-'+suffix
    # Original /output cell paths are relocated only in derived profile copies.
    # The complete old attempt remains mounted at its original host path, read-only.
    keep_targets = {'/models', '/verification/model-verification.json', '/predictor',
                    '/trace.json', '/prediction-receipt.json'}
    mounts = [tuple(row) for row in old['mounts'] if row[1] in keep_targets]
    mounts += [(str(source), '/opt/pdblend-src', 'ro'),
               (str(source/'manifest.json'), '/source-manifest.json', 'ro'),
               (str(out), '/reuse', 'ro'), (str(out), str(out), 'ro'),
               (str(ATTEMPT), str(ATTEMPT), 'ro'), (str(config), '/spec/config.json', 'ro')]
    argv = ['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
            '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
            '--entrypoint','/opt/venv/bin/python']
    for host, target, mode in mounts:
        argv += ['-v', f'{host}:{target}:{mode}']
    argv += ['-v','{attempt_dir}:/output:rw']
    old_argv = old['argv']
    for index, arg in enumerate(old_argv[:-1]):
        if arg == '-e':
            env = old_argv[index+1]
            if env.startswith('PDBLEND_SOURCE_SHA256='):
                env = 'PDBLEND_SOURCE_SHA256='+source_sha
            argv += ['-e', env]
    argv += [old['image_digest'], '-B','-m','pdblend_baselines.dynamollm.run_v1',
             '--config','/spec/config.json','--trace','/trace.json','--out','/output/dynamo',
             '--duration','100','--seed','701','--mode','functional']
    payload = copy.deepcopy(old)
    payload.update(prepare_only=True, deferred=False, container_name=job_id, argv=argv, mounts=mounts,
        source_revision=source_sha, source_sha256=source_sha, source_snapshot=str(source),
        source_base_revision=old['source_sha256'], config_path=str(config),
        depends_on=[], after_terminal=list(after_terminal),
        exact_inputs_sha256=identity, prior_attempt=str(ATTEMPT),
        execution=dict(duration_s=100, seed=701, requests=17, profile_cells_reused=18,
            profile_cells_new=0, output_dir='/output/dynamo', slo_ttft_s=5., slo_tpot_s=.15,
            lease_gpu_placeholder='{lease_local_indices}', lease_port_placeholder='{lease_port}'))
    job = dict(job_id=job_id, payload=payload, priority=100, max_attempts=1)
    save(out/'jobs.json', [job])
    save(out/'manifest.json', dict(schema='dynamo-reroute-retry-v1', prepare_only=True,
        source_snapshot=str(source), source_sha256=source_sha, job_id=job_id,
        jobs_sha256=sha(out/'jobs.json'), old_job_id=old_job['job_id'],
        files={p.name: sha(p) for p in out.iterdir() if p.is_file()},
        tests_pending=True, hardware_started=False))
    helper.verify_snapshot(source, frozen['files'])
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--after-terminal', nargs='*', default=[])
    args = parser.parse_args()
    job = prepare(args.out, after_terminal=args.after_terminal)
    print(json.dumps(dict(job_id=job['job_id'], jobs=str(args.out.resolve()/'jobs.json'),
                          source_sha256=job['payload']['source_sha256'])))
