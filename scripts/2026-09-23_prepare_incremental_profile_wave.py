#!/usr/bin/env python3
"""Prepare an immutable 4+2+1+1 incremental calibration wave; never enqueue."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc'
VERIFY = ROOT/'results/2026-09-22/three-model/profile-receipts/model-verification-99fabb0721f21aa50eb2a8518877acdf05cc76df32f0f769900be7b4d4471fc8.json'
LOCAL_REVIEW = ROOT/'results/2026-09-23/14b-tp4-power-diagnosis/runner-review-v2.json'
EXPECTED = {
    '14b-tp4-power-mixed': ('Qwen2.5-14B-Instruct', 4, 'local_power'),
    '32b-tp2-longctx': ('Qwen2.5-32B-Instruct', 2, 'followup'),
    '7b-tp1-longctx': ('Qwen2.5-7B-Instruct', 1, 'followup'),
    '14b-tp1-longctx': ('Qwen2.5-14B-Instruct', 1, 'training'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True)+'\n')


def freeze_source(base, files):
    spec = importlib.util.spec_from_file_location('incremental_freezer',
        ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.verify_snapshot(base, json.loads((base/'manifest.json').read_text())['files'])
    with tempfile.TemporaryDirectory(prefix='incremental-profile-wave-') as temporary:
        staged = Path(temporary)/'src'; shutil.copytree(base, staged)
        for name in files:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('invalid source overlay path')
            target = staged/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/'src'/relative, target)
        return helper.freeze_source(staged, ROOT/'results/2026-09-23/incremental-profile-sources')


def check_member(name, member):
    expected = EXPECTED[name]
    if tuple(member.get(k) for k in ('model_id', 'tp', 'kind')) != expected:
        raise ValueError('unexpected incremental member identity: '+name)
    roots = [Path(value).resolve(strict=True) for value in member['readonly_roots']]
    def require_mounted(path):
        path = Path(path).resolve(strict=True)
        if not any(path == root or (root.is_dir() and path.is_relative_to(root)) for root in roots):
            raise ValueError('immutable input missing read-only mount: '+str(path))
    if member['kind'] == 'training':
        path = Path(member['plan']); plan = json.loads(path.read_text())
        require_mounted(path); require_mounted(plan['training_source'])
        if ((plan['model_id'], plan['tp'], plan['pp']) != (expected[0], expected[1], 1)
                or plan.get('fit_existing_holdout') is not False
                or digest(plan['training_source']) != plan['training_source_sha256']):
            raise ValueError('training plan identity or immutable source differs')
        return digest(path)
    package = Path(member['package']); manifest = json.loads((package/'manifest.json').read_text())
    require_mounted(package)
    if (manifest['model_id'], manifest['tp'], manifest['pp']) != (expected[0], expected[1], 1):
        raise ValueError('incremental package belongs to another topology')
    for binding in manifest['inputs'].values():
        require_mounted(binding['path'])
        if digest(binding['path']) != binding['sha256']:
            raise ValueError('incremental input changed: '+binding['path'])
    return digest(package/'manifest.json')


def build(*, review, out, dependencies, overlay_files, cache_manifest=None):
    local = json.loads(LOCAL_REVIEW.read_text())
    followup = json.loads(review.read_text())
    members = dict(followup['members'])
    members['14b-tp4-power-mixed'] = dict(model_id='Qwen2.5-14B-Instruct', tp=4,
        kind='local_power', package=local['package'],
        readonly_roots=local['exact_path_readonly_roots'], expected_power_points=12,
        expected_mixed_points=4)
    if set(members) != set(EXPECTED):
        raise ValueError('expected exactly the declared four independent members')
    bindings = {name:check_member(name, member) for name, member in members.items()}
    source, source_sha = freeze_source(Path(local['source_snapshot']),
        ['pdblend/profile/local_power_job.py', *overlay_files])
    # A source overlay must not invalidate the already tested local power package.
    manifest = json.loads((Path(local['package'])/'manifest.json').read_text())
    for name, sha in manifest['implementation_sha256'].items():
        if digest(source/'pdblend'/name) != sha:
            raise ValueError('overlay changes local power package implementation: '+name)
    identity = dict(source_sha256=source_sha, member_bindings=bindings,
        long_review_sha256=digest(review), local_review_sha256=digest(LOCAL_REVIEW),
        depends_on=dependencies, image_digest=IMAGE, seed=701)
    caches = {}
    if cache_manifest:
        archive = json.loads(cache_manifest.read_text())['members']
        for name, member in members.items():
            matching = [value for value in archive.values() if
                (value['model_id'], value['tp'], value['pp'], value['image_digest']) ==
                (member['model_id'], member['tp'], 1, IMAGE)]
            if len(matching) != 1:
                raise ValueError('compiler cache must uniquely match model/topology/image')
            value = matching[0]
            if any(digest(Path(value['path'])/file) != sha for file, sha in value['files'].items()):
                raise ValueError('archived compiler cache changed')
            caches[name] = value
        identity['compiler_cache_manifest_sha256'] = digest(cache_manifest)
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cohort = 'incremental-profile-4-2-1-1-'+suffix
    out.mkdir(parents=True, exist_ok=False)
    wave = out/'wave'; wave.mkdir()
    write_new(wave/'wave.json', dict(cohort_id=cohort, coordinator=True,
        members=list(EXPECTED), synchronize_parallel_windows=True,
        keep_peers_resident_until_all_done=True, qualification_frequency_mhz=2100,
        interference_limit=.05, purpose='incremental_calibration_4_plus_2_plus_1_plus_1'))
    jobs = []
    for index, name in enumerate(EXPECTED):
        member = members[name]; tp = member['tp']
        job_id = 'incremental-'+name+'-'+suffix
        mounts = [(str(source), '/opt/pdblend-src', 'ro'), ('/home/models', '/models', 'ro'),
            (str(VERIFY), '/verification/model-verification.json', 'ro'),
            ('{attempt_dir}', '/output', 'rw'), (str(wave), '/wave', 'rw'),
            (str(ROOT/'results/2026-09-22/three-model/profile-wave'), '/coord', 'rw')]
        for value in sorted(set(member['readonly_roots'])):
            path = Path(value).resolve(strict=True)
            mounts.append((str(path), str(path), 'ro'))
        if name in caches:
            working_cache = out/'compiler-cache'/name
            shutil.copytree(caches[name]['path'], working_cache)
            mounts.append((str(working_cache), '/root/.cache/vllm', 'rw'))
        if member['kind'] == 'local_power':
            command = ['pdblend.profile.local_power_job', '--package', member['package']]
            required = ['queue-completion.json']
        elif member['kind'] == 'followup':
            command = ['pdblend.profile.long_context_followup', 'run', '--package', member['package']]
            required = ['completion.json']
        else:
            command = ['pdblend.profile.long_context_collect', '--plan', member['plan']]
            required = ['completion.json']
        argv = ['docker','run','--rm','--name',job_id,'--gpus','all','--cap-add','SYS_ADMIN',
            '--ipc=host','--network=host','--shm-size=16g','--ulimit','nofile=65536:65536',
            '--entrypoint','/opt/venv/bin/python']
        for host, target, mode in mounts: argv += ['-v', f'{host}:{target}:{mode}']
        environment = dict(PYTHONPATH='/opt/pdblend-src', PYTHONDONTWRITEBYTECODE='1',
            PDBLEND_MODELS_DIR='/models', PDBLEND_MODEL_VERIFICATION_RECEIPT='/verification/model-verification.json',
            PDBLEND_SOURCE_SHA256=source_sha, PDBLEND_IMAGE_ID=IMAGE, PDBLEND_HARDWARE_ID='8xL20-lease',
            PDBLEND_VLLM_VERSION='0.10.1.1', CUDA_VERSION='12.8.1',
            PDBLEND_GPU_UUIDS='{lease_gpu_uuids}', PDBLEND_COORD_DIR='/coord',
            PDBLEND_PROFILE_WAVE='/wave', PDBLEND_PROFILE_MEMBER=name,
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        for key, value in environment.items(): argv += ['-e', key+'='+value]
        for key in ('PDBLEND_CONCURRENCY_ENVIRONMENT','PDBLEND_CONCURRENCY_ENVIRONMENT_SHA256'):
            argv += ['-e', key]
        argv += [IMAGE, '-B', '-m', *command, '--model', '/models/'+member['model_id'],
            '--gpus', *map(str, range(tp)), '--base-port', '{lease_port}', '--out', '/output']
        payload = dict(schema=1, system='pdblend', model_id=member['model_id'], tp=tp, pp=1,
            gpu_count=tp, container_name=job_id, source_sha256=source_sha, source_snapshot=str(source),
            image_digest=IMAGE, cohort_id=cohort, cohort_dir=str(wave), cohort_member=name,
            profile_wave_member=name, profile_wave_members=list(EXPECTED), depends_on=dependencies,
            formal_eligible=False, energy_comparable=False, exclusive=False, global_lock=False,
            timeout_s=7200, required_receipts=required, argv=argv, immutable_input_sha256=bindings[name],
            measurement_scope=member, completion_is_formal_qualification=False)
        if name in caches:
            payload['compiler_cache_reuse'] = dict(archive=caches[name]['path'],
                manifest_sha256=identity['compiler_cache_manifest_sha256'],
                working_copy=str(out/'compiler-cache'/name))
        # The four-card member is claimed first; each subsequent group fits the
        # remaining contiguous GPUs. The allocator still binds physical UUIDs.
        jobs.append(dict(job_id=job_id, payload=payload, priority=330-index, max_attempts=1))
    write_new(out/'jobs.json', jobs)
    write_new(out/'manifest.json', dict(identity, cohort_id=cohort, source_snapshot=str(source),
        gpu_count=sum(j['payload']['gpu_count'] for j in jobs), prepare_only=True,
        jobs_sha256=digest(out/'jobs.json')))
    return out/'jobs.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depends-on', nargs='+', required=True)
    parser.add_argument('--overlay-files', nargs='+', required=True)
    parser.add_argument('--cache-manifest', type=Path)
    args = parser.parse_args()
    print(build(review=args.long_review.resolve(), out=args.out.resolve(),
        dependencies=args.depends_on, overlay_files=args.overlay_files,
        cache_manifest=args.cache_manifest))


if __name__ == '__main__':
    main()
