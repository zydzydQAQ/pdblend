#!/usr/bin/env python3
"""Prepare, never enqueue, the 14B TP4 holdout + 7B TP4 training cohort.

Root owns source freezing and the eventual atomic replacement of the unstarted
14B job.  This script has no live-queue mutation API or Docker/GPU invocation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def readonly_mounts(argv, image):
    prefix = argv[:argv.index(image)]
    mounts = {}
    for i, value in enumerate(prefix[:-1]):
        if value == '-v':
            host, container, mode = prefix[i+1].rsplit(':', 2)
            if container in mounts:
                raise ValueError(f'duplicate container mount: {container}')
            mounts[container] = (host, mode)
    for required in ('/models', '/candidate', '/verification/model-verification.json', '/opt/pdblend-src'):
        if required not in mounts or mounts[required][1] != 'ro':
            raise ValueError(f'existing job does not bind {required} read-only')
    return mounts


def choose_unstarted_holdout(state):
    matches = [j for j in state['jobs'].values() if j.get('status') == 'queued' and
               j.get('payload', {}).get('model_id') == 'Qwen2.5-14B-Instruct' and
               j['payload'].get('tp') == 4 and
               j['payload'].get('evidence_class') == 'independent_calibration_holdout']
    if len(matches) != 1:
        raise ValueError('expected exactly one queued 14B TP4 holdout')
    job = matches[0]
    if job.get('attempts', 0) != 0 or job.get('lease_id') is not None:
        raise ValueError('14B TP4 replacement is allowed only before its first attempt')
    if any(x.get('job_id') == job['job_id'] for x in state.get('leases', {}).values()):
        raise ValueError('14B TP4 already has a lease record')
    if job['payload'].get('gpu_count') != 4 or job['payload'].get('exclusive') or job['payload'].get('global_lock'):
        raise ValueError('paired cohort requires a nonexclusive four-GPU holdout')
    return job


def source_binding(snapshot, source_hash):
    if (snapshot is None) != (source_hash is None):
        raise ValueError('source snapshot and hash must be provided together')
    if snapshot is None:
        return '{root_frozen_source_dir}', '{root_frozen_source_sha256}', False
    snapshot = Path(snapshot).resolve()
    if len(source_hash) != 64 or any(c not in '0123456789abcdef' for c in source_hash):
        raise ValueError('invalid frozen source SHA256')
    spec = importlib.util.spec_from_file_location('freeze_profile_source', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    recorded = json.loads((snapshot/'manifest.json').read_text())
    helper.verify_snapshot(snapshot, recorded['files'])
    if recorded['source_sha256'] != source_hash:
        raise ValueError('source snapshot differs from declared hash')
    for required in ('pdblend/profile/long_context_collect.py', 'pdblend/profile/calibration.py',
                     'pdblend/profile/window_sampling.py'):
        if required not in recorded['files']:
            raise ValueError(f'frozen source lacks required collector: {required}')
    return str(snapshot), source_hash, True


def build_review(state, *, plan_path, review_dir, snapshot=None, source_hash=None):
    old = choose_unstarted_holdout(state)
    payload = old['payload']
    image = payload['image_digest']
    mounts = readonly_mounts(payload['argv'], image)
    candidate_dir = Path(payload['candidate_dir']).resolve()
    candidate_manifest = json.loads((candidate_dir/'manifest.json').read_text())
    if (candidate_manifest.get('system'), candidate_manifest.get('model_id'), candidate_manifest.get('tp'),
            candidate_manifest.get('pp')) != ('pdblend', 'Qwen2.5-14B-Instruct', 4, 1):
        raise ValueError('existing holdout candidate has a different system/model/topology')
    if sha256(candidate_dir/'candidate.json') != candidate_manifest['candidate_sha256']:
        raise ValueError('existing holdout frozen candidate checksum mismatch')
    holdout_raw = Path(candidate_manifest['training_raw']).resolve()
    if sha256(holdout_raw) != candidate_manifest['training_raw_sha256']:
        raise ValueError('existing holdout training raw checksum mismatch')
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    training_raw = Path(plan['training_source']).resolve()
    if sha256(training_raw) != plan['training_source_sha256']:
        raise ValueError('long-context training source checksum mismatch')
    training_identity = json.loads(training_raw.read_text())
    if any(training_identity.get(k) != plan.get(k) for k in ('system', 'model_id', 'tp', 'pp')):
        raise ValueError('long-context plan and original training identities differ')
    if (plan.get('system'), plan.get('model_id'), plan.get('tp'), plan.get('pp')) != (
            'pdblend', 'Qwen2.5-7B-Instruct', 4, 1):
        raise ValueError('paired training member must be independent 7B TP4 PP1')
    if ({x.get('freq_mhz') for x in plan.get('training', [])} != {900, 1200, 1500, 1800, 2100, 2520} or
            any(x.get('purpose') != 'training_extension' for x in plan['training']) or
            plan.get('fit_existing_holdout') is not False):
        raise ValueError('training member must retain all six frequencies and exclude holdout points')
    source_dir, source_sha, frozen = source_binding(snapshot, source_hash)
    identity = dict(source_sha256=source_sha, image_digest=image,
                    candidate_sha256=candidate_manifest['candidate_sha256'],
                    plan_sha256=sha256(plan_path), training_source_sha256=sha256(training_raw),
                    original_holdout_payload_sha256=object_hash(payload))
    suffix = object_hash(identity)[:20]
    cohort_id = f'paired-14b-holdout-7b-longctx-{suffix}'
    review_dir = Path(review_dir).resolve()
    wave_dir = review_dir/'wave'
    members = ['14b-tp4-holdout', '7b-tp4-longctx']
    wave = dict(cohort_id=cohort_id, coordinator=True, members=members,
                qualification_frequency_mhz=2100, representative_layout_check_only=True,
                profile_frequency_coverage=[900, 1200, 1500, 1800, 2100, 2520],
                qualification_note='Existing ProfileWave protocol: paired isolated/concurrent B8 context1024 at 2100 MHz; not per-frequency interference validation.')

    def container_prefix(name, member, extra_mounts):
        # Preserve image/runtime options and all unrelated environment entries,
        # replacing only this cohort's source, identity and role-specific mounts.
        before = payload['argv'][:payload['argv'].index(image)]
        result = []
        index = 0
        replace_env = {'PDBLEND_SOURCE_SHA256', 'PDBLEND_PROFILE_MEMBER', 'PDBLEND_PROFILE_WAVE'}
        removed = {'/opt/pdblend-src', '/wave', '/candidate', str(holdout_raw)}
        while index < len(before):
            value = before[index]
            if value == '--name':
                result.extend([value, name]); index += 2; continue
            if value == '-v':
                mapping = before[index+1]
                host, target, mode = mapping.rsplit(':', 2)
                if target not in removed and host != str(holdout_raw):
                    result.extend([value, mapping])
                index += 2; continue
            if value == '-e':
                env = before[index+1]
                if env.split('=', 1)[0] not in replace_env:
                    result.extend([value, env])
                index += 2; continue
            result.append(value); index += 1
        result += ['-v', f'{source_dir}:/opt/pdblend-src:ro', '-v', f'{wave_dir}:/wave:rw',
                   '-e', 'PDBLEND_SOURCE_SHA256='+source_sha,
                   '-e', 'PDBLEND_PROFILE_WAVE=/wave', '-e', 'PDBLEND_PROFILE_MEMBER='+member]
        for host, target in extra_mounts:
            result += ['-v', f'{host}:{target}:ro']
        return result + [image, '-B', '-m']

    holdout_id = f'holdout-14b-tp4-paired-{suffix}'
    train_id = f'longctx-training-7b-tp4-paired-{suffix}'
    holdout = copy.deepcopy(payload)
    holdout.update(system='pdblend', pp=1, container_name=holdout_id, source_sha256=source_sha, cohort_id=cohort_id,
        profile_wave_members=members, profile_wave_member=members[0],
        qualification_frequency_mhz=2100, shared_prefill_windows=True,
        argv=container_prefix(holdout_id, members[0], [(candidate_dir, '/candidate'), (holdout_raw, '/training/raw.json')]) + [
            'pdblend.profile.calibration', '--candidate-dir', '/candidate', '--training-raw', '/training/raw.json',
            '--model', '/models/Qwen2.5-14B-Instruct', '--gpus', '0', '1', '2', '3',
            '--base-port', '{lease_port}', '--out', '/output', '--shared-prefill-windows'])
    train = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1,
        gpu_count=4, exclusive=False, global_lock=False, formal_eligible=False, energy_comparable=False,
        source_sha256=source_sha, image_digest=image, container_name=train_id,
        evidence_class='incremental_training_evidence', independent_holdout=False,
        training_plan_sha256=identity['plan_sha256'], training_raw_sha256=sha256(training_raw),
        expected_training_points=len(plan['training']), holdout_points_consumed=0,
        collection_scope=plan.get('completion_scope', 'declared_training_points'),
        original_training_points=plan.get('original_training_point_count', len(plan['training'])),
        deferred_training_points=len(plan.get('deferred_training', [])),
        coverage_constraints=copy.deepcopy(plan.get('coverage_constraints', {})),
        timeout_s=10800, required_receipts=['completion.json'], max_model_len=8192,
        cohort_id=cohort_id, profile_wave_members=members, profile_wave_member=members[1],
        qualification_frequency_mhz=2100, depends_on=list(payload.get('depends_on', [])),
        topology=dict(model_id='Qwen2.5-7B-Instruct', tp=4, pp=1, gpu_count=4),
        argv=container_prefix(train_id, members[1], [(plan_path, '/plan/long-context.json'), (training_raw, '/training/raw.json')]) + [
            'pdblend.profile.long_context_collect', '--plan', '/plan/long-context.json', '--training-raw', '/training/raw.json',
            '--model', '/models/Qwen2.5-7B-Instruct', '--gpus', '0', '1', '2', '3',
            '--base-port', '{lease_port}', '--out', '/output'])
    jobs = [dict(job_id=holdout_id, payload=holdout, priority=old['priority'], max_attempts=1),
            dict(job_id=train_id, payload=train, priority=old['priority'], max_attempts=1)]
    if jobs[0]['payload']['depends_on'] != jobs[1]['payload']['depends_on']:
        raise ValueError('cohort members must become eligible together')
    replacement = dict(original_job_id=old['job_id'], replacement_job_id=holdout_id,
        additional_job_id=train_id, required_status='queued', required_attempts=0, required_lease_id=None,
        expected_original_payload_sha256=object_hash(payload),
        reason='Unstarted third-wave holdout shares a qualified disjoint 4+4 GPU cohort with incremental 7B long-context training; original frozen candidate is unchanged.')
    return dict(schema=1, mode='prepare_only', live_queue_modified=False, source_frozen=frozen,
        ready_for_root_final_review=frozen, automatic_enqueue_allowed=False, identity=identity,
        jobs=jobs, replacement=replacement, wave=wave, wave_dir=str(wave_dir),
        immutable_inputs=[dict(path=str(p), sha256=sha256(p)) for p in
            (candidate_dir/'candidate.json', candidate_dir/'manifest.json', holdout_raw, plan_path, training_raw,
             Path(mounts['/verification/model-verification.json'][0]))],
        qualification_scope=wave['qualification_note'],
        commit_instructions=['Freeze current source with the established content-addressed helper.',
            'Re-run prepare with --source-snapshot and --source-sha256; review both manifests.',
            'Recheck only the original job status/attempts/lease/payload and immutable input hashes immediately before replacement.',
            'Root blocks only the unstarted original 14B job and enqueues both reviewed jobs together; do not mutate running jobs or existing wave files.'])


def write_immutable(path, value):
    data = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != data:
        raise ValueError(f'review artifact differs; choose a new output directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=ROOT/'results/2026-09-22/three-model/queue.json')
    parser.add_argument('--plan', type=Path, default=ROOT/'results/2026-09-23/long-context-plan/Qwen2.5-7B-Instruct-tp4.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--source-sha256')
    parser.add_argument('--prepare-only', action='store_true', help='explicit marker; this script always prepares only')
    args = parser.parse_args()
    state = json.loads(args.queue.read_text())
    review = build_review(state, plan_path=args.plan, review_dir=args.out,
                          snapshot=args.source_snapshot, source_hash=args.source_sha256)
    write_immutable(args.out/'wave/wave.json', review['wave'])
    write_immutable(args.out/'jobs.json', review['jobs'])
    write_immutable(args.out/'replacement.json', review['replacement'])
    write_immutable(args.out/'review.json', review)
    print(json.dumps(dict(mode='prepare_only', source_frozen=review['source_frozen'],
        output=str(args.out.resolve()), jobs=[x['job_id'] for x in review['jobs']],
        replacement=review['replacement']['original_job_id'], live_queue_modified=False), indent=2))


if __name__ == '__main__':
    main()
