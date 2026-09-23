#!/usr/bin/env python3
"""Prepare a fresh synchronized cohort, retaining the completed 14B holdout cells.

Stop and release the old owned cohort first. This script never interrupts a
running job or mutates its evidence. Enqueue the four-card member first so its
prior physical UUIDs can be checked before dispatching the other members.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from pdblend.experimentation.lease import GPULeaseQueue


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--cache-manifest', type=Path)
    parser.add_argument('--enqueue', action='store_true')
    args = parser.parse_args()
    queue = GPULeaseQueue(ROOT/'results/2026-09-22/three-model/queue.json')
    state = queue.snapshot()
    old = json.loads(args.spec.read_text())
    entries = [entry for entry in old['jobs'] if
               entry['payload'].get('cohort_id', '').startswith('quad-profile-4-2-1-1-')]
    if len(entries) != 4:
        raise ValueError('expected exactly four old quad members')
    prior = None
    for entry in entries:
        job = state['jobs'][entry['job_id']]
        if job['status'] not in ('failed', 'cancelled', 'blocked'):
            raise ValueError('old member must be stopped and released: '+entry['job_id'])
        leases = [lease for lease in state['leases'].values() if lease['job_id'] == entry['job_id']]
        lease = max(leases, key=lambda item: item['claimed_at'])
        if lease['status'] == 'active':
            raise ValueError('old member still owns an active lease')
        folder = Path(lease['attempt_dir'])
        raw = json.loads((folder/'raw.json').read_text())
        if entry['payload']['profile_wave_member'] == '14b-tp4-holdout':
            prior = dict(folder=folder, raw_sha256=digest(folder/'raw.json'),
                         gpu_uuids=lease['gpu_uuids'], raw=raw)
        elif any(raw.get(section) for section in ('prefill', 'decode', 'mixed')):
            raise ValueError('non-holdout member has completed points; add an explicit reuse path first')
    if prior is None:
        raise ValueError('14B prior holdout missing')
    helper_spec = importlib.util.spec_from_file_location('freeze_quad', ROOT/'scripts/2026-09-22_enqueue_parallel_profiles.py')
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    bases = {next(x.split(':/opt/pdblend-src:ro')[0] for x in entry['payload']['argv']
                  if x.endswith(':/opt/pdblend-src:ro')) for entry in entries}
    if len(bases) != 1:
        raise ValueError('expected one previous frozen source')
    with tempfile.TemporaryDirectory(prefix='synchronized-quad-overlay-') as directory:
        staging = Path(directory)/'src'; shutil.copytree(next(iter(bases)), staging)
        for filename in args.files:
            relative = Path(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe source overlay path')
            (staging/relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/'src'/relative, staging/relative)
        source, sha = helper.freeze_source(staging, ROOT/'results/2026-09-23/synchronized-quad-sources')
    identity = dict(source_sha256=sha, prior_raw_sha256=prior['raw_sha256'],
                    parent_spec_sha256=digest(args.spec), prior_gpu_uuids=prior['gpu_uuids'])
    caches = {}
    if args.cache_manifest:
        caches = json.loads(args.cache_manifest.read_text())['members']
        identity['compiler_cache_manifest_sha256'] = digest(args.cache_manifest)
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    wave_dir = output/'wave'
    wave_dir.mkdir()
    members = [entry['payload']['profile_wave_member'] for entry in entries]
    cohort = 'quad-profile-4-2-1-1-synchronized-'+suffix
    wave = dict(cohort_id=cohort, coordinator=True, members=members,
                synchronize_parallel_windows=True, keep_peers_resident_until_all_done=True,
                qualification_frequency_mhz=2100, interference_limit=.05,
                preserve_prior_complete_cells=True, prior_raw_sha256=prior['raw_sha256'])
    (wave_dir/'wave.json').write_text(json.dumps(wave, indent=2)+'\n')
    replacements, jobs = {}, []
    for entry in entries:
        job = deepcopy(entry); payload = job['payload']; member = payload['profile_wave_member']
        name = 'quad-sync-'+member+'-'+suffix
        replacements[entry['job_id']] = name
        job.update(job_id=name, priority=310, max_attempts=1)
        payload.update(container_name=name, source_sha256=sha, source_snapshot=str(source),
                       cohort_id=cohort, cohort_dir=str(wave_dir), cohort_member=member,
                       supersedes=entry['job_id'], timeout_s=10800)
        argv = payload['argv']; argv[argv.index('--name')+1] = name
        argv[:] = [str(source)+':/opt/pdblend-src:ro' if x.endswith(':/opt/pdblend-src:ro') else
                   str(wave_dir)+':/wave:rw' if x.endswith(':/wave:rw') else
                   'PDBLEND_SOURCE_SHA256='+sha if x.startswith('PDBLEND_SOURCE_SHA256=') else x for x in argv]
        if caches:
            cache = caches[member]; archive = Path(cache['path'])
            for key in ('model_id', 'tp', 'pp', 'image_digest'):
                if cache[key] != payload[key]: raise ValueError('compiler cache identity mismatch: '+key)
            if any(digest(archive/name) != expected for name, expected in cache['files'].items()):
                raise ValueError('compiler cache snapshot changed')
            working_cache = output/'compiler-cache'/member
            shutil.copytree(archive, working_cache)
            image_index = argv.index(payload['image_digest'])
            argv[image_index:image_index] = ['-v', str(working_cache)+':/root/.cache/vllm:rw']
            payload['compiler_cache_reuse'] = dict(archive=str(archive), working_copy=str(working_cache),
                manifest_sha256=identity['compiler_cache_manifest_sha256'], source_job=cache['source_job'])
        if member == '14b-tp4-holdout':
            image_index = argv.index(payload['image_digest'])
            argv[image_index:image_index] = ['-v', str(prior['folder'])+':/prior:ro']
            argv += ['--prior-holdout', '/prior', '--prior-raw-sha256', prior['raw_sha256']]
            payload.update(prior_raw_sha256=prior['raw_sha256'], prior_holdout=str(prior['folder']),
                           required_prior_gpu_uuids=prior['gpu_uuids'],
                           reused_points={section:len(prior['raw'].get(section, [])) for section in ('prefill', 'decode')})
        jobs.append(job)
    jobs.sort(key=lambda item: item['payload']['profile_wave_member'] != '14b-tp4-holdout')
    combined = deepcopy(old)
    combined['jobs'] = [entry for entry in combined['jobs'] if entry['job_id'] not in replacements]+jobs
    combined.update(parent_spec=str(args.spec.resolve()), synchronized_quad=dict(identity, replacements=replacements))
    for filename, value in [('jobs.json', jobs), ('combined-spec.json', combined), ('identity.json', identity)]:
        path = output/filename
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(value, indent=2)+'\n')
    if args.enqueue:
        first = jobs[0]
        queue.enqueue(**first)
        deadline = time.monotonic()+60
        while True:
            fresh = queue.snapshot(); record = fresh['jobs'][first['job_id']]
            if record['status'] == 'running':
                actual = fresh['leases'][record['lease_id']]['gpu_uuids']
                if actual != prior['gpu_uuids']:
                    raise RuntimeError('prior GPU placement differs; remaining members were not enqueued')
                break
            if record['status'] != 'queued' or time.monotonic() > deadline:
                raise RuntimeError('first member did not acquire prior GPUs; remaining members not enqueued')
            time.sleep(.5)
        for job in jobs[1:]: queue.enqueue(**job)
    print(json.dumps(dict(spec=str(output/'combined-spec.json'), jobs=[j['job_id'] for j in jobs],
                          reused=jobs[0]['payload']['reused_points'], enqueued=args.enqueue), indent=2))


if __name__ == '__main__': main()
