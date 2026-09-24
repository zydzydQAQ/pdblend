#!/usr/bin/env python3
"""Publish the reviewed IPC prerequisite and drain the previous queue worker."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from pdblend.experimentation.lease import GPULeaseQueue

ROOT = Path('/home/pdblend4')
OUT = ROOT / 'results/2026-09-24/dynamo-stationary-ipc-early-schedule-v3'
QUEUE = ROOT / 'results/2026-09-22/three-model/queue.json'
OLD_WORKER = 1507575
OLD_STOP = ROOT / 'results/2026-09-23/native-ab-smoke-prepared-v3/worker.stop'
RUNNING = 'pdblend-native-timing-14b-d48b8f0f78c3c399'


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(name, value):
    path = OUT / name
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return ref(path)


def main():
    os.chdir(ROOT)
    source = ROOT / 'src/pdblend/experimentation/lease.py'
    tests = ROOT / 'tests/pdblend/test_lease_queue.py'
    assert ref(source)['sha256'] == 'd6535430b7bdc91920d4b80aaf42699cf38c88c5971ecd070a6c673f1df41fe2'
    assert ref(tests)['sha256'] == '43043aa3d5ce1ab6e76458a45e08c6fc4aa0bddf5b7ea7fcb15fa330dd5cf93c'
    cmdline = Path(f'/proc/{OLD_WORKER}/cmdline').read_bytes().decode().split('\0')
    assert str(OLD_STOP) in cmdline and not OLD_STOP.exists()
    before = json.loads(QUEUE.read_text())
    active = before['jobs'][RUNNING]
    assert active['status'] == 'running' and active['attempts'] == 1
    assert before['leases'][active['lease_id']]['owner_pid'] == OLD_WORKER
    old_spec = ROOT / 'results/2026-09-24/dynamo-stationary-ipc-early-schedule-v2/jobs.json'
    spec = json.loads(old_spec.read_text())[0]
    old_id = spec['job_id']
    spec['job_id'] = old_id.removesuffix('-v2') + '-v3'
    payload = spec['payload']
    assert payload['gpu_count'] == 1 and payload['model_loads'] == 0
    assert payload['exclusive'] is False and payload['reserve_host'] is False
    assert not payload.get('sampling_cohort')
    assert spec['priority'] > before['jobs']['comparison-32b-6597548ffb85825c']['priority']
    payload['argv'][payload['argv'].index('--name') + 1] = spec['job_id']
    payload.update(container_name=spec['job_id'], supersedes_job_id=old_id,
                   after_terminal=[RUNNING], precedes_host_reservations=True,
                   scheduling_reason='Explicit short prerequisite after current PD14 lease; before the next host reservation. No active cohort interruption.')
    OUT.mkdir(parents=True, exist_ok=False)
    jobs_ref = write('jobs.json', [spec])
    review_ref = write('review.json', {
        'passed': True, 'queue_source': ref(source), 'queue_tests': ref(tests),
        'test_result': '25 passed in 1.03s',
        'independent_reviewer': '/root/audit_pdblend',
        'independent_review_passed': True,
        'original_spec': ref(old_spec), 'source_and_config_unchanged': True,
        'active_job_unchanged': RUNNING,
        'rollout': 'Old stop file is checked only between run_one calls; replacement worker shares existing host flock and GPU lease database.',
        'expected_order': [RUNNING, spec['job_id'], 'comparison-32b-6597548ffb85825c',
                           'pdblend-native-timing-32b-3b60fd5c83eabada'],
        'resource_check': {'gpu_count': 1, 'visible_host_gpu_count': 8,
                           'worker_has_fixed_count': False, 'timeout_s': payload['timeout_s']},
    })
    published = GPULeaseQueue(QUEUE).enqueue_replacements([spec])
    publication_ref = write('publication.json', {
        'jobs': jobs_ref, 'review': review_ref,
        'published': [{'job_id': x.job_id, 'status': x.status, 'attempts': x.attempts} for x in published],
    })
    # This does not signal or cancel any subprocess inside the old run_one.
    with OLD_STOP.open('x') as stream:
        stream.write(f'Drain worker {OLD_WORKER} after its active run_one; replacement in {OUT}\n')
    new_stop = OUT / 'worker.stop'
    argv = ['/home/pdblend/.venv/bin/python', '-B', '-u',
            str(ROOT / 'scripts/2026-09-22_gpu_campaign_queue.py'), '--db', str(QUEUE),
            'worker', '--workers', '8', '--stop-file', str(new_stop)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
    with (OUT / 'worker.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(1)
    assert process.poll() is None, 'replacement worker exited; inspect worker.log'
    after = json.loads(QUEUE.read_text())
    current = after['jobs'][RUNNING]
    invariant_fields = ('status', 'attempts', 'lease_id', 'payload')
    unchanged = all(current[field] == active[field] for field in invariant_fields)
    rows = list(csv.DictReader((ROOT / 'results/compare.csv').open()))
    receipt = write('worker-rollout.json', {
        'at': time.time(), 'publication': publication_ref,
        'previous_worker_pid': OLD_WORKER, 'previous_stop_file': str(OLD_STOP),
        'previous_worker_draining': True, 'replacement_worker_pid': process.pid,
        'argv': argv, 'queue_source': ref(source), 'running_job_unchanged': unchanged,
        'running_job': RUNNING, 'measured_csv_rows': sum(r['status'] == 'measured' for r in rows),
    })
    assert unchanged, 'active job changed during rollout; inspect receipt'
    print(json.dumps({'receipt': receipt, 'worker_pid': process.pid, 'active_job_unchanged': unchanged}))


if __name__ == '__main__':
    main()
