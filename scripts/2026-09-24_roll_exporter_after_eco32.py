#!/usr/bin/env python3
"""Publish the reviewed export update after the current measurement lease."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/pdblend4')
QUEUE=ROOT/'results/2026-09-22/three-model/queue.json'
AFTER='comparison-32b-6597548ffb85825c'
OLD_PID=1994443
OUT=ROOT/'results/2026-09-24/comparison-export-update-rollout-v1'
READY=ROOT/'results/2026-09-24/comparison-export-update-readiness-v1/review.json'
CSV=ROOT/'results/compare.csv'


def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(name,value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def numerical_rows():
    rows=list(csv.DictReader(CSV.open()))
    prefixes=('energy_','service_mean_','goodput_','cohort_','throughput_','ttft_','tpot_','gpu','window_good_')
    exact={'offered_requests','successful_requests','failed_requests','output_tokens','joint_slo_requests',
           'success_rate','joint_slo_rate','tail_s','slo_pass','evidence_valid','formal_eligible','baseline_frozen'}
    return {r['receipt_sha256']:{k:v for k,v in r.items() if (k in exact or k.startswith(prefixes))
                              and not k.startswith('energy_rank')}
            for r in rows if r['status']=='measured'}


def safe_boundary(state):
    job=state['jobs'][AFTER]
    if job['status'] not in ('succeeded','failed','cancelled') or job.get('lease_id') is not None:
        return False
    running=[j for j in state['jobs'].values() if j['status']=='running']
    for current in running:
        kind=current['payload'].get('kind')
        if kind in ('same_gpu_stationary_cuda_ipc_primitive','source_KV_lifetime_only_not_TP_conversion'):
            continue
        if current['job_id']=='pdblend-native-timing-32b-3b60fd5c83eabada':
            lease=state['leases'][current['lease_id']]
            if not (Path(lease['attempt_dir'])/'native-timing/runtime').exists():
                continue
        return False
    return True


def main():
    OUT.mkdir(exist_ok=False)
    old_cmd=Path(f'/proc/{OLD_PID}/cmdline').read_bytes().decode().split('\0')
    old_cmd=[x for x in old_cmd if x]
    assert 'pdblend.bench.comparison_campaign' in old_cmd and str(CSV) in old_cmd
    write('waiting.json',dict(after_job=AFTER,old_pid=OLD_PID,argv=old_cmd,readiness=ref(READY),started_s=time.time()))
    while not safe_boundary(json.loads(QUEUE.read_text())):
        time.sleep(1)
    review=json.loads(READY.read_text())
    for item in review['sources'].values():
        assert ref(Path(item['path']))['sha256']==item['sha256'], 'reviewed exporter source changed'
    before=numerical_rows()
    proc_path=Path(f'/proc/{OLD_PID}/cmdline')
    if proc_path.exists():
        actual=[x for x in proc_path.read_bytes().decode().split('\0') if x]
        assert actual==old_cmd, 'previous exporter PID was reused'
        os.kill(OLD_PID,signal.SIGTERM)
        for _ in range(30):
            if not proc_path.exists():break
            time.sleep(.1)
        assert not proc_path.exists(), 'previous exporter did not stop'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OUT/'watcher.log').open('x') as log:
        child=subprocess.Popen(old_cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write('launched.json',dict(at_s=time.time(),pid=child.pid,argv=old_cmd,readiness=ref(READY),
        old_measured_rows=len(before),first_full_hash_outside_measurement=True))
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        with CSV.open() as stream:
            header=next(csv.reader(stream))
        if 'observed_comparison_scope' in header:break
        assert child.poll() in (None,0), 'new exporter failed; inspect watcher.log'
        time.sleep(1)
    else:raise TimeoutError('new exporter did not publish the reviewed columns')
    after=numerical_rows()
    assert all(after.get(key)==value for key,value in before.items()), 'frozen numerical rows changed'
    write('completion.json',dict(status='passed',finished_s=time.time(),pid=child.pid,
        old_measured_rows=len(before),new_measured_rows=len(after),
        previous_frozen_numerics_unchanged=True,observed_columns_published=True,csv=ref(CSV),
        baseline_files_modified=False,old_process_stopped=True))
    print(json.dumps(dict(status='passed',pid=child.pid,receipt=str(OUT/'completion.json'))),flush=True)


if __name__=='__main__':main()
