#!/usr/bin/env python3
"""Start the next CSV watcher during a functional probe or engine loading."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/pdblend4')
QUEUE=ROOT/'results/2026-09-22/three-model/queue.json'
OUT=ROOT/'results/2026-09-24/eco-unattempted-v7-export-rollout-v1'
CAMPAIGN=ROOT/'results/2026-09-24/resident-comparison-eco-unattempted-v7/campaign.json'
CSV=ROOT/'results/compare.csv'
PROBES={'dynamo-stationary-ipc-d7bd3ee0d78dae9a-schedule-v1',
        'dynamo-stationary-kv-8fe989bce5d21ba4-schedule-v1'}
ECO='comparison-7b-ac1ffa51e9a42ef0'


def write(name,value):
    with (OUT/name).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


def measured():
    prefixes=('energy_','service_mean_','goodput_','cohort_','throughput_','ttft_','tpot_','gpu','window_good_')
    exact={'offered_requests','successful_requests','failed_requests','output_tokens','joint_slo_requests',
           'success_rate','joint_slo_rate','tail_s','slo_pass','evidence_valid','formal_eligible','baseline_frozen',
           'point_sha256','trace_sha256','revision'}
    with CSV.open() as stream:rows=list(csv.DictReader(stream))
    return {r['receipt_sha256']:{k:v for k,v in r.items() if (k in exact or k.startswith(prefixes))
                              and not k.startswith('energy_rank')}
            for r in rows if r['status']=='measured'}


def safe_boundary(state):
    running=[j for j in state['jobs'].values() if j['status']=='running']
    if running and all(j['job_id'] in PROBES for j in running):return 'functional_probe'
    if len(running)==1 and running[0]['job_id']==ECO:
        lease=state['leases'][running[0]['lease_id']]
        if 0<=time.time()-lease['claimed_at']<20:return 'initial_engine_loading'
    if not running and all(state['jobs'][j]['status'] not in ('queued','running') for j in PROBES):
        return 'no_active_gpu_job'
    return None


def main():
    OUT.mkdir(exist_ok=False)
    old=json.loads((ROOT/'results/2026-09-24/comparison-export-update-rollout-v1/launched.json').read_text())
    argv=old['argv'];argv[argv.index('--campaign')+1]=str(CAMPAIGN)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'pdblend.bench.comparison_campaign' in cmd and str(CSV).encode() in cmd:
            raise RuntimeError('an existing CSV exporter is still running')
    write('waiting.json',dict(argv=argv,started_s=time.time(),safe_window='functional probe or first model loading'))
    while True:
        boundary=safe_boundary(json.loads(QUEUE.read_text()))
        if boundary:break
        time.sleep(1)
    before=measured();write('before-frozen-metrics.json',before)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (OUT/'watcher.log').open('x') as log:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write('launched.json',dict(pid=child.pid,argv=argv,at_s=time.time(),boundary=boundary))
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        with CSV.open() as stream:first=next(csv.DictReader(stream))
        if first['campaign_id']==CAMPAIGN.parent.name:break
        if child.poll() not in (None,0):raise RuntimeError('CSV exporter failed; inspect watcher.log')
        time.sleep(1)
    else:raise TimeoutError('CSV watcher did not publish the new campaign')
    after=measured()
    if any(after.get(key)!=value for key,value in before.items()):
        raise RuntimeError('frozen numerical/identity fields changed')
    write('completion.json',dict(status='passed',pid=child.pid,finished_s=time.time(),
        old_measured_rows=len(before),new_measured_rows=len(after),frozen_metrics_unchanged=True))


if __name__=='__main__':main()
