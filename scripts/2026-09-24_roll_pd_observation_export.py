#!/usr/bin/env python3
"""Replace the sole CSV writer while a PD resident group is loading."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/home/pdblend4')
CSV=ROOT/'results/compare.csv'
QUEUE=ROOT/'results/2026-09-22/three-model/queue.json'


def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')


def frozen_values():
    rows=list(csv.DictReader(CSV.open()))
    prefixes=('energy_','service_mean_','goodput_','cohort_','throughput_',
              'ttft_','tpot_','gpu','window_good_')
    exact={'offered_requests','successful_requests','failed_requests','output_tokens',
           'joint_slo_requests','success_rate','joint_slo_rate','tail_s','slo_pass',
           'evidence_valid','formal_eligible','baseline_frozen','point_sha256',
           'trace_sha256','revision','receipt_sha256'}
    return {r['receipt_sha256']:{k:v for k,v in r.items()
            if (k in exact or k.startswith(prefixes)) and not k.startswith('energy_rank')}
            for r in rows if r['baseline_frozen']=='True'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--analysis-policy',choices=['all_recorded_windows/v1'])
    parser.add_argument('--previous-launch',type=Path,
        default=ROOT/'results/2026-09-24/eco-unattempted-v7-export-rollout-v1/launched.json')
    parser.add_argument('--boundary-jobs',type=Path,
        help='Allow the next switch only while one of these resident jobs is loading')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False)
    launch=json.loads(args.previous_launch.read_text())
    old_pid=launch['pid'];expected=[str(x) for x in launch['argv']]
    campaign=args.campaign.resolve();jobs=json.loads((args.boundary_jobs or campaign.parent/'jobs.json').read_text())
    ids={j['job_id'] for j in jobs}
    argv=list(expected);argv[argv.index('--campaign')+1]=str(campaign)
    if '--qualification-evidence' not in argv:
        argv+=['--qualification-evidence',str(ROOT/'results/2026-09-24/comparison-qualification-attempt-evidence-v1/manifest.json')]
    if args.analysis_policy:
        if '--analysis-policy' in argv:
            argv[argv.index('--analysis-policy')+1]=args.analysis_policy
        else:
            argv+=['--analysis-policy',args.analysis_policy]
    write(out/'waiting.json',dict(old_pid=old_pid,argv=argv,waiting_s=time.time(),
        safe_window='first 30 seconds after a new PD resident lease is acquired'))
    while True:
        state=json.loads(QUEUE.read_text())
        running=[j for j in state['jobs'].values() if j['status']=='running']
        if len(running)==1 and running[0]['job_id'] in ids:
            lease=state['leases'][running[0]['lease_id']]
            windows=Path(lease['attempt_dir'])/'session/windows'
            loading=not windows.exists() or not any(windows.iterdir())
            if 0<=time.time()-lease['claimed_at']<30 or loading:
                boundary=dict(job_id=running[0]['job_id'],claimed_s=lease['claimed_at']);break
        if all(state['jobs'].get(j,{}).get('status') in ('succeeded','failed','cancelled','blocked') for j in ids) and not running:
            boundary=dict(no_active_gpu_job=True);break
        time.sleep(1)
    before=frozen_values();write(out/'before-frozen-values.json',before)
    path=Path(f'/proc/{old_pid}/cmdline')
    if path.exists():
        actual=[x.decode() for x in path.read_bytes().split(b'\0') if x]
        if actual!=expected:
            raise RuntimeError('old exporter PID was reused or command changed')
        os.kill(old_pid,signal.SIGTERM)
        deadline=time.monotonic()+10
        while path.exists() and path.read_bytes() and time.monotonic()<deadline:
            time.sleep(.1)
        if path.exists() and path.read_bytes():
            raise RuntimeError('old exporter has not stopped')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        if b'pdblend.bench.comparison_campaign' in cmd and str(CSV).encode() in cmd:
            raise RuntimeError('another CSV writer exists')
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'watcher.log').open('x') as log:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'launched.json',dict(pid=child.pid,argv=argv,at_s=time.time(),boundary=boundary))
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        rows=list(csv.DictReader(CSV.open()))
        if rows and rows[0]['campaign_id']==campaign.parent.name:break
        if child.poll() not in (None,0):raise RuntimeError('new CSV exporter failed')
        time.sleep(1)
    else:raise TimeoutError('new CSV was not published')
    after=frozen_values()
    if any(after.get(key)!=value for key,value in before.items()):
        raise RuntimeError('frozen measurement values changed')
    write(out/'completion.json',dict(status='passed',pid=child.pid,
        prior_frozen_rows=len(before),current_frozen_rows=len(after),frozen_values_unchanged=True,
        finished_s=time.time()))


if __name__=='__main__':main()
