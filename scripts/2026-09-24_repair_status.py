#!/usr/bin/env python3
"""Compact, read-only repair status; no lease credentials or raw queue payloads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def summarize(root,queue_path):
    root=Path(root).resolve();queue_path=Path(queue_path).resolve()
    queue=read(queue_path) or {};attempt_root=queue_path.parent/'queue-attempts'
    active=read(root/'active-low-m.json') or {}
    package=Path(active.get('package',str(root/'low-m-v3')))
    driver=read(package/'driver/state.json') or {}
    rows=[]
    for stage in driver.get('stages',[]):
        job=queue.get('jobs',{}).get(stage['job_id'],{});trials=[]
        for trial_id in stage['trial_ids']:
            paths=sorted((attempt_root/stage['job_id']).glob('attempt-*/session/trials/'+trial_id+'/verdict.json'))
            for path in paths:
                verdict=read(path)
                if verdict is None:continue
                metrics=verdict.get('metrics',{})
                requests=read(path.parent/'comparison-metrics.json') or {}
                meter=read(path.parent/'comparison-metering.json') or {}
                energy=[meter.get(k) for k in ('energy_service_j','energy_tail_j','energy_service_tail_j')]
                complete=(meter.get('energy_comparable') is True and meter.get('gpu_count')==8
                    and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in energy)
                    and math.isclose(energy[0]+energy[1],energy[2],rel_tol=1e-8,abs_tol=.01))
                trials.append(dict(trial_id=trial_id,accepted=verdict.get('accepted'),reason=verdict.get('error'),
                    total_energy_kj=energy[2]/1000 if complete else None,full_energy_observed=complete,
                    requests=requests.get('offered_requests'),successful_requests=requests.get('successful_requests'),
                    joint_slo_requests=requests.get('joint_slo_requests'),
                    stable_low_m_s=metrics.get('stable_low_m_s'),verdict=str(path)))
        rows.append(dict(stage=stage['index'],kind=stage['kind'],job_id=stage['job_id'],
            status=job.get('status',stage.get('status')),planned_trials=len(stage['trial_ids']),
            complete_raw_trials=len(trials),accepted_raw_trials=sum(t['accepted'] is True for t in trials),
            rejected_raw_trials=sum(t['accepted'] is False for t in trials),trials=trials))
    followup=read(root/'followup/followup-status.json') or {}
    latest=read(root/'report-watch/latest.json') or {};snapshot=read(latest.get('snapshot',{}).get('path','')) or {}
    return dict(captured_at=datetime.now(timezone.utc).isoformat(),hardware_executed=False,goal_complete=False,
        coordination=read(root/'external-scope-conflict.json'),worker_stop_present=(root/'worker-v2.stop').exists(),
        note='Raw completed trials include in-progress sessions; full capacity qualification still requires final cleanup and frozen revalidation.',
        running_jobs=[dict(job_id=k,priority=j.get('priority')) for k,j in queue.get('jobs',{}).items() if j.get('status')=='running'],
        queued_jobs=sum(j.get('status')=='queued' for j in queue.get('jobs',{}).values()),
        tuning_package=str(package),tuning_revision=active.get('source_revision'),
        tuning_driver_status=driver.get('status'),tuning_diagnosis=driver.get('diagnosis'),tuning_stages=rows,
        raw_trial_counts=dict(Counter('accepted' if t['accepted'] is True else 'rejected' if t['accepted'] is False else 'unknown'
                                     for stage in rows for t in stage['trials'])),
        followup_status=followup.get('status'),matrix_path=followup.get('matrix_path'),
        report=latest.get('path'),baseline_conditions=snapshot.get('baseline_conditions'),
        baseline_total=snapshot.get('baseline_selected_points'),baseline_energy_complete=snapshot.get('baseline_complete_energy_points'),
        baseline_supplements_selected=snapshot.get('baseline_supplement_selected_points'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/pdblend4/results/2026-09-24/pdblend-energy-repair-v1'))
    p.add_argument('--queue',type=Path,default=Path('/home/pdblend4/results/2026-09-22/three-model/queue.json'))
    args=p.parse_args();print(json.dumps(summarize(args.root,args.queue),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
