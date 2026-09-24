#!/usr/bin/env python3
"""Summarize the priority native A/B campaign from actual persisted receipts."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def snapshot(prepared, queue):
    review, state = read(prepared/'review.json'), read(queue)
    rows = []
    for jid in review['campaign_jobs']:
        job = state['jobs'].get(jid, {})
        leases = [v for v in state['leases'].values() if v['job_id'] == jid]
        lease = max(leases, key=lambda v: v['claimed_at']) if leases else {}
        attempt = Path(lease['attempt_dir']) if lease else None
        pd8 = '-pd8-' in jid
        member = '7b-pd8' if pd8 else jid.split('-ab-')[1].split('-')[0]
        cohort = read(prepared/('pd-eight-cohort' if pd8 else 'cohort')/(member+'.json'))
        stages = cohort.get('stages', {})
        functional = read(attempt/'functional/completion.json') if attempt else {}
        completion = read(attempt/'completion.json') if attempt else {}
        verdict = read(attempt/'audit.json') if attempt else {}
        arms = {arm: read(attempt/arm/'summary.json') for arm in ('A', 'B')} if attempt else {}
        stage = list(stages)[-1] if stages else ('loading_or_waiting_load_lock' if lease else 'queued')
        for arm in ('A', 'B'):
            if attempt and (attempt/arm/'controller.jsonl').exists() and not arms[arm]:
                stage = arm+'_warmup_or_service'
        rows.append(dict(job_id=jid, queue_status=job.get('status', 'not_published'),
            stage=stage, gpu_indices=lease.get('gpu_indices', []),
            lease_status=lease.get('status'), attempt_dir=str(attempt) if attempt else None,
            functional_passed=functional.get('functional_passed'),
            completed_arms=[arm for arm, summary in arms.items() if summary],
            execution_complete=completion.get('complete', False),
            audit_status=verdict.get('status'), classification=verdict.get('classification'),
            errors=completion.get('errors', []) or ([completion['error']] if completion.get('error') else []),
            formal_eligible=False, energy_comparable=False))
    return dict(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source_sha256=review['source_sha256'], prepared=str(prepared.resolve()),
        scope='synthetic_development_fixed_plan_vs_periodic_replanning', jobs=rows,
        formal_eligible=False, energy_comparable=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--queue', type=Path,
        default=Path('results/2026-09-22/three-model/queue.json'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        report = render(args)
        if not args.watch or all(row['queue_status'] in ('succeeded', 'failed', 'cancelled')
                                 for row in report['jobs']):
            break
        time.sleep(30.)


def render(args):
    report = snapshot(args.prepared, args.queue)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        temporary = args.out/'current.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.out/'current.json')
        lines = [f"更新：{report['updated_at']}", '',
            '本轮是 seed 701 合成开发轨迹上的固定计划与周期规划消融；不属于正式排名或总能耗比较。', '',
            '| 任务 | 队列 | 当前阶段 | GPU | 功能通过 | 完成臂 | 独立审计 |',
            '|---|---|---|---|---|---|---|']
        for row in report['jobs']:
            link = f"[{row['job_id']}]({row['attempt_dir']})" if row['attempt_dir'] else row['job_id']
            lines.append(f"| {link} | {row['queue_status']} | {row['stage']} | {row['gpu_indices']} | "
                f"{row['functional_passed']} | {row['completed_arms']} | {row['audit_status']} |")
            if row['errors']:
                lines.append('\n错误：'+repr(row['errors'])+'\n')
        temporary = args.out/'current.md.tmp'
        temporary.write_text('\n'.join(lines)+'\n')
        temporary.replace(args.out/'current.md')
    if not args.watch:
        print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
