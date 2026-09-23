#!/usr/bin/env python3
"""Refresh seed-701 sampling/calibration/function/formal status; read-only inputs."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MODELS = tuple('Qwen2.5-'+size+'-Instruct' for size in ('7B', '14B', '32B'))
SYSTEMS = ('mixed', 'distserve', 'dynamollm', 'ecoserve', 'pdblend')
TERMINAL = {'succeeded', 'failed', 'cancelled'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def gpu_snapshot():
    """Read instantaneous utilization; this is not an energy measurement."""
    value = dict(sampled_at_s=time.time(), scope='instantaneous_gpu_utilization', rows=[])
    try:
        result = subprocess.run(['nvidia-smi',
            '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            index, uuid, utilization, memory, power = [part.strip() for part in line.split(',')]
            value['rows'].append(dict(index=int(index), uuid=uuid,
                utilization_percent=float(utilization), memory_mib=float(memory), power_w=float(power)))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        value['error'] = str(exc)
    return value


def receipt(path):
    path = Path(path)
    if not path.is_file():
        return None
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError):
        return dict(path=str(path), sha256=sha_bytes(data), status='unreadable_partial_receipt', complete=False)
    keys = ('status', 'complete', 'scope', 'error', 'failures', 'cleanup_errors', 'seed', 'model_id',
            'complete_reproduction', 'formal_eligible', 'energy_comparable', 'controller_hierarchy_qualified',
            'automatic_policy_triggered', 'missing_required_actions', 'periods_s', 'requests', 'successful_requests')
    return dict(path=str(path.resolve()), sha256=sha_bytes(data), **{k:value[k] for k in keys if k in value})


def latest_lease(queue, jid):
    values = [x for x in queue.get('leases', {}).values() if x.get('job_id')==jid]
    return max(values, key=lambda x:x.get('claimed_at', 0)) if values else None


def sampling_progress(spec, attempt):
    """Count persisted measurements, without treating them as qualified points."""
    scope = spec.get('measurement_scope', {})
    if not scope:
        return None
    phases = []
    def add(name, relative, field, expected):
        if expected is None:
            return
        path = Path(attempt) / relative if attempt else None
        raw, readable = {}, False
        if path and path.is_file():
            try:
                raw = json.loads(path.read_text())
                readable = True
            except (ValueError, OSError):
                pass
        rows = raw.get(field, [])
        count = (sum(len(r.get('repeats', [])) == r.get('point', {}).get('repeats')
                     for r in rows.values()) if isinstance(rows, dict) else len(rows))
        phases.append(dict(phase=name, measured_points=count, expected_points=expected,
                           raw_path=str(path) if path else None, raw_readable=readable,
                           calibration_qualified=False))
    if scope.get('kind') == 'local_power':
        add('power_holdout', 'raw.json', 'decode', scope.get('expected_power_points'))
        add('mixed_repair', 'mixed-repair/raw.json', 'mixed', scope.get('expected_mixed_points'))
    elif scope.get('kind') == 'experimental_short_and_optional_qualified_long_resident':
        add('independent_long_holdout', 'long-holdout/raw.json', 'decode', scope.get('long_holdout_points'))
        add('experimental_short_training', 'short/raw.json', 'training', scope.get('short_training_points'))
        add('experimental_short_holdout', 'short/raw.json', 'holdout', scope.get('short_holdout_points'))
    else:
        add('endpoint_training', 'raw.json', 'decode', scope.get('expected_training_points'))
        if scope.get('expected_holdout_points', 0):
            add('independent_holdout', 'long-holdout/raw.json', 'decode', scope['expected_holdout_points'])
    done = (Path(spec['cohort_dir']) / (spec['cohort_member'] + '.done.json')
            if spec.get('cohort_dir') and spec.get('cohort_member') else None)
    return dict(phases=phases, scope='measurement_progress_only',
                collection_done=bool(done and done.is_file()),
                measured_points=sum(p['measured_points'] for p in phases),
                expected_points=sum(p['expected_points'] for p in phases))


def blocked_by(queue, jid, visited=()):
    """Mirror actual lease admission, including terminal-but-still-leased jobs."""
    if jid in visited:
        return [dict(job_id=jid, reason='dependency_cycle', chain=list(visited)+( [jid]))]
    job = queue['jobs'].get(jid, {})
    blockers = []
    for field in ('depends_on', 'after_terminal'):
        for dep in job.get('payload', {}).get(field, []):
            target = queue['jobs'].get(dep, {})
            active = target.get('lease_id') is not None or any(x.get('job_id')==dep and x.get('status')=='active' for x in queue.get('leases', {}).values())
            ready = target.get('status')=='succeeded' if field=='depends_on' else target.get('status') in TERMINAL and not active
            if ready:
                continue
            item = dict(job_id=dep, required='succeeded' if field=='depends_on' else 'terminal_and_lease_released',
                        status=target.get('status', 'missing'), active_lease=active,
                        superseded_by=target.get('superseded_by'), error=target.get('last_error'))
            item['upstream'] = blocked_by(queue, dep, visited+(jid,)) if target else []
            blockers.append(item)
    return blockers


def job_summary(queue, jid):
    job = queue['jobs'][jid]
    spec, lease = job.get('payload', {}), latest_lease(queue, jid)
    row = dict(job_id=jid, queue_status=job['status'], model_id=spec.get('model_id'), system=spec.get('system'),
               tp=spec.get('tp'), pp=spec.get('pp'), gpu_count=spec.get('gpu_count'), source_sha256=spec.get('source_sha256'),
               superseded_by=job.get('superseded_by'), error=job.get('last_error'),
               updated_at=job.get('updated_at'), created_at=job.get('created_at'), required_receipts=[])
    row['blocked_by'] = blocked_by(queue, jid) if job['status']=='queued' else []
    row['scheduler_state'] = 'superseded' if row['superseded_by'] else ('waiting_dependencies' if row['blocked_by'] else job['status'])
    row['sampling_progress'] = sampling_progress(spec, lease['attempt_dir'] if lease else None)
    if lease:
        row.update(attempt_dir=lease['attempt_dir'], gpu_indices=lease.get('gpu_indices', []),
                   gpu_uuids=lease.get('gpu_uuids', []), claimed_at=lease.get('claimed_at'), lease_status=lease.get('status'))
        row['required_receipts'] = [receipt(Path(lease['attempt_dir'])/name) for name in spec.get('required_receipts', [])]
    return row


def latest_stage(jobs, predicate):
    rows = [x for x in jobs if predicate(x) and not x['superseded_by']]
    if not rows:
        return None
    key=lambda x:(x.get('created_at') or 0, x.get('claimed_at') or 0)
    selected=dict(max(rows,key=key))
    succeeded=[x for x in rows if x['queue_status']=='succeeded']
    selected['last_succeeded_stage']=max(succeeded,key=key) if succeeded else None
    return selected


def evidence_stage(stage, relative=None):
    if stage is None:
        return dict(status='missing', receipt=None)
    proof = receipt(Path(stage['attempt_dir'])/relative) if relative and stage.get('attempt_dir') else None
    if relative is None:
        proof = next((r for r in stage['required_receipts'] if r), None)
    # Receipt-level success may arrive before a resident pair's second system finishes.
    status = proof.get('status') if proof else stage['scheduler_state']
    if status=='passed' and proof.get('complete') is not True:
        status='incomplete_receipt'
    last_passed=None
    if stage.get('last_succeeded_stage'):
        previous=evidence_stage(stage['last_succeeded_stage'],relative)
        if previous['status']=='passed':
            last_passed={k:previous[k] for k in ('job_id','receipt','source_sha256')}
    return dict(status=status, job_id=stage['job_id'], queue_status=stage['queue_status'], receipt=proof,
                blocked_by=stage['blocked_by'], source_sha256=stage['source_sha256'],last_passed=last_passed)


def matrix(queue, jobs, registry):
    rows=[]
    for model in MODELS:
        own = [x for x in jobs if x['model_id']==model]
        resident = latest_stage(own, lambda x:x['job_id'].startswith('dist-eco-resident-'))
        probe = latest_stage(own, lambda x:x['job_id'].startswith('native-probe-') and x['queue_status']=='succeeded')
        for system in SYSTEMS:
            if system in ('mixed', 'pdblend'):
                functional = evidence_stage(latest_stage(own, lambda x:x['job_id'].startswith('smoke-independent-'+system+'-')))
                mechanism = evidence_stage(probe, 'public-mixed-pd.json') if system=='pdblend' else functional
            elif system=='dynamollm':
                functional = evidence_stage(latest_stage(own, lambda x:x['job_id'].startswith(('dynamo-functional-', 'dynamo-reroute-retry-'))))
                mechanism = evidence_stage(latest_stage(own, lambda x:x['job_id'].startswith('native-dynamo-')))
            elif system=='distserve':
                functional = evidence_stage(resident, 'campaign/distserve/completion.json')
                mechanism = evidence_stage(latest_stage(own, lambda x:x['job_id'].startswith('native-distserve-pipeline-')))
            else:
                functional = evidence_stage(resident, 'campaign/ecoserve/completion.json')
                mechanism = evidence_stage(latest_stage(own, lambda x:x['job_id'].startswith(('native-eco4-', 'ecoserve-auto-macro-'))))
            gates = {
                'mixed': ['fixed_tp_placement', 'full_slo_and_total_energy_matrix'],
                'distserve': ['independent_stage_profile_calibration', 'qualified_symmetric_tp_pp1_search_and_deployment', 'real_pipeline_batch_boundary', 'full_slo_and_total_energy_matrix'],
                'dynamollm': ['model_bound_predictor_holdout', 'independent_profile_calibration', 'automatic_scaleinst_1800s', 'automatic_scaleshard_300s', 'automatic_scalefreq_5s', 'full_slo_and_total_energy_matrix'],
                'ecoserve': ['independent_csv_profile_calibration', 'automatic_macro_rotation_split_merge_parking', 'full_slo_and_total_energy_matrix'],
                'pdblend': ['independent_profile_calibration', 'offline_tp', 'resident_hetero_tp', 'slow_reshard_tp', 'full_slo_and_total_energy_matrix'],
            }[system]
            sampling = [dict(job_id=x['job_id'],status=x['scheduler_state'],tp=x['tp']) for x in own
                        if (x['system']==system or (system=='dynamollm' and x['job_id'].startswith('dynamo-gap-') and '-profile-' in x['job_id']))
                        and x['job_id'].startswith(('profile-', 'incremental-', 'holdout-', 'power-holdout-', 'resident-domain-', 'dynamo-gap-')) and not x['superseded_by']]
            versions = [v for v in registry.get('versions', []) if v['model_id']==model and v['system']==system]
            rows.append(dict(model_id=model, system=system, seed=701, pp1_pdblend=True,
                sampling=dict(jobs=sampling, independent=True, full_profile_qualified=False,
                    scope='fixed_placement_no_fitted_policy' if system=='mixed' else 'own_system_evidence_only'),
                calibration=dict(status='bounded_components_passed' if versions and all(v['calibration_components_passed'] for v in versions) else 'not_fully_qualified',
                    versions=[dict(version_id=v['version_id'],tp=v['tp'],power_passed=v['power_passed'],effective_timing_passed=v['effective_timing_passed']) for v in versions]),
                functional=functional, mechanism=mechanism, remaining_mechanism_and_campaign_gates=gates,
                complete_reproduction=False, formal_status='inconclusive', formal_eligible=False, energy_comparable=False))
    return rows


def build(queue, queue_binding, registry=None, now=None):
    jobs=[job_summary(queue, jid) for jid in queue['jobs']]
    active=[x for x in jobs if x['queue_status'] in ('running','queued') and not x['superseded_by']]
    leases=[x for x in queue.get('leases',{}).values() if x.get('status')=='active']
    gpu_ids=[uuid for lease in leases for uuid in lease.get('gpu_uuids', [])]
    return dict(schema=1,updated_at_s=now or time.time(), queue=queue_binding,
        seeds=[701],single_seed=True,profile_window_repeats=3,
        gpu_lease_occupancy=dict(active_groups=len(leases),gpu_count=len(gpu_ids),gpu_uuids=gpu_ids,
                                 mutual_exclusion=len(gpu_ids)==len(set(gpu_ids)), actual_compute_utilization='not_measured_by_this_cpu_status'),
        job_status_counts=dict(Counter(x['queue_status'] for x in jobs)),active_jobs=active,
        superseded_jobs=[dict(job_id=x['job_id'],superseded_by=x['superseded_by']) for x in jobs if x['superseded_by']],
        matrix=matrix(queue,jobs,registry or {}),formal_eligible=False,energy_comparable=False,
        limits=['GPU lease occupancy is not GPU compute utilization.',
                'Queue success, sampling completion, bounded calibration, functional smoke and formal acceptance are different gates.',
                'Manual transition/macro primitives do not prove full-duration automatic policies.',
                'No missing seed 1701/2701 gate is applied.'])


def markdown(value):
    lines=['# 实时状态（seed 701）','', '更新：'+datetime.fromtimestamp(value['updated_at_s'],ZoneInfo('Asia/Shanghai')).isoformat(),'',
           '活跃租约：%s 组 / %s GPU；这里统计占卡，不能据此声称 GPU 始终繁忙。' % (value['gpu_lease_occupancy']['active_groups'],value['gpu_lease_occupancy']['gpu_count']), '',
           ]
    sample = value.get('gpu_snapshot', {})
    if sample.get('rows') and not sample.get('error'):
        lines += ['GPU 瞬时利用率：'+', '.join('%s: %g%%' % (r['index'], r['utilization_percent'])
                    for r in sample['rows'])+'；这是单次采样，不代表整个窗口。', '']
    if value.get('first_batch_spec'):
        batch = value['first_batch_spec']
        lines += ['首批配置：%s 点；已冻结 %s/%s 条共同评价轨迹。配置就绪与正式通过分别验收，'
                  '本页未将配置计为正式完成。' % (batch['summary'].get('points', 0),
                    batch['summary'].get('frozen_trace_sets', 0), batch['summary'].get('trace_sets', 0)), '']
    lines += ['|模型|系统|功能请求链|局部机制证据|校准|正式|','|---|---|---|---|---|---|']
    for r in value['matrix']:
        lines.append('|%s|%s|%s|%s|%s|%s|'%(r['model_id'],r['system'],r['functional']['status'],r['mechanism']['status'],r['calibration']['status'],r['formal_status']))
    lines+=['','功能 passed 只覆盖对应 receipt 的 scope；完整自动策略、SLO 和整机总能耗比较尚未由该状态表认定通过。','','## 当前队列','']
    for j in value['active_jobs']:
        reason=', '.join(x['job_id']+' ('+x['status']+', '+x['required']+')' for x in j['blocked_by'])
        progress = j.get('sampling_progress')
        measured = ('；采样 '+', '.join('%s %s/%s' % (p['phase'], p['measured_points'], p['expected_points'])
                    for p in progress['phases'])) if progress else ''
        if progress and progress.get('collection_done') and j['queue_status']=='running':
            measured += '；采集结束，等待同波次收尾与校准审计'
        lines.append('- `%s`：%s%s%s'%(j['job_id'],j['scheduler_state'],measured,'；等待 '+reason if reason else ''))
    lines+=['','逐条 receipt SHA、GPU UUID、递归阻塞链、被替换任务和各系统剩余门槛见 [current.json](current.json)。',
            'Profiler 每点三次窗口重复仍保留；workload 只要求 seed 701。','']
    return '\n'.join(lines)


def refresh(args):
    data=args.queue.read_bytes(); queue=json.loads(data)
    registry=json.loads(args.registry.read_text()) if args.registry.is_file() else {}
    value=build(queue,dict(path=str(args.queue.resolve()),sha256=sha_bytes(data)),registry)
    value['gpu_snapshot'] = gpu_snapshot()
    campaign_path = getattr(args, 'campaign', None)
    if campaign_path and campaign_path.is_file():
        campaign_data = campaign_path.read_bytes()
        campaign = json.loads(campaign_data)
        value['first_batch_spec'] = dict(path=str(campaign_path.resolve()), sha256=sha_bytes(campaign_data),
            summary=campaign.get('summary', {}), scope='configuration_and_frozen_traces_only')
    incremental_path = ROOT/'results/2026-09-23/incremental-wave-closeout-v1/audit.json'
    if incremental_path.is_file():
        incremental = json.loads(incremental_path.read_text())
        value['incremental_calibration_audit'] = dict(path=str(incremental_path), sha256=sha_bytes(incremental_path.read_bytes()))
        for row in value['matrix']:
            components = [dict(member=name, qualification_status=component['qualification_status'],
                               tp=component.get('tp'), domain=component.get('domain'),
                               version_creation_ready=component.get('version_creation_ready', False),
                               calibration_components_passed=component.get('calibration_components_passed', False))
                          for name, component in incremental.get('members', {}).items()
                          if component.get('model_id') == row['model_id'] and component.get('system') == row['system']]
            row['calibration']['incremental_components'] = components
            if any(c['calibration_components_passed'] for c in components):
                row['calibration']['status'] = 'bounded_components_passed'
    if registry:
        value['calibration_registry']=dict(path=str(args.registry.resolve()),sha256=sha_bytes(args.registry.read_bytes()))
    args.out.mkdir(parents=True,exist_ok=True)
    for name,text in [('current.json',json.dumps(value,indent=2,allow_nan=False)+'\n'),('current.md',markdown(value))]:
        temp=args.out/('.'+name+'.'+str(os.getpid())+'.tmp');temp.write_text(text);temp.replace(args.out/name)
    print(json.dumps(dict(output=str(args.out.resolve()),active_jobs=len(value['active_jobs']),matrix_rows=len(value['matrix']))), flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,default=ROOT/'results/2026-09-22/three-model/queue.json')
    parser.add_argument('--registry',type=Path,default=ROOT/'results/2026-09-23/calibration-versions-v1/registry.json')
    parser.add_argument('--out',type=Path,default=ROOT/'results/2026-09-23/status')
    parser.add_argument('--campaign',type=Path,default=ROOT/'results/2026-09-23/first-five-system-batch-v2/spec.json')
    parser.add_argument('--watch',action='store_true',help='refresh until this queue has no running/queued jobs')
    parser.add_argument('--interval',type=float,default=30)
    args=parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error('--interval must be between 1 and 60 seconds')
    while True:
        value=refresh(args)
        if not args.watch or not value['active_jobs']:
            return
        time.sleep(args.interval)


if __name__=='__main__':
    main()
