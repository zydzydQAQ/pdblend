#!/usr/bin/env python3
"""Read-only closeout of incremental measurements, qualification and domains.

The current snapshot may be refreshed while jobs run. Final immutable reports
are written only after all four jobs are terminal and their leases are released.
No collector, fit using holdout data, GPU command or queue mutation is invoked.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
MEMBERS=('14b-tp4-power-mixed','32b-tp2-longctx','7b-tp1-longctx','14b-tp1-longctx')
QUEUE=ROOT/'results/2026-09-22/three-model/queue.json'
COHORT='incremental-profile-4-2-1-1-b941f027dc7ae9b609a0'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def check(path,digest):
    if not digest or sha(path)!=digest:raise ValueError('checksum mismatch: '+str(path))


def write(path,value,immutable=False):
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if immutable and path.exists():
        if path.read_text()!=text:raise ValueError('immutable closeout differs: '+str(path))
        return
    temp=path.with_name('.'+path.name+'.tmp');temp.write_text(text);temp.replace(path)


def choose(queue,cohort):
    result={}
    for jid,job in queue['jobs'].items():
        p=job.get('payload',{});member=p.get('cohort_member')
        if p.get('cohort_id')!=cohort or member not in MEMBERS or job.get('superseded_by'):continue
        if member not in result or job.get('created_at',0)>result[member][1].get('created_at',0):result[member]=(jid,job)
    return result


def released(queue,jid,job):
    return job['status'] in ('succeeded','failed','cancelled') and job.get('lease_id') is None and not any(
        l.get('job_id')==jid and l.get('status')=='active' for l in queue.get('leases',{}).values())


def progress(root):
    def count(relative):
        path=root/relative
        return len(read(path).get('decode',[])) if path.is_file() else 0
    return dict(primary_decode=count('raw.json'),fresh_holdout=count('long-holdout/raw.json'),
                mixed=len(read(root/'mixed-repair/raw.json').get('mixed',[])) if (root/'mixed-repair/raw.json').is_file() else 0)


def _audit(entry):
    from pdblend.profile import power_calibration as pc, local_power as lp, local_mixed_repair as mr
    from pdblend.profile import long_context_collect as lc, long_context_followup as lf
    from pdblend.profile.local_power_job import queue_receipt
    from pdblend.profile.identity import sha256_value
    from pdblend.profile.calibration import evaluate_holdout
    root=Path(entry['root']);scope=entry['payload']['measurement_scope'];kind=scope['kind']
    source=Path(entry['payload']['source_snapshot']);source_manifest=read(source/'manifest.json')
    if not Path(pc.__file__).resolve().is_relative_to(source.resolve()):raise ValueError('wrong audit implementation')
    if source_manifest['source_sha256']!=entry['payload']['source_sha256']:raise ValueError('source identity mismatch')
    for name,digest in source_manifest['files'].items():check(source/name,digest)
    raw=read(root/'raw.json');completion=read(root/'completion.json');check(root/'raw.json',completion['raw_sha256'])
    for key in ('model_id','tp','pp','system'):
        if raw[key]!=entry['payload'][key]:raise ValueError('raw model/topology mismatch: '+key)
    if (raw['environment']['source_hash']!=entry['payload']['source_sha256'] or
            sorted(raw['environment']['gpu_uuids'])!=sorted(entry['gpu_uuids'])):raise ValueError('raw source/lease UUID mismatch')
    files={name:bound(root/name) for name in ('raw.json','completion.json')}
    result=dict(measurement_complete=completion.get('complete') is True,
        qualification_status='not_evaluated',calibration_components_passed=False,version_creation_ready=False,
        system=raw['system'],model_id=raw['model_id'],tp=raw['tp'],pp=raw['pp'],kind=kind,
        evidence=files,formal_eligible=False,energy_comparable=False,full_profile_qualified=False,
        missing_gates=completion.get('missing_gates',[]),environment=raw['environment'])
    if kind=='local_power':
        package=Path(scope['package']);manifest,plan,model=lp.load_package(package)
        check(root/'composite-audit.json',completion['composite_audit_sha256'])
        if queue_receipt(completion,root)!=read(root/'queue-completion.json'):raise ValueError('queue adapter receipt differs')
        power=pc.audit_power(raw,root,plan['points'],model,raw['power_holdout_binding'],windows_per_frequency=6)
        if power!=read(root/'power-only-audit.json'):raise ValueError('power audit differs from raw reconstruction')
        mr.checked_completed(package=package,out=root/'mixed-repair',manifest=manifest)
        saved=read(root/'mixed-repair/repaired-timing-audit.json')
        check(root/'mixed-repair/combined-timing-view.json',saved['combined_view_sha256'])
        check(root/'mixed-repair/completion.json',saved['repair_completion_sha256'])
        original,roots,_=mr.original_view(manifest);roots['mixed-repair']=root/'mixed-repair'
        view=read(root/'mixed-repair/combined-timing-view.json')
        frozen=read(Path(manifest['original_holdout'])/'frozen-fit.json')
        timing=pc.timing_component(evaluate_holdout(view,pc.PerfModel.load(manifest['inputs']['base_candidate']['path']),
            root,expected_plan=frozen['plan'],evidence_roots=roots))
        for key,value in timing.items():
            if saved[key]!=value:raise ValueError('repaired timing differs from actual inherited/raw samples: '+key)
        passed=power['passed'] and timing['passed']
        result.update(qualification_status='passed' if passed else 'failed',calibration_components_passed=passed,
            power=dict(points=len(raw['decode']),windows=len(power['points']),by_frequency=power['by_frequency'],
                       maximum=max(x['max_error'] for x in power['by_frequency'].values()),failures=power['failures']),
            timing=dict(passed=timing['passed'],maximum=timing['timing_max'],mixed_median=timing['mixed_timing_median'],
                        failures=timing['failures'],fresh_mixed_points=4,retained_mixed_points=8,original_failed_receipt_unchanged=True),
            domain=dict(validated_power_batches=[1,128],frequencies=list(model.freqs),
                        unvalidated_other_power_shapes=True,timing_domain_unchanged=True),
            consumer_requirements=['new_explicit_component_version_descriptor','local_12_point_power_scope_not_full_grid'],
            missing_gates=['other_power_shapes_if_required','full_profile_provenance_revalidation','native_mechanisms','campaign_acceptance'])
        files.update({name:bound(root/name) for name in ('composite-audit.json','queue-completion.json','power-only-audit.json',
            'mixed-repair/completion.json','mixed-repair/raw.json','mixed-repair/repaired-timing-audit.json','mixed-repair/combined-timing-view.json')})
        files['package_manifest']=bound(package/'manifest.json')
    else:
        plan=read(root/'training-plan.json')
        lf.verify_training(raw,root,plan['training'],sha(root/'training-plan.json'))
        result['training']=dict(verified_points=len(raw['decode']),fit_uses_holdout=False)
        if kind=='training':
            if completion.get('candidate_fit_performed') or completion.get('fit_performed') or completion.get('holdout_points_consumed'):
                raise ValueError('training-only job claims fitted/holdout evidence')
            result.update(qualification_status='training_only',domain=dict(exact_batches=scope['exact_batches'],
                batch_interpolation_qualified=False),consumer_requirements=['training_only_candidate_fit','fresh_independent_holdout'])
            # Diagnostic budget derived only from the now-verified training
            # archive. No candidate is published and no holdout is fabricated.
            proposal=dict(schema=1,kind=lf.KIND,**lf.identity(raw),exact_batches=scope['exact_batches'],nodes={})
            for f in lf.FREQUENCIES:
                for b in proposal['exact_batches']:
                    group=sorted((r for r in raw['decode'] if (r['freq_mhz'],r['batch'])==(f,b)),key=lambda r:r['context_tokens'])
                    if [r['context_tokens'] for r in group]!=[5120,7168,7680]:raise ValueError('three training-only anchors required')
                    proposal['nodes'][f'{f}/{b}']=[dict(context=statistics.fmean(p['effective_context_tokens'] for p in r['repeats']),
                        step_seconds=statistics.fmean(p['step_seconds'] for p in r['repeats']),
                        power_w=statistics.fmean(p['power_w'] for p in r['repeats'])) for r in group]
            next_plan=lf.holdout_plan(proposal,dict(raw,decode=[]),raw,raw['kv_capacity_tokens'])
            result['minimum_followup']=dict(expected_points=next_plan['expected_points'],feasible_points=len(next_plan['points']),
                missing_points=next_plan['missing_points'],repeats_per_point=3,
                settle_measure_lower_bound_s=next_plan['expected_points']*3*7,
                exact_batches=proposal['exact_batches'],roles=['interior6144','campaign_endpoint7679'],
                output_budget_range=[min(p['max_tokens'] for p in next_plan['points']),max(p['max_tokens'] for p in next_plan['points'])],
                own_gpu_count=1,compatible_with_dynamo_32b_four_gpu_budget=True,
                requires_new_concurrency_qualification=True,old_training_reused=True,
                reason_training_cannot_replace_holdout='All 36 points are candidate training anchors; no independent long-domain test exists.',
                required_implementation='14B exact-batch training-candidate and holdout-only adapter; current fit whitelist is 7B/32B',
                plan_is_diagnostic_not_executed=True)
        else:
            package=Path(scope['package']);manifest,endpoint_plan,prior=lf.load_package(package)
            frozen=root/'long-candidate';candidate=read(frozen/'candidate.json');fmanifest=read(frozen/'manifest.json')
            for name,key in [('candidate.json','candidate_sha256'),('holdout-plan.json','plan_sha256'),('endpoint-training-archive.json','endpoint_archive_sha256')]:check(frozen/name,fmanifest[key])
            endpoint=read(frozen/'endpoint-training-archive.json')
            lf.verify_training(endpoint,root,endpoint_plan['training'],manifest['endpoint_plan_sha256'])
            reconstructed=lf.fit_candidate(prior,endpoint)  # Training-only deterministic reconstruction.
            reconstructed['training_binding']=dict(package_manifest_sha256=sha(package/'manifest.json'),
                prior_raw_sha256=manifest['inputs']['raw.json']['sha256'],endpoint_rows_sha256=sha256_value(endpoint['decode']))
            if reconstructed!=candidate:raise ValueError('frozen candidate differs from exclusively training-derived fit')
            hroot=root/'long-holdout';hraw=read(hroot/'raw.json');hcompletion=read(hroot/'completion.json')
            check(hroot/'raw.json',hcompletion['raw_sha256']);check(hroot/'holdout-audit.json',hcompletion['audit_sha256'])
            audit=lf.audit(candidate,hraw,hroot,read(frozen/'holdout-plan.json'))
            if audit!=read(hroot/'holdout-audit.json'):raise ValueError('long holdout audit differs from raw reconstruction')
            if hcompletion['calibration_passed']!=audit['passed'] or completion['extended_calibration_passed']!=audit['passed']:
                raise ValueError('completion qualification differs from actual holdout')
            result.update(qualification_status='passed' if audit['passed'] else 'failed',
                calibration_components_passed=audit['passed'],holdout=dict(complete=audit['complete'],
                    verified_points=len(hraw['decode']),errors=audit['errors'],failures=audit['failures']),
                domain=dict(exact_batches=candidate['exact_batches'],batch_interpolation_qualified=False,
                    context_intervals={key:[nodes[0]['context'],nodes[-1]['context']] for key,nodes in candidate['nodes'].items()},
                    target_context=7679,whole_window_checked=True,short_profile_unchanged=True),
                consumer_requirements=['explicit_exact_batch_long_candidate_loader','never_interpolate_B1_to_B4_or_B4_to_B8'])
            files.update({str(p.relative_to(root)):bound(p) for p in (frozen/'candidate.json',frozen/'holdout-plan.json',frozen/'manifest.json',
                frozen/'endpoint-training-archive.json',hroot/'raw.json',hroot/'completion.json',hroot/'holdout-audit.json')})
            files['package_manifest']=bound(package/'manifest.json')
    result['version_creation_ready']=result['measurement_complete'] and result['calibration_components_passed'] and entry['queue_status']=='succeeded'
    return result


def audit_entry(entry):
    env=dict(os.environ,PYTHONPATH=entry['payload']['source_snapshot'],PYTHONDONTWRITEBYTECODE='1')
    run=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--audit-entry'],input=json.dumps(entry),
                       text=True,capture_output=True,env=env)
    if run.returncode:raise ValueError(run.stderr[-6000:])
    return json.loads(run.stdout)


def build(queue,cohort):
    selected=choose(queue,cohort);members={};all_released=len(selected)==len(MEMBERS)
    for member in MEMBERS:
        if member not in selected:
            members[member]=dict(status='missing_job',measurement_complete=False,qualification_status='inconclusive');continue
        jid,job=selected[member];ready=released(queue,jid,job);all_released&=ready
        leases=[x for x in queue['leases'].values() if x['job_id']==jid]
        lease=max(leases,key=lambda x:x['claimed_at']) if leases else None
        value=dict(job_id=jid,queue_status=job['status'],released=ready,measurement_complete=False,
                   qualification_status='pending_terminal',error=job.get('last_error'))
        if lease:
            root=Path(lease['attempt_dir']);value.update(root=str(root),progress=progress(root),gpu_uuids=lease['gpu_uuids'])
            if ready:
                try:value.update(audit_entry(dict(root=str(root),payload=job['payload'],gpu_uuids=lease['gpu_uuids'],queue_status=job['status'])))
                except (ValueError,OSError,KeyError) as exc:value.update(qualification_status='audit_error',audit_error=str(exc),version_creation_ready=False)
        members[member]=value
    return dict(schema=1,cohort_id=cohort,all_jobs_released=all_released,members=members,
                seeds=[701],formal_eligible=False,energy_comparable=False,source=bound(__file__))


def markdown(value):
    lines=['# 增量测量收尾审计','',f"全部任务终态且租约已释放：{value['all_jobs_released']}",'',
        '|成员|任务|采样完整|校准资格|可准备新局部版本|','|---|---|---|---|---|']
    for member,row in value['members'].items():
        lines.append('|%s|%s|%s|%s|%s|'%(member,row.get('queue_status','missing'),row['measurement_complete'],row['qualification_status'],row.get('version_creation_ready',False)))
        if row.get('audit_error'):lines.append('\n审计错误：`'+member+'`：'+row['audit_error'].splitlines()[-1]+'\n')
    lines+=['','所有 error、逐频率最大误差、时延/功率 MAPE、mixed median、精确 batch/context 域和 raw SHA 在 audit.json/current.json。',
        'training_only 只完成训练样本收集；measurement_complete 不代表 calibration passed。',
        '局部候选不得扩大 batch 或 context 域；旧失败记录和现有 registry 不改，完整机制/SLO/整机能耗门槛保持独立。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',type=Path,default=QUEUE)
    parser.add_argument('--cohort-id',default=COHORT)
    parser.add_argument('--out',type=Path,default=ROOT/'results/2026-09-23/incremental-wave-closeout-v1')
    parser.add_argument('--audit-entry',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    if args.audit_entry:print(json.dumps(_audit(json.load(sys.stdin)),allow_nan=False));return
    data=args.queue.read_bytes();value=build(json.loads(data),args.cohort_id)
    value['queue_snapshot_sha256']=hashlib.sha256(data).hexdigest()
    write(args.out/'current.json',dict(value,updated_at_s=time.time()))
    (args.out/'current.md').write_text(markdown(value))
    if value['all_jobs_released']:
        # Queue heartbeats unrelated to this cohort must not change an immutable audit.
        final={k:v for k,v in value.items() if k!='queue_snapshot_sha256'}
        write(args.out/'audit.json',final,immutable=True)
        report=args.out/'report.md';text=markdown(final)
        if report.exists() and report.read_text()!=text:raise ValueError('immutable report differs')
        report.write_text(text)
    print(json.dumps(dict(all_jobs_released=value['all_jobs_released'],members={k:v['qualification_status'] for k,v in value['members'].items()})))


if __name__=='__main__':main()
