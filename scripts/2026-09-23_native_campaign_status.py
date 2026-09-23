#!/usr/bin/env python3
"""Summarize real receipts for the five-system GPU smoke, without ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def read_receipt(path):
    if not path.is_file():
        return None,None
    data=path.read_bytes()
    return json.loads(data),dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'results/2026-09-23/five-system-gpu-status.json')
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text())
    queue=json.loads((ROOT/'results/2026-09-22/three-model/queue.json').read_text())
    models={}
    for entry in spec['jobs']:
        name=entry['job_id']
        if not name.startswith('native-'):
            continue
        job=queue['jobs'].get(name,{})
        model=entry['payload']['model_id']
        phase=entry['payload']['kind']
        leases=[x for x in queue['leases'].values() if x['job_id']==name]
        lease=max(leases,key=lambda x:x['claimed_at']) if leases else None
        stage=dict(job_id=name,status=job.get('status','not_enqueued'),
                   gpu_count=entry['payload']['gpu_count'],gpu_indices=lease['gpu_indices'] if lease else [],
                   source_sha256=entry['payload']['source_sha256'],
                   receipts={},completion=None)
        blocked=[dep for dep in entry['payload'].get('depends_on',[]) if
                 queue['jobs'].get(dep,{}).get('status') in ('blocked','failed')]
        if blocked and stage['status']=='queued':
            stage.update(status='blocked_dependency',blocked_by=blocked)
        if lease:
            folder=Path(lease['attempt_dir'])
            # The lease worker owns attempt/events.jsonl. Dynamo's Journal
            # writes to its own subdirectory and its required receipt records
            # that location explicitly in the immutable queue payload.
            required=entry['payload'].get('required_receipts', [])
            if required:
                folder=folder/Path(required[0]).parent
            stage['artifact_root']=str(folder)
            for filename in ('completion.json','native-primitives.json','public-mixed-pd.json',
                             'distserve-scheduler.json','mechanism.json'):
                value,binding=read_receipt(folder/filename)
                if value is not None:
                    stage['receipts'][filename]=dict(binding,status=value.get('status'),complete=value.get('complete'))
                    if filename=='completion.json':
                        stage['completion']={key:value.get(key) for key in
                            ('status','complete','error','missing_required_actions','independent_profile_collection',
                             'complete_reproduction','controller_hierarchy_qualified')}
        models.setdefault(model,dict(stages={},systems={} ))['stages'][phase]=stage
    for model,row in models.items():
        stages=row['stages']
        probe=stages.get('probe',{})
        receipts=probe.get('receipts',{})
        paths={'mixed':receipts.get('public-mixed-pd.json'),
               'pdblend':receipts.get('public-mixed-pd.json'),
               'distserve':(stages.get('distserve_pipeline',{}).get('receipts',{}).get('completion.json')
                            if 'distserve_pipeline' in stages else receipts.get('distserve-scheduler.json')),
               'dynamollm':stages.get('dynamo',{}).get('receipts',{}).get('completion.json'),
               'ecoserve':stages.get('eco4',{}).get('receipts',{}).get('completion.json')}
        for system,receipt in paths.items():
            phase={'dynamollm':'dynamo','ecoserve':'eco4'}.get(system,'probe')
            if system=='distserve' and 'distserve_pipeline' in stages:
                phase='distserve_pipeline'
            phase_status=stages.get(phase,{}).get('status','not_enqueued')
            absent_status=(phase_status if phase_status in ('failed','blocked','blocked_dependency') else 'pending_gpu')
            passed = (phase_status == 'succeeded' and receipt is not None
                      and receipt['complete'] is True and receipt['status'] == 'passed')
            receipt_status = receipt['status'] if receipt else absent_status
            if receipt_status == 'passed' and not passed:
                receipt_status = 'incomplete_or_unverified'
            row['systems'][system]=dict(status='passed' if passed else receipt_status,receipt=receipt,
                scope='functional_protocol' if system in ('mixed','pdblend') else 'native_mechanism_primitives',
                formal_eligible=False,energy_comparable=False,
                complete_reproduction=False)
            if system=='distserve':
                row['systems'][system]['scheduler_primitive_receipt']=receipts.get('distserve-scheduler.json')
                if phase=='distserve_pipeline':
                    row['systems'][system]['scope']='independent_native_request_pipeline'
            if system=='ecoserve' and stages.get('eco4',{}).get('source_sha256','').startswith('3c2c90d1c630'):
                row['systems'][system]['native_time_semantics']='response_completed_not_scheduler_snapshot'
                row['systems'][system]['strict_snapshot_token_alignment_qualified']=False
    result=dict(updated_at_s=time.time(),spec=str(args.spec.resolve()),spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                source_sha256=spec['source_sha256'],models=models,seed=701,single_seed=True,
                formal_eligible=False,energy_comparable=False,
                smoke_complete=len(models)==3 and all(x['status']=='passed' for m in models.values() for x in m['systems'].values()),
                limits=['Functional diagnostics are not an energy comparison.',
                        'Manual primitive receipts do not qualify the complete original controller hierarchy.',
                        'Independent profiles, calibration, automatic policy actions and full SLO matrix remain separate gates.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{s:v['status'] for s,v in r['systems'].items()} for m,r in models.items()},indent=2))


if __name__=='__main__':main()
