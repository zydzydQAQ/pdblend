"""Explicit, sequential continuation of the original node campaign and its authorized budget.

``template``, ``prepare-smoke`` and ``prepare-return`` only write CPU artifacts.
``execute`` acquires the existing Campaign lease and never creates a new budget.
Future stages remain conditional intentions, not passed evidence or reservations.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .campaign_followup_setup import read, write, initial_layout
from .budget import read_budget


MODULE='ecopadg.serving.campaign_pipeline'


def template(root,out):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('refusing to overwrite pipeline template')
    value=dict(status='conditional_template_not_executed',campaign_root=str(root),
        active_campaign=str(root/'validation-instant-cost-resume.campaign.json'),
        initial_layout_evidence=str(root/'validate-physical-transitions/raw.json'),
        legacy_search_completion=str(root/'configuration-search.json'),
        post_campaign=str(root/'post-profile-validation.campaign.json'),
        eco_result=str(root/'eco-validation-current/raw.json'),
        smoke_out=str(root/'controller-smoke-v2'),
        admission_out=str(root/'kv-admission-v2'),
        dynamo_campaign=str(root/'dynamo-revalidation-shard.pending.campaign.json'),
        dynamo_setup=str(root/'dynamo-revalidation-shard-v2'),
        followup=str(root/'followup-templates/followup.json'),
        mechanism_collection_out=str(root/'mechanism-evidence-v2'),
        mechanisms_out=str(root/'mechanisms.certified-v2.json'),
        return_manifest=str(out.parent/'after-dynamo.restore.json'),
        return_out=str(root/'after-dynamo-restoration'),return_limit_s=600,
        note='Preserve original elapsed time and use the verified budget ledger; each stage is capped by remaining time. Calibration and paired methods retain their explicit allocation gates; formal work is separate.')
    write(out,value);return value


def existing_budget(root):
    value=read_budget(root)
    if (not isinstance(value.get('started_s'),(int,float)) or not math.isfinite(value['started_s'])
            or value['started_s']<=0):
        raise ValueError('an already-started original campaign budget is required')
    if value['remaining_s']<=60:
        raise ValueError('effective authorized campaign deadline exhausted')
    return value


def gate(path,*,keys=('passed',)):
    value=read(path)
    if any(value.get(k) is not True for k in keys):
        raise RuntimeError('required evidence did not pass: '+str(path))
    return value


def queue_completion(plan,budget):
    """Called only after acquiring Campaign's exclusive lease, without nesting."""
    from .evidence import freeze_files
    active=read(plan['active_campaign'])
    last=active['stages'][-1]['name']
    if (budget.get('stage')!=last or budget.get('last_finished_s',0)<budget.get('last_started_s',float('inf'))
            or type(budget.get('last_exit_code')) is not int or budget['last_exit_code']!=0
            or budget.get('last_error') is not None or budget.get('last_interrupted') is not False):
        raise ValueError('preceding queue has not recorded its last completed stage')
    receipt_path=Path(plan['active_campaign']).with_suffix('.execution-result.json')
    receipt=read(receipt_path)
    from .evidence import sha256
    if (type(receipt.get('exit_code')) is not int or receipt['exit_code']!=0
            or receipt.get('complete') is not True or receipt.get('error') is not None
            or receipt.get('manifest_sha256')!=sha256(plan['active_campaign'])):
        raise ValueError('preceding campaign has no successful unchanged execution receipt')
    prior=initial_layout(dict(initial_layout_evidence=plan['initial_layout_evidence']))
    # The old search is only a completion checkpoint, never a calibrated input.
    read(plan['legacy_search_completion'])
    return dict(complete=True,checkpoint_only=True,formal_eligible=False,initial_instances=prior,
        original_started_s=budget['started_s'],original_limit_s=budget.get('original_limit_s',budget['limit_s']),
        effective_limit_s=budget['limit_s'],budget_revision_seq=budget.get('revision_seq',0),
        artifacts=freeze_files([plan['active_campaign'],plan['initial_layout_evidence'],
            plan['legacy_search_completion'],str(receipt_path)]))


def prepare_smoke(plan):
    from .controller_smoke import generate
    followup=read(plan['followup']);cal=read(followup['calibration_template'])
    initial=initial_layout(dict(initial_layout_evidence=plan['initial_layout_evidence']))
    # Keep original IDs/ports so Dynamo's explicit original layout accepts this
    # four-replica subset. No broadly shared ownership root is necessary.
    choices=sorted((i for i in initial if i['tp']==1),key=lambda i:min(i['gpus']))[:4]
    if len(choices)!=4:raise ValueError('four proven TP1 replicas required for smoke')
    choices=[dict(i,role='mixed') for i in choices]
    out=Path(plan['smoke_out']).resolve()
    manifest=dict(campaign_root=plan['campaign_root'],image=cal['image'],instances=choices,
        initial_instances=initial,profiles=cal['profiles'],transfers=cal['transfers'],interconnect=cal['interconnect'],
        frequency_costs=cal['frequency_costs'],role_costs=str(Path(plan['campaign_root'])/'role-profiling-tp1-instant/role_costs.json'),
        engine_template=cal['engine_template'],retained_weights=cal['retained_weights'],
        runtime_dir=str(out/'runtime'),ownership_root=str(out),limit_s=900)
    return generate(manifest,out)


def prepare_return(plan):
    """Authorize only the proven prior layout plus this Dynamo setup's engines."""
    from .calibration_setup import spec, physical, verify_artifacts
    from .evidence import freeze_files
    from .topology import validate_layout
    followup=read(plan['followup']);cal=read(followup['calibration_template'])
    setup=Path(plan['dynamo_setup']).resolve();campaign_root=Path(plan['campaign_root']).resolve()
    if setup==campaign_root or not setup.is_relative_to(campaign_root):
        raise ValueError('Dynamo ownership root must be a dedicated campaign child directory')
    old=initial_layout(dict(initial_layout_evidence=plan['initial_layout_evidence']))
    restoration=read(setup/'restoration.json')
    evidence=read(setup/'input-evidence.json')['artifacts'];verify_artifacts(evidence)
    if str((setup/'restoration.json').resolve()) not in evidence:
        raise ValueError('Dynamo restoration must be fingerprinted by its generated setup')
    if (restoration['image']!=cal['image'] or restoration.get('retained_weights')!=cal.get('retained_weights')
            or Path(restoration['ownership_root']).resolve()!=setup
            or not Path(restoration['engine_template']).resolve().is_relative_to(setup)):
        raise ValueError('Dynamo image, cache, template or ownership root changed')
    desired=[spec(i) for i in restoration['instances']];validate_layout(desired,range(8))
    previous=[spec(i) for i in restoration['initial_instances']]
    allowed={physical(spec(i)) for i in old}|{physical(i) for i in desired}
    if {physical(i) for i in previous}!=allowed:
        raise ValueError('Dynamo setup authorizes unexpected prior engines')
    result=dict(instances=old,initial_instances=old+[i.endpoint() for i in desired],
        image=cal['image'],engine_template=restoration['engine_template'],
        retained_weights=cal.get('retained_weights'),ownership_root=str(setup))
    target=Path(plan['return_manifest'])
    if target.exists():raise ValueError('refusing to overwrite after-Dynamo restoration')
    write(target,result)
    write(target.with_suffix('.evidence.json'),dict(status='prepared_not_executed',
        purpose='restore only the validated original layout; inter-experiment preparation, not serving energy',
        artifacts=freeze_files([str(p) for p in (plan['initial_layout_evidence'],setup/'restoration.json',
            setup/'input-evidence.json',target)])))
    return result


def stage_list(path,root):
    value=read(path)
    effective=read_budget(root)['limit_s'] if (Path(root)/'budget.json').is_file() else 86400
    declared=value.get('budget_s',86400)
    if (Path(value['output']).resolve()!=Path(root).resolve() or type(declared) not in (int,float)
            or not math.isfinite(declared) or not 0<declared<=effective):
        raise ValueError('all stages must share the original authorized campaign')
    names=set()
    for stage in value['stages']:
        if (stage['name'] in names or not set(stage.get('requires',()))<=names
                or not math.isfinite(stage['limit_s']) or stage['limit_s']<=0
                or stage.get('formal')):
            raise ValueError('invalid, unordered, or formal stage in development continuation')
        names.add(stage['name'])
    return value['stages']


def execute(plan,path):
    from .campaign import Campaign
    from . import calibration_setup,method_selection
    existing_budget(plan['campaign_root'])
    campaign=Campaign(plan['campaign_root'],read_budget(plan['campaign_root'])['limit_s'])
    status=dict(complete=False,formal_eligible=False,started_s=time.time(),segments=[])
    def one(stage):
        campaign.run(stage['name'],stage['argv'],stage['limit_s'],gpu=stage.get('gpu',True))
    def segment(name,stages):
        # These are maximum timeouts, not elapsed measurements or prebooked
        # capacity. Campaign caps every child to the unchanged absolute deadline.
        record=dict(name=name,stage_upper_bound_s=sum(s['limit_s'] for s in stages),
            remaining_before_s=campaign.remaining_s,complete=False)
        status['segments'].append(record)
        for stage in stages:one(stage)
        record['complete']=True
    def cpu(name,action,limit=600):
        one(dict(name=name,gpu=False,limit_s=limit,
            argv=[sys.executable,'-m',MODULE,action,'--manifest',str(path)]))
    try:
        checkpoint=queue_completion(plan,campaign.state)
        checkpoint_path=Path(path).with_suffix('.queue-completion.json')
        if checkpoint_path.exists():raise ValueError('pipeline already started; do not replay completed stages')
        write(checkpoint_path,checkpoint)
        segment('instant-profiles-and-eco',stage_list(plan['post_campaign'],plan['campaign_root']))
        gate(plan['eco_result'],keys=('complete','passed'))
        followup=read(plan['followup'])
        calibration_setup.validated_inputs(read(followup['calibration_template']))
        cpu('prepare-controller-smoke','prepare-smoke')
        segment('controller-smoke',stage_list(Path(plan['smoke_out'])/'campaign.json',plan['campaign_root']))
        smoke=gate(Path(plan['smoke_out'])/'run/summary.json')
        if smoke.get('status')!='controller_smoke_passed':raise RuntimeError('controller smoke has no successful correctness status')
        one(dict(name='validate-real-kv-admission',gpu=True,limit_s=240,argv=[sys.executable,'-m',
            'ecopadg.serving.admission_validation','--manifest',str(path),'--out',plan['admission_out']]))
        gate(Path(plan['admission_out'])/'raw.json',keys=('complete','passed'))
        dynamo=stage_list(plan['dynamo_campaign'],plan['campaign_root'])
        if len(dynamo)!=3 or dynamo[0].get('gpu',True):raise ValueError('expected CPU setup then two Dynamo hardware stages')
        segment('dynamo-cpu-setup',dynamo[:1])
        cpu('prepare-after-dynamo-restoration','prepare-return')
        try:
            segment('dynamo-real-periods',dynamo[1:])
        finally:
            calibration_setup.verify_artifacts(read(Path(plan['return_manifest']).with_suffix('.evidence.json'))['artifacts'])
            one(dict(name='restore-proven-layout-after-dynamo',limit_s=plan['return_limit_s'],
                argv=[sys.executable,'-m','ecopadg.serving.calibration_setup','restore',
                    '--manifest',plan['return_manifest'],'--out',plan['return_out']]))
        # Dynamo's CLI can finish valid cycles with no physical actions. That
        # diagnostic is preserved, but cannot open dependent baseline work.
        gate(Path(plan['dynamo_setup'])/'run/mechanisms.json')
        one(dict(name='prepare-followup-calibration',gpu=False,limit_s=600,argv=[sys.executable,'-m',
            'ecopadg.serving.campaign_followup_setup','prepare-calibration','--manifest',plan['followup']]))
        segment('independent-baseline-calibration',stage_list(Path(followup['calibration_out'])/'campaign.json',plan['campaign_root']))
        gate(Path(followup['calibration_out'])/'summary.json')
        one(dict(name='collect-baseline-mechanism-evidence',gpu=False,limit_s=600,argv=[sys.executable,'-m',
            'ecopadg.serving.mechanism_evidence','--manifest',str(path),'--out',plan['mechanism_collection_out']]))
        collected=gate(Path(plan['mechanism_collection_out'])/'summary.json',keys=('complete',))
        status.update(baseline_mechanisms_complete=collected['baseline_mechanisms_complete'],
            missing_baseline_mechanisms=collected['missing'])
        one(dict(name='prepare-followup-methods',gpu=False,limit_s=600,argv=[sys.executable,'-m',
            'ecopadg.serving.campaign_followup_setup','prepare-methods','--manifest',plan['followup']]))
        segment('complete-paired-development',stage_list(Path(followup['method_out'])/'paired/campaign.json',plan['campaign_root']))
        selection=method_selection.summarize(Path(followup['method_out'])/'paired/prepared.json')
        if selection.get('selected_variant') is None:raise RuntimeError('no method passed development selection')
        status.update(complete=True,development_selection=selection['selected_variant'])
    except BaseException as exc:
        status['error']=repr(exc);raise
    finally:
        status['finished_s']=time.time();write(Path(path).with_suffix('.status.json'),status);campaign.close()
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('template','prepare-smoke','prepare-return','execute'))
    parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.action=='template':
        if not args.root or not args.out:parser.error('template requires --root and --out')
        result=template(args.root,args.out)
    else:
        if not args.manifest:parser.error('action requires --manifest')
        plan=read(args.manifest)
        result=(prepare_smoke(plan) if args.action=='prepare-smoke' else
            prepare_return(plan) if args.action=='prepare-return' else execute(plan,args.manifest.resolve()))
    print(json.dumps({k:v for k,v in result.items() if k in ('status','complete','formal_eligible')},allow_nan=False))


if __name__=='__main__':main()
