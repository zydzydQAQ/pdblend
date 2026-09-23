"""Exact-batch long-context endpoint training, freeze, and fresh resident holdout.

The original short-context PerfModel is never modified.  This separately bound
candidate rejects unmeasured batches and every context outside its training
means, even when a request's reserved max_tokens would be memory-feasible.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import statistics
import time
from pathlib import Path

from pdblend.profile.collection import long_context_collect as lc
from pdblend.profile.identity import sha256_value
from pdblend.profile.long_context_plan import FREQUENCIES
from pdblend.profile.calibration.power_calibration import write_immutable
from pdblend.profile.collection.wave import atomic_json

KIND='exact_batch_piecewise_context_training_means_v1'
MODELS={'Qwen2.5-7B-Instruct':(1,(1,4,8)), 'Qwen2.5-32B-Instruct':(2,(1,4))}
TARGET=7679


def source_hashes():
    from pdblend.source_inventory import implementation_hashes as inventory_hashes
    return inventory_hashes()


def check_archive(raw):
    if raw.get('identity_sha256')!=sha256_value({k:v for k,v in raw.items() if k!='identity_sha256'}):
        raise ValueError('long-context archive checksum mismatch')


def identity(raw):return {k:raw[k] for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp')}


def verify_training(raw,root,points,plan_sha):
    check_archive(raw)
    if raw.get('holdout_independent') is not False or raw.get('system')!='pdblend':
        raise ValueError('long candidate may only consume own training observations')
    expected={lc.point_key(p):p for p in points}
    if len(raw['decode'])!=len(expected) or {lc.point_key(r) for r in raw['decode']}!=set(expected):
        raise ValueError('long training matrix is incomplete or duplicate')
    for row in raw['decode']:
        point=expected[lc.point_key(row)]
        if row.get('evidence_class')!='training_extension' or row.get('independent_holdout') is not False or len(row['repeats'])!=3:
            raise ValueError('holdout/partial row cannot train long candidate')
        lc.validate_repeats(root,row['repeats'],point,plan_sha,complete=True)
        for rep in row['repeats']:
            if 'long_training_qualification_history' in raw:
                qualifier=rep.get('qualification_sha256');binding=raw['long_training_qualification_history'].get(qualifier)
                if not binding or binding['samples_sha256']!=qualifier:
                    raise ValueError('endpoint training repeat has no original qualification receipt')
                qualification_receipt(root,binding)
            if round(rep['mean_freq_mhz'])!=point['freq_mhz']:
                raise ValueError('long training frequency differs from requested tier')


def output_budget(rows):
    """Seven seconds, measured pre-barrier lead, and a 15% faster margin."""
    reps=[p for r in rows for p in r['repeats']]
    lead=max(p['observed_context_min']-r['context_tokens']-2/p['step_seconds']
             for r in rows for p in r['repeats'])
    return math.ceil(7/(.85*min(p['step_seconds'] for p in reps)))+math.ceil(max(16,lead))+32


def prepare(*,prior,out):
    prior,out=Path(prior).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('long followup package must use a new immutable directory')
    raw=json.loads((prior/'raw.json').read_text());completion=json.loads((prior/'completion.json').read_text())
    oldplan=json.loads((prior/'training-plan.json').read_text())
    if completion.get('complete') is not True or completion.get('raw_sha256')!=lc.digest(prior/'raw.json'):
        raise ValueError('prior long training is not complete/checksum-bound')
    tp,batches=MODELS.get(raw.get('model_id'),(None,()))
    if raw.get('tp')!=tp or raw.get('pp')!=1:raise ValueError('unsupported own long topology')
    expected={(f,b,c) for f in FREQUENCIES for b in batches for c in (5120,7168)}
    if {(r['freq_mhz'],r['batch'],r['context_tokens']) for r in raw['decode']}!=expected:
        raise ValueError('prior training must cover the explicit exact batches and two anchors')
    verify_training(raw,prior,oldplan['training'],lc.digest(prior/'training-plan.json'))
    points=[]
    for f in FREQUENCIES:
        for b in batches:
            group=[r for r in raw['decode'] if (r['freq_mhz'],r['batch'])==(f,b)]
            required=output_budget(group)
            point=dict(freq_mhz=f,batch=b,context_tokens=7680,max_tokens=512,repeats=3,settle_s=2,measure_s=5,
                purpose='training_extension',training_derived_min_output_tokens=required)
            if required>=512 or lc.point_capacity_error(point,raw['kv_capacity_tokens']):
                raise ValueError(f'endpoint training cannot safely fit seven-second windows: {f}/{b}/{required}')
            points.append(point)
    plan=dict(schema=1,**{k:raw[k] for k in ('system','model_id','tp','pp')},fit_existing_holdout=False,
        training=points,holdout=[],training_source=str(prior/'raw.json'),training_source_sha256=lc.digest(prior/'raw.json'),
        exact_batches=list(batches),batch_interpolation_qualified=False,formal_eligible=False)
    out.mkdir(parents=True);write_immutable(out/'endpoint-training-plan.json',plan)
    manifest=dict(schema=1,kind=KIND,**identity(raw),exact_batches=list(batches),
        prior=str(prior),inputs={name:dict(path=str(prior/name),sha256=lc.digest(prior/name))
            for name in ('raw.json','training-plan.json','completion.json')},
        endpoint_plan_sha256=lc.digest(out/'endpoint-training-plan.json'),implementation_sha256=source_hashes(),
        candidate_family_fixed_before_holdout=True,fit_existing_holdout=False,short_profile_unchanged=True,
        whole_campaign_qualified=False,formal_eligible=False,energy_comparable=False,
        expected_training_points=len(points),expected_holdout_points=len(points)*2,target_context=TARGET)
    write_immutable(out/'manifest.json',manifest);load_package(out)
    return manifest


def load_package(package):
    package=Path(package);m=json.loads((package/'manifest.json').read_text())
    if m['implementation_sha256']!=source_hashes() or m['kind']!=KIND:
        raise ValueError('long followup implementation/package changed')
    for binding in m['inputs'].values():
        if lc.digest(binding['path'])!=binding['sha256']:raise ValueError('long training input changed')
    if lc.digest(package/'endpoint-training-plan.json')!=m['endpoint_plan_sha256']:
        raise ValueError('endpoint training plan changed')
    raw=json.loads(Path(m['inputs']['raw.json']['path']).read_text())
    if identity(raw)!=identity(m):raise ValueError('long package model identity differs')
    oldplan=json.loads(Path(m['inputs']['training-plan.json']['path']).read_text())
    verify_training(raw,Path(m['prior']),oldplan['training'],m['inputs']['training-plan.json']['sha256'])
    plan=json.loads((package/'endpoint-training-plan.json').read_text());lc.validate_training_plan(plan,raw)
    return m,plan,raw


def prepare_training_only(*,training_manifest,out):
    """14B TP1's missing own anchors; collects training only, no fit/holdout."""
    from pdblend.profile.calibration.core import _checkpoint_points
    training_manifest,out=Path(training_manifest).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('training-only plan needs a new immutable directory')
    provenance=json.loads(training_manifest.read_text());source=Path(provenance['training_raw'])
    if lc.digest(source)!=provenance['training_raw_sha256']:raise ValueError('own original training checksum changed')
    raw=json.loads(source.read_text());check_archive(raw)
    if (tuple(raw.get(k) for k in ('system','model_id','tp','pp'))!=('pdblend','Qwen2.5-14B-Instruct',1,1) or
        raw.get('holdout_candidate_sha256') or raw.get('holdout_independent') is True or
        max(r['context_tokens'] for r in raw['decode'])!=4096):
        raise ValueError('requires original 14B TP1 short-context training, not a holdout/repeated long panel')
    # This plan uses decode speed only; legacy prefill summary rows are not
    # being reused as measurements or predictor training for the new panel.
    speed_rows=[r for r in raw['decode'] if r['batch'] in (1,4)]
    _checkpoint_points(dict(prefill=[],decode=speed_rows),source.parent)
    points=[]
    for f in FREQUENCIES:
        for b in (1,4):
            rows=[r for r in raw['decode'] if r['freq_mhz']==f and r['batch']==b]
            if not rows:raise ValueError('missing own training speed for output-budget preflight')
            required=math.ceil(7/(.85*min(p['step_seconds'] for r in rows for p in r['repeats'])))+64
            for c in (5120,7168,7680):
                point=dict(freq_mhz=f,batch=b,context_tokens=c,max_tokens=min(1024,8192-c),repeats=3,
                    settle_s=2,measure_s=5,purpose='training_extension',training_derived_min_output_tokens=required)
                if required>=point['max_tokens'] or lc.point_capacity_error(point,raw['kv_capacity_tokens']):
                    raise ValueError('14B new training point exceeds conservative output/memory budget')
                points.append(point)
    plan=dict(schema=1,system='pdblend',model_id=raw['model_id'],tp=1,pp=1,training=points,holdout=[],
        training_source=str(source),training_source_sha256=lc.digest(source),fit_existing_holdout=False,
        training_only=True,fit_performed=False,independent_holdout=False,formal_eligible=False,
        exact_batches=[1,4],batch_interpolation_qualified=False,short_profile_unchanged=True)
    out.mkdir(parents=True);write_immutable(out/'training-plan.json',plan)
    manifest=dict(**identity(raw),kind='training_only',expected_training_points=36,expected_holdout_points=0,
        plan_sha256=lc.digest(out/'training-plan.json'),implementation_sha256=source_hashes(),
        inputs=dict(training_raw=dict(path=str(source),sha256=lc.digest(source)),
            original_training_manifest=dict(path=str(training_manifest),sha256=lc.digest(training_manifest))),
        fit_performed=False,independent_holdout=False,formal_eligible=False,energy_comparable=False)
    write_immutable(out/'manifest.json',manifest)
    return manifest


def fit_candidate(prior,endpoint):
    if identity(prior)!=identity(endpoint):raise ValueError('long training model identities differ')
    tp,batches=MODELS[prior['model_id']]
    if tp!=prior['tp']:raise ValueError('long training topology mismatch')
    rows=prior['decode']+endpoint['decode'];nodes={}
    if any(r['batch'] not in batches or r['freq_mhz'] not in FREQUENCIES for r in rows):
        raise ValueError('unqualified training batch/frequency cannot enter long candidate')
    for f in FREQUENCIES:
        for b in batches:
            group=sorted((r for r in rows if (r['freq_mhz'],r['batch'])==(f,b)),key=lambda r:r['context_tokens'])
            if [r['context_tokens'] for r in group]!=[5120,7168,7680]:raise ValueError('three training anchors required')
            if any(r.get('independent_holdout') is not False or r.get('evidence_class')!='training_extension' for r in group):
                raise ValueError('holdout rows cannot fit long candidate')
            values=[dict(context=statistics.fmean(p['effective_context_tokens'] for p in r['repeats']),
                step_seconds=statistics.fmean(p['step_seconds'] for p in r['repeats']),
                power_w=statistics.fmean(p['power_w'] for p in r['repeats'])) for r in group]
            if any(values[i]['context']>=values[i+1]['context'] for i in range(2)) or any(not math.isfinite(v) or v<=0 for n in values for v in n.values()):
                raise ValueError('invalid or unordered long training means')
            nodes[f'{f}/{b}']=values
    return dict(schema=1,kind=KIND,**identity(prior),exact_batches=list(batches),nodes=nodes,
        domain_definition='closed interval of observed training means at each exact frequency/batch',
        batch_interpolation_qualified=False,short_profile_unchanged=True,holdout_used=False,formal_eligible=False)


from pdblend.profile.query.long_context import predict


def holdout_plan(candidate,prior,endpoint,capacity):
    points=[];missing=[]
    for f in FREQUENCIES:
        for b in candidate['exact_batches']:
            group=[r for r in prior['decode']+endpoint['decode'] if (r['freq_mhz'],r['batch'])==(f,b)]
            required=output_budget(group)
            upper=candidate['nodes'][f'{f}/{b}'][-1]['context']
            lead=min(p['observed_context_max']-r['context_tokens'] for r in group if r['context_tokens']>=7168 for p in r['repeats'])
            for role,context in [('interior',6144),('campaign_endpoint',math.ceil(TARGET-lead+16))]:
                budget=min(1024,8192-context,math.floor(upper)-context-8)
                point=dict(freq_mhz=f,batch=b,context_tokens=context,max_tokens=budget,repeats=3,settle_s=2,measure_s=5,
                    purpose='independent_holdout_repair',long_context_role=role,training_derived_min_output_tokens=required)
                reason=None
                try:
                    if budget<required:raise ValueError('bounded_output_budget_too_short_for_steady_window')
                    if lc.point_capacity_error(point,capacity):raise ValueError('bounded_output_reservation_exceeds_KV')
                    for c in (context,context+budget-1):predict(candidate,'step_seconds',f,b,c)
                except ValueError as exc:reason=str(exc)
                if reason:missing.append(dict(point=point,status='missing_profile',reason=reason))
                else:points.append(point)
    return dict(schema=1,points=points,missing_points=missing,candidate_sha256=sha256_value(candidate),
        target_context=TARGET,expected_points=12*len(candidate['exact_batches']),
        exact_batches=candidate['exact_batches'],formal_eligible=False)


def qualification_receipt(root,binding):
    path=(Path(root)/binding['samples_file']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or lc.digest(path)!=binding['samples_sha256']:
        raise ValueError('long holdout qualification receipt checksum/path differs')
    receipt=json.loads(path.read_text())
    if (receipt.get('complete') is not True or receipt.get('cross_job') is not True or
        not (receipt.get('passed') is True or receipt.get('fallback')=='serial_cohort')):
        raise ValueError('long holdout requires real parallel/serial-cohort qualification')
    return path


def audit(candidate,raw,root,plan):
    if sha256_value(candidate)!=plan['candidate_sha256']:
        raise ValueError('holdout candidate differs from frozen plan')
    rows=[];failures=[];errors={'step_seconds':[],'power_w':[]};seen=set()
    expected={lc.point_key(p):p for p in plan['points']}
    if len(expected)!=len(plan['points']) or len(plan['points'])+len(plan['missing_points'])!=plan['expected_points']:
        raise ValueError('holdout plan has missing/duplicate declarations')
    for row in raw['decode']:
        key=lc.point_key(row)
        if key in seen or key not in expected:raise ValueError('extra or duplicate long holdout point')
        seen.add(key);point=expected[key]
        if row.get('independent_holdout') is not True or len(row['repeats'])!=3:
            raise ValueError('fresh independent complete holdout required')
        lc.validate_repeats(root,row['repeats'],point,raw['measurement_plan_sha256'],complete=True)
        for rep in row['repeats']:
            qualifier=rep.get('qualification_sha256')
            binding=raw.get('qualification_history',{}).get(qualifier)
            if not binding or binding['samples_sha256']!=qualifier:
                raise ValueError('holdout repeat is missing its actual qualification receipt')
            qualification_receipt(root,binding)
            lc.validate_repeat(root,rep,expected_point=point,expected_plan_sha256=raw['measurement_plan_sha256'])
            if round(rep['mean_freq_mhz'])!=point['freq_mhz']:raise ValueError('holdout frequency differs')
            for metric in errors:
                item=dict(point=key,repeat=rep['repeat'],metric=metric)
                try:
                    # Validate every observed token's context range, not only its average.
                    for c in (rep['observed_context_min'],rep['observed_context_max']):
                        predict(candidate,metric,row['freq_mhz'],row['batch'],c)
                    value=predict(candidate,metric,row['freq_mhz'],row['batch'],rep['effective_context_tokens'])
                    error=abs(value/rep[metric]-1);item.update(relative_error=error,predicted=value,observed=rep[metric])
                    errors[metric].append(error)
                    if not math.isfinite(error) or error>.10:failures.append(item)
                except (ValueError,ZeroDivisionError) as exc:
                    item.update(error=str(exc),status='outside_coverage');failures.append(item)
                rows.append(item)
        if point['long_context_role']=='campaign_endpoint' and any(not p['observed_context_min']<=TARGET<=p['observed_context_max'] for p in row['repeats']):
            failures.append(dict(point=key,error='each fresh endpoint window must actually include context 7679'))
    if seen!=set(expected) or plan['missing_points']:
        failures.append(dict(metric='incomplete_matrix',missing=sorted(set(expected)-seen),unsupported=plan['missing_points']))
    return dict(complete=seen==set(expected) and not plan['missing_points'],passed=not failures,failures=failures,points=rows,
        errors={k:dict(maximum=max(v,default=None),mape=statistics.fmean(v) if v else None) for k,v in errors.items()},
        scope='exact_batch_long_context_only',validated_batches=candidate['exact_batches'],batch_interpolation_qualified=False,
        short_profile_unchanged=True,formal_eligible=False,energy_comparable=False)


class ResidentFollowup:
    def __init__(self,package):
        self.package=Path(package);self.fit_complete=False;self.holdout_points=0;self.holdout_complete=False;self.holdout_passed=False
    def progress_metadata(self):
        return dict(training_collector_fit_performed=False,holdout_points_consumed_by_training=0,
            candidate_fit_performed=self.fit_complete,fit_performed=self.fit_complete,
            fresh_holdout_points=self.holdout_points,fresh_holdout_complete=self.holdout_complete,
            fresh_holdout_calibration_passed=self.holdout_passed,
            candidate_fit_uses_holdout=False,evidence_class='training_extension_and_fresh_independent_holdout')
    def after_qualification(self,profiler):
        root=Path(profiler.out_dir);binding=profiler.raw.get('external_interference') or {}
        source=qualification_receipt(root,binding);sha=binding['samples_sha256']
        history=profiler.raw.setdefault('long_training_qualification_history',{})
        for old in history.values():qualification_receipt(root,old)
        dest=root/'samples'/('long-training-qualification-'+sha+'.json');dest.parent.mkdir(exist_ok=True)
        if dest.exists() and lc.digest(dest)!=sha:raise ValueError('training qualification archive changed')
        if not dest.exists():dest.write_bytes(source.read_bytes())
        history[sha]=dict(samples_file=str(dest.relative_to(root)),samples_sha256=sha)
        profiler.raw['active_long_training_qualification']=sha
        profiler._checkpoint()
    async def __call__(self,*,profiler,client,gpus):
        m,endpoint_plan,prior=load_package(self.package);out=Path(profiler.out_dir);frozen=out/'long-candidate';frozen.mkdir(exist_ok=True)
        verify_training(profiler.raw,out,endpoint_plan['training'],m['endpoint_plan_sha256'])
        if identity(profiler.raw)!=identity(prior):raise ValueError('live endpoint model differs from own training')
        for k in ('image_digest','torch','vllm','cuda','hardware_id'):
            if not prior['environment'].get(k) or prior['environment'][k]!=profiler.raw['environment'].get(k):
                raise ValueError('long extension changed engine/hardware environment: '+k)
        candidate=fit_candidate(prior,profiler.raw)
        candidate['training_binding']=dict(package_manifest_sha256=lc.digest(self.package/'manifest.json'),
            prior_raw_sha256=m['inputs']['raw.json']['sha256'],endpoint_rows_sha256=sha256_value(profiler.raw['decode']))
        write_immutable(frozen/'candidate.json',candidate)
        plan=holdout_plan(candidate,prior,profiler.raw,profiler.raw['kv_capacity_tokens']);write_immutable(frozen/'holdout-plan.json',plan)
        snapshot=frozen/'endpoint-training-archive.json'
        if not snapshot.exists():snapshot.write_bytes((out/'raw.json').read_bytes())
        archived=json.loads(snapshot.read_text());check_archive(archived)
        if sha256_value(archived['decode'])!=candidate['training_binding']['endpoint_rows_sha256']:
            raise ValueError('immutable endpoint snapshot rows changed')
        write_immutable(frozen/'manifest.json',dict(candidate_sha256=lc.digest(frozen/'candidate.json'),
            plan_sha256=lc.digest(frozen/'holdout-plan.json'),endpoint_archive_sha256=lc.digest(snapshot),
            endpoint_samples_root=str(out),frozen_before_fresh_holdout=True,
            endpoint_qualification_source='per-repeat qualification_sha256 and long_training_qualification_history',
            **candidate['training_binding']))
        self.fit_complete=True
        before=copy.deepcopy(profiler.raw);child=copy.copy(profiler);child.out_dir=out/'long-holdout';child.out_dir.mkdir(exist_ok=True)
        current_qualification=profiler.raw.get('external_interference') or {}
        qualified_path=qualification_receipt(out,current_qualification)
        qualified_sha=current_qualification['samples_sha256']
        saved_receipt=child.out_dir/'samples'/('qualification-'+qualified_sha+'.json')
        saved_receipt.parent.mkdir(parents=True,exist_ok=True)
        if saved_receipt.exists() and lc.digest(saved_receipt)!=qualified_sha:
            raise ValueError('immutable prior qualification receipt changed')
        if not saved_receipt.exists():saved_receipt.write_bytes(qualified_path.read_bytes())
        child.raw=copy.deepcopy(profiler.raw)
        for field in ('external_interference','parallel_interference','concurrency_environment','identity_sha256','evidence_bindings',
                      'decode_pending','missing_training_points','training_extension','training_plan_sha256',
                      'long_training_qualification_history','active_long_training_qualification'):
            child.raw.pop(field,None)
        binding=dict(candidate_sha256=lc.digest(frozen/'candidate.json'),plan_sha256=lc.digest(frozen/'holdout-plan.json'),
            package_manifest_sha256=lc.digest(self.package/'manifest.json'))
        child.raw.update(prefill=[],decode=[],mixed=[],transfer=[],static={},decode_pending={},
            config=dict(child.raw['config'],incremental_long_context=False,independent_long_context_holdout=True),
            measurement_plan_sha256=binding['plan_sha256'],long_holdout_binding=binding,parent_training_artifact_root=str(out),
            evidence_class='independent_long_context_holdout')
        if (child.out_dir/'raw.json').exists():
            child.resume()
            if child.raw.get('long_holdout_binding')!=binding:raise ValueError('holdout resume candidate/qualification differs')
        history=child.raw.setdefault('qualification_history',{})
        for old in history.values():qualification_receipt(child.out_dir,old)
        history[qualified_sha]=dict(samples_file=str(saved_receipt.relative_to(child.out_dir)),samples_sha256=qualified_sha)
        done=lc.resume_points(child.raw,child.out_dir,plan['points']);result=dict(complete=False,calibration_passed=False,
            independent_holdout=True,fit_performed=False,candidate_frozen_before_holdout=True,
            formal_eligible=False,energy_comparable=False)
        self.holdout_points=len(done)
        try:
            previous_frequency=None
            for point in plan['points']:
                key=lc.point_key(point)
                if key in done:continue
                if previous_frequency!=point['freq_mhz']:
                    child._lock(point['freq_mhz'],gpus);previous_frequency=point['freq_mhz'];await asyncio.sleep(2)
                def checkpoint(repeats):
                    for rep in repeats:rep.setdefault('qualification_sha256',qualified_sha)
                    child.raw['decode_pending'][key]=repeats;child._checkpoint()
                try:
                    row=await lc.collect_bounded_decode_point(child,client,gpus,point,purpose='independent_holdout_repair',
                        previous=child.raw['decode_pending'].get(key,()),on_window=checkpoint)
                except lc._EarlyEnd as exc:
                    child.raw.setdefault('missing_holdout_points',{})[key]=dict(error=str(exc),reason='bounded_output_budget_exhausted')
                    child._checkpoint();continue
                child.raw['decode'].append(row);child.raw['decode_pending'].pop(key,None);done.add(key);child._checkpoint()
                self.holdout_points=len(done)
                print(f'long holdout f={point["freq_mhz"]} B={point["batch"]} role={point["long_context_role"]}',flush=True)
            checked=audit(candidate,child.raw,child.out_dir,plan);atomic_json(child.out_dir/'holdout-audit.json',checked)
            if lc.digest(frozen/'candidate.json')!=binding['candidate_sha256'] or lc.digest(frozen/'holdout-plan.json')!=binding['plan_sha256']:
                raise ValueError('long candidate/plan changed during holdout')
            result.update(complete=checked['complete'],calibration_passed=checked['passed'],status='completed' if checked['complete'] else 'inconclusive',
                audit_sha256=lc.digest(child.out_dir/'holdout-audit.json'),candidate_sha256=binding['candidate_sha256'],
                measured_points=len(child.raw['decode']),expected_points=plan['expected_points'])
            self.holdout_complete=checked['complete'];self.holdout_passed=checked['passed']
        except BaseException as exc:
            result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            child._checkpoint();result['raw_sha256']=lc.digest(child.out_dir/'raw.json');atomic_json(child.out_dir/'completion.json',result)
            if profiler.raw!=before:raise ValueError('long holdout changed training archive')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    t=sub.add_parser('prepare-training');t.add_argument('--training-manifest',type=Path,required=True);t.add_argument('--out',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--package',type=Path,required=True);r.add_argument('--model',required=True)
    r.add_argument('--gpus',type=int,nargs='+',required=True);r.add_argument('--base-port',type=int,required=True);r.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(prior=args.prior,out=args.out)
    elif args.command=='prepare-training':result=prepare_training_only(training_manifest=args.training_manifest,out=args.out)
    else:
        m,_,_=load_package(args.package)
        result=lc.run(plan_path=args.package/'endpoint-training-plan.json',training_raw_path=Path(m['inputs']['raw.json']['path']),
            model_path=args.model,gpus=args.gpus,base_port=args.base_port,out=args.out,resident_followup=ResidentFollowup(args.package))
    print(json.dumps(result,indent=2),flush=True)
    if args.command=='run':raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
