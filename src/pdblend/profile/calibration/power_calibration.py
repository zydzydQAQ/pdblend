"""Incremental independent decode-power validation with immutable timing reuse.

``prepare`` is CPU-only. ``run`` starts exactly one resident TP group only when
explicitly invoked. A fresh power receipt never rewrites the old completion or
turns a failed original calibration into a passed original result.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from pdblend.profile.calibration.core import _checkpoint_points, digest, evaluate_holdout
from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.power_table import KIND, PowerCoverageError, context_bounds, validate
from pdblend.profile.collection.wave import atomic_json
from pdblend.profile.collection.window_sampling import _background, _running, summarize_window

FREQUENCIES = (900, 1200, 1500, 1800, 2100, 2520)
SOURCE_FILES = ('profile/power_calibration.py', 'profile/power_table.py', 'profile/model.py',
    'profile/calibration.py', 'profile/acceptance.py', 'profile/window_sampling.py', 'control/planner.py',
    'bench/gate_g5.py', 'profile/profiler.py', 'profile/wave.py', 'profile/parallel.py', 'engine/client.py')


def implementation_hashes():
    from pdblend.source_inventory import implementation_hashes as inventory_hashes
    return inventory_hashes()


def write_immutable(path, value):
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'immutable prepared artifact differs: {path}')
    path.write_text(text)


def timing_fields(model):
    """Fields required to preserve all existing latency predictions/coverage."""
    return {key: copy.deepcopy(getattr(model, key)) for key in (
        'freqs','prefill_time','decode_time','decode_overrides','bounded_coverage',
        'transfer','freq_switch_s','kv_capacity_tokens','kv_bytes_per_token','model','system','tp','pp','profile_key')}


def timing_component(audit):
    """Remove only explicitly typed power errors; unknown failures stay fatal."""
    failures, excluded = [], []
    for item in audit['failures']:
        item = copy.deepcopy(item)
        if item['metric'] in ('decode_power','decode_power_repeat'):
            excluded.append(item)
        elif item['metric'] == 'shared_window_prediction_failure':
            power = [d for d in item['details'] if d.get('metric') in ('decode_power','decode_power_max')]
            other = [d for d in item['details'] if d.get('metric') not in ('decode_power','decode_power_max')]
            if power:
                excluded.append(dict(item, details=power))
            if other or not item['details']:
                failures.append(dict(item, details=other))
        else:
            failures.append(item)
    return dict(passed=not failures, failures=failures, timing_max=audit.get('timing_max'),
        mixed_timing_median=audit.get('mixed_median'), excluded_original_power_failures=excluded,
        scope='original_prefill_decode_timing_and_mixed_timing_only', original_calibration_passed=audit['passed'],
        original_completion_unchanged=True, formal_eligible=False)


def proposal_model(proposal, training, base):
    wanted=('pdblend','Qwen2.5-7B-Instruct',4,1)
    if (proposal.get('system'),proposal.get('model_id'),proposal.get('tp'),proposal.get('pp')) != wanted:
        raise ValueError('power proposal is restricted to independent 7B TP4 PP1 PDBlend')
    if ((training.get('system'),training.get('model_id'),training.get('tp'),training.get('pp')) != wanted or
            (base.system,Path(base.model).name,base.tp,base.pp) != wanted or training.get('holdout_independent') is not False):
        raise ValueError('power proposal/training/base identity mismatch or holdout used as training')
    if (proposal.get('family') != 'bounded_table_linear_batch' or
            proposal.get('override_scope') != 'decode_power_only' or base.decode_power_overrides or
            tuple(base.freqs) != FREQUENCIES or proposal.get('formal_eligible') is not False):
        raise ValueError('unsupported or already-promoted power proposal')
    if any(proposal.get(k) != training.get(k) for k in ('model_hash','tokenizer_hash')):
        raise ValueError('power proposal model/tokenizer differs from training')
    result=copy.deepcopy(base)
    for f in base.freqs:
        rows=[r for r in training['decode'] if r['freq_mhz']==f]
        nodes=[dict(batch=r['batch'],nominal_context_tokens=r['context_tokens'],
            context_min=min(x['effective_context_tokens'] for x in r['repeats']),
            context_max=max(x['effective_context_tokens'] for x in r['repeats']),
            power_w=statistics.fmean(x['power_w'] for x in r['repeats'])) for r in rows]
        planned=proposal['tables'][str(f)]
        order=lambda n:(n['batch'],n['nominal_context_tokens'])
        for actual,expected in zip(sorted(nodes,key=order),sorted(planned,key=order)):
            if actual.keys()!=expected.keys() or any(not math.isclose(actual[k],expected[k],rel_tol=1e-12,abs_tol=1e-9) for k in actual):
                raise ValueError('power table differs from checksum-bound training observations')
        if len(nodes)!=len(planned) or len(nodes)!=24:
            raise ValueError('power table requires all 24 training shapes per frequency')
        spec=dict(kind=KIND,batch_interpolation='linear',nodes=nodes,
            training_raw_sha256=proposal['training_raw_sha256'],validation_status='training_only')
        validate(spec);result.decode_power_overrides[f]=spec
    result.quality['decode_power_calibration']=dict(status='training_only',independent_holdout=False,
        training_raw_sha256=proposal['training_raw_sha256'],base_candidate_sha256=proposal['base_candidate_sha256'])
    if timing_fields(result)!=timing_fields(base):
        raise AssertionError('power-only update changed timing')
    return result


def reserve_points(training, model, training_root):
    """Fit the entire reserved generation inside both frozen domains.

    Real training start counts include prefill/barrier lead, which is often
    larger than 2/step_seconds for a large batch. Reserve that measured lead,
    three full windows, 15% faster decoding and 32 terminal guard tokens.
    """
    points=[]
    for f in FREQUENCIES:
        for batch,target,reason in ((1,1024,'single_sequence_regime'),(64,2048,'measured_batch_power_trough'),
                                   (256,4096,'high_batch_high_context'),(192,2048,'unseen_batch_interpolation')):
            measured=[r for r in training['decode'] if r['freq_mhz']==f]
            bs=sorted({r['batch'] for r in measured})
            brackets=[batch] if batch in bs else [max(b for b in bs if b<batch),min(b for b in bs if b>batch)]
            anchors=[]
            for b in brackets:
                rows=[r for r in measured if r['batch']==b]
                near=min(rows,key=lambda r:abs(r['effective_context_tokens']-target))
                anchors.append(near)
            samples=[]
            for r in anchors:
                for rep in r['repeats']:
                    path=(training_root/rep['samples_file']).resolve()
                    if not path.is_relative_to(training_root.resolve()) or digest(path)!=rep['samples_sha256']:
                        raise ValueError('training reservation sample checksum mismatch')
                    sample=json.loads(path.read_text())
                    samples.append((rep,sample))
            fastest=min(rep['step_seconds'] for rep,_ in samples)
            if not math.isfinite(fastest) or fastest<=0:
                raise ValueError('missing measured reservation latency')
            lead=max(max(s['start_token_counts']) for _,s in samples)
            tail=math.ceil(21/(fastest*.85))+lead+32
            low,high=context_bounds(model.decode_power_overrides[f],batch)
            lower=math.ceil(low)+8;upper=math.floor(high)-tail-8
            if lower>upper:
                raise ValueError('missing_profile: cannot reserve three power windows without extrapolation')
            preferred=(upper if batch==256 else target-math.ceil(4.5/fastest)-max(0,lead-math.ceil(2/fastest)))
            prompt=min(upper,max(lower,preferred))
            point=dict(freq_mhz=f,batch=batch,context_tokens=prompt,max_tokens=tail,repeats=3,
                settle_s=2.,measure_s=5.,purpose='independent_power_holdout',reason=reason,
                reservation=dict(fastest_training_step_s=fastest,measured_start_count_lead=lead,
                    generation_speed_margin=.15,terminal_guard_tokens=32,
                    power_context_bounds=[low,high],full_reserved_context=[prompt,prompt+tail-1],
                    nominal_target_context=target,training_anchor_shapes=[dict(batch=r['batch'],context_tokens=r['context_tokens']) for r in anchors]),
                sampling_method='three_windows_shared_prefill',automatic_prediction_failure_retry=False)
            for context in (prompt,prompt+tail-1):
                if not model.decode_supported(batch,context,f) or not model.decode_power_supported(batch,context,f):
                    raise ValueError('power holdout reservation exceeds unchanged timing or new power domain')
            if prompt+tail>8192 or batch*(prompt+tail)>.9*training['kv_capacity_tokens']:
                raise ValueError('power holdout output reservation exceeds memory/model length')
            points.append(point)
    return points


def scheduling_proxy(training,points):
    rows=[]
    for point in points:
        anchors=sorted((r for r in training['prefill'] if r['freq_mhz']==point['freq_mhz']),key=lambda r:r['input_tokens'])
        context=point['context_tokens']
        lower=[r for r in anchors if r['input_tokens']<=context]
        upper=[r for r in anchors if r['input_tokens']>=context]
        if not lower or not upper:
            raise ValueError('power scheduling proxy would exceed measured prefill lengths')
        lo,hi=lower[-1],upper[0]
        weight=(context-lo['input_tokens'])/(hi['input_tokens']-lo['input_tokens']) if hi['input_tokens']!=lo['input_tokens'] else 0
        seconds=lo['seconds']+weight*(hi['seconds']-lo['seconds'])
        rows.append(dict(freq_mhz=point['freq_mhz'],batch=point['batch'],context_tokens=context,
            single_request_prefill_seconds=seconds,serial_prefill_work_proxy_seconds=point['batch']*seconds,
            window_seconds=21.,prefill_anchors=[dict(input_tokens=r['input_tokens'],seconds=r['seconds']) for r in (lo,hi)]))
    work=sum(r['serial_prefill_work_proxy_seconds'] for r in rows)
    return dict(rows=rows,serial_prefill_work_proxy_seconds=work,decode_window_seconds=len(rows)*21,
        combined_proxy_seconds=work+len(rows)*21,actual_batch_runtime_measured=False,not_a_bound=True,
        excluded=['model_load','token_barrier','clock_changes','paired_qualification','cleanup','retries'])


def prepare(*, proposal_path, original_holdout, out):
    proposal_path,original_holdout,out=map(Path,(proposal_path,original_holdout,out))
    proposal=json.loads(proposal_path.read_text())
    training_path=Path(proposal['training_raw']);base_path=Path(proposal['base_candidate'])
    if digest(training_path)!=proposal['training_raw_sha256'] or digest(base_path)!=proposal['base_candidate_sha256']:
        raise ValueError('power proposal source checksum mismatch')
    training=json.loads(training_path.read_text());base=PerfModel.load(base_path)
    # Verify every original raw sample and its actual context/power, not only
    # the representative samples used for reservation estimates.
    for row in training['decode']:
        for rep in row['repeats']:
            path=(training_path.parent/rep['samples_file']).resolve()
            if not path.is_relative_to(training_path.parent.resolve()) or digest(path)!=rep['samples_sha256']:
                raise ValueError('power training raw sample checksum mismatch')
            evidence=json.loads(path.read_text())
            context=row['context_tokens']+statistics.fmean((a+b)/2 for a,b in zip(evidence['start_token_counts'],evidence['end_token_counts']))
            watts=statistics.fmean(sum(p) for _,p in evidence['power'])
            if not math.isclose(context,rep['effective_context_tokens'],abs_tol=1e-8) or not math.isclose(watts,rep['power_w'],rel_tol=1e-9):
                raise ValueError('training context/power does not match raw evidence')
            if rep['steady_window_s']<5 or rep['min_steps']<8 or len(evidence['power'])<2 or not evidence['frequency']:
                raise ValueError('training sample window gates failed')
    model=proposal_model(proposal,training,base)
    points=reserve_points(training,model,training_path.parent)
    original_raw_path=original_holdout/'raw.json';original_completion=original_holdout/'completion.json'
    completion=json.loads(original_completion.read_text());old_raw=json.loads(original_raw_path.read_text())
    if (completion.get('complete') is not True or completion.get('independent_holdout') is not True or
            completion.get('candidate_sha256')!=digest(base_path) or completion.get('raw_sha256')!=digest(original_raw_path)):
        raise ValueError('original timing archive is not complete/independent/bound to the unchanged candidate')
    for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp'):
        if old_raw.get(k)!=training.get(k):
            raise ValueError(f'original timing identity differs: {k}')
    original_manifest_path=base_path.parent/'manifest.json'
    original_manifest=json.loads(original_manifest_path.read_text())
    if original_manifest['candidate_sha256']!=digest(base_path):
        raise ValueError('original timing manifest checksum mismatch')
    _checkpoint_points(old_raw,original_holdout)
    timing=timing_component(evaluate_holdout(old_raw,base,original_holdout,expected_plan=original_manifest['plan']))
    # Preserve timing failures too; preparing does not mean eligible.
    serialized=json.loads(model.to_json())
    write_immutable(out/'candidate.json',serialized)
    write_immutable(out/'original-timing-component-audit.json',timing)
    plan=dict(schema=1,purpose='independent_decode_power_holdout',system='pdblend',model_id=training['model_id'],
        model_hash=training['model_hash'],tokenizer_hash=training['tokenizer_hash'],tp=4,pp=1,
        points=points,repeats=3,minimum_decode_window_seconds=504,independent_holdout=True,fit_performed=False,
        power_gate=dict(mape_max=.10,max_error_max=.15,each_independent_window_max_error=.10),
        full_reservation_inside_frozen_power_and_timing_domain=True,point_count=24,
        no_automatic_retry_of_prediction_failures=True,prompt_seed_rule='1000 * batch + request_index',
        qualification_frequency=2100,qualification_note='existing representative paired-layout protocol; six frequencies apply to profile sampling',
        cpu_scheduling_proxy=scheduling_proxy(training,points),
        formal_eligible=False,energy_comparable=False,original_timing_recollection=False)
    write_immutable(out/'power-plan.json',plan)
    inputs=dict(proposal=proposal_path,training_raw=training_path,base_candidate=base_path,
        original_raw=original_raw_path,original_completion=original_completion,original_manifest=original_manifest_path)
    manifest=dict(schema=1,status='prepared_independent_power_validation',formal_eligible=False,
        model_id=training['model_id'],model_hash=training['model_hash'],tokenizer_hash=training['tokenizer_hash'],system='pdblend',tp=4,pp=1,
        candidate_sha256=digest(out/'candidate.json'),plan_sha256=digest(out/'power-plan.json'),
        timing_component_sha256=digest(out/'original-timing-component-audit.json'),
        inputs={k:dict(path=str(p.resolve()),sha256=digest(p)) for k,p in inputs.items()},
        implementation_sha256=implementation_hashes(),training_environment=training['environment'],
        original_timing_environment=old_raw['environment'],timing_fields_unchanged=True,
        timing_component_passed=timing['passed'],original_completion_status=completion.get('calibration_status'),
        required_remaining=['fresh_power_holdout','provenance_and_mixed_evidence','transfer_protocol_revalidation','native_mechanisms','campaign_acceptance'])
    write_immutable(out/'manifest.json',manifest)
    return manifest


def load_package(package):
    manifest=json.loads((package/'manifest.json').read_text())
    if (digest(package/'candidate.json')!=manifest['candidate_sha256'] or
            digest(package/'power-plan.json')!=manifest['plan_sha256'] or
            digest(package/'original-timing-component-audit.json')!=manifest['timing_component_sha256'] or
            implementation_hashes()!=manifest['implementation_sha256']):
        raise ValueError('prepared power package or implementation changed; prepare a new package')
    for row in manifest['inputs'].values():
        if digest(row['path'])!=row['sha256']:
            raise ValueError('immutable power package input changed')
    model=PerfModel.load(package/'candidate.json');base=PerfModel.load(Path(manifest['inputs']['base_candidate']['path']))
    if timing_fields(model)!=timing_fields(base):
        raise ValueError('power-only candidate changed original timing fields')
    plan=json.loads((package/'power-plan.json').read_text())
    if len(plan['points'])!=24 or {(p['freq_mhz'],p['batch']) for p in plan['points']}!={(f,b) for f in FREQUENCIES for b in (1,64,192,256)}:
        raise ValueError('power holdout is not the declared 24-point matrix')
    for p in plan['points']:
        if p['purpose']!='independent_power_holdout' or p['repeats']!=3 or p['settle_s']<2 or p['measure_s']<5:
            raise ValueError('power holdout sampling gates changed')
        if any(not model.decode_power_supported(p['batch'],c,p['freq_mhz']) or not model.decode_supported(p['batch'],c,p['freq_mhz'])
               for c in (p['context_tokens'],p['context_tokens']+p['max_tokens']-1)):
            raise ValueError('power plan full output reservation is outside frozen support')
    return manifest,plan,model


def point_key(point):
    return f'{point["freq_mhz"]}-{point["batch"]}-{point["context_tokens"]}'


def validate_repeat(root,rep,point,*,binding):
    path=(root/rep['samples_file']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path)!=rep['samples_sha256']:
        raise ValueError('power window sample checksum mismatch')
    raw=json.loads(path.read_text())
    if raw['point']!=point or raw['binding']!=binding or raw['purpose']!='independent_power_holdout':
        raise ValueError('power window belongs to another shape, model or plan')
    if raw['repeat']!=rep['repeat'] or raw['shared_decode_run']!=rep['shared_decode_run']:
        raise ValueError('power window repeat/run identity differs from raw')
    actual=summarize_window(token_times=raw['token_times_s'],context=point['context_tokens'],
        start_s=raw['start_s'],end_s=raw['end_s'],power=raw['power'],frequency=raw['frequency'],
        gpu_count=len(rep['measured_gpu_ids']),settle_s=raw['start_s']-raw['settle_start_s'],measurement_s=point['measure_s'])
    for key in ('effective_context_tokens','observed_context_min','observed_context_max','step_seconds','power_w',
                'min_steps','power_samples','frequency_samples','steady_window_s','mean_freq_mhz'):
        if not math.isclose(rep[key],actual[key],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError(f'power window summary differs from raw: {key}')
    if actual['observed_context_max']>=point['context_tokens']+point['max_tokens']:
        raise ValueError('power output exceeded full reservation')
    return actual


def resume_windows(raw,root,points,binding):
    expected={point_key(p):p for p in points};completed=set()
    def validate_sequence(repeats,point):
        if len({r['samples_sha256'] for r in repeats})!=len(repeats):
            raise ValueError('power repeats reuse the same measurement window')
        for index,rep in enumerate(repeats):
            if rep['repeat']!=index or (index and rep['start_s']<repeats[index-1]['end_s']+2-1e-6):
                raise ValueError('power repeat indices/settle intervals are invalid')
            validate_repeat(root,rep,point,binding=binding)
    for row in raw.get('decode',[]):
        key=point_key(row)
        if key not in expected or key in completed or len(row['repeats'])!=3:
            raise ValueError('unexpected/duplicate/incomplete power checkpoint point')
        validate_sequence(row['repeats'],expected[key])
        completed.add(key)
    for key,repeats in raw.get('power_pending',{}).items():
        if key not in expected or key in completed or len(repeats)>3:
            raise ValueError('invalid partial power checkpoint')
        validate_sequence(repeats,expected[key])
    return completed


async def collect_power_point(profiler,client,gpus,point,*,model,binding,previous=(),on_window=None,
                              _clock=time.time,_sleep=asyncio.sleep,_background_factory=None):
    if (len(gpus)!=model.tp or len(set(gpus))!=len(gpus) or
            point['batch']*(point['context_tokens']+point['max_tokens'])>.9*profiler.raw['kv_capacity_tokens']):
        raise ValueError('power measurement GPU group or full reservation is invalid')
    repeats=list(previous)
    for rep in repeats:
        validate_repeat(profiler.out_dir,rep,point,binding=binding)
    if len(repeats)>3:
        raise ValueError('too many previous power windows')
    if len(repeats)<3:
        run_id=uuid.uuid4().hex
        factory=_background_factory or _background
        async with factory(profiler,client,point,'power-'+point_key(point)+'-'+run_id) as (live,tasks):
            if len(live)!=point['batch']:
                raise ValueError('power measurement live batch differs from plan')
            for index in range(len(repeats),3):
                settle_start=_clock();await _sleep(point['settle_s']);_running(tasks)
                sampler=profiler.meter.sampler(gpus);sampler.start();start=_clock()
                try:
                    await _sleep(point['measure_s']);end=_clock();_running(tasks)
                finally:
                    sampler.stop()
                if sampler.error:
                    raise RuntimeError(f'power sampler failed: {sampler.error}')
                evidence=dict(point=point,binding=binding,purpose='independent_power_holdout',
                    shared_decode_run=run_id,repeat=index,start_s=start,end_s=end,settle_start_s=settle_start,
                    token_times_s=[list(r.token_times_s) for r in live],
                    power=[x for x in sampler.samples if start<=x[0]<=end],
                    frequency=[x for x in sampler.frequency_samples if start<=x[0]<=end])
                path=profiler.out_dir/'samples'/f'power-{point_key(point)}-{run_id}-r{index}.json'
                atomic_json(path,evidence)
                rep=summarize_window(token_times=evidence['token_times_s'],context=point['context_tokens'],
                    start_s=start,end_s=end,power=evidence['power'],frequency=evidence['frequency'],
                    gpu_count=len(gpus),settle_s=start-settle_start,measurement_s=point['measure_s'])
                rep.update(freq_mhz=point['freq_mhz'],measured_gpu_ids=list(gpus),repeat=index,shared_decode_run=run_id,
                    samples_file=str(path.relative_to(profiler.out_dir)),samples_sha256=digest(path))
                # Keep raw evidence/checkpoint even when the prediction fails.
                repeats.append(rep)
                if on_window: on_window(list(repeats))
                validate_repeat(profiler.out_dir,rep,point,binding=binding)
                for context in (rep['observed_context_min'],rep['effective_context_tokens'],rep['observed_context_max']):
                    if not model.decode_power_supported(point['batch'],context,point['freq_mhz']):
                        raise PowerCoverageError('missing_profile: actual power window escaped frozen support')
    powers=[r['power_w'] for r in repeats]
    return dict(freq_mhz=point['freq_mhz'],batch=point['batch'],context_tokens=point['context_tokens'],
        effective_context_tokens=statistics.median(r['effective_context_tokens'] for r in repeats),
        power_w=statistics.median(powers),power_repeats=powers,repeats=repeats,
        sampling_method='shared_prefill_power_windows',prefill_runs=len({r['shared_decode_run'] for r in repeats}),
        independent_holdout=True,formal_eligible=False)


def audit_power(raw,root,points,model,binding,*,windows_per_frequency=12):
    expected={point_key(p):p for p in points};rows=[];failures=[]
    try:
        complete=resume_windows(raw,root,points,binding)
    except (ValueError,KeyError,OSError) as exc:
        return dict(passed=False,failures=[dict(metric='power_raw_evidence',error=str(exc))],points=[],formal_eligible=False)
    if complete!=set(expected):
        failures.append(dict(metric='power_matrix_coverage',missing=sorted(set(expected)-complete)))
    for row in raw.get('decode',[]):
        point=expected[point_key(row)];powers=[]
        for index,rep in enumerate(row['repeats']):
            value=dict(freq_mhz=row['freq_mhz'],batch=row['batch'],repeat=index,
                effective_context_tokens=rep['effective_context_tokens'],observed=rep['power_w'],predicted=None,relative_error=None)
            try:
                for context in (rep['observed_context_min'],rep['effective_context_tokens'],rep['observed_context_max']):
                    if not model.decode_power_supported(row['batch'],context,row['freq_mhz']):
                        raise PowerCoverageError('actual power context outside frozen support')
                predicted=model.decode_power_w(row['batch'],row['freq_mhz'],ctx=rep['effective_context_tokens'])
                if rep['power_w']<=0: raise ValueError('non-positive observed power')
                error=abs(predicted/rep['power_w']-1)
                value.update(predicted=predicted,relative_error=error)
                if not math.isfinite(error) or error>.10:
                    failures.append(dict(value,metric='independent_power_window',limit=.10))
                if round(rep['mean_freq_mhz'])!=row['freq_mhz']:
                    failures.append(dict(value,metric='power_clock_identity',observed_mhz=rep['mean_freq_mhz']))
                powers.append(rep['power_w'])
            except PowerCoverageError as exc:
                value.update(status='outside_coverage',error_class='measurement_domain_error')
                failures.append(dict(value,metric='power_domain',error=str(exc)))
            except ValueError as exc:
                failures.append(dict(value,metric='invalid_power_evidence',error=str(exc)))
            rows.append(value)
        if len(powers)==3 and statistics.stdev(powers)/statistics.fmean(powers)>.10:
            failures.append(dict(metric='power_repeat_noise',point=point_key(point)))
    by_frequency={}
    for f in model.freqs:
        values=[r['relative_error'] for r in rows if r['freq_mhz']==f and r['relative_error'] is not None]
        by_frequency[str(f)]=dict(mape=statistics.fmean(values) if values else None,max_error=max(values) if values else None,
                                  windows=len(values),expected_windows=windows_per_frequency)
        if len(values)!=windows_per_frequency or statistics.fmean(values)>.10 or max(values)>.15:
            failures.append(dict(metric='power_frequency_gate',freq_mhz=f,**by_frequency[str(f)]))
    return dict(passed=not failures,failures=failures,points=rows,by_frequency=by_frequency,
        scope='fresh_decode_power_only',formal_eligible=False,fit_performed=False)


def composite_audit(package,raw,out,manifest,plan,model,binding):
    load_package(package)  # Recheck immutable source bytes at the end too.
    power=audit_power(raw,out,plan['points'],model,binding)
    original=Path(manifest['inputs']['original_raw']['path'])
    base=PerfModel.load(Path(manifest['inputs']['base_candidate']['path']))
    original_raw=json.loads(original.read_text())
    original_manifest=json.loads(Path(manifest['inputs']['original_manifest']['path']).read_text())
    timing=timing_component(evaluate_holdout(original_raw,base,original.parent,expected_plan=original_manifest['plan']))
    write_immutable(out/'power-only-audit.json',power)
    write_immutable(out/'reused-timing-audit.json',timing)
    result=dict(schema=1,calibration_components_passed=power['passed'] and timing['passed'],formal_eligible=False,
        power=power,timing=timing,old_completion_unchanged=True,original_power_retained_as_diagnostic=True,
        binding=binding,power_raw_sha256=digest(out/'raw.json'),
        component_receipts=dict(power=digest(out/'power-only-audit.json'),timing=digest(out/'reused-timing-audit.json')),
        original_sources=manifest['inputs'],power_environment=raw['environment'],
        original_timing_environment=manifest['original_timing_environment'],implementation_sha256=manifest['implementation_sha256'],
        missing_gates=['full_profile_provenance_revalidation','mixed_power_if_required','transfer_protocol_revalidation','native_system_mechanisms','campaign_acceptance'])
    write_immutable(out/'composite-audit.json',result)
    return result


def timing_package_binding(package):
    """Bind optional independent timing inputs without changing any package."""
    package=Path(package).resolve()
    manifest=json.loads((package/'manifest.json').read_text())
    if (manifest.get('model_id'),manifest.get('tp'),manifest.get('pp'))!=('Qwen2.5-32B-Instruct',4,1):
        raise ValueError('resident timing overlay requires its own 32B TP4 PP1 package')
    files={}
    for path in sorted(package.rglob('*')):
        if path.is_symlink():raise ValueError('timing package symlinks are not immutable inputs')
        if path.is_file():files[str(path.relative_to(package))]=digest(path)
    if not {'manifest.json','candidate.json','timing-plan.json'}<=set(files):
        raise ValueError('timing package candidate/plan/manifest inputs are incomplete')
    return dict(package=str(package),files=files)


async def collect_resident_timing(*,profiler,client,gpus,package,out,power_model,power_plan):
    """Run an independent collector while keeping power evidence read-only."""
    from pdblend.profile.calibration.timing_calibration import collect_existing
    package,out=Path(package),Path(out)
    binding=timing_package_binding(package)
    protected_raw=copy.deepcopy(profiler.raw)
    protected_model=power_model.to_json()
    protected_plan=copy.deepcopy(power_plan)
    try:
        report=await collect_existing(profiler=profiler,client=client,gpus=gpus,package=package,out=out)
    finally:
        if (profiler.raw!=protected_raw or power_model.to_json()!=protected_model or power_plan!=protected_plan):
            raise ValueError('timing overlay modified protected power raw/model/plan')
        if timing_package_binding(package)!=binding:
            raise ValueError('timing overlay modified its immutable package')
    receipt=out/'completion.json'
    if not receipt.is_file() or report.get('receipt_sha256')!=digest(receipt):
        raise ValueError('timing overlay has no checksum-bound completion')
    stored=json.loads(receipt.read_text())
    if (report.get('complete') is not True or stored.get('complete') is not True
            or type(report.get('timing_passed')) is not bool
            or stored.get('timing_passed') is not report['timing_passed']
            or stored.get('formal_eligible') is not False or stored.get('energy_comparable') is not False):
        raise ValueError('timing overlay completion or independent scope is invalid')
    return dict(requested=True,complete=True,timing_passed=report['timing_passed'],
        completion=str(receipt),receipt_sha256=digest(receipt),package_binding=binding,
        formal_eligible=False,energy_comparable=False,original_power_and_timing_results_unchanged=True)


def run(*,package,model_path,gpus,base_port,out,timing_package=None,panel_adapter=None):
    from pdblend.profile.collection.profiler import Profiler, _load_flock
    from pdblend.profile.collection.wave import ProfileWave
    from pdblend.engine.launcher import Fleet
    from pdblend.engine.client import EngineClient
    package,out=Path(package),Path(out)
    loader=load_package if panel_adapter is None else panel_adapter.load_package
    manifest,plan,model=loader(package)
    if panel_adapter is not None and timing_package is not None:
        raise ValueError('local power panels cannot silently attach another timing package')
    if timing_package is not None:
        timing_binding=timing_package_binding(timing_package)
        if manifest.get('model_id')!='Qwen2.5-32B-Instruct':
            raise ValueError('the independent timing panel is only attached to its own 32B power member')
    binding=dict(candidate_sha256=manifest['candidate_sha256'],plan_sha256=manifest['plan_sha256'],
        package_manifest_sha256=digest(package/'manifest.json'))
    profiler=Profiler(model_path,gpus,tp=4,pp=1,system='pdblend',out_dir=out,hardware_id='8xL20-lease',
                      base_port=base_port,kv_connector='P2pNcclConnector')
    if len(gpus)!=4 or len(profiler.specs)!=1:
        raise ValueError('power-only holdout requires exactly one resident TP4 group')
    if (profiler.model_spec.model_hash,profiler.model_spec.tokenizer_hash)!=(manifest['model_hash'],manifest['tokenizer_hash']):
        raise ValueError('live model/tokenizer differs from frozen power candidate')
    for key in ('image_digest','vllm','torch','cuda','hardware_id'):
        old=manifest['original_timing_environment'].get(key);new=profiler.raw['environment'].get(key)
        if not old or old!=new:
            raise ValueError(f'cannot compose old timing with changed/missing {key}')
    profiler.raw['config']['power_only_holdout']=True
    if panel_adapter is not None:
        profiler.raw['config']['power_panel_scope']=panel_adapter.scope
        profiler.raw['config']['concurrency_mode']=panel_adapter.concurrency_mode
    profiler.raw['power_holdout_binding']=binding
    if (out/'raw.json').is_file():
        profiler.resume()
        if profiler.raw.get('power_holdout_binding')!=binding:
            raise ValueError('power checkpoint belongs to another immutable package')
    completed=resume_windows(profiler.raw,out,plan['points'],binding)
    profiler.raw.setdefault('power_pending',{})
    wave=ProfileWave.from_environment() if panel_adapter is None else panel_adapter.make_wave(profiler)
    if wave is None:
        raise ValueError('power-only holdout requires coordinated ProfileWave qualification')
    result=dict(status='running',complete=False,formal_eligible=False,energy_comparable=False,
        scope='independent_decode_power_only',independent_holdout=True,fit_performed=False,
        started_s=time.time(),binding=binding,prefill_grid_points=0,mixed_grid_points=0,
        timing_overlay_requested=timing_package is not None)
    if timing_package is not None:result['timing_package_binding']=timing_binding
    if panel_adapter is not None:result['scope']=panel_adapter.scope
    atomic_json(out/'frozen-package.json',manifest)
    try:
        followup_needed=(panel_adapter is not None and hasattr(panel_adapter,'needs_followup') and
                         panel_adapter.needs_followup(package,out))
        if len(completed)<len(plan['points']) or timing_package is not None or followup_needed:
            with Fleet(profiler.specs,out/'logs') as fleet:
                with _load_flock():fleet.start_all()
                instance=fleet[profiler.specs[0].instance_id]
                profiler.raw['kv_capacity_tokens']=profiler._kv_capacity(instance)
                async def sample():
                    await wave.qualify_external(profiler,fleet)
                    async with wave.measurement():
                        async with EngineClient(instance.spec.instance_id,instance.spec.base_url) as client:
                            previous_frequency=None
                            for point in plan['points']:
                                key=point_key(point)
                                if key in completed:continue
                                if previous_frequency!=point['freq_mhz']:
                                    profiler._lock(point['freq_mhz'],instance.spec.gpus)
                                    previous_frequency=point['freq_mhz'];await asyncio.sleep(2)
                                def checkpoint(repeats):
                                    profiler.raw['power_pending'][key]=repeats;profiler._checkpoint()
                                row=await collect_power_point(profiler,client,instance.spec.gpus,point,model=model,binding=binding,
                                    previous=profiler.raw['power_pending'].get(key,()),on_window=checkpoint)
                                profiler.raw['decode'].append(row);profiler.raw['power_pending'].pop(key,None)
                                completed.add(key);profiler._checkpoint()
                                print(f'power holdout f={point["freq_mhz"]} B={point["batch"]} windows={len(row["repeats"])} '+
                                      f'contexts={[round(r["effective_context_tokens"],1) for r in row["repeats"]]}',flush=True)
                            if panel_adapter is not None and hasattr(panel_adapter,'after_samples'):
                                result['panel_followup']=await panel_adapter.after_samples(
                                    profiler=profiler,client=client,gpus=instance.spec.gpus,
                                    package=package,out=out/'mixed-repair')
                            if timing_package is not None:
                                result['timing_overlay']=await collect_resident_timing(profiler=profiler,client=client,
                                    gpus=instance.spec.gpus,package=timing_package,out=out/'timing-overlay',
                                    power_model=model,power_plan=plan)
                asyncio.run(sample())
        profiler._checkpoint()
        auditor=composite_audit if panel_adapter is None else panel_adapter.audit
        audit=auditor(package,profiler.raw,out,manifest,plan,model,binding)
        result.update(status='completed',complete=True,calibration_components_passed=audit['calibration_components_passed'],
            power_passed=audit['power']['passed'],reused_timing_passed=audit['timing']['passed'],
            measured_decode_points=len(profiler.raw['decode']),composite_audit_sha256=digest(out/'composite-audit.json'))
        if panel_adapter is not None:
            result.update(full_profile_qualified=False,validation_scope=panel_adapter.scope,
                          concurrency_qualified=audit.get('concurrency_qualified',False))
            if 'repaired_timing' in audit:
                result['repaired_timing_passed']=audit['repaired_timing']['passed']
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        wave.write('error',dict(error=result['error']))
    finally:
        profiler._checkpoint();profiler.meter.reset_all()
        result.update(finished_s=time.time(),raw_sha256=digest(out/'raw.json'))
        atomic_json(out/'completion.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--original-holdout',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--package',type=Path,required=True);r.add_argument('--model',required=True)
    r.add_argument('--gpus',type=int,nargs='+',required=True);r.add_argument('--base-port',type=int,required=True)
    r.add_argument('--out',type=Path,required=True);r.add_argument('--timing-package',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        result=prepare(proposal_path=args.proposal,original_holdout=args.original_holdout,out=args.out)
    else:
        result=run(package=args.package,model_path=args.model,gpus=args.gpus,base_port=args.base_port,out=args.out,
                   timing_package=args.timing_package)
    print(json.dumps(result,indent=2),flush=True)
    if args.command=='run':raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
