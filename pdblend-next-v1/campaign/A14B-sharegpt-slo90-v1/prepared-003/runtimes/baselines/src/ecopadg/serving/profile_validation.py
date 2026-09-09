"""Merge measured tables and validate envelopes on independent held-out runs.

Validation never fits the envelope to its own observations. Failures and
uncovered regions remain explicit; a revised model needs new held-out data.
"""
import argparse
from collections import Counter
from dataclasses import asdict,replace
import json
import math
from pathlib import Path

from .evidence import sha256
from .profiles import ProfilePoint,ProfileStore,validate_profile_observations


RESIDENCY_MODEL_MULTIPLIER=1.05


def validate_heldout_matrix(points,expected):
    """Check the declared shape matrix, including multiplicity and omissions."""
    fields=('role','tp','frequency_mhz','input_tokens','context_tokens','batch')
    def key(point):
        value=tuple(point[name] for name in fields)
        if (value[0] not in ('mixed','prefill','decode') or value[1] not in (1,2,4,8)
                or any(type(v) is not int or v<=0 for v in value[1:])):
            raise ValueError('invalid expected held-out shape')
        return value
    wanted=Counter(key(point) for point in expected)
    if not wanted:raise ValueError('nonempty expected held-out matrix required')
    observed=Counter(tuple(getattr(point,name) for name in fields) for point in points)
    def difference(counts):
        return [dict(zip(fields,point),count=count) for point,count in sorted(counts.items())]
    return dict(passed=wanted==observed,expected_observations=sum(wanted.values()),
                actual_observations=sum(observed.values()),missing=difference(wanted-observed),
                unexpected=difference(observed-wanted),
                method='exact declared role/TP/frequency/input/context/batch multiset')


def apply_settled_residency(points,settled):
    """Reconcile the training model with its separately measured idle floor.

    This changes model components, not observations or error envelopes. A zero
    active-phase field retains ProfilePoint's fallback to whole-instance power.
    Validate before taking maxima: a floor must never sanitize invalid inputs.
    """
    points=tuple(points)
    validate_profile_observations(points)
    for evidence in settled.values():
        if (not math.isfinite(evidence['watts']) or evidence['watts']<0
                or not evidence.get('source_sha256')):
            raise ValueError('invalid measured settled residency')
    updated=[];adjustments=[]
    fields=('residency_w','power_w','prefill_power_w','decode_power_w')
    for index,point in enumerate(points):
        evidence=settled.get((point.tp,point.frequency_mhz))
        if evidence is None:
            updated.append(point);continue
        floor=evidence['watts']*RESIDENCY_MODEL_MULTIPLIER
        active=('prefill','decode') if point.role=='mixed' else (point.role,)
        values=dict(residency_w=floor,power_w=max(point.power_w,floor))
        for phase in ('prefill','decode'):
            field=phase+'_power_w';original=getattr(point,field)
            values[field]=(max(original,floor) if original else 0) if phase in active else floor
        revised=replace(point,**values)
        updated.append(revised)
        changed=[field for field in fields if getattr(point,field)!=getattr(revised,field)]
        if changed:
            adjustments.append(dict(point_index=index,role=point.role,tp=point.tp,
                frequency_mhz=point.frequency_mhz,input_tokens=point.input_tokens,
                context_tokens=point.context_tokens,batch=point.batch,source_sha256=point.source_sha256,
                original={field:getattr(point,field) for field in fields},modeled=values,
                original_effective_phase_w={phase:point.phase_power(phase) for phase in ('prefill','decode')},
                modeled_effective_phase_w={phase:revised.phase_power(phase) for phase in ('prefill','decode')},
                active_phases=list(active),changed_fields=changed,settled_evidence=dict(evidence),
                model_floor_w=floor,residency_multiplier=RESIDENCY_MODEL_MULTIPLIER,
                reason='training model consistency with conservative settled residency; inactive phase is idle',
                measurement_changed=False))
    ProfileStore(updated)
    return updated,adjustments


def validate_envelope(store,points):
    points=tuple(points)
    validate_profile_observations(points)
    rows=[]
    for actual in points:
        ratios={}; power_ratios={}; covered=True; queries={}
        phases=(('prefill',) if actual.role=='prefill' else
                ('decode',) if actual.role=='decode' else ('prefill','decode','mixed_bucket_prefill'))
        for label in phases:
            phase='prefill' if label=='mixed_bucket_prefill' else label
            # Match online stage queries: the new mixed prefill is one request,
            # while the decode point covers the resident batch and its context.
            context=actual.input_tokens+1 if label=='prefill' else actual.context_tokens
            batch=1 if label=='prefill' and actual.role=='mixed' else actual.batch
            expected=store.lookup(actual.role,actual.tp,actual.frequency_mhz,
                                  actual.input_tokens,context,batch)
            queries[label]=dict(context_tokens=context,batch=batch)
            covered &= expected is not None
            if expected:
                key='prefill_s' if phase=='prefill' else 'iteration_s'
                metric='mixed_bucket_prefill_s' if label=='mixed_bucket_prefill' else key
                duration=getattr(actual,key)
                bound=expected.phase_time_bound(phase)
                if duration<=0:
                    ratios[metric]=None
                    continue
                ratios[metric]=duration/bound if bound>0 else None
                predicted_power=expected.phase_power_bound(phase)
                actual_power=actual.phase_power(phase)
                power_ratios[label]=actual_power/predicted_power if predicted_power>0 else None
                # A dominating shape bounds execution time, not instantaneous
                # watts. Short NVML-unresolved phases carry a device-limit power
                # upper bound. Validate latency/power error envelopes with the
                # online residency convention. The nominal routing objective
                # remains a central estimate, not this conservative upper bound.
                for incremental in (False,True):
                    # Both sides use the settled modeled baseline. A held-out
                    # short idle prelude can retain power from preceding work;
                    # subtracting it would hide underestimated dynamic energy.
                    observed=max(0,actual_power-(expected.residency_w if incremental else 0))*duration
                    predicted=max(0,predicted_power-(expected.residency_w if incremental else 0))*bound
                    name=label+('_incremental_energy_j' if incremental else '_energy_j')
                    if observed>0: ratios[name]=observed/predicted if predicted>0 else None
        rows.append(dict(role=actual.role,tp=actual.tp,frequency_mhz=actual.frequency_mhz,
            input_tokens=actual.input_tokens,context_tokens=actual.context_tokens,batch=actual.batch,
            covered=covered,queries=queries,ratios=ratios,power_ratios_diagnostic=power_ratios,
            passed=covered and bool(ratios) and all(v is not None and v<=1 for v in ratios.values())))
    return dict(passed=bool(rows) and all(r['passed'] for r in rows),observations=rows,
                incremental_reference='common modeled residency for predicted and held-out phase',
                method='dominating measured bucket with training-only phase history uncertainty; latency and phase energy bounds; no fit to held-out results')


def merge(manifest):
    artifacts={};raw_by_hash={}
    image_ids=set();engine_versions=set();models=set();engine_sources=set()
    for path in manifest['raw']:
        p=Path(path).resolve();digest=sha256(p);raw=json.loads(p.read_text())
        if not raw.get('complete') or raw.get('sampling_error'):
            raise ValueError('incomplete source measurements: '+str(p))
        provenance=raw.get('engine_provenance',[])
        if not provenance: raise ValueError('missing raw engine/image provenance')
        for engine in provenance:
            image_ids.add(engine['image_id']);engine_versions.add(engine['engine_version'])
            models.add(engine['model'])
            engine_sources.add(next((v for p,v in engine['source_files_at_import'].items()
                                     if p.endswith('/serving/engine.py')),None))
        artifacts[str(p)]=digest;raw_by_hash[digest]=raw
    if (image_ids!={manifest['engine_image']} or engine_versions!={'0.9.2'} or len(models)!=1
            or len(engine_sources)!=1 or None in engine_sources):
        raise ValueError('mixed engine images or versions in profile evidence')
    tables=[];points=[];heldout=[];phase_power_sources=[]
    instant_prefill_groups=set();instant_heldout_groups=set()
    for category in ('tables','heldout_tables'):
        for path in manifest.get(category,[]):
            p=Path(path).resolve();table=json.loads(p.read_text())
            if (not table.get('frequency_commands_verified') or table.get('source_sha256') not in raw_by_hash
                    or table.get('frequency_samples_source_sha256')!=table['source_sha256']):
                raise ValueError('table has no certified clock/source measurement: '+str(p))
            if table.get('schema')!=2 or table.get('measurement')!='hardware':
                raise ValueError('only measured tables can be merged')
            for reference_path,reference_hash in table.get('residency_reference',{}).get('artifacts',{}).items():
                if reference_hash not in raw_by_hash or sha256(Path(reference_path))!=reference_hash:
                    raise ValueError('settled reference is missing or changed in raw certification evidence')
            artifacts[str(p)]=sha256(p)
            phase_power_sources.extend(dict(row,table_path=str(p),table_sha256=artifacts[str(p)],
                raw_source_sha256=table['source_sha256'],partition=category,
                power_source=table.get('power_source',dict(mode='legacy_average',
                    api='nvmlDeviceGetPowerUsage',averaging_window_s=1.)))
                for row in table.get('phase_power_sources',[]))
            values=[ProfilePoint(**point) for point in table['points']]
            validate_profile_observations(values)
            if any(value.source_sha256!=table['source_sha256'] for value in values):
                raise ValueError('profile point source differs from its measured table')
            if table.get('power_source',{}).get('mode')=='instant' and table.get('power_source_verified'):
                if category=='tables':
                    instant_prefill_groups.update((v.role,v.tp,v.frequency_mhz) for v in values
                        if v.role in ('mixed','prefill') and v.batch==1)
                else:
                    instant_heldout_groups.update((v.role,v.tp,v.frequency_mhz) for v in values)
            if category=='tables': tables.append(table);points.extend(values)
            else: heldout.extend(values)
    if not points: raise ValueError('no measured profile points')
    expected_matrix=manifest.get('expected_heldout_points')
    if manifest.get('require_complete_heldout_matrix') and not expected_matrix:
        raise ValueError('explicit complete held-out matrix required')
    matrix=validate_heldout_matrix(heldout,expected_matrix) if expected_matrix is not None else None
    # Reject invalid original observations before any envelope or model floor.
    validate_profile_observations(points)
    training_hashes={p.source_sha256 for p in points}
    if training_hashes.intersection(p.source_sha256 for p in heldout):
        raise ValueError('held-out observations cannot be the training measurements')
    interference=[]
    for path in manifest.get('interference',[]):
        p=Path(path).resolve();digest=sha256(p)
        if digest not in raw_by_hash: raise ValueError('interference raw source not included')
        raw=raw_by_hash[digest]
        if not raw.get('frequency_samples'): raise ValueError('interference requires actual SM samples')
        gpus=raw['topology']['mixed']['gpus']
        for run in raw['runs']:
            if run.get('skipped'): continue
            commands=run.get('commanded_frequencies',{})
            if any(commands.get(str(g),commands.get(g))!=run['frequency_mhz'] for g in gpus):
                raise ValueError('interference frequency label differs from command')
            interference.append(dict(run,source_sha256=digest,context_upper=run.get('background_context',raw['background_context'])+
                run.get('background_output',raw['background_output'])))
    # Isolated M/D decode performs the same stage on different GPU groups.
    # Training-side variation between those groups enters the energy error;
    # a power fluctuation must not inflate the TTFT/TPOT latency bound.
    envelopes={}
    for point in points:
        key=(point.tp,point.frequency_mhz,point.input_tokens,point.context_tokens,point.batch)
        values=envelopes.setdefault(key,{})
        if point.role in ('mixed','decode'):
            values['decode_power_w']=max(values.get('decode_power_w',0),point.phase_power('decode'))
    adjusted=[]
    for point in points:
        key=(point.tp,point.frequency_mhz,point.input_tokens,point.context_tokens,point.batch)
        errors=[point.energy_error_fraction]
        for field,maximum in envelopes[key].items():
            if field=='decode_power_w' and point.role in ('mixed','decode'):
                actual=getattr(point,field) or point.power_w
                if actual>0: errors.append(maximum/actual-1)
        adjusted.append(replace(point,energy_error_fraction=max(errors)))
    points=adjusted
    gaps=[];updated=[]
    for point in points:
        if point.role!='mixed' or point.batch==1:
            updated.append(point);continue
        matches=[r for r in interference if r['tp']==point.tp and r['frequency_mhz']==point.frequency_mhz
                 and r['input_tokens']>=point.input_tokens and r['background_batch']>=point.batch-1
                 and r['context_upper']>=point.context_tokens]
        if matches:
            key=min((r['input_tokens'],r['background_batch'],r['context_upper']) for r in matches)
            values=[r['incremental_delay_s'] for r in matches
                    if (r['input_tokens'],r['background_batch'],r['context_upper'])==key]
            point=replace(point,interference_s=max(point.interference_s,*values))
        else:
            gaps.append(dict(tp=point.tp,frequency=point.frequency_mhz,input=point.input_tokens,
                             context=point.context_tokens,batch=point.batch))
        updated.append(point)
    parked={};loaded={}
    for path in manifest.get('residency',[]):
        digest=sha256(path);raw=raw_by_hash.get(digest)
        if raw is None or not raw.get('wakeup',{}).get('passed'):
            raise ValueError('missing resident-power and clock-wakeup evidence')
        for index,item in enumerate(raw['residency']):
            if not math.isfinite(item['watts']) or item['watts']<0:
                raise ValueError('invalid measured settled residency')
            if item['parked']:
                parked[item['tp']]=max(parked.get(item['tp'],0),item['watts']*RESIDENCY_MODEL_MULTIPLIER)
            else:
                key=(item['tp'],item['frequency_mhz'])
                if key not in loaded or item['watts']>loaded[key]['watts']:
                    loaded[key]=dict(watts=item['watts'],source_sha256=digest,
                        source_path=str(Path(path).resolve()),sample_index=index)
    # Replace the operator run's short prelude with separately settled idle
    # power. NVML averaging can retain part of the previous workload's power
    # in a short idle prelude, especially on the decode endpoint.
    updated,residency_adjustments=apply_settled_residency(updated,loaded)
    from .profile_history import apply_prefill_history_bounds
    updated,history_bounds=apply_prefill_history_bounds(updated,tables)
    options=dict(gpu_count=8,node_residency_w=max(t.get('node_residency_w',0) for t in tables),
                 idle_unallocated_gpu_w=max(t.get('idle_unallocated_gpu_w',0) for t in tables))
    unallocated_evidence=None
    if manifest.get('unallocated_residency_reference'):
        from .profile_reference import measured_unallocated_reference
        watts,unallocated_evidence,reference_artifacts=measured_unallocated_reference(
            manifest['unallocated_residency_reference'])
        options['idle_unallocated_gpu_w']=watts
        artifacts.update(reference_artifacts)
    elif manifest.get('require_unallocated_residency_reference'):
        raise ValueError('independent empty-GPU residency evidence required')
    store=ProfileStore(updated,**options)
    validation=validate_envelope(store,heldout)
    trained_groups={(p.role,p.tp,p.frequency_mhz) for p in updated}
    validated_groups={(p.role,p.tp,p.frequency_mhz) for p in heldout}
    missing_groups=sorted(trained_groups-validated_groups)
    validation['missing_role_tp_frequency_groups']=missing_groups
    validation['passed']=validation['passed'] and not missing_groups
    if matrix is not None:
        validation['matrix']=matrix
        validation['passed']=validation['passed'] and matrix['passed']
    required_prefill_groups={key for key in trained_groups if key[0] in ('mixed','prefill')}
    instant_prefill_complete=required_prefill_groups<=instant_prefill_groups
    instant_heldout_complete=trained_groups<=instant_heldout_groups
    instant_complete=instant_prefill_complete and instant_heldout_complete
    residency_complete=({p.tp for p in points}<=set(parked) and
                        {(p.tp,p.frequency_mhz) for p in points}<=set(loaded))
    return dict(schema=2,measurement='hardware',model='Qwen2.5-14B-Instruct',
        points=[asdict(p) for p in updated],**options,engine_image=manifest['engine_image'],
        interference_points=[dict(tp=r['tp'],frequency_mhz=r['frequency_mhz'],input_tokens=r['input_tokens'],
            context_tokens=r['context_upper'],background_batch=r['background_batch'],
            delay_s=r['incremental_delay_s'],source_sha256=r['source_sha256']) for r in interference],
        frequency_commands_verified=True,mixed_interference_measured=bool(interference) and not gaps,
        phase_power_sources=phase_power_sources,
        residency_model_adjustments=residency_adjustments,
        prefill_history_bounds=history_bounds,
        heldout_calibration_complete=validation['passed'],
        instant_prefill_calibration_complete=instant_prefill_complete,
        instant_heldout_calibration_complete=instant_heldout_complete,
        parked_residency_w_by_tp=parked,resident_idle_measured=residency_complete,
        unallocated_residency_reference=unallocated_evidence,
        certification_artifacts=artifacts,interference_gaps=gaps,heldout_validation=validation,
        status='validated_envelope' if (validation['passed'] and not gaps and residency_complete
            and (not manifest.get('require_instant_calibration') or instant_complete)) else 'development_only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('refusing to overwrite profile evidence')
    result=merge(json.loads(args.manifest.read_text()))
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status=result['status'],points=len(result['points']),
        interference_gaps=len(result['interference_gaps']),heldout=result['heldout_calibration_complete'])))


if __name__=='__main__': main()
