#!/usr/bin/env python3
"""Derive reviewable SLO endpoints from bound terminal admission rejections.

This external CPU helper changes neither a native observation nor its manifest.
The request-level classification is supplied by the separately reviewed validator.
"""
from copy import deepcopy
import csv
import importlib.util
from pathlib import Path

from pdblend.bench.comparison_campaign import load_bound
from pdblend.bench.single_observation_slo_boundary import read_extension_manifest


def classification_module():
    path=Path(__file__).with_name('explicit_rejection_boundary.py')
    spec=importlib.util.spec_from_file_location('explicit_rejection_boundary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validator(ref, *, load_bound=load_bound):
    return classification_module().validate_explicit_rejection_sidecar(ref,load_bound=load_bound)


def annotate_temporary_csv(path, refs, *, annotate=None):
    """Append derived endpoint columns before the driver's existing metric guard."""
    if not refs:return
    path=Path(path)
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=list(reader.fieldnames or []);before=list(reader)
    annotate=annotate or classification_module().annotate_classified_boundary_rows
    after=annotate(deepcopy(before),refs)
    if len(before)!=len(after) or any(any(new.get(k)!=v for k,v in old.items()) for old,new in zip(before,after)):
        raise ValueError('classified CSV annotation must preserve every existing field and row')
    extra=sorted({key for row in after for key in row}-set(fields))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+extra);writer.writeheader();writer.writerows(after)


def same_history(source_ref, target_ref, *, load=load_bound):
    source=read_extension_manifest(source_ref,load_bound=load)['manifest']
    target=read_extension_manifest(target_ref,load_bound=load)['manifest']
    if any(source.get(k)!=target.get(k) for k in ('policy','run_id','model_id','revision','engine_signature')):
        raise ValueError('classification belongs to another immutable boundary identity')
    for key in ('observations','decisions'):
        old,new=source[key],target[key]
        if new[:len(old)]!=old:
            raise ValueError('classification observation or decision prefix was changed')
    ref=target_ref;seen=set()
    while ref!=source_ref:
        identity=(ref['path'],ref['sha256'])
        if identity in seen:raise ValueError('cyclic classification manifest ancestry')
        seen.add(identity);value=load(ref)
        if value.get('policy')!=source['policy']:
            raise ValueError('classification ancestor changed policy')
        ref=value.get('previous_manifest')
        if not ref:raise ValueError('classification is not bound to an ancestor of the selected head')
    return source,target


def apply_selection(value, refs, campaign, *, validate=None, load=load_bound):
    """Return a new selection. Original manifest fields remain untouched."""
    result=deepcopy(value)
    if not refs:return result
    validate=validate or validator
    active=campaign.get('active_extension_policy_refs',campaign.get('extension_policy_refs',[]))
    applied=[];datasets=set()
    for ref in refs:
        classified=validate(ref,load_bound=load)
        policy=load(classified['policy'])
        if (classified['policy'] not in active or policy['run_id']!=value['run_id']
                or campaign['run_id']!=value['run_id']):
            raise ValueError('classification sidecar is outside the active campaign authority')
        if policy['model_id']!=value['model_id']:
            continue
        if not value.get('manifest'):
            raise ValueError('classification cannot replace an absent selected manifest')
        _,head=same_history(classified['source_head'],value['manifest'],load=load)
        if head['policy']!=classified['policy']:
            raise ValueError('classification policy differs from selected manifest')
        dataset=classified['dataset']
        if dataset in datasets:raise ValueError('multiple sidecars cannot choose the same dataset endpoint')
        datasets.add(dataset)
        lower,upper=classified['lower'],classified['upper']
        if not (0<lower['scale']<upper['scale'] and upper['scale']/lower['scale']<=1.25+1e-12):
            raise ValueError('classified endpoints do not form the frozen 1.25 observed bracket')
        for endpoint,expected_verdict in ((lower,'pass'),(upper,'incomplete')):
            rows=[r for r in head['observations'] if r['dataset']==dataset and r['scale']==endpoint['scale']]
            if (len(rows)!=1 or rows[0]['point']!=endpoint['point']
                    or rows[0]['receipt']!=endpoint['receipt'] or rows[0]['verdict']!=expected_verdict):
                raise ValueError('classified endpoint no longer matches its exact native observation')
            if expected_verdict=='incomplete' and rows[0]['reason']!='incomplete_or_inconsistent_request_timing':
                raise ValueError('execution/drain failures cannot be promoted by admission classification')
        raw=deepcopy(head['boundaries'][dataset])
        if raw.get('status')!='incomplete_observation' or raw.get('passed_lower')!=lower['scale']:
            raise ValueError('classification must use the highest unchanged successful observation')
        # A later observation in this dataset requires its own reviewed sidecar;
        # other datasets may continue while this dataset's native state is frozen.
        if any(r['dataset']==dataset and r['scale']>upper['scale'] for r in head['observations']):
            raise ValueError('classification is stale after a later same-dataset observation')
        result['datasets'][dataset]=dict(raw,status='bracketed',
            passed_lower=lower['scale'],failed_upper=upper['scale'],
            passed_lower_point=lower['point'],passed_lower_receipt=lower['receipt'],
            failed_upper_point=upper['point'],failed_upper_receipt=upper['receipt'],
            lower=lower['point'],upper=upper['point'],saturation_observed=True,
            next_rate_scale=None,relative_width=upper['scale']/lower['scale']-1,
            unique_boundary=not raw.get('nonmonotonic',False),raw_boundary_state=raw,
            classification='explicit_terminal_admission_rejection',classification_sidecars=[ref])
        applied.append(ref)
    if applied:result['classification_sidecars']=applied
    return result


def validate_selection(selection, campaign, *, authorized_refs=None, validate=None, load=load_bound):
    refs=selection.get('classification_sidecars',[])
    if not refs:
        if any(any(key in row for key in ('classification','classification_sidecars','raw_boundary_state'))
               for row in selection.get('datasets',{}).values()):
            raise ValueError('orphan dataset classification lacks top-level sidecar references')
        return selection
    if authorized_refs is not None and any(ref not in authorized_refs for ref in refs):
        raise ValueError('selection sidecar lacks explicit orchestration authorization')
    head=read_extension_manifest(selection['manifest'],load_bound=load)['manifest']
    plain=deepcopy(selection);plain.pop('classification_sidecars',None)
    plain['datasets']={dataset:dict(state,
        unique_boundary=state.get('status')=='bracketed' and not state.get('nonmonotonic'),
        lower=state.get('passed_lower_point') if state.get('status')=='bracketed' else None,
        upper=state.get('failed_upper_point') if state.get('status')=='bracketed' else None)
        for dataset,state in head['boundaries'].items()}
    expected=apply_selection(plain,refs,campaign,validate=validate,load=load)
    if expected!=selection:
        raise ValueError('classified selection differs from bound request evidence and native history')
    return selection
