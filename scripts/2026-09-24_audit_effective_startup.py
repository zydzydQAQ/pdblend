#!/usr/bin/env python3
"""Read-only, receipt-bound startup audit outside the algorithm source inventory.

Standard points use their precomputed contract. Adaptive points recompute from
their own frozen choice, Profile and independent planning trace, never outputs.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path


def builder():
    path=Path(__file__).with_name('2026-09-24_prepare_saturation_round.py')
    spec=importlib.util.spec_from_file_location('external_startup_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fresh_contract(module,point,choice,config):
    ref,frozen=module.verify_source(point['source_manifest']['path'])
    if ref!=point['source_manifest']:raise ValueError('actual runtime source binding differs')
    module.activate_source(ref)
    from pdblend.bench.pdblend_observation_plan import decode_plan
    from pdblend.bench.independent_dispatch import request_rows
    from pdblend.profile.query.versions import load_profile
    helper_ref=config['startup_helper'];module.load_bound(point['inputs']['planning_trace'])
    if module.binding(helper_ref['path'])!=helper_ref:raise ValueError('startup helper binding differs')
    spec=importlib.util.spec_from_file_location('pdblend.bench.comparison_startup',helper_ref['path'])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    planning=module.load_bound(point['inputs']['planning_trace'])
    if planning.get('selection_split') not in ('calibration','tuning'):
        raise ValueError('startup audit cannot select from evaluation input')
    plan=decode_plan(choice,observation=True);profile=config['profile'];module.load_bound(profile)
    loaded=load_profile(profile['path'],system='pdblend',model_id=point['model_id'],
                        tp=plan.tp,pp=plan.pp,usage='development')
    return helper.startup_contract(point,loaded.model,plan,config['pdblend_runtime'],request_rows(planning),
        profile=profile,planning_trace=point['inputs']['planning_trace'])


def signature(event):
    identity=event['plan_identity']
    return dict(counts=event['counts'],f_P=event['f_P'],f_D=event['f_D'],f_M=event['f_M'],
                tau=event['tau'],tp=identity['tp'],pp=identity['pp'],profile_key=identity['profile_key'])


def control_keys(source_ref):
    """Read the selected source's literal contract keys without importing GPUs."""
    path=Path(source_ref['path']).parent/'pdblend/bench/pdblend_runtime_options.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROL_OPTIONS' for t in node.targets):
            keys=ast.literal_eval(node.value)
            if isinstance(keys,(list,tuple)) and all(isinstance(k,str) for k in keys):return keys
    raise ValueError('selected source does not expose literal control option keys')


def audit(receipt_path):
    m=builder();ref=m.binding(receipt_path);receipt=m.load_bound(ref);window=Path(ref['path']).parent
    point_ref=m.binding(window/'point.json')
    if receipt.get('artifacts',{}).get('point.json')!=point_ref['sha256']:
        raise ValueError('receipt point artifact is not bound')
    point=m.load_bound(point_ref)
    if receipt.get('point_sha256')!=m.digest(point):raise ValueError('point identity differs')
    source_ref,frozen=m.verify_source(point['source_manifest']['path'])
    if source_ref!=point['source_manifest']:raise ValueError('runtime source binding differs')
    result_ref=m.binding(window/'result.json');result=m.load_bound(result_ref)
    if (receipt.get('artifacts',{}).get('result.json')!=result_ref['sha256']
            or result!=receipt.get('result')
            or result.get('identity',{}).get('source_sha256')!=frozen['source_sha256']):
        raise ValueError('result does not bind the actual selected source')
    choice=m.load_bound(point['inputs']['offline_choice']);config=m.load_bound(point['inputs']['system_config'])
    if (config.get('startup_contract_mode')!='external_audit'
            or choice.get('startup_contract_mode')!='external_audit'):
        raise ValueError('point is not in external audit mode')
    options=config['pdblend_runtime']
    if choice.get('runtime_options')!=options:
        raise ValueError('choice and config runtime options differ')
    if choice.get('startup_helper')!=config['startup_helper'] or m.binding(config['startup_helper']['path'])!=config['startup_helper']:
        raise ValueError('startup helper identity differs')
    contract=choice.get('startup_contract');recomputed=contract is None
    if contract is None:
        if choice.get('startup_contract_pending')!='recompute_from_generated_point_independent_tuning':
            raise ValueError('missing explicit independent extension startup audit requirement')
        contract=fresh_contract(m,point,choice,config)
    if (contract['profile_sha256']!=config['profile']['sha256']
            or contract['planning_trace_sha256']!=point['inputs']['planning_trace']['sha256']
            or contract.get('control_options_sha256')!=m.digest({k:options[k] for k in control_keys(source_ref)})):
        raise ValueError('startup contract inherited another point Profile or planning trace')
    if not recomputed and (contract.get('artifact_bindings_sha256')!=m.digest({k:options.get(k) for k in (
            'capacity_floor_path','transition_catalog_path','incremental_energy_path')})
            or contract.get('capacity_workload_binding_sha256')!=m.digest(options.get('capacity_workload_binding'))):
        raise ValueError('startup contract artifact or capacity workload binding differs')
    controller_ref=m.binding(window/'run/controller.jsonl')
    if receipt.get('artifacts',{}).get('run/controller.jsonl')!=controller_ref['sha256']:
        raise ValueError('controller events not bound by receipt')
    events=[json.loads(line) for line in Path(controller_ref['path']).read_text().splitlines()]
    first=next((e for e in events if e.get('kind')=='plan'),None)
    if first is None:raise ValueError('receipt has no executed first plan')
    actual=signature(first)
    native_ref=m.binding(window/'run/native-result.json');native=m.load_bound(native_ref)
    if receipt.get('artifacts',{}).get('run/native-result.json')!=native_ref['sha256']:
        raise ValueError('native service readiness evidence not bound')
    ready=native.get('initial_plan_ready') or {};service=native.get('service_started_s')
    transition=next((e for e in events if e.get('kind')=='transition_complete'
                     and e.get('transition_id')==ready.get('transition_id')),None)
    times=[service,ready.get('ready_s'),ready.get('transition_finished_s'),first.get('t')]
    physical_ready=bool(transition and all(type(t) in (int,float) and math.isfinite(t) for t in times)
        and ready.get('evidence')=='controller_completed_physical_transition'
        and ready.get('transition_finished_s')==transition['finished_s']
        and first['t']<=transition['finished_s']<=ready['ready_s']<=service
        and transition['started_s']<=first['t'])
    return dict(schema='external-effective-startup-audit/v1',receipt=ref,point=point_ref,
        choice=point['inputs']['offline_choice'],config=point['inputs']['system_config'],controller=controller_ref,
        source_manifest=point['source_manifest'],helper=config['startup_helper'],
        contract_recomputed_from_independent_inputs=recomputed,contract_sha256=m.digest(contract),
        expected_first_plan=contract['expected_first_plan'],actual_first_plan=actual,
        logical_plan_matches=actual==contract['expected_first_plan'],physical_transition_ready_before_service=physical_ready,
        matches=actual==contract['expected_first_plan'] and physical_ready,evaluation_outputs_used_for_selection=False,
        native_result=native_ref,initial_plan_ready=ready,service_started_s=service,
        first_plan_at_s=first['t'],audit_scope='first logical plan plus bound native initial transition readiness; actual SM frequency is separate evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();result=audit(args.receipt);print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['matches'] else 2)


if __name__=='__main__':main()
