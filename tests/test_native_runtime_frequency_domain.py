"""Synthetic CPU journals for explicit scoped runtime frequency revisions."""
import asyncio
from copy import deepcopy
import hashlib
import json

import pytest

from pdblend.profile.collection.native_frequency_domain import make_domain,domain_fields,with_domain,validate_collection_inputs
from pdblend.profile.collection.native_runtime_collect import NativeRuntimeCollector,build_runtime_plan,collect_runtime,write_new,RUNTIME_PLAN
from pdblend.profile.collection.native_runtime_audit import replay_runtime
from pdblend.profile.collection.native_timing_plan import digest
from tests.test_native_runtime_collect import artifact,inventory,UUIDS


def domain_ref(tmp_path,model_id='Qwen2.5-7B-Instruct'):
    return write_new(tmp_path/'domain.json',make_domain(model_id=model_id,high_mhz=2100,revision='runtime-test'))


def runtime_fixture(tmp_path,model_id='Qwen2.5-7B-Instruct'):
    # New-domain synthetic observations are created here; production never
    # transforms old raw clocks or labels into new-domain evidence.
    if model_id=='Qwen2.5-32B-Instruct':
        from tests.test_native_runtime_tp2 import tp2_artifact
        report,power,rows=tp2_artifact(tmp_path)
    else:report,power,rows=artifact(tmp_path)
    ref=domain_ref(tmp_path,model_id);plan=build_runtime_plan(ref);domain=plan['frequency_domain'];tp=domain['tp']
    report.update({k:plan[k] for k in ('model_id','tp','pp','frequency_domain_ref','frequency_domain','frequency_domain_sha256')})
    report.update(runtime_plan=plan,runtime_plan_binding=write_new(tmp_path/'domain-runtime-plan.json',plan),
        scope=plan['scope'],handoff_prediction_qualified=False,full_profile_qualified=False)
    identity=dict(system='pdblend',model_id=domain['model_id'],tp=tp,pp=1,model_hash='model',tokenizer_hash='tokenizer',
        source_revision='source',image_digest='image',engine_revision='engine')
    report['component_identity']=with_domain(identity,domain)
    for cap in report['initial_capabilities'].values():cap.update({k:v for k,v in identity.items() if k!='system'})
    def clock(iid,stamp=90):
        gpu=int(iid[1:])*tp;return dict(ack=dict(acknowledged=True,success=True,requested_frequency_mhz=2100,
            gpus=[dict(gpu_uuid=UUIDS[gpu+r]) for r in range(tp)]),observations=[dict(at_s=stamp,frequencies_mhz=[2100]*tp)])
    report['restoration']['clock_mhz']=2100
    for row in report['restoration']['instances']:row['clock']=clock(row['instance_id'])
    for row in rows:
        row.update(frequency_domain_sha256=plan['frequency_domain_sha256'],runtime_plan_sha256=digest(plan))
        if row['kind']=='capacity':row['capability']=report['initial_capabilities'][row['instance_id']]
        if 'state' in row and isinstance(row['state'],str):row['state']=row['state'].replace('2520','2100')
        if 'operation' in row:
            row['operation']=row['operation'].replace('2520','2100')
            if row['operation']=='unpark':row['receipt']['clock']=clock(row['instance_id'],row['started_s'])
            if row['operation']=='wake':
                row['receipt']['clock']=clock(row['instance_id'],row['started_s'])
                row['receipt']['capability']=report['initial_capabilities'][row['instance_id']]
            if row['operation'].startswith('clock_'):
                receipt=row['receipt']
                if receipt.get('ack',{}).get('requested_frequency_mhz')==2520:
                    receipt['ack']['requested_frequency_mhz']=2100
                    for r in receipt['observations']:r['frequencies_mhz']=[2100]*tp
    for _,values in power['frequency_samples']:
        for i,value in enumerate(values):
            if value==2520:values[i]=2100
    power.update(frequency_domain_sha256=plan['frequency_domain_sha256'],runtime_plan_sha256=digest(plan))
    report['power']=write_new(tmp_path/'domain-power.json',power)
    save_rows(tmp_path,report,rows,'domain-journal.jsonl')
    return report,power,rows


def save_rows(tmp_path,report,rows,name):
    p=tmp_path/name;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report['journal']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_new_scoped_runtime_raw_replays_without_handoff_or_full_profile(tmp_path):
    report,_,_=runtime_fixture(tmp_path);audit=replay_runtime(report)
    assert audit['raw_components_complete'],audit['errors']
    assert audit['scoped_runtime_qualified'] and len(audit['capacity'])==8
    assert not audit['handoff_prediction_qualified'] and not audit['full_profile_qualified'] and not audit['formal_eligible']
    assert any(r['component']=='clock_1500_to_2100' for r in audit['holdout_comparisons'])
    assert not any('2520' in r['component'] or 'transfer' in r['component'] for r in audit['holdout_comparisons'])
    assert report['runtime_plan']['holdout_limits']==RUNTIME_PLAN['holdout_limits']


@pytest.mark.parametrize('fault',['missing_domain','wrong_row_domain','wrong_plan','old_restore','old_power',
    'old_static_clock','handoff_claim','transfer_row','mixed_model','missing_wake_clock'])
def test_new_domain_raw_cannot_mix_legacy_or_change_physical_evidence(tmp_path,fault):
    report,power,rows=runtime_fixture(tmp_path)
    if fault=='missing_domain':report.pop('frequency_domain_ref')
    elif fault=='wrong_row_domain':rows[-1]['frequency_domain_sha256']='old'
    elif fault=='wrong_plan':rows[-1]['runtime_plan_sha256']='old'
    elif fault=='old_restore':report['restoration']['instances'][0]['clock']['ack']['requested_frequency_mhz']=2520
    elif fault=='old_power':power.pop('frequency_domain_sha256')
    elif fault=='old_static_clock':
        row=next(r for r in rows if r.get('state')=='active_idle@2100')
        for t,values in power['frequency_samples']:
            if row['started_s']<=t<row['finished_s']:values[0]=2520
    elif fault=='handoff_claim':report['handoff_prediction_qualified']=True
    elif fault=='transfer_row':rows.append(dict(sequence=len(rows),kind='transfer',frequency_domain_sha256=report['frequency_domain_sha256'],runtime_plan_sha256=digest(report['runtime_plan'])))
    elif fault=='mixed_model':report['initial_capabilities']['i1']['model_hash']='other'
    else:next(r for r in rows if r.get('operation')=='wake')['receipt'].pop('clock')
    report['power']=write_new(tmp_path/'mutated-power.json',power);save_rows(tmp_path,report,rows,'mutated-journal.jsonl')
    audit=replay_runtime(report);assert not audit['raw_components_complete'] and audit['errors']


def test_plan_and_invocation_require_explicit_scoped_runtime_binding(tmp_path):
    ref=domain_ref(tmp_path);runtime=build_runtime_plan(ref)
    assert build_runtime_plan()==RUNTIME_PLAN and runtime['transfer_input_tokens']==[]
    plan=dict(model_id=runtime['model_id'],tp=1,pp=1,frequency_domain_ref=ref,**domain_fields(runtime['frequency_domain']))
    inputs=dict(plan,timing_first=True,runtime_include_transfer=False,runtime_plan=runtime,runtime_scope=runtime['scope'])
    assert validate_collection_inputs(plan,inputs,collect_runtime=True)==(1500,2100)
    for key in ('runtime_include_transfer','runtime_plan','runtime_scope'):
        bad=deepcopy(inputs);bad.pop(key)
        with pytest.raises(ValueError):validate_collection_inputs(plan,bad,collect_runtime=True)
    with pytest.raises(ValueError):validate_collection_inputs(plan,inputs,collect_runtime=True,request_cycles=True)
    specs,fleet,meter,sampler=inventory()
    with pytest.raises(ValueError,match='include_transfer=False'):
        asyncio.run(collect_runtime(specs,fleet,meter,sampler,tmp_path/'not-created',gpu_uuids=UUIDS,frequency_domain_ref=ref))
    assert not (tmp_path/'not-created').exists()


def test_native_collector_records_domain_and_rejects_diagnostic_transfer(tmp_path):
    ref=domain_ref(tmp_path);specs,fleet,meter,sampler=inventory()
    runner=NativeRuntimeCollector(specs,fleet,meter,sampler,tmp_path,gpu_uuids=UUIDS,frequency_domain_ref=ref)
    assert runner.frequencies==(1500,2100) and runner.restore_frequency==2100
    row=runner.journal(dict(kind='capacity',instance_id='i0'))
    assert row['frequency_domain_sha256']==runner.runtime_plan['frequency_domain_sha256']
    with pytest.raises(ValueError,match='diagnostic only'):asyncio.run(runner.transfers(specs[0],specs[1]))
