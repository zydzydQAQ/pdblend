"""Separate four-TP2 receipts keep the frozen TP1 path unchanged."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pdblend.bench import comparison_ecoserve32_inputs as inputs32
from pdblend.bench.comparison_ecoserve32_acceptance import audit_ecoserve_window
from pdblend.bench.resident_session import digest, engine_signature
from pdblend_baselines.native_profile import _sha
from test_comparison_ecoserve import eco_fixture, put, save_change, state
from test_comparison_acceptance import drain_rows


def fixture(tmp_path, monkeypatch, *, slow=False):
    args=eco_fixture(tmp_path,monkeypatch,slow=slow);point=args['point'];identity=args['engine_identity']
    model='Qwen2.5-32B-Instruct';point['model_id']=model
    identity['instances']=identity['instances'][:4]
    for index,row in enumerate(identity['instances']):
        row.update(tp=2,gpu_uuids=identity['fleet_gpu_uuids'][index*2:index*2+2])
    specs=identity['instances'];startup=args['startup_qualification']
    startup['engine_signature']=engine_signature(identity)
    startup['actual_launch_identity']['instances']=startup['actual_launch_identity']['instances'][:4]
    from pdblend_runtime.probe import NativeSpec
    for index,(launch,spec) in enumerate(zip(startup['actual_launch_identity']['instances'],specs)):
        launch.update(gpu_uuids=spec['gpu_uuids'],argv=NativeSpec(spec['instance_id'],(index*2,index*2+1),
            20000+index*4,'/models/'+model,tp=2,pp=1,**spec['launch_options']).command())
        launch['environment']['CUDA_VISIBLE_DEVICES']=f'{index*2},{index*2+1}'
    startup['capabilities']={s['instance_id']:dict(startup['capabilities'][s['instance_id']],
        model_id=model,tp=2,gpu_uuids=s['gpu_uuids'],state=state(2,91.,4)) for s in specs}
    for cap in startup['capabilities'].values():
        for rank in cap['state']['ranks']:rank['retained_kv_supported']=True
    startup['ordinary_reference']=startup['ordinary_reference'][:4]
    startup['drain']=drain_rows(specs,92.,4)
    reset=args['reset'];reset.update(drain=drain_rows(specs,98.,4),
        generation={s['instance_id']:5 for s in specs},
        reopen_ack={s['instance_id']:dict(acknowledged=True,generation=5) for s in specs},
        reopen_state={s['instance_id']:state(2,99.,5) for s in specs})
    args['drain']['states']=drain_rows(specs,250.05,5)
    trace=json.loads(Path(point['trace']['path']).read_text());trace['model_id']=model
    point['trace']=args['raw_refs']['trace']=put(tmp_path,'trace.json',trace)
    inputs=point['inputs'];inputs['trace']=point['trace']
    config=json.loads(Path(inputs['system_config']['path']).read_text())
    config.update(model_id=model,eco_initial_instances=4,instances=[dict(id=s['instance_id'],gpus=[i*2,i*2+1],tp=2,pp=1) for i,s in enumerate(specs)])
    inputs['system_config']=put(tmp_path,'config.json',config)
    profile=json.loads(Path(inputs['eco_profile_manifest']['path']).read_text())
    profile['model']=model;profile['metadata']['tp']=2;profile['metadata']['gpu_uuids']=specs[0]['gpu_uuids']
    for row in profile['rows']:
        for repetition in row['repetitions']:
            other=deepcopy(repetition['ranks'][0]);other['rank']=1;repetition['ranks'].append(other)
        row['samples_sha256']=[_sha(r) for r in row['repetitions']]
    inputs['eco_profile_manifest']=put(tmp_path,'eco.csv.manifest.json',profile)
    mechanism=json.loads(Path(inputs['eco_mechanism_completion']['path']).read_text())
    mechanism['capabilities']=startup['capabilities']
    inputs['eco_mechanism_completion']=put(tmp_path,'mechanism.json',mechanism)
    source=json.loads(Path(startup['source_manifest']['path']).read_text())
    monkeypatch.setattr(inputs32,'PROFILE_SOURCE',source['source_sha256'])
    monkeypatch.setattr(inputs32,'MECHANISM_SOURCE',source['source_sha256'])
    monkeypatch.setattr(inputs32,'PROTECTED',('pdblend_runtime/serve.py',))
    monkeypatch.setattr(inputs32,'REVIEWED_WRAPPERS',{})
    monkeypatch.setattr(inputs32,'CERTIFIED',{model:{k:inputs[k]['sha256'] for k in (
        'eco_profile_csv','eco_profile_manifest','eco_mechanism_completion','eco_automatic_completion','eco_mechanism_review')}})
    events=[json.loads(r) for r in Path(args['raw_refs']['events']['path']).read_text().splitlines()]
    events=[r for r in events if r['kind']!='eco_http_receipt']
    clocks=[dict(kind='eco_http_receipt',instance_id=s['instance_id'],method='POST',path='/baseline/clock',
        at_s=99.01+i*.01,body=dict(frequency_mhz=2520),response=dict(acknowledged=True,success=True,
            requested_frequency_mhz=2520,gpus=[dict(gpu_uuid=u,frequency_mhz=2520) for u in s['gpu_uuids']])) for i,s in enumerate(specs)]
    events=clocks+events
    next(r for r in events if r['kind']=='eco_startup')['groups']=[['mixed0','mixed1'],['mixed2','mixed3']]
    for r in events:
        if r['kind']=='eco_scale_observation':r.update(available_instances=[],groups=[['mixed0','mixed1'],['mixed2','mixed3']])
    args['raw_refs']['events']=put(tmp_path,'eco-events.jsonl',events,journal=True)
    native=args['native_result'];native.update(model_id=model,trace_sha256=point['trace']['sha256'],
        events_sha256=args['raw_refs']['events']['sha256'],journal_rows=len(events),
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        initial_native_states={s['instance_id']:state(2,99.9,5) for s in specs},
        drain_receipts={r['instance_id']:r['drain'] for r in args['drain']['states']})
    power=json.loads(Path(args['raw_refs']['power']['path']).read_text())
    for stamp,values in power['frequency_samples']:values[:]=[2520]*8
    save_change(args,'power',power)
    for key in ('startup_qualification','reset','drain','native_result'):save_change(args,key,args[key])
    return args


@pytest.mark.parametrize('slow',[False,True])
def test_four_tp2_complete_window_uses_both_ranks_and_all_boards(tmp_path,monkeypatch,slow):
    args=fixture(tmp_path,monkeypatch,slow=slow)
    result=inputs32.validate_ecoserve_inputs(args['point'],args['engine_identity'])
    assert result['preflight_ready'],result['gate_failures']
    result=audit_ecoserve_window(**args)
    assert result['formal_eligible'],result['gate_failures']
    assert result['slo_pass'] is not slow


@pytest.mark.parametrize('change',['rank','tp','initial3','missing_board','launch_tp','profile_tp','clock_second_rank','launch_memory'])
def test_32b_checks_cannot_be_satisfied_by_tp1_or_probe_initial_layout(tmp_path,monkeypatch,change):
    args=fixture(tmp_path,monkeypatch)
    if change=='rank':
        args['native_result']['drain_receipts']['mixed0']['ranks'].pop()
        save_change(args,'native_result',args['native_result'])
    elif change=='tp':args['engine_identity']['instances'][0]['tp']=1
    elif change=='missing_board':args['engine_identity']['instances'][0]['gpu_uuids'].pop()
    elif change in ('launch_tp','launch_memory'):
        argv=args['startup_qualification']['actual_launch_identity']['instances'][0]['argv']
        argv[argv.index('--tensor-parallel-size' if change=='launch_tp' else '--gpu-memory-utilization')+1]='1' if change=='launch_tp' else '.55'
        save_change(args,'startup_qualification',args['startup_qualification'])
    elif change=='clock_second_rank':
        power=json.loads(Path(args['raw_refs']['power']['path']).read_text());power['frequency_samples'][15][1][1]=900
        save_change(args,'power',power)
    else:
        key='system_config' if change=='initial3' else 'eco_profile_manifest';ref=args['point']['inputs'][key]
        value=json.loads(Path(ref['path']).read_text())
        if change=='initial3':value['eco_initial_instances']=3
        else:value['metadata']['tp']=1
        args['point']['inputs'][key]=put(tmp_path,Path(ref['path']).name,value)
    result=audit_ecoserve_window(**args)
    assert not result['formal_eligible']


def test_32b_catalog_is_separate_from_frozen_tp1_catalog():
    from pdblend.bench.comparison_ecoserve_inputs import CERTIFIED as frozen
    assert 'Qwen2.5-32B-Instruct' not in frozen
    assert set(inputs32.CERTIFIED)=={'Qwen2.5-32B-Instruct'}
