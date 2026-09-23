import copy

import pytest

from pdblend.profile import short_domain as sd, short_version as sv


def measured():
    plan=sd.make_plan(dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',model_hash='m',tokenizer_hash='t',tp=1,pp=1))
    result={}
    for phase in ('training','holdout'):
        result[phase]={}
        for p in plan[phase]:
            n=p['input_tokens'];values=dict(seconds=.05+n*.0001)
            if p['role']=='short_mixed':values.update(seconds=1+n*.001,power_w=100+n*.02,energy_j=100+n*.03)
            result[phase][sd.point_key(p)]=dict(point=p,repeats=[dict(values) for _ in range(p['repeats'])])
    return plan,result


def test_complete_request_query_scope_rejects_decode_and_unmeasured_work():
    plan,rows=measured();c=sv.fit_direct(plan,rows['training']);m=sv.ShortComponentModel(c,'test')
    query=dict(role='mixed',frequency=2100,input_tokens=80,output_tokens=64,batch=1,execution=sv.EXECUTION,usage='experimental')
    answer=m.query(**query)
    assert answer['seconds']==pytest.approx(1.08)
    assert answer['energy_j']==pytest.approx(102.4)
    assert answer['role']=='mixed' and not answer['pure_decode_power_qualified'] and not answer['formal_eligible']
    for changes in ({'input_tokens':28},{'input_tokens':225},{'output_tokens':32},{'batch':2},{'role':'decode'},
                    {'frequency':2000},{'execution':'poisson_requests'},{'usage':'formal'}):
        with pytest.raises(sv.MissingShortProfile):m.query(**(query|changes))
    pre=m.query(**(query|dict(role='prefill_timing',output_tokens=1,input_tokens=128)))
    assert pre['seconds']==pytest.approx(.0628) and 'power_w' not in pre and 'energy_j' not in pre
    with pytest.raises(sv.MissingShortProfile):m.query(**(query|dict(role='prefill_timing',output_tokens=1,input_tokens=129)))


def test_independent_holdout_checks_each_mixed_metric_without_refitting():
    plan,rows=measured();candidate=sv.fit_direct(plan,rows['training']);before=copy.deepcopy(candidate)
    assert sv.audit_direct(candidate,plan,rows['holdout'])['passed']
    row=next(row for row in rows['holdout'].values() if row['point']['role']=='short_mixed')
    row['repeats'][0]['energy_j']*=1.2
    audit=sv.audit_direct(candidate,plan,rows['holdout'])
    assert not audit['passed'] and audit['failures'][0]['metric']=='energy_j'
    assert audit['maximum_error']>.10 and candidate==before
    del rows['holdout'][next(iter(rows['holdout']))]
    with pytest.raises(ValueError,match='exact independent'):sv.audit_direct(candidate,plan,rows['holdout'])


def test_mixed_energy_includes_entire_window_and_host_gaps(monkeypatch):
    monkeypatch.setattr(sd,'summarize',lambda evidence:dict(mixed_power_w=100))
    evidence=dict(point={'role':'short_mixed'},start_s=0,end_s=10,
        power=[(2,[100]),(8,[200])],requests=[{},{}])
    row=sv.mixed_observation(evidence)
    assert row['seconds']==5 and row['energy_j']==750 and row['power_w']==150


@pytest.mark.asyncio
@pytest.mark.historical
async def test_published_component_reconstructs_raw_and_rejects_tampering(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from pdblend.profile import short_domain_collect as collect
    # Reuse the CPU protocol clock/client used to test the actual collector.
    spec=importlib.util.spec_from_file_location('short_collector_protocol_fixtures',Path(__file__).with_name('test_short_domain_collect.py'))
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    root=Path(__file__).resolve().parents[2]
    audit=json.loads((root/'results/2026-09-23/incremental-wave-closeout-v1/audit.json').read_text())
    training=Path(audit['members']['7b-tp1-longctx']['root']);package=tmp_path/'package'
    collect.prepare(base_candidate=root/'results/2026-09-22/three-model/calibration-candidates/7b-tp1-4ddc49563cef6321c0b5/candidate.json',
        dataset_manifest=root/'datasets/prepared/2026-09-22-7b-v1/manifest.json',identity_raw=training/'raw.json',out=package)
    clock=fixtures.Clock();p=fixtures.profiler(tmp_path/'parent',training,clock);client=fixtures.Client(clock)
    layout={'a':['GPU-a'],'b':['GPU-b']};qualifier=p.out_dir/'qualifier.json'
    qualifier.write_text(json.dumps(dict(complete=True,cross_job=True,passed=True,cohort_id='test-epoch',members=['a','b'],
        isolated=[dict(gpu_uuids=layout[k]) for k in layout],parallel=[dict(gpu_uuids=layout[k]) for k in layout])))
    def guard():return dict(epoch_id='test-epoch',qualification_path=str(qualifier),qualification_sha256=collect.digest(qualifier),
        layout=layout,layout_sha256=sv.sha256_value(layout))
    actual=collect.measure_repeat
    async def measure(*args,**kwargs):return await actual(*args,**kwargs,_clock=clock)
    monkeypatch.setattr(collect,'measure_repeat',measure)
    archive=tmp_path/'archive'
    result=await collect.run_existing(package=package,profiler=p,client=client,gpus=[0],out=archive,qualification_guard=guard)
    assert result['complete'] and result['experimental_components_passed']
    version=sv.publish(package=package,archive=archive,out=tmp_path/'version')
    identity=dict(system='pdblend',model_id='Qwen2.5-7B-Instruct',tp=1,pp=1,usage='experimental')
    model=sv.load(tmp_path/'version/version.json',**identity)
    assert model.version_id==version['version_id'] and not version['formal_eligible']
    with pytest.raises(sv.MissingShortProfile):sv.load(tmp_path/'version/version.json',**(identity|dict(usage='formal')))
    raw=json.loads((archive/'raw.json').read_text());row=next(iter(raw['training'].values()))
    (archive/row['repeats'][0]['samples_file']).write_text('{}')
    with pytest.raises(ValueError,match='raw sample changed'):sv.load(tmp_path/'version/version.json',**identity)
