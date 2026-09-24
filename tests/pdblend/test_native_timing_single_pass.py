"""Single-pass designs and complete raw replay, CPU only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from pdblend.profile.collection import native_timing_single_pass as single
from pdblend.profile.collection import native_timing_replay as replay
from pdblend.profile.collection.native_timing_plan import digest
from pdblend.profile.collection.native_timing_capacity import partition_windows, fit_measured_partition
from test_native_timing_capacity import plans, capacity_fixture
from test_native_timing_replay import collected, put
from test_native_timing_replay_v2 import rebind_completion
from test_comparison_acceptance import state


def make_plan(tmp_path, plans, model='Qwen2.5-7B-Instruct'):
    return single.build_plan(put(tmp_path/'parent.json', plans['plans'][model]))


def test_single_pass_preserves_all_shapes_and_independent_holdout(tmp_path, plans):
    for model in ('Qwen2.5-7B-Instruct', 'Qwen2.5-14B-Instruct'):
        plan = make_plan(tmp_path, plans, model)
        assert single.validate_plan(plan) == plan
        parent = plans['plans'][model]
        assert plan['points'] == [dict(p, repeats=1) for p in parent['points']]
        assert plan['holdout_limits'] == parent['holdout_limits']
        assert not plan['parallel_qualified'] and not plan['component_qualified']
        assert plan['interference_repeats'] == 0 and not plan['evaluation_used_for_selection']


@pytest.mark.parametrize('change', ['repeat', 'boolean_repeat', 'shape', 'split', 'missing', 'threshold', 'promotion'])
def test_no_point_holdout_or_qualification_relabeling(tmp_path, plans, change):
    plan = make_plan(tmp_path, plans)
    if change == 'repeat': plan['points'][0]['repeats'] = 3
    elif change == 'boolean_repeat': plan['points'][0]['repeats'] = True
    elif change == 'shape': plan['points'][0]['prompt_tokens'] += 1
    elif change == 'split': plan['points'][-1]['purpose'] = 'training'
    elif change == 'missing': plan['points'].pop()
    elif change == 'threshold': plan['holdout_limits']['max_relative_error'] = .9
    else: plan['component_qualified'] = True
    with pytest.raises(ValueError, match='changes'): single.validate_plan(plan)


def test_32b_not_relabelled_and_no_vacuous_parallel_qualification(tmp_path, plans):
    with pytest.raises(ValueError, match='7B/14B'):
        make_plan(tmp_path, plans, 'Qwen2.5-32B-Instruct')
    uuids = [f'GPU-{i}' for i in range(8)]
    q = single.development_qualification([], uuids)
    assert not q['qualified'] and not q['parallel_qualified'] and not q['interference_measured']
    with pytest.raises(ValueError): single.development_qualification([dict(passed=True)], uuids)
    with pytest.raises(ValueError): single.development_qualification([], uuids[:-1]+uuids[:1])


@pytest.fixture
def development(collected, plans):
    x = collected
    plan = make_plan(x.tmp, plans)
    inputs = json.loads(Path(x.inputs_ref['path']).read_text())
    inputs.update(schema=single.INPUT_SCHEMA, timing_first=False,
                  point_plan=put(x.tmp/'single.json', plan))
    x.inputs_ref = put(Path(x.inputs_ref['path']), inputs)
    for name in ('samples', 'interference'): shutil.rmtree(x.root/name)
    uuids = [f'GPU-{i}' for i in range(8)]
    q = single.development_qualification([], uuids)
    qref = put(x.root/'measurement-qualification.json', q)
    windows = []; owners = []; refs = []
    for index, point in enumerate(plan['points']):
        name = f'{digest(point)[:20]}-0'
        # Eight overlapping engines, with nonoverlapping windows per engine.
        raw = x.raw_factory(dict(point, repeat=0), index % 8, 1000.+12.*(index//8), name)
        ref = put(x.root/'samples'/(name+'.json'), raw)
        owner = 'pd-timing-'+str(index % 8)
        refs.append(ref); owners.append(dict(instance_id=owner, raw=ref))
        windows.append(dict(instance_id=owner, raw=raw))
    partition = partition_windows(plan, windows, identities=x.caps)
    fitted = fit_measured_partition(partition, identity=dict(system='pdblend', **x.identity),
        raw_bindings=refs, measurement_qualification=dict(q, receipt=qref), limits=plan['holdout_limits'])
    assert fitted['component'] and not fitted['component_qualified']
    report = json.loads((x.root/'completion.json').read_text())
    end = 1000.+12.*((len(plan['points'])+7)//8)
    report.update(schema=single.COLLECTION_SCHEMA, point_plan=inputs['point_plan'],
        capacity_policy=plan['capacity_policy'], raw_bindings=refs, window_owners=owners,
        measurement_qualification=qref, window_partition=put(x.root/'window-partition.json', partition),
        timing_component=put(x.root/'timing-component.json', fitted), component_qualified=False,
        measured_windows=len(refs), unsupported_windows=0,
        final_drains=[dict(instance_id=iid, received_s=end+1.,
            drain=dict(state(1,end+1.,5), acknowledged=True, drained=True), state=state(1,end+1.,5))
            for iid in x.caps],
        physical_cleanup=dict(passed=True, started_s=end+2., finished_s=end+2.2,
            observations=[dict(at_s=end+2.1, devices=[dict(gpu=i,gpu_uuid=u,compute_pids=[]) for i,u in enumerate(uuids)])]),
        actual_engine_starts={iid:[dict(instance=iid,kind='start',pid=1000+i,t_s=900.)] for i,iid in enumerate(x.caps)},
        engine_loads=8)
    manifest=json.loads((x.attempt/'manifest.json').read_text())
    manifest['payload']['input_manifest']=x.inputs_ref
    put(x.attempt/'manifest.json',manifest)
    queue=json.loads(x.queue.read_text());queue['jobs']['timing-job']['payload']=manifest['payload'];put(x.queue,queue)
    execution=json.loads((x.attempt/'execution.json').read_text());execution['finished_s']=end+3.
    put(x.attempt/'execution.json',execution);rebind_completion(x,report)
    x.plan=plan;x.report=report
    return x


def test_complete_single_pass_parallel_raw_replays_without_qualification(development):
    x=development;ref=replay.capture_evidence(x.attempt,x.queue,x.evidence)
    result=replay.replay_evidence(ref)
    assert result['schema']==single.REPLAY_SCHEMA
    assert result['replayed_windows']==len(x.plan['points'])
    assert result['replayed_interference_windows']==0
    assert not result['component_qualified'] and not result['parallel_qualified']
    assert result['component']['models'] and all(r['qualified'] for r in result['component']['models'])
    relocated=x.tmp/'relocated';shutil.copytree(x.tmp,relocated,ignore=shutil.ignore_patterns('relocated'))
    shutil.rmtree(x.attempt)
    assert replay.replay_evidence(ref,path_map=[(str(x.tmp),str(relocated))])['schema']==single.REPLAY_SCHEMA


def test_rehashing_single_pass_qualification_cannot_promote_component(development):
    x=development;report=deepcopy(x.report)
    q=json.loads(Path(report['measurement_qualification']['path']).read_text());q['qualified']=True
    report['measurement_qualification']=put(Path(report['measurement_qualification']['path']),q)
    rebind_completion(x,report)
    ref=replay.capture_evidence(x.attempt,x.queue,x.evidence)
    with pytest.raises(ValueError,match='qualification does not reproduce'):replay.replay_evidence(ref)
