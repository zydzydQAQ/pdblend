"""Small synthetic evidence regressions; never certify real hardware."""
from copy import deepcopy
import json
import pytest
from ecopadg.serving.evidence import (evaluate,baseline_gaps,sha256,checked_mechanism_collection)
from test_evidence import evidence_fixture


def test_matching_pair_cannot_reduce_the_frozen_output_work(tmp_path):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    for row in rows:row['generated_tokens']=1
    result=evaluate(rows,cells,mechanisms,freeze)
    assert result['verdict']=='evidence_insufficient'
    assert all(v['invalid_pairs'] for v in result['comparisons'].values())


@pytest.mark.parametrize('field,value',[('expected_generated_tokens',None),('expected_generated_tokens',1),
    ('generated_tokens',True),('expected_generated_tokens',True)])
def test_work_fields_are_required_integer_trace_totals(tmp_path,field,value):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    for row in rows:row[field]=value
    assert evaluate(rows,cells,mechanisms,freeze)['verdict']=='evidence_insufficient'


@pytest.mark.parametrize('flag',[False,0,1,'pending','false',None])
def test_mechanism_passed_requires_the_boolean_true(tmp_path,flag):
    _,_,mechanisms,_=evidence_fixture(tmp_path)
    mechanisms=deepcopy(mechanisms)
    mechanisms['distserve']['kv_admission']['passed']=flag
    assert 'kv_admission' in baseline_gaps(mechanisms)['distserve']


def test_stale_registry_digest_cannot_borrow_a_successful_collection(tmp_path):
    _,_,mechanisms,freeze=evidence_fixture(tmp_path)
    links=freeze['formal_evidence'];registry=tmp_path/'mechanisms.json'
    value=json.loads(registry.read_text());value['distserve']['kv_admission']['scope']='changed'
    registry.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='collector'):
        checked_mechanism_collection(value,registry,links['mechanism_collection'])


@pytest.mark.parametrize('defect',['incomplete','missing','component_skipped','registry_link','missing_digest','empty_sources','source_changed'])
def test_collection_requires_actual_completion_and_unchanged_sources(tmp_path,defect):
    _,_,mechanisms,freeze=evidence_fixture(tmp_path)
    links=freeze['formal_evidence'];path=tmp_path/'collection.json';value=json.loads(path.read_text())
    if defect=='incomplete':value['complete']=False
    elif defect=='missing':value['missing']={'dynamollm':['scale_shard_300s']}
    elif defect=='component_skipped':value['components']['dynamo_hardware']['passed']='skipped'
    elif defect=='registry_link':value['registry']=str(tmp_path/'different.json')
    elif defect=='missing_digest':value.pop('registry_sha256')
    elif defect=='empty_sources':value['source_files']={}
    else:(tmp_path/'source').write_text('changed source')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='collector'):
        checked_mechanism_collection(mechanisms,links['mechanisms'],path)


def test_report_cannot_substitute_unfrozen_collection_even_with_valid_content(tmp_path):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    original=tmp_path/'collection.json';copy=tmp_path/'unfrozen.json';copy.write_bytes(original.read_bytes())
    freeze['formal_evidence']['mechanism_collection']=str(copy)
    result=evaluate(rows,cells,mechanisms,freeze)
    assert result['verdict']=='evidence_insufficient'
    assert any('not frozen' in reason for reason in result['reasons'])


def test_report_cannot_substitute_expected_cells_outside_protocol(tmp_path):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    copy=tmp_path/'unfrozen-cells.json';copy.write_text(json.dumps(cells))
    freeze['formal_evidence']['expected_cells']=str(copy)
    assert evaluate(rows,cells,mechanisms,freeze)['verdict']=='evidence_insufficient'


def test_partial_paired_groups_still_cannot_pass_with_new_links(tmp_path):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    rows=[r for r in rows if r['dataset']=='alpaca' and r['load']=='medium']
    result=evaluate(rows,cells,mechanisms,freeze)
    assert result['verdict']=='evidence_insufficient'
    assert all(not v['passed'] for v in result['comparisons'].values())


@pytest.mark.parametrize('output',[None,0,-1,True,64.0])
def test_frozen_trace_requires_real_positive_integer_output_work(tmp_path,output):
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    for trace_path in freeze['groups']['traces']:
        path=__import__('pathlib').Path(trace_path);before=sha256(path);trace=json.loads(path.read_text())
        trace['requests'][0]['output_len']=output;path.write_text(json.dumps(trace));digest=sha256(path)
        freeze['files'][trace_path]=digest
        for row in rows:
            if row['trace_sha256']==before:row['trace_sha256']=digest
    assert evaluate(rows,cells,mechanisms,freeze)['verdict']=='evidence_insufficient'
