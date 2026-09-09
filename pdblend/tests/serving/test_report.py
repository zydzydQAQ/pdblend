import json
import pytest

from ecopadg.serving.report import collect,write_report


def test_development_energy_is_reported_but_cannot_satisfy_formal_evidence(tmp_path):
    path=tmp_path/'run.json'
    path.write_text(json.dumps(dict(system='pdblend',split='development',energy_j=1,
        slo_attainment=1,completed=500,n_expected=500,validity='ok')))
    result=collect(dict(summaries=[str(path)]))
    assert result['verdict']['verdict']=='evidence_insufficient'
    assert result['rows'][0]['energy_j']==1
    out=tmp_path/'report';write_report(result,out)
    assert '证据不足' in (out/'report.md').read_text()
    assert (out/'points.csv').exists()
    with pytest.raises(ValueError,match='duplicate'): collect(dict(summaries=[str(path),str(path)]))


def formal_report_fixture(tmp_path):
    from test_evidence import evidence_fixture
    rows,cells,mechanisms,freeze=evidence_fixture(tmp_path)
    freeze_path=tmp_path/'freeze.json';freeze_path.write_text(json.dumps(freeze))
    paths=[]
    for index,row in enumerate(rows):
        path=tmp_path/('run-'+str(index)+'.json');path.write_text(json.dumps(row));paths.append(str(path))
    return dict(summaries=paths,expected_cells=freeze['formal_evidence']['expected_cells'],
        mechanisms=freeze['formal_evidence']['mechanisms'],freeze=str(freeze_path))


def test_full_explicit_report_uses_frozen_collection_and_prescribed_work(tmp_path):
    result=collect(formal_report_fixture(tmp_path))
    assert result['verdict']['verdict']=='target_achieved'
    assert len(result['rows'])==180
    write_report(result,tmp_path/'output')
    assert '达到目标' in (tmp_path/'output/report.md').read_text()


def test_report_matching_incomplete_output_cannot_claim_a_global_win(tmp_path):
    from pathlib import Path
    manifest=formal_report_fixture(tmp_path)
    for name in manifest['summaries']:
        path=Path(name);value=json.loads(path.read_text());value['generated_tokens']=1
        path.write_text(json.dumps(value))
    result=collect(manifest)
    assert result['verdict']['verdict']=='evidence_insufficient'
    assert all(len(c['invalid_pairs'])==30 for c in result['verdict']['comparisons'].values())


def test_report_cannot_replace_the_collected_registry_with_another_valid_looking_one(tmp_path):
    from pathlib import Path
    manifest=formal_report_fixture(tmp_path)
    value=json.loads(Path(manifest['mechanisms']).read_text())
    value['distserve']['kv_admission']['scope']='uncollected replacement'
    path=tmp_path/'replacement.json';path.write_text(json.dumps(value));manifest['mechanisms']=str(path)
    result=collect(manifest)
    assert result['verdict']['verdict']=='evidence_insufficient'
    assert any('collector' in reason for reason in result['verdict']['reasons'])
