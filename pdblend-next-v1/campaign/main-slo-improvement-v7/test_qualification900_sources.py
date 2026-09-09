import json
import pytest
import verify_qualification900 as v


def fixture(tmp_path):
    host=tmp_path/'host';host.mkdir();(host/'manifest.json').write_text('{}')
    run=tmp_path/'run';(run/'qualification900').mkdir(parents=True)
    rows=run/'qualification900/requests.json';rows.write_text('[]')
    source=dict(original_binding={'path':'/original','sha256':'x'},
        capacity_binding={'path':'/capacity','sha256':'x'},config={'path':'/config','sha256':'x'},
        host_manifest=v.p.ref(host/'manifest.json'))
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(
        original_binding=source['original_binding'],capacity_binding=source['capacity_binding'],
        config=source['config'],host_release=str(host),files={str(host/'manifest.json'):v.p.sha(host/'manifest.json')})))
    (run/'spec-reference.json').write_text(json.dumps(v.p.ref(spec)))
    return spec,run,dict(source=source,artifacts={str(rows):v.p.sha(rows)})


def test_actual_frozen_invocation_and_rows(tmp_path):
    spec,run,result=fixture(tmp_path)
    v.actual_source(spec,run,result)


def test_wrong_source_rejected_even_matching_shape(tmp_path):
    spec,run,result=fixture(tmp_path);result['source']['config']={'path':'/other','sha256':'x'}
    with pytest.raises(ValueError,match='source differs'):v.actual_source(spec,run,result)


def test_changed_rows_rejected(tmp_path):
    spec,run,result=fixture(tmp_path);(run/'qualification900/requests.json').write_text('[{}]')
    with pytest.raises(ValueError,match='artifacts changed'):v.actual_source(spec,run,result)


def test_another_actual_invocation_rejected(tmp_path):
    spec,run,result=fixture(tmp_path);(run/'spec-reference.json').write_text('{}')
    with pytest.raises(ValueError,match='another frozen specification'):v.actual_source(spec,run,result)
