import json
from pathlib import Path
import pytest
import protocol as p
from prepare_dynamic_release import certificate_inputs

def put(path, value):
    path.write_text(json.dumps(value));return p.ref(path)

def test_retain_nested_actual_requests_trace_inventory_and_raw(tmp_path):
    raw=tmp_path/'power.csv';raw.write_text('actual samples')
    requests=tmp_path/'requests.json';requests.write_text('[]')
    trace=put(tmp_path/'trace.json', {'requests':[]})
    inventory=put(tmp_path/'inventory.json', {'complete':True})
    measurement=put(tmp_path/'measurement.json', {'artifacts':{str(raw):p.sha(raw)}})
    result=put(tmp_path/'result.json', {'raw_measurement':measurement, 'trace':trace,
                                      'artifacts':{str(requests):p.sha(requests)}})
    group=put(tmp_path/'group.json', {'members':[{'result':result,'inventory':inventory}]})
    cert=put(tmp_path/'certificate.json', {'evidence_groups':[group],'raw_measurements':[measurement]})
    files=certificate_inputs(cert)
    assert set(files)=={str(q) for q in tmp_path.iterdir()}

def test_changed_group_dependency_blocks_release(tmp_path):
    trace=put(tmp_path/'trace.json', {'requests':[]})
    cert=put(tmp_path/'certificate.json', {'trace':trace})
    Path(trace['path']).write_text('{"changed":true}')
    with pytest.raises(ValueError,match='calibration source changed'):
        certificate_inputs(cert)
