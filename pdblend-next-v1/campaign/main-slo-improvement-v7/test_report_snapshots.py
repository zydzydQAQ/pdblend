import json
import pytest
from report import execution_statuses

def save(root, name, **changes):
    path = root/name/'status.json'; path.parent.mkdir(parents=True)
    value = dict(model='32b', stage='screen_fixed2', pid=123, started_s=100.,
                 updated_s=110., attempted=['a'], completed=[], failed=[])
    value.update(changes); path.write_text(json.dumps(value))
    return path

def test_mirrored_progress_is_one_invocation(tmp_path):
    save(tmp_path, 'progress-001')
    latest = save(tmp_path, 'progress-002', updated_s=120., attempted=['a','b'], completed=['a'])
    values = execution_statuses([tmp_path])
    assert len(values) == 1 and values[0][0] == latest
    assert values[0][1]['completed'] == ['a']

def test_distinct_restart_is_not_collapsed(tmp_path):
    save(tmp_path, 'first')
    save(tmp_path, 'second', pid=124, started_s=200., updated_s=210.)
    assert len(execution_statuses([tmp_path])) == 2

def test_failure_cannot_disappear_from_newer_snapshot(tmp_path):
    save(tmp_path, 'old', failed=[dict(cell_id='a', error='measured failure')])
    save(tmp_path, 'new', updated_s=120., completed=['a'])
    with pytest.raises(ValueError, match='erased earlier work or failure'):
        execution_statuses([tmp_path])

def test_progress_cannot_erase_completed_observation(tmp_path):
    save(tmp_path, 'old', completed=['a'])
    save(tmp_path, 'new', updated_s=120.)
    with pytest.raises(ValueError, match='erased earlier work or failure'):
        execution_statuses([tmp_path])
