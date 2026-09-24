from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.bench.comparison_campaign import binding
from pdblend.bench.comparison_runtime import pdblend_window_resources, read_native_measurement
from pdblend.planner.pool import Plan
from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.versions import VersionError
from pdblend.results.journal import CompactJournal


FIXTURE = Path(__file__).parent / 'fixtures' / 'legacy_profile.json'


def prepared(tmp_path, *, qualified):
    data = json.loads(FIXTURE.read_text())
    # Real serialized numerical fixture; qualification is synthetic only in
    # this isolated CPU test. It is never written into a campaign artifact.
    data['quality'] = dict(formal_eligible=qualified)
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(data))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', profile='profile.json')))
    choice = tmp_path / 'choice.json'
    choice.write_text(json.dumps(dict(system='pdblend', model_id='Qwen2.5-7B-Instruct',
        selection_split='tuning', evaluation_used_for_selection=False,
        profile_sha256=binding(profile)['sha256'],
        plan=asdict(Plan({'M': 2}, 2520, 2520, 2520, 0, 100, 1, .1, tp=1, pp=1)))))
    point = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct',
        inputs=dict(system_config=binding(config), profiles=[binding(profile)], offline_choice=binding(choice)))
    specs = [SimpleNamespace(tp=1, pp=1, generation=5) for _ in range(2)]
    return point, specs


def change_choice(point, change):
    path = Path(point['inputs']['offline_choice']['path'])
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    point['inputs']['offline_choice'] = binding(path)


def test_real_profile_deserializes_to_model_and_explicit_plan_binds_actual_epoch(tmp_path):
    point, specs = prepared(tmp_path, qualified=True)
    loaded, plan = pdblend_window_resources(point, specs)
    expected = PerfModel.from_json(FIXTURE.read_text())
    assert loaded.model.step_seconds(8, 512, 2520) == expected.step_seconds(8, 512, 2520)
    assert isinstance(plan, Plan) and plan.generation == 5 and plan.tp == 1
    assert json.loads(plan.profile_key) == loaded.profile_key
    assert loaded.qualification['usage'] == 'formal'
    assert loaded.qualification['formal_eligible'] is True


def test_real_unqualified_fixture_is_not_promoted_by_dispatch_receipts(tmp_path):
    point, specs = prepared(tmp_path, qualified=False)
    with pytest.raises(VersionError, match='formal qualification'):
        pdblend_window_resources(point, specs)


@pytest.mark.parametrize('change,message', [
    (lambda c: c.update(selection_split='evaluation'), 'calibration/tuning'),
    (lambda c: c.update(profile_sha256='wrong'), 'selected profile'),
    (lambda c: c['plan'].update(tp=2), 'topology'),
    (lambda c: c['plan'].update(counts={'M': 1}), 'inventory'),
    (lambda c: c['plan'].update(counts={'M': True, 'off': 1}), 'inventory'),
    (lambda c: c['plan'].update(f_M=1000), 'frequency'),
    (lambda c: c['plan'].update(profile_key='stale'), 'profile key'),
    (lambda c: c['plan'].update(unrecognized_field=1), 'unknown'),
])
def test_wrong_offline_choice_fails_before_consuming_a_different_plan(tmp_path, change, message):
    point, specs = prepared(tmp_path, qualified=True)
    change_choice(point, change)
    with pytest.raises(ValueError, match=message):
        pdblend_window_resources(point, specs)


def test_heterogeneous_inventory_is_not_silently_flattened(tmp_path):
    point, specs = prepared(tmp_path, qualified=True)
    specs[1].tp = 2
    with pytest.raises(ValueError, match='heterogeneous'):
        pdblend_window_resources(point, specs)


def test_mixed_compact_outcomes_are_restored_and_service_start_is_preserved(tmp_path):
    with CompactJournal(tmp_path / 'outcomes.jsonl.gz') as journal:
        journal.write(dict(idx=0, completion_tokens=2, correct=True))
    start, rows, journal_path = read_native_measurement('mixed', tmp_path, dict(started_s=100))
    assert start == 100 and rows == [dict(idx=0, completion_tokens=2, correct=True)]
    assert journal_path is None


def test_dynamo_json_outcomes_and_journal_origin_are_read(tmp_path):
    (tmp_path / 'outcomes.json').write_text(json.dumps([dict(request_id='dynamo-701-0', ok=True)]))
    with CompactJournal(tmp_path / 'events.jsonl.gz') as journal:
        journal.write(dict(event='dynamo_service_window_start', at_s=120))
    start, rows, journal_path = read_native_measurement('dynamollm', tmp_path, dict(started_s=50))
    assert start == 120 and rows[0]['request_id'] == 'dynamo-701-0'
    assert journal_path == tmp_path / 'events.jsonl.gz'


def test_distserve_uses_service_clock_not_earlier_setup_clock(tmp_path):
    start, _, _ = read_native_measurement('distserve', tmp_path,
        dict(started_s=100, service_window_started_s=120, outcomes=[]))
    assert start == 120
    with pytest.raises(ValueError, match='service origin'):
        read_native_measurement('distserve', tmp_path, dict(started_s=100, outcomes=[]))
