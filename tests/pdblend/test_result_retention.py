import csv
import json
from pathlib import Path
from pdblend.results.retention import prepare, apply


def make(tmp):
    (tmp/'results/old/run').mkdir(parents=True)
    (tmp/'results/current').mkdir()
    (tmp/'src').mkdir()
    (tmp/'results/old/run/summary.json').write_text(json.dumps(dict(model='m',policy='mixed',slo=dict(offered=1))))
    (tmp/'results/old/run/events.jsonl').write_text('{"event":"sample"}\n')
    return tmp/'results/old'


def test_current_transitive_reference_and_csv(tmp_path):
    old=make(tmp_path)
    (old/'profile.json').write_text(json.dumps({'raw':'raw.json'}))
    (old/'raw.json').write_text('{}')
    (tmp_path/'results/current/profile.json').write_text(json.dumps({'base':'results/old/profile.json'}))
    out=tmp_path/'results/maintenance/audit'
    p=prepare(tmp_path,[old],out)
    assert {e['path'] for e in p['entries'] if e['decision']=='retain_dependency'}=={'results/old/profile.json','results/old/raw.json'}
    r=apply(out/'plan.json')
    assert r['deleted_files']==2
    assert (old/'raw.json').exists()
    rows=list(csv.DictReader((tmp_path/'results/runs.csv').open()))
    assert rows[0]['evidence_status']=='raw_pruned'
    assert rows[0]['formal_eligible']=='False'


def test_changed_and_new_reference_skip(tmp_path):
    old=make(tmp_path); out=tmp_path/'results/maintenance/audit'
    prepare(tmp_path,[old],out)
    (old/'run/events.jsonl').write_text('changed')
    (tmp_path/'results/current/new.json').write_text(json.dumps({'proof':'results/old/run/summary.json'}))
    r=apply(out/'plan.json')
    assert r['deleted_files']==0
    assert {e['skip_reason'] for e in r['entries']}=={'file_changed','new_current_reference'}


def test_keep_csv_and_reject_external(tmp_path):
    import pytest
    old=make(tmp_path); (old/'existing.csv').write_text('x\n1\n')
    out=tmp_path/'results/maintenance/audit'
    prepare(tmp_path,[old],out);apply(out/'plan.json')
    assert (old/'existing.csv').is_file()
    with pytest.raises(ValueError):prepare(tmp_path,[tmp_path.parent],tmp_path/'bad')


def test_pruned_child_raw_marks_retained_summary(tmp_path):
    old=make(tmp_path);out=tmp_path/'results/maintenance/audit'
    (old/'run/raw').mkdir();(old/'run/raw/token.jsonl').write_text('{}\n')
    (tmp_path/'results/current/consumer.json').write_text(json.dumps({'summary':'results/old/run/summary.json'}))
    prepare(tmp_path,[old],out);apply(out/'plan.json')
    assert (old/'run/summary.json').exists()
    rows=list(csv.DictReader((tmp_path/'results/runs.csv').open()))
    assert rows[0]['evidence_status']=='raw_pruned'
    assert rows[0]['retention_manifest']==str(out/'plan.json')


def test_parent_directory_scan_does_not_reactivate_retired_audit(tmp_path):
    old=make(tmp_path);out=tmp_path/'results/maintenance/audit'
    archive=tmp_path/'results/maintenance/previous';archive.mkdir(parents=True)
    (archive/'identities.csv').write_text('path\nresults/old\n')
    (tmp_path/'results/current/root.json').write_text(json.dumps({'namespace':str(tmp_path/'results')}))
    p=prepare(tmp_path,[old],out,active_roots=[tmp_path/'results/current'])
    assert p['candidate_bytes']>0
    assert not any(e['decision']=='retain_dependency' for e in p['entries'])


def test_profile_point_tombstones(tmp_path):
    old=make(tmp_path);out=tmp_path/'results/maintenance/audit'
    point_file=tmp_path/'results/profile_points.csv'
    with point_file.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['point_id','raw_path','status','formal_eligible']);w.writeheader()
        w.writerow(dict(point_id='p',raw_path=str(old/'run/events.jsonl'),status='measured',formal_eligible=False))
    prepare(tmp_path,[old],out);apply(out/'plan.json')
    points=list(csv.DictReader(point_file.open()))
    assert points[0]['status']=='raw_pruned'
    assert (out/'historical_profile_points.csv').is_file()


def test_explicit_active_namespace_does_not_revive_retired_siblings(tmp_path):
    old=make(tmp_path);out=tmp_path/'results/maintenance/audit'
    sibling=tmp_path/'results/old-index';sibling.mkdir()
    (sibling/'frozen-before.json').write_text(json.dumps({'historical':'results/old'}))
    (tmp_path/'results/current/paths.json').write_text(json.dumps({'output_namespace':str(tmp_path/'results')}))
    p=prepare(tmp_path,[old],out,active_roots=[tmp_path/'results/current'])
    assert p['candidate_bytes']>0
    assert not any(e['decision']=='retain_dependency' for e in p['entries'])
