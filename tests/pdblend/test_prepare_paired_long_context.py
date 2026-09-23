import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2]/'scripts/2026-09-23_prepare_paired_long_context.py'
spec = importlib.util.spec_from_file_location('prepare_paired_long_context', path)
prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)


def fixture(tmp_path):
    candidate = tmp_path/'candidate'; candidate.mkdir()
    (candidate/'candidate.json').write_text('{}')
    holdout_raw = tmp_path/'14b.raw.json'; holdout_raw.write_text('{}')
    manifest = dict(system='pdblend', model_id='Qwen2.5-14B-Instruct', tp=4, pp=1,
        candidate_sha256=prepare.sha256(candidate/'candidate.json'), training_raw=str(holdout_raw),
        training_raw_sha256=prepare.sha256(holdout_raw))
    (candidate/'manifest.json').write_text(json.dumps(manifest))
    source = tmp_path/'7b.raw.json'
    source.write_text(json.dumps(dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1)))
    receipt = tmp_path/'receipt.json'; receipt.write_text('{}')
    plan = dict(system='pdblend', model_id='Qwen2.5-7B-Instruct', tp=4, pp=1,
        training_source=str(source), training_source_sha256=prepare.sha256(source), fit_existing_holdout=False,
        training=[dict(freq_mhz=f, purpose='training_extension') for f in (900, 1200, 1500, 1800, 2100, 2520)],
        holdout=[dict(freq_mhz=900, purpose='independent_holdout_after_candidate_freeze')])
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan))
    image = 'sha256:'+'i'*64
    argv = ['docker', 'run', '--rm', '--name', 'old', '--gpus', 'all',
        '-v', '/frozen/source:/opt/pdblend-src:ro', '-v', '/models-host:/models:ro',
        '-v', f'{candidate}:/candidate:ro', '-v', f'{receipt}:/verification/model-verification.json:ro',
        '-v', '{attempt_dir}:/output:rw', '-v', '/coord-host:/coord:rw', '-v', '/old-wave:/wave:rw',
        '-v', f'{holdout_raw}:{holdout_raw}:ro',
        '-e', 'PDBLEND_SOURCE_SHA256=old', '-e', 'PDBLEND_PROFILE_WAVE=/wave',
        '-e', 'PDBLEND_PROFILE_MEMBER=old', image, '-B', '-m', 'pdblend.profile.calibration']
    job = dict(job_id='old', status='queued', attempts=0, lease_id=None, priority=298,
        payload=dict(model_id='Qwen2.5-14B-Instruct', tp=4, gpu_count=4, exclusive=False,
                     global_lock=False, evidence_class='independent_calibration_holdout',
                     candidate_dir=str(candidate), image_digest=image, argv=argv,
                     depends_on=['prior-7b', 'prior-32b']))
    return dict(jobs={'old': job}, leases={}), plan_path


def test_prepare_preserves_live_state_and_builds_disjoint_paired_spec(tmp_path):
    state, plan = fixture(tmp_path)
    original = copy.deepcopy(state)
    review = prepare.build_review(state, plan_path=plan, review_dir=tmp_path/'review')
    assert state == original
    assert not review['live_queue_modified'] and not review['source_frozen']
    assert review['ready_for_root_final_review'] is False
    holdout, train = review['jobs']
    assert holdout['payload']['gpu_count'] == train['payload']['gpu_count'] == 4
    assert holdout['payload']['depends_on'] == train['payload']['depends_on']
    assert review['wave']['qualification_frequency_mhz'] == 2100
    assert len(review['wave']['profile_frequency_coverage']) == 6
    assert review['wave']['members'] == ['14b-tp4-holdout', '7b-tp4-longctx']
    assert '--shared-prefill-windows' in holdout['payload']['argv']
    assert 'pdblend.profile.long_context_collect' in train['payload']['argv']
    assert train['payload']['holdout_points_consumed'] == 0
    assert train['payload']['independent_holdout'] is False
    for job in (holdout, train):
        argv = job['payload']['argv']
        assert any(x.endswith(':/models:ro') for x in argv)
        assert any(x.endswith(':/training/raw.json:ro') for x in argv)
        assert any(x.endswith(':/opt/pdblend-src:ro') for x in argv)
        assert not any('/old-wave:' in x for x in argv)
    assert not any(x.endswith(':/candidate:ro') for x in train['payload']['argv'])


@pytest.mark.parametrize('change', ['attempt', 'lease', 'running'])
def test_prepare_refuses_previously_started_holdout(tmp_path, change):
    state, plan = fixture(tmp_path)
    if change == 'attempt': state['jobs']['old']['attempts'] = 1
    if change == 'lease': state['leases']['x'] = dict(job_id='old', status='failed')
    if change == 'running': state['jobs']['old']['status'] = 'running'
    with pytest.raises(ValueError):
        prepare.build_review(state, plan_path=plan, review_dir=tmp_path/'review')


def test_prepare_fails_if_plan_or_candidate_bytes_changed(tmp_path):
    state, plan = fixture(tmp_path)
    (tmp_path/'7b.raw.json').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        prepare.build_review(state, plan_path=plan, review_dir=tmp_path/'review')


def test_review_files_are_idempotent_and_never_overwritten(tmp_path):
    path = tmp_path/'review.json'
    prepare.write_immutable(path, dict(a=1))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    prepare.write_immutable(path, dict(a=1))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match='differs'):
        prepare.write_immutable(path, dict(a=2))
