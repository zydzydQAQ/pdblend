"""The CPU report observer never changes or replaces experiment evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/2026-09-24_watch_repair_reports.py'
SPEC = importlib.util.spec_from_file_location('watch_repair_reports', SCRIPT)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def prepare(package):
    return write(package / 'preparation.json',
                 {name: watch.binding(package / (name + '.json')) for name in ('campaign', 'jobs')})


@pytest.fixture
def package(tmp_path):
    repair = tmp_path / 'repair'
    attempts = tmp_path / 'attempts'
    historical = tmp_path / 'historical'
    attempts.mkdir()
    for name, job in [('paired-ab', 'ab'), ('paired-ab-v2', 'v2')]:
        write(repair / name / 'campaign.json', {'points': []})
        write(repair / name / 'jobs.json', [{'job_id': job}])
        prepare(repair / name)
    write(repair / 'baseline-supplements/campaign.json', {'points': []})
    write(repair / 'baseline-supplements/jobs.json', [{'job_id': 'baseline'}])
    prepare(repair / 'baseline-supplements')
    write(historical / 'points.json', [])
    program = tmp_path / 'program/report.py'
    program.parent.mkdir()
    program.write_text('# frozen CPU report\n')
    manifest = write(program.parent / 'manifest.json',
                     {'report_script': str(program), 'files': [watch.binding(program)]})
    return SimpleNamespace(repair_root=repair, attempt_root=attempts,
                           historical_dir=historical, program_manifest=manifest,
                           out=tmp_path / 'reports')


def receipt(package, job='ab', point='point', value=None):
    return write(package.attempt_root / job / 'attempt-0001-test/session/windows' / point / 'receipt.json',
                 {'result': {'metrics': {}}} if value is None else value)


def discover(package):
    return watch.discover(package.repair_root, package.attempt_root, package.historical_dir)


def runner(monkeypatch, action=None):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        target = Path(argv[argv.index('--out') + 1])
        target.mkdir(parents=True)
        write(target / 'snapshot.json', {'cpu_only': True})
        (target / 'report.md').write_text('Completed report\n')
        if action:
            return action(argv, target)
        return subprocess.CompletedProcess(argv, 0, 'done', '')

    monkeypatch.setattr(watch.subprocess, 'run', run)
    return calls


def test_discover_only_declared_jobs_and_followup_matrix(package):
    expected = [receipt(package), receipt(package, 'baseline')]
    ignored = receipt(package, 'not-declared')
    matrix = package.repair_root / 'followup/standard'
    write(matrix / 'campaign.json', {'points': []})
    write(matrix / 'jobs.json', [{'job_id': 'standard'}])
    prepare(matrix)
    write(package.repair_root / 'followup/followup-status.json', {'matrix_path': str(matrix)})
    expected.append(receipt(package, 'standard'))
    inputs = discover(package)
    assert inputs['job_dirs'] == sorted(str(package.attempt_root / job) for job in ('ab', 'v2', 'standard'))
    assert str(matrix / 'campaign.json') in inputs['campaigns']
    paths = {ref['path'] for ref in inputs['refs']}
    assert all(str(path) in paths for path in expected)
    assert str(matrix / 'preparation.json') in inputs['repair_preparations']
    assert all(path in paths for path in inputs['repair_preparations'])
    assert str(ignored) not in paths
    assert inputs['supplement_attempt_root'] == str(package.attempt_root)
    assert not (package.out / 'state.json').exists()


def test_status_heartbeat_does_not_change_report_identity(package):
    status = package.repair_root / 'followup/followup-status.json'
    write(status, {'status': 'waiting', 'updated_s': 1})
    first = discover(package)
    write(status, {'status': 'waiting', 'updated_s': 2})
    assert discover(package) == first


def test_command_is_explicit_cpu_argv_without_shell(package):
    inputs = discover(package)
    command = watch.command('/frozen/report.py', inputs, '/reports/new', python='/cpu/python')
    assert command[:3] == ['/cpu/python', '-B', '/frozen/report.py']
    assert '--all-models' in command
    assert command.count('--repair-preparation') == 2
    assert '--campaign' not in command and '--job-dir' not in command
    assert command[command.index('--attempt-root') + 1] == inputs['supplement_attempt_root']
    assert command[command.index('--out') + 1] == '/reports/new'
    assert command[command.index('--supplement-preparation') + 1] == inputs['supplement_preparation']
    assert 'docker' not in command and '--gpus' not in command


@pytest.mark.parametrize('identity', ['../escape', '/absolute', '.', '..', 'a/b', 'a\\b', '', None])
def test_reject_job_path_traversal(package, identity):
    write(package.repair_root / 'paired-ab/jobs.json', [{'job_id': identity}])
    prepare(package.repair_root / 'paired-ab')
    with pytest.raises(ValueError, match='job ID'):
        discover(package)


def test_reject_matrix_outside_followup(package):
    write(package.repair_root / 'followup/followup-status.json',
          {'matrix_path': str(package.repair_root / '../outside')})
    with pytest.raises(ValueError, match='matrix package is outside'):
        discover(package)


def test_matrix_is_not_discovered_until_preparation_is_written(package):
    matrix = package.repair_root / 'followup/standard'
    write(matrix / 'campaign.json', {'points': []})
    write(matrix / 'jobs.json', [{'job_id': 'standard'}])
    write(package.repair_root / 'followup/followup-status.json', {'matrix_path': str(matrix)})
    assert str(matrix / 'campaign.json') not in discover(package)['campaigns']
    prepare(matrix)
    assert str(matrix / 'campaign.json') in discover(package)['campaigns']


def test_preparation_binding_change_is_rejected_before_report(package):
    write(package.repair_root / 'paired-ab/jobs.json', [{'job_id': 'unbound-job'}])
    with pytest.raises(ValueError, match='binding differs'):
        discover(package)


def test_preparation_cannot_bind_an_external_campaign(package, tmp_path):
    path = package.repair_root / 'paired-ab/preparation.json'
    data = watch.load(path)
    data['campaign'] = watch.binding(write(tmp_path / 'outside.json', {'points': []}))
    write(path, data)
    with pytest.raises(ValueError, match='bound campaign is outside'):
        discover(package)


@pytest.mark.parametrize('link_level', ['job', 'attempt', 'window', 'receipt'])
def test_reject_symlink_escape(package, tmp_path, link_level):
    outside = tmp_path / 'outside'
    outside.mkdir()
    job = package.attempt_root / 'ab'
    attempt = job / 'attempt-0001-test'
    window = attempt / 'session/windows/point'
    path = window / 'receipt.json'
    if link_level == 'job':
        job.symlink_to(outside, target_is_directory=True)
    elif link_level == 'attempt':
        job.mkdir()
        attempt.symlink_to(outside, target_is_directory=True)
        write(outside / 'session/windows/point/receipt.json', {})
    elif link_level == 'window':
        window.parent.mkdir(parents=True)
        window.symlink_to(outside, target_is_directory=True)
        write(outside / 'receipt.json', {})
    else:
        window.mkdir(parents=True)
        path.symlink_to(write(outside / 'receipt.json', {}))
    with pytest.raises(ValueError, match='outside the declared directory'):
        discover(package)


def test_program_requires_bound_entrypoint(package, tmp_path):
    manifest = watch.load(package.program_manifest)
    script = tmp_path / 'unbound.py'
    script.write_text('# not frozen')
    manifest['report_script'] = str(script)
    write(package.program_manifest, manifest)
    with pytest.raises(ValueError, match='report_script is not bound'):
        watch.verify_program(package.program_manifest)


def test_program_rejects_changed_or_duplicate_file(package):
    manifest = watch.load(package.program_manifest)
    script = Path(manifest['report_script'])
    script.write_text('# changed')
    with pytest.raises(ValueError, match='program changed'):
        watch.verify_program(package.program_manifest)
    manifest['files'] = [watch.binding(script), watch.binding(script)]
    write(package.program_manifest, manifest)
    with pytest.raises(ValueError, match='duplicate'):
        watch.verify_program(package.program_manifest)


def test_success_repeat_skips_and_new_receipt_creates_separate_snapshot(package, monkeypatch):
    raw = receipt(package)
    before = raw.read_bytes()
    calls = runner(monkeypatch)
    first = watch.run_once(package)
    first_report = Path(first['latest_report'])
    contents = first_report.read_bytes()
    assert watch.run_once(package)['latest_report'] == str(first_report)
    assert len(calls) == 1
    receipt(package, 'v2')
    second = watch.run_once(package)
    assert len(calls) == 2
    assert second['latest_report'] != str(first_report)
    assert first_report.read_bytes() == contents
    assert raw.read_bytes() == before
    assert len(second['reports']) == 2
    assert watch.load(package.out / 'latest.json') == second['reports'][-1]
    invocation = watch.load(package.out / 'invocations' / (second['latest_input_sha256'] + '.json'))
    assert invocation['status'] == 'succeeded'
    assert invocation['hardware_executed'] is False
    assert calls[0][1]['env']['MPLBACKEND'] == 'Agg'
    assert calls[0][1]['env']['PYTHONDONTWRITEBYTECODE'] == '1'


@pytest.mark.parametrize('pointer', ['missing', 'previous'])
def test_state_commit_without_latest_pointer_recovers_without_rerun(package, monkeypatch, pointer):
    calls = runner(monkeypatch)
    first = watch.run_once(package)
    latest = package.out / 'latest.json'
    if pointer == 'missing':
        latest.unlink()
    else:
        write(latest, {'path': '/old-report'})
    watch.run_once(package)
    assert len(calls) == 1
    assert watch.load(latest) == first['reports'][-1]


def test_changed_published_snapshot_is_not_silently_accepted(package, monkeypatch):
    calls = runner(monkeypatch)
    first = watch.run_once(package)
    write(Path(first['reports'][-1]['snapshot']['path']), {'tampered': True})
    with pytest.raises(ValueError, match='published report snapshot changed'):
        watch.run_once(package)
    assert len(calls) == 1


@pytest.mark.parametrize('field', ['repair_root', 'attempt_root', 'historical_dir', 'program_manifest'])
def test_changed_watcher_identity_cannot_reuse_output(package, monkeypatch, tmp_path, field):
    runner(monkeypatch)
    watch.run_once(package)
    other = SimpleNamespace(**vars(package))
    if field == 'program_manifest':
        alternate = tmp_path / 'program/alternate-manifest.json'
        write(alternate, watch.load(package.program_manifest))
        setattr(other, field, alternate)
    else:
        setattr(other, field, tmp_path / ('other-' + field))
    with pytest.raises(ValueError, match='identity changed'):
        watch.run_once(other)


def test_partial_receipt_waits_without_replacing_published_report(package, monkeypatch):
    calls = runner(monkeypatch)
    first = watch.run_once(package)
    latest = (package.out / 'latest.json').read_bytes()
    path = receipt(package)
    path.write_text('{"result":')
    waiting = watch.run_once(package)
    assert waiting['status'] == 'waiting_for_input'
    assert waiting['input_problem']['path'] == str(path)
    assert len(calls) == 1
    assert (package.out / 'latest.json').read_bytes() == latest
    assert waiting['latest_report'] == first['latest_report']
    write(path, {'result': {'metrics': {}}})
    assert watch.run_once(package)['status'] == 'watching'
    assert len(calls) == 2


@pytest.mark.parametrize('failure', ['exit', 'timeout', 'launch', 'missing_snapshot', 'invalid_snapshot', 'missing_report'])
def test_failures_retain_raw_output_diagnostics_and_previous_latest(package, monkeypatch, failure):
    runner(monkeypatch)
    first = watch.run_once(package)
    latest = (package.out / 'latest.json').read_bytes()
    raw = receipt(package)
    original = raw.read_bytes()

    def fail(argv, target):
        (target / 'partial.txt').write_text('retain me')
        if failure == 'exit':
            return subprocess.CompletedProcess(argv, 3, 'partial stdout', 'process failed')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(argv, 1800, output=b'before timeout', stderr=b'timeout diagnostic')
        if failure == 'launch':
            raise OSError('launch failed')
        if failure == 'missing_snapshot':
            (target / 'snapshot.json').unlink()
        if failure == 'invalid_snapshot':
            (target / 'snapshot.json').write_text('{')
        if failure == 'missing_report':
            (target / 'report.md').unlink()
        return subprocess.CompletedProcess(argv, 0, '', '')

    calls = runner(monkeypatch, fail)
    failed = watch.run_once(package)
    assert failed['status'] == 'report_needs_diagnosis'
    invocation_path = Path(failed['failed_invocation'])
    evidence = invocation_path.read_bytes()
    invocation = json.loads(evidence)
    assert invocation['status'] == 'failed' and invocation['error']
    assert (package.out / 'snapshots' / invocation['input_sha256'] / 'partial.txt').read_text() == 'retain me'
    assert failed['latest_report'] == first['latest_report']
    assert (package.out / 'latest.json').read_bytes() == latest
    assert raw.read_bytes() == original
    watch.run_once(package)
    assert len(calls) == 1
    assert invocation_path.read_bytes() == evidence
    if failure == 'timeout':
        assert invocation['stdout'] == 'before timeout'
        assert invocation['stderr'] == 'timeout diagnostic'


def test_input_arriving_during_generation_is_not_published(package, monkeypatch):
    runner(monkeypatch)
    first = watch.run_once(package)
    latest = (package.out / 'latest.json').read_bytes()
    receipt(package)

    def append_receipt(argv, target):
        receipt(package, 'v2')
        return subprocess.CompletedProcess(argv, 0, '', '')

    runner(monkeypatch, append_receipt)
    superseded = watch.run_once(package)
    evidence = watch.load(superseded['failed_invocation'])
    assert evidence['status'] == 'superseded'
    assert evidence['post_input_sha256'] != evidence['input_sha256']
    assert superseded['latest_report'] == first['latest_report']
    assert (package.out / 'latest.json').read_bytes() == latest
    calls = runner(monkeypatch)
    final = watch.run_once(package)
    assert final['status'] == 'watching' and len(calls) == 1
    assert final['latest_input_sha256'] == evidence['post_input_sha256']
    assert len(final['reports']) == 2


def test_partial_receipt_during_generation_is_diagnostic_and_not_published(package, monkeypatch):
    def partial(argv, target):
        path = receipt(package)
        path.write_text('{')
        return subprocess.CompletedProcess(argv, 0, '', '')

    runner(monkeypatch, partial)
    state = watch.run_once(package)
    invocation = watch.load(state['failed_invocation'])
    assert invocation['status'] == 'superseded'
    assert invocation['input_problem']['path'].endswith('/receipt.json')
    assert not (package.out / 'latest.json').exists()


def test_program_change_during_generation_does_not_publish(package, monkeypatch):
    def change(argv, target):
        Path(argv[2]).write_text('# different program')
        return subprocess.CompletedProcess(argv, 0, '', '')

    runner(monkeypatch, change)
    state = watch.run_once(package)
    assert state['status'] == 'report_needs_diagnosis'
    assert 'program changed' in watch.load(state['failed_invocation'])['error']
    assert not (package.out / 'latest.json').exists()


def test_orphan_output_after_crash_is_never_overwritten(package, monkeypatch):
    key = watch.digest(discover(package))
    target = package.out / 'snapshots' / key
    target.mkdir(parents=True)
    (target / 'partial.txt').write_text('original crash evidence')
    calls = runner(monkeypatch)
    state = watch.run_once(package)
    assert state['status'] == 'report_needs_diagnosis'
    assert not calls
    assert (target / 'partial.txt').read_text() == 'original crash evidence'


def test_existing_invocation_after_crash_is_never_reexecuted(package, monkeypatch):
    key = watch.digest(discover(package))
    path = write(package.out / 'invocations' / (key + '.json'), {'status': 'failed'})
    original = path.read_bytes()
    calls = runner(monkeypatch)
    state = watch.run_once(package)
    assert state['status'] == 'report_needs_diagnosis'
    assert not calls and path.read_bytes() == original
