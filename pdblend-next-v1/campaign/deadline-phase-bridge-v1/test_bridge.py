import json
from pathlib import Path
import pytest
from bridge import scale_ready
import bridge


def fixture(root):
    rows = [dict(phase='scale', reuse_main_cell_id=f'cell-{i}', trace_sha256=f'sha-{i}')
            for i in range(18) for _ in range(2)]
    (root / 'runspec.json').write_text(json.dumps(dict(cells=rows)))
    folder = root / 'checkpoints' / 'main'
    folder.mkdir(parents=True)
    for i in range(18):
        (folder / f'{i}.json').write_text(json.dumps(dict(cell_id=f'cell-{i}', phase='main',
            source_trace_sha256=f'sha-{i}', measurement_valid=True, work_complete=False)))
    return dict(selected_phase='main', complete=True, phase='finished',
                baseline_preservation_verified=True)


@pytest.mark.parametrize('phase', ['finished', 'stopped_by_deadline'])
def test_scale_runs_after_clean_main_with_all_references_even_if_work_incomplete(tmp_path, phase):
    status = fixture(tmp_path)
    status['phase'] = phase
    assert len(scale_ready(tmp_path, status)) == 18


@pytest.mark.parametrize('change', [dict(complete=False), dict(phase='failed'),
    dict(phase='stopped_by_request'), dict(baseline_preservation_verified=False)])
def test_unclean_main_cannot_start_scale(tmp_path, change):
    status = fixture(tmp_path)
    status.update(change)
    with pytest.raises(RuntimeError):
        scale_ready(tmp_path, status)


def test_missing_reference_cannot_start_scale(tmp_path):
    status = fixture(tmp_path)
    (tmp_path / 'checkpoints/main/3.json').unlink()
    with pytest.raises(RuntimeError, match='missing valid same-work'):
        scale_ready(tmp_path, status)


def test_changed_trace_cannot_start_scale(tmp_path):
    status = fixture(tmp_path)
    path = tmp_path / 'checkpoints/main/3.json'
    value = json.loads(path.read_text())
    value['source_trace_sha256'] = 'different-work'
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='missing valid same-work'):
        scale_ready(tmp_path, status)


def test_stop_is_preserved(tmp_path):
    status = fixture(tmp_path)
    (tmp_path / 'STOP').write_text('user stop\n')
    with pytest.raises(RuntimeError, match='STOP'):
        scale_ready(tmp_path, status)
    assert (tmp_path / 'STOP').read_text() == 'user stop\n'


@pytest.mark.parametrize('fail_main', [False, True])
def test_process_handoff_starts_scale_only_after_main_success(tmp_path, monkeypatch, fail_main):
    root = tmp_path / 'package'
    root.mkdir()
    status = fixture(root)
    (root / 'package-manifest.json').write_text('{}\n')
    (root / 'invocations').mkdir()
    calls = []

    class Process:
        pid = 123

        def __init__(self, args, **kwargs):
            self.phase = args[args.index('--part') + 1]
            self.count = args[args.index('--max-cells') + 1]
            self.code = None
            calls.append((self.phase, self.count))

        def wait(self):
            self.code = 1 if fail_main and self.phase == 'main' else 0
            result = dict(status, selected_phase=self.phase,
                          selected_phase_execution_complete=self.code == 0,
                          checkpointed_cells=int(self.count))
            if self.code:
                result.update(complete=False, phase='failed')
            path = root / 'invocations' / (self.phase + '.json')
            path.write_text(json.dumps(result))
            return self.code

        def poll(self):
            return self.code

    monkeypatch.setattr(bridge.subprocess, 'Popen', Process)
    monkeypatch.setattr(bridge.signal, 'signal', lambda *a: None)
    digest = bridge.hashlib.sha256((root / 'package-manifest.json').read_bytes()).hexdigest()
    output = tmp_path / 'attempt'
    if fail_main:
        with pytest.raises(RuntimeError, match='runner failed'):
            bridge.execute(root, output, digest)
        assert calls == [('main', '60')]
    else:
        bridge.execute(root, output, digest)
        assert calls == [('main', '60'), ('scale', '36')]
    assert json.loads((output / 'status.json').read_text())['complete'] is (not fail_main)
