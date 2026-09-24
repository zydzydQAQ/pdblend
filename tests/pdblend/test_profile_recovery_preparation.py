"""Preparation control flow only: no queue mutation or GPU execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pdblend.profile.collection.native_timing_plan import binding, read_bound
from pdblend.profile.collection import native_timing_plan_v2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return binding(path)


def fixture(tmp_path, monkeypatch, size):
    script = Path(__file__).resolve().parents[2]/'scripts/2026-09-24_prepare_pdblend_profile_recovery.py'
    spec = importlib.util.spec_from_file_location('test_profile_recovery_preparation', script)
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(native_timing_plan_v2, 'validate_plan', lambda plan: plan)
    for name in ('native_readiness.py', 'native_runtime_audit.py', 'native_timing_replay.py',
                 'native_timing_stage.py'):
        write(tmp_path/'src/pdblend/profile/collection'/name, {})
    common = write(tmp_path/'common.json', {})
    source = write(tmp_path/'source/manifest.json', {})
    model = f'Qwen2.5-{size.upper()}-Instruct'
    plan = dict(model_id=model, tp=2 if size == '32b' else 1, pp=1,
                query_ledger=common, query_provenance=common,
                points=[dict(repeats=3, purpose='training'), dict(repeats=3, purpose='holdout')])
    design = write(tmp_path/f'results/2026-09-24/pdblend-native-timing-model-plans-v2/{size}-point-plan.json', plan)
    old = tmp_path/'prior'
    manifest = dict(point_plan=design, source_manifest=source)
    if size == '32b':
        manifest['layout_energy_plan'] = common
    write(old/'manifest.json', manifest)
    monkeypatch.setattr(recovery, 'verify_prepared_bindings',
                        lambda _: (plan, dict(source_sha256='original-source')))
    queue = tmp_path/'queue/queue.json'
    write(queue, dict(jobs={}))
    attempt = queue.parent/'queue-attempts/pdblend-native-timing-old/attempt-1'
    write(attempt/'native-timing/completion.json', {})
    old_inputs = write(attempt/'inputs.json', dict(point_plan=design))
    evidence = write(attempt/'evidence.json', {})
    monkeypatch.setattr(recovery, 'inspect_attempt', lambda *a, **kw: dict(model_id=model,
        input_manifest=old_inputs, source={'source_sha256': 'original-source'},
        qualified_timing=True, timing_evidence=evidence))
    calls = []

    def freeze(target, ledger, provenance, **kwargs):
        calls.append(kwargs)
        jobs = write(target/'jobs.json', [dict(job_id='new-layout-job', payload={'execution_ready': True})])
        value = dict(jobs=jobs, point_plan=design, source_manifest=source)
        if kwargs.get('layout_energy_plan'):
            value['layout_energy_plan'] = common
        return value

    monkeypatch.setattr(recovery, 'freezer', lambda: SimpleNamespace(prepare=freeze))
    return recovery, old, queue, calls


@pytest.mark.parametrize('size', ['7b', '14b'])
def test_reused_timing_retains_unqualified_canonical_dependency_chain(tmp_path, monkeypatch, size):
    recovery, old, queue, calls = fixture(tmp_path, monkeypatch, size)
    result = recovery.prepare(tmp_path/'new', queue=queue, preparations={size: old})
    assert calls == [] and read_bound(result['jobs']) == []
    assert result['models'][size]['collection_prepared'] is False
    nodes = {row['node']: row for row in read_bound(result['dependencies'])}
    assert nodes[size+'-terminal-replay']['status'] == 'reused_qualified'
    assert len(nodes[size+'-terminal-replay']['evidence']) == 1
    assert nodes[size+'-canonical-power-and-handoff']['status'].startswith('blocked_missing_qualified')
    assert nodes[size+'-canonical-query-selection']['depends_on'] == [
        size+'-terminal-replay', size+'-canonical-power-and-handoff']
    assert nodes[size+'-selected-layout-holdout']['status'] == 'blocked_missing_frozen_selection'
    assert not result['full_profile_qualified']


def test_qualified_timing_does_not_skip_unqualified_later_layout_job(tmp_path, monkeypatch):
    recovery, old, queue, calls = fixture(tmp_path, monkeypatch, '32b')
    result = recovery.prepare(tmp_path/'new', queue=queue, preparations={'32b': old})
    assert len(calls) == 1
    assert calls[0]['collect_runtime'] and calls[0]['timing_first'] and calls[0]['layout_energy_plan']
    assert read_bound(result['jobs'])[0]['job_id'] == 'new-layout-job'
    assert not result['models']['32b']['reuse']['collect_timing']
    assert result['models']['32b']['reuse']['layout_requires_new_resident_timing_stage']
    assert not result['full_profile_qualified'] and not result['historical_blocked_reason_inherited']


def test_after_terminal_is_bound_in_execution_jobs_without_success_or_manual_gate(tmp_path, monkeypatch):
    recovery, old, queue, calls = fixture(tmp_path, monkeypatch, '32b')
    result = recovery.prepare(tmp_path/'new', queue=queue, preparations={'32b': old},
                              after_terminal=['ab-control', 'ab-candidate'])
    jobs = read_bound(result['jobs'])
    assert jobs[0]['payload']['after_terminal'] == ['ab-control', 'ab-candidate']
    assert jobs[0]['payload']['execution_ready']
    assert 'depends_on' not in jobs[0]['payload'] and 'blocked_reason' not in jobs[0]['payload']
    assert read_bound(result['models']['32b']['jobs']) == jobs
    assert 'after_terminal' not in read_bound(result['models']['32b']['collection_jobs'])[0]['payload']
    assert result['after_terminal'] == ['ab-control', 'ab-candidate']
