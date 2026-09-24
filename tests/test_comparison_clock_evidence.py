"""Clock annotation is separate from immutable window/SLO qualification."""
from copy import deepcopy
import hashlib
import json

import pytest

from pdblend.bench import comparison_clock_evidence as C


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + '\n')
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(tmp_path, monkeypatch, *, system='ecoserve', legacy=False, state='pass'):
    name = system + '-point'
    window = tmp_path / 'session/windows' / name
    run = window / 'run'
    sources = tmp_path / 'source'
    source_name = C._PD if system == 'pdblend' else C._ECO
    text = ('synthetic reviewed contract ' + system + str(legacy)).encode()
    source_file = sources / source_name
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(text)
    source_hash = hashlib.sha256(text).hexdigest()
    attr = '_PD_EXPLICIT' if system == 'pdblend' else '_ECO_LEGACY' if legacy else '_ECO_EXPLICIT'
    monkeypatch.setattr(C, attr, source_hash)
    files = {source_name: source_hash}
    source_ref = save(sources / 'manifest.json', dict(files=files, source_sha256=C._digest(files)))
    startup_ref = save(tmp_path / 'session/qualification.json', dict(source_manifest=source_ref))
    trace_ref = save(tmp_path / 'trace.json', {})
    point = dict(name=name, system=system, source_manifest=source_ref, trace=trace_ref)
    artifacts = {}
    raw = dict(trace=trace_ref, startup_qualification=startup_ref)
    paths = {'events': 'run/events.jsonl', 'outcomes': 'run/outcomes.json', 'power': 'run/power.json',
             'native_result': 'run/native-result.json', 'canonical_requests': 'run/comparison-requests.json',
             'metering': 'run/comparison-metering.json', 'reset': 'reset.json', 'drain': 'run/native-drain.json'}
    if system == 'pdblend':
        paths.update(frequencies='run/freq.jsonl', controller='run/controller.jsonl',
                     transition_measurements='run/transition-measurements.json')
    for key, rel in paths.items():
        raw[key] = save(window / rel, {})
        artifacts[rel] = raw[key]['sha256']
    checked = {'raw.startup_qualification', 'binding.startup_qualification', 'binding.trace'}
    if system == 'pdblend':
        checked |= {'raw.frequencies', 'raw.controller', 'raw.transition_measurements', 'pdblend.actual_window',
                    'pdblend.inventory', 'pdblend.full_physical_inventory', 'pdblend.startup', 'pdblend.reset',
                    'pdblend.controller_actions'}
        gate = 'pdblend.physical_clocks'
    else:
        checked |= {'raw.events', 'raw.power', 'raw.native_result', 'eco.observation_boundary',
                    'eco.fixed_fleet', 'eco.startup', 'eco.reset', 'binding.native_result'}
        gate = 'eco.raw_protocol_and_canonical_metrics' if legacy else 'eco.observed_active_frequency'
        if not legacy:
            checked.add('eco.raw_protocol_and_canonical_metrics')
    if state == 'pass':
        checked.add(gate)
    failures = {gate: 'actual frequency differs'} if state == 'fail' else {}
    acceptance = dict(schema=system + '-single-observation-acceptance-v1', checked_gates=sorted(checked),
        gate_failures=failures, evidence_sha256=C._digest(raw), evidence_valid=state != 'fail', slo_pass=False)
    result = dict(acceptance=acceptance, qualification=startup_ref, identity=dict(source_sha256=C._digest(files)),
                  evidence_valid=state != 'fail', formal_eligible=state != 'fail', metrics={'slo_pass': False})
    receipt = dict(point=name, point_sha256=C._digest(point), result=result, evidence_valid=state != 'fail',
                   baseline_frozen=state != 'fail', artifacts=artifacts)
    def freeze():
        for rel, value in [('point.json', point), ('result.json', result), ('run/acceptance.json', acceptance)]:
            artifacts[rel] = save(window / rel, value)['sha256']
        save(window / 'receipt.json', receipt)
    freeze()
    return point, result, receipt, window / 'receipt.json', freeze, raw


@pytest.mark.parametrize('system,legacy,state,expected', [
    ('ecoserve', False, 'pass', 'pass'), ('ecoserve', False, 'fail', 'fail'),
    ('ecoserve', False, 'missing', 'unknown'), ('ecoserve', True, 'pass', 'pass'),
    ('ecoserve', True, 'fail', 'unknown'), ('pdblend', False, 'pass', 'pass'),
    ('pdblend', False, 'fail', 'fail'), ('pdblend', False, 'missing', 'unknown'),
    ('mixed', False, 'pass', 'unknown'), ('distserve', False, 'pass', 'unknown'),
    ('dynamollm', False, 'pass', 'unknown')])
def test_bound_gates_and_reviewed_sources_are_required_without_mutating_qualification(tmp_path, monkeypatch, system, legacy, state, expected):
    point, result, receipt, path, _, _ = fixture(tmp_path, monkeypatch, system=system, legacy=legacy, state=state)
    before = deepcopy((point, result, receipt))
    value = C.annotate(point, result, receipt=receipt, receipt_path=path)
    assert value['common_clock_evidence'] == expected
    assert value['common_clock_scope'] == C.SCOPE
    assert C.qualified(value) is (expected == 'pass')
    assert (point, result, receipt) == before
    assert result['metrics']['slo_pass'] is False


@pytest.mark.parametrize('fault', ['source_unreviewed', 'source_changed', 'manifest_digest', 'result_identity',
    'point', 'acceptance', 'result', 'raw_hash', 'raw_digest', 'raw_prerequisite', 'interval_provenance',
    'startup_ref', 'source_ref', 'ambiguous_gate', 'receipt_context', 'artifact_escape'])
def test_untrusted_or_incomplete_binding_never_yields_pass(tmp_path, monkeypatch, fault):
    p, r, receipt, path, freeze, raw = fixture(tmp_path, monkeypatch)
    a = r['acceptance']
    if fault == 'source_unreviewed':
        monkeypatch.setattr(C, '_ECO_EXPLICIT', '0' * 64)
    elif fault == 'source_changed':
        (tmp_path / 'source' / C._ECO).write_text('changed')
    elif fault == 'manifest_digest':
        manifest = json.loads((tmp_path / 'source/manifest.json').read_text())
        manifest['source_sha256'] = '0' * 64
        ref = save(tmp_path / 'source/manifest.json', manifest)
        p['source_manifest'] = ref
        r['qualification'] = save(tmp_path / 'session/qualification.json', dict(source_manifest=ref))
        freeze()
    elif fault == 'result_identity':
        r['identity']['source_sha256'] = '0' * 64
        freeze()
    elif fault == 'point':
        p['name'] = 'foreign-point'
    elif fault == 'acceptance':
        save(path.parent / 'run/acceptance.json', dict(a, checked_gates=[]))
    elif fault == 'result':
        r['metrics']['slo_pass'] = True
    elif fault == 'raw_hash':
        receipt['artifacts']['run/power.json'] = '0' * 64
        freeze()
    elif fault == 'raw_digest':
        a['evidence_sha256'] = '0' * 64
        freeze()
    elif fault == 'raw_prerequisite':
        a['checked_gates'].remove('raw.power')
        freeze()
    elif fault == 'interval_provenance':
        a['checked_gates'].remove('eco.raw_protocol_and_canonical_metrics')
        freeze()
    elif fault == 'startup_ref':
        r['qualification']['sha256'] = '0' * 64
        freeze()
    elif fault == 'source_ref':
        p['source_manifest'] = dict(p['source_manifest'], sha256='0' * 64)
        receipt['point_sha256'] = C._digest(p)
        freeze()
    elif fault == 'ambiguous_gate':
        a['gate_failures']['eco.observed_active_frequency'] = 'fail and pass'
        freeze()
    elif fault == 'receipt_context':
        path = None
    elif fault == 'artifact_escape':
        # A symlink cannot retarget the trusted small bound artifact outside the
        # window even when the outside bytes have the original hash.
        original = path.parent / 'run/acceptance.json'
        original.rename(tmp_path / 'outside.json')
        original.symlink_to(tmp_path / 'outside.json')
    value = C.annotate(p, r, receipt=receipt, receipt_path=path)
    assert value['common_clock_evidence'] == 'unknown'
    assert not C.qualified(value)


def test_large_raw_files_are_not_reopened_after_exporter_checksum_validation(tmp_path, monkeypatch):
    p, r, receipt, path, _, raw = fixture(tmp_path, monkeypatch)
    from pathlib import Path
    original = Path.read_bytes
    forbidden = {raw[k]['path'] for k in ('power', 'events', 'outcomes')}
    def checked_read(self):
        assert str(self) not in forbidden, 'large raw data was redundantly read'
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', checked_read)
    assert C.qualified(C.annotate(p, r, receipt=receipt, receipt_path=path))


def test_qualified_requires_both_exact_status_and_scope():
    assert not C.qualified({})
    assert not C.qualified({'common_clock_evidence': 'pass'})
    assert not C.qualified({'common_clock_evidence': True, 'common_clock_scope': C.SCOPE})
    assert not C.qualified({'common_clock_evidence': 'pass', 'common_clock_scope': 'old'})
