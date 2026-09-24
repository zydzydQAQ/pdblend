"""Conservative clock annotations derived from frozen window acceptance.

This never changes a historical result or grants window/SLO qualification.
Call after the exporter's existing artifact checksum validation. Large raw
journals are not read a second time: their immutable artifact hashes must
reconstruct the acceptance evidence digest. Only small binding/source files are
read here. Unreviewed acceptance implementations remain unknown.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCOPE = 'observed_requested_active_clock/v1'
_ECO = 'pdblend/bench/comparison_ecoserve_acceptance.py'
_ECO32 = 'pdblend/bench/comparison_ecoserve32_acceptance.py'
_PD = 'pdblend/bench/comparison_pdblend_acceptance.py'
# Exact reviewed source bytes, not a substring test for a familiar gate name.
_ECO_LEGACY = '4fd9d6c4560c1ac1a54a617d2e54cf6b9be8328013dc24b47797107f8364478d'
_ECO_EXPLICIT = '7510ff9cdb47d5bcdb0b138c6e6fa4b838d22364bb861cffd04d37ea89ba6382'
_ECO32_EXPLICIT = 'a1b9a07991aa8153be2860b72e4386358f89ddb5f93d9ae14c343269822b48a5'
_PD_EXPLICIT = 'cc8d8294ef48ad3d7e19199ec791bff5b764e8ca6a6192bbaa466a8561b87e61'


def qualified(row):
    """Pure predicate for the derived common comparison requirement."""
    return row.get('common_clock_evidence') == 'pass' and row.get('common_clock_scope') == SCOPE


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(ref):
    _need(isinstance(ref, dict) and isinstance(ref.get('path'), str), 'missing evidence reference')
    path = Path(ref['path'])
    _need(path.is_absolute(), 'evidence path is not absolute')
    raw = path.read_bytes()
    _need(hashlib.sha256(raw).hexdigest() == ref.get('sha256'), 'evidence checksum differs: ' + str(path))
    return json.loads(raw)


def _binding(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def annotate(point, result, *, receipt=None, receipt_path=None):
    """Return pass/fail/unknown without mutating point, result, or receipt.

    ``fail`` means a reviewed physical-clock gate failed, including missing
    observations; it does not itself diagnose throttling. ``unknown`` includes
    absent gates, unreviewed source, or incomplete provenance. Passed clocks do
    not imply that any other acceptance gate or SLO passed.
    """
    refs = []
    def answer(status, reason):
        return dict(common_clock_evidence=status, common_clock_scope=SCOPE,
                    common_clock_reason=reason, common_clock_evidence_refs=refs)
    try:
        _need(receipt_path is not None, 'bound window receipt context is missing')
        path = Path(receipt_path).resolve()
        receipt_ref = _binding(path)
        disk = _read(receipt_ref)
        _need(receipt is None or disk == receipt, 'supplied receipt differs from disk')
        receipt = disk
        refs.append(dict(kind='receipt', **receipt_ref))
        artifacts = receipt.get('artifacts', {})
        def artifact(name, *, read=False):
            _need(name in artifacts, 'bound window artifact is missing: ' + name)
            target = (path.parent / name).resolve()
            _need(target.is_relative_to(path.parent), 'window artifact escapes directory')
            ref = dict(path=str(target), sha256=artifacts[name])
            return _read(ref) if read else ref
        _need(artifact('point.json', read=True) == point and receipt.get('point_sha256') == _digest(point)
              and receipt.get('point') == point.get('name'), 'historical point binding differs')
        _need(artifact('result.json', read=True) == result
              and (receipt.get('result') is None or receipt['result'] == result), 'historical result binding differs')
        acceptance = artifact('run/acceptance.json', read=True)
        _need(acceptance == result.get('acceptance'), 'acceptance differs from bound result')
        refs.append(dict(kind='acceptance', **artifact('run/acceptance.json')))
        qualification_ref = result.get('qualification')
        startup = _read(qualification_ref)
        refs.append(dict(kind='startup', **qualification_ref))
        source_ref = startup.get('source_manifest')
        _need(source_ref == point.get('source_manifest'), 'point and startup source bindings differ')
        manifest = _read(source_ref)
        _need(manifest.get('source_sha256') == _digest(manifest['files'])
              == result.get('identity', {}).get('source_sha256'), 'executed source inventory digest differs')
        refs.append(dict(kind='source_manifest', **source_ref))
        source_root = Path(source_ref['path']).parent
        def source(name, expected):
            _need(manifest['files'].get(name) == expected, 'clock acceptance source has not been reviewed')
            ref = dict(path=str(source_root / name), sha256=expected)
            _need(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == expected,
                  'reviewed clock acceptance source bytes changed')
            refs.append(dict(kind='clock_acceptance_source', **ref))
        checked = set(acceptance.get('checked_gates', []))
        failures = acceptance.get('gate_failures', {})
        _need(isinstance(failures, dict) and not checked.intersection(failures), 'ambiguous acceptance gates')

        # Reconstruct the frozen runtime's complete raw_refs identity, without
        # reading/compressing the already checksum-validated large artifacts.
        names = {'requests': 'requests.json', 'power': 'power.json', 'native_result': 'native-result.json',
                 'canonical_requests': 'comparison-requests.json', 'metering': 'comparison-metering.json',
                 'drain': 'native-drain.json', 'controller': 'controller.jsonl', 'routes': 'routes.jsonl',
                 'native_cleanup': 'native-cleanup.json', 'transition_measurements': 'transition-measurements.json',
                 'frequencies': 'freq.jsonl', 'metering_method': 'metering-method.json'}
        raw_refs = {key: artifact('run/' + name) for key, name in names.items() if 'run/' + name in artifacts}
        for key, choices in [('events', ('events.jsonl.gz', 'events.jsonl')),
                             ('outcomes', ('outcomes.jsonl.gz', 'outcomes.jsonl', 'outcomes.json'))]:
            found = ['run/' + name for name in choices if 'run/' + name in artifacts]
            if found:
                raw_refs[key] = artifact(found[0])
        raw_refs.update(trace=point['trace'], startup_qualification=qualification_ref, reset=artifact('reset.json'))
        session = Path(qualification_ref['path']).parent
        for key, filename in [('concurrency_environment', 'concurrency-environment.json'), ('lease_manifest', 'lease-manifest.json')]:
            if (session / filename).is_file():
                raw_refs[key] = _binding(session / filename)
        _need(_digest(raw_refs) == acceptance.get('evidence_sha256'), 'raw artifact identities differ from acceptance digest')
        _need({'raw.startup_qualification', 'binding.startup_qualification', 'binding.trace'} <= checked,
              'source/trace acceptance bindings did not pass')

        schema = acceptance.get('schema')
        if schema == 'mixed-single-observation-acceptance-v1' and point.get('system') == 'mixed':
            return answer('unknown', 'frozen Mixed checks requested reset clock; actual service-window frequency was not collected or gated')
        if schema in ('ecoserve-single-observation-acceptance-v1', 'ecoserve32-single-observation-acceptance-v1') and point.get('system') == 'ecoserve':
            legacy = manifest['files'].get(_ECO) == _ECO_LEGACY
            source(_ECO, _ECO_LEGACY if legacy else _ECO_EXPLICIT)
            if schema == 'ecoserve32-single-observation-acceptance-v1':
                _need(not legacy, 'unreviewed legacy 32B clock implementation')
                source(_ECO32, _ECO32_EXPLICIT)
            required = {'raw.events', 'raw.power', 'raw.native_result', 'eco.observation_boundary',
                        'eco.fixed_fleet', 'eco.startup', 'eco.reset', 'binding.native_result'}
            _need(required <= checked, 'physical clock scope/startup/raw prerequisites did not pass')
            refs.extend(dict(kind='raw_' + name, **raw_refs[name]) for name in ('events', 'power', 'native_result'))
            gate = 'eco.raw_protocol_and_canonical_metrics' if legacy else 'eco.observed_active_frequency'
            if gate in checked:
                # Explicit gate uses clock/park events; its protocol gate proves
                # those transitions and initial active inventory are genuine.
                _need('eco.raw_protocol_and_canonical_metrics' in checked,
                      'active clock/park interval provenance did not pass')
                return answer('pass', 'bound reviewed ' + gate + ' proves requested active clocks within 30 MHz and 1 s coverage after observed transitions')
            if not legacy and gate in failures:
                # A clock failure may still be explained even when unrelated
                # request metrics fail; this never grants qualification.
                return answer('fail', 'bound reviewed physical clock gate failed: ' + str(failures[gate]))
            return answer('unknown', 'combined legacy protocol failed or physical clock gate is absent; clock failure cannot be isolated')
        if schema == 'pdblend-single-observation-acceptance-v1' and point.get('system') == 'pdblend':
            source(_PD, _PD_EXPLICIT)
            required = {'raw.frequencies', 'raw.controller', 'raw.transition_measurements', 'pdblend.actual_window',
                        'pdblend.inventory', 'pdblend.full_physical_inventory', 'pdblend.startup', 'pdblend.reset',
                        'pdblend.controller_actions'}
            _need(required <= checked, 'PD physical clock plan/transition/raw prerequisites did not pass')
            refs.extend(dict(kind='raw_' + name, **raw_refs[name]) for name in ('frequencies', 'controller', 'transition_measurements'))
            gate = 'pdblend.physical_clocks'
            if gate in checked:
                return answer('pass', 'bound reviewed PD physical clock gate checks requested active/parked values within 30 MHz and 1 s coverage outside completed transitions')
            if gate in failures:
                return answer('fail', 'bound reviewed physical clock gate failed: ' + str(failures[gate]))
        return answer('unknown', 'no reviewed observed-frequency acceptance gate is bound to this result')
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        return answer('unknown', 'clock evidence provenance incomplete: ' + str(exc))
