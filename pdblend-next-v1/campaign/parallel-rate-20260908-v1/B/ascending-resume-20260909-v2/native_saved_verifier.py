"""Saved B32B original-native baseline qualification; controller strategy unchanged."""
import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapters import control as b
p, g, R = b.p, b.g, b.R


def verify(reference):
    from adapters import source_contract
    source_contract()
    contract = p.checked(reference)
    assert contract['schema'] == 'B32B-ascending-baseline-saved-qualification-v1'
    system = contract['system']
    assert system in b.OLD_BINDINGS
    for path, digest in contract['files'].items():
        assert p.sha(path) == digest, path
    historical = p.checked(contract['original_selected_binding'])
    assert contract['original_selected_binding'] == p.ref(b.OLD_BINDINGS[system])
    qualified = p.checked(contract['fresh_binding'])
    boot = p.read(b.QUAL / 'bootstrap.json')
    invocation = p.read(b.QUAL / 'gate-invocation.json')
    gate = p.read(b.QUAL / 'original27/status.json')
    runtime = {p.read(i['engine_config'])['runtime_dir'] for i in boot['instances']}
    assert len(runtime) == 1
    assert invocation['argv'][1:] == ['-B', str(b.GATE), '--binding', str(b.QUAL / 'bootstrap.json'), '--runtime-dir', runtime.pop(),
                                    '--out', str(b.QUAL / 'original27'), '--run']
    assert invocation['bootstrap'] == p.ref(b.QUAL / 'bootstrap.json') and invocation['gate_source'] == p.ref(b.GATE)
    assert invocation['host_manifest'] == p.ref(b.ECOHOST / 'manifest.json')
    assert not invocation['stop_requested'] and invocation['exitcode'] in (0, 1)
    assert invocation['started_s'] <= gate['started_s'] < gate['finished_s'] <= invocation['finished_s']
    helper = g.load(R / 'B/baseline-return-after-external-source-v1/execution.py', 'ascending_B_native_evidence')
    helper.load_common(qualified['host_release'])
    if system == 'ecoserve':
        q = g.load(HERE / 'qualify_eco_drained_baseline_v2.py', 'ascending_B_Eco_saved_native')
        assert q.audit_binding(contract['fresh_binding'])['passed']
    else:
        assert b.derive_other(system) == qualified, 'baseline native mechanism reconstruction differs'
    assert qualified['host_release'] == historical['host_release']
    assert qualified['system'] == system and qualified['model'] == '32b'
    for name, path in qualified['configs'].items():
        assert p.sha(path) == p.sha(historical['configs'][name]), 'original policy/profile configuration bytes changed'
    actual = p.checked(contract['executed_binding'])
    expected = copy.deepcopy(qualified)
    expected.update(output=contract['execution_output'], files=contract['execution_files'])
    assert actual == expected, 'execution binding may change only output and append exact frozen evidence'
    assert all(actual['files'].get(path) == digest for path, digest in qualified['files'].items())
    for path, digest in actual['files'].items():
        assert p.sha(path) == digest, path
    host = Path(actual['host_release'])
    assert contract['host_manifest'] == p.ref(host / 'manifest.json')
    return dict(passed=True, independently_recomputed=True, node='B', system=system, model='32b',
                binding=contract['executed_binding'], host_manifest=contract['host_manifest'],
                qualification=reference, original_policy_bytes_unchanged=True,
                legacy_temporal_false_preserved=(system == 'ecoserve'), files=contract['files'])
