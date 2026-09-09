"""Freeze eight exact B32B baseline rows after the ascending PDB boundary."""
import copy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_control_v1 as c
import verify_baseline_v1 as q
p, g, R = c.p, c.g, c.R


def prepare():
    boundary_ref = p.ref(HERE / 'baseline-boundary-001.json')
    boundary = p.checked(boundary_ref)
    actual, pdb_release = c.terminal()
    assert boundary == actual
    contract_path = Path(pdb_release['declaration_contract']['path'])
    declaration = g.load(contract_path, 'ascending_B_baseline_rows')
    refs = p.read(c.QUAL / 'bindings.json')
    releases = {}
    previous = p.ref(HERE / 'pdb-performance-001/status.json')
    for system in ('mixed', 'distserve', 'dynamollm', 'ecoserve'):
        root = HERE / ('baseline-' + system + '-release-001')
        assert not root.exists()
        root.mkdir()
        base = p.checked(refs[system])
        rows = [declaration.lookup(boundary['declaration'], '32b', 'alpaca', 4.5, system, n) for n in (1, 2)]
        host = Path(base['host_release'])
        binding = copy.deepcopy(base)
        binding['output'] = str(HERE / ('baseline-' + system + '-performance-001') / 'results')
        required_refs = [boundary_ref, pdb_release['declaration'], p.ref(contract_path), refs[system],
                         p.ref(c.OLD_BINDINGS[system]), p.ref(host / 'manifest.json'), p.ref(Path(__file__)),
                         p.ref(HERE / 'verify_baseline_v1.py'), p.ref(HERE / 'baseline_control_v1.py'),
                         p.ref(HERE / 'run_cells_v1.py'), p.ref(R / 'audit_cooperative_arrivals_v1.py')]
        for value in required_refs:
            binding['files'][value['path']] = value['sha256']
        for row in rows:
            binding['files'][row['trace']] = row['trace_sha256']
        p.save(root / 'binding.json', binding)
        fields = dict(schema='B32B-ascending-baseline-saved-qualification-v1', system=system,
            original_selected_binding=p.ref(c.OLD_BINDINGS[system]), fresh_binding=refs[system],
            host_manifest=p.ref(host / 'manifest.json'), executed_binding=p.ref(root / 'binding.json'),
            execution_output=binding['output'], execution_files=binding['files'], files=dict(binding['files']))
        fields['files'][str(root / 'binding.json')] = p.sha(root / 'binding.json')
        p.save(root / 'qualification.json', fields)
        q.verify(p.ref(root / 'qualification.json'))
        # Each predecessor is consumed only after actual clean process exit.
        release = dict(schema='B32B-ascending-cells-release-v1', node='B', system=system, created_s=time.time(),
            declaration=boundary['declaration'], declaration_contract=p.ref(contract_path), rows=rows,
            host_manifest=p.ref(host / 'manifest.json'), binding=p.ref(root / 'binding.json'),
            qualification=p.ref(root / 'qualification.json'), qualification_validator=p.ref(HERE / 'verify_baseline_v1.py'),
            boundary=boundary_ref, predecessors=[previous],
            arrival_limits=dict(max_s=1., p99_s=.1, method='linear at(n-1)*0.99'), files=dict(fields['files']))
        release['files'][str(root / 'qualification.json')] = p.sha(root / 'qualification.json')
        p.save(root / 'release.json', release)
        # The first predecessor is already terminal; later stages are strictly
        # sequenced by the owning pipeline, without pre-hashing mutable status.
        c.r.load_release(p.ref(root / 'release.json'))
        releases[system] = p.ref(root / 'release.json')
    p.save(HERE / 'baseline-releases-001.json', releases)
    return releases


if __name__ == '__main__':
    print(json.dumps(prepare()))
