"""Path-only adapters around immutable B32B qualification and measurement code.

Fresh source module objects keep the old modules and their historical paths intact.
Only output roots, the old terminal reader, and verifier dependency objects change.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'ascending-rate-v1'
sys.path.insert(0, str(OLD))
import baseline_control_v2 as historical

p, g, R = historical.p, historical.g, historical.R
control = g.load(OLD / 'baseline_control_v2.py', 'B32B_resume_control')
control.HERE = HERE
control.RESTORE = HERE / 'baseline-restoration-001'
control.QUAL = HERE / 'baseline-qualification-001'
control.terminal = historical.terminal

verifier = g.load(OLD / 'verify_baseline_v2.py', 'B32B_resume_verifier')
verifier.HERE = HERE
verifier.b = control

cells = g.load(OLD / 'run_cells_v1.py', 'B32B_resume_cells')
cells.HERE = HERE


def source_contract():
    manifest = p.read(HERE / 'manifest.json')
    assert manifest['schema'] == 'B32B-cold-resume-source-v1'
    for path, digest in manifest['files'].items():
        assert p.sha(path) == digest, 'resume source/evidence changed: ' + path
    declaration = p.checked(manifest['original_declaration'])
    assert declaration['schema'] == 'B32B-ascending-baseline-continuation-v1'
    assert declaration['maximum_new_baseline_runs'] == 8
    assert declaration['systems'] == ['mixed', 'distserve', 'dynamollm', 'ecoserve']
    assert declaration['unknown_failure_stops_successor'] is True
    for path, digest in declaration['files'].items():
        assert p.sha(path) == digest, 'original frozen input changed: ' + path
    boundary, release = historical.terminal()
    assert boundary == p.read(OLD / 'baseline-boundary-001.json')
    assert boundary['cap_rate_rps'] == 4.5
    assert declaration['pdb_release'] == p.ref(OLD / 'pdb-release-002/release.json')
    assert not (OLD / 'STOP').exists() and not (HERE / 'STOP').exists()
    assert not (OLD / 'baseline-releases-001.json').exists(), 'old successor produced baseline releases'
    for system in declaration['systems']:
        assert not (OLD / ('baseline-' + system + '-performance-001')).exists(), 'old baseline attempt exists'
    return manifest, boundary, release
