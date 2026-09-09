"""Bind the original B32B P4 policy/profile to equivalent P12 OFF and fresh TP2."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cells_v1 as runner
g, p, R = runner.g, runner.p, runner.R


def prepare(declaration, destination):
    destination = destination.resolve()
    assert not destination.exists(), 'immutable release directory required'
    contract_path = R / 'common/ascending-rate-execution-v2/contract.py'
    shared_manifest = Path(declaration['path']).parent / 'manifest.json'
    assert p.sha(shared_manifest) == '09c5005710998560c51b96cac5bf95d1eac105ce9fc73883dcd63d29bbe52842'
    assert declaration['sha256'] == 'a566e7f203a77abe64aa2f122e66c47b87c44ab0b91f60c3453235ca69c4c32f'
    contract = g.load(contract_path, 'ascending_B_prepare_declaration')
    group = contract.resolve_group(declaration, '32b', 'alpaca', actual_host='B')
    selected = contract.select_group(group, [])
    assert selected['phase'] == 'pdblend' and selected['rate_rps'] == 4.5
    rows = [contract.lookup(declaration, '32b', 'alpaca', 4.5, 'pdblend', n) for n in (1, 2)]
    assert [t['row'] for t in selected['next_tasks']] == rows
    qual = p.ref(HERE / 'pdb-qualification-001.json')
    q = g.load(HERE / 'verify_pdb_v1.py', 'ascending_B_prepare_saved')
    checked = q.verify(qual)
    original_ref = p.ref(R / 'B/completion-release-p4-001/release.json')
    original = p.checked(original_ref)
    host = Path(checked['host_manifest']['path']).parent
    off_ref = p.ref(R / 'common/idle-domain-off-existing95-reuse-declaration-v7.json')
    off = p.checked(off_ref)
    model = next(m for m in off['models'] if m['model'] == '32b')
    assert model['P12_manifest'] == checked['host_manifest']
    assert off['passed'] and off['independently_recomputed'] and off['actual_CB_P4_count'] == 83
    b = copy.deepcopy(p.checked(checked['binding']))
    b.update(host_release=str(host), configs={'alpaca': original['configs']['fixed2']['alpaca']['path']},
             output=str(HERE / 'pdb-performance-001/results'), historical_binding_only=False,
             experiment_scope='B32B ascending Alpaca4.5; P4 exact policy/profile, equivalent P12 OFF, fresh original TP2',
             fresh_ascending_binding=True, output_correctness_verified=True,
             correctness_gate_required_before_performance=False, qualification=qual)
    b.pop('fresh_ablation_binding', None)
    b['files'].update(checked['files'])
    b['files'].update({str(host / name): digest for name, digest in p.read(host / 'manifest.json')['files'].items()})
    references = [declaration, qual, original_ref, off_ref, checked['host_manifest'], p.ref(contract_path),
                  p.ref(shared_manifest), p.ref(Path(declaration['path']).parent / 'node-B.json'),
                  p.ref(HERE / 'runner-cpu-validation-001.json'),
                  p.ref(R / 'cpu-validation-p12.json'), original['configs']['fixed2']['alpaca'], original['profile_refs']['alpaca']]
    for reference in references:
        b['files'][reference['path']] = reference['sha256']
    for row in rows:
        b['files'][row['trace']] = row['trace_sha256']
    for f in (Path(__file__), HERE / 'run_cells_v1.py', HERE / 'verify_pdb_v1.py',
              R / 'audit_cooperative_arrivals_v1.py'):
        b['files'][str(f)] = p.sha(f)
    destination.mkdir()
    p.save(destination / 'binding.json', b)
    release = dict(schema='B32B-ascending-cells-release-v1', created_s=time.time(), node='B', system='pdblend',
                   declaration=declaration, declaration_contract=p.ref(contract_path), rows=rows,
                   host_manifest=checked['host_manifest'], profile=checked['profile'],
                   original_performance_release=original_ref, off_source_equivalence=off_ref,
                   qualification=qual, qualification_validator=p.ref(HERE / 'verify_pdb_v1.py'),
                   binding=p.ref(destination / 'binding.json'),
                   predecessors=[p.ref(HERE / 'restore-pdb-001/status.json'), p.ref(HERE / 'qualification-001/status.json')],
                   arrival_limits=dict(max_s=1., p99_s=.1, method='linear at(n-1)*0.99'), files=dict(b['files']))
    release['files'][str(destination / 'binding.json')] = p.sha(destination / 'binding.json')
    p.save(destination / 'release.json', release)
    runner.load_release(p.ref(destination / 'release.json'))
    return p.ref(destination / 'release.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--declaration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(prepare(p.ref(args.declaration), args.out)))
