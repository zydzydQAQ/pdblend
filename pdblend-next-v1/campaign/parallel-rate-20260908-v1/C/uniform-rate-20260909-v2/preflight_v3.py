"""CPU-only preflight of every remaining assigned baseline row."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
U = ROOT / 'C/uniform-rate-20260909-v2'
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
import contract


def main():
    plan_ref = p.ref(U / 'plan-003.json')
    plan = p.checked(plan_ref)
    out = U / 'preflight-003'
    p.need(not out.exists(), 'fresh preflight required')
    out.mkdir()
    coverage = {}
    for system, href in plan['handoffs'].items():
        h = p.checked(href)
        binding = p.checked(p.checked(h['qualification'])['binding'])
        coverage[system] = sorted(binding['configs'])
        p.need(set(plan['datasets']) <= set(binding['configs']), 'missing dataset configuration: ' + system)
    requests = []
    groups = {}
    for dataset in plan['datasets']:
        obs = [r for r in plan['initial_observations'] if p.checked(r)['dataset'] == dataset]
        group = contract.resolve_group(plan['declaration'], plan['model'], dataset, actual_host=plan['node'])
        decision = contract.select_group(group, [p.checked(r) for r in obs])
        p.need(decision['pdb_boundary_complete'], 'PDB boundary not complete')
        groups[dataset] = decision
        for task in decision['baseline_tasks']:
            if task['action'] != 'execute':
                continue
            row = task['row']
            h = p.checked(plan['handoffs'][row['system']])
            dest = out / row['cell_id']
            kwargs = dict(declaration=plan['declaration'], qualification=h['qualification'],
                qualification_validator=h['qualification_validator'], node=plan['node'], model=plan['model'],
                dataset=dataset, rate=row['rate_rps'], system=row['system'], out=str(dest / 'release'),
                scheduling_observations=obs, predecessors=[*h.get('predecessors', []), plan['last_cell_status']],
                repeats=[row['repeat']], measurement_purpose=row.get('measurement_purpose', 'normal'),
                stop_paths=plan['stop_paths'], extra_files=[plan_ref, p.ref(__file__), p.ref(U / 'pipeline_v4.py')])
            p.save(dest / 'request.json', dict(rows=[row], kwargs=kwargs))
            requests.append((row['cell_id'], dest))
    def run(item):
        cell_id, dest = item
        with (dest / 'prepare.log').open('xb') as stream:
            proc = subprocess.run([sys.executable, '-B', str(U / 'prepare_request_v3.py'),
                '--request', str(dest / 'request.json'), '--out', str(dest / 'release-reference.json')],
                stdout=stream, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        return dict(cell_id=cell_id, passed=proc.returncode == 0, exitcode=proc.returncode,
                    log=p.ref(dest / 'prepare.log'), release=p.read(dest / 'release-reference.json') if proc.returncode == 0 else None)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run, requests):
            results.append(result)
            print(len(results), len(requests), result['cell_id'], result['passed'], flush=True)
    p.save(out / 'result.json', dict(schema='uniform-v2-all-missing-rows-cpu-preflight', cpu_only=True,
        no_measurement=True, plan=plan_ref, coverage=coverage, pdb_groups=groups,
        initial_observations=plan['initial_observations'], rows=results, passed=all(r['passed'] for r in results)))
    p.need(all(r['passed'] for r in results), 'at least one future row did not prepare')


if __name__ == '__main__':
    main()
