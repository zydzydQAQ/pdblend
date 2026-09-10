"""Small adapter for node dispatchers; no GPU work on its dry-run CLI."""
import argparse
import json
from pathlib import Path
import contract as c


def next_group(declaration, model, dataset, *, node, observations=()):
    values = [c.checked(o) if set(o) == {'path', 'sha256'} else o for o in observations]
    return c.select_group(c.resolve_group(declaration, model, dataset, actual_host=node), values)


def prepare_dispatch(*, rows, **kwargs):
    import prepare_release
    c.need(rows and len({(r['model'], r['dataset'], r['system'], r['rate_rps'],
                          r.get('measurement_purpose', 'normal')) for r in rows}) == 1,
           'one purpose/system/rate per immutable measurement release')
    row = rows[0]
    for candidate in rows:
        c.need(c.lookup_cell(kwargs['declaration'], candidate['cell_id']) == candidate, 'undeclared row')
    kwargs.update(model=row['model'], dataset=row['dataset'], node=row['node'], system=row['system'],
                  rate=row['rate_rps'], repeats=tuple(r['repeat'] for r in rows),
                  measurement_purpose=row.get('measurement_purpose', 'normal'))
    return prepare_release.prepare(**kwargs)


def dry_run(declaration, node=None):
    d = c.load_declaration(declaration)
    result = []
    for group in d['groups']:
        if node is not None and group['node'] != node:
            continue
        decision = next_group(declaration, group['model'], group['dataset'], node=group['node'])
        rows = [t['row'] for t in decision.get('baseline_tasks', []) if t['action'] == 'execute']
        result.append(dict(node=group['node'], model=group['model'], dataset=group['dataset'],
            phase=decision['phase'], cap_rate_rps_decimal=decision.get('cap_rate_rps_decimal'),
            normal_baseline_missing=sum(r['measurement_purpose'] == 'normal' for r in rows),
            metric_supplements=sum(r['measurement_purpose'] == 'metric_supplement' for r in rows),
            next_pdb_cell_ids=[t['cell_id'] for t in decision.get('next_tasks', [])],
            pdb_boundary_complete=decision.get('pdb_boundary_complete', False),
            five_system_complete=decision.get('five_system_complete', False)))
    return dict(cpu_only=True, declaration=declaration, groups=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--node', choices=('C', 'B', 'Anew20260909'))
    args = parser.parse_args()
    print(json.dumps(dry_run(c.ref(args.declaration), args.node), indent=2))
