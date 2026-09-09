"""Completion concerns coverage and evidence; negative performance never fails it."""
from collections import defaultdict


def assess(points,pairs,grid,baselines,boundaries):
    required=[p for p in points if p.get('required_execution',True)]
    pdb_gaps=[p['cell_id'] for p in required if not p.get('measurement_valid') or not p.get('work_complete')]
    original_gaps=[p['original_cell_id'] for p in grid if not p['repeat_requirement_complete']]
    matched=defaultdict(set)
    for pair in pairs:matched[pair['cell_id']].add(pair['baseline_system'])
    systems={'mixed','distserve','dynamollm','ecoserve'}
    pair_gaps=[dict(cell_id=p['cell_id'],systems=sorted(systems-matched[p['cell_id']]))
        for p in required if p.get('measurement_valid') and systems-matched[p['cell_id']]]
    baseline_gaps=[p['cell_id'] for p in baselines if not p.get('measurement_valid')]
    versions=defaultdict(set)
    for point in points:
        if point.get('measurement_valid'):
            versions[(point['model'],point['dataset'])].add(point['version_id'])
    expected={(m,d) for m in ('7b','14b','32b') for d in ('alpaca','sharegpt','longbench')}
    version_gaps=[dict(model=m,dataset=d,versions=sorted(versions[(m,d)]))
        for m,d in sorted(expected) if len(versions[(m,d)])!=1]
    stopped={(b['model'],b['dataset']) for b in boundaries if b['requested_stop_condition_observed']}
    boundary_gaps=[dict(model=m,dataset=d) for m,d in sorted(expected-stopped)]
    complete=(len(grid)==90 and bool(required) and not any((pdb_gaps,original_gaps,pair_gaps,
        baseline_gaps,version_gaps,boundary_gaps)))
    return dict(complete=complete,required_pdb_executions=len(required),
        pdb_work_or_evidence_gaps=pdb_gaps,original_logical_gaps=original_gaps,
        baseline_pair_gaps=pair_gaps,new_baseline_evidence_gaps=baseline_gaps,
        final_version_gaps=version_gaps,first_loss_gaps=boundary_gaps,
        valid_baseline_incomplete_negatives=sum(p.get('measurement_valid') and not p.get('work_complete') for p in baselines),
        performance_pass_rate_is_not_completion_requirement=True,
        independent_seeds_and_boundary_bisection_not_required=True,
        scope='Final selected rate comparison only; historical suffix and delivery packaging are separate')
