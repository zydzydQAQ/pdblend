"""Declare fresh work without changing original rows, traces, or device state."""
import argparse
from collections import Counter
import copy
from pathlib import Path
import time
import protocol as p

def build():
    originals = p.original_points()
    by_pair = {}
    for point in originals:
        by_pair.setdefault(p.pair_identity(point), []).append(point)
    cells = []
    sources = {str(p.SNAPSHOT / name): digest for name, digest in p.PINNED.items()}
    for model in p.MODELS:
        selected = [v for v in originals if v['model'] == model and v['system'] == 'pdblend']
        p.need(len(selected) == 30, 'wrong original model domain')
        for point in selected:
            key = p.pair_identity(point)
            p.need({x['system'] for x in by_pair[key]} == {'pdblend', *p.BASELINES}, 'unpaired original workload')
            checkpoint_path = Path(point['checkpoint_path'])
            checkpoint = p.read(checkpoint_path)
            row = copy.deepcopy(checkpoint['row'])
            p.need(row['cell_id'] == point['cell_id'] and row['trace_sha256'] == point['trace_sha256'], 'source row mismatch')
            baseline_refs = {v['system']: v['cell_id'] for v in by_pair[key] if v['system'] != 'pdblend'}
            screen = point['rate_rps'] in p.SCREEN[model][point['dataset']]
            arms = ('fixed2', 'dynamic') if screen else ('dynamic',)
            for arm in arms:
                for repeat in (1, 2):
                    cell = dict(schema=1, model=model, dataset=point['dataset'],
                        arm=arm, repeat=repeat, stage=('screen_' + arm if screen else 'confirm_dynamic'),
                        source_row=row, original_point=point, baseline_cell_ids=baseline_refs,
                        original_checkpoint=p.ref(checkpoint_path),
                        original_receipt=p.ref(point['receipt_path']),
                        trace={'path': row['trace'], 'sha256': row['trace_sha256']},
                        original_cell_id=point['cell_id'],
                        cell_id='slo-improve-v7-' + arm + '-' + point['cell_id'] + '-repeat' + str(repeat))
                    cells.append(cell)
            sources[str(checkpoint_path)] = p.sha(checkpoint_path)
    stage_order = {'screen_fixed2': 0, 'screen_dynamic': 1, 'confirm_dynamic': 2}
    # High-load risks lead the screen; confirmation includes every remaining rate.
    cells.sort(key=lambda c: (p.MODELS.index(c['model']), stage_order[c['stage']],
        c['repeat'], p.DATASETS.index(c['dataset']),
        -c['original_point']['rate_rps'] if c['stage'].startswith('screen') else c['original_point']['rate_rps']))
    p.need(len(cells) == len({c['cell_id'] for c in cells}) == 220, 'new declaration domain mismatch')
    p.need(Counter(c['arm'] for c in cells) == {'fixed2': 40, 'dynamic': 180}, 'new arm counts mismatch')
    return dict(schema='main-slo-improvement-work-v7', implementation_series='v7', prior_v4_v5_v6_observations_are_separate_development_evidence=True, created_s=time.time(),
        deadline_s=p.DEADLINE, authorization='User approved implementation of the three-model improvement plan',
        criterion='work_complete AND E_PDB <= E_baseline AND SLO_PDB >= min(.90,SLO_baseline)',
        original_snapshot_only=True, baseline_rerun=False, seed=701,
        arrival_window_s=100, request_hard_timeout_s=120, drain_after_arrival_window_s=120,
        automatic_retries=False, independent_arrival_seeds=False,
        fixed2_screen_count=40, dynamic_full_grid_count=180, cells=cells, sources=sources,
        dynamic_development_trajectories=dict(separate_from_main=True, phase_seconds=[300, 300, 300],
            shape_domains=list(p.DATASETS), arms=['fixed2', 'dynamic'], paired_traces_per_shape=1, require_actual_transitions=True),
        calibration=dict(repeats=3, initial_instances=2, minimum_instances=2,
            maximum_instances={'7b': 8, '14b': 8, '32b': 4}, initially_allowed_transitions=[],
            first_transition_to_calibrate='2<->3', empirical_bounds_not_statistical_guarantees=True))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    declaration = build()
    if args.out:
        p.write(args.out, declaration, exclusive=True)
        print({'path': str(args.out), 'sha256': p.sha(args.out), 'cells': len(declaration['cells'])})
    else:
        print({'cpu_only': True, 'cells': len(declaration['cells']), 'deadline_s': p.DEADLINE})

if __name__ == '__main__':
    main()
