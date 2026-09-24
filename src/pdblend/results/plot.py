"""Plot qualified runs.csv rows using complete common measurement identity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .catalog import comparison_key, sha


def comparison_groups(rows, systems):
    groups={}
    for row in rows:
        try:key=comparison_key(row)
        except ValueError:continue
        if key is None or row.get('system') not in systems:continue
        groups.setdefault(key,{}).setdefault(row['system'],[]).append(row)
    qualified,excluded=[],[]
    for key,group in groups.items():
        if set(group)!=set(systems) or any(len(values)!=1 for values in group.values()):
            excluded.append(dict(key=list(key),reason='missing_system_or_ambiguous_attempt',
                                 counts={system:len(values) for system,values in group.items()}))
        else:qualified.append([group[system][0] for system in systems])
    return qualified,excluded


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=Path('results/runs.csv'))
    parser.add_argument('--out',type=Path,default=Path('results/plots'))
    parser.add_argument('--systems',nargs='+',default=['mixed','distserve','dynamollm','ecoserve','pdblend'])
    parser.add_argument('--metric',choices=['j_per_good_token','j_per_token','energy_total_j','joint_slo_rate'],default='j_per_good_token')
    args=parser.parse_args(argv)
    with args.csv.open(newline='') as handle:rows=list(csv.DictReader(handle))
    groups,excluded=comparison_groups(rows,args.systems)
    args.out.mkdir(parents=True,exist_ok=True)
    report=dict(source=str(args.csv.resolve()),source_sha256=sha(args.csv),groups=len(groups),excluded=excluded,plots=[])
    for group in groups:
        if args.metric != 'joint_slo_rate' and (any(row.get('energy_scope') not in
                ('explicit_total','full_lifecycle') or row.get('energy_total_j','')=='' for row in group)
                or len({row.get('energy_scope') for row in group})!=1):
            report['excluded'].append(dict(run_ids=[row['run_id'] for row in group],reason='total_energy_scope_unproven_or_mismatched'))
            continue
        if any(row.get(args.metric,'')=='' for row in group):
            report['excluded'].append(dict(run_ids=[row['run_id'] for row in group],reason='missing_metric'))
            continue
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        key=comparison_key(group[0])
        identity=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
        path=args.out/f'{identity}-{args.metric}.pdf'
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(args.systems,[float(row[args.metric]) for row in group])
        ax.set_ylabel(args.metric)
        first=group[0]
        ax.set_title(f"{first['model_id']} · {first['dataset']} · seed {first['seed']} · {first['duration_s']} s")
        fig.tight_layout();fig.savefig(path);plt.close(fig)
        report['plots'].append(dict(path=str(path.resolve()),run_ids=[row['run_id'] for row in group],comparison_identity=list(key)))
    (args.out/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
