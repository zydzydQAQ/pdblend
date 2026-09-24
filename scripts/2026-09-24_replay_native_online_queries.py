#!/usr/bin/env python3
"""CPU-only native calibration query development; never launches GPU work."""
from pathlib import Path
import argparse
import json

from pdblend.profile.collection.native_timing_plan import binding,read_bound
from pdblend.profile.query.native_online_shadow import build_ledger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--raw',type=Path,action='append',default=[])
    parser.add_argument('--collection',type=Path,help='Completed layout phase receipt; reads its bound windows.')
    parser.add_argument('--phase',choices=['training','holdout'],default='training')
    parser.add_argument('--training-ledger',type=Path)
    parser.add_argument('--priors',type=Path,help='JSON mapping dataset to pre-window training-only prior binding; omitted fields remain missing.')
    parser.add_argument('--period-s',type=float,default=10.)
    parser.add_argument('--max-state-age-s',type=float,default=1.)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    refs=[binding(p) for p in args.raw]
    if args.collection:
        collection=read_bound(binding(args.collection))
        if collection.get('phase')!=args.phase:
            raise ValueError('collection split differs')
        refs += [row['raw'] for row in collection['windows']]
    result=build_ledger(binding(args.plan),refs,source_manifest=binding(args.source_manifest),
        out=args.out,phase=args.phase,training_ledger=binding(args.training_ledger) if args.training_ledger else None,
        prior_refs=json.loads(args.priors.read_text()) if args.priors else None,
        period_s=args.period_s,max_state_age_s=args.max_state_age_s)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
