#!/usr/bin/env python3
"""Create an explicit old-baseline registry from a bound handoff inventory.

Host-only, no queue or runtime operations; all prior receipt metrics stay fixed.
"""
import argparse
from pathlib import Path
from pdblend.bench.comparison_trace_equivalence import binding, load_bound, make_registry, write_new


def prepare(history_path, out):
    history_ref=binding(history_path);history=load_bound(history_ref)
    candidates=[dict(point=binding(Path(r['path']).parent/'point.json'),receipt=r)
                for r in history['completed_receipts']]
    registry=make_registry(candidates,history_inventory=history_ref)
    return write_new(out,registry)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-inventory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(prepare(args.history_inventory,args.out))
