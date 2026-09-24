#!/usr/bin/env python3
"""Capture NEW post-completion timing evidence and independently replay it."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from pdblend.profile.collection.native_timing_replay import capture_evidence,replay_evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',type=Path)
    p.add_argument('--queue',type=Path)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--evidence-sha256',help='Required for replay of an existing externally bound manifest')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--path-map',action='append',default=[],metavar='RECORDED=ACTUAL')
    args=p.parse_args();mapping=[tuple(v.split('=',1)) for v in args.path_map]
    if args.out.exists():raise FileExistsError('refusing to overwrite native timing replay output')
    if args.attempt:
        if not args.queue:raise ValueError('--queue required when capturing a completed attempt')
        ref=capture_evidence(args.attempt,args.queue,args.evidence,path_map=mapping)
    else:
        if not args.evidence_sha256:raise ValueError('externally bound --evidence-sha256 required')
        ref=dict(path=str(args.evidence.resolve()),sha256=args.evidence_sha256)
    value=replay_evidence(ref,path_map=mapping)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(out=str(args.out),component_qualified=(value.get('component') or {}).get('component_qualified',False),
        replayed_windows=value['replayed_windows'],formal_eligible=False)))


if __name__=='__main__':main()
