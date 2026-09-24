#!/usr/bin/env python3
"""Prepare independent tuning, audit raw receipts, or run inside an owned lease."""
import argparse
import asyncio
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output',required=True,type=Path)
    prep.add_argument('--corpus',required=True,type=Path)
    prep.add_argument('--profile',required=True,type=Path)
    prep.add_argument('--recovery',required=True,type=Path,help='explicit recovery-policy JSON')
    prep.add_argument('--source-root',type=Path)
    prep.add_argument('--seeds',type=int,nargs=3,default=[8801,8802,8803])
    prep.add_argument('--frequencies',type=int,nargs='+',default=[2100,1800,1500,1200,900])
    audit = commands.add_parser('summarize')
    audit.add_argument('--manifest',required=True,type=Path)
    audit.add_argument('--receipts',type=Path,nargs='*',default=[])
    audit.add_argument('--output',required=True,type=Path)
    audit.add_argument('--freeze',type=Path,help='emit floor only when a complete qualified candidate exists')
    run = commands.add_parser('run',help='GPU operation: queue owner must supply its exclusive lease environment')
    run.add_argument('--group',required=True,type=Path)
    run.add_argument('--output',required=True,type=Path)
    run.add_argument('--base-port',type=int,default=8700)
    run.add_argument('--trial-ids',nargs='+')
    args = parser.parse_args()
    from pdblend.bench.resident_session import write_new
    if args.command == 'prepare':
        from pdblend.bench.low_m_tuning import prepare
        manifest = prepare(output=args.output,corpus=args.corpus,profile=args.profile,
            recovery=json.loads(args.recovery.read_text()),source_root=args.source_root,
            seeds=tuple(args.seeds),frequencies=tuple(args.frequencies))
        print(json.dumps(dict(manifest=str(args.output.resolve()/'manifest.json'),trials=len(manifest['trials']),
                              qualification='unmeasured',formal_eligible=False)))
    elif args.command == 'summarize':
        from pdblend.bench.capacity_floor_v2 import summarize
        from pdblend.bench.comparison_campaign import binding
        result = summarize(args.manifest,args.receipts)
        args.output.parent.mkdir(parents=True,exist_ok=True); write_new(args.output,result)
        if args.freeze:
            if not result['selected']:
                raise ValueError('no complete three-seed/stress, zero-miss, full-energy candidate; no floor issued')
            manifest = json.loads(args.manifest.read_text())
            write_new(args.freeze,dict(kind='pdblend_capacity_floor_v2',identity=manifest['identity'],
                tuning_manifest=binding(args.manifest),floors=result['selected'],
                trial_receipts=[row['receipt'] for row in result['rows']],formal_eligible=False))
        print(json.dumps(dict(accepted_trials=len(result['rows']),selected=len(result['selected']),
                              missing_trials=len(result['missing_trial_ids']),formal_eligible=False)))
    else:
        from pdblend.bench.low_m_tuning_runtime import run_group
        result = asyncio.run(run_group(args.group,args.output,base_port=args.base_port,trial_ids=args.trial_ids))
        print(json.dumps(dict(trials=len(result),accepted=sum(r['accepted'] for r in result.values()))))


if __name__ == '__main__':
    main()
