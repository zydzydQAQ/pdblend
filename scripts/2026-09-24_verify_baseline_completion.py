#!/usr/bin/env python3
"""Read-only final completion predicate for the twenty authorized baseline gaps.

Pass candidate receipt files explicitly (or bounded session directories).
Returns exit code 2 while any gap still lacks a matching complete measurement.
This neither resubmits failed jobs nor treats queue terminal status as success.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def load_builder():
    path=Path(__file__).with_name('2026-09-24_prepare_saturation_round.py')
    spec=importlib.util.spec_from_file_location('baseline_completion_builder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify(campaign_path, receipts=(), sessions=()):
    builder=load_builder();campaign=builder.load_bound(builder.binding(campaign_path))
    contract=campaign['baseline_completion_contract']
    ownership=builder.load_bound(campaign['baseline_ownership'])
    paths={Path(p).resolve() for p in receipts}
    for ref in ownership.get('completion_receipts',[]):
        builder.load_bound(ref);paths.add(Path(ref['path']).resolve())
    for root in sessions:
        # A session root is an explicitly supplied bounded attempt directory;
        # never search results recursively or read outcomes/power samples.
        paths.update(Path(root).glob('windows/*/receipt.json'))
    refs=[builder.binding(p) for p in sorted(paths)]
    report=builder.baseline_ownership(contract['authorized_gaps'],ownership['external_campaigns'],
                                     completion_refs=refs)
    return dict(schema='authorized-baseline-completion/v1',campaign=builder.binding(campaign_path),
        authorized_count=report['authorized_count'],completed_count=report['completed_count'],
        all_complete=report['all_complete'],rows=report['rows'],
        queue_terminal_used_as_completion=False,observed_receipts=len(refs))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--receipt',type=Path,action='append',default=[])
    p.add_argument('--session',type=Path,action='append',default=[])
    args=p.parse_args();report=verify(args.campaign,args.receipt,args.session)
    print(json.dumps(report,indent=2,sort_keys=True))
    raise SystemExit(0 if report['all_complete'] else 2)


if __name__=='__main__':main()
