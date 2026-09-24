#!/usr/bin/env python3
"""Prepare CPU-only model-owned v2 timing plans; no collector or queue changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True

from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_timing_plan_v2 import build_plan,DATASETS
from pdblend.bench.resident_session import write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ledger','provenance','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--corpus-root',type=Path,default=ROOT/'datasets/prepared')
    parser.add_argument('--models',nargs='+',choices=['7b','14b','32b'],default=['7b','14b','32b'])
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('new immutable v2 plan directory required')
    out.mkdir(parents=True);plans={}
    ledger,provenance=binding(args.ledger),binding(args.provenance)
    for size in args.models:
        model='Qwen2.5-'+size.upper()+'-Instruct'
        corpus=args.corpus_root/('2026-09-22-'+size+'-v1')
        refs={dataset:binding(corpus/(dataset+'.json')) for dataset in DATASETS}
        plan=build_plan(ledger,provenance,refs,model_id=model)
        path=out/(size+'-point-plan.json');write_new(path,plan)
        plans[model]=dict(point_plan=binding(path),tp=plan['tp'],resident_instances=plan['resident_instances'],
            points=len(plan['points']),windows=sum(p['repeats'] for p in plan['points']),
            unsupported_query_shapes=len(plan['unsupported_queries']),blocked_ledger_entries=len(plan['blocked_ledger_entries']))
    value=dict(schema='pdblend-native-timing-v2-plan-preflight/v1',created_s=time.time(),models=plans,
        hardware_executed=False,enqueued=False,formal_eligible=False,collector_integration_required=True,
        builder=binding(__file__),implementation={name:binding(ROOT/'src/pdblend/profile/collection'/name)
            for name in ('native_timing_plan_v2.py','native_timing_capacity.py')},
        next_step='freeze a separate capacity-aware collector and all-model raw replay; v1 remains unchanged')
    write_new(out/'preflight.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
