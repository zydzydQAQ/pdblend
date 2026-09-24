#!/usr/bin/env python3
"""Prepare only: immutable PD32 Poisson layout calibration, never GPU/enqueue."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pdblend.profile.collection.native_timing_plan import binding
from pdblend.profile.collection.native_runtime_collect import write_new
from pdblend.profile.collection.native_layout_energy import build_layout_plan


def prepare(out,ledger,provenance,*,frequency_domain=None):
    out=Path(out);plan=build_layout_plan(binding(ledger),binding(provenance),
        frequency_domain_ref=binding(frequency_domain) if frequency_domain is not None else None)
    out.mkdir(parents=True,exist_ok=False);reference=write_new(out/'plan.json',plan)
    return write_new(out/'review.json',dict(schema='pdblend-native-layout-preparation/v1',plan=reference,
        prepared_only=True,hardware_executed=False,enqueued=False,formal_eligible=False,
        training_windows=sum(p['purpose']=='training' for p in plan['points']),
        independent_150s_holdout_windows=sum(p['purpose']=='holdout' for p in plan['points']),
        minimum_service_s=plan['training_service_s']+plan['holdout_service_s'],
        excludes_load_drain_tail_and_timing_seconds=True,
        order=['qualified_native_timing_runtime_replay','training','freeze_candidate',
               'full_12_group_2_frequency_selection_replay','freeze_selection','holdout','independent_component_replay'],
        scope='32B only; canonical M4/TP2; Poisson; 3 datasets; 4 confirmed rate scales; full 8 physical boards'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True);parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frequency-domain',type=Path,help='Explicit new PD32 calibration domain; never relabel old observations')
    args=parser.parse_args()
    print(prepare(args.out,args.ledger,args.bindings,frequency_domain=args.frequency_domain))
