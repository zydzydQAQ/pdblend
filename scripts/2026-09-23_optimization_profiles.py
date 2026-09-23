#!/usr/bin/env python3
"""Publish explicit development consumers for immutable component versions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from pdblend.profile.calibration.optimization_profiles import digest
from pdblend.profile.calibration.power_calibration import write_immutable
from pdblend.profile.calibration.runtime_components import audit_runtime
from pdblend.profile.query.versions import load_profile


def publish(registry,out):
    registry,out=Path(registry).resolve(),Path(out).resolve()
    rows=[]
    def binding(path):return dict(path=str(path),sha256=digest(path))
    for row in json.loads(registry.read_text())['versions']:
        target=out/row['version_id']
        base=Path(row['evidence']['power_candidate']['path'])
        raw=Path(row['original_inputs']['training_raw']['path'])
        audit=target/'runtime-audit.json'
        measured=audit_runtime(base,raw,out=audit)
        selection=target/'selection.json'
        write_immutable(selection,dict(kind='pdblend_profile_selection_v1',registry=str(registry),
            version_id=row['version_id'],runtime_base=dict(profile=binding(base),raw=binding(raw),audit=binding(audit))))
        loaded=load_profile(selection,**{k:row[k] for k in ('system','model_id','tp','pp')})
        loaded.model.require_runtime_components('capacity','static','transfer','clock_transition')
        receipt=dict(selection=binding(selection),runtime_audit=binding(audit),
            manifest=loaded.manifest_fields(),consumer_queries=dict(kv_capacity_tokens=loaded.model.kv_capacity_tokens,
                parked_power_w=loaded.model.static_power_w('parked'),transfer_512_s=loaded.model.transfer_seconds(512),
                clock_transition_s=loaded.model.freq_switch_s,decode_32_ctx1200_f1500_s=loaded.model.step_seconds(32,1200,1500),
                decode_32_ctx1200_f1500_w=loaded.model.decode_power_w(32,1500,ctx=1200)),
            hardware_executed=False,formal_eligible=False,independent_runtime_holdout_passed=False)
        write_immutable(target/'consumer-receipt.json',receipt)
        rows.append(dict(version_id=row['version_id'],selection=str(selection),
            receipt=str(target/'consumer-receipt.json'),runtime_measurements_passed=measured['passed']))
    report=dict(versions=rows,hardware_executed=False,formal_eligible=False,immutable_versions_unchanged=True)
    write_immutable(out/'completion.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(publish(args.registry,args.out),indent=2))


if __name__=='__main__':main()
