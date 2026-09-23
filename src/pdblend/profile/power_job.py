"""Lease-worker adapter for an explicitly coordinated power-only cohort.

The underlying raw and completion receipts remain unchanged. Queue completion
means measurement completion; power/timing/composite qualification stay separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .power_calibration import digest,run
from .wave import atomic_json


def require_paired_cohort():
    root=os.environ.get('PDBLEND_PROFILE_WAVE');member=os.environ.get('PDBLEND_PROFILE_MEMBER')
    if not root or not member:
        raise ValueError('power jobs require a new explicit paired ProfileWave')
    wave=json.loads((Path(root)/'wave.json').read_text())
    members=wave.get('members',[])
    if (len(members)!=2 or len(set(members))!=2 or member not in members or
            not wave.get('coordinator') or not wave.get('cohort_id') or
            wave.get('purpose')!='independent_power_holdouts_4_plus_4'):
        raise ValueError('power job cannot claim single-member, native-peer or existing quad qualification')
    return wave


def queue_receipt(result,out):
    path=out/'completion.json'
    if not path.is_file() or json.loads(path.read_text())!=result:
        raise ValueError('underlying power completion must exist unchanged')
    complete=result.get('complete') is True and result.get('status')=='completed'
    power=result.get('power_passed');timing=result.get('reused_timing_passed')
    if complete:
        if (type(power) is not bool or type(timing) is not bool or
                result.get('calibration_components_passed') is not (power and timing)):
            raise ValueError('power completion has inconsistent qualification fields')
        if digest(out/'composite-audit.json')!=result.get('composite_audit_sha256'):
            raise ValueError('power completion composite receipt checksum mismatch')
    receipt=dict(schema=1,status='passed' if complete else 'failed',complete=complete,
        queue_receipt_semantics='measurement_completion_only; qualification_is_reported_separately',
        calibration_status='passed' if complete and power and timing else 'failed',
        power_status='passed' if complete and power else 'failed',
        reused_timing_status='passed' if complete and timing else 'failed',
        composite_status='passed' if complete and power and timing else 'failed',
        formal_eligible=False,energy_comparable=False,fit_performed=False,
        underlying_completion_sha256=digest(path),composite_audit_sha256=result.get('composite_audit_sha256'),
        original_completion_unchanged=True,error=result.get('error'))
    if result.get('timing_overlay_requested'):
        receipt['timing_overlay_requested']=True
        receipt['timing_overlay']=result.get('timing_overlay',dict(complete=False,status='not_completed'))
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True);p.add_argument('--base-port',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--timing-package',type=Path);a=p.parse_args()
    wave=require_paired_cohort()
    result=run(package=a.package,model_path=a.model,gpus=a.gpus,base_port=a.base_port,out=a.out,
               timing_package=a.timing_package)
    receipt=queue_receipt(result,a.out)
    receipt['cohort_id']=wave['cohort_id']
    atomic_json(a.out/'queue-completion.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    raise SystemExit(0 if receipt['complete'] else 1)


if __name__=='__main__':main()
