"""Four resident fixed-TP EcoServe instances on an exact GPU lease.

This invokes the independent controller and retains live-KV/action receipts.
The explicit membership actions qualify primitives, not automatic scaling.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from pdblend.engine.launcher import Fleet
from pdblend.bench.metering import Gpus
from pdblend_baselines.native_profile import collect
from pdblend_baselines.ecoserve.mechanism_four import run
from .probe import NativeSpec


def specs_for(model, tp, gpus, base_port):
    legal = 2 if Path(model).name == 'Qwen2.5-32B-Instruct' else 1
    if Path(model).name not in ('Qwen2.5-7B-Instruct','Qwen2.5-14B-Instruct','Qwen2.5-32B-Instruct'):
        raise ValueError('explicit campaign model required')
    if tp != legal or len(gpus) != 4*tp or len(set(gpus)) != len(gpus) or len(gpus)>8:
        raise ValueError('four nonoverlapping fixed-TP groups within eight GPUs required')
    return [NativeSpec('eco'+str(i),tuple(gpus[i*tp:(i+1)*tp]),base_port+16*i,
                       model,tp=tp,max_num_seqs=32,extra_args=('--enforce-eager',)) for i in range(4)]


def pressure_count(profile, *, ttft_s=5.):
    """Bounded real demand, chosen from this deployment's own prefill CSV."""
    rows=[row for row in profile['rows'] if row['input_tokens']==7168]
    if len(rows)!=1 or not math.isfinite(rows[0]['minimum_ms']) or rows[0]['minimum_ms']<=0:
        raise ValueError('one positive independently measured 7168-token prefill is required')
    return min(12, math.ceil(ttft_s*1000/rows[0]['minimum_ms'])+3)


def execute(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    specs=specs_for(args.model,args.tp,args.gpus,args.base_port)
    result=dict(system='ecoserve',model_id=Path(args.model).name,tp=args.tp,pp=1,seed=701,
        started_s=time.time(),status='running',complete=False,formal_eligible=False,
        complete_reproduction=False,automatic_policy_triggered=False,energy_comparable=False,
        source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'),image_digest=os.environ.get('PDBLEND_IMAGE_ID'),
        specs=[asdict(spec) for spec in specs])
    meter=Gpus(args.gpus);sampler=meter.sampler(interval_s=.1);fleet=Fleet(specs,out/'logs')
    try:
        sampler.start()
        # Stagger weight loads; completed instances remain resident.
        result['startup']={}
        for spec in specs:
            instance=fleet[spec.instance_id];instance.start()
            result['startup'][spec.instance_id]=instance.wait_ready(timeout_s=1200)
        from pdblend_baselines.native_profile import call
        caps=[call(spec.base_url,'GET','/baseline/capability') for spec in specs]
        result['capabilities']=caps
        for spec,cap in zip(specs,caps):
            if not cap.get('supported') or cap.get('tp')!=args.tp or cap.get('pp')!=1:
                raise RuntimeError('four-member native capability incomplete')
            for key in ('model_id','model_hash','tokenizer_hash','image_digest','source_revision'):
                if not cap.get(key) or cap[key]!=caps[0].get(key):
                    raise RuntimeError('four-member identity differs: '+key)
        csv=out/'ecoserve-prefill.csv'
        profile=collect(SimpleNamespace(url=specs[0].base_url,system='ecoserve',
            model=result['model_id'],out=str(csv),frequency=2520,
            **{key:caps[0][key] for key in ('model_hash','tokenizer_hash','engine_version',
                                           'image_digest','source_revision','gpu_uuids')}))
        result['profile_metadata']=profile['metadata']
        config=dict(instances=[dict(id=s.instance_id,gpus=list(s.gpus)) for s in specs],
            eco_prefill_csv=str(csv),slo_ttft_s=5.,slo_tpot_s=.15,eco_initial_instances=3,
            eco_macro_lower=2,eco_macro_upper=3,eco_active_frequency_mhz=2520,
            eco_state_poll_s=.05,eco_scale_period_s=5,request_timeout_s=180,eco_drain_timeout_s=120,
            eco_probe_pressure_requests=pressure_count(profile))
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        mechanism=asyncio.run(run(config,{s.instance_id:s.base_url for s in specs},out/'mechanism.json'))
        result.update(status=mechanism['status'],complete=mechanism['complete'],
            mechanism_validated=mechanism['mechanism_validated'],checks=mechanism['checks'],
            missing_required_actions=mechanism['missing_required_actions'],errors=mechanism['errors'])
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
    finally:
        from .cleanup import cleanup_owned
        result['cleanup_errors']=cleanup_owned(fleet,meter,sampler)
        if result['cleanup_errors']:
            result.update(status='failed',complete=False)
        result.update(finished_s=time.time(),energy_j=sampler.total_energy_j(),sampler_error=sampler.error)
        (out/'power.json').write_text(json.dumps(dict(samples=sampler.samples,frequency_samples=sampler.frequency_samples)))
        (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--tp',type=int,required=True)
    parser.add_argument('--gpus',required=True);parser.add_argument('--base-port',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.gpus=[int(x) for x in args.gpus.split(',')]
    result=execute(args)
    print(json.dumps({key:result.get(key) for key in ('status','model_id','tp','error','missing_required_actions')},indent=2),flush=True)
    return 0 if result['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
