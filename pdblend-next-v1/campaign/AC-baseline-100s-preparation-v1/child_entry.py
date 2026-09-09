"""Small run_cell entry for a separately frozen/leased five-system wrapper.

The required preflight hook performs the wrapper's complete read-only live gate.
Outer request ownership, cancellation, power, and native cleanup belong to the
parent wrapper and remain required on any exception from this function.
"""
import argparse
import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from bind import PROTOCOL,SLOS,read,require,sha

def cell_args(config,trace,out,*,scale):
    c=read(config);t=read(trace);dataset=t['dataset']
    require(c.get('evaluation_protocol')=='evaluation-v3' and c.get('measurement_window_protocol')==PROTOCOL,
        'new five-system protocol configuration required')
    require(t.get('protocol_id')==PROTOCOL and t.get('measurement_schema')==3 and t.get('seed')==701
        and t.get('arrival_window_s')==100 and t.get('duration_s')==100 and t.get('split')=='development',
        'wrong actual paired trace protocol')
    require(c.get('arrival_window_s')==100 and scale in (.5,1.,2.) and c.get('slo_scale')==scale,
        'config/trace window or scale differs')
    require((c.get('slo_ttft_s'),c.get('slo_tpot_s'))==tuple(x*scale for x in SLOS[dataset]),
        'actual Controller SLO differs from paired scoring SLO')
    return SimpleNamespace(config=Path(config),trace=Path(trace),out=Path(out),strategy=None,
        split='development',dataset=dataset,load=t['load'],seed=701,timeout=120.,
        freeze=None,mechanisms=None,slo_ttft_s=c['slo_ttft_s'],slo_tpot_s=c['slo_tpot_s'],
        slo_scale=scale,evaluation_plan=None,evaluation_cell_id=None)

async def execute(args,host,preflight):
    from ecopadg.serving.cell import run_cell
    require(Path(inspect.getfile(run_cell)).resolve()==Path(host).resolve()/'src/ecopadg/serving/cell.py',
        'wrong frozen 100-second host import')
    proof=await preflight(read(args.config),read(args.trace))
    require(isinstance(proof,dict) and proof.get('identity_verified') is True,
        'complete source/model/container preflight did not pass; zero Controller construction')
    return await run_cell(args)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','trace','out','host','preflight-hook'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--preflight-sha256',required=True);p.add_argument('--scale',type=float,default=1.)
    a=p.parse_args();require(sha(a.preflight_hook)==a.preflight_sha256,'preflight hook differs from frozen wrapper')
    spec=importlib.util.spec_from_file_location('five_system_live_preflight',a.preflight_hook)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    result=asyncio.run(execute(cell_args(a.config,a.trace,a.out,scale=a.scale),a.host,module.verify_before_dispatch))
    print(json.dumps(dict(measurement_valid=result.get('measurement_valid'),energy_j=result.get('energy_j'))))

if __name__=='__main__':main()
