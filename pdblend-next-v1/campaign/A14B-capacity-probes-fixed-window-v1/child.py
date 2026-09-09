"""One unchanged schema3 cell with durable pre-dispatch request ownership records."""
import asyncio
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import aiohttp
from ecopadg.serving.cell import run_cell
from ecopadg.serving.runtime import Controller
from benchmarks.scripts import bench_vllm

ROOT=Path(__file__).resolve().parent
HOST=ROOT.parents[1]/'releases/io-v1.2.1-fixed-window-v2-runtime'


class EpochGate:
    def __init__(self, original, limits, record):
        self.original, self.limits, self.record = original, limits, record
        self.checked = False

    def __call__(self, api_base, protocol, arrival_s, dispatch_s):
        if not self.checked:
            if (protocol != bench_vllm.EVALUATION_V3 or arrival_s != dispatch_s
                    or arrival_s > self.limits['latest_arrival_epoch_s']):
                raise RuntimeError('actual arrival epoch exceeded reserved phase startup budget')
            self.record(dict(actual_arrival_epoch_s=arrival_s,
                             latest_arrival_epoch_s=self.limits['latest_arrival_epoch_s'],
                             checked_before_any_request_dispatch=True))
            self.checked = True
        return self.original(api_base, protocol, arrival_s, dispatch_s)


def owned_dispatch(method,url,kwargs):
    parsed=urlparse(str(url));rid=(kwargs.get('headers') or {}).get('X-Request-Id')
    if method.upper()!='POST' or parsed.path!='/v1/completions' or parsed.port not in (24306,24307):return None
    if not (isinstance(rid,str) and len(rid)==32 and all(c in '0123456789abcdef' for c in rid)):
        raise RuntimeError('unexpected mixed request identity before dispatch')
    return dict(port=parsed.port,request_id=rid,started_s=time.time())


def cell_arguments(row):
    return SimpleNamespace(config=ROOT/'inputs/controller.fixed.json',trace=Path(row['trace']),
        out=ROOT/'cells'/row['cell_id'],dataset=row['dataset'],load=row['load'],seed=row['seed'],split='development',
        strategy=None,freeze=None,mechanisms=None,slo_ttft_s=row['slo_ttft_s'],slo_tpot_s=row['slo_tpot_s'],slo_scale=row['slo_scale'],timeout=120)


async def main(cell_id):
    assert Path(inspect.getfile(run_cell)).resolve()==HOST/'src/ecopadg/serving/cell.py'
    assert Path(inspect.getfile(Controller)).resolve()==HOST/'src/ecopadg/serving/runtime.py'
    spec=json.loads((ROOT/'runspec.json').read_text());row=next(r for r in spec['cells'] if r['cell_id']==cell_id)
    operation=ROOT/'operations'/cell_id
    original=aiohttp.ClientSession._request
    original_headers=bench_vllm.evaluation_headers
    limits=json.loads((operation/'deadline-limits.json').read_text())
    gate=EpochGate(original_headers,limits,lambda value:
        (operation/'actual_epoch_gate.json').write_text(json.dumps(value,indent=2)+'\n'))
    bench_vllm.evaluation_headers=gate
    with (operation/'dispatch.jsonl').open('x',buffering=1) as log:
        async def request(self,method,url,**kwargs):
            record=owned_dispatch(method,url,kwargs)
            if record is not None:log.write(json.dumps(record)+'\n')
            return await original(self,method,url,**kwargs)
        aiohttp.ClientSession._request=request
        try:
            result=await run_cell(cell_arguments(row))
            if not gate.checked:raise RuntimeError('actual benchmark epoch gate did not execute')
            result.update(slo_protocol=row['slo_protocol'],declared_dataset_slo={k:row[k] for k in ('slo_ttft_s','slo_tpot_s')})
            target=ROOT/'cells'/cell_id/'summary.json'
            target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            return bool(result.get('measurement_valid') and result.get('post_measurement_cleanup',{}).get('cleanup_complete'))
        finally:
            aiohttp.ClientSession._request=original
            bench_vllm.evaluation_headers=original_headers


if __name__=='__main__':raise SystemExit(0 if asyncio.run(main(sys.argv[1])) else 1)
