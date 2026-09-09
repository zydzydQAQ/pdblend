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

ROOT=Path(__file__).resolve().parent
HOST=Path(json.loads((ROOT/'binding.json').read_text())['host_release'])


def owned_dispatch(method,url,kwargs):
    parsed=urlparse(str(url));rid=(kwargs.get('headers') or {}).get('X-Request-Id')
    if method.upper()!='POST' or parsed.path!='/v1/completions' or parsed.port not in (33500,33501):return None
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
    import benchmarks.scripts.bench_vllm as benchmark
    from epoch import EpochGuard
    limits=json.loads((operation/'limits.json').read_text())
    original_headers=benchmark.evaluation_headers
    guard=EpochGuard(limits,operation/'actual_epoch_gate.json',original_headers)
    benchmark.evaluation_headers=guard
    original=aiohttp.ClientSession._request
    with (operation/'dispatch.jsonl').open('x',buffering=1) as log:
        async def request(self,method,url,**kwargs):
            record=owned_dispatch(method,url,kwargs)
            if record is not None:log.write(json.dumps(record)+'\n')
            return await original(self,method,url,**kwargs)
        aiohttp.ClientSession._request=request
        try:
            result=await run_cell(cell_arguments(row))
            result.update(slo_protocol=row['slo_protocol'],declared_dataset_slo={k:row[k] for k in ('slo_ttft_s','slo_tpot_s')})
            target=ROOT/'cells'/cell_id/'summary.json'
            target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            return bool(result.get('measurement_valid') and result.get('post_measurement_cleanup',{}).get('cleanup_complete'))
        finally:
            aiohttp.ClientSession._request=original
            benchmark.evaluation_headers=original_headers


if __name__=='__main__':raise SystemExit(0 if asyncio.run(main(sys.argv[1])) else 1)
