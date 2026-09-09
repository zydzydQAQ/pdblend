"""One declared 100-second cell, preserving the original serving algorithms."""
import asyncio
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse


async def execute(job_path):
    import aiohttp
    from ecopadg.serving.cell import run_cell
    from benchmarks.scripts import bench_vllm
    job = json.loads(Path(job_path).read_text())
    row = job['row']
    operation = Path(job_path).parent
    original_request = aiohttp.ClientSession._request
    original_headers = bench_vllm.evaluation_headers
    checked = False
    ports = set(job['engine_ports'])

    def epoch_headers(api, protocol, arrival, dispatch):
        nonlocal checked
        if not checked:
            if (protocol != bench_vllm.EVALUATION_V3 or arrival != dispatch
                    or arrival > job['latest_arrival_epoch_s']):
                raise RuntimeError('arrival epoch exceeds declared execution budget')
            (operation / 'actual-epoch.json').write_text(json.dumps(dict(
                arrival_epoch_s=arrival, checked_before_dispatch=True)) + '\n')
            checked = True
        return original_headers(api, protocol, arrival, dispatch)

    with (operation / 'dispatch.jsonl').open('x', buffering=1) as log:
        async def owned_request(self, method, url, **kwargs):
            parsed = urlparse(str(url))
            if method.upper() == 'POST' and parsed.path == '/v1/completions' and parsed.port in ports:
                rid = (kwargs.get('headers') or {}).get('X-Request-Id')
                if not isinstance(rid, str) or not rid or len(rid) > 256:
                    raise RuntimeError('missing engine dispatch ownership identity')
                log.write(json.dumps(dict(port=parsed.port, request_id=rid, dispatched_s=time.time())) + '\n')
            return await original_request(self, method, url, **kwargs)

        aiohttp.ClientSession._request = owned_request
        bench_vllm.evaluation_headers = epoch_headers
        args = SimpleNamespace(config=Path(job['config']), trace=Path(row['trace']),
            out=Path(job['out']), dataset=row['dataset'], load=row['load'],
            seed=row['seed'], split='development', strategy=None, freeze=None,
            mechanisms=None, slo_ttft_s=row['slo_ttft_s'], slo_tpot_s=row['slo_tpot_s'],
            slo_scale=row['slo_scale'], timeout=120)
        try:
            result = await run_cell(args)
            if not checked:
                raise RuntimeError('benchmark epoch gate did not execute')
            return bool(result.get('measurement_valid') and
                result.get('post_measurement_cleanup', {}).get('cleanup_complete'))
        finally:
            aiohttp.ClientSession._request = original_request
            bench_vllm.evaluation_headers = original_headers


if __name__ == '__main__':
    raise SystemExit(0 if asyncio.run(execute(sys.argv[1])) else 1)
