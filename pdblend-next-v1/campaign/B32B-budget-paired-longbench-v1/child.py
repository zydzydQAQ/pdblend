"""One original 64-request trace through the unchanged frozen schema3 host cell."""
import asyncio
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from ecopadg.serving.cell import run_cell
from ecopadg.serving.runtime import Controller

ROOT=Path(__file__).resolve().parent
HOST=ROOT.parents[1]/'releases/io-v1.2.1-runtime'


async def main(tokens):
    assert tokens in (8192,2048)
    assert Path(inspect.getfile(run_cell)).resolve()==HOST/'src/ecopadg/serving/cell.py'
    assert Path(inspect.getfile(Controller)).resolve()==HOST/'src/ecopadg/serving/runtime.py'
    path=ROOT/f'budget{tokens}.config.json';config=json.loads(path.read_text())
    probe=Controller(config)
    try:
        assert probe.strategy=='pdblend-joint'
        assert probe.planner.independent_idle_mixed_on_stale_tail is True
        assert not probe.eco_scheduler and not probe.dynamo_scheduler and not probe.pd_topology
    finally:await probe.planning_executor.close()
    (ROOT/f'planner-preflight-{tokens}.json').write_text(json.dumps(dict(host=str(HOST),
        output_prior=config['output_prior'],strategy=probe.strategy,fallback=True,allow_pd=False,
        dynamic_pools=False,temporal_scheduler=False))+'\n')
    result=await run_cell(SimpleNamespace(config=path,trace=ROOT.parent/'B32B-io-v1/longbench.trace.json',
        out=ROOT/f'cell-longbench-budget{tokens}',dataset='longbench',load='pilot',seed=11,split='development',
        strategy=None,freeze=None,mechanisms=None,slo_ttft_s=None,slo_tpot_s=None,timeout=120))
    return bool(result.get('measurement_valid') and result.get('post_measurement_cleanup',{}).get('cleanup_complete'))


if __name__=='__main__':raise SystemExit(0 if asyncio.run(main(int(sys.argv[1]))) else 1)
