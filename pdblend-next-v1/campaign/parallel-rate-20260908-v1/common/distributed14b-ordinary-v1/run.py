"""Bounded original deterministic native gate after distributed deployment."""
import argparse, asyncio, json, os, signal, sys
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / 'distributed14b-deployment-v1'
sys.path.insert(0, str(DEPLOY))
import deploy

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--deployment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    receipt = deploy.read(a.deployment / 'deployment-receipt.json')
    deploy.require(receipt['complete'] and receipt['measurement_valid'] and not receipt['errors'], 'deployment not qualified')
    br = receipt['binding_base']
    deploy.require(deploy.sha(br['path']) == br['sha256'], 'deployment binding changed')
    binding = deploy.read(br['path'])
    common = deploy.adapter.load_runtime(binding['host_release'], Path(binding['executor']).parent)
    common.validate_binding(binding)
    if not a.run:
        print(json.dumps(dict(cpu_only=True, binding=br))); return
    deploy.require('PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh owner required')
    from ecopadg.serving.campaign import node_lease
    async def execute(lease):
        task = asyncio.current_task(); loop = asyncio.get_running_loop(); cancelled = False
        def stop():
            nonlocal cancelled
            if not cancelled:
                cancelled = True; task.cancel()
        for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, stop)
        result = await deploy.measured_ordinary(common, binding, a.out, lease=lease)
        deploy.write(a.out / 'lineage.json', dict(deployment=deploy.reference(a.deployment / 'deployment-receipt.json'), binding=br, source=deploy.reference(Path(__file__)), deployment_source=deploy.reference(DEPLOY / 'manifest.json')))
        return result
    with node_lease() as lease: result = asyncio.run(execute(lease))
    print(json.dumps({k: result.get(k) for k in ('passed', 'complete', 'measurement_valid', 'errors')}))

if __name__ == '__main__': main()
