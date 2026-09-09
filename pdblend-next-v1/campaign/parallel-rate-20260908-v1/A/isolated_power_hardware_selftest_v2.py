"""Exclusive read-only all-eight-GPU observer qualification; no hardware writes."""
import asyncio, hashlib, json, sys, time
from pathlib import Path
A=Path(__file__).resolve().parent; R=A.parent
CODE=A/'load-p8-isolated-code-003'; HOST=R/'hosts/14b-capacity-p8'
OUT=A/'isolated-power-hardware-selftest-002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
sys.path[:0]=[str(CODE),str(HOST/'src'),str(HOST),'/root/workspace/pdblend/.runtime-deps']
from ecopadg.serving.campaign import node_lease
from meter_evidence import install
async def measure():
    from capacity_backend import TransitionMeter
    meter=await TransitionMeter(OUT/'raw').start()
    await asyncio.sleep(8)
    result=await meter.finish()
    assert result['measurement_valid'] is True
    assert result['gpu_indices']==list(range(8)) and result['energy_j']>0
    assert result['duration_s']>=8 and result['power_observer_stopped']
    return result
if __name__=='__main__':
    with node_lease():
        OUT.mkdir()
        finish=install(OUT/'isolated-samplers',ref(HOST/'manifest.json'),
                       ref(A/'isolated-power-v2/manifest.json'),
                       ref(A/'dynamic-execution-isolated-power-002/sampler_hooks.py'))
        result=None
        try:result=asyncio.run(measure())
        finally:finish()
        terminal=json.loads((OUT/'isolated-observers-terminal.json').read_text())
        assert terminal['complete']
        receipt=dict(passed=True,read_only=True,hardware_writes=False,new_requests=0,
                     result=result,terminal=ref(OUT/'isolated-observers-terminal.json'),
                     source=ref(__file__),finished_s=time.time())
        (OUT/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(validation=ref(OUT/'validation.json'),energy_j=result['energy_j'],duration_s=result['duration_s'])))
