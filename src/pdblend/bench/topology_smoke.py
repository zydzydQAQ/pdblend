"""Real launch/output/KV smoke for one model and one TP/PP layout.

This is a screening artifact. It never promotes a topology to formal status:
native rank ACK, KV reclaim and cancellation evidence are separate gates.
"""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from ..engine.launcher import Fleet, make_specs
from ..engine.client import EngineClient, PDTransfer
from ..model_registry import ModelRegistry
from .gates import _gate_kv, kv_bytes_per_token, random_prompt
from .metering import Gpus


async def outputs(fleet):
    rows = []
    for inst in fleet.instances.values():
        async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as client:
            for repeat in range(2):
                r = await client.complete(random_prompt(128, 701), 16, f'smoke-{inst.spec.instance_id}-{repeat}', seed=701)
                rows.append(dataclasses.asdict(r))
    return rows


async def cancellation_probe(fleet):
    inst = next(iter(fleet.instances.values()))
    async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as client:
        request_id = f'cancel-smoke-{time.time_ns()}'
        task = asyncio.create_task(client.complete(random_prompt(7168, 2701), 512, request_id))
        await asyncio.sleep(.1)
        receipt = await client.cancel(request_id)
        result = await task
        return {"request_id": request_id, "receipt": receipt,
                "finished": result.finished_s is not None,
                "completion_tokens": result.completion_tokens,
                "native_cancel_supported": bool(receipt.get("supported"))}


def run(model, tp, pp, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec = ModelRegistry('/models').get(model); spec.validate_topology(tp, pp)
    used = list(range((8 // (tp * pp)) * tp * pp))
    # Full-device lease includes boards with no assigned instance.
    specs = make_specs(spec.model_path, used, tp=tp, pp=pp, kv_connector='P2pNcclConnector' if pp == 1 and len(used) // tp >= 2 else None,
                       max_num_seqs=32)
    meter = Gpus(list(range(8)))
    sampler = meter.sampler(interval_s=.1)
    result = dict(schema=1, model_id=spec.model_id, tp=tp, pp=pp, complete=False,
                  status='running', formal_eligible=False, energy_comparable=False,
                  evidence_class='topology_smoke', missing_gates=['native_rank_generation_ack', 'kv_reclaim', 'cancellation_recovery', 'profile_holdout'],
                  source_sha256=os.environ.get('PDBLEND_SOURCE_SHA256'), image_digest=os.environ.get('PDBLEND_IMAGE_ID'),
                  versions={name: importlib.metadata.version(name) for name in ('vllm','torch')},
                  cuda=os.environ.get('CUDA_VERSION'), specs=[dataclasses.asdict(s) for s in specs])
    result['hardware'] = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
    result['topology'] = subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
    fleet = Fleet(specs, out / 'logs')
    try:
        meter.reset_all()
        for gpu in used: meter.set_clock(gpu, 2520)
        sampler.start(); result['started_s'] = time.time()
        result['startup_s'] = fleet.start_all()
        result['startup_finished_s'] = time.time()
        result['outputs'] = asyncio.run(outputs(fleet))
        rows = result['outputs']
        result['repeat_output_match'] = all(rows[i]['text'] == rows[i+1]['text'] for i in range(0,len(rows),2))
        result['ordinary_ok'] = all(not r['error'] and r['completion_tokens'] == 16 for r in rows) and result['repeat_output_match']
        result['cancellation'] = asyncio.run(cancellation_probe(fleet))
        if not result['cancellation']['native_cancel_supported']:
            result['missing_gates'].append('native_cancellation_endpoint')
        if specs[0].kv_connector:
            # Three deterministic repeats are required to distinguish a
            # transfer corruption from a one-off greedy tie.  A single repeat
            # is useful for a quick launch check but cannot qualify KV.
            result['kv'] = asyncio.run(_gate_kv(fleet, PDTransfer('P2pNcclConnector',{s.instance_id:s.zmq_address for s in specs}),
                                               (512,2048,7168),3,16,kv_bytes_per_token(specs[0].model_path)))
            # _gate_kv emits one row per (input length, repeat): three
            # lengths times three repeats, with per-length medians in summary.
            expected_rows = 3 * 3
            summary = result['kv'].get('summary', {})
            result['kv_ok'] = (
                len(result['kv']['rows']) == expected_rows and
                set(summary) == {'512', '2048', '7168'} and
                all(summary[n].get('runs') == 3 and summary[n].get('token_ids_match_all')
                    for n in ('512', '2048', '7168')) and
                all(not r.get('error') and r.get('token_ids_match') for r in result['kv']['rows']))
        else:
            result['kv_status'] = 'unsupported_engine' if pp != 1 else 'insufficient_gpus'
            result['kv_ok'] = None
        result['health'] = [i.health_report() for i in fleet.instances.values()]
        result['status'] = 'passed' if result['ordinary_ok'] and result['kv_ok'] is not False else 'failed'
        result['complete'] = result['status'] == 'passed'
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['service_finished_s'] = time.time()
        fleet.stop_all()
        result['events'] = fleet.events()
        meter.reset_all()
        meter.wait_released(used)
        sampler.stop()
        result['finished_s'] = time.time()
        result['energy_j'] = sampler.total_energy_j()
        result['sampler_error'] = sampler.error
        (out / 'power.json').write_text(json.dumps({'samples':sampler.samples,'frequency_samples':sampler.frequency_samples}))
        if sampler.error: result.update(status='failed', complete=False)
        (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--tp',type=int,required=True); p.add_argument('--pp',type=int,default=1); p.add_argument('--out',required=True)
    a=p.parse_args(); result=run(a.model,a.tp,a.pp,a.out)
    print(json.dumps({k:result.get(k) for k in ('status','model_id','tp','pp','ordinary_ok','kv_ok','error')}),flush=True)
    raise SystemExit(0 if result['complete'] else 1)
if __name__ == '__main__': main()
