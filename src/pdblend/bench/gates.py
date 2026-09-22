"""Hardware gates G0 (cross-GPU KV transfer) and G1 (parking states) with JSON evidence."""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from ..engine.client import EngineClient, PDTransfer, pd_complete
from ..engine.launcher import Fleet, make_specs
from .metering import Gpus


def random_prompt(n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(1000, 60000) for _ in range(n_tokens)]


def kv_bytes_per_token(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text())
    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return layers * 2 * kv_heads * head_dim * 2


async def _gate_kv(fleet: Fleet, transfer: PDTransfer, lengths, repeats: int, decode_tokens: int,
                   kv_bpt: int) -> dict:
    p, d = list(fleet.instances.values())[:2]
    async with EngineClient(p.spec.instance_id, p.spec.base_url) as pc, \
            EngineClient(d.spec.instance_id, d.spec.base_url) as dc:
        rows = []
        await dc.complete(random_prompt(64, 0), 4, "warm-m")
        await pd_complete(transfer, pc, dc, random_prompt(64, 0), 4, "warm-pd")
        for n in lengths:
            for r in range(repeats):
                prompt = random_prompt(n, 100 * n + r)
                mixed = await dc.complete(prompt, decode_tokens, f"mixed-{n}-{r}")
                pre, dec = await pd_complete(transfer, pc, dc, prompt, decode_tokens, f"pd-{n}-{r}")
                if dec is None:
                    rows.append(dict(input_tokens=n, repeat=r, error=pre.error, stage="prefill"))
                    continue
                row = dict(input_tokens=n, repeat=r, kv_bytes=n * kv_bpt,
                           mixed_ttft_s=mixed.ttft_s, mixed_tpot_s=mixed.tpot_s, mixed_text=mixed.text,
                           prefill_leg_s=pre.finished_s - pre.submitted_s,
                           decode_leg_ttft_s=dec.ttft_s, decode_tpot_s=dec.tpot_s, pd_text=dec.text,
                           pd_ttft_s=(dec.first_token_s - pre.submitted_s) if dec.first_token_s else None,
                           error=dec.error or mixed.error, handoff=pre.kv_transfer_params or pre.request_id)
                if row["pd_ttft_s"] is not None and mixed.ttft_s is not None:
                    row["transfer_overhead_s"] = row["pd_ttft_s"] - mixed.ttft_s
                    row["effective_gbps"] = (n * kv_bpt / max(row["transfer_overhead_s"], 1e-6)) / 1e9
                    row["text_match"] = mixed.text == dec.text
                rows.append(row)
    summary = {}
    for n in lengths:
        ok = [r for r in rows if r["input_tokens"] == n and not r.get("error") and "transfer_overhead_s" in r]
        if ok:
            summary[str(n)] = dict(
                runs=len(ok),
                mixed_ttft_ms=statistics.median(r["mixed_ttft_s"] for r in ok) * 1e3,
                pd_ttft_ms=statistics.median(r["pd_ttft_s"] for r in ok) * 1e3,
                transfer_overhead_ms=statistics.median(r["transfer_overhead_s"] for r in ok) * 1e3,
                effective_gbps=statistics.median(r["effective_gbps"] for r in ok),
                text_match_all=all(r["text_match"] for r in ok))
    return dict(rows=rows, summary=summary)


def gate_kv(model: str, gpus=(0, 1), connector: str = "P2pNcclConnector", lengths=(512, 2048, 7168),
            repeats: int = 3, decode_tokens: int = 16, out: Path = Path("results/v2/gate-g0.json")) -> dict:
    specs = make_specs(model, gpus[:2], tp=1, kv_connector=connector)
    transfer = PDTransfer(connector, {s.instance_id: s.zmq_address for s in specs})
    out.parent.mkdir(parents=True, exist_ok=True)
    kv_bpt = kv_bytes_per_token(specs[0].model_path)
    with Fleet(specs, out.parent / "logs") as fleet:
        startup = fleet.start_all()
        result = asyncio.run(_gate_kv(fleet, transfer, lengths, repeats, decode_tokens, kv_bpt))
        result.update(gate="G0", model=model, connector=connector, gpus=list(gpus), startup_s=startup,
                      kv_bytes_per_token=kv_bpt, events=fleet.events())
    out.write_text(json.dumps(result, indent=1, default=str))
    return result


async def _probe(inst) -> dict:
    async with EngineClient(inst.spec.instance_id, inst.spec.base_url) as c:
        started = time.time()
        r = await c.complete(random_prompt(128, 7), 8, f"probe-{started:.0f}")
        return dict(ok=r.error is None, latency_s=time.time() - started, error=r.error)


def gate_park(model: str, gpu: int = 0, window_s: float = 8.0, out: Path = Path("results/v2/gate-g1.json")) -> dict:
    gpus = Gpus([gpu])
    states = []

    def record(name, **extra):
        m = gpus.settle_and_measure(window_s, settle_s=3.0)
        m.update(state=name, freq_mhz=gpus.current_freq(gpu), mem_mhz=gpus.mem_freq(gpu), **extra)
        states.append(m)
        return m

    gpus.reset_clock(gpu)
    record("no_process")
    spec = make_specs(model, [gpu], tp=1, kv_connector=None)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    with Fleet([spec], out.parent / "logs") as fleet:
        inst = fleet[spec.instance_id]
        t0 = time.time()
        inst.start()
        ready_s = inst.wait_ready()
        asyncio.run(_probe(inst))
        record("ready_idle_default_clock", start_to_ready_s=ready_s)
        gpus.set_clock(gpu, 900)
        record("ready_idle_locked_900")
        gpus.reset_clock(gpu)
        record("ready_idle_after_reset_clock")
        t = time.time()
        gpus.park(gpu)
        record("mem_parked", park_s=time.time() - t)
        t = time.time()
        gpus.unpark(gpu)
        gpus.set_clock(gpu, 2520)
        unpark_s = time.time() - t
        probe = asyncio.run(_probe(inst))
        states.append(dict(state="wake_from_mem_parked", unpark_s=unpark_s, probe=probe))
        record("ready_idle_locked_2520_after_unpark")
        gpus.reset_clock(gpu)
        for level in (1, 2):
            sleep_s = inst.sleep(level)
            record(f"sleep_level_{level}", sleep_s=sleep_s)
            wake_s = inst.wake_up()
            probe = asyncio.run(_probe(inst))
            states.append(dict(state=f"wake_from_level_{level}", wake_s=wake_s, probe=probe))
        inst.stop()
        record("process_stopped")
        t0 = time.time()
        inst.start()
        ready_s = inst.wait_ready()
        probe = asyncio.run(_probe(inst))
        states.append(dict(state="restart_from_off", start_to_ready_s=ready_s, probe=probe))
    gpus.reset_clock(gpu)
    result = dict(gate="G1", model=model, gpu=gpu, window_s=window_s, states=states)
    out.write_text(json.dumps(result, indent=1, default=str))
    return result
