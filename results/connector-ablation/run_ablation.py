"""Ablate the P2pNccl patches: patched (kv_both + batched PUT_ASYNC) vs upstream vLLM 0.10.1.1.

Three phases, GPUs 2/3 (0/1 are busy with an external profile job), clocks locked at 2520 MHz:
  1. mixed baseline: single connector-less instance on GPU3 (upstream instances cannot serve
     plain requests: the producer parses ___decode_addr_ out of every request id and raises).
  2. patched fleet:  i0/i1 with kv_both (image-patched P2pNcclConnector).
  3. upstream fleet: i0 kv_producer + i1 kv_consumer, connector class P2pNcclConnectorUpstream
     loaded via kv_connector_module_path from the p2p_upstream package (bit-identical to
     upstream, sha256 matches the migration manifest).

Per variant: PD TTFT vs mixed TTFT for prompt lengths 512/2048/7168 (5 reps, median),
text agreement between PD output and mixed output (upstream stream bug -> garbage), and a
16-way concurrent burst of 1024-token PD requests (upstream serialises at ~18 rps).
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

from pdblend.bench.gates import random_prompt  # noqa: E402
from pdblend.engine.client import EngineClient, PDTransfer, pd_complete  # noqa: E402
from pdblend.engine.launcher import Fleet, InstanceSpec  # noqa: E402

MODEL = "Qwen2.5-7B-Instruct"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results.json"
LOGS = ROOT / "logs"
GPUS = (2, 3)
PORT_P, PORT_D = 8102, 8103
LENGTHS = (512, 2048, 7168)
REPS = 5
DECODE_TOKENS = 16
FREQ_MHZ = 2520


def lock_freq() -> None:
    r = subprocess.run(["nvidia-smi", "-i", ",".join(map(str, GPUS)), "-lgc", f"{FREQ_MHZ},{FREQ_MHZ}"],
                       capture_output=True, text=True)
    print(f"lock clocks: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}", flush=True)


def unlock_freq() -> None:
    subprocess.run(["nvidia-smi", "-i", ",".join(map(str, GPUS)), "-rgc"], capture_output=True)


def check_gpus_free() -> None:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True)
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in r.stdout.strip().splitlines()}
    for g in GPUS:
        if used.get(g, 0) > 2000:
            raise RuntimeError(f"GPU {g} has {used[g]} MiB used by another job; aborting")


def upstream_kv_config(port: int, role: str) -> str:
    return json.dumps({
        "kv_connector": "P2pNcclConnectorUpstream",
        "kv_connector_module_path": "p2p_upstream",
        "kv_role": role,
        "kv_buffer_size": "1e9",
        "kv_port": str(port + 20000),
        "kv_connector_extra_config": {"http_port": str(port), "send_type": "PUT_ASYNC",
                                      "nccl_num_channels": "8", "mem_pool_size_gb": "4"}})


def patched_specs() -> list[InstanceSpec]:
    return [InstanceSpec("i0", (GPUS[0],), PORT_P, MODEL, kv_connector="P2pNcclConnector"),
            InstanceSpec("i1", (GPUS[1],), PORT_D, MODEL, kv_connector="P2pNcclConnector")]


def upstream_specs() -> list[InstanceSpec]:
    return [InstanceSpec("i0", (GPUS[0],), PORT_P, MODEL, kv_connector=None,
                         extra_args=("--kv-transfer-config", upstream_kv_config(PORT_P, "kv_producer"))),
            InstanceSpec("i1", (GPUS[1],), PORT_D, MODEL, kv_connector=None,
                         extra_args=("--kv-transfer-config", upstream_kv_config(PORT_D, "kv_consumer")))]


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def assert_own_engines(fleet: Fleet) -> None:
    for inst in fleet.instances.values():
        pid = inst.process.pid if inst.process else None
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes() if pid else b""
        if not inst.alive() or f"--port\x00{inst.spec.port}".encode() not in cmdline:
            raise RuntimeError(f"{inst.spec.instance_id} on :{inst.spec.port} is not our engine (pid {pid})")


async def measure_mixed() -> dict:
    """TTFT + reference text on a plain instance (no connector)."""
    spec = InstanceSpec("i1", (GPUS[1],), PORT_D, MODEL, kv_connector=None)
    rows = {}
    with Fleet([spec], LOGS / "mixed") as fleet:
        fleet.start_all()
        assert_own_engines(fleet)
        async with EngineClient("i1", spec.base_url) as c:
            warm = await c.complete(random_prompt(128, 7), 4, "mw0")
            if warm.error:
                raise RuntimeError(f"mixed warmup: {warm.error}")
            for n in LENGTHS:
                samples = []
                for r in range(REPS):
                    prompt = random_prompt(n, 1000 * n + r)
                    res = await c.complete(prompt, DECODE_TOKENS, f"m-{n}-{r}")
                    if res.error:
                        raise RuntimeError(f"mixed n={n}: {res.error}")
                    samples.append(dict(ttft_s=res.ttft_s, text=res.text))
                rows[n] = dict(ttft_s=statistics.median(s["ttft_s"] for s in samples),
                               texts=[s["text"] for s in samples])
                print(f"mixed n={n}: ttft {rows[n]['ttft_s']*1e3:.1f} ms", flush=True)
    return rows


async def measure_variant(name: str, specs: list[InstanceSpec], mixed: dict) -> dict:
    transfer = PDTransfer("P2pNcclConnector", {s.instance_id: s.zmq_address for s in specs})
    out: dict = {"latency": [], "burst": {}}
    with Fleet(specs, LOGS / name) as fleet:
        fleet.start_all()
        assert_own_engines(fleet)
        p, d = specs[0], specs[1]
        async with EngineClient("i0", p.base_url) as pc, EngineClient("i1", d.base_url) as dc:
            pre, dec = await pd_complete(transfer, pc, dc, random_prompt(256, 1), 4, f"{name}-warm")
            if dec is None or dec.error:
                raise RuntimeError(f"{name} warmup: {pre.error} {dec and dec.error}")
            for n in LENGTHS:
                for r in range(REPS):
                    prompt = random_prompt(n, 1000 * n + r)
                    pre, dec = await pd_complete(transfer, pc, dc, prompt, DECODE_TOKENS, f"lat-{n}-{r}")
                    row = dict(n=n, rep=r, error=pre.error or (dec.error if dec else "no decode leg"))
                    if dec is not None and not dec.error:
                        row["prefill_leg_s"] = pre.finished_s - pre.submitted_s
                        row["pd_ttft_s"] = dec.first_token_s - pre.submitted_s
                        row["overhead_s"] = row["pd_ttft_s"] - mixed[n]["ttft_s"]
                        row["text_exact"] = dec.text == mixed[n]["texts"][r]
                        row["text_prefix"] = common_prefix(dec.text, mixed[n]["texts"][r])
                    out["latency"].append(row)
                    ok = [x for x in out["latency"] if x["n"] == n and "overhead_s" in x]
                    med = statistics.median(x["overhead_s"] for x in ok) if ok else float("nan")
                    print(f"{name} n={n} r={r}: overhead {row.get('overhead_s', float('nan'))*1e3:.1f} ms "
                          f"(median {med*1e3:.1f}) text_exact={row.get('text_exact')} err={row['error']}",
                          flush=True)
            # ---- concurrent bursts: short prompts expose the upstream per-(request, layer)
            # handshake serialisation; long prompts are compute-bound and should not differ ----
            for burst_n, burst_conc in ((128, 48), (1024, 16)):
                prompts = [random_prompt(burst_n, 7000 + i) for i in range(burst_conc)]
                t0 = time.time()
                try:
                    pairs = await asyncio.wait_for(
                        asyncio.gather(*(pd_complete(transfer, pc, dc, prompts[i], DECODE_TOKENS,
                                                     f"burst{burst_n}-{i}")
                                         for i in range(burst_conc)), return_exceptions=True),
                        timeout=300.0)
                except TimeoutError:
                    pairs = []
                    out["burst"][f"{burst_n}x{burst_conc}"] = dict(error="timeout after 300 s")
                if pairs:
                    pre_ok, dec_ok, errors = [], [], []
                    for item in pairs:
                        if isinstance(item, Exception):
                            errors.append(repr(item))
                            continue
                        pr, dc_ = item
                        if pr.error:
                            errors.append(f"prefill: {pr.error}")
                        elif dc_ is None or dc_.error:
                            errors.append(f"decode: {dc_ and dc_.error}")
                        else:
                            pre_ok.append(pr)
                            dec_ok.append(dc_)
                    if dec_ok:
                        prefill_window = max(pr.finished_s for pr in pre_ok) - t0
                        out["burst"][f"{burst_n}x{burst_conc}"] = dict(
                            n=burst_n, concurrency=burst_conc, ok=len(dec_ok), errors=errors,
                            prefill_window_s=prefill_window,
                            prefill_rps=len(pre_ok) / max(prefill_window, 1e-6),
                            all_done_s=max(x.finished_s for x in dec_ok) - t0,
                            median_pd_ttft_s=statistics.median(
                                x.first_token_s - pr.submitted_s
                                for pr, x in zip(pre_ok, dec_ok)))
                    else:
                        out["burst"][f"{burst_n}x{burst_conc}"] = dict(
                            n=burst_n, concurrency=burst_conc, ok=0, errors=errors)
                print(f"{name} burst {burst_n}x{burst_conc}: "
                      f"{json.dumps(out['burst'][f'{burst_n}x{burst_conc}'])}", flush=True)
    return out


async def main() -> None:
    check_gpus_free()
    lock_freq()
    result: dict = dict(model=MODEL, gpus=GPUS, freq_mhz=FREQ_MHZ, lengths=LENGTHS, reps=REPS,
                        decode_tokens=DECODE_TOKENS, started=time.time())
    try:
        result["mixed"] = await measure_mixed()
        for name, specs in (("patched", patched_specs()), ("upstream", upstream_specs())):
            result[name] = await measure_variant(name, specs, result["mixed"])
            # checkpoint after each variant so a later failure keeps earlier data
            OUT.write_text(json.dumps(result, indent=1))
    finally:
        unlock_freq()
    result["elapsed_s"] = time.time() - result["started"]
    OUT.write_text(json.dumps(result, indent=1))
    print(f"done in {result['elapsed_s']:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
