# -*- coding: utf-8 -*-
"""EcoServe MacroInstance ported onto vLLM 0.9.2 HTTP backends.

Scheduler logic follows EcoServe/ecoserve/macro_instance.py
(schedule / _check_constraints / _switch_instance). send_output hold follows
instance.py: do not forward streamed tokens until the instance is released
or the TTFT budget elapses. That delay lands in first-token TTFT (warehouse
metric), which is the wait span without switching the main table to the
paper's second-token definition.

This is not the 0.7.3 container and not joint_temporal / strict_padg / A9.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path
                if _os.path.abspath(p or _os.getcwd()) != _PKG_DIR]

import argparse
import asyncio
import csv
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

BLOCK_SIZE = 16
# L20 14B stock vLLM 0.9.2 engine_telemetry (one GPU). Never 1e9.
MEASURED_L20_14B_GPU_BLOCKS = 15060
# n80 1×TP1 @ 2520 ShareGPT TTFT/pin. Production prefill table uses this.
MEASURED_TP1_2520_S_PER_TOK = 0.0005659482421875
MEASURED_TP1_SOURCE = "n80-1x-tp1-2520"
IMPLEMENTATION_LABEL = "ecoserve-macro-8x-tp1-vllm092"
_CACHE_USAGE_RE = re.compile(
    r"^vllm(?::|_)gpu_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$",
    re.M)
_CACHE_INFO_RE = re.compile(
    r"vllm(?::|_)cache_config_info\{([^}]*)\}")
_NUM_BLOCKS_LABEL = re.compile(r'num_gpu_blocks="(\d+)"')


@dataclass
class RequestState:
    request_id: str
    arrival_time: float
    num_iterations: int
    ttft: float
    predict_time: float
    predict_length: int
    prefill_blocks: int


@dataclass
class InstanceState:
    instance_id: int
    requests: Deque[RequestState] = field(default_factory=deque)
    waiting_queue: List[str] = field(default_factory=list)
    free_blocks: int = 0
    prefill_mode: bool = False
    schedule_time: float = 0.0
    num_gpu_blocks: int = 0


class MacroScheduler:
    """Literal port of EcoServe MacroInstance.schedule + constraints.

    Times are milliseconds, matching the source.
    """

    def __init__(self, instance_count: int, ttft_ms: int, tpot_ms: int,
                 prefill_data: Optional[Dict[int, float]] = None,
                 num_gpu_blocks: int = 0):
        if instance_count < 1:
            raise ValueError("instance_count must be >= 1")
        self.instance_count = int(instance_count)
        self.TTFT = int(ttft_ms)
        self.TPOT = int(tpot_ms)
        self.prefill_data = dict(prefill_data or {})
        blocks = int(num_gpu_blocks or 0)
        if blocks <= 0:
            blocks = MEASURED_L20_14B_GPU_BLOCKS
        self.num_gpu_blocks = blocks
        now = time.time() * 1000.0
        self.instance_states = [
            InstanceState(i, schedule_time=now, free_blocks=blocks,
                          num_gpu_blocks=blocks)
            for i in range(self.instance_count)
        ]
        self.prefill_instance = 0
        self.send_output = [True] * self.instance_count
        self.prefill_time = [now] * self.instance_count
        self.state_degraded = not self.prefill_data
        self.switch_count = 0
        self.kv_from_metrics = False

    def _get_predict_time(self, num_tokens: int) -> int:
        """Interpolate the measured TP1 table. 0.2×n is not a production path."""
        if not self.prefill_data:
            raise ValueError("prefill_data required; 0.2*n is not a production path")
        n = max(int(num_tokens), 1)
        if n in self.prefill_data:
            return max(int(self.prefill_data[n]), 1)
        keys = sorted(self.prefill_data)
        if n <= keys[0]:
            return max(int(self.prefill_data[keys[0]] * n / float(keys[0])), 1)
        if n >= keys[-1]:
            return max(int(self.prefill_data[keys[-1]] * n / float(keys[-1])), 1)
        lo = max(k for k in keys if k <= n)
        hi = min(k for k in keys if k >= n)
        if lo == hi:
            return max(int(self.prefill_data[lo]), 1)
        t0 = float(self.prefill_data[lo])
        t1 = float(self.prefill_data[hi])
        frac = (n - lo) / float(hi - lo)
        return max(int(t0 + (t1 - t0) * frac), 1)

    def _check_constraints(self, num_blocks: int, predict_time: float) -> bool:
        """Source MacroInstance._check_constraints (ms, KV, saved-TPOT)."""
        TPOT_left_time: List[float] = []
        TTFT_left_time: List[float] = []
        need_blocks = int(num_blocks)
        need_time = float(predict_time)
        instance_state = self.instance_states[self.prefill_instance]
        for request_state in instance_state.requests:
            if request_state.request_id in instance_state.waiting_queue:
                TTFT_left_time.append(self.TTFT)
                need_blocks += request_state.prefill_blocks
                need_time += request_state.predict_time
            else:
                TPOT_left_time.append(
                    request_state.ttft
                    + request_state.num_iterations * self.TPOT
                    - instance_state.schedule_time
                    + request_state.arrival_time)

        if need_blocks > instance_state.free_blocks:
            return False
        if TPOT_left_time:
            TPOT_left_time = [max(TPOT_left_time)]
        left_time = TTFT_left_time + TPOT_left_time
        if not left_time:
            return True
        avail_time = min(left_time)
        if avail_time > need_time:
            return True
        if need_time > self.TTFT:
            return False
        TPOT_left_time = []
        next_instance = (self.prefill_instance + 1) % self.instance_count
        instance_state = self.instance_states[next_instance]
        now = time.time() * 1000.0
        for request_state in instance_state.requests:
            TPOT_left_time.append(
                request_state.ttft
                + request_state.num_iterations * self.TPOT
                - now
                + request_state.arrival_time)
        if not TPOT_left_time:
            return False
        leftover = max(TPOT_left_time)
        return leftover < self.TTFT * (self.instance_count - 1) / self.instance_count

    def _switch_instance(self) -> int:
        now = time.time() * 1000.0
        cur = self.prefill_instance
        self.instance_states[cur].waiting_queue = []
        self.send_output[cur] = True
        nxt = (cur + 1) % self.instance_count
        self.send_output[nxt] = False
        self.prefill_time[nxt] = now
        self.instance_states[nxt].schedule_time = now
        self.switch_count += 1
        return nxt

    def schedule(self, request_id: str, prompt_len: int) -> int:
        now = time.time() * 1000.0
        predict_time = self._get_predict_time(int(prompt_len))
        num_blocks = (int(prompt_len) + BLOCK_SIZE) // BLOCK_SIZE
        request_state = RequestState(
            request_id, now, 0, float(self.TTFT),
            float(predict_time), -1, num_blocks)
        if self._check_constraints(num_blocks, predict_time):
            schedule_instance = self.prefill_instance
        else:
            schedule_instance = self._switch_instance()
        self.prefill_instance = schedule_instance
        inst = self.instance_states[schedule_instance]
        inst.requests.append(request_state)
        inst.waiting_queue.append(request_state.request_id)
        inst.prefill_mode = True
        inst.free_blocks = max(0, inst.free_blocks - num_blocks)
        return schedule_instance

    def apply_kv_snapshot(self, instance_id: int, free_blocks: int,
                          num_gpu_blocks: Optional[int] = None) -> None:
        inst = self.instance_states[int(instance_id)]
        if num_gpu_blocks is not None and int(num_gpu_blocks) > 0:
            inst.num_gpu_blocks = int(num_gpu_blocks)
            self.num_gpu_blocks = int(num_gpu_blocks)
        inst.free_blocks = max(0, int(free_blocks))
        self.kv_from_metrics = True

    def refresh_schedule_time(self, instance_id: Optional[int] = None) -> None:
        now = time.time() * 1000.0
        if instance_id is None:
            for inst in self.instance_states:
                inst.schedule_time = now
            return
        self.instance_states[int(instance_id)].schedule_time = now

    def mark_first_token(self, instance_id: int, request_id: str) -> None:
        inst = self.instance_states[int(instance_id)]
        now = time.time() * 1000.0
        inst.schedule_time = now
        if request_id in inst.waiting_queue:
            inst.waiting_queue.remove(request_id)
        for req in inst.requests:
            if req.request_id == request_id:
                if req.num_iterations == 0:
                    req.ttft = now - req.arrival_time
                req.num_iterations += 1
                break
        inst.prefill_mode = bool(inst.waiting_queue)

    def mark_output_tokens(self, instance_id: int, request_id: str,
                           n_tokens: int = 1) -> None:
        inst = self.instance_states[int(instance_id)]
        now = time.time() * 1000.0
        inst.schedule_time = now
        extra = max(int(n_tokens), 0)
        if extra <= 0:
            return
        for req in inst.requests:
            if req.request_id == request_id:
                req.num_iterations += extra
                break

    def finish_request(self, instance_id: int, request_id: str,
                       used_blocks: int = 0) -> None:
        inst = self.instance_states[int(instance_id)]
        inst.requests = deque(
            r for r in inst.requests if r.request_id != request_id)
        if request_id in inst.waiting_queue:
            inst.waiting_queue.remove(request_id)
        cap = inst.num_gpu_blocks or self.num_gpu_blocks
        if used_blocks:
            inst.free_blocks = min(cap, inst.free_blocks + int(used_blocks))
        inst.prefill_mode = bool(inst.waiting_queue)

    def should_release(self, instance_id: int) -> bool:
        idx = int(instance_id)
        if self.send_output[idx]:
            return True
        elapsed = time.time() * 1000.0 - self.prefill_time[idx]
        return elapsed > float(self.TTFT)


def load_prefill_csv(path: str) -> Dict[int, float]:
    data: Dict[int, float] = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            length = int(row["Length"])
            data[length] = float(row["Prefill Time"])
    return data


def measured_tp1_prefill_s_per_tok(freq_mhz: int = 2520) -> float:
    """n80 1×TP1 overlay; baked constant if the profile file is absent."""
    here = _os.path.dirname(_os.path.abspath(__file__))
    profile = _os.path.normpath(_os.path.join(
        here, "..", "..", "script", "bench", "pdblend_profile_14b.json"))
    try:
        with open(profile, encoding="utf-8") as fh:
            data = json.load(fh)
        row = (data.get("tp1") or {}).get(str(int(freq_mhz))) or {}
        spt = float(row.get("prefill_s_per_tok") or 0.0)
        if spt > 0.0:
            return spt
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return MEASURED_TP1_2520_S_PER_TOK


def write_prefill_csv(path: str, *, opmodel=None, tp: int = 1,
                      freq_mhz: int = 2520) -> str:
    """Length,Prefill Time (ms) from measured 1×TP1, not 0.2×n."""
    del opmodel, tp
    lengths = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    spt = measured_tp1_prefill_s_per_tok(int(freq_mhz))
    rows = [(length, max(spt * float(length) * 1000.0, 1.0))
            for length in lengths]
    parent = _os.path.dirname(path)
    if parent:
        _os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Length", "Prefill Time"])
        writer.writerows(rows)
    return path


def default_prefill_csv_path() -> str:
    here = _os.path.dirname(_os.path.abspath(__file__))
    return _os.path.normpath(_os.path.join(
        here, "..", "..", "script", "bench", "prefill_14b_tp1.csv"))


def ensure_prefill_csv(path: str = "") -> str:
    dest = path or default_prefill_csv_path()
    if _os.path.isfile(dest):
        return dest
    write_prefill_csv(dest, tp=1)
    return dest


def parse_vllm_kv_metrics(text: str) -> Tuple[Optional[float], Optional[int]]:
    """Return (gpu_cache_usage_0_1, num_gpu_blocks) from vLLM /metrics."""
    usage = None
    match = _CACHE_USAGE_RE.search(text or "")
    if match:
        try:
            usage = float(match.group(1))
        except (TypeError, ValueError):
            usage = None
        if usage is not None and usage > 1.0:
            usage = usage / 100.0
    blocks = None
    info = _CACHE_INFO_RE.search(text or "")
    if info:
        label = _NUM_BLOCKS_LABEL.search(info.group(1))
        if label:
            blocks = int(label.group(1))
    return usage, blocks


def count_completion_tokens(raw: bytes) -> int:
    """Count streamed completion events. One vLLM SSE chunk ≈ one token."""
    n = 0
    for line in raw.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            obj = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for choice in obj.get("choices") or ():
            text = choice.get("text")
            if text in (None, ""):
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("text")
            if text not in (None, ""):
                n += 1
    return n


def load_tokenizer(path: str):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:
        print("[ecoserve] tokenizer 加载失败,回退启发式: %s" % exc, flush=True)
        return None


def prompt_len_from_body(body: dict, tokenizer=None) -> int:
    from ecopadg.router import prompt_tokens

    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = "".join(str(x) for x in prompt)
    text = str(prompt or "")
    return max(1, int(prompt_tokens(text, body, tokenizer)))


class EcoServeProxy:  # aiohttp handlers; imported only in _run()
    def __init__(self, backends: Sequence[str], scheduler: MacroScheduler,
                 tokenizer=None):
        self.backends = list(backends)
        if len(self.backends) != scheduler.instance_count:
            raise ValueError("backends and instance_count must match")
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self._release = [asyncio.Event() for _ in self.backends]
        for ev, flag in zip(self._release, scheduler.send_output):
            if flag:
                ev.set()
        self._lock = asyncio.Lock()
        self._req_seq = 0
        self._kv_session = None

    def _sync_release_flags(self) -> None:
        for i, flag in enumerate(self.scheduler.send_output):
            if flag or self.scheduler.should_release(i):
                self._release[i].set()
            else:
                self._release[i].clear()

    async def refresh_kv(self, session, idx: int) -> None:
        import aiohttp
        url = self.backends[int(idx)].rstrip("/") + "/metrics"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    return
                text = await resp.text()
        except Exception:
            return
        usage, blocks = parse_vllm_kv_metrics(text)
        total = blocks or self.scheduler.instance_states[int(idx)].num_gpu_blocks
        if usage is None or not total:
            if blocks:
                async with self._lock:
                    inst = self.scheduler.instance_states[int(idx)]
                    if inst.num_gpu_blocks <= 0:
                        self.scheduler.apply_kv_snapshot(idx, blocks, blocks)
            return
        free = max(0, int(round(float(total) * (1.0 - float(usage)))))
        async with self._lock:
            self.scheduler.apply_kv_snapshot(idx, free, total)

    async def _kv_loop(self) -> None:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._kv_session = session
            while True:
                for idx in range(len(self.backends)):
                    await self.refresh_kv(session, idx)
                await asyncio.sleep(0.2)

    async def handle_completion(self, request):
        from aiohttp import web
        import aiohttp
        body = await request.json()
        prompt_len = prompt_len_from_body(body, self.tokenizer)
        if self._kv_session is not None:
            for idx in range(len(self.backends)):
                await self.refresh_kv(self._kv_session, idx)
        async with self._lock:
            self._req_seq += 1
            rid = str(body.get("request_id") or ("es-%d" % self._req_seq))
            idx = self.scheduler.schedule(rid, prompt_len)
            self._sync_release_flags()
        base = self.backends[idx]
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)
        first = True
        used_blocks = (prompt_len + BLOCK_SIZE) // BLOCK_SIZE
        try:
            timeout = aiohttp.ClientTimeout(total=3600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(base + "/v1/completions",
                                        json=body) as r:
                    if r.status != 200:
                        await resp.write(b"data: {\"error\": \"backend\"}\n\n")
                    else:
                        async for raw in r.content:
                            if first:
                                if not self.scheduler.should_release(idx):
                                    remain = max(
                                        0.0,
                                        (self.scheduler.TTFT
                                         - (time.time() * 1000.0
                                            - self.scheduler.prefill_time[idx]))
                                        / 1000.0)
                                    try:
                                        await asyncio.wait_for(
                                            self._release[idx].wait(),
                                            timeout=remain)
                                    except asyncio.TimeoutError:
                                        pass
                                async with self._lock:
                                    self.scheduler.mark_first_token(idx, rid)
                                first = False
                            else:
                                extra = count_completion_tokens(raw)
                                if extra:
                                    async with self._lock:
                                        self.scheduler.mark_output_tokens(
                                            idx, rid, extra)
                            await resp.write(raw)
        except Exception:  # noqa: BLE001
            pass
        finally:
            async with self._lock:
                self.scheduler.finish_request(idx, rid, used_blocks=used_blocks)
        await resp.write_eof()
        return resp

    async def handle_models(self, request):
        from aiohttp import web
        del request
        return web.json_response(dict(data=[dict(id=IMPLEMENTATION_LABEL)]))

    async def handle_health(self, request):
        from aiohttp import web
        del request
        return web.json_response(dict(
            ok=True,
            implementation=IMPLEMENTATION_LABEL,
            prefill_instance=self.scheduler.prefill_instance,
            switch_count=self.scheduler.switch_count,
            state_degraded=self.scheduler.state_degraded,
            send_output=list(self.scheduler.send_output),
            kv_from_metrics=self.scheduler.kv_from_metrics,
            num_gpu_blocks=self.scheduler.num_gpu_blocks,
            free_blocks=[inst.free_blocks
                         for inst in self.scheduler.instance_states],
            tokenizer=self.tokenizer is not None,
            note="MacroInstance-on-vLLM0.9.2 first-token TTFT; not paper second-token",
        ))


async def _run(args) -> None:
    from aiohttp import web
    csv_path = ensure_prefill_csv(args.prefill_csv)
    prefill = load_prefill_csv(csv_path)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tok = load_tokenizer(str(getattr(args, "tokenizer", "") or ""))
    sched = MacroScheduler(
        instance_count=len(backends),
        ttft_ms=int(round(float(args.slo_ttft) * 1000.0)),
        tpot_ms=int(round(float(args.slo_tpot) * 1000.0)),
        prefill_data=prefill,
        num_gpu_blocks=int(getattr(args, "num_gpu_blocks", 0) or 0))
    if not prefill:
        sched.state_degraded = True
    if tok is None:
        sched.state_degraded = True
    proxy = EcoServeProxy(backends, sched, tokenizer=tok)
    app = web.Application()
    app.router.add_post("/v1/completions", proxy.handle_completion)
    app.router.add_get("/v1/models", proxy.handle_models)
    app.router.add_get("/health", proxy.handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    asyncio.create_task(proxy._kv_loop())
    print("[ecoserve] ready :%d -> %s csv=%s impl=%s degraded=%s tok=%s blocks=%s" % (
        args.port, proxy.backends, csv_path, IMPLEMENTATION_LABEL,
        sched.state_degraded, tok is not None, sched.num_gpu_blocks),
          flush=True)
    await asyncio.Event().wait()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EcoServe MacroInstance HTTP proxy")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", default="",
                    help="comma-separated vLLM base URLs")
    ap.add_argument("--slo-ttft", type=float, default=5.0)
    ap.add_argument("--slo-tpot", type=float, default=0.15)
    ap.add_argument("--prefill-csv", default="")
    ap.add_argument("--tokenizer", default="",
                    help="HF tokenizer / model dir for real prompt_len")
    ap.add_argument("--num-gpu-blocks", type=int, default=0,
                    help="init KV free_blocks; 0 probes /metrics then L20 14B")
    ap.add_argument("--write-prefill-csv", default="",
                    help="generate measured TP1 Length,Prefill Time CSV and exit")
    args = ap.parse_args(argv)
    if args.write_prefill_csv:
        write_prefill_csv(args.write_prefill_csv, tp=1)
        print(args.write_prefill_csv)
        return 0
    if not args.backends:
        raise SystemExit("--backends is required unless --write-prefill-csv")
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
