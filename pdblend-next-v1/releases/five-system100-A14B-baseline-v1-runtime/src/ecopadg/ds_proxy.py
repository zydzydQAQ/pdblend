#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ds_proxy.py — DS-PD 代理(DistServe 思想复现):桥接门控 + 两阶段握手 + 遥测。

对照 DistServe 源码:
  - prefill 池:FCFS token 批(由 prefill 实例 vLLM V0 内部实现,语义同
    ContextStageFCFSScheduler);
  - decode 池:continuous batching(decode 实例 vLLM V0 默认语义同
    DecodingStageFCFSScheduler);
  - 桥接:本代理实现 post_process 的 waiting_block_prop_threshold 门控 ——
    prefill 完成但 decode 未接受的请求停在 unaccepted 桥,待 decode 侧
    等待块 < 阈值×最大块数才触发迁移(第二跳);
  - KV 传输:NCCL P2P(PyNcclConnector,prefill 实例 KV producer → decode
    实例 KV consumer,对应 worker.py::migrate_blocks 的 IPC+NCCL 拉取原语);
  - 逐请求遥测:prefill 阶段时长(含 KV 传输)、ttft/tpot/slo_ok → bench.csv。

用法(容器内):
  python3 ds_proxy.py --prefill http://localhost:8100 \\
      --decode http://localhost:8200 --port 8000 --out <dir>
"""
from __future__ import annotations

# 防御:直跑本文件时 Python 会把 ecopadg/ 目录插入 sys.path[0],包内 types.py
# 会遮蔽标准库 types → 解释器在首个 stdlib import 即崩(历史率扫全灭根因)。
# sys/os 在解释器启动时已加载,此处 import 不触发文件系统查找。
import os as _os
import sys as _sys
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path
                if _os.path.abspath(p or _os.getcwd()) != _PKG_DIR]

import argparse
import asyncio
import csv
import json
import os
import sys
import time

import aiohttp
from aiohttp import web

WS_ROOT = os.environ.get("PDBLEND_WS", "/root/workspace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS_ROOT)

from ecopadg.ds_bridge import SegmentBridge
from ecopadg.types import SloSpec
from ecopadg.metrics import summarize_bench


class DsProxy:
    """多段 1P1D 代理:--prefill/--decode 逗号分隔按下标配对;
    least-loaded 选段(DistServe Scheduler 语义)+ 每段独立桥接门控。"""

    def __init__(self, args):
        self.args = args
        prefills = [u.strip() for u in args.prefill.split(",") if u.strip()]
        decodes = [u.strip() for u in args.decode.split(",") if u.strip()]
        if len(prefills) != len(decodes):
            raise ValueError("prefill/decode 段数不一致: %d vs %d"
                             % (len(prefills), len(decodes)))
        self.segments = [dict(prefill=p, decode=d, inflight=0,
                              waiting_blocks=0)
                         for p, d in zip(prefills, decodes)]
        self.rows = []
        self._rid = 0
        self._lock = asyncio.Lock()
        self.slo = SloSpec(ttft_s=args.slo_ttft, tpot_s=args.slo_tpot)

    def _pick_segment(self) -> int:
        m = min(s["inflight"] for s in self.segments)
        for i, s in enumerate(self.segments):
            if s["inflight"] == m:
                return i
        return 0

    # ------------------------------------------------------------------
    async def handle_completion(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body.get("prompt", "")
        output_len = int(body.get("max_tokens", 128))
        ignore_eos = bool(body.get("ignore_eos", False))
        rid = self._rid
        self._rid += 1
        row = dict(rid=rid, success=0, prompt_len=len(prompt.split()),
                   output_len=output_len, latency_s="", ttft_s="", tpot_s="",
                   slo_ok=0, error="", prefill_s="", bridge_wait_s="",
                   segment=-1)
        self.rows.append(row)
        # 选段(least-loaded)后做该段桥接门控:decode 侧等待块超阈值则等待。
        # 门控设 300s 上限:泄漏/后端卡死时请求快速失败而非永久自旋。
        need = (row["prompt_len"] + 15) // 16
        t_bridge = time.perf_counter()
        si = -1
        while True:
            async with self._lock:
                si = self._pick_segment()
                seg = self.segments[si]
                if seg["waiting_blocks"] < self.args.max_blocks * self.args.threshold:
                    seg["waiting_blocks"] += need
                    seg["inflight"] += 1
                    break
            if time.perf_counter() - t_bridge > 300:
                row["error"] = "bridge-timeout"
                row["latency_s"] = round(time.perf_counter() - t_bridge, 4)
                resp = web.StreamResponse()
                await resp.prepare(request)
                await resp.write_eof()
                return resp
            await asyncio.sleep(0.05)
        row["segment"] = si
        row["bridge_wait_s"] = round(time.perf_counter() - t_bridge, 4)
        t_start = time.perf_counter()
        ttft = None
        n_tok = 0
        last_ts = None
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)
        # 配额释放必须在 finally:客户端断开触发 CancelledError(BaseException),
        # 否则 inflight/waiting_blocks 泄漏,门控逐渐收死整段(挂死根因)。
        try:
            try:
                # sock_read:流中断 120s 视为后端卡死,快速失败释放段配额
                timeout = aiohttp.ClientTimeout(total=600, sock_read=120)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # 阶段 1:prefill(max_tokens=1)→ KV 经 NCCL 传至段内 decode
                    prefill_payload = dict(body)
                    prefill_payload["max_tokens"] = 1
                    prefill_payload["stream"] = False
                    t_pf = time.perf_counter()
                    async with session.post(seg["prefill"] + "/v1/completions",
                                            json=prefill_payload) as pr:
                        await pr.read()
                        if pr.status != 200:
                            row["error"] = "prefill-http-%d" % pr.status
                    row["prefill_s"] = round(time.perf_counter() - t_pf, 4)
                    if not row["error"]:
                        # 阶段 2:decode(原请求,流式;KV 亲和:必须同段 consumer)
                        async with session.post(seg["decode"] + "/v1/completions",
                                                json=body) as dr:
                            if dr.status != 200:
                                row["error"] = "decode-http-%d" % dr.status
                            else:
                                async for raw in dr.content:
                                    line = raw.decode("utf-8", "ignore").strip()
                                    if not line.startswith("data:"):
                                        continue
                                    await resp.write(raw)
                                    ts = time.perf_counter()
                                    if ttft is None:
                                        ttft = ts - t_start
                                    last_ts = ts
                                    n_tok += 1
            except Exception as e:  # noqa: BLE001
                row["error"] = "stream-error:%s" % e
            if ttft is not None:
                row["ttft_s"] = round(ttft, 4)
            if ttft is not None and n_tok > 1 and last_ts is not None:
                row["tpot_s"] = round((last_ts - t_start - ttft) / (n_tok - 1),
                                      6)
            row["latency_s"] = round(time.perf_counter() - t_start, 4)
            row["success"] = 1 if row["error"] == "" and ttft is not None else 0
            row["slo_ok"] = 1 if (row["success"] and row["tpot_s"] != ""
                                  and float(row["ttft_s"]) < self.slo.ttft_s
                                  and float(row["tpot_s"]) < self.slo.tpot_s) \
                else 0
        finally:
            async with self._lock:
                seg["waiting_blocks"] = max(0, seg["waiting_blocks"] - need)
                seg["inflight"] = max(0, seg["inflight"] - 1)
        await resp.write_eof()
        return resp


    async def handle_models(self, request: web.Request) -> web.Response:
        return web.json_response(dict(data=[dict(id="ds-pd")]))

    def flush(self):
        os.makedirs(self.args.out, exist_ok=True)
        with open(os.path.join(self.args.out, "bench_proxy.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys())
                               if self.rows else ["rid"])
            if self.rows:
                w.writeheader()
                w.writerows(self.rows)
        if self.rows:
            s = summarize_bench([dict(r) for r in self.rows], self.slo)
            print("[ds-proxy] attainment=%.3f completed=%d"
                  % (s["slo_attainment"], s["completed"]), flush=True)
        print("[ds-proxy] flushed %d rows -> %s" % (len(self.rows),
                                                    self.args.out))


async def main():
    ap = argparse.ArgumentParser(description="DS-PD 代理(桥接门控,多段)")
    ap.add_argument("--prefill", default="http://localhost:8100",
                    help="逗号分隔的 prefill base URL(与 --decode 按下标配对)")
    ap.add_argument("--decode", default="http://localhost:8200",
                    help="逗号分隔的 decode base URL")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slo-ttft", type=float, default=5.0)
    ap.add_argument("--slo-tpot", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="waiting_block_prop_threshold(DistServe post_process)")
    ap.add_argument("--max-blocks", type=int, default=2048,
                    help="decode 侧最大 KV 块数(block_size=16)")
    args = ap.parse_args()
    proxy = DsProxy(args)
    app = web.Application()
    app.router.add_post("/v1/completions", proxy.handle_completion)
    app.router.add_get("/v1/models", proxy.handle_models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print("[ds-proxy] ready :%d -> prefill=%s decode=%s threshold=%.2f"
          % (args.port, args.prefill, args.decode, args.threshold), flush=True)
    import signal
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await stop.wait()
    finally:
        proxy.flush()


if __name__ == "__main__":
    asyncio.run(main())
