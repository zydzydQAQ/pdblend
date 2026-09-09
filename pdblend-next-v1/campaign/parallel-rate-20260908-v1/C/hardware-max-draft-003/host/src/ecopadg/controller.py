#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M̃1 单实例代理:OpenAI 兼容入口 + 窗口准入 + 整卡 DVFS。

不是 S3,也不在 run_cell.sh 主矩阵。不是严格 temporal PaDG:
只控制请求何时进入 vLLM 以及锁哪一档 SM 时钟;引擎内仍可混相。
  - /v1/completions → AdmissionGate + PaDGWindowScheduler 释放;
  - prefill 窗锁最高频、decode 窗 slack 选频、过载连续混批;
  - --force-continuous 时退化为 M0-D(无窗口,只做 slack DVFS)。

用法(宿主机):
  python3 controller.py --model-key 32b --engine http://localhost:8000 \\
      --port 8001 --out script/bench/results/<exp_tag> \\
      --slo-ttft 5.0 --slo-tpot 0.1
"""
from __future__ import annotations

# 防御:直跑本文件时 ecopadg/ 目录会进 sys.path[0],包内 types.py 遮蔽标准库
# types → 解释器崩溃(历史 ecopadg 率扫全灭根因)。首行清除后再 import。
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
import signal
import sys
import threading
import time
from typing import Dict, Optional

import aiohttp
from aiohttp import web

WS_ROOT = os.environ.get("PDBLEND_WS", "/root/workspace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS_ROOT)

from ecopadg.dvfs import DvfsController
from ecopadg.online_scheduler import PaDGWindowScheduler
from ecopadg.router import (
    AdmissionGate, RequestRecord, admit_request, estimate_prompt_tokens,
)
from ecopadg.types import SloSpec, SystemConfig
from ecopadg.measure.backends import get_backend
from ecopadg.measure.power import PowerSampler


from ecopadg.opmodel import load_opmodel, model_runtime_cfg

# M̃1 单实例消融。S3 mixed-only 用: python -m ecopadg.ecospd_controller --pools mixed
MODEL_CFG = {
    "14b": model_runtime_cfg("14b", WS_ROOT),
    "32b": model_runtime_cfg("32b", WS_ROOT),
    "72b": model_runtime_cfg("72b", WS_ROOT),
}


class Controller:
    """代理 + 窗口调度 + DVFS 的组装。"""

    def __init__(self, args, mcfg):
        self.args = args
        self.mcfg = mcfg
        self.backend = get_backend("pynvml")
        self.dvfs = DvfsController(backend=self.backend, min_dwell_s=3.0)
        self.opmodel = load_opmodel(mcfg["tables"])
        self.cfg = SystemConfig(model=args.model_key)
        self.slo = SloSpec(ttft_s=float(args.slo_ttft),
                           tpot_s=float(args.slo_tpot))
        self.cfg.slo = self.slo
        self.cfg.force_continuous = bool(
            getattr(args, "force_continuous", False))
        if args.model_key == "72b" and float(args.slo_tpot) <= 0.1:
            self.cfg.decode_floor_mhz = 2520
        if self.opmodel is None:
            print("[ctrl] 警告:未找到操作点表,decode 固定频率 %d MHz"
                  % mcfg["fallback_decode_freq"], flush=True)
        self.gate = AdmissionGate()
        self.sched = PaDGWindowScheduler(
            self.opmodel, self.cfg,
            mixed_capacity_req_s=self.mcfg["capacity_req_s"])
        if self.opmodel is None:
            self.sched._decode_freq = lambda: self.mcfg["fallback_decode_freq"]
        self.records: Dict[int, RequestRecord] = {}
        self.events: Dict[int, asyncio.Event] = {}
        self._rid = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.sampler = PowerSampler(gpus=mcfg["gpus"], interval=0.02,
                                    backend=self.backend)
        self.engine_ready = asyncio.Event()

    # ------------------------------------------------------------------
    # 引擎就绪探测(引擎加载完成前不放行请求)
    # ------------------------------------------------------------------
    async def _engine_probe(self):
        import aiohttp as _aiohttp
        while not self._stop.is_set():
            try:
                async with _aiohttp.ClientSession() as s:
                    async with s.get(self.args.engine + "/v1/models",
                                     timeout=_aiohttp.ClientTimeout(total=3)) as r:
                        if r.status == 200:
                            self.engine_ready.set()
                            print("[ctrl] engine ready", flush=True)
                            return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(2)

    # ------------------------------------------------------------------
    # 窗口调度线程
    # ------------------------------------------------------------------
    def _sched_loop(self, loop):
        sched = self.sched
        period = self.cfg.global_period_s
        while not self._stop.is_set():
            time.sleep(period)
            now = time.time()
            d = sched.step(now)
            if not sched.buffer and not sched.active:
                self.dvfs.force_reset(self.mcfg["gpus"])  # 空闲解锁,避免空转锁频
            else:
                self.dvfs.apply({g: d.freq_mhz for g in self.mcfg["gpus"]}, now)
            if d.released:
                self.gate.release_due(now, limit=len(d.released))
            for rid in d.released:
                with self._lock:
                    ev = self.events.get(rid)
                if ev is not None:
                    loop.call_soon_threadsafe(ev.set)
            print("[ctrl] t=%.1f mode=%s freq=%d released=%d buffer=%d"
                  % (now, d.mode, d.freq_mhz, len(d.released),
                     len(sched.buffer)), flush=True)


    # ------------------------------------------------------------------
    # OpenAI 代理:排队 + 门控 + 转发
    # ------------------------------------------------------------------
    async def handle_completion(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body.get("prompt", "")
        output_len = int(body.get("max_tokens", 128))
        ignore_eos = bool(body.get("ignore_eos", False))
        with self._lock:
            rid = self._rid
            self._rid += 1
            rec = RequestRecord(rid=rid, arrival_s=time.time(),
                                prompt_len=estimate_prompt_tokens(prompt, body),
                                output_len=output_len)
            self.records[rid] = rec
            ev = asyncio.Event()
            self.events[rid] = ev
        admit_request(self.gate, self.sched, rec)
        # 引擎未就绪先等待(加载期不放行)
        try:
            await asyncio.wait_for(self.engine_ready.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            rec.error = "engine-not-ready"
        # 等待窗口释放(连续模式即时释放)
        try:
            await asyncio.wait_for(ev.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            rec.error = "gate-timeout"
        rec.release_s = time.time()

        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)

        payload = dict(model=body.get("model", self.args.model_key),
                       prompt=prompt, max_tokens=output_len,
                       temperature=0, top_p=1.0,
                       ignore_eos=ignore_eos, stream=True)
        t_start = time.perf_counter()
        ttft = None
        n_tok = 0
        last_ts = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.args.engine + "/v1/completions",
                                        json=payload) as r:
                    if r.status != 200:
                        rec.error = "engine-http-%d" % r.status
                        await resp.write(b"data: {\"error\": \"engine\"}\n\n")
                    else:
                        async for raw in r.content:
                            line = raw.decode("utf-8", "ignore").strip()
                            if not line.startswith("data:"):
                                continue
                            await resp.write(raw)
                            token_ts = time.perf_counter()
                            if ttft is None:
                                ttft = token_ts - t_start
                            last_ts = token_ts
                            n_tok += 1
        except Exception as e:  # noqa: BLE001
            rec.error = "stream-error:%s" % e
        rec.ttft_s = ttft
        if ttft is not None and n_tok > 1 and last_ts is not None:
            rec.tpot_s = (last_ts - t_start - ttft) / (n_tok - 1)
        rec.latency_s = time.perf_counter() - t_start
        rec.success = rec.error == "" and ttft is not None
        self.sched.complete(rec.rid)  # active 清理(b_hat 估计用)
        await resp.write_eof()
        return resp


    def _flush_outputs(self):
        os.makedirs(self.args.out, exist_ok=True)
        rows = [r.as_row(self.slo) for r in
                sorted(self.records.values(), key=lambda x: x.rid)]
        with open(os.path.join(self.args.out, "bench_ctrl.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                               else ["rid"])
            if rows:
                w.writeheader()
                w.writerows(rows)
        self.sampler.stop()
        self.sampler.to_csv(os.path.join(self.args.out, "power_ctrl.csv"))
        print("[ctrl] flushed %d records -> %s" % (len(rows), self.args.out))

    def shutdown(self):
        self._stop.set()
        self.dvfs.force_reset(self.mcfg["gpus"])
        self._flush_outputs()
        print("[ctrl] shutdown:频率已解锁")


async def main():
    ap = argparse.ArgumentParser(description="PDblend M̃1 窗口代理")
    ap.add_argument("--model-key", default="32b")
    ap.add_argument("--engine", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slo-ttft", type=float, default=5.0)
    ap.add_argument("--slo-tpot", type=float, default=0.1)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--force-continuous", action="store_true",
                    help="M0-D:不开时间窗,只做 slack DVFS")
    args = ap.parse_args()
    mcfg = MODEL_CFG[args.model_key]
    ctrl = Controller(args, mcfg)
    ctrl.cfg.global_period_s = args.period

    app = web.Application()
    app.router.add_post("/v1/completions", ctrl.handle_completion)
    app.router.add_get("/v1/models",
                       lambda r: web.json_response(dict(data=[dict(id="eco")])))
    loop = asyncio.get_event_loop()
    th = threading.Thread(target=ctrl._sched_loop, args=(loop,), daemon=True)
    th.start()
    loop.create_task(ctrl._engine_probe())
    ctrl.sampler.start()

    def _sig(signum, frame):
        ctrl.shutdown()
        os._exit(0)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print("[ctrl] 代理就绪 :%d -> %s" % (args.port, args.engine), flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        ctrl.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
