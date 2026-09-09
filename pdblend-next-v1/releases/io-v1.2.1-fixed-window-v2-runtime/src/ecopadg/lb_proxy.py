#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lb_proxy.py — mixed 多副本 least-inflight 负载均衡代理(B1 等资源基线用)。

DistServe Scheduler::_find_best_worker_and_queue 同款 least-loaded 语义
(以 in-flight 请求数为负载,平局取下标小者);流式透传不改语义。

用法(容器内):
  python3 -m ecopadg.lb_proxy --port 8000 \\
      --backends http://localhost:8300,http://localhost:8301
"""
from __future__ import annotations

# 防御:直跑时 ecopadg/ 目录会进 sys.path[0],包内 types.py 遮蔽标准库 types。
import os as _os
import sys as _sys
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [p for p in _sys.path
                if _os.path.abspath(p or _os.getcwd()) != _PKG_DIR]

import argparse
import asyncio

import aiohttp
from aiohttp import web


class LbProxy:
    def __init__(self, backends):
        self.backends = list(backends)
        self.inflight = [0] * len(self.backends)

    def _pick(self) -> int:
        m = min(self.inflight)
        return self.inflight.index(m)

    async def handle_completion(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        idx = self._pick()
        base = self.backends[idx]
        self.inflight[idx] += 1
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)
        try:
            timeout = aiohttp.ClientTimeout(total=3600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(base + "/v1/completions",
                                        json=body) as r:
                    if r.status != 200:
                        await resp.write(b"data: {\"error\": \"backend\"}\n\n")
                    else:
                        async for raw in r.content:
                            await resp.write(raw)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.inflight[idx] -= 1
        await resp.write_eof()
        return resp

    async def handle_models(self, request: web.Request) -> web.Response:
        return web.json_response(dict(data=[dict(id="lb-mixed")]))


async def main():
    ap = argparse.ArgumentParser(description="mixed 多副本 LB 代理")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True,
                    help="逗号分隔的后端 base URL")
    args = ap.parse_args()
    proxy = LbProxy([b.strip() for b in args.backends.split(",") if b.strip()])
    app = web.Application()
    app.router.add_post("/v1/completions", proxy.handle_completion)
    app.router.add_get("/v1/models", proxy.handle_models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()
    print("[lb] ready :%d -> %s" % (args.port, proxy.backends), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
