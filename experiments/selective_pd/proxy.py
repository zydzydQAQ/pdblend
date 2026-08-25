#!/usr/bin/env python3
"""Minimal aiohttp proxy for colocated and 1P1D vLLM serving."""

from __future__ import annotations

import argparse
import itertools
import json
import uuid
from typing import Any

import aiohttp
from aiohttp import web


class BackendFailure(RuntimeError):
    def __init__(self, status: int, body: bytes):
        super().__init__(f"backend returned HTTP {status}")
        self.status = status
        self.body = body


async def _stream_backend(
    request: web.Request,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> web.StreamResponse:
    session: aiohttp.ClientSession = request.app["session"]
    try:
        async with session.post(url, json=payload, headers=headers) as backend:
            if backend.status != 200:
                body = await backend.read()
                raise BackendFailure(backend.status, body)
            response = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": backend.headers.get(
                        "Content-Type", "text/event-stream"
                    ),
                    "Cache-Control": "no-cache",
                },
            )
            await response.prepare(request)
            async for chunk in backend.content.iter_chunked(16 * 1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    except BackendFailure as exc:
        return web.Response(
            status=exc.status,
            body=exc.body,
            content_type="application/json",
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        return web.json_response(
            {"error": f"backend unavailable: {type(exc).__name__}: {exc}"},
            status=503,
        )


async def _handle_completion(request: web.Request) -> web.StreamResponse:
    payload = await request.json()
    mode = request.app["mode"]
    if mode == "colocated":
        backend = next(request.app["backend_cycle"])
        return await _stream_backend(request, backend, payload)

    session: aiohttp.ClientSession = request.app["session"]
    prefill_payload = dict(payload)
    prefill_payload["max_tokens"] = 1
    prefill_payload["stream"] = False
    if request.app["connector"] == "nixl":
        prefill_payload["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        request_id = str(uuid.uuid4())
    else:
        request_id = (
            f"___prefill_addr_{request.app['prefill_zmq']}"
            f"___decode_addr_{request.app['decode_zmq']}_{uuid.uuid4().hex}"
        )
    headers = {"X-Request-Id": request_id}
    try:
        async with session.post(
            request.app["prefill_url"],
            json=prefill_payload,
            headers=headers,
        ) as prefill:
            body = await prefill.read()
            if prefill.status != 200:
                return web.Response(
                    status=prefill.status,
                    body=body,
                    content_type="application/json",
                )
            if request.app["connector"] == "nixl":
                prefill_response = json.loads(body)
                transfer = prefill_response.get("kv_transfer_params")
                if not transfer:
                    return web.json_response(
                        {"error": "prefill returned no kv_transfer_params"},
                        status=502,
                    )
                payload["kv_transfer_params"] = transfer
    except (aiohttp.ClientError, TimeoutError) as exc:
        return web.json_response(
            {"error": f"prefill unavailable: {type(exc).__name__}: {exc}"},
            status=503,
        )
    return await _stream_backend(
        request, request.app["decode_url"], payload, headers
    )


async def _health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _startup(app: web.Application) -> None:
    timeout = aiohttp.ClientTimeout(total=600)
    app["session"] = aiohttp.ClientSession(timeout=timeout)


async def _cleanup(app: web.Application) -> None:
    await app["session"].close()


def create_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app["mode"] = args.mode
    if args.mode == "colocated":
        if len(args.backends) != len(args.weights):
            raise ValueError("--backends and --weights lengths must match")
        weighted = [
            backend
            for backend, weight in zip(args.backends, args.weights)
            for _ in range(weight)
        ]
        if not weighted:
            raise ValueError("at least one positive backend weight is required")
        app["backend_cycle"] = itertools.cycle(weighted)
    else:
        app["prefill_url"] = args.prefill_url
        app["decode_url"] = args.decode_url
        app["connector"] = args.connector
        app["prefill_zmq"] = args.prefill_zmq
        app["decode_zmq"] = args.decode_zmq
    app.router.add_get("/health", _health)
    app.router.add_post("/v1/completions", _handle_completion)
    app.on_startup.append(_startup)
    app.on_cleanup.append(_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["colocated", "disagg"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[
            "http://127.0.0.1:8100/v1/completions",
            "http://127.0.0.1:8200/v1/completions",
        ],
    )
    parser.add_argument("--weights", nargs="+", type=int, default=[1, 1])
    parser.add_argument(
        "--prefill-url",
        default="http://127.0.0.1:8100/v1/completions",
    )
    parser.add_argument(
        "--decode-url",
        default="http://127.0.0.1:8200/v1/completions",
    )
    parser.add_argument("--prefill-zmq", default="127.0.0.1:21001")
    parser.add_argument("--decode-zmq", default="127.0.0.1:22001")
    parser.add_argument(
        "--connector", choices=["p2p", "nixl"], default="nixl"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "host": args.host,
                "port": args.port,
                "backends": args.backends if args.mode == "colocated" else None,
                "weights": args.weights if args.mode == "colocated" else None,
                "prefill_url": (
                    args.prefill_url if args.mode == "disagg" else None
                ),
                "decode_url": args.decode_url if args.mode == "disagg" else None,
                "prefill_zmq": (
                    args.prefill_zmq if args.mode == "disagg" else None
                ),
                "decode_zmq": (
                    args.decode_zmq if args.mode == "disagg" else None
                ),
                "connector": args.connector if args.mode == "disagg" else None,
            }
        ),
        flush=True,
    )
    web.run_app(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
