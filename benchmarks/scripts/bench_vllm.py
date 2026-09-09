#!/usr/bin/env python3
"""Open-loop, fixed-work streaming benchmark. All timestamps use Unix seconds.

Only server usage counts are evidence of generated work. SSE text chunks are
not tokens. Arrival includes client-side semaphore and scheduling delay.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path

import aiohttp


async def send_request(session, api_base: str, model: str, prompt,
                       output_len: int, *, arrival_s=None,
                       arrival_monotonic=None, request_id="") -> dict:
    start = time.perf_counter() if arrival_monotonic is None else arrival_monotonic
    arrival = time.time() if arrival_s is None else arrival_s
    result = dict(request_id=request_id, arrival_s=arrival, success=False,
                  error="", ttft=None, latency=None, text="", itl=[],
                  n_text=0, generated_tokens=0, input_tokens=0,
                  token_itl=[], token_ids=[], token_events_exact=True,
                  http_status=None,admission_rejection=None,
                  token_count_source="missing", max_itl=None,
                  max_itl_pos=None, max_itl_at=None)
    payload = dict(model=model, prompt=prompt, max_tokens=int(output_len),
                   temperature=0, top_p=1.0, ignore_eos=True, stream=True,
                   stream_options={"include_usage": True}, seed=0)
    last = None
    last_token = None
    done = False
    try:
        async with session.post(api_base.rstrip("/") + "/v1/completions",
                                json=payload,
                                headers={"X-Request-Id": request_id}) as response:
            if response.status != 200:
                result['http_status']=response.status
                message=await response.text()
                try:
                    failure=json.loads(message).get('error',{})
                    if (response.status==429 and isinstance(failure,dict)
                            and failure.get('type')=='admission_rejection'
                            and failure.get('code') in ('admission_queue_full','admission_deadline')):
                        result['admission_rejection']=failure['code']
                except (ValueError,AttributeError): pass
                raise RuntimeError("HTTP %s: %s" % (
                    response.status, message[:300]))
            # readline handles split TCP packets; each SSE data line is a JSON event.
            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                usage = event.get("usage")
                if usage is not None and "completion_tokens" in usage:
                    result.update(generated_tokens=int(usage["completion_tokens"]),
                                  input_tokens=int(usage["prompt_tokens"]),
                                  token_count_source="server_usage")
                choices = event.get("choices") or []
                delta = choices[0].get("text", "") if choices else ""
                ids = event.get("token_ids") or []
                if ids:
                    if event.get('token_index',len(result['token_ids'])+len(ids))!=len(result['token_ids'])+len(ids):
                        raise RuntimeError('out_of_order_or_duplicate_token_event')
                    now = time.perf_counter()
                    if len(ids) != 1:
                        result["token_events_exact"] = False
                    if result["ttft"] is None:
                        result["ttft"] = now - start
                    if last_token is not None:
                        result["token_itl"].append(now - last_token)
                    last_token = now
                    result["token_ids"].extend(ids)
                if delta:
                    now = time.perf_counter()
                    result["n_text"] += 1
                    if result["ttft"] is None:
                        result["ttft"] = now - start
                    if last is not None:
                        result["itl"].append(now - last)
                    last = now
                    result["text"] += delta
            if not done:
                raise RuntimeError("truncated_stream")
            if last is None and last_token is None:
                raise RuntimeError("empty_stream")
            if result["token_count_source"] != "server_usage":
                raise RuntimeError("missing_token_usage")
            if result["generated_tokens"] != int(output_len):
                raise RuntimeError("incomplete_output:%s/%s" % (
                    result["generated_tokens"], output_len))
            if result['token_ids'] and len(result['token_ids'])!=result['generated_tokens']:
                raise RuntimeError('token_stream_usage_mismatch')
            result["success"] = True
    except (Exception, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    end = time.perf_counter()
    result["finish_s"] = arrival + (end - start)
    result["latency"] = (last_token if last_token is not None else
                          last if last is not None else end) - start
    if result["itl"]:
        maximum = max(result["itl"])
        index = result["itl"].index(maximum)
        result.update(max_itl=maximum, max_itl_pos=index + 1,
                      max_itl_at=result["ttft"] + sum(result["itl"][:index]))
    return result


async def run_trace(trace, api_base, model, max_concurrency=0, timeout_s=900):
    reqs, prompts = trace["requests"], trace["prompts"]
    if len(reqs) != len(prompts):
        raise ValueError("request/prompt count mismatch")
    if any(float(r["arrival_s"]) < 0 for r in reqs):
        raise ValueError("negative arrival")
    if any(float(a["arrival_s"]) > float(b["arrival_s"])
           for a, b in zip(reqs, reqs[1:])):
        raise ValueError("arrivals must be ordered")
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    mono, wall = time.perf_counter(), time.time()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker(i, req):
            scheduled = mono + float(req["arrival_s"])
            await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
            async def dispatch():
                return await send_request(
                    session, api_base, model, prompts[i], req["output_len"],
                    arrival_s=wall + float(req["arrival_s"]),
                    arrival_monotonic=scheduled, request_id=str(i))
            if semaphore is not None:
                async with semaphore:
                    return await dispatch()
            return await dispatch()
        outputs = await asyncio.gather(*(worker(i, r) for i, r in enumerate(reqs)))
    return outputs, time.perf_counter() - mono


def bench_rows(trace, outputs, slo_ttft, slo_tpot):
    rows = []
    for i, (request, output) in enumerate(zip(trace["requests"], outputs)):
        count = output["generated_tokens"]
        tpot = ((output["latency"] - output["ttft"]) / (count - 1)
                if count > 1 and output["ttft"] is not None
                else (0.0 if count == 1 else None))
        good = (output["success"] and output["ttft"] is not None
                and output["ttft"] < slo_ttft and tpot is not None
                and tpot < slo_tpot)
        rows.append(dict(
            idx=i, request_id=output["request_id"], success=int(output["success"]),
            arrival_s=output["arrival_s"], finish_s=output["finish_s"],
            prompt_len=request["prompt_len"], output_len=request["output_len"],
            input_tokens=output["input_tokens"], generated_tokens=count,
            token_count_source=output["token_count_source"],
            token_ids_verified=int(count>0 and len(output['token_ids'])==count),
            latency_s=output["latency"], ttft_s=output["ttft"], tpot_s=tpot,
            slo_ok=int(good), error=output["error"], n_text_chunks=output["n_text"],
            http_status=output.get('http_status'),admission_rejection=output.get('admission_rejection'),
            chunk_itl_s=json.dumps(output["itl"]),
            token_itl_s=json.dumps(output["token_itl"]),
            token_itl_exact=int(output["token_events_exact"] and
                                output['token_count_source']=='server_usage' and count>0 and
                                len(output["token_ids"]) == count and len(output['token_itl'])==count-1),
            output_token_sha256=hashlib.sha256(json.dumps(output["token_ids"]).encode()).hexdigest()
                if output["token_ids"] else "",
            max_itl_s=output["max_itl"], max_itl_pos=output["max_itl_pos"],
            max_itl_at_s=output["max_itl_at"],
            output_text_sha256=hashlib.sha256(output["text"].encode()).hexdigest()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--slo-ttft", type=float, default=5.0)
    parser.add_argument("--slo-tpot", type=float, default=0.1)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--max-concurrency", type=int, default=0)
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text())
    outputs, duration = asyncio.run(run_trace(
        trace, args.api_base, args.model, args.max_concurrency))
    rows = bench_rows(trace, outputs, args.slo_ttft, args.slo_tpot)
    summary = dict(completed=sum(r["success"] for r in rows),
                   n_expected=len(rows), duration_s=duration,
                   output_tokens=sum(r["generated_tokens"] for r in rows),
                   slo_attainment=sum(r["slo_ok"] for r in rows) / max(len(rows), 1))
    summary["request_throughput"] = summary["completed"] / max(duration, 1e-9)
    print(json.dumps(summary, indent=2))
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["idx"])
            writer.writeheader()
            writer.writerows(rows)
        Path(args.out_csv).with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
