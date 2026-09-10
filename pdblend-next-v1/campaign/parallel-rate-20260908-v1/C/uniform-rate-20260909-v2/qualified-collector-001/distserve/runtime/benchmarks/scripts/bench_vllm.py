#!/usr/bin/env python3
"""Open-loop, fixed-work streaming benchmark. All timestamps use Unix seconds.

Server usage retains its original role in complete-work and latency checks.
Independently indexed token IDs also prove received output on interrupted
streams. SSE text chunks are not tokens. Arrival includes scheduling delay.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import ipaddress
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

EVALUATION_V3 = "evaluation-v3"
REQUEST_TIMEOUT_S = 120.0


def evaluation_headers(api_base, protocol, arrival_s, dispatch_s):
    """Only a loopback controller may receive the trusted benchmark clock."""
    if protocol != EVALUATION_V3:
        return {}
    host = urlsplit(api_base).hostname
    try:
        loopback = ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError("evaluation-v3 requires a loopback controller")
    return {"X-PDBlend-Evaluation-Protocol": EVALUATION_V3,
            "X-PDBlend-Planned-Arrival-S": format(arrival_s, ".17g"),
            "X-PDBlend-Actual-Dispatch-S": format(dispatch_s, ".17g")}


def empty_result(request_id, arrival):
    return dict(request_id=request_id, arrival_s=arrival, success=False,
                error="", ttft=None, latency=None, text="", itl=[],
                n_text=0, generated_tokens=0, input_tokens=0,
                token_itl=[], token_ids=[], token_events_exact=True,
                token_event_records=[], token_sequence_verified=True,
                token_stream_opened=False, token_stream_parse_error=False,
                http_status=None, admission_rejection=None,
                token_count_source="missing", max_itl=None,
                max_itl_pos=None, max_itl_at=None)


async def send_request(session, api_base: str, model: str, prompt,
                       output_len: int, *, arrival_s=None,
                       arrival_monotonic=None, request_id="",
                       evaluation_protocol=None, _result_sink=None) -> dict:
    start = time.perf_counter() if arrival_monotonic is None else arrival_monotonic
    arrival = time.time() if arrival_s is None else arrival_s
    result = empty_result(request_id, arrival)
    dispatch = arrival + (time.perf_counter() - start)
    headers = {"X-Request-Id": request_id}
    headers.update(evaluation_headers(api_base, evaluation_protocol, arrival, dispatch))
    if evaluation_protocol == EVALUATION_V3:
        result.update(evaluation_protocol=EVALUATION_V3, planned_arrival_s=arrival,
                      actual_dispatch_s=dispatch, first_token_s=None, last_token_s=None,
                      stream_end_s=None, request_deadline_s=arrival + REQUEST_TIMEOUT_S,
                      request_timeout=False)
        if _result_sink is not None:
            _result_sink[request_id]=result
    payload = dict(model=model, prompt=prompt, max_tokens=int(output_len),
                   temperature=0, top_p=1.0, ignore_eos=True, stream=True,
                   stream_options={"include_usage": True}, seed=0)
    last = None
    last_token = None
    done = False
    try:
        async with session.post(api_base.rstrip("/") + "/v1/completions",
                                json=payload,
                                headers=headers) as response:
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
            result['token_stream_opened'] = True
            # readline handles split TCP packets; each SSE data line is a JSON event.
            async for raw in response.content:
                try:
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError:
                    result['token_stream_parse_error'] = True
                    raise
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    if evaluation_protocol == EVALUATION_V3:
                        result["stream_end_s"] = arrival + (time.perf_counter() - start)
                    break
                try:
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError('non_object_stream_event')
                except (ValueError, TypeError):
                    result['token_stream_parse_error'] = True
                    raise
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
                if delta and not ids:
                    result['token_sequence_verified'] = False
                if ids:
                    expected_index = len(result['token_ids']) + len(ids)
                    verified_index = (type(event.get('token_index')) is int
                                      and event['token_index'] == expected_index
                                      and isinstance(ids, list)
                                      and all(type(token) is int for token in ids))
                    result['token_sequence_verified'] &= verified_index
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
                    if evaluation_protocol == EVALUATION_V3:
                        result["first_token_s"] = arrival + result["ttft"]
                        result["last_token_s"] = arrival + (now - start)
                    result["token_ids"].extend(ids)
                    result['token_event_records'].append(dict(
                        token_index=event.get('token_index'), token_ids=list(ids)))
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
        if isinstance(exc, asyncio.CancelledError) and evaluation_protocol != EVALUATION_V3:
            raise
        if evaluation_protocol == EVALUATION_V3 and isinstance(exc, (asyncio.CancelledError, asyncio.TimeoutError)):
            result.update(error="request_hard_timeout", request_timeout=True)
        else:
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


async def run_trace(trace, api_base, model, max_concurrency=0, timeout_s=900, *,
                    evaluation_protocol=None):
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
    if evaluation_protocol == EVALUATION_V3:
        evaluation_headers(api_base, evaluation_protocol, wall, wall)
        timeout_s = REQUEST_TIMEOUT_S
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    partial_outputs={}
    def interrupted_result(i, req, error="request_hard_timeout_before_dispatch"):
        arrival = wall + float(req["arrival_s"])
        partial=partial_outputs.get(str(i))
        result = partial if partial is not None else empty_result(str(i), arrival)
        finish = wall + time.perf_counter() - mono
        if partial is None:
            result.update(actual_dispatch_s=None,first_token_s=None,last_token_s=None,stream_end_s=None)
        result.update(evaluation_protocol=EVALUATION_V3, planned_arrival_s=arrival,success=False,
            request_deadline_s=arrival + REQUEST_TIMEOUT_S,
            finish_s=finish, latency=(result.get('last_token_s') or finish)-arrival,
            request_timeout=error.startswith("request_hard_timeout"), error=error)
        return result
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker(i, req):
            scheduled = mono + float(req["arrival_s"])
            await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
            async def dispatch():
                return await send_request(
                    session, api_base, model, prompts[i], req["output_len"],
                    arrival_s=wall + float(req["arrival_s"]),
                    arrival_monotonic=scheduled, request_id=str(i),
                    **({"evaluation_protocol": evaluation_protocol,"_result_sink":partial_outputs}
                       if evaluation_protocol == EVALUATION_V3 else {}))
            async def admitted_dispatch():
                if semaphore is not None:
                    async with semaphore:
                        return await dispatch()
                return await dispatch()
            if evaluation_protocol != EVALUATION_V3:
                return await admitted_dispatch()
            try:
                return await asyncio.wait_for(admitted_dispatch(),
                    timeout=max(0.0, scheduled + REQUEST_TIMEOUT_S - time.perf_counter()))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                return interrupted_result(i,req)
        tasks = [asyncio.create_task(worker(i, r)) for i, r in enumerate(reqs)]
        if evaluation_protocol == EVALUATION_V3 and tasks:
            # This is the same absolute bound as the final request's timeout.
            deadline = mono + float(reqs[-1]["arrival_s"]) + REQUEST_TIMEOUT_S
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, deadline-time.perf_counter()))
            for task in pending:
                task.cancel()
        outputs = await asyncio.gather(*tasks,return_exceptions=evaluation_protocol==EVALUATION_V3)
        if evaluation_protocol==EVALUATION_V3:
            outputs=[interrupted_result(i,req,"request_hard_timeout_before_dispatch"
                if isinstance(result,asyncio.CancelledError) else 'client_worker_error: '+repr(result))
                if isinstance(result,BaseException) else result
                for i,(req,result) in enumerate(zip(reqs,outputs))]
            for result in outputs:
                result['open_loop_independent']=semaphore is None
    return outputs, time.perf_counter() - mono


def received_token_evidence(output):
    """Additional evidence only; never alter legacy work, SLO or latency fields."""
    ids = output['token_ids']
    records = output.get('token_event_records', [])
    flattened = []
    ordered = True
    for event in records:
        values = event.get('token_ids', [])
        ordered &= (isinstance(values, list) and bool(values)
                    and all(type(value) is int for value in values)
                    and type(event.get('token_index')) is int
                    and event['token_index'] == len(flattened) + len(values))
        flattened.extend(values)
    exact = bool(output.get('token_sequence_verified')
                 and not output.get('token_stream_parse_error')
                 and output.get('token_stream_opened') and ordered and flattened == ids)
    if output['token_count_source'] == 'server_usage':
        exact &= len(ids) == output['generated_tokens']
    return dict(token_evidence_schema=2,
                received_token_ids=json.dumps(ids),
                received_token_events=json.dumps(records, separators=(',', ':')),
                received_token_count=len(ids), received_token_count_exact=int(exact),
                token_sequence_verified=int(bool(output.get('token_sequence_verified'))),
                token_stream_opened=int(bool(output.get('token_stream_opened'))),
                token_stream_parse_error=int(bool(output.get('token_stream_parse_error'))),
                output_count_scope='received indexed server tokens; interrupted streams retain prefix',
                terminal_usage_observed=int(output['token_count_source'] == 'server_usage'))


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
        rows[-1].update(received_token_evidence(output))
        if output.get("evaluation_protocol") == EVALUATION_V3:
            rows[-1].update({key: output.get(key) for key in (
                "evaluation_protocol", "planned_arrival_s", "actual_dispatch_s",
                "first_token_s", "last_token_s", "stream_end_s", "request_deadline_s",
                "request_timeout", "open_loop_independent")})
            rows[-1]["dispatch_delay_s"] = (output["actual_dispatch_s"] - output["planned_arrival_s"]
                if output.get("actual_dispatch_s") is not None else None)
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
    parser.add_argument("--evaluation-protocol", choices=[EVALUATION_V3])
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text())
    outputs, duration = asyncio.run(run_trace(
        trace, args.api_base, args.model, args.max_concurrency,
        evaluation_protocol=args.evaluation_protocol))
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
